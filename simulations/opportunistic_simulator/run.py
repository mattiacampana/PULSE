"""Command-line entry point for an opportunistic-learning simulation."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

# Allow this file to be executed directly from the project root, e.g.
# ``python3 run.py --config configs/example.yaml``.  In that case Python adds
# the package directory itself to sys.path, whereas importing
# ``opportunistic_simulator`` requires its parent directory.
if __package__ in {None, ""}:
    project_parent = Path(__file__).resolve().parent.parent
    if str(project_parent) not in sys.path:
        sys.path.insert(0, str(project_parent))

from opportunistic_simulator.data import cache_node_embeddings, assign_trace_nodes, load_mobility_trace, make_node_splits
from opportunistic_simulator.nodes import create_node
from opportunistic_simulator.simulator import Simulator
from opportunistic_simulator.utils.config import Config
from opportunistic_simulator.utils import commons as common_utils



def init_simulation(
        config: Config,
        seed: int,
        config_path: str
    ) -> tuple[dict[str, common_utils.Node], list[common_utils.ContactEvent], dict[str, str]]:
    """
    Initialize the simulation environment, including loading datasets, mobility traces, and creating nodes.
    """

    common_utils.seed_everything(seed)
    
    print("Opportunistic-learning simulator", flush=True)
    print(f"  config: {config_path}", flush=True)
    print(f"  seed: {seed} | device: {config.training.device}", flush=True)
    print("[setup] Loading and splitting the local datasets ...", flush=True)

    # Load the dataset and produce deterministic splits for each node -----------------------------------
    # The simulator will only use the splits for nodes that are actually assigned
    # to the mobility trace, but this allows the user to specify a larger
    # dataset and a smaller number of nodes to simulate.  The node selection
    # strategy is deterministic and reproducible, so the same config always
    # yields the same node splits, even if the dataset contains more nodes than requested.
    splits, dataset = make_node_splits(
        config.data.path,
        config.data.num_nodes,
        config.data.train_fraction, config.data.val_fraction, config.data.test_fraction,
        seed,
        config.data.node_selection,
        config.data.source_split,
        config.data.allow_full_fallback,
        config.data.loading_mode if config.data.loading_mode != "cache_embeddings" else "lazy_hdf5",
    )

    print(
        f"[setup] Loaded {len(splits)} data nodes from split {dataset.selected_split!r} "
        f"({sum(len(split.train) for split in splits.values()):,} train windows).",
        flush=True,
    )

    # Load the mobility trace --------------------------------------------------------------------------
    print("[setup] Loading and assigning the mobility trace ...", flush=True)
    raw_events = load_mobility_trace(config.mobility.path)
    events, node_assignment = assign_trace_nodes(
        raw_events, splits, config.mobility.node_assignment, seed,
    )
    print(
        f"[setup] Loaded {len(raw_events):,} raw contact events; "
        f"{len(events):,} events remain after node assignment.",
        flush=True,
    )

    # Initialize the encoder ----------------------------------------------------------------------------
    config.model.num_classes = (
        dataset.num_classes if config.model.num_classes == "auto" else int(config.model.num_classes)
    )
    encoder = common_utils.init_encoder(config=config, descriptor=dataset)

    if config.data.loading_mode == "cache_embeddings":
        if not config.pretraining.checkpoint:
            raise ValueError("data.loading_mode='cache_embeddings' requires pretraining.checkpoint.")
        if not config.pretraining.freeze_encoder:
            raise ValueError(
                "data.loading_mode='cache_embeddings' requires pretraining.freeze_encoder=true. "
                "Cached embeddings would become stale if the encoder were trainable."
            )
        cache_started_at = time.perf_counter()
        total_windows = sum(
            len(part)
            for node_splits in splits.values()
            for part in (node_splits.train, node_splits.val, node_splits.test)
        )
        print(
            f"[setup] Caching {total_windows:,} frozen-encoder embeddings in RAM "
            f"(batch size {config.data.cache_batch_size}, "
            f"workers {config.data.cache_num_workers}) ...",
            flush=True,
        )
        last_reported_percent = -10

        def report_cache_progress(done: int, total: int) -> None:
            nonlocal last_reported_percent
            percent = int(100 * done / max(total, 1))
            if percent == 100 or percent >= last_reported_percent + 10:
                print(f"[setup] Embedding cache: {done:,}/{total:,} ({percent}%)", flush=True)
                last_reported_percent = percent

        splits = cache_node_embeddings(
            splits, encoder, config.training.device, config.data.cache_batch_size,
            config.data.cache_num_workers, config.data.cache_pin_memory,
            config.data.cache_prefetch_factor, report_cache_progress,
        )
        encoder = torch.nn.Identity()
        print(
            f"[setup] Cached embeddings in {time.perf_counter() - cache_started_at:.1f}s; "
            "HDF5 will not be read during the timeline.",
            flush=True,
        )

    elif config.data.loading_mode == "in_memory":
        print(
            "[setup] Raw input windows are shared in CPU memory; "
            "HDF5 will not be read during the timeline.",
            flush=True,
        )


    # Create nodes --------------------------------------------------------------------------------------
    print(
        f"[setup] Creating {config.algorithm.name} nodes "
        f"(encoder {'frozen' if config.pretraining.freeze_encoder else 'trainable'}) ...",
        flush=True,
    )
    nodes = {}
    head_fingerprints = {}
    for node_id, node_splits in splits.items():

        # Used to inizialize all the MLP (classifiers) in the same way
        init_id = "common"

        classifier = common_utils.make_seeded_classifier(
            config.model.embedding_dim, config.model.num_classes,
            config.model.classifier_hidden_dims, seed, init_id,
            batch_norm=(config.algorithm.name == "optimist"),
        )

        head_fingerprints[str(node_id)] = common_utils.classifier_fingerprint(classifier)

        kwargs = dict(
            node_id=node_id,
            encoder=copy.deepcopy(encoder),
            classifier=classifier,
            train_data=node_splits.train,
            val_data=node_splits.val,
            test_data=node_splits.test,
            config=config.training,
            device=config.training.device,
        )

        nodes[node_id] = create_node(
            config=config,
            node_kwargs=kwargs,
            node_index=list(splits.keys()).index(node_id),
            num_nodes=len(splits),
            seed=seed
        )

    print(
        "[setup] Initial head fingerprints (seed + node ID): "
        + ", ".join(f"{node_id}={fingerprint}" for node_id, fingerprint in head_fingerprints.items()),
        flush=True,
    )

    return nodes, events, node_assignment


def main() -> None:

    # Read the configuration file ---------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    # Load the configuration and set the random seeds for reproducibility -----------------------------------
    config : Config = Config.from_yaml(args.config)

    for seed in config.seeds:
        print(f"\n[setup] Starting simulation with seed {seed} ...", flush=True)

        started_at = time.perf_counter()
        nodes, events, node_assignment = init_simulation(config, seed, str(args.config.resolve()))
        print(f"[setup] Ready in {time.perf_counter() - started_at:.1f}s. Starting timeline ...", flush=True)

        # Run simulation ------------------------------------------------------------------------------------
        Simulator(
            nodes=nodes,
            contact_events=events,
            config=config,
            node_assignment=node_assignment,
            seed=seed
        ).run()


if __name__ == "__main__":
    main()
