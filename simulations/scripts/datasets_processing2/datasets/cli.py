"""Command-line interface for all DeepRAP dataset adapters."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__


# Parser construction must stay dependency-light: importing the adapters loads
# the numerical/HDF5 stack, which is unnecessary for --help and --version.
DATASET_NAMES = (
    "audio_imu_cough", "cifar10", "fashion_mnist", "hhar", "pads", "pulse_transit_time",
    "ringauth", "sisfall", "uci_har", "wesad",
)


def build_parser() -> argparse.ArgumentParser:
    """Create the documented CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", type=Path, help="YAML configuration for one or more datasets.")
    parser.add_argument("--dataset", choices=(*DATASET_NAMES, "all"), help="Prepare one dataset, or every registered adapter.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--num-pretrain-users", type=int)
    parser.add_argument("--num-validation-users", type=int)
    parser.add_argument("--num-simulation-users", type=int)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--window-seconds", type=float)
    parser.add_argument("--stride-seconds", type=float)
    parser.add_argument("--target-rate-hz", type=float)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--verbose", action="store_true", help="Show source-file, tensor-shape, and download-progress details.")
    output.add_argument("--quiet", action="store_true", help="Suppress normal progress messages.")
    return parser


def _from_arguments(args: argparse.Namespace):
    from .config import DatasetConfig, PipelineConfig, SplitConfig

    if not args.dataset:
        raise ValueError("Provide --config or --dataset.")
    names = DATASET_NAMES if args.dataset == "all" else (args.dataset,)
    split = SplitConfig(args.num_pretrain_users, args.num_validation_users, args.num_simulation_users, args.split_seed)
    entries = tuple(DatasetConfig(name, args.raw_dir / name, args.output_dir / name, args.window_seconds, args.stride_seconds, args.target_rate_hz, split) for name in names)
    return PipelineConfig(entries, args.force_download, args.force_preprocess)


def main(argv: list[str] | None = None) -> None:
    """Run selected adapters and stop at the first invalid dataset."""
    parser = build_parser()
    args = parser.parse_args(argv)
    from . import adapters as _adapters  # noqa: F401 - registers adapters.
    from .adapters.base import available_adapters, create_adapter
    from .config import load_config
    from .reporting import REPORTER, configure_reporting

    configure_reporting(verbose=args.verbose, quiet=args.quiet)
    if args.list_datasets:
        print("\n".join(available_adapters()))
        return
    try:
        pipeline = load_config(args.config) if args.config else _from_arguments(args)
        REPORTER.info("pipeline", "START", f"Preparing {len(pipeline.datasets)} dataset(s)")
        for config in pipeline.datasets:
            create_adapter(config.name).run(config, pipeline.force_download, pipeline.force_preprocess)
        REPORTER.info("pipeline", "DONE", f"Successfully prepared {len(pipeline.datasets)} dataset(s)")
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
