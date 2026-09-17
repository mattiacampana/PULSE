"""Model-ready HDF5 datasets and mobility-trace utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


class HDF5SampleDataset(Dataset):
    """Lazy view over selected rows of a model-ready ``data.h5`` file.

    The file handle is opened on first access in each process, which keeps the
    dataset safe when DataLoader workers are enabled later.
    """

    def __init__(self, path: str | Path, indices: Iterable[int], branches: Iterable[str]) -> None:
        self.path = str(Path(path).resolve())
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.branches = tuple(branches)
        self._archive: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self._archive is None:
            self._archive = h5py.File(self.path, "r")
        index = int(self.indices[position])
        inputs = {
            name: torch.as_tensor(np.asarray(self._archive[f"inputs/{name}"][index]), dtype=torch.float32)
            for name in self.branches
        }
        label = torch.as_tensor(self._archive["labels"][index], dtype=torch.long)
        return inputs, label

    def __getitems__(self, positions: Sequence[int]) -> list[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Read a complete DataLoader batch with one HDF5 selection per branch.

        Recent PyTorch versions call ``__getitems__`` for map-style batches.
        Sorting the physical rows satisfies h5py's indexing requirements and
        avoids tens of thousands of tiny, random HDF5 reads.
        """
        if self._archive is None:
            self._archive = h5py.File(self.path, "r")
        positions_array = np.asarray(positions, dtype=np.int64)
        physical = self.indices[positions_array]
        order = np.argsort(physical, kind="stable")
        sorted_physical = physical[order]
        if len(np.unique(sorted_physical)) != len(sorted_physical):
            raise ValueError("A cache batch contains duplicate HDF5 sample indices.")
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        branch_batches = {
            name: np.asarray(self._archive[f"inputs/{name}"][sorted_physical])[inverse]
            for name in self.branches
        }
        label_batch = np.asarray(self._archive["labels"][sorted_physical])[inverse]
        return [
            (
                {name: torch.as_tensor(values[row], dtype=torch.float32) for name, values in branch_batches.items()},
                torch.as_tensor(label_batch[row], dtype=torch.long),
            )
            for row in range(len(positions_array))
        ]

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def __del__(self) -> None:
        self.close()


class InMemorySampleDataset(Dataset):
    """Zero-copy local view over one shared CPU-resident raw-sample cache."""

    def __init__(
        self,
        inputs: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        indices: Iterable[int],
    ) -> None:
        self.inputs = dict(inputs)
        self.labels = labels
        self.indices = torch.as_tensor(np.asarray(list(indices), dtype=np.int64), dtype=torch.long)
        if any(len(values) != len(self.labels) for values in self.inputs.values()):
            raise ValueError("All in-memory input branches must have the same length as labels.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        index = int(self.indices[position])
        return {name: values[index] for name, values in self.inputs.items()}, self.labels[index]


class EmbeddingDataset(Dataset):
    """In-memory CPU dataset produced by a frozen encoder."""

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        if len(embeddings) != len(labels):
            raise ValueError("Embeddings and labels must have the same length.")
        self.embeddings = embeddings.detach().cpu().contiguous()
        self.labels = labels.detach().cpu().long().contiguous()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, position: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return {"embedding": self.embeddings[position]}, self.labels[position]


class IndexedEmbeddingDataset(Dataset):
    """Zero-copy view over one shared, in-memory embedding cache."""

    def __init__(self, cache: EmbeddingDataset, positions: Iterable[int]) -> None:
        self.cache = cache
        self.positions = torch.as_tensor(np.asarray(list(positions), dtype=np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, position: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return self.cache[int(self.positions[position])]


def dataset_labels(dataset: Dataset) -> torch.Tensor:
    """Return labels without assuming a particular dataset implementation.

    The returned tensor follows the dataset's logical order.  In particular,
    indexed embedding views must not expose the entire shared cache, and lazy
    HDF5 views must preserve their (usually shuffled) local split order.
    """
    if isinstance(dataset, IndexedEmbeddingDataset):
        labels = dataset.cache.labels.index_select(0, dataset.positions)
    elif isinstance(dataset, EmbeddingDataset):
        labels = dataset.labels
    elif isinstance(dataset, InMemorySampleDataset):
        labels = dataset.labels.index_select(0, dataset.indices)
    elif isinstance(dataset, HDF5SampleDataset):
        physical = dataset.indices
        order = np.argsort(physical, kind="stable")
        sorted_physical = physical[order]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        with h5py.File(dataset.path, "r") as archive:
            labels = torch.as_tensor(
                np.asarray(archive["labels"][sorted_physical])[inverse],
                dtype=torch.long,
            )
    elif isinstance(dataset, TensorDataset):
        if len(dataset.tensors) < 2:
            raise ValueError("A TensorDataset must contain inputs and labels.")
        labels = dataset.tensors[1]
    elif isinstance(dataset, Subset):
        parent_labels = dataset_labels(dataset.dataset)
        positions = torch.as_tensor(dataset.indices, dtype=torch.long)
        labels = parent_labels.index_select(0, positions)
    else:
        raise TypeError(
            "Cannot extract labels from dataset type "
            f"{type(dataset).__name__}; use a supported simulator dataset."
        )

    labels = torch.as_tensor(labels, dtype=torch.long).detach().cpu().contiguous()
    if labels.ndim != 1 or len(labels) != len(dataset):
        raise ValueError("A simulation dataset must expose one scalar label per sample.")
    return labels


@dataclass(frozen=True)
class NodeSplits:
    train: Dataset
    val: Dataset
    test: Dataset


@dataclass(frozen=True)
class DatasetDescriptor:
    root: Path
    h5_path: Path
    manifest_path: Path
    metadata: Mapping[str, object]
    branches: tuple[str, ...]
    input_shapes: Mapping[str, tuple[int, ...]]
    num_classes: int
    selected_split: str


@dataclass(frozen=True)
class ContactEvent:
    time: int
    node_a: str
    node_b: str


@dataclass(frozen=True)
class ContactAttempt:
    peer_id: str
    event: ContactEvent
    completed: bool = True
    details: Mapping[str, object] | None = None


def _cache_dataset_embeddings(
    dataset: Dataset,
    encoder: nn.Module,
    device: str | torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    progress: Callable[[int, int], None] | None = None,
) -> EmbeddingDataset:
    embeddings, labels = [], []
    processed = 0
    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    with torch.inference_mode():
        for inputs, target in loader:
            inputs = {name: value.to(device, non_blocking=True) for name, value in inputs.items()}
            embeddings.append(encoder(**inputs).detach().cpu())
            labels.append(target.cpu())
            processed += len(target)
            if progress is not None:
                progress(processed, len(dataset))
    return EmbeddingDataset(torch.cat(embeddings), torch.cat(labels))


def cache_node_embeddings(
    splits: Mapping[str, NodeSplits],
    encoder: nn.Module,
    device: str | torch.device,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, NodeSplits]:
    """Encode all selected HDF5 rows once and expose zero-copy local views."""
    if batch_size <= 0:
        raise ValueError("data.cache_batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("data.cache_num_workers must be non-negative.")
    if prefetch_factor <= 0:
        raise ValueError("data.cache_prefetch_factor must be positive.")
    datasets = [
        dataset
        for node_splits in splits.values()
        for dataset in (node_splits.train, node_splits.val, node_splits.test)
    ]
    if not datasets or not all(isinstance(dataset, HDF5SampleDataset) for dataset in datasets):
        raise TypeError("Embedding caching requires HDF5SampleDataset node splits.")
    first = datasets[0]
    if not all(dataset.path == first.path and dataset.branches == first.branches for dataset in datasets):
        raise ValueError("All cached node splits must refer to the same HDF5 dataset and branches.")

    all_indices = np.concatenate([dataset.indices for dataset in datasets])
    unique_indices = np.unique(all_indices)
    if len(unique_indices) != len(all_indices):
        raise ValueError("Node train/validation/test splits overlap; refusing to cache leaked samples.")
    global_dataset = HDF5SampleDataset(first.path, unique_indices, first.branches)
    device_object = torch.device(device)
    effective_pin_memory = bool(pin_memory and device_object.type == "cuda")
    encoder = encoder.to(device_object).eval()
    shared_cache = _cache_dataset_embeddings(
        global_dataset, encoder, device_object, batch_size,
        num_workers, effective_pin_memory, prefetch_factor, progress,
    )

    cached: dict[str, NodeSplits] = {}
    for node_id, node_splits in splits.items():
        views = []
        for dataset in (node_splits.train, node_splits.val, node_splits.test):
            positions = np.searchsorted(unique_indices, dataset.indices)
            if not np.array_equal(unique_indices[positions], dataset.indices):
                raise RuntimeError("Internal error while mapping samples into the embedding cache.")
            views.append(IndexedEmbeddingDataset(shared_cache, positions))
        cached[node_id] = NodeSplits(*views)
    return cached


def make_node_splits(
    path: str | Path,
    num_nodes: int | None,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    node_selection: str = "first",
    source_split: str = "simulation",
    allow_full_fallback: bool = False,
    loading_mode: str = "lazy_hdf5",
) -> tuple[dict[str, NodeSplits], DatasetDescriptor]:
    """Select subjects from one manifest split and build local dataset views."""
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("Local train, validation, and test fractions must be positive.")
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("Local train, validation, and test fractions must sum to 1.0.")

    requested = Path(path).expanduser().resolve()
    root = requested if requested.is_dir() else requested.parent
    h5_path = requested if requested.suffix in {".h5", ".hdf5"} else root / "data.h5"
    manifest_path = root / "manifest.csv"
    metadata_path = root / "dataset.json"
    for required in (h5_path, manifest_path, metadata_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required model-ready dataset file not found: {required}")

    import json
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str, "split": str})
    required_columns = {"sample_index", "subject_id", "split"}
    if not required_columns.issubset(manifest.columns):
        raise ValueError(f"{manifest_path} must contain {sorted(required_columns)}.")
    if manifest["sample_index"].duplicated().any():
        raise ValueError("manifest.csv contains duplicate sample_index values.")

    if loading_mode not in {"lazy_hdf5", "in_memory"}:
        raise ValueError("make_node_splits supports loading_mode 'lazy_hdf5' or 'in_memory'.")
    shared_inputs: dict[str, torch.Tensor] | None = None
    shared_labels: torch.Tensor | None = None
    with h5py.File(h5_path, "r") as archive:
        if "inputs" not in archive or "labels" not in archive:
            raise ValueError(f"{h5_path} must contain inputs/ and labels.")
        branches = tuple(archive["inputs"].keys())
        length = len(archive["labels"])
        input_shapes = {name: tuple(map(int, archive[f"inputs/{name}"].shape[1:])) for name in branches}
        labels = np.asarray(archive["labels"])
        if loading_mode == "in_memory":
            shared_inputs = {
                name: torch.from_numpy(np.asarray(archive[f"inputs/{name}"], dtype=np.float32)).contiguous()
                for name in branches
            }
            shared_labels = torch.from_numpy(labels).long().contiguous()
    if len(manifest) != length or set(manifest["sample_index"].astype(int)) != set(range(length)):
        raise ValueError("manifest.csv sample_index values must cover every HDF5 row exactly once.")
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or len(labels) == 0 or labels.min() < 0:
        raise ValueError("Simulation requires one non-negative integer class label per HDF5 sample.")

    available = set(manifest["split"].dropna().astype(str))
    selected_split = source_split
    if selected_split not in available:
        if allow_full_fallback and "full" in available:
            selected_split = "full"
        else:
            raise ValueError(
                f"Requested data.source_split={source_split!r}, but available splits are {sorted(available)}. "
                "Set data.allow_full_fallback=true only for an ex-novo simulation dataset prepared without roles."
            )
    selected = manifest.loc[manifest["split"] == selected_split].copy()
    if selected.empty:
        raise ValueError(f"Split {selected_split!r} contains no samples.")

    node_ids = sorted(selected["subject_id"].unique().tolist())
    if num_nodes is not None:
        if num_nodes > len(node_ids):
            raise ValueError(f"Requested {num_nodes} nodes, but split {selected_split!r} contains only {len(node_ids)} subjects.")
        if node_selection == "random":
            node_ids = sorted(np.random.default_rng(seed).choice(node_ids, size=num_nodes, replace=False).tolist())
        elif node_selection == "first":
            node_ids = node_ids[:num_nodes]
        else:
            raise ValueError("node_selection must be 'first' or 'random'.")

    rng = np.random.default_rng(seed)
    result: dict[str, NodeSplits] = {}
    for node_id in node_ids:
        indices = selected.loc[selected["subject_id"] == node_id, "sample_index"].to_numpy(dtype=np.int64, copy=True)
        rng.shuffle(indices)
        n_train = int(len(indices) * train_fraction)
        n_val = int(len(indices) * val_fraction)
        if n_train == 0 or n_val == 0 or len(indices) - n_train - n_val == 0:
            raise ValueError(f"Node {node_id} does not have enough windows for all local splits.")
        split_indices = (indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:])
        if loading_mode == "in_memory":
            assert shared_inputs is not None and shared_labels is not None
            result[node_id] = NodeSplits(*[
                InMemorySampleDataset(shared_inputs, shared_labels, part) for part in split_indices
            ])
        else:
            result[node_id] = NodeSplits(*[
                HDF5SampleDataset(h5_path, part, branches) for part in split_indices
            ])

    class_names = metadata.get("class_names")
    num_classes = len(class_names) if isinstance(class_names, list) and class_names else int(labels.max()) + 1
    descriptor = DatasetDescriptor(
        root=root, h5_path=h5_path, manifest_path=manifest_path, metadata=metadata,
        branches=branches, input_shapes=input_shapes, num_classes=num_classes,
        selected_split=selected_split,
    )
    return result, descriptor


def load_mobility_trace(path: str | Path) -> list[ContactEvent]:
    frame = pd.read_csv(path)
    required = {"time", "node1", "node2"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Mobility CSV must contain {sorted(required)}.")
    return sorted([
        ContactEvent(int(row.time), str(row.node1), str(row.node2))
        for row in frame.itertuples(index=False) if str(row.node1) != str(row.node2)
    ], key=lambda event: event.time)


def assign_trace_nodes(events: list[ContactEvent], dataset_nodes: Iterable[str], mode: str, seed: int) -> tuple[list[ContactEvent], dict[str, str]]:
    data_nodes = sorted(map(str, dataset_nodes))
    trace_nodes = sorted({event.node_a for event in events} | {event.node_b for event in events})
    if mode == "identity":
        missing = set(data_nodes) - set(trace_nodes)
        if missing:
            raise ValueError(f"Mobility trace does not contain dataset node IDs: {sorted(missing)}.")
        mapping = {node: node for node in data_nodes}
    elif mode == "random":
        if len(trace_nodes) < len(data_nodes):
            raise ValueError(f"Mobility trace has {len(trace_nodes)} nodes, fewer than the {len(data_nodes)} simulated nodes.")
        rng = np.random.default_rng(seed)
        mapping = dict(zip(rng.choice(trace_nodes, len(data_nodes), replace=False).tolist(), rng.permutation(data_nodes).tolist()))
    else:
        raise ValueError("mobility.node_assignment must be 'identity' or 'random'.")
    remapped = [ContactEvent(event.time, mapping[event.node_a], mapping[event.node_b]) for event in events if event.node_a in mapping and event.node_b in mapping]
    return remapped, mapping
