"""Supervised pretraining for every model-ready DeepRAP encoder.

The preferred input is a processed dataset directory containing ``data.h5``,
``dataset.json`` and ``manifest.csv``.  Legacy NPZ archives with ``x``/``y``
remain supported for UCI HAR experiments.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


def _make_project_package_importable() -> None:
    """Add the source checkout root when this file is run from ``scripts/``.

    A direct invocation such as ``python scripts/pretrain_encoders.py`` puts
    only ``scripts/`` on ``sys.path``.  In the DeepRAP source tree the Python
    package is its sibling, so locate its root from this file instead of
    depending on the caller's current working directory.
    """
    try:
        __import__("opportunistic_simulator.models")
        return
    except ModuleNotFoundError as error:
        if error.name not in {"opportunistic_simulator", "opportunistic_simulator.models"}:
            raise

    script_path = Path(__file__).resolve()
    for root in script_path.parents:
        if (root / "opportunistic_simulator" / "models.py").is_file():
            sys.path.insert(0, str(root))
            return

    raise ModuleNotFoundError(
        "Cannot locate opportunistic_simulator/models.py. Place this script "
        "inside the DeepRAP checkout or install the project package."
    )


_make_project_package_importable()

from opportunistic_simulator.models import MLPClassifier, build_encoder


class ModelReadyDataset(Dataset):
    """Lazy HDF5 reader which is safe for large, multimodal datasets."""

    def __init__(self, path: Path, branches: Sequence[str]) -> None:
        self.path = path
        self.branches = tuple(branches)
        self._archive: h5py.File | None = None
        with h5py.File(path, "r") as archive:
            self.length = len(archive["labels"])
            missing = [name for name in self.branches if f"inputs/{name}" not in archive]
            if missing:
                raise ValueError(f"Missing HDF5 input branches: {', '.join(missing)}.")
            for name in self.branches:
                if len(archive[f"inputs/{name}"]) != self.length:
                    raise ValueError(f"Branch {name!r} and labels have different sample counts.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self._archive is None:
            self._archive = h5py.File(self.path, "r")
        inputs = {
            name: torch.as_tensor(self._archive[f"inputs/{name}"][index], dtype=torch.float32)
            for name in self.branches
        }
        label = torch.as_tensor(self._archive["labels"][index], dtype=torch.long)
        return inputs, label

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None


class CachedModelReadyDataset(ModelReadyDataset):
    """In-memory variant for compact datasets such as CIFAR-10.

    The regular reader deliberately fetches samples lazily because wearable
    datasets may be large and multimodal.  Image benchmarks are small enough
    to cache, and random scalar reads from compressed HDF5 otherwise dominate
    the training time.
    """

    def __init__(self, path: Path, branches: Sequence[str]) -> None:
        super().__init__(path, branches)
        try:
            with h5py.File(path, "r") as archive:
                self._inputs = {
                    name: np.ascontiguousarray(archive[f"inputs/{name}"][:], dtype=np.float32)
                    for name in self.branches
                }
                self._labels = np.ascontiguousarray(archive["labels"][:], dtype=np.int64)
        except MemoryError as error:
            raise RuntimeError(
                "Unable to cache this dataset in memory. Retry without "
                "--cache-in-memory or use a machine with more RAM."
            ) from error

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return (
            {name: torch.from_numpy(values[index]) for name, values in self._inputs.items()},
            torch.as_tensor(self._labels[index], dtype=torch.long),
        )


class LegacyNPZDataset(Dataset):
    def __init__(self, path: Path) -> None:
        archive = np.load(path, allow_pickle=False)
        if not {"x", "y"}.issubset(archive.files):
            raise ValueError("Legacy NPZ input must contain x and y arrays.")
        self.x = archive["x"].astype(np.float32, copy=False)
        self.y = archive["y"].astype(np.int64, copy=False)
        self.split = archive["split"].astype(str) if "split" in archive.files else None
        if self.x.ndim != 3 or len(self.x) != len(self.y):
            raise ValueError(f"Expected x [samples, channels, time] aligned with y; got {self.x.shape} and {self.y.shape}.")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return {"imu": torch.from_numpy(self.x[index])}, torch.as_tensor(self.y[index], dtype=torch.long)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Processed dataset directory, data.h5, or legacy NPZ.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--encoder", help="Override the encoder declared in dataset.json.")
    parser.add_argument("--num-classes", type=int, help="Inferred from class_names/labels when omitted.")
    parser.add_argument("--embedding-dim", type=int, help="Default: value in dataset.json, otherwise 64.")
    parser.add_argument(
        "--input-channels",
        type=int,
        help="Optional channel-count check; inferred from every HDF5 input branch.",
    )
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument(
        "--cache-in-memory",
        action="store_true",
        help="Load all HDF5 inputs into RAM before training; recommended for CIFAR-10 and Fashion-MNIST.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--train-splits", default="pretrain,train,full", help="Comma-separated split priority.")
    parser.add_argument("--eval-splits", default="validation,test", help="Comma-separated evaluation split priority.")
    parser.add_argument("--log-interval", default=50, type=int)
    parser.add_argument(
        "--early-stopping-patience",
        default=5,
        type=int,
        help="Stop after this many epochs without an eval-F1 improvement (default: 5).",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        default=0.001,
        type=float,
        help="Minimum absolute eval-F1 increase required to reset patience (default: 0).",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    if (
        args.weight_decay < 0
        or args.num_workers < 0
        or args.log_interval < 0
        or args.early_stopping_patience <= 0
        or args.early_stopping_min_delta < 0
    ):
        parser.error(
            "weight-decay, num-workers, log-interval, and early-stopping-min-delta "
            "must be non-negative; early-stopping-patience must be positive"
        )
    return args


def _load(args: argparse.Namespace) -> tuple[Dataset, dict[str, Any], list[str], np.ndarray | None]:
    path = args.data
    if path.is_dir():
        data_path, descriptor_path, manifest_path = path / "data.h5", path / "dataset.json", path / "manifest.csv"
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        data_path, descriptor_path, manifest_path = path, path.with_name("dataset.json"), path.with_name("manifest.csv")
    elif path.suffix.lower() == ".npz":
        dataset = LegacyNPZDataset(path)
        return dataset, {"dataset": path.stem, "encoder": "imu_encoder_tiny", "embedding_dim": 64}, ["imu"], dataset.split
    else:
        raise ValueError("--data must be a processed directory, data.h5/.hdf5, or legacy .npz archive.")
    if not data_path.is_file():
        raise FileNotFoundError(f"Model-ready HDF5 file not found: {data_path}")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8")) if descriptor_path.is_file() else {}
    with h5py.File(data_path, "r") as archive:
        if "inputs" not in archive or "labels" not in archive:
            raise ValueError("HDF5 input must contain /inputs and /labels.")
        branches = list(archive["inputs"].keys())
    dataset_type = CachedModelReadyDataset if args.cache_in_memory else ModelReadyDataset
    dataset = dataset_type(data_path, branches)
    split = _read_manifest_splits(manifest_path, len(dataset))
    return dataset, descriptor, branches, split


def _read_manifest_splits(path: Path, length: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != length or any("split" not in row for row in rows):
        raise ValueError(f"{path} must contain one split value for every HDF5 sample.")
    return np.asarray([row["split"] for row in rows], dtype=str)


def _input_artifacts(path: Path) -> dict[str, str | None]:
    """Return the physical files used to construct the train/eval subsets."""
    if path.is_dir():
        base, data_path = path, path / "data.h5"
    else:
        base, data_path = path.parent, path
    if path.suffix.lower() == ".npz":
        return {"data": str(path.resolve()), "descriptor": None, "manifest": None}
    descriptor = base / "dataset.json"
    manifest = base / "manifest.csv"
    return {
        "data": str(data_path.resolve()),
        "descriptor": str(descriptor.resolve()) if descriptor.is_file() else None,
        "manifest": str(manifest.resolve()) if manifest.is_file() else None,
    }


def _selected_source_files(manifest_path: str | None, indices: np.ndarray) -> list[str]:
    """Extract source-file references when preprocessing retained them."""
    if not manifest_path or not indices.size:
        return []
    selected = set(map(int, indices))
    discovered: set[str] = set()

    def collect(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                collect(nested_value, str(nested_key))
        elif isinstance(value, list):
            for item in value:
                collect(item, key)
        elif isinstance(value, str) and any(token in key.lower() for token in ("file", "path", "source")):
            if value.strip():
                discovered.add(value.strip())

    with Path(manifest_path).open(newline="", encoding="utf-8") as stream:
        for position, row in enumerate(csv.DictReader(stream)):
            sample_index = int(row.get("sample_index", position))
            if sample_index not in selected:
                continue
            for key, value in row.items():
                if key == "metadata_json" and value:
                    try:
                        collect(json.loads(value))
                    except json.JSONDecodeError:
                        pass
                elif value and any(token in key.lower() for token in ("file", "path", "source")):
                    discovered.add(value.strip())
    return sorted(discovered)


def _split_summary(split: np.ndarray | None, indices: np.ndarray) -> list[str]:
    return sorted(set(split[indices].tolist())) if split is not None and indices.size else []


def _choose_indices(split: np.ndarray | None, train_names: str, eval_names: str, length: int) -> tuple[np.ndarray, np.ndarray]:
    if split is None:
        return np.arange(length), np.empty(0, dtype=np.int64)

    def first_available(names: str) -> np.ndarray:
        for name in (item.strip() for item in names.split(",")):
            indices = np.flatnonzero(split == name)
            if len(indices):
                return indices
        return np.empty(0, dtype=np.int64)

    train = first_available(train_names)
    evaluate = first_available(eval_names)
    if not len(train):
        available = ", ".join(sorted(set(split)))
        raise ValueError(f"No requested training split found; available values: {available}.")
    if np.intersect1d(train, evaluate).size:
        raise ValueError("Training and evaluation splits overlap.")
    return train, evaluate


def _all_labels(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, ModelReadyDataset):
        with h5py.File(dataset.path, "r") as archive:
            labels = np.asarray(archive["labels"])
    else:
        labels = np.asarray(dataset.y)  # type: ignore[attr-defined]
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Supervised encoder pretraining requires one integer class label per sample.")
    return labels.astype(np.int64, copy=False)


def _encoder_kwargs(name: str, branches: Sequence[str], dataset: Dataset, embedding_dim: int, input_channels: int | None) -> dict[str, int]:
    kwargs = {"embedding_dim": embedding_dim}
    if name in {"imu_encoder_tiny", "imu_encoder_tiny_tcn", "dual_wrist_imu_encoder_tiny"}:
        branch = "left_wrist" if name == "dual_wrist_imu_encoder_tiny" else "imu"
    elif name == "image_encoder_tiny_cnn":
        branch = "image"
    else:
        return kwargs

    if branch not in branches:
        raise ValueError(f"Encoder {name!r} requires branch {branch!r}.")
    if isinstance(dataset, ModelReadyDataset):
        with h5py.File(dataset.path, "r") as archive:
            channels = int(archive[f"inputs/{branch}"].shape[1])
    else:
        channels = int(dataset.x.shape[1])  # type: ignore[attr-defined]
    if input_channels is not None and input_channels != channels:
        raise ValueError(f"Dataset has {channels} channels, but --input-channels is {input_channels}.")
    kwargs["input_channels"] = channels
    return kwargs


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return result


def _forward(encoder: nn.Module, inputs: Mapping[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    # Some datasets expose auxiliary arrays (RingAuth's valid_mask) which are
    # useful to downstream algorithms but are not consumed by every encoder.
    parameters = inspect.signature(encoder.forward).parameters
    accepts_kwargs = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    values = {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in inputs.items()
        if accepts_kwargs or name in parameters
    }
    required = {
        name for name, item in parameters.items()
        if name != "self" and item.default is inspect.Parameter.empty
        and item.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    missing = required.difference(values)
    if len(missing) == 1:
        # The generic single-stream encoders expose ``forward(x)`` while the
        # model-ready files give that tensor a semantic branch name (usually
        # ``imu``).  Resolve this harmless naming difference only when there
        # is exactly one unconsumed data branch, so multimodal models still
        # require their explicit branch names.
        auxiliary = {"valid_mask", "mask", "attention_mask", "lengths"}
        candidates = [
            (name, tensor) for name, tensor in inputs.items()
            if name not in values and name not in auxiliary
        ]
        if len(candidates) == 1:
            parameter = next(iter(missing))
            values[parameter] = candidates[0][1].to(device, non_blocking=True)
            missing.clear()
    if missing:
        available = ", ".join(sorted(inputs)) or "none"
        raise ValueError(
            f"Dataset does not provide encoder inputs: {', '.join(sorted(missing))} "
            f"(available branches: {available})."
        )
    return encoder(**values)


class _ClassificationMetrics:
    """Accumulate a confusion matrix without retaining epoch predictions."""

    def __init__(self, num_classes: int) -> None:
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = logits.detach().argmax(1).cpu().numpy()
        expected = targets.detach().cpu().numpy()
        np.add.at(self.confusion, (expected, predictions), 1)

    def compute(self) -> tuple[float, float]:
        seen = int(self.confusion.sum())
        accuracy = float(np.trace(self.confusion) / seen) if seen else 0.0
        true_positive = np.diag(self.confusion).astype(np.float64)
        false_positive = self.confusion.sum(axis=0) - true_positive
        false_negative = self.confusion.sum(axis=1) - true_positive
        denominator = 2.0 * true_positive + false_positive + false_negative
        f1_per_class = np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator != 0,
        )
        # Binary F1 conventionally reports the positive class (label 1).
        # For multiclass tasks, macro F1 gives every class equal importance.
        f1 = float(f1_per_class[1] if len(f1_per_class) == 2 else f1_per_class.mean())
        return accuracy, f1


@torch.no_grad()
def _evaluate(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    num_classes: int,
) -> tuple[float | None, float | None, float | None]:
    if loader is None:
        return None, None, None
    encoder.eval(); head.eval()
    loss_sum = seen = 0
    metrics = _ClassificationMetrics(num_classes)
    for inputs, labels in loader:
        labels = labels.to(device, non_blocking=True)
        logits = head(_forward(encoder, inputs, device))
        batch = len(labels)
        loss_sum += nn.functional.cross_entropy(logits, labels).item() * batch
        metrics.update(logits, labels)
        seen += batch
    accuracy, f1 = metrics.compute()
    return loss_sum / seen, accuracy, f1


def main() -> None:
    args = _arguments()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset, descriptor, branches, split = _load(args)
    labels = _all_labels(dataset)
    if len(labels) == 0 or labels.min() < 0:
        raise ValueError("Labels must be non-negative and the dataset must not be empty.")
    num_classes = args.num_classes or len(descriptor.get("class_names", [])) or int(labels.max()) + 1
    if labels.max() >= num_classes:
        raise ValueError(f"Label {labels.max()} is outside [0, {num_classes}).")
    encoder_name = args.encoder or descriptor.get("encoder")
    if not encoder_name:
        raise ValueError("No encoder was specified and dataset.json does not declare one.")
    embedding_dim = args.embedding_dim or int(descriptor.get("embedding_dim", 64))
    train_indices, eval_indices = _choose_indices(split, args.train_splits, args.eval_splits, len(dataset))
    if not len(eval_indices):
        available = ", ".join(sorted(set(split))) if split is not None else "no split metadata"
        raise ValueError(
            "Early stopping requires a non-empty evaluation split, but none of "
            f"[{args.eval_splits}] was found (available: {available})."
        )
    device = _device(args.device)
    encoder_kwargs = _encoder_kwargs(
        str(encoder_name), branches, dataset, embedding_dim, args.input_channels
    )
    encoder = build_encoder(str(encoder_name), **encoder_kwargs).to(device)
    head = MLPClassifier(embedding_dim, num_classes, hidden_dims=[64]).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    common = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(dataset, train_indices.tolist()), shuffle=True, generator=generator, **common)
    eval_loader = DataLoader(Subset(dataset, eval_indices.tolist()), shuffle=False, **common) if len(eval_indices) else None

    print(f"Pretraining {encoder_name} on {descriptor.get('dataset', args.data.stem)}")
    cached = " | inputs cached in RAM" if args.cache_in_memory else ""
    print(
        f"  data: {args.data} | stored samples: {len(dataset):,} "
        f"({len(train_indices):,} train, {len(eval_indices):,} eval){cached}"
    )
    print(
        f"  branches: {', '.join(branches)} | classes: {num_classes} | "
        f"embedding: {embedding_dim} | encoder args: {encoder_kwargs}"
    )
    print(f"  device: {device} | epochs: {args.epochs} | batch size: {args.batch_size}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output.with_name(args.output.stem + "_training_log.csv")
    summary_path = args.output.with_name(args.output.stem + "_training_summary.json")
    f1_kind = "binary" if num_classes == 2 else "macro"
    artifacts = _input_artifacts(args.data)
    training_summary: dict[str, Any] = {
        "format_version": 1,
        "status": "running",
        "checkpoint": str(args.output.resolve()),
        "training_log": str(metrics_path.resolve()),
        "model": {
            "encoder": str(encoder_name),
            "encoder_kwargs": encoder_kwargs,
            "embedding_dim": embedding_dim,
            "input_branches": branches,
            "num_classes": num_classes,
            "classifier": "MLPClassifier(hidden_dims=[64])",
        },
        "training_parameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "loss": "cross_entropy",
            "seed": args.seed,
            "num_workers": args.num_workers,
            "cache_in_memory": args.cache_in_memory,
            "device_requested": args.device,
            "device_resolved": str(device),
            "train_split_priority": [item.strip() for item in args.train_splits.split(",") if item.strip()],
            "eval_split_priority": [item.strip() for item in args.eval_splits.split(",") if item.strip()],
            "f1_average": f1_kind,
            "log_interval": args.log_interval,
            "early_stopping": {
                "monitor": "eval_f1",
                "mode": "max",
                "patience": args.early_stopping_patience,
                "min_delta": args.early_stopping_min_delta,
                "restore_best": True,
            },
        },
        "dataset": {
            "name": descriptor.get("dataset", args.data.stem),
            "requested_path": str(args.data),
            "files": artifacts,
            "total_samples": len(dataset),
            "training": {
                "splits": _split_summary(split, train_indices),
                "num_samples": int(len(train_indices)),
                "data_file": artifacts["data"],
                "source_files": _selected_source_files(artifacts["manifest"], train_indices),
            },
            "evaluation": {
                "splits": _split_summary(split, eval_indices),
                "num_samples": int(len(eval_indices)),
                "data_file": artifacts["data"] if len(eval_indices) else None,
                "source_files": _selected_source_files(artifacts["manifest"], eval_indices),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
    }
    summary_path.write_text(json.dumps(training_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metric_fields = [
        "epoch", "epoch_seconds", "f1_average", "train_loss", "train_accuracy",
        "train_f1", "eval_loss", "eval_accuracy", "eval_f1", "is_best",
        "epochs_without_improvement",
    ]
    started = time.perf_counter()
    best_eval_f1 = float("-inf")
    early_stopping_reference = float("-inf")
    best_eval_loss: float | None = None
    best_eval_accuracy: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    stopped_early = False
    epochs_completed = 0
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_stream:
        metrics_writer = csv.DictWriter(metrics_stream, fieldnames=metric_fields)
        metrics_writer.writeheader()
        metrics_stream.flush()
        for epoch in range(1, args.epochs + 1):
            encoder.train(); head.train()
            loss_sum = seen = 0
            train_metrics = _ClassificationMetrics(num_classes)
            epoch_started = time.perf_counter()
            for step, (inputs, labels_batch) in enumerate(train_loader, 1):
                labels_batch = labels_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = head(_forward(encoder, inputs, device))
                loss = nn.functional.cross_entropy(logits, labels_batch)
                loss.backward(); optimizer.step()
                batch = len(labels_batch)
                loss_sum += loss.item() * batch
                train_metrics.update(logits, labels_batch)
                seen += batch
                if args.log_interval and (step % args.log_interval == 0 or step == len(train_loader)):
                    partial_acc, partial_f1 = train_metrics.compute()
                    print(
                        f"  epoch {epoch:>3}/{args.epochs} | batch {step:>4}/{len(train_loader)} "
                        f"| loss {loss_sum/seen:.4f} | acc {partial_acc:.3f} | f1 {partial_f1:.3f}",
                        flush=True,
                    )
            epoch_seconds = time.perf_counter() - epoch_started
            train_acc, train_f1 = train_metrics.compute()
            eval_loss, eval_acc, eval_f1 = _evaluate(encoder, head, eval_loader, device, num_classes)
            assert eval_loss is not None and eval_acc is not None and eval_f1 is not None
            is_best = best_epoch is None or eval_f1 > best_eval_f1
            if is_best:
                best_eval_f1 = eval_f1
                best_eval_loss = eval_loss
                best_eval_accuracy = eval_acc
                best_epoch = epoch
                # Save only the encoder. Writing at every improvement also
                # preserves the best checkpoint if a later epoch is interrupted.
                best_state = {
                    name: value.detach().cpu()
                    for name, value in encoder.state_dict().items()
                }
                torch.save(best_state, args.output)
            significant_improvement = (
                early_stopping_reference == float("-inf")
                or eval_f1 > early_stopping_reference + args.early_stopping_min_delta
            )
            if significant_improvement:
                early_stopping_reference = eval_f1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            epochs_completed = epoch
            metrics_writer.writerow({
                "epoch": epoch,
                "epoch_seconds": f"{epoch_seconds:.6f}",
                "f1_average": f1_kind,
                "train_loss": f"{loss_sum/seen:.8f}",
                "train_accuracy": f"{train_acc:.8f}",
                "train_f1": f"{train_f1:.8f}",
                "eval_loss": f"{eval_loss:.8f}",
                "eval_accuracy": f"{eval_acc:.8f}",
                "eval_f1": f"{eval_f1:.8f}",
                "is_best": str(is_best).lower(),
                "epochs_without_improvement": epochs_without_improvement,
            })
            metrics_stream.flush()
            message = (
                f"Epoch {epoch:>3}/{args.epochs} completed in {epoch_seconds:.1f}s "
                f"| train loss {loss_sum/seen:.4f} | train acc {train_acc:.3f} | train f1 {train_f1:.3f}"
            )
            message += f" | eval loss {eval_loss:.4f} | eval acc {eval_acc:.3f} | eval f1 {eval_f1:.3f}"
            if is_best:
                message += " | new best"
            print(message, flush=True)
            if epochs_without_improvement >= args.early_stopping_patience:
                stopped_early = True
                print(
                    f"Early stopping at epoch {epoch}: eval_f1 did not improve by more than "
                    f"{args.early_stopping_min_delta:g} for {args.early_stopping_patience} epochs. "
                    f"Best epoch: {best_epoch} (eval_f1={best_eval_f1:.4f}).",
                    flush=True,
                )
                break

    # The checkpoint was updated only when validation improved. It remains a
    # plain state_dict for compatibility with encoder.load_state_dict().
    assert best_epoch is not None and best_eval_loss is not None and best_eval_accuracy is not None
    metadata = {
        "format_version": 4, "encoder_name": str(encoder_name), "encoder_kwargs": encoder_kwargs,
        "embedding_dim": embedding_dim,
        "branches": branches, "num_classes": num_classes,
        "dataset": descriptor.get("dataset", args.data.stem), "seed": args.seed,
        "training_log": metrics_path.name, "training_summary": summary_path.name,
        "f1_average": f1_kind, "selection_metric": "eval_f1",
        "best_epoch": best_epoch, "best_eval_f1": best_eval_f1,
    }
    metadata_path = args.output.with_name(args.output.name + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    training_summary["status"] = "completed"
    training_summary["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    training_summary["result"] = {
        "epochs_completed": epochs_completed,
        "stopped_early": stopped_early,
        "stop_reason": "early_stopping" if stopped_early else "max_epochs_reached",
        "best_epoch": best_epoch,
        "best_eval_loss": best_eval_loss,
        "best_eval_accuracy": best_eval_accuracy,
        "best_eval_f1": best_eval_f1,
        "checkpoint_contains": "best_encoder_only",
    }
    summary_path.write_text(json.dumps(training_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if isinstance(dataset, ModelReadyDataset):
        dataset.close()
    print(f"Saved best encoder checkpoint (epoch {best_epoch}) to: {args.output}")
    print(f"Saved training summary to: {summary_path}")
    print(f"Saved checkpoint metadata to: {metadata_path}")
    print(f"Saved training metrics to: {metrics_path}")
    print(f"Total pretraining time: {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
