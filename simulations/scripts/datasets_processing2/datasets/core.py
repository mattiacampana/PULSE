"""Shared records, signal processing, splitting, and output writing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np
import pandas as pd
from scipy import signal

from .config import SplitConfig
from .reporting import REPORTER, describe_paths


@dataclass
class Sample:
    """A single model-ready example with one or more encoder inputs."""

    inputs: Mapping[str, np.ndarray]
    label: int | float
    subject_id: str
    session_id: str = ""
    device_id: str = ""
    timestamp: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


def sliding_windows(values: np.ndarray, size: int, stride: int) -> Iterable[tuple[int, np.ndarray]]:
    """Yield complete channel-first windows without padding continuous data."""
    if size <= 0 or stride <= 0:
        raise ValueError("Window size and stride must be positive.")
    for start in range(0, values.shape[-1] - size + 1, stride):
        yield start, values[..., start:start + size]


def resample(values: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    """Polyphase-resample a channel-first signal to a deterministic length."""
    if source_rate == target_rate:
        return values.astype(np.float32, copy=False)
    from fractions import Fraction
    ratio = Fraction(target_rate / source_rate).limit_denominator(1000)
    return signal.resample_poly(values, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)


def robust_window_normalize(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Normalize each channel using its own window mean and standard deviation."""
    mean = values.mean(axis=-1, keepdims=True)
    std = values.std(axis=-1, keepdims=True)
    return ((values - mean) / np.maximum(std, epsilon)).astype(np.float32)


def allocate_subjects(subjects: Iterable[str], request: SplitConfig) -> dict[str, list[str] | None]:
    """Create reproducible, disjoint subject groups or a single full group."""
    unique = np.asarray(sorted(set(map(str, subjects))))
    if not request.enabled:
        return {"full": unique.tolist(), "pretrain": None, "validation": None, "simulation": None, "unused": []}
    rng = np.random.default_rng(request.seed)
    shuffled = unique[rng.permutation(len(unique))].tolist()
    n_pre = request.pretrain_users or 0
    n_val = request.validation_users or 0
    n_sim = len(unique) - n_pre - n_val if request.simulation_users == "remaining" else (request.simulation_users or 0)
    if not all(isinstance(value, int) and value >= 0 for value in (n_pre, n_val, n_sim)):
        raise ValueError("Split counts must be non-negative integers or simulation_users: remaining.")
    if n_pre + n_val + n_sim > len(unique):
        raise ValueError(f"Requested {n_pre + n_val + n_sim} users, but only {len(unique)} valid subjects are available.")
    return {
        "full": None, "pretrain": shuffled[:n_pre], "validation": shuffled[n_pre:n_pre + n_val],
        "simulation": shuffled[n_pre + n_val:n_pre + n_val + n_sim], "unused": shuffled[n_pre + n_val + n_sim:],
    }


def write_dataset(
    samples: list[Sample],
    output_dir: Path,
    descriptor: dict[str, object],
    split: SplitConfig,
    *,
    explicit_splits: list[str] | None = None,
    split_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate and atomically materialize HDF5, manifest, schema, and split files."""
    if not samples:
        raise ValueError("No model-ready samples were generated.")
    keys = tuple(samples[0].inputs)
    for sample in samples:
        if tuple(sample.inputs) != keys:
            raise ValueError("All samples must expose the same ordered input branches.")
        if any(not np.isfinite(value).all() for value in sample.inputs.values()):
            raise ValueError("A model-ready input contains NaN or infinity.")
    shapes = {key: samples[0].inputs[key].shape for key in keys}
    for sample in samples:
        if any(sample.inputs[key].shape != shapes[key] for key in keys):
            raise ValueError("Each input branch must have a fixed shape; use padding and a mask for events.")

    dataset = str(descriptor.get("dataset", "dataset"))
    REPORTER.info(dataset, "VALIDATE", f"Checking {len(samples):,} samples for consistent branches, shapes, and finite values")
    allowed_splits = {"pretrain", "validation", "simulation", "unused"}
    if explicit_splits is None:
        groups = allocate_subjects((sample.subject_id for sample in samples), split)
        assignments = None
        split_mode = "subject" if split.enabled else "full"
        selection_unit = "subject"
    else:
        if len(explicit_splits) != len(samples):
            raise ValueError("Explicit split assignments must contain one entry per sample.")
        invalid = sorted(set(explicit_splits) - allowed_splits)
        if invalid:
            raise ValueError(f"Unsupported explicit split names: {invalid}")
        assignments = list(explicit_splits)
        groups = {
            name: sorted({sample.subject_id for sample, assigned in zip(samples, assignments) if assigned == name})
            for name in allowed_splits
        }
        groups["full"] = None
        split_mode = "explicit"
        selection_unit = str((split_metadata or {}).get("selection_unit", "sample"))
    split_counts = {name: len(members or []) for name, members in groups.items()}
    split_text = ", ".join(f"{name}={count}" for name, count in split_counts.items() if members_exist(name, groups))
    REPORTER.info(dataset, "SPLIT", f"mode={'subject' if split.enabled else 'full'}; users: {split_text}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "data.h5.tmp"
    with h5py.File(temporary, "w") as archive:
        inputs = archive.create_group("inputs")
        for key in keys:
            inputs.create_dataset(key, data=np.stack([sample.inputs[key] for sample in samples]), compression="gzip", shuffle=True)
        archive.create_dataset("labels", data=np.asarray([sample.label for sample in samples]))
    temporary.replace(output_dir / "data.h5")
    REPORTER.detail(dataset, "TENSORS", ", ".join(f"{key}=[{len(samples)}, {', '.join(map(str, shapes[key]))}]" for key in keys))

    membership = {}
    for name, members in groups.items():
        if members:
            membership.update({subject: name for subject in members})
    manifest = pd.DataFrame({
        "sample_index": np.arange(len(samples)), "subject_id": [item.subject_id for item in samples],
        "session_id": [item.session_id for item in samples], "device_id": [item.device_id for item in samples],
        "timestamp": [item.timestamp for item in samples], "label": [item.label for item in samples],
        "split": assignments if assignments is not None else [membership.get(item.subject_id, "unused") for item in samples],
        "metadata_json": [json.dumps(item.metadata, sort_keys=True) for item in samples],
    })
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    schema = dict(descriptor)
    schema.update({"format_version": 1, "storage": "hdf5", "input_shapes": {key: list(shape) for key, shape in shapes.items()}, "num_samples": len(samples)})
    _write_json(output_dir / "dataset.json", schema)
    split_document = {
        "seed": split.seed,
        "selection_unit": selection_unit,
        "split_mode": split_mode,
        **groups,
    }
    if split_metadata:
        split_document.update(split_metadata)
    _write_json(output_dir / "splits.json", split_document)
    _write_json(output_dir / "preprocessing_report.json", {
        "status": "valid", "num_samples": len(samples), "num_subjects": manifest.subject_id.nunique(),
        "label_counts": {str(key): int(value) for key, value in manifest.label.value_counts().sort_index().items()},
        "contains_non_finite": False,
    })
    artifacts = [output_dir / name for name in ("data.h5", "manifest.csv", "dataset.json", "splits.json", "preprocessing_report.json")]
    sample_split_counts = {str(key): int(value) for key, value in manifest["split"].value_counts().items()}
    REPORTER.info(dataset, "WRITE", describe_paths(artifacts))
    REPORTER.info(dataset, "SUMMARY", f"{manifest.subject_id.nunique():,} users; {len(samples):,} samples; samples by split: " + ", ".join(f"{key}={value:,}" for key, value in sorted(sample_split_counts.items())))
    return {"num_samples": len(samples), "num_subjects": int(manifest.subject_id.nunique()), "sample_split_counts": sample_split_counts, "artifacts": artifacts}


def members_exist(name: str, groups: Mapping[str, list[str] | None]) -> bool:
    """Include meaningful zero-valued split groups while hiding inactive full mode fields."""
    return groups[name] is not None


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
