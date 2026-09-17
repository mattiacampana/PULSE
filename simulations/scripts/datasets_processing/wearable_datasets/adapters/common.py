"""Reusable adapter-level parsing and window construction utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wearable_datasets.core import Sample, robust_window_normalize, sliding_windows
from wearable_datasets.reporting import REPORTER


def column(frame: pd.DataFrame, *candidates: str) -> str:
    """Resolve a case-insensitive column alias."""
    lookup = {str(item).strip().lower(): str(item) for item in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Missing column; expected one of {candidates} in {list(frame.columns)}")


def window_labeled_frame(
    frame: pd.DataFrame, channels: list[str], label_column: str, subject: str, session: str,
    device: str, size: int, stride: int, label_map: dict[str, int], normalize: bool = True,
) -> list[Sample]:
    """Window contiguous runs while never crossing a label boundary."""
    result: list[Sample] = []
    labels = frame[label_column].astype(str).str.strip().str.lower()
    run = (labels != labels.shift()).cumsum()
    for _, group in frame.assign(_label=labels, _run=run).groupby("_run", sort=False):
        label = group["_label"].iloc[0]
        if label not in label_map:
            continue
        values = group[channels].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(np.float32).T
        for start, window in sliding_windows(values, size, stride):
            result.append(Sample({"imu": robust_window_normalize(window) if normalize else window}, label_map[label], subject, session, device, float(start)))
    return result


def require_local_files(raw_dir: Path, patterns: tuple[str, ...], dataset: str, instructions: str) -> Path:
    """Return a local source only when at least one expected file is present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    matches = [path for pattern in patterns for path in raw_dir.rglob(pattern)]
    if not matches:
        raise FileNotFoundError(f"{dataset} source files were not found in {raw_dir}. {instructions}")
    REPORTER.info(dataset, "LOCAL", f"Found {len(set(matches)):,} matching source files in {raw_dir}")
    REPORTER.detail(dataset, "PATTERNS", ", ".join(patterns))
    return raw_dir
