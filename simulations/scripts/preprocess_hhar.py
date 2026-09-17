"""
Convert raw HHAR accelerometer records into simulator-ready HAR windows.

The input is the ``Phones_accelerometer.csv`` file from HHAR.  Windows are
grouped by user, device, and activity so that a window never mixes recordings
from different devices, labels, or temporal segments.  The resulting archive
uses the simulator's node identity (the user), while device remains a
preprocessing safeguard rather than a simulated node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# This order matches the six activities used in the preliminary HHAR analysis.
DEFAULT_LABEL_MAP = {
    "sit": 0,
    "stairsup": 1,
    "stairsdown": 2,
    "walk": 3,
    "stand": 4,
    "bike": 5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw HHAR Phones_accelerometer.csv file.")
    parser.add_argument("--output", required=True, type=Path, help="Output .npz file containing x, y, and node_ids.")
    parser.add_argument("--window-size", default=128, type=int, help="Samples per window.")
    parser.add_argument("--stride", default=128, type=int, help="Window stride in samples; defaults to non-overlapping windows.")
    parser.add_argument("--max-gap-ms", default=None, type=float, help="Discard windows containing a temporal gap larger than this value.")
    parser.add_argument("--normalization", choices=("none", "per_window"), default="per_window")
    parser.add_argument("--label-map", type=Path, default=None, help="Optional JSON object mapping raw activity names to integer labels.")
    args = parser.parse_args()

    if args.window_size <= 0 or args.stride <= 0:
        raise ValueError("window-size and stride must be positive.")

    frame = pd.read_csv(args.input)
    columns = _resolve_columns(frame)
    label_map = _load_label_map(args.label_map)
    frame = _canonicalize(frame, columns, label_map)
    x, y, node_ids = _window_records(frame, args.window_size, args.stride, args.max_gap_ms, args.normalization)
    if not len(x):
        raise ValueError("No valid windows were generated. Check labels, window size, and temporal-gap threshold.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=x.astype(np.float32),
        y=y.astype(np.int64),
        node_ids=node_ids.astype(str),
        class_names=np.asarray([name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]),
    )
    print(f"Saved {len(x)} windows for {len(np.unique(node_ids))} users to {args.output}.")


def _resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Resolve the small set of column-name variants encountered in HHAR exports."""
    candidates = {
        "x": ("x", "X"), "y": ("y", "Y"), "z": ("z", "Z"),
        "user": ("User", "user"), "label": ("gt", "label", "activity"),
        "device": ("Device", "device", "Model", "model"),
        "time": ("Creation_Time", "Arrival_Time", "time", "timestamp"),
    }
    resolved = {}
    for key, options in candidates.items():
        match = next((column for column in options if column in frame.columns), None)
        if match is None and key == "device":
            # A missing device column is safe: user and label still prevent mixing activities.
            continue
        if match is None:
            raise ValueError(f"Missing HHAR column for {key}. Expected one of {options}.")
        resolved[key] = match
    return resolved


def _load_label_map(path: Path | None) -> dict[str, int]:
    if path is None:
        return DEFAULT_LABEL_MAP.copy()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return {str(name).strip().lower(): int(value) for name, value in loaded.items()}


def _canonicalize(frame: pd.DataFrame, columns: dict[str, str], label_map: dict[str, int]) -> pd.DataFrame:
    data = pd.DataFrame({
        "x": pd.to_numeric(frame[columns["x"]], errors="coerce"),
        "y": pd.to_numeric(frame[columns["y"]], errors="coerce"),
        "z": pd.to_numeric(frame[columns["z"]], errors="coerce"),
        "user": frame[columns["user"]].astype(str),
        "label": frame[columns["label"]].astype(str).str.strip().str.lower(),
        "time": pd.to_numeric(frame[columns["time"]], errors="coerce"),
    })
    data["device"] = frame[columns["device"]].astype(str) if "device" in columns else "unknown"
    data["target"] = data["label"].map(label_map)
    return data.dropna(subset=["x", "y", "z", "user", "time", "target"]).sort_values(["user", "device", "label", "time"])


def _window_records(frame: pd.DataFrame, window_size: int, stride: int, max_gap_ms: float | None, normalization: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    owners: list[str] = []
    for (user, device, label), group in frame.groupby(["user", "device", "target"], sort=False):
        samples = group[["x", "y", "z"]].to_numpy(dtype=np.float32)
        times = group["time"].to_numpy(dtype=np.float64)
        for start in range(0, len(group) - window_size + 1, stride):
            stop = start + window_size
            if max_gap_ms is not None and np.diff(times[start:stop]).max(initial=0.0) > max_gap_ms:
                continue
            window = samples[start:stop].T  # [3, 128], compatible with UCI HAR total acceleration.
            if normalization == "per_window":
                window = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-6)
            windows.append(window)
            labels.append(int(label))
            owners.append(str(user))
    return np.stack(windows), np.asarray(labels), np.asarray(owners)


if __name__ == "__main__":
    main()
