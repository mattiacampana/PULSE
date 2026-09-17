"""Download and preprocess the UCI HAR source and HHAR target datasets.

This is a standalone script: it intentionally has no dependency on
``preprocess_hhar.py``.

The script creates the two NPZ archives consumed by the simulator:

* ``uci_har.npz``: source windows and labels for encoder pretraining;
* ``hhar_by_user.npz``: target windows, labels, and simulated user IDs.

Both archives are standardized *per window and per accelerometer axis*.  This
is essential because UCI HAR and HHAR use different sensor devices and their
raw accelerometer values have different physical scales.  The transformation
does not use labels or dataset-wide target statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


UCI_HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
HHAR_URL = "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"
_NORMALIZATION = "per_window_per_axis_zscore"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("uci_har", "hhar", "all"), default="all")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory for downloaded archives and extracted files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Directory for simulator-ready NPZ archives.")
    parser.add_argument("--stats-dir", type=Path, default=None, help="Directory for dataset statistics (defaults to --output-dir).")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-gap-ms", type=float, default=None, help="Optional HHAR maximum gap within a window.")
    parser.add_argument("--force-download", action="store_true", help="Download archives again even if present locally.")
    parser.add_argument("--force-preprocess", action="store_true", help="Overwrite existing NPZ archives.")
    args = parser.parse_args()

    if args.window_size != 128:
        raise ValueError("The current CNN-1D and UCI HAR source data use 128-sample windows.")
    if args.stride <= 0:
        raise ValueError("stride must be positive.")

    raw_dir, output_dir = args.raw_dir.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = (args.stats_dir or output_dir).resolve()
    stats_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "uci_har.npz"
    target_path = output_dir / "hhar_by_user.npz"

    if args.dataset in {"uci_har", "all"}:
        _prepare_uci_har(raw_dir, source_path, stats_dir, args.force_download, args.force_preprocess)
    if args.dataset in {"hhar", "all"}:
        _prepare_hhar(raw_dir, target_path, stats_dir, args, args.force_download, args.force_preprocess)


def _prepare_uci_har(raw_dir: Path, output_path: Path, stats_dir: Path, force_download: bool, force_preprocess: bool) -> None:
    if output_path.exists() and not force_preprocess:
        archive = np.load(output_path, allow_pickle=False)
        if "normalization" in archive.files and str(archive["normalization"]) == _NORMALIZATION:
            print(f"Reusing {output_path}.")
            _write_uci_statistics(
                stats_dir, archive["y"],
                archive["subject_ids"] if "subject_ids" in archive.files else None,
                archive["split"].astype(str) if "split" in archive.files else None,
                _archive_split_counts(archive),
                class_names=archive["class_names"].astype(str).tolist() if "class_names" in archive.files else _uci_class_names(),
                reused_archive="subject_ids" not in archive.files,
            )
            return
        raise ValueError(f"{output_path} was generated with an older preprocessing scheme; rerun with --force-preprocess.")

    dataset_dir = _ensure_dataset(raw_dir / "uci_har", UCI_HAR_URL, "UCI HAR Dataset", force_download)
    x_train, y_train, subjects_train = _load_uci_split(dataset_dir, "train")
    x_test, y_test, subjects_test = _load_uci_split(dataset_dir, "test")
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    subjects = np.concatenate((subjects_train, subjects_test), axis=0)
    split = np.concatenate((np.repeat("train", len(y_train)), np.repeat("test", len(y_test))))
    x = _normalize_windows(x)
    np.savez_compressed(
        output_path, x=x.astype(np.float32), y=y.astype(np.int64),
        subject_ids=subjects.astype(np.int64), split=split.astype(str),
        class_names=np.asarray(_uci_class_names()), normalization=np.asarray(_NORMALIZATION),
    )
    _write_uci_statistics(stats_dir, y, subjects, split, (len(y_train), len(y_test)), _uci_class_names())
    print(f"Saved {len(x)} UCI HAR windows to {output_path}.")


def _prepare_hhar(raw_dir: Path, output_path: Path, stats_dir: Path, args: argparse.Namespace, force_download: bool, force_preprocess: bool) -> None:
    if output_path.exists() and not force_preprocess:
        print(f"Reusing {output_path}.")
        archive = np.load(output_path, allow_pickle=False)
        if "normalization" not in archive.files or str(archive["normalization"]) != _NORMALIZATION:
            raise ValueError(f"{output_path} was generated with an older preprocessing scheme; rerun with --force-preprocess.")
        class_names = archive["class_names"].astype(str).tolist() if "class_names" in archive.files else _hhar_class_names()
        _write_hhar_statistics(stats_dir, archive["y"], archive["node_ids"].astype(str), None, class_names, args)
        return
    hhar_dir = _ensure_dataset(raw_dir / "hhar", HHAR_URL, None, force_download)
    csv_path = next(hhar_dir.rglob("Phones_accelerometer.csv"), None)
    if csv_path is None:
        raise FileNotFoundError(f"Phones_accelerometer.csv was not found beneath {hhar_dir}.")

    frame = pd.read_csv(csv_path)
    data = _canonicalize_hhar(frame, _resolve_hhar_columns(frame), _hhar_label_map())
    x, y, node_ids = _window_hhar_records(data, args.window_size, args.stride, args.max_gap_ms)
    if not len(x):
        raise ValueError("No valid HHAR windows were generated.")
    x = _normalize_windows(x)
    class_names = np.asarray([name for name, _ in sorted(_hhar_label_map().items(), key=lambda item: item[1])])
    np.savez_compressed(output_path, x=x.astype(np.float32), y=y.astype(np.int64), node_ids=node_ids.astype(str), class_names=class_names, normalization=np.asarray(_NORMALIZATION))
    devices_per_user = data.groupby("user")["device"].agg(lambda devices: sorted(set(devices.astype(str))))
    _write_hhar_statistics(stats_dir, y, node_ids.astype(str), devices_per_user, class_names.astype(str).tolist(), args)
    print(f"Saved {len(x)} HHAR windows for {len(np.unique(node_ids))} users to {output_path}.")


def _ensure_dataset(destination: Path, url: str, expected_directory: str | None, force_download: bool) -> Path:
    if destination.exists() and any(destination.rglob("*.csv" if expected_directory is None else "total_acc_x_train.txt")) and not force_download:
        return _find_root(destination, expected_directory)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "dataset.zip"
    if force_download or not archive_path.exists():
        print(f"Downloading {url} ...")
        with urllib.request.urlopen(url) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output)
    _extract_all_zips(archive_path, destination)
    return _find_root(destination, expected_directory)


def _extract_all_zips(archive_path: Path, destination: Path) -> None:
    """Extract the official archive and any nested ZIP archives it contains."""
    pending = [archive_path]
    extracted: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in extracted:
            continue
        with zipfile.ZipFile(current) as archive:
            archive.extractall(destination)
        extracted.add(current)
        pending.extend(path for path in destination.rglob("*.zip") if path not in extracted)


def _find_root(destination: Path, expected_directory: str | None) -> Path:
    if expected_directory is not None:
        found = next((path for path in destination.rglob(expected_directory) if path.is_dir()), None)
        if found is None:
            raise FileNotFoundError(f"{expected_directory} was not found beneath {destination}.")
        return found
    return destination


def _load_uci_split(dataset_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inertial = dataset_dir / split / "Inertial Signals"
    channels = [np.loadtxt(inertial / f"total_acc_{axis}_{split}.txt", dtype=np.float32) for axis in ("x", "y", "z")]
    x = np.stack(channels, axis=1)
    y = np.loadtxt(dataset_dir / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(dataset_dir / split / f"subject_{split}.txt", dtype=np.int64)
    return x, y, subjects


def _normalize_windows(x: np.ndarray) -> np.ndarray:
    """Z-score each channel independently within every [3, time] window."""
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def _hhar_label_map() -> dict[str, int]:
    """Return the activity mapping shared by HHAR preprocessing and simulation."""
    return {"sit": 0, "stairsup": 1, "stairsdown": 2, "walk": 3, "stand": 4, "bike": 5}


def _resolve_hhar_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Resolve known HHAR column-name variants without relying on another script."""
    candidates = {
        "x": ("x", "X"), "y": ("y", "Y"), "z": ("z", "Z"),
        "user": ("User", "user"), "label": ("gt", "label", "activity"),
        "device": ("Device", "device", "Model", "model"),
        "time": ("Creation_Time", "Arrival_Time", "time", "timestamp"),
    }
    resolved: dict[str, str] = {}
    for key, options in candidates.items():
        column = next((name for name in options if name in frame.columns), None)
        if column is None and key == "device":
            continue
        if column is None:
            raise ValueError(f"Missing HHAR column for {key}. Expected one of {options}.")
        resolved[key] = column
    return resolved


def _canonicalize_hhar(frame: pd.DataFrame, columns: dict[str, str], label_map: dict[str, int]) -> pd.DataFrame:
    """Keep valid accelerometer records and give them a stable internal schema."""
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
    return data.dropna(subset=["x", "y", "z", "user", "time", "target"]).sort_values(
        ["user", "device", "label", "time"]
    )


def _window_hhar_records(frame: pd.DataFrame, window_size: int, stride: int, max_gap_ms: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create [channels, samples] windows without mixing users, devices, or activities."""
    windows: list[np.ndarray] = []
    labels: list[int] = []
    owners: list[str] = []
    for (user, _device, label), group in frame.groupby(["user", "device", "target"], sort=False):
        samples = group[["x", "y", "z"]].to_numpy(dtype=np.float32)
        times = group["time"].to_numpy(dtype=np.float64)
        for start in range(0, len(group) - window_size + 1, stride):
            stop = start + window_size
            if max_gap_ms is not None and np.diff(times[start:stop]).max(initial=0.0) > max_gap_ms:
                continue
            windows.append(samples[start:stop].T)
            labels.append(int(label))
            owners.append(str(user))
    if not windows:
        return np.empty((0, 3, window_size), dtype=np.float32), np.empty(0, dtype=np.int64), np.empty(0, dtype=str)
    return np.stack(windows), np.asarray(labels), np.asarray(owners)


def _uci_class_names() -> list[str]:
    return ["walk", "walk_upstairs", "walk_downstairs", "sit", "stand", "lay"]


def _hhar_class_names() -> list[str]:
    return ["sit", "stairsup", "stairsdown", "walk", "stand", "bike"]


def _archive_split_counts(archive: np.lib.npyio.NpzFile) -> tuple[int, int] | None:
    if "split" not in archive.files:
        return None
    split = archive["split"].astype(str)
    return int(np.sum(split == "train")), int(np.sum(split == "test"))


def _write_uci_statistics(
    stats_dir: Path,
    labels: np.ndarray,
    subject_ids: np.ndarray | None,
    split: np.ndarray | None,
    split_counts: tuple[int, int] | None,
    class_names: list[str],
    reused_archive: bool = False,
) -> None:
    """Write source-dataset summaries; UCI subjects are analysis-only nodes."""
    class_rows = _class_rows("uci_har", labels, class_names, split)
    _write_csv(stats_dir / "uci_har_class_summary.csv", class_rows)
    summary = {
        "dataset": "uci_har", "num_windows": int(len(labels)), "num_classes": len(class_names),
        "class_names": class_names, "normalization": _NORMALIZATION,
        "split_windows": {"train": split_counts[0], "test": split_counts[1]} if split_counts else None,
        "statistics_from_reused_legacy_archive": reused_archive,
    }
    if subject_ids is not None and split is not None:
        rows = []
        for subject in sorted(np.unique(subject_ids)):
            mask = subject_ids == subject
            rows.append(_node_row(str(subject), labels[mask], class_names, {"source_split": _split_description(split[mask])}))
        _write_csv(stats_dir / "uci_har_node_summary.csv", rows)
        summary["num_subjects"] = int(len(np.unique(subject_ids)))
    else:
        summary["num_subjects"] = None
    _write_json(stats_dir / "uci_har_dataset_summary.json", summary)


def _write_hhar_statistics(
    stats_dir: Path,
    labels: np.ndarray,
    node_ids: np.ndarray,
    devices_per_user: object,
    class_names: list[str],
    args: argparse.Namespace,
) -> None:
    """Write target summaries where each row corresponds to one simulated user node."""
    _write_csv(stats_dir / "hhar_class_summary.csv", _class_rows("hhar", labels, class_names, None))
    rows = []
    for node_id in sorted(np.unique(node_ids)):
        device_list = []
        if devices_per_user is not None and node_id in devices_per_user.index:
            device_list = devices_per_user.loc[node_id]
        rows.append(_node_row(node_id, labels[node_ids == node_id], class_names, {
            "num_devices": len(device_list), "devices": ";".join(device_list),
        }))
    _write_csv(stats_dir / "hhar_node_summary.csv", rows)
    _write_json(stats_dir / "hhar_dataset_summary.json", {
        "dataset": "hhar", "num_windows": int(len(labels)), "num_nodes": int(len(rows)),
        "node_definition": "user", "num_classes": len(class_names), "class_names": class_names,
        "window_size": args.window_size, "stride": args.stride, "max_gap_ms": args.max_gap_ms,
        "normalization": _NORMALIZATION, "device_metadata_available": devices_per_user is not None,
    })


def _class_rows(dataset: str, labels: np.ndarray, class_names: list[str], split: np.ndarray | None) -> list[dict[str, object]]:
    rows = []
    for class_id, class_name in enumerate(class_names):
        row: dict[str, object] = {"dataset": dataset, "class_id": class_id, "class_name": class_name, "num_windows": int(np.sum(labels == class_id))}
        if split is not None:
            row.update({f"num_windows_{name}": int(np.sum((labels == class_id) & (split == name))) for name in ("train", "test")})
        rows.append(row)
    return rows


def _node_row(node_id: str, labels: np.ndarray, class_names: list[str], extra: dict[str, object]) -> dict[str, object]:
    counts = np.bincount(labels, minlength=len(class_names))
    present = counts[counts > 0]
    proportions = present / present.sum() if len(present) else np.asarray([])
    entropy = float(-(proportions * np.log(proportions)).sum()) if len(proportions) else 0.0
    row: dict[str, object] = {
        "node_id": node_id, "num_windows": int(len(labels)), "num_classes_present": int(np.sum(counts > 0)),
        "missing_classes": ";".join(name for name, count in zip(class_names, counts) if count == 0),
        "class_entropy": entropy,
        "imbalance_ratio": float(present.max() / present.min()) if len(present) else None,
    }
    row.update({f"windows_{name}": int(count) for name, count in zip(class_names, counts)})
    row.update(extra)
    return row


def _split_description(values: np.ndarray) -> str:
    return ";".join(f"{name}:{int(np.sum(values == name))}" for name in sorted(np.unique(values)))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, content: dict[str, object]) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
