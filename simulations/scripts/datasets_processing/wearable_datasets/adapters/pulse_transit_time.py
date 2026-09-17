"""Pulse Transit Time adapter producing compact physiological feature vectors."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from wearable_datasets.config import DatasetConfig
from wearable_datasets.core import Sample
from wearable_datasets.download import download
from .base import DatasetAdapter, register
from .common import column


PHYSIONET_CSV_ROOT = "https://physionet.org/files/pulse-transit-time-ppg/1.1.0/csv"


def _validate_csv(path: Path) -> None:
    if path.stat().st_size < 1_000_000:
        raise ValueError("file is too small to be an official PTT waveform CSV")
    header = path.open("r", encoding="utf-8", errors="replace").readline().lower()
    required = ("ecg", "pleth_1", "a_x", "a_y", "a_z")
    if not all(name in header for name in required):
        raise ValueError(f"CSV header does not contain the official PTT channels: {required}")


@register
class PulseTransitTimeAdapter(DatasetAdapter):
    name = "pulse_transit_time"
    descriptor = {"task_type": "regression_or_activity", "encoder": "physio_feature_mlp_tiny", "feature_names": ["heart_rate_bpm", "rr_mean_s", "rr_std_s", "pulse_arrival_mean_s", "pulse_arrival_std_s", "ppg_mean", "ppg_std", "ecg_std", "acc_magnitude_mean", "acc_magnitude_std", "signal_quality"], "normalization": "dimensionless_or_physical_features", "embedding_dim": 64}

    def acquire(self, raw_dir: Path, force: bool) -> Path:
        csv_dir = raw_dir / "csv"
        existing = [path for path in csv_dir.glob("s*_*.csv") if path.name != "subjects_info.csv"]
        if len(existing) >= 66 and not force:
            for path in existing:
                _validate_csv(path)
            return raw_dir
        csv_dir.mkdir(parents=True, exist_ok=True)
        for subject in range(1, 23):
            for activity in ("sit", "walk", "run"):
                name = f"s{subject}_{activity}.csv"
                download(f"{PHYSIONET_CSV_ROOT}/{name}", csv_dir / name, force, self.name, _validate_csv)
        return raw_dir

    def preprocess(self, source: Path, config: DatasetConfig) -> list[Sample]:
        rate = float(config.options.get("source_rate_hz", 500))
        duration = config.window_seconds or 10.0
        size, stride = round(duration * rate), round((config.stride_seconds or duration) * rate)
        activities: dict[str, int] = {}
        result = []
        for path in sorted(source.rglob("*.csv")):
            if path.name.lower() == "subjects_info.csv":
                continue
            frame = pd.read_csv(path)
            try:
                ecg_name, ppg_name = column(frame, "ecg", "ecg1"), column(frame, "pleth_1", "ppg", "pleth", "ppg1")
            except ValueError:
                continue
            acc_names = []
            for aliases in (("a_x", "acc_x", "ax"), ("a_y", "acc_y", "ay"), ("a_z", "acc_z", "az")):
                try: acc_names.append(column(frame, *aliases))
                except ValueError: pass
            values = frame[[ecg_name, ppg_name, *acc_names]].apply(pd.to_numeric, errors="coerce").interpolate().fillna(0).to_numpy(np.float32)
            match = re.fullmatch(r"s(\d+)_(sit|walk|run)", path.stem, re.I)
            activity = match.group(2).lower() if match else path.parent.name.lower()
            activities.setdefault(activity, len(activities))
            subject = match.group(1) if match else path.stem.split("_")[0]
            for start in range(0, len(values) - size + 1, stride):
                window = values[start:start + size]
                features = self._features(window, rate)
                result.append(Sample({"physio_features": features[:, None]}, activities[activity], subject, path.stem, "physio_recorder", start / rate, {"activity": activity}))
        self.descriptor = {**self.descriptor, "class_names": [name for name, _ in sorted(activities.items(), key=lambda item: item[1])]}
        return result

    @staticmethod
    def _features(values: np.ndarray, rate: float) -> np.ndarray:
        ecg, ppg = values[:, 0], values[:, 1]
        distance = max(1, round(0.3 * rate))
        r_peaks, _ = signal.find_peaks(ecg, distance=distance, prominence=max(np.std(ecg), 1e-6))
        p_peaks, _ = signal.find_peaks(ppg, distance=distance, prominence=max(np.std(ppg) * 0.5, 1e-6))
        rr = np.diff(r_peaks) / rate
        arrivals = []
        for peak in r_peaks:
            future = p_peaks[(p_peaks > peak) & (p_peaks < peak + 0.5 * rate)]
            if len(future): arrivals.append((future[0] - peak) / rate)
        acc = np.linalg.norm(values[:, 2:], axis=1) if values.shape[1] >= 5 else np.zeros(len(values))
        quality = min(len(r_peaks), len(p_peaks)) / max(len(r_peaks), len(p_peaks), 1)
        return np.asarray([60 / np.mean(rr) if len(rr) else 0, np.mean(rr) if len(rr) else 0, np.std(rr) if len(rr) else 0, np.mean(arrivals) if arrivals else 0, np.std(arrivals) if arrivals else 0, np.mean(ppg), np.std(ppg), np.std(ecg), np.mean(acc), np.std(acc), quality], np.float32)