"""WESAD multimodal wrist adapter with late-fusion-ready branches."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from datasets.config import DatasetConfig
from datasets.core import Sample, resample, robust_window_normalize, sliding_windows
from datasets.download import download, extract, require_zip_members
from datasets.reporting import REPORTER
from .base import DatasetAdapter, register


@register
class WesadAdapter(DatasetAdapter):
    name = "wesad"
    descriptor = {"task_type": "multiclass", "class_names": ["baseline", "stress", "amusement"], "encoder": "physio_encoder_tiny_multibranch", "branches": {"acc": ["acc_x", "acc_y", "acc_z"], "bvp": ["bvp"], "eda": ["eda"], "temperature": ["temperature"]}, "sampling_rates_hz": {"acc": 32, "bvp": 64, "eda": 4, "temperature": 4}, "normalization": "per_window_per_channel_zscore", "embedding_dim": 64}
    # Public Kaggle mirror requested by the project. Kaggle's dataset-download
    # API follows redirects and does not require credentials for this dataset.
    url = "https://www.kaggle.com/api/v1/datasets/download/orvile/wesad-wearable-stress-affect-detection-dataset"

    def acquire(self, raw_dir: Path, force: bool) -> Path:
        validator = require_zip_members("*/S*.pkl", "S*.pkl", minimum_size=10 * 1024 * 1024)
        try:
            archive = download(self.url, raw_dir / "wesad.zip", force, self.name, validator)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                "WESAD acquisition failed: the complete subject archive was not received. "
                f"Remove/refresh {raw_dir / 'wesad.zip'} or download the official archive "
                f"from its Kaggle page and save it there. Details: {error}"
            ) from error
        if next(raw_dir.rglob("S*.pkl"), None) is None or force:
            extract(archive, raw_dir, self.name)
        subjects = sorted(raw_dir.rglob("S*.pkl"))
        if not subjects:
            raise RuntimeError(
                f"WESAD extraction completed but no subject pickle files (S*.pkl) were found under {raw_dir}."
            )
        REPORTER.info(self.name, "VALIDATED", f"Found {len(subjects)} WESAD subject files")
        return raw_dir

    def preprocess(self, source: Path, config: DatasetConfig) -> list[Sample]:
        duration = config.window_seconds or 60.0
        stride_s = config.stride_seconds or duration
        label_map = {1: 0, 2: 1, 3: 2}
        rates = self.descriptor["sampling_rates_hz"]
        result: list[Sample] = []
        for path in sorted(source.rglob("S*.pkl")):
            with path.open("rb") as stream:
                record = pickle.load(stream, encoding="latin1")
            subject = str(record.get("subject", path.stem))
            wrist = record["signal"]["wrist"]
            labels = np.asarray(record["label"]).reshape(-1)  # Chest-rate labels at 700 Hz.
            reference_rate = 700
            count = int(duration * reference_rate)
            stride = int(stride_s * reference_rate)
            for start, label_window in sliding_windows(labels[None, :], count, stride):
                valid = label_window[0]
                selected = valid[np.isin(valid, list(label_map))]
                if len(selected) < 0.95 * len(valid):
                    continue
                original_label = int(np.bincount(selected.astype(int)).argmax())
                inputs = {}
                for key, source_key in (("acc", "ACC"), ("bvp", "BVP"), ("eda", "EDA"), ("temperature", "TEMP")):
                    rate = rates[key]
                    first = round(start / reference_rate * rate)
                    last = first + round(duration * rate)
                    values = np.asarray(wrist[source_key], np.float32).reshape(-1, 3 if key == "acc" else 1).T[:, first:last]
                    if values.shape[-1] != round(duration * rate):
                        break
                    inputs[key] = robust_window_normalize(values)
                else:
                    result.append(Sample(inputs, label_map[original_label], subject, path.stem, "empatica_e4", start / reference_rate))
        return result
