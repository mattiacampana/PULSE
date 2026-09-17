"""HHAR smartphone accelerometer adapter."""

from pathlib import Path

import pandas as pd

from datasets.config import DatasetConfig
from datasets.download import download, extract_nested_zips, require_zip_members
from .base import DatasetAdapter, register
from .common import column, window_labeled_frame


@register
class HharAdapter(DatasetAdapter):
    name = "hhar"
    descriptor = {"task_type": "multiclass", "class_names": ["sit", "stairsup", "stairsdown", "walk", "stand", "bike"], "encoder": "imu_encoder_tiny", "sampling_rate_hz": 50, "channels": ["acc_x", "acc_y", "acc_z"], "normalization": "per_window_per_channel_zscore", "embedding_dim": 64}
    url = "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"

    def acquire(self, raw_dir: Path, force: bool) -> Path:
        validator = require_zip_members(
            "*Activity*recognition*.zip", "*Phones_accelerometer.csv",
            minimum_size=100 * 1024 * 1024,
        )
        archive = download(self.url, raw_dir / "hhar.zip", force, self.name, validator)
        if next(raw_dir.rglob("Phones_accelerometer.csv"), None) is None or force:
            extract_nested_zips(
                archive, raw_dir, self.name,
                ready=lambda: next(raw_dir.rglob("Phones_accelerometer.csv"), None) is not None,
            )
        if next(raw_dir.rglob("Phones_accelerometer.csv"), None) is None:
            raise FileNotFoundError(
                "The official HHAR activity-recognition CSV was not found after extracting the UCI archive."
            )
        return raw_dir

    def preprocess(self, source: Path, config: DatasetConfig):
        path = next(source.rglob("Phones_accelerometer.csv"), None)
        if path is None:
            raise FileNotFoundError("Phones_accelerometer.csv is missing from HHAR.")
        frame = pd.read_csv(path)
        aliases = {key: column(frame, *names) for key, names in {"x": ("x",), "y": ("y",), "z": ("z",), "user": ("user",), "label": ("gt", "label", "activity"), "device": ("device", "model")}.items()}
        size = int((config.window_seconds or 2.56) * (config.target_rate_hz or 50))
        stride = int((config.stride_seconds or 2.56) * (config.target_rate_hz or 50))
        result = []
        for (user, device), group in frame.groupby([aliases["user"], aliases["device"]], sort=False):
            result += window_labeled_frame(group, [aliases["x"], aliases["y"], aliases["z"]], aliases["label"], str(user), str(device), str(device), size, stride, {name: i for i, name in enumerate(self.descriptor["class_names"])})
        return result
