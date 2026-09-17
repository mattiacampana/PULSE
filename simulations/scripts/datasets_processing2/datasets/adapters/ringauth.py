"""RingAuth segmented gesture adapter for a tiny shared IMU encoder."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.config import DatasetConfig
from datasets.core import Sample, resample, robust_window_normalize
from datasets.download import download, extract
from datasets.reporting import REPORTER, human_size
from .base import DatasetAdapter, register
from .common import column


RINGAUTH_PARTS = (
    ("data.7z.001", "https://ora.ox.ac.uk/objects/uuid%3A66bc3ee9-056f-4164-8ca3-1fd28e97e7a5/files/d70795819f", 100 * 1024 * 1024),
    ("data.7z.002", "https://ora.ox.ac.uk/objects/uuid%3A66bc3ee9-056f-4164-8ca3-1fd28e97e7a5/files/dkd17ct47h", 100 * 1024 * 1024),
    ("data.7z.003", "https://ora.ox.ac.uk/objects/uuid%3A66bc3ee9-056f-4164-8ca3-1fd28e97e7a5/files/dt435gd59q", 50 * 1024 * 1024),
)


def _validate_multipart_part(path: Path, minimum_size: int) -> None:
    """Reject login/error pages and obviously truncated ORA parts."""
    size = path.stat().st_size
    if size < minimum_size:
        preview = path.read_bytes()[:80].decode("utf-8", errors="replace").replace("\n", " ")
        raise ValueError(
            f"{path.name} is only {human_size(size)}; expected a RingAuth archive part "
            f"larger than {human_size(minimum_size)} (payload starts with {preview!r})"
        )


def _csv_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".csv")


@register
class RingAuthAdapter(DatasetAdapter):
    name = "ringauth"
    descriptor = {"task_type": "multitarget", "targets": ["gesture_class", "user_id"], "encoder": "imu_encoder_tiny", "sampling_rate_hz": 100, "channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"], "normalization": "per_window_per_channel_zscore", "embedding_dim": 64}
    def acquire(self, raw_dir: Path, force: bool) -> Path:
        raw_dir.mkdir(parents=True, exist_ok=True)
        existing = _csv_files(raw_dir)
        if existing and not force:
            REPORTER.info(self.name, "LOCAL", f"Found {len(existing):,} extracted CSV files")
            return raw_dir

        parts = []
        for filename, url, minimum_size in RINGAUTH_PARTS:
            validator = lambda path, limit=minimum_size: _validate_multipart_part(path, limit)
            try:
                parts.append(download(url, raw_dir / filename, force, self.name, validator))
            except OSError as error:
                raise RuntimeError(
                    f"RingAuth download failed for {filename}. You can download all three parts "
                    f"from the Oxford ORA page and place them in {raw_dir}: {error}"
                ) from error

        REPORTER.info(self.name, "VALIDATE", f"All {len(parts)} multipart archive files are present")
        extract(parts[0], raw_dir, self.name)
        extracted = _csv_files(raw_dir)
        if not extracted:
            raise RuntimeError(
                f"RingAuth extraction completed but no CSV files were found under {raw_dir}. "
                "Check the 7z output and verify that data.7z.001/.002/.003 belong to the same download."
            )
        REPORTER.info(self.name, "EXTRACTED", f"Found {len(extracted):,} RingAuth CSV files")
        return raw_dir

    def preprocess(self, source: Path, config: DatasetConfig) -> list[Sample]:
        target_rate = config.target_rate_hz or 100
        length = round((config.window_seconds or 3.0) * target_rate)
        gesture_names: dict[str, int] = {}
        result = []
        for path in _csv_files(source):
            frame = pd.read_csv(path)
            try:
                channels = [column(frame, *names) for names in (("acc_x", "ax", "accelerometer_x"), ("acc_y", "ay", "accelerometer_y"), ("acc_z", "az", "accelerometer_z"), ("gyro_x", "gx", "gyroscope_x"), ("gyro_y", "gy", "gyroscope_y"), ("gyro_z", "gz", "gyroscope_z"))]
            except ValueError:
                continue
            subject_match = re.search(r"(?:user|subject|participant)[_-]?(\d+)", str(path), re.I)
            subject = subject_match.group(1) if subject_match else path.parent.name
            gesture = str(frame[column(frame, "gesture", "activity", "label")].iloc[0]).lower() if any(str(c).lower() in {"gesture", "activity", "label"} for c in frame.columns) else path.stem.split("_")[0].lower()
            gesture_names.setdefault(gesture, len(gesture_names))
            values = frame[channels].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(np.float32).T
            source_rate = float(config.options.get("source_rate_hz", target_rate))
            values = resample(values, source_rate, target_rate)
            mask = np.zeros(length, np.float32)
            clipped = values[:, :length]
            padded = np.zeros((6, length), np.float32)
            padded[:, :clipped.shape[1]] = clipped
            mask[:clipped.shape[1]] = 1
            result.append(Sample({"imu": robust_window_normalize(padded), "valid_mask": mask[None, :]}, gesture_names[gesture], subject, path.stem, path.parent.name, metadata={"gesture": gesture, "user_target": subject}))
        return result
