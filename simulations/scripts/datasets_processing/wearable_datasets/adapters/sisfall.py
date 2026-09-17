"""SisFall waist IMU adapter for binary fall detection."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from wearable_datasets.config import DatasetConfig
from wearable_datasets.core import Sample, robust_window_normalize, sliding_windows
from wearable_datasets.download import download, extract, require_zip_members
from wearable_datasets.reporting import REPORTER
from .base import DatasetAdapter, register


@register
class SisFallAdapter(DatasetAdapter):
    name = "sisfall"
    descriptor = {"task_type": "binary", "class_names": ["adl", "fall"], "encoder": "imu_encoder_tiny_tcn", "sampling_rate_hz": 200, "channels": ["acc1_x", "acc1_y", "acc1_z", "gyro_x", "gyro_y", "gyro_z"], "normalization": "per_window_per_channel_zscore", "embedding_dim": 64}

    url = "https://www.kaggle.com/api/v1/datasets/download/nvnikhil0001/sis-fall-original-dataset"

    def acquire(self, raw_dir: Path, force: bool) -> Path:
        """Download the public Kaggle mirror and validate the real recordings."""
        validator = require_zip_members("*D??_S???_R??.txt", minimum_size=10 * 1024 * 1024)
        recordings = list(raw_dir.rglob("*.txt")) if raw_dir.exists() else []
        if not recordings or force:
            archive = download(self.url, raw_dir / "sisfall.zip", force, self.name, validator)
            extract(archive, raw_dir, self.name)
            recordings = list(raw_dir.rglob("*.txt"))
        valid = [path for path in recordings if self._parse_name(path) is not None]
        if not valid:
            raise RuntimeError(
                f"SisFall data were extracted under {raw_dir}, but no recordings matching "
                "Dxx_SA/SEyy_Rzz.txt were found. The archive layout is unsupported or incomplete."
            )
        REPORTER.info(self.name, "VALIDATED", f"Found {len(valid):,} SisFall recordings")
        return raw_dir

    def preprocess(self, source: Path, config: DatasetConfig) -> list[Sample]:
        rate = 200
        size = round((config.window_seconds or 2.56) * rate)
        stride = round((config.stride_seconds or 1.28) * rate)
        result = []
        parsed_files = skipped_files = 0
        for path in sorted(source.rglob("*.txt")):
            parsed = self._parse_name(path)
            if parsed is None:
                skipped_files += 1
                continue
            try:
                # Original SisFall rows are comma-separated and end in ';'.
                # Reading text first avoids turning the ninth sensor into NaN.
                rows = [line.strip().rstrip(";") for line in path.read_text(errors="replace").splitlines() if line.strip()]
                values = np.asarray([[float(item) for item in row.split(",")[:9]] for row in rows], dtype=np.float32)
            except (OSError, ValueError):
                skipped_files += 1
                continue
            if values.ndim != 2 or values.shape[1] < 9 or values.shape[0] < size or not np.isfinite(values[:, :9]).all():
                skipped_files += 1
                continue
            parsed_files += 1
            # SisFall units: ADXL345 /256 g; ITG3200 /14.375 deg/s. Use the
            # first accelerometer and gyroscope, converted to g and rad/s.
            imu = np.vstack((values[:, :3].T / 256.0, np.deg2rad(values[:, 6:9].T / 14.375)))
            activity, subject, trial = parsed
            label = int(activity.startswith("F"))
            for start, window in sliding_windows(imu, size, stride):
                result.append(Sample({"imu": robust_window_normalize(window)}, label, subject, trial, "waist_imu", start / rate, {"activity": activity}))
        REPORTER.info(self.name, "PARSED", f"Read {parsed_files:,} recordings; skipped {skipped_files:,} non-recording or invalid text files")
        return result

    @staticmethod
    def _parse_name(path: Path) -> tuple[str, str, str] | None:
        match = re.search(
            r"(?P<activity>[ADF]\d{2})_(?P<subject>S(?:A|E)\d{2})_R(?P<trial>\d{2})",
            path.stem,
            re.IGNORECASE,
        )
        if match is None:
            return None
        return tuple(match.group(key).upper() for key in ("activity", "subject", "trial"))
