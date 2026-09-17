"""PADS dual-wrist IMU adapter with shared-encoder fusion inputs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from datasets.config import DatasetConfig
from datasets.core import Sample, robust_window_normalize, sliding_windows
from datasets.download import download, extract, require_zip_members
from datasets.reporting import REPORTER
from .base import DatasetAdapter, register


@register
class PadsAdapter(DatasetAdapter):
    name = "pads"
    descriptor = {"task_type": "multiclass", "encoder": "dual_wrist_imu_encoder_tiny", "sampling_rate_hz": 100, "branches": {"left_wrist": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"], "right_wrist": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]}, "normalization": "per_window_per_channel_zscore", "embedding_dim": 64}

    # Direct public archive linked by the PhysioNet project page.  This is the
    # complete 1.0.0 release, not an HTML directory listing.
    url = (
        "https://physionet.org/static/published-projects/"
        "parkinsons-disease-smartwatch/"
        "parkinsons-disease-smartwatch-1.0.0.zip"
    )

    def acquire(self, raw_dir: Path, force: bool) -> Path:
        """Download and validate the public PhysioNet release."""
        recordings = self._recordings(raw_dir) if raw_dir.exists() else []
        if not recordings or force:
            validator = require_zip_members(
                "*movement*/*.txt",
                "*movement*/*/*.txt",
                "*LeftWrist.txt",
                "*RightWrist.txt",
                minimum_size=1024 * 1024,
            )
            try:
                archive = download(
                    self.url, raw_dir / "pads-1.0.0.zip", force,
                    self.name, validator,
                )
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    "PADS acquisition failed: the complete PhysioNet archive was not received. "
                    f"You can download it from the PADS 1.0.0 project page and save it as "
                    f"{raw_dir / 'pads-1.0.0.zip'}. Details: {error}"
                ) from error
            extract(archive, raw_dir, self.name)
            recordings = self._recordings(raw_dir)
        if not recordings:
            raise RuntimeError(
                f"PADS extraction completed, but no paired-wrist movement TXT files were found under {raw_dir}."
            )
        REPORTER.info(self.name, "VALIDATED", f"Found {len(recordings):,} PADS movement recordings")
        return raw_dir

    @staticmethod
    def _recordings(raw_dir: Path) -> list[Path]:
        """Return sensor recordings, ignoring PhysioNet documentation TXT files."""
        return sorted(
            path for path in raw_dir.rglob("*.txt")
            if "movement" in {part.lower() for part in path.parts}
            and ("leftwrist" in path.name.lower() or "rightwrist" in path.name.lower())
        )

    def preprocess(self, source: Path, config: DatasetConfig) -> list[Sample]:
        rate = 100
        size = round((config.window_seconds or 2.56) * rate)
        stride = round((config.stride_seconds or 2.56) * rate)
        # Official filenames/parents contain subject, task, and wrist. Pair by
        # the normalized path after removing the side marker.
        pairs: dict[str, dict[str, Path]] = {}
        for path in source.rglob("*.txt"):
            text = str(path).lower()
            side = "left" if "left" in text else "right" if "right" in text else None
            if side:
                key = re.sub(r"left|right", "wrist", text)
                pairs.setdefault(key, {})[side] = path
        task_ids: dict[str, int] = {}
        result = []
        for pair in pairs.values():
            if set(pair) != {"left", "right"}:
                continue
            left, right = (np.loadtxt(pair[side], delimiter=",", dtype=np.float32) for side in ("left", "right"))
            if left.ndim != 2 or right.ndim != 2 or left.shape[1] < 6 or right.shape[1] < 6:
                continue
            # PhysioNet's movement/timeseries files contain exactly six sensor
            # columns and no time column. Tolerate a legacy seven-column export
            # only when explicitly configured by its extra leading column.
            left = left[:, -6:].T
            right = right[:, -6:].T
            length = min(left.shape[-1], right.shape[-1])
            subject_match = re.search(r"(?:subject|patient|observation)[_-]?(\d+)", str(pair["left"]), re.I)
            if subject_match is None:
                subject_match = re.match(r"(\d{3})_", pair["left"].name)
            subject = subject_match.group(1) if subject_match else pair["left"].parent.name
            task = pair["left"].stem.lower().replace("left", "").replace("right", "").strip("_- ")
            task_ids.setdefault(task, len(task_ids))
            for start in range(0, length - size + 1, stride):
                inputs = {"left_wrist": robust_window_normalize(left[:, start:start + size]), "right_wrist": robust_window_normalize(right[:, start:start + size])}
                result.append(Sample(inputs, task_ids[task], subject, task, "dual_apple_watch", start / rate, {"task": task}))
        self.descriptor = {**self.descriptor, "class_names": [name for name, _ in sorted(task_ids.items(), key=lambda item: item[1])]}
        return result
