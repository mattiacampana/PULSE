"""UCI HAR adapter producing the common 50 Hz accelerometer encoder view."""

from pathlib import Path

import numpy as np

from wearable_datasets.config import DatasetConfig
from wearable_datasets.core import Sample, robust_window_normalize
from wearable_datasets.download import download, extract_nested_zips, require_zip_members
from .base import DatasetAdapter, register


@register
class UciHarAdapter(DatasetAdapter):
    name = "uci_har"
    descriptor = {"task_type": "multiclass", "class_names": ["walk", "walk_upstairs", "walk_downstairs", "sit", "stand", "lay"], "encoder": "imu_encoder_tiny", "sampling_rate_hz": 50, "channels": ["acc_x", "acc_y", "acc_z"], "normalization": "per_window_per_channel_zscore", "embedding_dim": 64}
    url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

    def acquire(self, raw_dir: Path, force: bool) -> Path:
        # The UCI repository bundle is currently ~58 MiB and may contain the
        # original "UCI HAR Dataset.zip" as an embedded archive.
        validator = require_zip_members(
            "*UCI*HAR*Dataset*.zip", "*subject_train.txt", "*y_train.txt",
            minimum_size=10 * 1024 * 1024,
        )
        archive = download(self.url, raw_dir / "uci_har.zip", force, self.name, validator)
        root = self._find_dataset_root(raw_dir)
        if root is None or force:
            extract_nested_zips(
                archive, raw_dir, self.name,
                ready=lambda: self._find_dataset_root(raw_dir) is not None,
            )
            root = self._find_dataset_root(raw_dir)

        if root is None:
            discovered = sorted(path.name for path in raw_dir.iterdir()) if raw_dir.exists() else []
            raise FileNotFoundError(
                "The official UCI HAR data files were not found after extracting "
                f"the downloaded archive. Top-level entries: {discovered or ['<none>']}"
            )
        return root

    @staticmethod
    def _find_dataset_root(raw_dir: Path) -> Path | None:
        """Find a valid UCI HAR root independently of wrapper directory names."""
        required = (
            Path("train/y_train.txt"),
            Path("train/subject_train.txt"),
            Path("test/y_test.txt"),
            Path("test/subject_test.txt"),
        )
        if not raw_dir.exists():
            return None
        for marker in raw_dir.rglob("subject_train.txt"):
            candidate = marker.parent.parent
            signals = tuple(
                Path(f"{split}/Inertial Signals/total_acc_{axis}_{split}.txt")
                for split in ("train", "test") for axis in "xyz"
            )
            if all((candidate / relative).is_file() for relative in (*required, *signals)):
                return candidate
        return None

    def preprocess(self, source: Path, config: DatasetConfig) -> list[Sample]:
        result = []
        for split in ("train", "test"):
            folder = source / split / "Inertial Signals"
            values = np.stack([np.loadtxt(folder / f"total_acc_{axis}_{split}.txt", dtype=np.float32) for axis in "xyz"], axis=1)
            labels = np.loadtxt(source / split / f"y_{split}.txt", dtype=int) - 1
            subjects = np.loadtxt(source / split / f"subject_{split}.txt", dtype=int)
            result.extend(Sample({"imu": robust_window_normalize(x)}, int(y), str(subject), split, "smartphone", float(i), {"official_split": split}) for i, (x, y, subject) in enumerate(zip(values, labels, subjects)))
        return result
