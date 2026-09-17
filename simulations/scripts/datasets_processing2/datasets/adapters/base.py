"""Adapter interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from datasets.config import DatasetConfig
from datasets.core import Sample, write_dataset
from datasets.reporting import REPORTER


class DatasetAdapter(ABC):
    """Base class implemented by every raw dataset integration."""

    name: str
    descriptor: dict[str, object]

    @abstractmethod
    def acquire(self, raw_dir: Path, force: bool) -> Path:
        """Return a directory containing extracted official data."""

    @abstractmethod
    def preprocess(self, source: Path, config: DatasetConfig) -> list[Sample]:
        """Convert official files into fixed-shape encoder inputs."""

    def run(self, config: DatasetConfig, force_download: bool, force_preprocess: bool) -> None:
        """Execute acquisition, preprocessing, validation, and output writing."""
        REPORTER.section(self.name, "Preparing model-ready dataset")
        REPORTER.info(self.name, "CONFIG", self._configuration_summary(config))
        completed = config.output_dir / "dataset.json"
        if completed.exists() and not force_preprocess:
            REPORTER.info(self.name, "REUSE", f"Completed output already exists at {config.output_dir}; use --force-preprocess to rebuild")
            return
        REPORTER.info(self.name, "ACQUIRE", f"Resolving raw source in {config.raw_dir}")
        source = self.acquire(config.raw_dir, force_download)
        source_files = sum(1 for item in source.rglob("*") if item.is_file()) if source.is_dir() else 1
        REPORTER.info(self.name, "SOURCE", f"Ready at {source} ({source_files:,} files)")
        REPORTER.info(self.name, "PREPROCESS", self._preprocessing_summary(config))
        samples = self.preprocess(source, config)
        subjects = len({sample.subject_id for sample in samples})
        REPORTER.info(self.name, "PREPARED", f"Generated {len(samples):,} fixed-shape samples from {subjects:,} users")
        if not samples:
            raise RuntimeError(
                f"{self.name} preprocessing generated no samples. Verify that the raw dataset is complete "
                "and that the configured window/stride settings match the source recordings."
            )
        write_dataset(
            samples,
            config.output_dir,
            {"dataset": self.name, **self.descriptor},
            config.split,
            explicit_splits=getattr(self, "_explicit_splits", None),
            split_metadata=getattr(self, "_split_metadata", None),
        )
        REPORTER.info(self.name, "DONE", f"Model-ready dataset written to {config.output_dir}")

    def _configuration_summary(self, config: DatasetConfig) -> str:
        split = config.split
        mode = "subject split" if split.enabled else "full dataset"
        counts = f"pretrain={split.pretrain_users or 0}, validation={split.validation_users or 0}, simulation={split.simulation_users or 0}" if split.enabled else "no user-role assignment"
        return f"mode={mode}; {counts}; seed={split.seed}"

    def _preprocessing_summary(self, config: DatasetConfig) -> str:
        """Describe model-facing operations without exposing per-sample noise."""
        branches = self.descriptor.get("branches") or self.descriptor.get("channels") or ["model input"]
        branch_names = ", ".join(branches) if isinstance(branches, dict) else "imu"
        window = f"{config.window_seconds:g}s" if config.window_seconds is not None else "dataset default"
        rate = f"{config.target_rate_hz:g} Hz" if config.target_rate_hz is not None else "native/default rate"
        return f"Parsing signals, encoding labels, windowing ({window}), resampling ({rate}), normalizing, and building branches: {branch_names}"


_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register(adapter: type[DatasetAdapter]) -> type[DatasetAdapter]:
    """Register an adapter class by its stable CLI name."""
    _REGISTRY[adapter.name] = adapter
    return adapter


def create_adapter(name: str) -> DatasetAdapter:
    """Instantiate a registered adapter or report all valid names."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]()


def available_adapters() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
