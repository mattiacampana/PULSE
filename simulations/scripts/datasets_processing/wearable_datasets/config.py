"""Configuration models and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SplitConfig:
    """Subject-level split request. Missing counts select full-dataset mode."""

    pretrain_users: int | None = None
    validation_users: int | None = None
    simulation_users: int | str | None = None
    seed: int = 42

    @property
    def enabled(self) -> bool:
        return any(value is not None for value in (self.pretrain_users, self.validation_users, self.simulation_users))


@dataclass(frozen=True)
class DatasetConfig:
    """One adapter invocation, including encoder-view parameters."""

    name: str
    raw_dir: Path
    output_dir: Path
    window_seconds: float | None = None
    stride_seconds: float | None = None
    target_rate_hz: float | None = None
    split: SplitConfig = field(default_factory=SplitConfig)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level preparation configuration."""

    datasets: tuple[DatasetConfig, ...]
    force_download: bool = False
    force_preprocess: bool = False


def load_config(path: Path) -> PipelineConfig:
    """Load and validate the public YAML configuration format."""
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root_raw = Path(document.get("raw_dir", "data/raw"))
    root_output = Path(document.get("output_dir", "data/processed"))
    seed = int(document.get("seed", 42))
    entries: list[DatasetConfig] = []
    for name, value in (document.get("datasets") or {}).items():
        value = value or {}
        known = {"pretrain_users", "validation_users", "simulation_users", "window_seconds", "stride_seconds", "target_rate_hz"}
        split = SplitConfig(value.get("pretrain_users"), value.get("validation_users"), value.get("simulation_users"), seed)
        entries.append(DatasetConfig(
            name=name, raw_dir=root_raw / name, output_dir=root_output / name,
            window_seconds=value.get("window_seconds"), stride_seconds=value.get("stride_seconds"),
            target_rate_hz=value.get("target_rate_hz"), split=split,
            options={key: item for key, item in value.items() if key not in known},
        ))
    if not entries:
        raise ValueError("The configuration must contain at least one dataset.")
    return PipelineConfig(tuple(entries), bool(document.get("force_download", False)), bool(document.get("force_preprocess", False)))

