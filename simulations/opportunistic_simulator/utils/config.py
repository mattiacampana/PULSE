from __future__ import annotations

from dataclasses import dataclass, field
import torch
from pathlib import Path
import yaml


@dataclass
class MobilityConfig:
    path: str
    node_assignment: str


@dataclass
class DataConfig:
    path: str
    num_nodes: int | None
    train_fraction: float
    val_fraction: float
    test_fraction: float
    node_selection: str = "first"
    source_split: str = "simulation"
    allow_full_fallback: bool = False
    loading_mode: str = "lazy_hdf5"
    cache_batch_size: int = 1024
    cache_num_workers: int = 4
    cache_pin_memory: bool = True
    cache_prefetch_factor: int = 2


@dataclass
class ModelConfig:
    encoder: str
    input_channels: int | None
    embedding_dim: int | str
    num_classes: int | str
    classifier_hidden_dims: list[int]


@dataclass
class PretrainingConfig:
    checkpoint: str | None = None
    freeze_encoder: bool = True


@dataclass
class TrainingConfig:
    interval: int
    local_steps: int
    batch_size: int
    learning_rate: float
    evaluation_batch_size: int = 1024
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class EvaluationConfig:
    interval: int


@dataclass
class AlgorithmConfig:
    name: str


@dataclass
class LoggingConfig:
    output_dir: str
    progress_interval: int


# Algorithm-specific hyperparameters
@dataclass
class OpportunisticFLConfig:
    similarity_threshold: float = 0.0
    lambda_weight: float = 1.0
    encounter_rounds: int = 1
    aggregation: str = "momentum"


@dataclass
class PULSEConfig:
    alpha_max: float = 0.5
    utility_momentum: float = 0.2
    initial_utility: float = 0.5
    ucb_exploration: float = 0.25
    norm_epsilon: float = 1.0e-12


@dataclass
class Config:
    seeds: list[int]
    mobility: MobilityConfig
    data: DataConfig
    model: ModelConfig
    pretraining: PretrainingConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    algorithm: AlgorithmConfig
    logging: LoggingConfig

    # Algorithm-specific hyperparameters.
    opportunistic_fl: OpportunisticFLConfig
    pulse: PULSEConfig


    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        config_path = Path(path).expanduser().resolve()
        base_dir = config_path.parent

        def resolve_path(value: str | None) -> str | None:
            if not value:
                return value

            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate

            return str(candidate.resolve())

        with config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)

        # Resolve every path declared in the YAML.
        raw["mobility"]["path"] = resolve_path(raw["mobility"]["path"])
        raw["data"]["path"] = resolve_path(raw["data"]["path"])
        pretraining = raw.get("pretraining", {})
        # Backward compatibility with the original simulator configuration.
        if "checkpoint" not in pretraining and "checkpoint" in raw["model"]:
            pretraining["checkpoint"] = raw["model"].pop("checkpoint")
        if "freeze_encoder" not in pretraining and "freeze_encoder" in raw["model"]:
            pretraining["freeze_encoder"] = raw["model"].pop("freeze_encoder")
        pretraining["checkpoint"] = resolve_path(pretraining.get("checkpoint"))
        raw["model"].setdefault("encoder", "auto")
        raw["model"].setdefault("input_channels", None)
        raw["model"].setdefault("embedding_dim", "auto")
        raw["model"].setdefault("num_classes", "auto")
        raw["data"].setdefault("loading_mode", "lazy_hdf5")
        raw["data"].setdefault("cache_batch_size", 1024)
        raw["data"].setdefault("cache_num_workers", 4)
        raw["data"].setdefault("cache_pin_memory", True)
        raw["data"].setdefault("cache_prefetch_factor", 2)
        if raw["data"]["loading_mode"] not in {"lazy_hdf5", "in_memory", "cache_embeddings"}:
            raise ValueError(
                "data.loading_mode must be 'lazy_hdf5', 'in_memory', or 'cache_embeddings'."
            )
        if raw["data"]["loading_mode"] == "cache_embeddings" and not pretraining["checkpoint"]:
            raise ValueError("data.loading_mode='cache_embeddings' requires pretraining.checkpoint.")
        raw["logging"]["output_dir"] = resolve_path(raw["logging"]["output_dir"])

        return cls(
            seeds=raw["seeds"],
            mobility=MobilityConfig(**raw["mobility"]),
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            pretraining=PretrainingConfig(**pretraining),
            training=TrainingConfig(**raw["training"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
            algorithm=AlgorithmConfig(**raw["algorithm"]),
            logging=LoggingConfig(**raw["logging"]),
            opportunistic_fl=OpportunisticFLConfig(
                **raw.get("opportunistic_fl", {})
            ),
            pulse=PULSEConfig(
                **raw.get("pulse", {})
            )
        )
