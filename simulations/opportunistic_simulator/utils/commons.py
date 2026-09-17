import numpy as np
import torch
import random
import hashlib
import inspect
import json
from pathlib import Path
from typing import Mapping

from torch import nn

from opportunistic_simulator.data import DatasetDescriptor
from opportunistic_simulator.models import MLPClassifier, build_encoder
from opportunistic_simulator.utils.config import Config


def seed_everything(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class EncoderInputAdapter(nn.Module):
    """Call an encoder from semantic HDF5 branch names."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        signature = inspect.signature(encoder.forward)
        self.parameter_names = tuple(
            name for name, parameter in signature.parameters.items()
            if name != "self" and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        )

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        values = {name: inputs[name] for name in self.parameter_names if name in inputs}
        missing = [name for name in self.parameter_names if name not in values]
        if len(missing) == 1:
            auxiliary = {"valid_mask", "mask", "attention_mask", "lengths"}
            candidates = [value for name, value in inputs.items() if name not in values and name not in auxiliary]
            if len(candidates) == 1:
                values[missing[0]] = candidates[0]
                missing.clear()
        if missing:
            raise ValueError(
                f"Encoder requires branches {list(self.parameter_names)}; dataset provides {sorted(inputs)}."
            )
        return self.encoder(**values)


def _encoder_kwargs(name: str, descriptor: DatasetDescriptor, embedding_dim: int) -> dict[str, int]:
    kwargs = {"embedding_dim": embedding_dim}
    if name in {"imu_encoder_tiny", "imu_encoder_tiny_tcn"}:
        branch = "imu" if "imu" in descriptor.input_shapes else descriptor.branches[0]
        kwargs["input_channels"] = descriptor.input_shapes[branch][0]
    elif name == "dual_wrist_imu_encoder_tiny":
        kwargs["input_channels"] = descriptor.input_shapes["left_wrist"][0]
    return kwargs


def init_encoder(config: Config, descriptor: DatasetDescriptor) -> nn.Module:
    """
    Initialize the CNN1DEncoder with a given seed for reproducibility.
    Args:
        config (Config): The configuration object.
    Returns:
        CNN1DEncoder: An instance of the CNN1DEncoder model.
    """
    checkpoint = Path(config.pretraining.checkpoint) if config.pretraining.checkpoint else None
    metadata_path = checkpoint.with_name(checkpoint.name + ".json") if checkpoint else None
    checkpoint_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path is not None and metadata_path.is_file() else {}
    )
    dataset_encoder = str(descriptor.metadata.get("encoder", ""))
    encoder_name = dataset_encoder if config.model.encoder == "auto" else config.model.encoder
    if not encoder_name:
        raise ValueError("model.encoder is auto, but the simulation dataset does not declare an encoder.")
    checkpoint_encoder = checkpoint_metadata.get("encoder_name")
    if checkpoint_encoder and checkpoint_encoder != encoder_name:
        raise ValueError(
            f"Checkpoint encoder {checkpoint_encoder!r} is incompatible with simulation encoder {encoder_name!r}."
        )
    embedding_dim = (
        int(descriptor.metadata.get("embedding_dim", 64))
        if config.model.embedding_dim == "auto" else int(config.model.embedding_dim)
    )
    if config.model.input_channels is not None:
        primary = "left_wrist" if encoder_name == "dual_wrist_imu_encoder_tiny" else (
            "imu" if "imu" in descriptor.input_shapes else descriptor.branches[0]
        )
        actual_channels = descriptor.input_shapes[primary][0]
        if int(config.model.input_channels) != actual_channels:
            raise ValueError(
                f"model.input_channels={config.model.input_channels} but simulation branch "
                f"{primary!r} has {actual_channels} channels."
            )
    checkpoint_embedding = checkpoint_metadata.get("embedding_dim")
    if checkpoint_embedding is not None and int(checkpoint_embedding) != embedding_dim:
        raise ValueError(f"Checkpoint embedding dimension {checkpoint_embedding} != requested {embedding_dim}.")

    encoder = build_encoder(encoder_name, **_encoder_kwargs(encoder_name, descriptor, embedding_dim))
    if checkpoint is None:
        print("[setup] Initializing encoder from scratch (no pretrained checkpoint).", flush=True)
    else:
        print(f"[setup] Loading pretrained encoder: {checkpoint}", flush=True)
        try:
            encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        except RuntimeError as error:
            raise ValueError(
                "The checkpoint architecture/input shape is incompatible with the simulation dataset. "
                f"Pretraining and simulation datasets may differ, but their encoder contract must match: {error}"
            ) from error

    # Freeze the encoder if requested. 
    # This is a global setting, so all nodes will share the same frozen encoder. 
    # The local heads are always trainable.
    encoder.requires_grad_(not config.pretraining.freeze_encoder)
    config.model.encoder = encoder_name
    config.model.embedding_dim = embedding_dim
    return EncoderInputAdapter(encoder)


def make_seeded_classifier(
    embedding_dim: int,
    num_classes: int,
    hidden_dims: list[int],
    simulation_seed: int,
    node_id: str,
    batch_norm: bool = False,
) -> MLPClassifier:
    """
    Create a node-specific head reproducibly, without advancing the global RNG.
    The same simulation seed and node identifier always yield exactly the same
    initial head, independently of the selected collaboration algorithm.
    """
    seed_material = f"{simulation_seed}:{node_id}".encode("utf-8")
    head_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little") % (2**63 - 1)
    with torch.random.fork_rng():
        torch.manual_seed(head_seed)
        return MLPClassifier(embedding_dim, num_classes, hidden_dims, batch_norm=batch_norm)


def classifier_fingerprint(classifier: MLPClassifier) -> str:
    """
    Return a compact, stable identifier for a classifier's initial weights.
    This is used to verify that the same seed and node ID always yield the same initial head,
    independently of the collaboration algorithm.
    """
    digest = hashlib.sha256()
    for name, parameter in classifier.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:12]
