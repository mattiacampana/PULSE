"""Small neural building blocks shared by all simulated nodes.

Every encoder in this module maps one model-ready sample (or a batch of
samples) to a fixed-size embedding.  Keeping that contract independent of the
sensor layout lets the simulator reuse the same personalized classifier and
federated-learning algorithms for all supported datasets.
"""

from __future__ import annotations

import torch
from torch import nn


def _require_channels(x: torch.Tensor, channels: int, name: str) -> None:
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, channels, time], got {tuple(x.shape)}.")
    if x.size(1) != channels:
        raise ValueError(f"{name} must have {channels} channels, got {x.size(1)}.")


def _require_image_channels(x: torch.Tensor, channels: int, name: str) -> None:
    """Validate channel-first batches of fixed-size or arbitrary-size images."""
    if x.ndim != 4:
        raise ValueError(
            f"{name} must have shape [batch, channels, height, width], "
            f"got {tuple(x.shape)}."
        )
    if x.size(1) != channels:
        raise ValueError(f"{name} must have {channels} channels, got {x.size(1)}.")


class _ConvBranch(nn.Module):
    """Length-agnostic temporal feature extractor used by sensor branches."""

    def __init__(self, input_channels: int, output_dim: int = 64) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor, name: str = "input") -> torch.Tensor:
        _require_channels(x, self.input_channels, name)
        return self.projection(self.network(x).flatten(1))


class CNN1DEncoder(nn.Module):
    """Compact length-agnostic IMU encoder (legacy public class name)."""

    def __init__(self, input_channels: int = 3, embedding_dim: int = 64) -> None:
        super().__init__()
        self.encoder = _ConvBranch(input_channels, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, "imu")


class IMUEncoderTiny(CNN1DEncoder):
    """Encoder for accelerometer-only or six-channel accelerometer/gyro IMU."""


class TemporalResidualBlock(nn.Module):
    """Causal-free dilated residual block for short sensor windows."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class IMUEncoderTinyTCN(nn.Module):
    """Dilated TCN used for high-rate six-channel SisFall windows."""

    def __init__(self, input_channels: int = 6, embedding_dim: int = 64) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, 5, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(TemporalResidualBlock(64, 1), TemporalResidualBlock(64, 2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_channels(x, self.input_channels, "imu")
        return self.projection(self.pool(self.blocks(self.stem(x))).flatten(1))


class DualWristIMUEncoderTiny(nn.Module):
    """Shared-weight dual-wrist encoder with symmetric feature fusion."""

    def __init__(self, input_channels: int = 6, embedding_dim: int = 64) -> None:
        super().__init__()
        self.wrist_encoder = _ConvBranch(input_channels, embedding_dim)
        self.fusion = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.ReLU(inplace=True))

    def forward(self, left_wrist: torch.Tensor, right_wrist: torch.Tensor) -> torch.Tensor:
        left = self.wrist_encoder(left_wrist, "left_wrist")
        right = self.wrist_encoder(right_wrist, "right_wrist")
        # Sum and absolute difference make the representation invariant to the
        # ordering of the two wrists without discarding inter-wrist contrast.
        return self.fusion(torch.cat((left + right, (left - right).abs()), dim=1))


class PhysioEncoderTinyMultibranch(nn.Module):
    """WESAD encoder for asynchronous ACC, BVP, EDA, and temperature branches."""

    BRANCH_CHANNELS = {"acc": 3, "bvp": 1, "eda": 1, "temperature": 1}

    def __init__(self, embedding_dim: int = 64, branch_dim: int = 32) -> None:
        super().__init__()
        self.branches = nn.ModuleDict({name: _ConvBranch(channels, branch_dim) for name, channels in self.BRANCH_CHANNELS.items()})
        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * len(self.branches), embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, acc: torch.Tensor, bvp: torch.Tensor, eda: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        values = {"acc": acc, "bvp": bvp, "eda": eda, "temperature": temperature}
        encoded = [self.branches[name](values[name], name) for name in self.BRANCH_CHANNELS]
        return self.fusion(torch.cat(encoded, dim=1))


class PhysioFeatureMLPTiny(nn.Module):
    """Encoder for the 11 engineered pulse-transit-time features."""

    def __init__(self, input_features: int = 11, embedding_dim: int = 64) -> None:
        super().__init__()
        self.input_features = input_features
        self.network = nn.Sequential(
            nn.Linear(input_features, 64), nn.LayerNorm(64), nn.ReLU(inplace=True),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, physio_features: torch.Tensor) -> torch.Tensor:
        if physio_features.ndim == 3 and physio_features.size(-1) == 1:
            physio_features = physio_features.squeeze(-1)
        if physio_features.ndim != 2 or physio_features.size(1) != self.input_features:
            raise ValueError(f"physio_features must have shape [batch, {self.input_features}] or [batch, {self.input_features}, 1].")
        return self.network(physio_features)


class AudioIMUEncoderTiny(nn.Module):
    """Fusion encoder for a log-spectrum image and synchronized 3-axis IMU."""

    def __init__(self, embedding_dim: int = 64, branch_dim: int = 64) -> None:
        super().__init__()
        self.audio = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, branch_dim),
        )
        self.imu = _ConvBranch(3, branch_dim)
        self.fusion = nn.Sequential(nn.Linear(branch_dim * 2, embedding_dim), nn.ReLU(inplace=True))

    def forward(self, audio_log_spectrum: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        if audio_log_spectrum.ndim == 3:
            audio_log_spectrum = audio_log_spectrum.unsqueeze(1)
        if audio_log_spectrum.ndim != 4 or audio_log_spectrum.size(1) != 1:
            raise ValueError("audio_log_spectrum must have shape [batch, frequency, frames] or [batch, 1, frequency, frames].")
        return self.fusion(torch.cat((self.audio(audio_log_spectrum), self.imu(imu, "imu")), dim=1))


class ImageEncoderTinyCNN(nn.Module):
    """Compact 2-D CNN for CIFAR-10 and Fashion-MNIST.

    The adaptive pooling layer makes the encoder work with both CIFAR-10 RGB
    images (``[3, 32, 32]``) and Fashion-MNIST grayscale images
    (``[1, 28, 28]``).  The final projection preserves the common
    64-dimensional embedding contract used by the personalized classifier.
    """

    def __init__(self, input_channels: int = 3, embedding_dim: int = 64) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        self.input_channels = input_channels
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(96, embedding_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _require_image_channels(image, self.input_channels, "image")
        return self.projection(self.features(image).flatten(1))


ENCODER_REGISTRY: dict[str, type[nn.Module]] = {
    "imu_encoder_tiny": IMUEncoderTiny,
    "imu_encoder_tiny_tcn": IMUEncoderTinyTCN,
    "dual_wrist_imu_encoder_tiny": DualWristIMUEncoderTiny,
    "physio_encoder_tiny_multibranch": PhysioEncoderTinyMultibranch,
    "physio_feature_mlp_tiny": PhysioFeatureMLPTiny,
    "audio_imu_encoder_tiny": AudioIMUEncoderTiny,
    "image_encoder_tiny_cnn": ImageEncoderTinyCNN,
}


def build_encoder(name: str, **kwargs: object) -> nn.Module:
    """Instantiate the encoder named by a dataset descriptor."""
    try:
        encoder_type = ENCODER_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(ENCODER_REGISTRY))
        raise ValueError(f"Unknown encoder {name!r}; available encoders: {available}.") from error
    return encoder_type(**kwargs)


class MLPClassifier(nn.Module):
    """Small personalized MLP head operating on an encoder embedding.

    ``hidden_dims=[]`` is supported for a linear-head ablation.  The final
    output layer remains explicitly accessible so class-wise algorithms can
    exchange only the parameters associated with a selected activity.
    """
    def __init__(
        self, embedding_dim: int, num_classes: int, hidden_dims: list[int] | None = None,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or []
        if any(width <= 0 for width in hidden_dims):
            raise ValueError("All classifier hidden dimensions must be positive.")
        layers: list[nn.Module] = []
        in_features = embedding_dim
        for width in hidden_dims:
            layers.append(nn.Linear(in_features, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            in_features = width
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.feature_layers(embeddings))
