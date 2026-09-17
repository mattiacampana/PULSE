"""Build the portable C core and compare it with the canonical PyTorch equations."""

from __future__ import annotations

import ctypes
import _ctypes
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CHANNELS = 3
WINDOW = 128
BATCH = 16
EMBEDDING = 64
HIDDEN = 64
CLASSES = 6
HEAD_PARAMETERS = 4550
POOLED_VALUES = 32 * 64


class ConstTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)), ("count", ctypes.c_size_t)]


class EncoderArtifact(ctypes.Structure):
    _fields_ = [
        ("conv1_weight", ConstTensor),
        ("conv1_bias", ConstTensor),
        ("conv2_weight", ConstTensor),
        ("conv2_bias", ConstTensor),
        ("projection_weight", ConstTensor),
        ("projection_bias", ConstTensor),
    ]


class Head(ctypes.Structure):
    _fields_ = [("parameters", ctypes.c_float * HEAD_PARAMETERS)]


class Gradient(ctypes.Structure):
    _fields_ = [("parameters", ctypes.c_float * HEAD_PARAMETERS)]


class BatchArtifact(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("sample_float_count", ctypes.c_size_t),
        ("sample_stride_floats", ctypes.c_size_t),
        ("labels", ctypes.POINTER(ctypes.c_uint8)),
        ("label_count", ctypes.c_size_t),
        ("batch_size", ctypes.c_size_t),
    ]


class SampleBatchArtifact(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("sample_float_count", ctypes.c_size_t),
        ("sample_stride_floats", ctypes.c_size_t),
        ("batch_size", ctypes.c_size_t),
    ]


class EmbeddingBatch(ctypes.Structure):
    _fields_ = [("values", ctypes.c_float * (BATCH * EMBEDDING))]


class Workspace(ctypes.Structure):
    _fields_ = [
        ("pooled", ctypes.c_float * POOLED_VALUES),
        ("encoder_features", ctypes.c_float * 64),
        ("embedding", ctypes.c_float * 64),
        ("hidden_pre", ctypes.c_float * 64),
        ("hidden", ctypes.c_float * 64),
        ("hidden_gradient", ctypes.c_float * 64),
        ("logits", ctypes.c_float * 6),
        ("probabilities", ctypes.c_float * 6),
    ]


class MixStats(ctypes.Structure):
    _fields_ = [
        ("agreement", ctypes.c_float),
        ("alpha", ctypes.c_float),
        ("remote_scale", ctypes.c_float),
        ("local_norm_sq", ctypes.c_float),
        ("remote_norm_sq", ctypes.c_float),
    ]


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(64, 64)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(values).squeeze(-1))


class HeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(64, 6)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.w2(self.relu(self.w1(values)))


def contiguous(values: torch.Tensor, dtype: np.dtype = np.float32) -> np.ndarray:
    return np.ascontiguousarray(values.detach().cpu().numpy(), dtype=dtype)


def tensor_view(values: np.ndarray) -> ConstTensor:
    return ConstTensor(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.size,
    )


def build_library(root: Path, output: Path) -> None:
    compiler = os.environ.get("CC", "gcc")
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-shared",
        f"-I{root / 'src' / 'pulse'}",
        str(root / "src" / "pulse" / "pulse_math.c"),
        "-lm",
        "-o",
        str(output),
    ]
    if os.name != "nt":
        command.insert(4, "-fPIC")
    subprocess.run(command, check=True)


def configure_library(library: ctypes.CDLL) -> None:
    library.pulse_encoder_forward.argtypes = [
        ctypes.POINTER(EncoderArtifact),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(Workspace),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.pulse_encoder_forward.restype = ctypes.c_int
    library.pulse_encoder_batch_forward.argtypes = [
        ctypes.POINTER(EncoderArtifact),
        ctypes.POINTER(SampleBatchArtifact),
        ctypes.POINTER(Workspace),
        ctypes.POINTER(EmbeddingBatch),
    ]
    library.pulse_encoder_batch_forward.restype = ctypes.c_int
    library.pulse_head_batch_gradient.argtypes = [
        ctypes.POINTER(EncoderArtifact),
        ctypes.POINTER(Head),
        ctypes.POINTER(BatchArtifact),
        ctypes.POINTER(Workspace),
        ctypes.POINTER(Gradient),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.pulse_head_batch_gradient.restype = ctypes.c_int
    library.pulse_head_batch_gradient_from_embeddings.argtypes = [
        ctypes.POINTER(Head),
        ctypes.POINTER(EmbeddingBatch),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(Workspace),
        ctypes.POINTER(Gradient),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.pulse_head_batch_gradient_from_embeddings.restype = ctypes.c_int
    library.pulse_head_apply_peer_update.argtypes = [
        ctypes.POINTER(Head),
        ctypes.POINTER(Gradient),
        ctypes.POINTER(Gradient),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(MixStats),
    ]
    library.pulse_head_apply_peer_update.restype = ctypes.c_int


def make_batch(values: np.ndarray, labels: np.ndarray) -> BatchArtifact:
    return BatchArtifact(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.size,
        CHANNELS * WINDOW,
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        labels.size,
        BATCH,
    )


def require_ok(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} returned pulse_status_t {status}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    library_name = "pulse_math.dll" if os.name == "nt" else "libpulse_math.so"

    torch.manual_seed(20260309)
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False

    encoder = Encoder().eval()
    head_model = HeadModel().eval()
    first_samples = torch.randn(BATCH, CHANNELS, WINDOW, dtype=torch.float32)
    first_labels = torch.randint(0, CLASSES, (BATCH,), dtype=torch.int64)
    second_samples = torch.randn(BATCH, CHANNELS, WINDOW, dtype=torch.float32)
    second_labels = torch.randint(0, CLASSES, (BATCH,), dtype=torch.int64)

    encoder_arrays = [contiguous(parameter) for parameter in encoder.parameters()]
    encoder_artifact = EncoderArtifact(*(tensor_view(values) for values in encoder_arrays))
    head_values = contiguous(torch.cat([parameter.detach().reshape(-1) for parameter in head_model.parameters()]))
    if head_values.size != HEAD_PARAMETERS:
        raise AssertionError(f"unexpected head size {head_values.size}")

    first_sample_values = contiguous(first_samples)
    first_label_values = contiguous(first_labels, np.uint8)
    second_sample_values = contiguous(second_samples)
    second_label_values = contiguous(second_labels, np.uint8)
    first_batch = make_batch(first_sample_values, first_label_values)
    second_batch = make_batch(second_sample_values, second_label_values)
    first_sample_batch = SampleBatchArtifact(
        first_sample_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        first_sample_values.size,
        CHANNELS * WINDOW,
        BATCH,
    )

    with tempfile.TemporaryDirectory(prefix="pulse_math_reference_") as temporary:
        library_path = Path(temporary) / library_name
        build_library(root, library_path)
        library = ctypes.CDLL(str(library_path))
        configure_library(library)

        c_head = Head((ctypes.c_float * HEAD_PARAMETERS)(*head_values))
        c_workspace = Workspace()
        c_embedding = (ctypes.c_float * EMBEDDING)()
        require_ok(
            library.pulse_encoder_forward(
                ctypes.byref(encoder_artifact),
                first_sample_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                CHANNELS * WINDOW,
                ctypes.byref(c_workspace),
                c_embedding,
            ),
            "pulse_encoder_forward",
        )

        with torch.no_grad():
            reference_embedding = contiguous(encoder(first_samples[:1])[0])
        c_embedding_values = np.ctypeslib.as_array(c_embedding).copy()
        embedding_error = float(np.max(np.abs(c_embedding_values - reference_embedding)))

        def c_gradient(batch: BatchArtifact) -> tuple[Gradient, float]:
            result = Gradient()
            loss = ctypes.c_float()
            require_ok(
                library.pulse_head_batch_gradient(
                    ctypes.byref(encoder_artifact),
                    ctypes.byref(c_head),
                    ctypes.byref(batch),
                    ctypes.byref(c_workspace),
                    ctypes.byref(result),
                    ctypes.byref(loss),
                ),
                "pulse_head_batch_gradient",
            )
            return result, float(loss.value)

        def torch_gradient(
            sample_tensor: torch.Tensor,
            label_tensor: torch.Tensor,
        ) -> tuple[list[torch.Tensor], float]:
            embeddings = encoder(sample_tensor).detach()
            loss = F.cross_entropy(head_model(embeddings), label_tensor)
            gradients = list(torch.autograd.grad(loss, tuple(head_model.parameters())))
            return gradients, float(loss.item())

        first_c_gradient, first_c_loss = c_gradient(first_batch)
        second_c_gradient, _ = c_gradient(second_batch)
        c_embedding_batch = EmbeddingBatch()
        split_c_gradient = Gradient()
        split_c_loss = ctypes.c_float()
        require_ok(
            library.pulse_encoder_batch_forward(
                ctypes.byref(encoder_artifact),
                ctypes.byref(first_sample_batch),
                ctypes.byref(c_workspace),
                ctypes.byref(c_embedding_batch),
            ),
            "pulse_encoder_batch_forward",
        )
        require_ok(
            library.pulse_head_batch_gradient_from_embeddings(
                ctypes.byref(c_head),
                ctypes.byref(c_embedding_batch),
                first_label_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                first_label_values.size,
                ctypes.byref(c_workspace),
                ctypes.byref(split_c_gradient),
                ctypes.byref(split_c_loss),
            ),
            "pulse_head_batch_gradient_from_embeddings",
        )
        split_gradient_error = float(
            np.max(
                np.abs(
                    np.ctypeslib.as_array(split_c_gradient.parameters)
                    - np.ctypeslib.as_array(first_c_gradient.parameters)
                )
            )
        )
        split_loss_error = abs(float(split_c_loss.value) - first_c_loss)
        first_reference_gradients, first_reference_loss = torch_gradient(
            first_samples,
            first_labels,
        )
        second_reference_gradients, _ = torch_gradient(second_samples, second_labels)
        first_reference_flat = contiguous(
            torch.cat([value.detach().reshape(-1) for value in first_reference_gradients])
        )
        first_c_flat = np.ctypeslib.as_array(first_c_gradient.parameters).copy()
        gradient_error = float(np.max(np.abs(first_c_flat - first_reference_flat)))
        gradient_cosine = float(
            np.dot(first_c_flat, first_reference_flat)
            / (np.linalg.norm(first_c_flat) * np.linalg.norm(first_reference_flat))
        )

        utility_before = 0.35
        epsilon = 1.0e-12
        local_sq = sum(float(value.float().pow(2).sum().item()) for value in first_reference_gradients)
        remote_sq = sum(float(value.float().pow(2).sum().item()) for value in second_reference_gradients)
        dot = sum(
            float((local.float() * remote.float()).sum().item())
            for local, remote in zip(first_reference_gradients, second_reference_gradients)
        )
        denominator = math.sqrt(local_sq) * math.sqrt(remote_sq)
        agreement = 0.0 if denominator <= epsilon else max(-1.0, min(1.0, dot / denominator))
        alpha = 0.5 * max(utility_before, 0.0) * max(agreement, 0.0)
        scale = math.sqrt(local_sq / max(remote_sq, epsilon))
        expected_updated = head_values.copy()
        second_reference_flat = contiguous(
            torch.cat([value.detach().reshape(-1) for value in second_reference_gradients])
        )
        expected_updated -= 0.01 * (
            (1.0 - alpha) * first_reference_flat + alpha * scale * second_reference_flat
        )

        stats = MixStats()
        require_ok(
            library.pulse_head_apply_peer_update(
                ctypes.byref(c_head),
                ctypes.byref(first_c_gradient),
                ctypes.byref(second_c_gradient),
                ctypes.c_float(0.01),
                ctypes.c_float(0.5),
                ctypes.c_float(utility_before),
                ctypes.c_float(epsilon),
                ctypes.byref(stats),
            ),
            "pulse_head_apply_peer_update",
        )
        c_updated = np.ctypeslib.as_array(c_head.parameters).copy()
        update_error = float(np.max(np.abs(c_updated - expected_updated)))

        library_handle = library._handle
        del library
        if os.name == "nt":
            _ctypes.FreeLibrary(library_handle)
        else:
            _ctypes.dlclose(library_handle)

    print(f"encoder max_abs_error={embedding_error:.9g}")
    print(f"loss abs_error={abs(first_c_loss - first_reference_loss):.9g}")
    print(
        f"split_path gradient max_abs_error={split_gradient_error:.9g} "
        f"loss abs_error={split_loss_error:.9g}"
    )
    print(f"gradient cosine={gradient_cosine:.9g} max_abs_error={gradient_error:.9g}")
    print(f"updated_head max_abs_error={update_error:.9g}")

    if embedding_error > 2.0e-5:
        raise AssertionError("encoder error exceeds tolerance")
    if abs(first_c_loss - first_reference_loss) > 2.0e-5:
        raise AssertionError("loss error exceeds tolerance")
    if split_gradient_error != 0.0 or split_loss_error != 0.0:
        raise AssertionError("split and streaming paths differ")
    if gradient_cosine < 0.999999 or gradient_error > 2.0e-5:
        raise AssertionError("gradient mismatch exceeds tolerance")
    if update_error > 2.0e-6:
        raise AssertionError("updated head mismatch exceeds tolerance")


if __name__ == "__main__":
    main()
