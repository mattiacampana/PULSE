#!/usr/bin/env python3
"""Generate PULSE HAR model, replay, and PyTorch golden vectors.

The upstream PULSE repository does not contain a trained checkpoint or a frozen
HAR minibatch.  By default this generator therefore creates a clearly labelled
deterministic synthetic fixture.  ``--artifact-npz`` instead imports the strict
versioned artifact produced by ``tools/export_pulse_artifact.py``.  Both paths
use the exact model dimensions and parameter ordering from PULSE/models.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from pathlib import Path
from typing import Iterable, TextIO

import numpy as np


SEED = 0x50554C53
BATCHES = 4
BATCH_SIZE = 16
CHANNELS = 3
WINDOW = 128
CLASSES = 6
EMBEDDING = 64
HIDDEN = 64
LEARNING_RATE = np.float32(0.01)
ALPHA_MAX = np.float32(0.5)
# Match the accepted-path microbenchmark precondition: one canonical positive
# observation from U0=0 with utility momentum mu=0.2.
CORRECTNESS_UTILITY = np.float32(0.2)
NORM_EPSILON = np.float32(1.0e-12)
ARTIFACT_SCHEMA = "senswear.pulse.artifact.v1"
ARTIFACT_NORMALIZATION_TOLERANCE = np.float32(2.0e-4)

ARTIFACT_ARRAY_SPECS = {
    "encoder.features.0.weight": ((32, 3, 5), np.dtype(np.float32)),
    "encoder.features.0.bias": ((32,), np.dtype(np.float32)),
    "encoder.features.3.weight": ((64, 32, 5), np.dtype(np.float32)),
    "encoder.features.3.bias": ((64,), np.dtype(np.float32)),
    "encoder.projection.weight": ((64, 64), np.dtype(np.float32)),
    "encoder.projection.bias": ((64,), np.dtype(np.float32)),
    "head.feature_layers.0.weight": ((64, 64), np.dtype(np.float32)),
    "head.feature_layers.0.bias": ((64,), np.dtype(np.float32)),
    "head.output_layer.weight": ((6, 64), np.dtype(np.float32)),
    "head.output_layer.bias": ((6,), np.dtype(np.float32)),
    "head.flat": ((4550,), np.dtype(np.float32)),
    "replay.x": ((4, 16, 3, 128), np.dtype(np.float32)),
    "replay.y": ((4, 16), np.dtype(np.uint8)),
    "replay.source_indices": ((4, 16), np.dtype(np.int64)),
}
ARTIFACT_OWNER_KEY = "replay.owner_ids"
ARTIFACT_OWNER_SHAPE = (4, 16)

ARTIFACT_HEAD_ORDER = (
    "head.feature_layers.0.weight",
    "head.feature_layers.0.bias",
    "head.output_layer.weight",
    "head.output_layer.bias",
)

ARTIFACT_REPLAY_ROLES = (
    "scheduled_r_step_0",
    "scheduled_r_step_1",
    "initiator_local",
    "responder_remote",
)


def _f32(values: np.ndarray | Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def build_fixture() -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(SEED))

    # He-scaled deterministic parameters keep activations representative while
    # making it impossible to confuse this fixture with learned weights.
    conv1_w = _f32(rng.normal(0.0, np.sqrt(2.0 / 15.0), (32, 3, 5)))
    conv1_b = _f32(rng.normal(0.0, 0.01, 32))
    conv2_w = _f32(rng.normal(0.0, np.sqrt(2.0 / 160.0), (64, 32, 5)))
    conv2_b = _f32(rng.normal(0.0, 0.01, 64))
    projection_w = _f32(rng.normal(0.0, np.sqrt(2.0 / 64.0), (64, 64)))
    projection_b = _f32(rng.normal(0.0, 0.01, 64))
    head_w1 = _f32(rng.normal(0.0, np.sqrt(2.0 / 64.0), (64, 64)))
    head_b1 = _f32(rng.normal(0.0, 0.01, 64))
    head_w2 = _f32(rng.normal(0.0, np.sqrt(2.0 / 64.0), (6, 64)))
    head_b2 = _f32(rng.normal(0.0, 0.01, 6))

    samples = np.empty((BATCHES, BATCH_SIZE, CHANNELS, WINDOW), dtype=np.float32)
    labels = np.empty((BATCHES, BATCH_SIZE), dtype=np.uint8)
    t = np.arange(WINDOW, dtype=np.float32) / np.float32(WINDOW)
    for batch in range(BATCHES):
        for sample in range(BATCH_SIZE):
            label = (sample + 2 * batch) % CLASSES
            labels[batch, sample] = label
            for channel in range(CHANNELS):
                fundamental = np.float32(label + 1 + channel)
                phase = np.float32(0.19 * batch + 0.31 * sample + 0.47 * channel)
                signal = (
                    np.sin(np.float32(2.0 * np.pi) * fundamental * t + phase)
                    + np.float32(0.35)
                    * np.cos(np.float32(2.0 * np.pi) * (fundamental + 0.5) * t)
                    + _f32(rng.normal(0.0, 0.08, WINDOW))
                ).astype(np.float32)
                mean = np.mean(signal, dtype=np.float32)
                std = np.std(signal, dtype=np.float32)
                samples[batch, sample, channel] = (signal - mean) / max(
                    float(std), 1.0e-6
                )

    head = np.concatenate(
        [head_w1.reshape(-1), head_b1, head_w2.reshape(-1), head_b2]
    ).astype(np.float32)
    return {
        "conv1_w": conv1_w,
        "conv1_b": conv1_b,
        "conv2_w": conv2_w,
        "conv2_b": conv2_b,
        "projection_w": projection_w,
        "projection_b": projection_b,
        "head": head,
        "samples": samples,
        "labels": labels,
    }


def fixture_hashes(data: dict[str, np.ndarray]) -> tuple[str, str]:
    encoder_keys = (
        "conv1_w",
        "conv1_b",
        "conv2_w",
        "conv2_b",
        "projection_w",
        "projection_b",
    )
    encoder_digest = hashlib.sha256()
    fixture_digest = hashlib.sha256()
    for key in (*encoder_keys, "head", "samples", "labels"):
        raw = np.ascontiguousarray(data[key]).tobytes(order="C")
        fixture_digest.update(key.encode("ascii") + b"\0" + raw)
        if key in encoder_keys:
            encoder_digest.update(key.encode("ascii") + b"\0" + raw)
    return fixture_digest.hexdigest(), encoder_digest.hexdigest()


def initial_head_fp32le_crc32(data: dict[str, np.ndarray]) -> int:
    """CRC-32/IEEE of the canonical little-endian FP32 head wire image."""

    return fp32le_crc32(data["head"])


def fp32le_crc32(values: np.ndarray) -> int:
    """CRC-32/IEEE of a contiguous canonical little-endian FP32 tensor."""

    wire_image = np.ascontiguousarray(values, dtype=np.dtype("<f4"))
    return zlib.crc32(wire_image.tobytes(order="C")) & 0xFFFFFFFF


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_payload_hash(arrays: dict[str, np.ndarray]) -> str:
    """Reproduce export_pulse_artifact.py's canonical payload hash."""

    digest = hashlib.sha256()
    for key in sorted((*ARTIFACT_ARRAY_SPECS, ARTIFACT_OWNER_KEY)):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _artifact_text_scalar(archive: np.lib.npyio.NpzFile, key: str) -> str:
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"artifact {key} must be a scalar Unicode string")
    return str(value.item())


def _artifact_text_vector(
    archive: np.lib.npyio.NpzFile,
    key: str,
    expected: tuple[str, ...],
) -> None:
    value = np.asarray(archive[key])
    if value.shape != (len(expected),) or value.dtype.kind != "U":
        raise ValueError(
            f"artifact {key} must be a {len(expected)}-element Unicode vector"
        )
    actual = tuple(str(item) for item in value.tolist())
    if actual != expected:
        raise ValueError(f"artifact {key} is {actual}, expected {expected}")


def load_artifact_fixture(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load and strictly validate an exported PULSE v1 artifact."""

    if not path.is_file():
        raise ValueError(f"PULSE artifact does not exist: {path}")
    metadata_keys = {
        "artifact.schema",
        "artifact.payload_sha256",
        "head.flat_order",
        "replay.roles",
    }
    required_keys = set(ARTIFACT_ARRAY_SPECS) | {ARTIFACT_OWNER_KEY} | metadata_keys
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(required_keys - set(archive.files))
            if missing:
                raise ValueError(f"PULSE artifact is missing keys: {missing}")

            schema = _artifact_text_scalar(archive, "artifact.schema")
            if schema != ARTIFACT_SCHEMA:
                raise ValueError(
                    f"unsupported PULSE artifact schema {schema!r}; "
                    f"expected {ARTIFACT_SCHEMA!r}"
                )
            embedded_payload_hash = _artifact_text_scalar(
                archive, "artifact.payload_sha256"
            )
            if len(embedded_payload_hash) != 64 or any(
                character not in "0123456789abcdef"
                for character in embedded_payload_hash
            ):
                raise ValueError(
                    "artifact.payload_sha256 must be a lowercase SHA-256 hex digest"
                )
            _artifact_text_vector(archive, "head.flat_order", ARTIFACT_HEAD_ORDER)
            _artifact_text_vector(archive, "replay.roles", ARTIFACT_REPLAY_ROLES)

            arrays: dict[str, np.ndarray] = {}
            for key, (shape, dtype) in ARTIFACT_ARRAY_SPECS.items():
                value = np.asarray(archive[key])
                if value.shape != shape:
                    raise ValueError(
                        f"artifact {key} has shape {value.shape}, expected {shape}"
                    )
                if value.dtype != dtype:
                    raise ValueError(
                        f"artifact {key} has dtype {value.dtype}, expected {dtype}"
                    )
                if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
                    raise ValueError(f"artifact {key} contains NaN or infinity")
                arrays[key] = np.ascontiguousarray(value).copy()
            owner_ids = np.asarray(archive[ARTIFACT_OWNER_KEY])
            if owner_ids.shape != ARTIFACT_OWNER_SHAPE:
                raise ValueError(
                    f"artifact {ARTIFACT_OWNER_KEY} has shape {owner_ids.shape}, "
                    f"expected {ARTIFACT_OWNER_SHAPE}"
                )
            if owner_ids.dtype.kind != "U":
                raise ValueError(
                    f"artifact {ARTIFACT_OWNER_KEY} has dtype {owner_ids.dtype}, "
                    "expected a Unicode string dtype"
                )
            if any(not owner for owner in owner_ids.reshape(-1).tolist()):
                raise ValueError(f"artifact {ARTIFACT_OWNER_KEY} contains an empty owner")
            arrays[ARTIFACT_OWNER_KEY] = np.ascontiguousarray(owner_ids).copy()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not read PULSE artifact {path}: {exc}") from exc

    calculated_payload_hash = _artifact_payload_hash(arrays)
    if calculated_payload_hash != embedded_payload_hash:
        raise ValueError(
            "PULSE artifact payload SHA-256 mismatch: "
            f"embedded {embedded_payload_hash}, calculated {calculated_payload_hash}"
        )

    expected_head = np.concatenate(
        [arrays[key].reshape(-1) for key in ARTIFACT_HEAD_ORDER]
    ).astype(np.float32, copy=False)
    if not np.array_equal(arrays["head.flat"], expected_head):
        raise ValueError("artifact head.flat is not exact W1,b1,W2,b2 order")

    labels = arrays["replay.y"]
    if np.any(labels >= CLASSES):
        raise ValueError("artifact replay.y contains a label outside [0, 5]")
    source_indices = arrays["replay.source_indices"]
    if np.any(source_indices < 0) or np.unique(source_indices).size != source_indices.size:
        raise ValueError(
            "artifact replay.source_indices must contain 64 distinct non-negative indices"
        )
    owner_ids = arrays[ARTIFACT_OWNER_KEY]
    initiator_owners = np.unique(owner_ids[:3]).tolist()
    responder_owners = np.unique(owner_ids[3]).tolist()
    if (
        len(initiator_owners) != 1
        or len(responder_owners) != 1
        or initiator_owners[0] == responder_owners[0]
    ):
        raise ValueError(
            "artifact replay owners must bind the first three roles to one initiator "
            "and the fourth role to one distinct responder"
        )

    samples = arrays["replay.x"]
    channel_means = samples.mean(axis=-1, dtype=np.float32)
    channel_stds = samples.std(axis=-1, dtype=np.float32)
    if np.max(np.abs(channel_means)) > ARTIFACT_NORMALIZATION_TOLERANCE:
        raise ValueError("artifact replay.x is not zero-mean per window/channel")
    std_error = np.minimum(np.abs(channel_stds), np.abs(channel_stds - np.float32(1.0)))
    if np.max(std_error) > ARTIFACT_NORMALIZATION_TOLERANCE:
        raise ValueError(
            "artifact replay.x standard deviation is not near 1 (or 0 for a constant channel)"
        )

    data = {
        "conv1_w": arrays["encoder.features.0.weight"],
        "conv1_b": arrays["encoder.features.0.bias"],
        "conv2_w": arrays["encoder.features.3.weight"],
        "conv2_b": arrays["encoder.features.3.bias"],
        "projection_w": arrays["encoder.projection.weight"],
        "projection_b": arrays["encoder.projection.bias"],
        "head": arrays["head.flat"],
        "samples": samples,
        "labels": labels,
    }
    provenance: dict[str, object] = {
        "kind": "exported_pulse_npz",
        "source": path.name,
        "source_path": str(path.resolve()),
        "artifact_sha256": _sha256_file(path),
        "payload_sha256": calculated_payload_hash,
        "schema": schema,
        "replay_source_indices": source_indices.tolist(),
        "replay_owner_ids": owner_ids.tolist(),
        "initiator_owner": initiator_owners[0],
        "responder_owner": responder_owners[0],
    }
    return data, provenance


def synthetic_provenance(data: dict[str, np.ndarray]) -> dict[str, object]:
    fixture_hash, _ = fixture_hashes(data)
    return {
        "kind": "deterministic_synthetic",
        "source": "tools/generate_pulse_fixture.py",
        "source_path": None,
        "artifact_sha256": fixture_hash,
        "payload_sha256": fixture_hash,
        "schema": "senswear-pulse-fixture-v1",
    }


def c_string(value: object) -> str:
    """Return an ASCII C string literal, preserving arbitrary UTF-8 source names."""

    escaped = []
    for byte in str(value).encode("utf-8"):
        if byte == ord("\\"):
            escaped.append("\\\\")
        elif byte == ord('"'):
            escaped.append('\\"')
        elif 32 <= byte <= 126:
            escaped.append(chr(byte))
        else:
            escaped.append(f"\\{byte:03o}")
    return '"' + "".join(escaped) + '"'


def c_float(value: np.float32 | float) -> str:
    value32 = np.float32(value)
    if not np.isfinite(value32):
        raise ValueError("fixture contains a non-finite value")
    return float(value32).hex() + "f"


def write_float_array(
    out: TextIO,
    declaration: str,
    values: np.ndarray,
    section: str,
    aggregate: bool = False,
) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    opening = "{{" if aggregate else "{"
    closing = "}};" if aggregate else "};"
    out.write(f"{declaration} PULSE_DATA_SECTION(\"{section}\") = {opening}\n")
    for start in range(0, flat.size, 6):
        row = ", ".join(c_float(value) for value in flat[start : start + 6])
        out.write(f"\t{row},\n")
    out.write(closing + "\n\n")


def write_fixture_source(
    path: Path,
    data: dict[str, np.ndarray],
    provenance: dict[str, object] | None = None,
) -> None:
    fixture_hash, encoder_hash = fixture_hashes(data)
    head_crc32 = initial_head_fp32le_crc32(data)
    provenance = provenance or synthetic_provenance(data)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(
            "/* Generated by tools/generate_pulse_fixture.py; do not edit. */\n"
            '#include "pulse_fixture.h"\n\n'
            "#include <zephyr/toolchain.h>\n\n"
            "#define PULSE_DATA_SECTION(name) __aligned(4) __attribute__((section(name)))\n\n"
        )
        write_float_array(out, "static const float conv1_weight[480]", data["conv1_w"], ".pulse.encoder")
        write_float_array(out, "static const float conv1_bias[32]", data["conv1_b"], ".pulse.encoder")
        write_float_array(out, "static const float conv2_weight[10240]", data["conv2_w"], ".pulse.encoder")
        write_float_array(out, "static const float conv2_bias[64]", data["conv2_b"], ".pulse.encoder")
        write_float_array(out, "static const float projection_weight[4096]", data["projection_w"], ".pulse.encoder")
        write_float_array(out, "static const float projection_bias[64]", data["projection_b"], ".pulse.encoder")
        write_float_array(out, "static const float initial_head[4550]", data["head"], ".pulse.head")
        for batch in range(BATCHES):
            write_float_array(
                out,
                f"static const float batch_{batch}_samples[{BATCH_SIZE * CHANNELS * WINDOW}]",
                data["samples"][batch],
                ".pulse.fixture",
            )
            labels = ", ".join(str(int(value)) for value in data["labels"][batch])
            out.write(
                f"static const uint8_t batch_{batch}_labels[{BATCH_SIZE}] "
                'PULSE_DATA_SECTION(".pulse.fixture") = {' + labels + "};\n\n"
            )

        out.write(
            "const pulse_encoder_artifact_t pulse_fixture_encoder = {\n"
            "\t.conv1_weight = {conv1_weight, 480U},\n"
            "\t.conv1_bias = {conv1_bias, 32U},\n"
            "\t.conv2_weight = {conv2_weight, 10240U},\n"
            "\t.conv2_bias = {conv2_bias, 64U},\n"
            "\t.projection_weight = {projection_weight, 4096U},\n"
            "\t.projection_bias = {projection_bias, 64U},\n"
            "};\n\n"
            "const pulse_head_artifact_t pulse_fixture_initial_head = {\n"
            "\t.parameters = {initial_head, 4550U},\n"
            "};\n\n"
            "const pulse_batch_t pulse_fixture_batches[PULSE_FIXTURE_BATCH_COUNT] = {\n"
        )
        for batch in range(BATCHES):
            out.write(
                "\t{"
                f"batch_{batch}_samples, {BATCH_SIZE * CHANNELS * WINDOW}U, "
                f"{CHANNELS * WINDOW}U, batch_{batch}_labels, {BATCH_SIZE}U, "
                f"{BATCH_SIZE}U"
                "},\n"
            )
        out.write(
            "};\n\n"
            f'const char pulse_fixture_sha256[] = "{fixture_hash}";\n'
            f'const char pulse_fixture_encoder_sha256[] = "{encoder_hash}";\n'
            f"const uint32_t pulse_fixture_initial_head_fp32le_crc32 = 0x{head_crc32:08x}U;\n"
            "const char pulse_fixture_artifact_kind[] = "
            f'{c_string(provenance["kind"])};\n'
            "const char pulse_fixture_artifact_source[] = "
            f'{c_string(provenance["source"])};\n'
            "const char pulse_fixture_artifact_sha256[] = "
            f'{c_string(provenance["artifact_sha256"])};\n'
        )


def torch_reference(
    data: dict[str, np.ndarray], *, require_pytorch: bool = False
) -> dict[str, np.ndarray | float]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        if require_pytorch:
            raise RuntimeError(
                "PyTorch is required for a correctness image; set "
                "PULSE_FIXTURE_PYTHON to an interpreter that can import torch"
            ) from exc
        return numpy_reference(data)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    def tensor(key: str, requires_grad: bool = False):
        return torch.tensor(data[key], dtype=torch.float32, requires_grad=requires_grad)

    conv1_w = tensor("conv1_w")
    conv1_b = tensor("conv1_b")
    conv2_w = tensor("conv2_w")
    conv2_b = tensor("conv2_b")
    projection_w = tensor("projection_w")
    projection_b = tensor("projection_b")

    def embedding(samples: np.ndarray):
        x = torch.tensor(samples, dtype=torch.float32)
        x = functional.relu(functional.conv1d(x, conv1_w, conv1_b, padding=2))
        x = functional.max_pool1d(x, 2)
        x = functional.relu(functional.conv1d(x, conv2_w, conv2_b, padding=2))
        x = x.mean(dim=2)
        return functional.linear(x, projection_w, projection_b)

    def gradient(batch_index: int):
        head = torch.tensor(data["head"], dtype=torch.float32, requires_grad=True)
        w1 = head[0:4096].reshape(64, 64)
        b1 = head[4096:4160]
        w2 = head[4160:4544].reshape(6, 64)
        b2 = head[4544:4550]
        z = embedding(data["samples"][batch_index])
        logits = functional.linear(functional.relu(functional.linear(z, w1, b1)), w2, b2)
        labels = torch.tensor(data["labels"][batch_index].astype(np.int64))
        loss = functional.cross_entropy(logits, labels, reduction="mean")
        loss.backward()
        return head.grad.detach().numpy().copy(), float(loss.detach())

    single_embedding = embedding(data["samples"][0, 0:1])
    initial = torch.tensor(data["head"], dtype=torch.float32)
    initial_w1 = initial[0:4096].reshape(64, 64)
    initial_b1 = initial[4096:4160]
    initial_w2 = initial[4160:4544].reshape(6, 64)
    initial_b2 = initial[4544:4550]
    single_logits = functional.linear(
        functional.relu(functional.linear(single_embedding, initial_w1, initial_b1)),
        initial_w2,
        initial_b2,
    )

    local_gradient, local_loss = gradient(2)
    remote_gradient, remote_loss = gradient(3)
    # Agreement exactly follows PULSENode._cosine_agreement: concatenate all
    # named gradients, then run one FP32 norm/dot reduction.  Remote scaling is
    # deliberately separate because _apply_agreement_calibrated_gradient sums
    # one FP32 squared-norm reduction per named parameter as Python floats.
    boundaries = (0, 4096, 4160, 4544, 4550)
    local_vector = torch.tensor(local_gradient, dtype=torch.float32)
    remote_vector = torch.tensor(remote_gradient, dtype=torch.float32)
    denominator_tensor = local_vector.norm() * remote_vector.norm()
    denominator = float(denominator_tensor.item())
    agreement = 0.0 if denominator <= float(NORM_EPSILON) else float(
        torch.clamp(
            torch.dot(local_vector, remote_vector) / denominator_tensor,
            -1.0,
            1.0,
        ).item()
    )
    local_parts = [torch.tensor(local_gradient[a:b]) for a, b in zip(boundaries, boundaries[1:])]
    remote_parts = [torch.tensor(remote_gradient[a:b]) for a, b in zip(boundaries, boundaries[1:])]
    local_norm_sq = sum(float(torch.sum(part**2).item()) for part in local_parts)
    remote_norm_sq = sum(float(torch.sum(part**2).item()) for part in remote_parts)
    alpha = float(ALPHA_MAX) * max(float(CORRECTNESS_UTILITY), 0.0) * max(agreement, 0.0)
    remote_scale = math.sqrt(local_norm_sq / max(remote_norm_sq, float(NORM_EPSILON)))
    aggregate = (
        (np.float32(1.0) - np.float32(alpha)) * local_gradient
        + np.float32(alpha) * (remote_gradient * np.float32(remote_scale))
    ).astype(np.float32)
    mixed_head = (data["head"] - LEARNING_RATE * aggregate).astype(np.float32)

    return {
        "embedding": single_embedding.detach().numpy().reshape(-1),
        "logits": single_logits.detach().numpy().reshape(-1),
        "local_gradient": local_gradient,
        "remote_gradient": remote_gradient,
        "mixed_head": mixed_head,
        "local_loss": local_loss,
        "remote_loss": remote_loss,
        "agreement": agreement,
        "alpha": alpha,
        "remote_scale": remote_scale,
        "torch_version": torch.__version__,
        "reference_name": "pytorch_cpu_float32",
    }


def numpy_reference(data: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    """Portable fallback with the same frozen PULSE topology and equations."""

    def embedding(samples: np.ndarray) -> np.ndarray:
        padded = np.pad(samples, ((0, 0), (0, 0), (2, 2)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, 5, axis=2)
        conv1 = np.einsum("bctk,ock->bot", windows, data["conv1_w"], optimize=False)
        conv1 = np.maximum(
            conv1.astype(np.float32) + data["conv1_b"][None, :, None], 0.0
        ).astype(np.float32)
        pooled = conv1.reshape(conv1.shape[0], 32, 64, 2).max(axis=3)
        padded = np.pad(pooled, ((0, 0), (0, 0), (2, 2)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, 5, axis=2)
        conv2 = np.einsum("bitk,oik->bot", windows, data["conv2_w"], optimize=False)
        conv2 = np.maximum(
            conv2.astype(np.float32) + data["conv2_b"][None, :, None], 0.0
        ).astype(np.float32)
        features = np.mean(conv2, axis=2, dtype=np.float32)
        return (
            features @ data["projection_w"].T + data["projection_b"][None, :]
        ).astype(np.float32)

    head = data["head"]
    w1 = head[0:4096].reshape(64, 64)
    b1 = head[4096:4160]
    w2 = head[4160:4544].reshape(6, 64)
    b2 = head[4544:4550]

    def gradient(batch_index: int) -> tuple[np.ndarray, float]:
        z = embedding(data["samples"][batch_index])
        hidden_pre = (z @ w1.T + b1[None, :]).astype(np.float32)
        hidden = np.maximum(hidden_pre, 0.0).astype(np.float32)
        logits = (hidden @ w2.T + b2[None, :]).astype(np.float32)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted).astype(np.float32)
        probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        labels = data["labels"][batch_index].astype(np.int64)
        loss = -np.mean(
            np.log(probabilities[np.arange(BATCH_SIZE), labels]), dtype=np.float32
        )
        delta2 = probabilities.copy()
        delta2[np.arange(BATCH_SIZE), labels] -= np.float32(1.0)
        delta2 *= np.float32(1.0 / BATCH_SIZE)
        grad_w2 = (delta2.T @ hidden).astype(np.float32)
        grad_b2 = np.sum(delta2, axis=0, dtype=np.float32)
        delta1 = (delta2 @ w2).astype(np.float32)
        delta1 *= hidden_pre > 0.0
        grad_w1 = (delta1.T @ z).astype(np.float32)
        grad_b1 = np.sum(delta1, axis=0, dtype=np.float32)
        return (
            np.concatenate(
                [grad_w1.reshape(-1), grad_b1, grad_w2.reshape(-1), grad_b2]
            ).astype(np.float32),
            float(loss),
        )

    single_embedding = embedding(data["samples"][0, 0:1])
    hidden = np.maximum(single_embedding @ w1.T + b1[None, :], 0.0)
    single_logits = (hidden @ w2.T + b2[None, :]).astype(np.float32)
    local_gradient, local_loss = gradient(2)
    remote_gradient, remote_loss = gradient(3)
    boundaries = (0, 4096, 4160, 4544, 4550)

    local_norm_sq = sum(
        float(np.sum(local_gradient[start:end] ** 2, dtype=np.float32))
        for start, end in zip(boundaries, boundaries[1:])
    )
    remote_norm_sq = sum(
        float(np.sum(remote_gradient[start:end] ** 2, dtype=np.float32))
        for start, end in zip(boundaries, boundaries[1:])
    )
    local_norm = np.float32(np.sqrt(np.sum(local_gradient**2, dtype=np.float32)))
    remote_norm = np.float32(np.sqrt(np.sum(remote_gradient**2, dtype=np.float32)))
    denominator = float(np.float32(local_norm * remote_norm))
    dot_value = np.sum(local_gradient * remote_gradient, dtype=np.float32)
    agreement = 0.0 if denominator <= float(NORM_EPSILON) else float(
        np.clip(np.float32(dot_value / np.float32(denominator)), -1.0, 1.0)
    )
    alpha = float(ALPHA_MAX) * max(float(CORRECTNESS_UTILITY), 0.0) * max(agreement, 0.0)
    remote_scale = math.sqrt(local_norm_sq / max(remote_norm_sq, float(NORM_EPSILON)))
    aggregate_gradient = (
        (np.float32(1.0) - np.float32(alpha)) * local_gradient
        + np.float32(alpha) * remote_gradient * np.float32(remote_scale)
    ).astype(np.float32)
    mixed_head = (head - LEARNING_RATE * aggregate_gradient).astype(np.float32)
    return {
        "embedding": single_embedding.reshape(-1),
        "logits": single_logits.reshape(-1),
        "local_gradient": local_gradient,
        "remote_gradient": remote_gradient,
        "mixed_head": mixed_head,
        "local_loss": local_loss,
        "remote_loss": remote_loss,
        "agreement": agreement,
        "alpha": alpha,
        "remote_scale": remote_scale,
        "torch_version": None,
        "reference_name": "pulse_numpy_float32_fallback",
    }


def write_golden_source(path: Path, reference: dict[str, np.ndarray | float]) -> None:
    remote_gradient_crc32 = fp32le_crc32(reference["remote_gradient"])
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(
            "/* Generated by tools/generate_pulse_fixture.py; do not edit. */\n"
            '#include "pulse_fixture.h"\n\n'
            "#include <zephyr/toolchain.h>\n\n"
            "#define PULSE_DATA_SECTION(name) __aligned(4) __attribute__((section(name)))\n\n"
        )
        write_float_array(out, "const float pulse_fixture_golden_embedding[64]", reference["embedding"], ".pulse.golden")
        write_float_array(out, "const float pulse_fixture_golden_logits[6]", reference["logits"], ".pulse.golden")
        write_float_array(out, "const pulse_gradient_t pulse_fixture_golden_local_gradient", reference["local_gradient"], ".pulse.golden", aggregate=True)
        write_float_array(out, "const pulse_gradient_t pulse_fixture_golden_remote_gradient", reference["remote_gradient"], ".pulse.golden", aggregate=True)
        write_float_array(out, "const pulse_head_t pulse_fixture_golden_mixed_head", reference["mixed_head"], ".pulse.golden", aggregate=True)
        out.write(
            f"const float pulse_fixture_golden_local_loss = {c_float(reference['local_loss'])};\n"
            f"const float pulse_fixture_golden_remote_loss = {c_float(reference['remote_loss'])};\n"
            f"const float pulse_fixture_golden_agreement = {c_float(reference['agreement'])};\n"
            f"const float pulse_fixture_golden_alpha = {c_float(reference['alpha'])};\n"
            f"const float pulse_fixture_golden_remote_scale = {c_float(reference['remote_scale'])};\n"
            f"const float pulse_fixture_golden_utility_before = {c_float(CORRECTNESS_UTILITY)};\n"
            "const uint32_t pulse_fixture_golden_remote_gradient_fp32le_crc32 = "
            f"0x{remote_gradient_crc32:08x}U;\n"
            f'const char pulse_fixture_reference_name[] = "{reference["reference_name"]}";\n'
        )


def write_manifest(
    path: Path,
    data: dict[str, np.ndarray],
    reference: dict[str, np.ndarray | float] | None,
    provenance: dict[str, object] | None = None,
) -> None:
    fixture_hash, encoder_hash = fixture_hashes(data)
    head_crc32 = initial_head_fp32le_crc32(data)
    provenance = provenance or synthetic_provenance(data)
    synthetic = provenance["kind"] == "deterministic_synthetic"
    manifest = {
        "schema": "senswear-pulse-fixture-v1",
        "source": (
            "deterministic synthetic replay; not a trained checkpoint or live capture"
            if synthetic
            else (
                "supplied PULSE encoder, provenance-bound canonical or supplied head, "
                "and normalized replay"
            )
        ),
    }
    if synthetic:
        manifest["seed"] = SEED
    manifest.update({
        "fixture_sha256": fixture_hash,
        "encoder_sha256": encoder_hash,
        "initial_head_fp32le_crc32": head_crc32,
        "artifact": provenance,
        "model": {
            "input": [CHANNELS, WINDOW],
            "classes": CLASSES,
            "embedding": EMBEDDING,
            "hidden": HIDDEN,
            "head_parameter_count": 4550,
            "head_order": ["W1[64,64]", "b1[64]", "W2[6,64]", "b2[6]"],
        },
        "batches": BATCHES,
        "batch_size": BATCH_SIZE,
    })
    if reference is not None:
        manifest["reference"] = {
            "implementation": reference["reference_name"],
            "torch_version": reference["torch_version"],
            "local_loss": reference["local_loss"],
            "remote_loss": reference["remote_loss"],
            "agreement": reference["agreement"],
            "alpha": reference["alpha"],
            "remote_scale": reference["remote_scale"],
            "remote_gradient_fp32le_crc32": fp32le_crc32(reference["remote_gradient"]),
            "utility_before": float(CORRECTNESS_UTILITY),
            "reduction_contract": (
                "agreement=concatenated_fp32_norm_dot;"
                "scale=sum_of_per_parameter_fp32_norm_squares_as_python_float"
            ),
        }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate C fixtures and optional golden vectors from either the "
            "deterministic synthetic default or a validated exported PULSE NPZ."
        )
    )
    parser.add_argument("--fixture-c", type=Path, required=True)
    parser.add_argument("--golden-c", type=Path)
    parser.add_argument(
        "--require-pytorch",
        action="store_true",
        help="fail instead of generating NumPy fallback golden vectors",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--artifact-npz",
        type=Path,
        help=(
            "optional senswear.pulse.artifact.v1 file produced by "
            "export_pulse_artifact.py; without it the synthetic fixture is used"
        ),
    )
    args = parser.parse_args(argv)

    if args.require_pytorch and args.golden_c is None:
        parser.error("--require-pytorch requires --golden-c")

    output_paths = [args.fixture_c.resolve(), args.manifest.resolve()]
    if args.golden_c is not None:
        output_paths.append(args.golden_c.resolve())
    if len(set(output_paths)) != len(output_paths):
        parser.error("--fixture-c, --golden-c, and --manifest must be distinct")
    if (
        args.artifact_npz is not None
        and args.artifact_npz.resolve() in output_paths
    ):
        parser.error("--artifact-npz must not also be an output path")

    args.fixture_c.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.golden_c is not None:
        args.golden_c.parent.mkdir(parents=True, exist_ok=True)

    if args.artifact_npz is None:
        data = build_fixture()
        provenance = synthetic_provenance(data)
    else:
        try:
            data, provenance = load_artifact_fixture(args.artifact_npz)
        except ValueError as exc:
            parser.error(str(exc))
    try:
        reference = (
            torch_reference(data, require_pytorch=args.require_pytorch)
            if args.golden_c is not None
            else None
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    write_fixture_source(args.fixture_c, data, provenance)
    if args.golden_c is not None and reference is not None:
        write_golden_source(args.golden_c, reference)
    write_manifest(args.manifest, data, reference, provenance)


if __name__ == "__main__":
    main()
