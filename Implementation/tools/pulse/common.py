"""Shared, dependency-free helpers for PULSE experiment tooling."""

from __future__ import annotations

import csv
import hashlib
import math
import re
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TRUE_VALUES = {"1", "true", "yes", "y", "pass", "passed", "ok", "success"}
FALSE_VALUES = {"0", "false", "no", "n", "fail", "failed", "error", "timeout"}

# Immutable/configuration context emitted in PULSE_META and inherited by every
# parsed firmware record.  Downstream tools use this exact list both to retain
# provenance and to prevent measurements from incompatible images or runtime
# configurations from being pooled into one statistic.
PULSE_BUILD_CONTEXT_FIELDS = (
    "build_context_sha256",
    "firmware_revision",
    "hardware_revision",
    "build_variant",
    "build_guard",
    "artifact_sha256",
    "fixture_hash",
    "artifact_kind",
    "artifact_source",
    "model_id",
    "model_version",
    "encoder_hash",
    "input_source",
    "tensor_precision",
    "batch_size",
    "local_steps",
    "ncs_version",
    "zephyr_version",
    "compiler_version",
    "optimization",
    "cpu_hz",
    "wall_counter_hz",
    "duration_us_semantics",
    "task",
    "class_count",
    "parameter_count",
    "activation",
    "loss_reduction",
    "serialized_bytes_per_parameter",
    "head_serialized_bytes",
    "mutable_head_bytes",
    "compute_workspace_bytes",
    "gradient_buffer_bytes_each",
    "gradient_buffer_count",
    "received_head_bytes",
    "utility_table_bytes",
    "peer_entry_bytes",
    "timing_counter_hz",
    "learning_rate",
    "alpha_max",
    "utility_mu",
    "initial_utility",
    "ucb_beta",
    "epsilon",
    "peer_limit",
    "tie_break_policy",
    "trial_state_policy",
    "enabled_sensors",
    "warmup_repetitions",
    "measured_repetitions",
    "timeout_ms",
    "discovery_settle_ms",
    "tx_power_dbm",
    "security_state",
    "execution_context",
    "cycle_timing_semantics",
    "stack_watermark_scope",
    "serialization_timing_semantics",
    "transfer_timing_semantics",
    "deserialization_timing_semantics",
    "responder_gradient_endpoint",
    "att_value_counter_semantics",
    "communication_accounting",
    "result_release_semantics",
    "link_layer_accounting",
    "capture_pacing",
    "capture_quiet_guard_ms",
    "capture_min_event_period_ms",
    "capture_envelope_semantics",
    "power_alignment",
    "event_workqueue_stack_capacity_bytes",
    "system_heap_capacity_bytes",
    "marker_available",
    "marker_event_gate",
    "marker_bit0",
    "marker_bit1",
    "marker_bit2",
    "marker_stage_0",
    "marker_stage_1",
    "marker_stage_2",
    "marker_stage_3",
    "marker_stage_4",
    "marker_stage_5",
    "marker_stage_6",
    "marker_stage_7",
)

# Negotiated per-event link settings that alter latency, radio energy, and
# packetization.  RSSI is intentionally not here: it is an observed covariate
# to retain/summarize, rather than a configuration stratum.
PULSE_EVENT_CONFIGURATION_FIELDS = (
    "att_mtu",
    "data_length",
    "tx_phy",
    "rx_phy",
    "connection_interval_units",
)


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 of an immutable analysis input."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_name(value: str) -> str:
    """Return a stable identifier suitable for matching CSV column aliases."""

    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_bool(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def format_number(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".12g")
    return str(value)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a comma/semicolon/tab-delimited file as dictionaries."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV header is missing")
        rows: list[dict[str, str]] = []
        for row in reader:
            if row is None:
                continue
            cleaned = {
                str(key).strip(): ("" if value is None else str(value).strip())
                for key, value in row.items()
                if key is not None
            }
            if any(cleaned.values()):
                rows.append(cleaned)
        return rows


def ordered_fields(rows: Sequence[Mapping[str, object]], preferred: Sequence[str] = ()) -> list[str]:
    if not rows:
        return list(dict.fromkeys(preferred))
    keys = {str(key) for row in rows for key in row}
    result = [key for key in preferred if key in keys]
    result.extend(sorted(keys.difference(result)))
    return result


def write_csv_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    preferred: Sequence[str] = (),
    fieldnames: Sequence[str] | None = None,
) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames) if fieldnames is not None else ordered_fields(materialized, preferred)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: format_number(row.get(key, "")) for key in fields})


def quantile(values: Sequence[float], probability: float) -> float:
    """R-7/NumPy-style linear sample quantile, implemented with stdlib only."""

    if not values:
        raise ValueError("cannot calculate a quantile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "q1": math.nan,
            "q3": math.nan,
            "iqr": math.nan,
            "p95": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": quantile(values, 0.5),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "p95": quantile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def row_success(row: Mapping[str, object]) -> bool | None:
    for key in ("success", "pass", "passed", "ok"):
        if key in row and str(row[key]).strip():
            return parse_bool(row[key])
    status = str(row.get("status", "")).strip().lower()
    if status:
        if status in TRUE_VALUES or status in {"complete", "completed", "accepted"}:
            return True
        if status in FALSE_VALUES or status in {"rejected", "skipped", "disconnected"}:
            # Rejected/skipped can be a successfully executed event path. Firmware
            # should emit an explicit success field when that distinction matters.
            if status in {"rejected", "skipped"}:
                return True
            return False
    return None


def is_post_warmup(row: Mapping[str, object]) -> bool:
    post = parse_bool(row.get("post_warmup"))
    if post is not None:
        return post
    warmup = parse_bool(row.get("warmup"))
    if warmup is not None:
        return not warmup
    return True


def normalized_row(row: Mapping[str, str]) -> dict[str, str]:
    return {normalized_name(key): value for key, value in row.items()}
