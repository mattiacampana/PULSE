"""Prepare Section-V figure tables and optional publication-style plots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from common import (
    PULSE_BUILD_CONTEXT_FIELDS,
    PULSE_EVENT_CONFIGURATION_FIELDS,
    is_post_warmup,
    parse_float,
    read_csv_rows,
    row_success,
    sha256_file,
    write_csv_rows,
)
from summarize import quality_valid, valid_metric_value

RESULT_CONTEXT_FIELDS = (
    *PULSE_BUILD_CONTEXT_FIELDS,
    *PULSE_EVENT_CONFIGURATION_FIELDS,
)


LATENCY_STAGE_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "peer_selection",
            "head_serialization",
            "head_transfer",
            "head_deserialization",
            "responder_encoder",
            "responder_head_forward_backward",
            "gradient_serialization",
            "gradient_transfer",
            "gradient_deserialization",
            "initiator_encoder",
            "initiator_head_forward_backward",
            "agreement",
            "normalization",
            "mixing",
            "utility_update",
            "ble_scan",
            "ble_connect",
            "local_step_1_encoder",
            "local_step_1_head_forward_backward",
            "local_step_1_update",
            "local_step_2_encoder",
            "local_step_2_head_forward_backward",
            "local_step_2_update",
            "end_to_end",
        )
    )
}

# Keep the publication panel deterministic and tied to the stages requested by
# Section V.  The figure-ready CSV still contains every measured stage.
LATENCY_PLOT_EVENT_KEYS = {
    ("local", "local", "connected"),
    ("no_contact", "local", "disconnected"),
    ("accepted_connected", "initiator", "connected"),
    ("accepted_connected", "responder", "connected"),
    ("accepted_discovery", "initiator", "disconnected"),
    ("accepted_discovery", "responder", "connected"),
}
LATENCY_PLOT_INITIATOR_STAGES = {
    "peer_selection",
    "head_serialization",
    "head_transfer",
    "initiator_encoder",
    "initiator_head_forward_backward",
    "gradient_transfer",
    "gradient_deserialization",
    "agreement",
    "normalization",
    "mixing",
    "utility_update",
}
LATENCY_PLOT_RESPONDER_STAGES = {
    "head_deserialization",
    "responder_encoder",
    "responder_head_forward_backward",
    "gradient_serialization",
}

SECTION_V_STAGE_REQUIREMENTS = {
    ("local", "local", "connected"): {
        "local_step_1_encoder",
        "local_step_1_head_forward_backward",
        "local_step_1_update",
        "local_step_2_encoder",
        "local_step_2_head_forward_backward",
        "local_step_2_update",
    },
    ("no_contact", "local", "disconnected"): {
        "ble_scan",
        "peer_selection",
        "local_step_1_encoder",
        "local_step_1_head_forward_backward",
        "local_step_1_update",
    },
    ("accepted_connected", "initiator", "connected"): {
        "peer_selection",
        "head_serialization",
        "head_transfer",
        "initiator_encoder",
        "initiator_head_forward_backward",
        "gradient_transfer",
        "gradient_deserialization",
        "agreement",
        "normalization",
        "mixing",
        "utility_update",
    },
    ("accepted_connected", "responder", "connected"): {
        "head_transfer",
        "head_deserialization",
        "responder_encoder",
        "responder_head_forward_backward",
        "gradient_serialization",
        "gradient_transfer",
    },
    ("accepted_discovery", "initiator", "disconnected"): {
        "ble_scan",
        "ble_connect",
        "peer_selection",
        "head_serialization",
        "head_transfer",
        "initiator_encoder",
        "initiator_head_forward_backward",
        "gradient_transfer",
        "gradient_deserialization",
        "agreement",
        "normalization",
        "mixing",
        "utility_update",
    },
    ("accepted_discovery", "responder", "connected"): {
        "head_transfer",
        "head_deserialization",
        "responder_encoder",
        "responder_head_forward_backward",
        "gradient_serialization",
        "gradient_transfer",
    },
}

PAPER_POWER_KEYS = {
    ("local", "local", "connected", "event_gate"),
    ("no_contact", "local", "disconnected", "event_gate"),
    ("accepted_connected", "initiator", "connected", "pair_union"),
    ("accepted_connected", "responder", "connected", "pair_union"),
    ("accepted_connected", "pair", "connected", "pair_union"),
    ("accepted_discovery", "initiator", "disconnected", "pair_union"),
    ("accepted_discovery", "responder", "connected", "pair_union"),
    ("accepted_discovery", "pair", "disconnected", "pair_union"),
}


def _read_from_dirs(directories: Iterable[Path], filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for directory in directories:
        path = directory / filename
        if path.exists():
            source_hash = sha256_file(path)
            rows.extend(
                {
                    **row,
                    "analysis_input_file": str(path),
                    "analysis_input_sha256": source_hash,
                }
                for row in read_csv_rows(path)
            )
    return rows


def _concrete_sha256(value: object) -> str:
    text = str(value or "").strip().lower()
    return (
        text
        if len(text) == 64
        and all(character in "0123456789abcdef" for character in text)
        else ""
    )


def _concrete_firmware_revision(value: object) -> str:
    text = str(value or "").strip().lower()
    return (
        text
        if len(text) in {40, 64}
        and all(character in "0123456789abcdef" for character in text)
        else ""
    )


MISSING_CONTEXT_VALUES = {
    "",
    "unknown",
    "unset",
    "operator_required",
    "external_required",
    "replace_with_numeric_volts",
    "replace_with_numeric_hz",
    "replace_with_exact_setting_or_none",
    "replace_with_numeric_metres",
    "na",
    "n_a",
}


def _context_value(row: dict[str, object], field: str) -> str:
    return str(row.get(field, "")).strip()


def _paper_release_context_issues(
    rows: list[dict[str, object]], label: str
) -> list[str]:
    """Validate and de-duplicate the complete release measurement contract."""

    if not rows:
        return [f"build_context:{label}:missing"]
    issues: list[str] = []
    identities: set[tuple[str, ...]] = set()
    for position, row in enumerate(rows, 1):
        missing = [
            field
            for field in PULSE_BUILD_CONTEXT_FIELDS
            if _context_value(row, field).lower() in MISSING_CONTEXT_VALUES
        ]
        if missing:
            issues.append(
                f"build_context:{label}:row_{position}:missing_{','.join(missing)}"
            )
            continue
        identities.add(
            tuple(_context_value(row, field) for field in PULSE_BUILD_CONTEXT_FIELDS)
        )
        if _context_value(row, "build_variant") != "pulse_release":
            issues.append(f"build_context:{label}:row_{position}:not_pulse_release")
        if _context_value(row, "build_guard") != "final_capture_requirements_enforced":
            issues.append(f"build_context:{label}:row_{position}:not_final_capture")
        if _context_value(row, "artifact_kind") != "exported_pulse_npz":
            issues.append(f"build_context:{label}:row_{position}:not_exported_artifact")
        for field in (
            "build_context_sha256",
            "artifact_sha256",
            "fixture_hash",
            "encoder_hash",
        ):
            if not _concrete_sha256(row.get(field)):
                issues.append(
                    f"build_context:{label}:row_{position}:invalid_{field}"
                )
        if not _concrete_firmware_revision(row.get("firmware_revision")):
            issues.append(
                f"build_context:{label}:row_{position}:invalid_firmware_revision"
            )
        expected_values = {
            "tensor_precision": "float32",
            "batch_size": "16",
            "local_steps": "2",
            "class_count": "6",
            "parameter_count": "4550",
        }
        for field, expected in expected_values.items():
            if _context_value(row, field) != expected:
                issues.append(
                    f"build_context:{label}:row_{position}:{field}_not_{expected}"
                )
    if len(identities) > 1:
        issues.append(
            f"build_context:{label}:mixed_release_contracts_{len(identities)}"
        )
    return issues


def _capture_environment_issues(
    row: dict[str, object], label: str
) -> list[str]:
    """Bind externally declared lab settings to an analyzer capture."""

    issues: list[str] = []
    declared_voltage = parse_float(row.get("supply_voltage_v"))
    mean_voltage = parse_float(row.get("mean_voltage_v"))
    declared_rate = parse_float(row.get("analyzer_sample_rate_hz"))
    observed_rate = parse_float(row.get("nominal_sample_rate_hz"))
    separation = parse_float(row.get("device_separation_m"))
    analyzer_filter = _context_value(row, "analyzer_filter").lower()

    if declared_voltage is None or declared_voltage <= 0.0:
        issues.append(f"capture_environment:{label}:invalid_supply_voltage_v")
    if mean_voltage is None or mean_voltage <= 0.0:
        issues.append(f"capture_environment:{label}:invalid_mean_voltage_v")
    if (
        declared_voltage is not None
        and declared_voltage > 0.0
        and mean_voltage is not None
        and mean_voltage > 0.0
        and abs(mean_voltage - declared_voltage) / declared_voltage > 0.01
    ):
        issues.append(f"capture_environment:{label}:voltage_not_bound")
    if declared_rate is None or declared_rate <= 0.0:
        issues.append(f"capture_environment:{label}:invalid_analyzer_sample_rate_hz")
    if observed_rate is None or observed_rate <= 0.0:
        issues.append(f"capture_environment:{label}:invalid_nominal_sample_rate_hz")
    if (
        declared_rate is not None
        and declared_rate > 0.0
        and observed_rate is not None
        and observed_rate > 0.0
        and abs(observed_rate - declared_rate) / declared_rate > 0.02
    ):
        issues.append(f"capture_environment:{label}:sample_rate_not_bound")
    if analyzer_filter in MISSING_CONTEXT_VALUES:
        issues.append(f"capture_environment:{label}:missing_analyzer_filter")
    if separation is None or separation <= 0.0:
        issues.append(f"capture_environment:{label}:invalid_device_separation_m")
    return issues


def _summary_binding_issues(
    summary_rows: list[dict[str, object]],
    raw_rows: list[dict[str, object]],
    group_fields: tuple[str, ...],
    label: str,
    *,
    include_failed_metrics: bool,
) -> list[str]:
    """Verify each selected aggregate against its exact hashed raw-row group."""

    issues: list[str] = []
    for position, summary in enumerate(summary_rows, 1):
        matches: list[dict[str, object]] = []
        for raw in raw_rows:
            if not is_post_warmup(raw) or not quality_valid(raw):
                continue
            if "outcome" in group_fields:
                outcome = (
                    "success"
                    if row_success(raw) is True
                    else "failure"
                    if row_success(raw) is False
                    else "unknown"
                )
                if outcome != str(summary.get("outcome", "")):
                    continue
            if all(
                field == "outcome"
                or str(raw.get(field, "")) == str(summary.get(field, ""))
                for field in group_fields
            ):
                matches.append(raw)

        expected_sources = {
            _concrete_sha256(row.get("analysis_input_sha256")) for row in matches
        }
        expected_sources.discard("")
        recorded_sources = {
            item.strip().lower()
            for item in str(summary.get("analysis_source_sha256_set", "")).split(";")
            if item.strip()
        }
        if not matches or not expected_sources or recorded_sources != expected_sources:
            issues.append(f"summary_provenance:{label}:row_{position}:source_set_not_bound")

        metric = str(summary.get("metric", ""))
        measured_rows = [
            row
            for row in matches
            if (include_failed_metrics or row_success(row) is not False)
            and valid_metric_value(metric, parse_float(row.get(metric)))
        ]
        count = parse_float(summary.get("n"))
        if count is None or int(count) != len(measured_rows):
            issues.append(f"summary_provenance:{label}:row_{position}:sample_count_not_bound")
    return issues


def _ngmo2_build_context_binding_issues(
    data: dict[str, list[dict[str, object]]],
) -> list[str]:
    """Bind autonomous power rows to an included, freshly hashed PULSE_META archive."""

    issues: list[str] = []
    identities_by_source: dict[str, set[tuple[str, ...]]] = {}
    for row in data.get("run_metadata", []):
        if str(row.get("build_variant", "")) != "pulse_release":
            continue
        source_hash = _concrete_sha256(row.get("analysis_input_sha256"))
        if not source_hash:
            continue
        identities_by_source.setdefault(source_hash, set()).add(
            tuple(_context_value(row, field) for field in PULSE_BUILD_CONTEXT_FIELDS)
        )
    for source_hash, identities in identities_by_source.items():
        if len(identities) != 1:
            issues.append(
                "ngmo2_build_context:included_metadata_"
                f"{source_hash[:12]}_has_{len(identities)}_release_contracts"
            )

    raw_rows = [
        row
        for row in data.get("power_events", [])
        if str(row.get("measurement_scope", "")) == "ngmo2_capture_envelope"
    ]
    identity_fields = (
        "run_id",
        "pair_id",
        "capture_id",
        "event_path",
        "firmware_sequence",
    )
    audits_by_capture: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for audit in data.get("ngmo2_capture_audit", []):
        key = tuple(str(audit.get(field, "")) for field in identity_fields)
        audits_by_capture.setdefault(key, []).append(audit)
    if not raw_rows:
        issues.append("ngmo2_build_context:missing_raw_power_rows")
        return issues
    for position, row in enumerate(raw_rows, 1):
        if str(row.get("capture_profile", "")) != (
            "ngmo2_autonomous_proof_joined_full_capture"
        ):
            issues.append(
                f"ngmo2_build_context:power_row_{position}:not_autonomous_profile"
            )
        if str(row.get("join_status", "")) != (
            "ngmo2_autonomous_exact_proof_sequence"
        ):
            issues.append(
                f"ngmo2_build_context:power_row_{position}:not_exact_proof_join"
            )
        source_hash = _concrete_sha256(row.get("build_context_source_sha256"))
        row_identity = tuple(
            _context_value(row, field) for field in PULSE_BUILD_CONTEXT_FIELDS
        )
        if not source_hash or source_hash not in identities_by_source:
            issues.append(
                f"ngmo2_build_context:power_row_{position}:source_not_included"
            )
        elif row_identity not in identities_by_source[source_hash]:
            issues.append(
                f"ngmo2_build_context:power_row_{position}:release_contract_mismatch"
            )
        capture_key = tuple(str(row.get(field, "")) for field in identity_fields)
        matching_audits = audits_by_capture.get(capture_key, [])
        if len(matching_audits) != 1:
            issues.append(
                f"ngmo2_build_context:power_row_{position}:"
                f"expected_one_capture_audit_found_{len(matching_audits)}"
            )
            continue
        audit = matching_audits[0]
        if (
            str(audit.get("status", "")) != "pass"
            or str(audit.get("audit_type", ""))
            != "ngmo2_autonomous_proof_join_full_capture"
            or str(audit.get("analysis_source_sha256", ""))
            != str(row.get("analysis_source_sha256", ""))
            or str(audit.get("build_context_source_sha256", "")) != source_hash
            or tuple(
                _context_value(audit, field) for field in PULSE_BUILD_CONTEXT_FIELDS
            )
            != row_identity
        ):
            issues.append(
                f"ngmo2_build_context:power_row_{position}:capture_audit_mismatch"
            )
    return issues


def _duration_to_ms(row: dict[str, str]) -> tuple[float | None, float]:
    if row.get("metric", "") not in {"duration_us", "wall_duration_us"}:
        return None, 1.0
    unit = row.get("unit", "")
    if unit == "us" or row.get("metric", "").endswith("_us"):
        return parse_float(row.get("median")), 1e-3
    if unit == "ms" or row.get("metric", "").endswith("_ms"):
        return parse_float(row.get("median")), 1.0
    if unit == "s" or row.get("metric", "").endswith("_s"):
        return parse_float(row.get("median")), 1e3
    return None, 1.0


def _layout_value(metadata: list[dict[str, str]], field: str) -> int | None:
    candidates = []
    for row in metadata:
        variant = str(row.get("build_variant", ""))
        if variant and variant != "pulse_release":
            continue
        value = parse_float(row.get(field))
        if value is not None and value >= 0 and float(value).is_integer():
            candidates.append(int(value))
    unique = set(candidates)
    if len(unique) > 1:
        raise ValueError(f"conflicting PULSE metadata values for {field}: {sorted(unique)}")
    return next(iter(unique)) if unique else None


def refine_pulse_ram_breakdown(
    rows: list[dict[str, str]], metadata: list[dict[str, str]]
) -> list[dict[str, object]]:
    """Split opaque linker objects using layout sizes emitted by that exact image.

    Zephyr's map gives authoritative totals, but C structure members share one
    input section.  This refinement only moves bytes between categories; it
    never changes the linker total and is skipped unless every required size is
    available and fits its conservative source category.
    """

    values = {
        field: _layout_value(metadata, field)
        for field in (
            "mutable_head_bytes",
            "compute_workspace_bytes",
            "gradient_buffer_bytes_each",
            "gradient_buffer_count",
            "received_head_bytes",
            "utility_table_bytes",
            "peer_entry_bytes",
            "peer_limit",
        )
    }
    required = (
        "mutable_head_bytes",
        "compute_workspace_bytes",
        "gradient_buffer_bytes_each",
        "gradient_buffer_count",
        "received_head_bytes",
        "utility_table_bytes",
    )
    if any(values[field] is None for field in required):
        return [dict(row) for row in rows]

    gradient_bytes = int(values["gradient_buffer_bytes_each"] or 0)
    gradient_count = int(values["gradient_buffer_count"] or 0)
    if gradient_count != 2:
        return [dict(row) for row in rows]

    moves = (
        ("pulse_state", "mutable_head", int(values["mutable_head_bytes"] or 0)),
        ("pulse_state", "peer_utility_table", int(values["utility_table_bytes"] or 0)),
        ("training_tensors", "compute_workspace", int(values["compute_workspace_bytes"] or 0)),
        ("training_tensors", "local_gradient", gradient_bytes),
        ("training_tensors", "remote_gradient", gradient_bytes),
        ("protocol_buffers", "tensor_wire_scratch", int(values["received_head_bytes"] or 0)),
    )
    source_totals = {
        category: sum(
            int(float(row.get("bytes", 0) or 0))
            for row in rows
            if row.get("image") == "pulse"
            and row.get("memory") == "static_ram"
            and row.get("category") == category
        )
        for category in {move[0] for move in moves}
    }
    required_totals: dict[str, int] = {}
    for source, _, amount in moves:
        required_totals[source] = required_totals.get(source, 0) + amount
    if any(source_totals.get(source, 0) < amount for source, amount in required_totals.items()):
        return [dict(row) for row in rows]

    refined: list[dict[str, object]] = [dict(row) for row in rows]
    metadata_source = "firmware_layout_metadata"
    for source, target, amount in moves:
        remaining = amount
        for row in refined:
            if (
                remaining > 0
                and row.get("image") == "pulse"
                and row.get("memory") == "static_ram"
                and row.get("category") == source
            ):
                available = int(float(row.get("bytes", 0) or 0))
                moved = min(available, remaining)
                row["bytes"] = available - moved
                remaining -= moved
        output: dict[str, object] = {
            "image": "pulse",
            "memory": "static_ram",
            "category": target,
            "bytes": amount,
            "source": metadata_source,
            "source_file": "run_metadata.csv",
        }
        if target == "peer_utility_table":
            output["peer_entry_bytes"] = values["peer_entry_bytes"]
            output["peer_limit"] = values["peer_limit"]
        if target == "tensor_wire_scratch":
            output["storage_aliasing"] = "head_tx_or_head_rx_or_gradient_wire"
        refined.append(output)
    return [row for row in refined if int(float(row.get("bytes", 0) or 0)) > 0]


def prepare_figure_csvs(
    data_dirs: Iterable[Path],
    output_dir: Path,
    *,
    require_complete: bool = False,
    profile: str = "instrumented",
) -> dict[str, list[dict[str, object]]]:
    directories = list(data_dirs)
    power_summary = _read_from_dirs(directories, "power_summary.csv")
    baseline_power_summary = _read_from_dirs(directories, "baseline_power_summary.csv")
    power_capture_metadata = _read_from_dirs(directories, "power_capture_metadata.csv")
    power_trace = _read_from_dirs(directories, "power_trace.csv")
    power_metrics = _read_from_dirs(directories, "event_power_metrics.csv")
    power_stages = _read_from_dirs(directories, "stage_power_metrics.csv")
    resource_breakdown = _read_from_dirs(directories, "resource_breakdown.csv")
    run_metadata = _read_from_dirs(directories, "run_metadata.csv")
    resource_breakdown = refine_pulse_ram_breakdown(resource_breakdown, run_metadata)
    resource_summary = _read_from_dirs(directories, "resource_summary.csv")
    resource_provenance = _read_from_dirs(directories, "resource_provenance.csv")
    runtime_ram = _read_from_dirs(directories, "runtime_ram.csv")
    event_summary = _read_from_dirs(directories, "event_summary.csv")
    stage_summary = _read_from_dirs(directories, "stage_summary.csv")
    event_metrics = _read_from_dirs(directories, "event_metrics.csv")
    stage_metrics = _read_from_dirs(directories, "stage_metrics.csv")
    link_summary = _read_from_dirs(directories, "link_summary.csv")
    link_event_metrics = _read_from_dirs(directories, "event_link_metrics.csv")
    link_audit = _read_from_dirs(directories, "link_audit.csv")
    link_capture_metadata = _read_from_dirs(
        directories, "link_capture_metadata.csv"
    )
    protocol_audit = _read_from_dirs(directories, "protocol_audit.csv")
    correctness_rows = _read_from_dirs(directories, "correctness.csv")
    communication_summary = _read_from_dirs(directories, "communication_summary.csv")
    communication_metrics = _read_from_dirs(
        directories, "event_communication_metrics.csv"
    )
    communication_audit = _read_from_dirs(directories, "communication_audit.csv")
    ngmo2_capture_audit = _read_from_dirs(directories, "ngmo2_capture_audit.csv")

    figure_power: list[dict[str, object]] = []
    for row in power_summary:
        if row.get("metric") not in {
            "energy_total_j",
            "energy_incremental_j",
            "average_power_w",
            "peak_power_w",
            "duration_s",
        }:
            continue
        output = dict(row)
        if row.get("metric", "").endswith("_j"):
            output["display_unit"] = "mJ"
            output["display_scale"] = 1000.0
        elif row.get("metric", "").endswith("_w"):
            output["display_unit"] = "mW"
            output["display_scale"] = 1000.0
        else:
            output["display_unit"] = row.get("unit", "")
            output["display_scale"] = 1.0
        figure_power.append(output)
    for row in baseline_power_summary:
        if row.get("metric") != "baseline_power_w" or row.get("capture_mode") != "baseline_only":
            continue
        output = dict(row)
        output.update(
            {
                "event_path": "baseline",
                "display_unit": "mW",
                "display_scale": 1000.0,
            }
        )
        figure_power.append(output)

    figure_latency: list[dict[str, object]] = []
    for source, rows in (("event", event_summary), ("stage", stage_summary)):
        preferred_metric = (
            "duration_us"
            if any(row.get("metric") == "duration_us" for row in rows)
            else "wall_duration_us"
        )
        for row in rows:
            if row.get("metric") != preferred_metric:
                continue
            median, scale = _duration_to_ms(row)
            if median is None:
                continue
            output: dict[str, object] = {
                "source": source,
                "event_path": row.get("event_path", ""),
                "role": row.get("role", ""),
                "connection_state": row.get("connection_state", ""),
                "remote_session_started": row.get("remote_session_started", ""),
                "stage": row.get("stage", "end_to_end" if source == "event" else ""),
                "metric": row.get("metric", ""),
                "timing_source": "firmware_wall_counter",
                "unit": "ms",
                "n": row.get("n", ""),
                "total_count": row.get("total_count", ""),
                "failure_count": row.get("failure_count", ""),
                "failure_rate": row.get("failure_rate", ""),
            }
            for field in RESULT_CONTEXT_FIELDS:
                output[field] = row.get(field, "")
            for field in (
                "analysis_source_count",
                "analysis_source_sha256_set",
                "analysis_source_set_sha256",
            ):
                output[field] = row.get(field, "")
            for field in ("median", "q1", "q3", "iqr", "p95", "min", "max"):
                value = parse_float(row.get(field))
                output[f"{field}_ms"] = value * scale if value is not None else None
            figure_latency.append(output)

    figure_link: list[dict[str, object]] = []
    for row in link_summary:
        if row.get("metric") not in {
            "link_layer_bytes",
            "unique_link_layer_bytes",
            "retransmitted_link_layer_bytes",
            "initiator_tx_link_layer_bytes",
            "responder_tx_link_layer_bytes",
            "retransmission_count",
            "retransmission_packet_rate",
            "retransmission_byte_rate",
        }:
            continue
        output = dict(row)
        output["source_definition"] = (
            "sniffer-observed CRC-valid BLE data-channel LL-PDU bytes: 2-byte LL data "
            "header plus the octet count encoded by its Length field; excludes preamble, "
            "access address, CRC, and inter-frame spacing"
        )
        figure_link.append(output)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "figure_power.csv",
        figure_power,
        preferred=[
            "event_path",
            "role",
            "connection_state",
            "remote_session_started",
            "measurement_scope",
            "baseline_source",
            "baseline_method",
            "outcome",
            "metric",
            "unit",
            "display_unit",
            "display_scale",
            "n",
            "median",
            "q1",
            "q3",
            "iqr",
            "p95",
            "failure_rate",
        ],
    )
    write_csv_rows(
        output_dir / "figure_power_trace.csv",
        power_trace,
        preferred=[
            "run_id",
            "pair_id",
            "board_id",
            "exchange_id",
            "event_index",
            "marker_event_index",
            "trial_id",
            "role",
            "power_channel_role",
            "remote_session_started",
            "event_path",
            "connection_state",
            "capture_time_s",
            "event_time_s",
            "current_a",
            "voltage_v",
            "power_w",
            "stage_code",
            "stage",
        ],
    )
    write_csv_rows(
        output_dir / "figure_power_events.csv",
        power_metrics,
        preferred=[
            "run_id",
            "pair_id",
            "board_id",
            "exchange_id",
            "event_index",
            "marker_event_index",
            "trial_id",
            "role",
            "logical_role",
            "measurement_scope",
            "serial_event_present",
            "power_channel_role",
            "remote_session_started",
            "event_path",
            "connection_state",
            "duration_s",
            "energy_total_j",
            "energy_incremental_j",
            "average_power_w",
            "peak_power_w",
            "success",
        ],
    )
    write_csv_rows(
        output_dir / "figure_power_stages.csv",
        power_stages,
        preferred=[
            "run_id",
            "pair_id",
            "board_id",
            "exchange_id",
            "event_index",
            "marker_event_index",
            "trial_id",
            "role",
            "logical_role",
            "measurement_scope",
            "power_channel_role",
            "remote_session_started",
            "event_path",
            "connection_state",
            "stage_sequence",
            "stage_code",
            "stage",
            "start_time_s",
            "end_time_s",
            "duration_s",
            "energy_incremental_j",
        ],
    )
    write_csv_rows(
        output_dir / "figure_resources.csv",
        resource_breakdown,
        preferred=["image", "memory", "category", "bytes", "source", "source_file"],
    )
    write_csv_rows(
        output_dir / "figure_resource_totals.csv",
        resource_summary,
        preferred=[
            "image",
            "memory",
            "used_bytes",
            "capacity_bytes",
            "free_bytes",
            "utilization",
            "source_file",
            "source_sha256",
        ],
    )
    write_csv_rows(
        output_dir / "figure_resource_provenance.csv",
        resource_provenance,
        preferred=[
            "image",
            "firmware_revision",
            "build_variant",
            "artifact_sha256",
            "elf_source_file",
            "elf_sha256",
            "map_source_file",
            "map_sha256",
            "run_metadata_source_file",
            "run_metadata_sha256",
            "correctness_source_file",
            "correctness_sha256",
            "firmware_revision_embedded_in_elf",
            "artifact_sha256_embedded_in_elf",
            "binding_method",
        ],
    )
    write_csv_rows(
        output_dir / "figure_runtime_ram.csv",
        runtime_ram,
        preferred=["event_path", "role", "peak_ram_bytes", "calculation_method", "source_file"],
    )
    write_csv_rows(
        output_dir / "figure_latency.csv",
        figure_latency,
        preferred=[
            "source",
            "event_path",
            "role",
            "connection_state",
            "remote_session_started",
            "stage",
            "metric",
            "timing_source",
            "unit",
            "n",
            "total_count",
            "failure_count",
            "failure_rate",
            "median_ms",
            "q1_ms",
            "q3_ms",
            "iqr_ms",
            "p95_ms",
            "min_ms",
            "max_ms",
        ],
    )
    write_csv_rows(
        output_dir / "figure_link.csv",
        figure_link,
        preferred=[
            "event_path",
            "role",
            "connection_state",
            "remote_session_started",
            "outcome",
            "metric",
            "unit",
            "n",
            "total_count",
            "quality_excluded_count",
            "median",
            "q1",
            "q3",
            "iqr",
            "p95",
            "min",
            "max",
            "source_definition",
        ],
    )
    write_csv_rows(
        output_dir / "figure_link_events.csv",
        link_event_metrics,
        preferred=[
            "run_id",
            "pair_id",
            "capture_id",
            "exchange_id",
            "event_index",
            "event_path",
            "role",
            "connection_state",
            "remote_session_started",
            "success",
            "quality_valid",
            "link_layer_bytes",
            "canonical_capture_sha256",
            "raw_pcap_path",
            "raw_pcap_sha256",
            "canonical_event_sha256",
            "run_metadata_sha256",
        ],
    )
    write_csv_rows(
        output_dir / "figure_link_audit.csv",
        link_audit,
        preferred=[
            "capture_id",
            "run_id",
            "pair_id",
            "audit_type",
            "status",
            "issues",
            "raw_pcap_path",
            "raw_pcap_sha256",
        ],
    )
    write_csv_rows(
        output_dir / "figure_link_capture_metadata.csv",
        link_capture_metadata,
        preferred=[
            "capture_id",
            "run_id",
            "pair_id",
            "raw_pcap_path",
            "raw_pcap_sha256",
            "canonical_csv_sha256",
            "event_windows_sha256",
            "run_metadata_sha256",
            "sync_method",
            "sync_valid",
        ],
    )
    write_csv_rows(
        output_dir / "figure_communication.csv",
        communication_summary,
        preferred=[
            "event_path",
            "role",
            "direction",
            "connection_state",
            "remote_session_started",
            "measurement_scope",
            "evidence_class",
            "outcome",
            "metric",
            "unit",
            "n",
            "median",
            "q1",
            "q3",
            "p95",
        ],
    )
    write_csv_rows(
        output_dir / "figure_communication_events.csv", communication_metrics
    )
    write_csv_rows(
        output_dir / "figure_communication_audit.csv", communication_audit
    )
    write_csv_rows(
        output_dir / "figure_ngmo2_capture_audit.csv", ngmo2_capture_audit
    )
    result = {
        "power": figure_power,
        "trace": power_trace,
        "power_events": power_metrics,
        "power_stages": power_stages,
        "resources": resource_breakdown,
        "resource_totals": resource_summary,
        "resource_provenance": resource_provenance,
        "runtime_ram": runtime_ram,
        "latency": figure_latency,
        "event_metrics": event_metrics,
        "stage_metrics": stage_metrics,
        "link": figure_link,
        "link_event_metrics": link_event_metrics,
        "link_audit": link_audit,
        "link_capture_metadata": link_capture_metadata,
        "power_capture_metadata": power_capture_metadata,
        "protocol_audit": protocol_audit,
        "correctness": correctness_rows,
        "run_metadata": run_metadata,
        "communication": communication_summary,
        "communication_metrics": communication_metrics,
        "communication_audit": communication_audit,
        "ngmo2_capture_audit": ngmo2_capture_audit,
    }
    if require_complete:
        _validate_section_v_coverage(result, profile=profile)
    return result


def _validate_instrumented_coverage(
    data: dict[str, list[dict[str, object]]]
) -> None:
    """Reject a paper figure that silently omits a required measured stratum."""

    missing: list[str] = []
    protocol_audit = data.get("protocol_audit", [])
    if not protocol_audit:
        missing.append("protocol_audit:missing")
    else:
        failed_audits = [
            row
            for row in protocol_audit
            if (
                str(row.get("required_stratum", "0")) == "1"
                and row.get("stratum_check") != "pass"
            )
            or row.get("integrity_check") == "fail"
        ]
        if failed_audits:
            missing.append(
                f"protocol_audit:{len(failed_audits)}_required_or_integrity_failure(s)"
            )

    release_rows: list[dict[str, object]] = [
        dict(row)
        for row in data.get("run_metadata", [])
        if str(row.get("build_variant", "")) == "pulse_release"
    ]
    release_rows.extend(
        dict(row)
        for row in data.get("latency", [])
        if (
            str(row.get("event_path", "")),
            str(row.get("role", "")),
            str(row.get("connection_state", "")),
        )
        in LATENCY_PLOT_EVENT_KEYS
    )
    release_rows.extend(
        dict(row)
        for row in data.get("power", [])
        if (
            str(row.get("event_path", "")),
            str(row.get("role", "")),
            str(row.get("connection_state", "")),
            str(row.get("measurement_scope", "")),
        )
        in PAPER_POWER_KEYS
    )
    release_rows.extend(
        dict(row)
        for row in data.get("link", [])
        if str(row.get("event_path", "")) in {
            "accepted_connected",
            "accepted_discovery",
        }
    )
    release_rows.extend(
        dict(row)
        for row in data.get("power_capture_metadata", [])
        if str(row.get("capture_mode", "")).startswith("events_")
    )
    missing.extend(_paper_release_context_issues(release_rows, "paper_measurements"))
    latency_keys = {
        (
            str(row.get("event_path", "")),
            str(row.get("role", "")),
            str(row.get("connection_state", "")),
        )
        for row in data["latency"]
        if row.get("source") == "event" and parse_float(row.get("median_ms")) is not None
    }
    for key in sorted(LATENCY_PLOT_EVENT_KEYS - latency_keys):
        missing.append("latency:" + "/".join(key))
    for key in sorted(LATENCY_PLOT_EVENT_KEYS):
        configurations = {
            tuple(str(row.get(field, "")) for field in PULSE_EVENT_CONFIGURATION_FIELDS)
            for row in data["latency"]
            if row.get("source") == "event"
            and tuple(
                str(row.get(field, ""))
                for field in ("event_path", "role", "connection_state")
            )
            == key
            and parse_float(row.get("median_ms")) is not None
        }
        if len(configurations) != 1 or any(
            not value for configuration in configurations for value in configuration
        ):
            missing.append("latency_config:" + "/".join(key))
    paper_event_latency = [
        row
        for row in data["latency"]
        if row.get("source") == "event"
        and tuple(
            str(row.get(field, ""))
            for field in ("event_path", "role", "connection_state")
        )
        in LATENCY_PLOT_EVENT_KEYS
    ]
    missing.extend(
        _summary_binding_issues(
            paper_event_latency,
            data.get("event_metrics", []),
            (
                *RESULT_CONTEXT_FIELDS,
                "event_path",
                "role",
                "connection_state",
                "remote_session_started",
            ),
            "event_latency",
            include_failed_metrics=False,
        )
    )

    stages_by_key: dict[tuple[str, str, str], set[str]] = {}
    for row in data["latency"]:
        if row.get("source") != "stage" or parse_float(row.get("median_ms")) is None:
            continue
        key = (
            str(row.get("event_path", "")),
            str(row.get("role", "")),
            str(row.get("connection_state", "")),
        )
        stages_by_key.setdefault(key, set()).add(str(row.get("stage", "")))
    for key, required_stages in SECTION_V_STAGE_REQUIREMENTS.items():
        for stage in sorted(required_stages - stages_by_key.get(key, set())):
            missing.append("stage:" + "/".join((*key, stage)))
        event_counts = {
            int(value)
            for row in data["latency"]
            if row.get("source") == "event"
            and tuple(
                str(row.get(field, ""))
                for field in ("event_path", "role", "connection_state")
            )
            == key
            for value in [parse_float(row.get("n"))]
            if value is not None
        }
        if len(event_counts) != 1:
            missing.append("stage_coverage:" + "/".join((*key, "ambiguous_event_count")))
            continue
        expected_count = next(iter(event_counts))
        for stage in sorted(required_stages):
            stage_counts = {
                int(value)
                for row in data["latency"]
                if row.get("source") == "stage"
                and tuple(
                    str(row.get(field, ""))
                    for field in ("event_path", "role", "connection_state")
                )
                == key
                and str(row.get("stage", "")) == stage
                for value in [parse_float(row.get("n"))]
                if value is not None
            }
            if stage_counts != {expected_count}:
                missing.append("stage_coverage:" + "/".join((*key, stage)))
    paper_stage_latency = [
        row
        for row in data["latency"]
        if row.get("source") == "stage"
        and str(row.get("stage", ""))
        in SECTION_V_STAGE_REQUIREMENTS.get(
            tuple(
                str(row.get(field, ""))
                for field in ("event_path", "role", "connection_state")
            ),
            set(),
        )
    ]
    missing.extend(
        _summary_binding_issues(
            paper_stage_latency,
            data.get("stage_metrics", []),
            (
                *RESULT_CONTEXT_FIELDS,
                "event_path",
                "role",
                "connection_state",
                "remote_session_started",
                "stage",
                "stage_code",
            ),
            "stage_latency",
            include_failed_metrics=False,
        )
    )

    power_by_key: dict[tuple[str, str, str, str], set[str]] = {}
    for row in data["power"]:
        if str(row.get("outcome", "")) != "success":
            continue
        key = (
            str(row.get("event_path", "")),
            str(row.get("role", "")),
            str(row.get("connection_state", "")),
            str(row.get("measurement_scope", "")),
        )
        power_by_key.setdefault(key, set()).add(str(row.get("metric", "")))
    for key in sorted(PAPER_POWER_KEYS):
        required_metrics = {
            "energy_total_j",
            "energy_incremental_j",
            "average_power_w",
            "peak_power_w",
        }
        absent_metrics = required_metrics - power_by_key.get(key, set())
        if absent_metrics:
            missing.append(
                "power:"
                + "/".join(key)
                + ":"
                + ",".join(sorted(absent_metrics))
            )
        metric_counts = {
            metric: {
                int(value)
                for row in data["power"]
                if str(row.get("outcome", "")) == "success"
                and tuple(
                    str(row.get(field, ""))
                    for field in (
                        "event_path",
                        "role",
                        "connection_state",
                        "measurement_scope",
                    )
                )
                == key
                and str(row.get("metric", "")) == metric
                for value in [parse_float(row.get("n"))]
                if value is not None
            }
            for metric in required_metrics
        }
        if (
            any(len(counts) != 1 for counts in metric_counts.values())
            or len({next(iter(counts)) for counts in metric_counts.values() if counts}) != 1
        ):
            missing.append("power_sample_coverage:" + "/".join(key))
        configurations = {
            tuple(str(row.get(field, "")) for field in PULSE_EVENT_CONFIGURATION_FIELDS)
            for row in data["power"]
            if str(row.get("outcome", "")) == "success"
            and tuple(
                str(row.get(field, ""))
                for field in (
                    "event_path",
                    "role",
                    "connection_state",
                    "measurement_scope",
                )
            )
            == key
        }
        if len(configurations) != 1 or any(
            not value for configuration in configurations for value in configuration
        ):
            missing.append("power_config:" + "/".join(key))
    paper_power_rows = [
        row
        for row in data["power"]
        if tuple(
            str(row.get(field, ""))
            for field in (
                "event_path",
                "role",
                "connection_state",
                "measurement_scope",
            )
        )
        in PAPER_POWER_KEYS
    ]
    missing.extend(
        _summary_binding_issues(
            paper_power_rows,
            data.get("power_events", []),
            (
                *RESULT_CONTEXT_FIELDS,
                "event_path",
                "role",
                "connection_state",
                "remote_session_started",
                "measurement_scope",
                "baseline_source",
                "baseline_method",
                "outcome",
            ),
            "event_power",
            include_failed_metrics=True,
        )
    )

    capture_metadata = data.get("power_capture_metadata", [])
    baseline_captures = [
        row
        for row in capture_metadata
        if row.get("capture_mode") == "baseline_only"
        and parse_float(row.get("baseline_power_w")) is not None
    ]
    baseline_roles = {str(row.get("role", "")) for row in baseline_captures}
    for role in sorted({"initiator", "responder"} - baseline_roles):
        missing.append(f"matched_baseline:{role}")
    for position, row in enumerate(baseline_captures, 1):
        missing_fields = [
            field
            for field in PULSE_BUILD_CONTEXT_FIELDS
            if _context_value(dict(row), field).lower() in MISSING_CONTEXT_VALUES
        ]
        if missing_fields:
            missing.append(
                f"matched_baseline:row_{position}:missing_{','.join(missing_fields)}"
            )
        if str(row.get("build_variant", "")) != "matched_baseline":
            missing.append(
                f"matched_baseline:row_{position}:wrong_or_missing_build_variant"
            )
        if str(row.get("build_guard", "")) != "final_capture_requirements_enforced":
            missing.append(f"matched_baseline:row_{position}:not_final_capture")
        if not _concrete_sha256(row.get("build_context_sha256")):
            missing.append(f"matched_baseline:row_{position}:missing_build_context")
        if not _concrete_firmware_revision(row.get("firmware_revision")):
            missing.append(
                f"matched_baseline:row_{position}:missing_firmware_revision"
            )
        if str(row.get("hardware_revision", "")).strip().lower() in MISSING_CONTEXT_VALUES:
            missing.append(f"matched_baseline:row_{position}:missing_hardware_revision")
        if not _concrete_sha256(row.get("source_sha256")):
            missing.append(f"matched_baseline:row_{position}:missing_raw_power_sha256")
        missing.extend(
            _capture_environment_issues(dict(row), f"baseline_row_{position}")
        )

    event_captures = [
        row
        for row in capture_metadata
        if str(row.get("capture_mode", "")).startswith("events_")
    ]
    event_capture_roles = {str(row.get("role", "")) for row in event_captures}
    for role in sorted({"initiator", "responder"} - event_capture_roles):
        missing.append(f"event_capture_metadata:{role}")
    for event_capture in event_captures:
        role = str(event_capture.get("role", ""))
        if not _concrete_sha256(event_capture.get("source_sha256")):
            missing.append(f"event_capture_metadata:{role}:missing_raw_power_sha256")
        missing.extend(
            _capture_environment_issues(dict(event_capture), f"event_{role}")
        )
        event_value = parse_float(event_capture.get("baseline_power_w"))
        source = str(event_capture.get("baseline_source", ""))
        if source == "diagnostic_intrace_gate_low":
            missing.append(f"baseline_binding:{role}:diagnostic_intrace_not_paper_valid")
            continue
        matches = [
            row
            for row in baseline_captures
            if str(row.get("role", "")) == role
            and event_value is not None
            and parse_float(row.get("baseline_power_w")) is not None
            and abs(float(parse_float(row.get("baseline_power_w")) or 0.0) - event_value)
            <= 1e-12
        ]
        if source == "baseline_metadata":
            bound_source = str(event_capture.get("baseline_capture_source_file", ""))
            matches = [
                row for row in matches if str(row.get("source_file", "")) == bound_source
            ]
            bound_metadata_hash = _concrete_sha256(
                event_capture.get("baseline_metadata_sha256")
            )
            matching_metadata_hashes = {
                _concrete_sha256(row.get("analysis_input_sha256"))
                for row in matches
            }
            matching_metadata_hashes.discard("")
            if (
                not bound_metadata_hash
                or bound_metadata_hash not in matching_metadata_hashes
            ):
                matches = []
        if len(matches) != 1:
            missing.append(
                f"baseline_binding:{role}:expected_one_exact_baseline_capture_found_{len(matches)}"
            )

    link_keys = {
        (
            str(row.get("event_path", "")),
            str(row.get("role", "")),
            str(row.get("connection_state", "")),
        )
        for row in data["link"]
        if row.get("metric") == "link_layer_bytes"
        and str(row.get("outcome", "")) == "success"
        and str(row.get("remote_session_started", "")) in {"1", "true", "True"}
        and parse_float(row.get("median")) is not None
    }
    required_link = {
        ("accepted_connected", "pair", "connected"),
        ("accepted_discovery", "pair", "disconnected"),
    }
    for key in sorted(required_link - link_keys):
        missing.append("link:" + "/".join(key))
    for key in sorted(required_link):
        configurations = {
            tuple(str(row.get(field, "")) for field in PULSE_EVENT_CONFIGURATION_FIELDS)
            for row in data["link"]
            if row.get("metric") == "link_layer_bytes"
            and str(row.get("outcome", "")) == "success"
            and tuple(
                str(row.get(field, ""))
                for field in ("event_path", "role", "connection_state")
            )
            == key
        }
        if len(configurations) != 1 or any(
            not value for configuration in configurations for value in configuration
        ):
            missing.append("link_config:" + "/".join(key))

    link_audit = data.get("link_audit", [])
    if not link_audit:
        missing.append("link_provenance:missing_link_audit")
    elif any(str(row.get("status", "")) != "pass" for row in link_audit):
        missing.append("link_provenance:link_audit_not_all_pass")

    link_captures = data.get("link_capture_metadata", [])
    capture_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    release_metadata_hashes = {
        _concrete_sha256(row.get("analysis_input_sha256"))
        for row in data.get("run_metadata", [])
        if str(row.get("build_variant", "")) == "pulse_release"
    }
    release_metadata_hashes.discard("")
    event_window_hashes = {
        _concrete_sha256(row.get("analysis_input_sha256"))
        for row in data.get("power_events", [])
    }
    event_window_hashes.discard("")
    if not link_captures:
        missing.append("link_provenance:missing_capture_metadata")
    for position, raw_capture in enumerate(link_captures, 1):
        capture = dict(raw_capture)
        key = tuple(
            str(capture.get(field, "")).strip()
            for field in ("run_id", "pair_id", "capture_id")
        )
        if not all(key) or key in capture_by_key:
            missing.append(
                f"link_provenance:capture_row_{position}:missing_or_duplicate_identity"
            )
            continue
        capture_by_key[key] = capture
        for field in (
            "canonical_csv_sha256",
            "raw_pcap_sha256",
            "event_windows_sha256",
            "run_metadata_sha256",
        ):
            if not _concrete_sha256(capture.get(field)):
                missing.append(
                    f"link_provenance:{'/'.join(key)}:invalid_{field}"
                )
        if _concrete_sha256(capture.get("run_metadata_sha256")) not in release_metadata_hashes:
            missing.append(f"link_provenance:{'/'.join(key)}:run_metadata_not_included")
        if _concrete_sha256(capture.get("event_windows_sha256")) not in event_window_hashes:
            missing.append(f"link_provenance:{'/'.join(key)}:event_windows_not_included")
        raw_path_text = str(capture.get("raw_pcap_path", "")).strip()
        if not raw_path_text:
            missing.append(f"link_provenance:{'/'.join(key)}:missing_raw_pcap_path")
        else:
            raw_path = Path(raw_path_text)
            try:
                raw_hash = sha256_file(raw_path) if raw_path.is_file() else ""
            except OSError:
                raw_hash = ""
            if raw_hash.lower() != _concrete_sha256(capture.get("raw_pcap_sha256")):
                missing.append(f"link_provenance:{'/'.join(key)}:raw_pcap_not_bound")
        canonical_path_text = str(capture.get("canonical_source_file", "")).strip()
        canonical_path = Path(canonical_path_text) if canonical_path_text else None
        try:
            canonical_hash = (
                sha256_file(canonical_path)
                if canonical_path is not None and canonical_path.is_file()
                else ""
            )
        except OSError:
            canonical_hash = ""
        if canonical_hash.lower() != _concrete_sha256(capture.get("canonical_csv_sha256")):
            missing.append(f"link_provenance:{'/'.join(key)}:canonical_csv_not_bound")
        if str(capture.get("sync_valid", "")) != "1":
            missing.append(f"link_provenance:{'/'.join(key)}:sync_not_valid")
        capture_audit_types = {
            str(audit.get("audit_type", ""))
            for audit in link_audit
            if str(audit.get("capture_id", "")) == key[2]
            and str(audit.get("run_id", "")) == key[0]
            and str(audit.get("pair_id", "")) == key[1]
            and str(audit.get("status", "")) == "pass"
        }
        for audit_type in {
            "capture_sync",
            "capture_packet_integrity",
        } - capture_audit_types:
            missing.append(
                f"link_provenance:{'/'.join(key)}:missing_passing_{audit_type}"
            )

    raw_link_events = data.get("link_event_metrics", [])
    if not raw_link_events:
        missing.append("link_provenance:missing_event_link_metrics")
    for position, raw_event in enumerate(raw_link_events, 1):
        if str(raw_event.get("event_path", "")) not in {
            "accepted_connected",
            "accepted_discovery",
        }:
            continue
        capture_key = tuple(
            str(raw_event.get(field, "")).strip()
            for field in ("run_id", "pair_id", "capture_id")
        )
        capture = capture_by_key.get(capture_key)
        if capture is None:
            missing.append(
                f"link_provenance:event_row_{position}:capture_metadata_not_found"
            )
            continue
        bindings = {
            "canonical_capture_sha256": "canonical_csv_sha256",
            "raw_pcap_sha256": "raw_pcap_sha256",
            "canonical_event_sha256": "event_windows_sha256",
            "run_metadata_sha256": "run_metadata_sha256",
        }
        for event_field, capture_field in bindings.items():
            if (
                not _concrete_sha256(raw_event.get(event_field))
                or str(raw_event.get(event_field, "")).lower()
                != str(capture.get(capture_field, "")).lower()
            ):
                missing.append(
                    f"link_provenance:event_row_{position}:{event_field}_not_bound"
                )
        matching_event_audits = [
            audit
            for audit in link_audit
            if str(audit.get("audit_type", "")) == "event_link_quality"
            and str(audit.get("status", "")) == "pass"
            and str(audit.get("capture_id", "")) == capture_key[2]
            and str(audit.get("run_id", "")) == capture_key[0]
            and str(audit.get("pair_id", "")) == capture_key[1]
            and str(audit.get("exchange_id", ""))
            == str(raw_event.get("exchange_id", ""))
            and str(audit.get("event_path", ""))
            == str(raw_event.get("event_path", ""))
            and str(audit.get("role", "")) == str(raw_event.get("role", ""))
        ]
        if len(matching_event_audits) != 1:
            missing.append(
                f"link_provenance:event_row_{position}:expected_one_passing_event_audit_"
                f"found_{len(matching_event_audits)}"
            )

    for position, summary_row in enumerate(data.get("link", []), 1):
        if summary_row.get("metric") != "link_layer_bytes":
            continue
        outcome = str(summary_row.get("outcome", ""))
        matches = []
        for raw_event in raw_link_events:
            raw_outcome = (
                "success"
                if row_success(raw_event) is True
                else "failure"
                if row_success(raw_event) is False
                else "unknown"
            )
            if raw_outcome != outcome or not is_post_warmup(raw_event):
                continue
            comparison_fields = (
                "event_path",
                "role",
                "connection_state",
                "remote_session_started",
                *RESULT_CONTEXT_FIELDS,
            )
            if all(
                str(raw_event.get(field, ""))
                == str(summary_row.get(field, ""))
                for field in comparison_fields
            ):
                matches.append(raw_event)
        expected_sources = {
            str(row.get("analysis_input_sha256", "")).lower()
            for row in matches
            if _concrete_sha256(row.get("analysis_input_sha256"))
        }
        recorded_sources = {
            item.strip().lower()
            for item in str(
                summary_row.get("analysis_source_sha256_set", "")
            ).split(";")
            if item.strip()
        }
        if not matches or recorded_sources != expected_sources:
            missing.append(
                f"link_provenance:summary_row_{position}:event_source_set_not_bound"
            )
        sample_count = parse_float(summary_row.get("n"))
        if sample_count is None or int(sample_count) != len(matches):
            missing.append(
                f"link_provenance:summary_row_{position}:sample_count_not_bound"
            )

    resource_keys = {
        (str(row.get("image", "")), str(row.get("memory", "")))
        for row in data["resource_totals"]
        if parse_float(row.get("used_bytes")) is not None
    }
    required_resources = {
        ("pulse", "flash"),
        ("pulse", "static_ram"),
        ("baseline", "flash"),
        ("baseline", "static_ram"),
    }
    for key in sorted(required_resources - resource_keys):
        missing.append("resource:" + "/".join(key))

    provenance_by_image: dict[str, list[dict[str, object]]] = {}
    for row in data.get("resource_provenance", []):
        provenance_by_image.setdefault(str(row.get("image", "")), []).append(row)
    for image in ("pulse", "baseline", "correctness"):
        rows = provenance_by_image.get(image, [])
        if len(rows) != 1:
            missing.append(
                f"resource_provenance:{image}:expected_one_row_found_{len(rows)}"
            )
            continue
        row = rows[0]
        expected_variant = {
            "pulse": "pulse_release",
            "baseline": "matched_baseline",
            "correctness": "correctness",
        }[image]
        if str(row.get("build_variant", "")) != expected_variant:
            missing.append(
                f"resource_provenance:{image}:expected_build_variant_{expected_variant}"
            )
        if not _concrete_sha256(row.get("elf_sha256")):
            missing.append(f"resource_provenance:{image}:missing_elf_sha256")
        if image in {"pulse", "baseline"} and not _concrete_sha256(
            row.get("map_sha256")
        ):
            missing.append(f"resource_provenance:{image}:missing_map_sha256")
        if image in {"pulse", "baseline", "correctness"}:
            if not _concrete_firmware_revision(row.get("firmware_revision")):
                missing.append(
                    f"resource_provenance:{image}:missing_firmware_revision"
                )
            if not _concrete_sha256(row.get("run_metadata_sha256")):
                missing.append(
                    f"resource_provenance:{image}:missing_run_metadata_sha256"
                )
            if str(row.get("firmware_revision_embedded_in_elf", "")) != "1":
                missing.append(
                    f"resource_provenance:{image}:revision_not_verified_in_elf"
                )
            if _concrete_sha256(row.get("artifact_sha256")) and str(
                row.get("artifact_sha256_embedded_in_elf", "")
            ) != "1":
                missing.append(
                    f"resource_provenance:{image}:artifact_not_verified_in_elf"
                )
        if image == "correctness" and not _concrete_sha256(
            row.get("correctness_sha256")
        ):
            missing.append(
                "resource_provenance:correctness:missing_correctness_sha256"
            )

    for image in ("pulse", "baseline"):
        rows = provenance_by_image.get(image, [])
        if len(rows) != 1:
            continue
        expected_map = _concrete_sha256(rows[0].get("map_sha256"))
        total_hashes = {
            _concrete_sha256(row.get("source_sha256"))
            for row in data["resource_totals"]
            if str(row.get("image", "")) == image
        }
        total_hashes.discard("")
        if expected_map and total_hashes != {expected_map}:
            missing.append(
                f"resource_provenance:{image}:resource_rows_not_bound_to_map_hash"
            )

    valid_runtime_methods = {
        "reported_peak_ram_bytes",
        "capacity_minus_minimum_free_ram",
        "static_plus_explicit_nonoverlapping_dynamic",
    }
    runtime_keys = {
        (str(row.get("event_path", "")), str(row.get("role", "")))
        for row in data["runtime_ram"]
        if parse_float(row.get("peak_ram_bytes")) is not None
        and float(parse_float(row.get("peak_ram_bytes")) or 0.0) > 0.0
        and row.get("calculation_method") in valid_runtime_methods
    }
    required_runtime = {
        ("local", "local"),
        ("no_contact", "local"),
        ("accepted_connected", "initiator"),
        ("accepted_connected", "responder"),
        ("accepted_discovery", "initiator"),
        ("accepted_discovery", "responder"),
    }
    for key in sorted(required_runtime - runtime_keys):
        missing.append("validated_runtime_ram:" + "/".join(key))

    release_artifacts = {
        (
            _concrete_sha256(row.get("artifact_sha256")),
            _concrete_sha256(row.get("fixture_hash")),
            _concrete_firmware_revision(row.get("firmware_revision")),
        )
        for row in data.get("run_metadata", [])
        if row.get("build_variant") == "pulse_release"
        and row.get("build_guard") == "final_capture_requirements_enforced"
        and row.get("artifact_kind") == "exported_pulse_npz"
        and _concrete_sha256(row.get("build_context_sha256"))
    }
    release_artifacts = {
        item for item in release_artifacts if all(item)
    }
    pulse_provenance = (
        provenance_by_image.get("pulse", [None])[0]
        if len(provenance_by_image.get("pulse", [])) == 1
        else None
    )
    baseline_provenance = (
        provenance_by_image.get("baseline", [None])[0]
        if len(provenance_by_image.get("baseline", [])) == 1
        else None
    )
    correctness_provenance = (
        provenance_by_image.get("correctness", [None])[0]
        if len(provenance_by_image.get("correctness", [])) == 1
        else None
    )
    pulse_revision = (
        _concrete_firmware_revision(pulse_provenance.get("firmware_revision"))
        if pulse_provenance
        else ""
    )
    baseline_revision = (
        _concrete_firmware_revision(
            baseline_provenance.get("firmware_revision")
        )
        if baseline_provenance
        else ""
    )
    correctness_revision = (
        _concrete_firmware_revision(
            correctness_provenance.get("firmware_revision")
        )
        if correctness_provenance
        else ""
    )
    pulse_metadata_hash = (
        _concrete_sha256(pulse_provenance.get("run_metadata_sha256"))
        if pulse_provenance
        else ""
    )
    baseline_metadata_hash = (
        _concrete_sha256(baseline_provenance.get("run_metadata_sha256"))
        if baseline_provenance
        else ""
    )
    correctness_metadata_hash = (
        _concrete_sha256(correctness_provenance.get("run_metadata_sha256"))
        if correctness_provenance
        else ""
    )
    release_metadata_inputs = {
        _concrete_sha256(row.get("analysis_input_sha256"))
        for row in data.get("run_metadata", [])
        if row.get("build_variant") == "pulse_release"
    }
    baseline_metadata_inputs = {
        _concrete_sha256(row.get("analysis_input_sha256"))
        for row in data.get("run_metadata", [])
        if row.get("build_variant") == "matched_baseline"
    }
    correctness_metadata_inputs = {
        _concrete_sha256(row.get("analysis_input_sha256"))
        for row in data.get("run_metadata", [])
        if row.get("build_variant") == "correctness"
    }
    if pulse_metadata_hash and pulse_metadata_hash not in release_metadata_inputs:
        missing.append("resource_provenance:pulse:metadata_hash_not_in_figure_inputs")
    if (
        baseline_metadata_hash
        and baseline_metadata_hash not in baseline_metadata_inputs
    ):
        missing.append(
            "resource_provenance:baseline:metadata_hash_not_in_figure_inputs"
        )
    if (
        correctness_metadata_hash
        and correctness_metadata_hash not in correctness_metadata_inputs
    ):
        missing.append(
            "resource_provenance:correctness:metadata_hash_not_in_figure_inputs"
        )
    valid_correctness = []
    for row in data.get("correctness", []):
        artifact_hash = _concrete_sha256(row.get("artifact_sha256"))
        fixture_hash = _concrete_sha256(row.get("fixture_hash"))
        fixture_sha256 = _concrete_sha256(row.get("fixture_sha256"))
        firmware_revision = _concrete_firmware_revision(
            row.get("firmware_revision")
        )
        correctness_input_hash = _concrete_sha256(
            row.get("analysis_input_sha256")
        )
        expected_correctness_hash = (
            _concrete_sha256(correctness_provenance.get("correctness_sha256"))
            if correctness_provenance
            else ""
        )
        if (
            row_success(row) is True
            and row.get("reference") == "pytorch_cpu_float32"
            and row.get("build_variant") == "correctness"
            and row.get("build_guard") == "final_capture_requirements_enforced"
            and _concrete_sha256(row.get("build_context_sha256"))
            and row.get("artifact_kind") == "exported_pulse_npz"
            and artifact_hash
            and fixture_hash
            and fixture_sha256 == fixture_hash
            and firmware_revision
            == pulse_revision
            == baseline_revision
            == correctness_revision
            and (artifact_hash, fixture_hash, firmware_revision) in release_artifacts
            and correctness_input_hash
            and correctness_input_hash == expected_correctness_hash
        ):
            valid_correctness.append(row)
    if not valid_correctness:
        missing.append(
            "correctness:no_pass_1_pytorch_row_with_same_release_artifact_and_fixture_hashes"
        )

    if _representative_event(data) is None:
        missing.append("power_trace:complete_post_warmup_pair_union_event")
    if missing:
        raise ValueError(
            "Section-V figure coverage is incomplete; refusing a partial paper figure: "
            + "; ".join(missing)
        )


def _validate_board_or_ngmo2_coverage(
    data: dict[str, list[dict[str, object]]], *, require_ngmo2: bool
) -> None:
    """Validate the no-sniffer evidence profile without inventing LL traffic."""

    missing: list[str] = []
    protocol_audit = data.get("protocol_audit", [])
    if not protocol_audit:
        missing.append("protocol_audit:missing")
    elif any(
        (
            str(row.get("required_stratum", "0")) == "1"
            and str(row.get("stratum_check", "")) != "pass"
        )
        or str(row.get("integrity_check", "")) == "fail"
        for row in protocol_audit
    ):
        missing.append("protocol_audit:required_or_integrity_failure")

    release_rows = [
        dict(row)
        for row in data.get("run_metadata", [])
        if str(row.get("build_variant", "")) == "pulse_release"
    ]
    release_rows.extend(
        dict(row)
        for row in data.get("latency", [])
        if tuple(
            str(row.get(field, ""))
            for field in ("event_path", "role", "connection_state")
        )
        in LATENCY_PLOT_EVENT_KEYS
    )
    release_rows.extend(dict(row) for row in data.get("communication", []))
    if require_ngmo2:
        release_rows.extend(dict(row) for row in data.get("power", []))
    missing.extend(_paper_release_context_issues(release_rows, "paper_measurements"))

    latency_keys = {
        tuple(
            str(row.get(field, ""))
            for field in ("event_path", "role", "connection_state")
        )
        for row in data.get("latency", [])
        if str(row.get("source", "")) == "event"
        and parse_float(row.get("median_ms")) is not None
    }
    for key in sorted(LATENCY_PLOT_EVENT_KEYS - latency_keys):
        missing.append("latency:" + "/".join(key))
    stages_by_key: dict[tuple[str, str, str], set[str]] = {}
    for row in data.get("latency", []):
        if str(row.get("source", "")) != "stage" or parse_float(
            row.get("median_ms")
        ) is None:
            continue
        key = tuple(
            str(row.get(field, ""))
            for field in ("event_path", "role", "connection_state")
        )
        stages_by_key.setdefault(key, set()).add(str(row.get("stage", "")))
    for key, required_stages in SECTION_V_STAGE_REQUIREMENTS.items():
        for stage in sorted(required_stages - stages_by_key.get(key, set())):
            missing.append("stage:" + "/".join((*key, stage)))

    communication_audit = data.get("communication_audit", [])
    if not communication_audit:
        missing.append("communication_audit:missing")
    elif any(str(row.get("status", "")) != "pass" for row in communication_audit):
        missing.append("communication_audit:not_all_pass")
    communication_paths = {
        str(row.get("event_path", ""))
        for row in data.get("communication_metrics", [])
        if str(row.get("role", "")) == "pair"
        and row_success(row) is True
        and is_post_warmup(row)
        and str(row.get("measurement_scope", "")) == "firmware_att_value_boundary"
        and str(row.get("evidence_class", "")) == "firmware_counter_not_over_air"
        and parse_float(row.get("att_value_bytes")) is not None
    }
    for path in sorted({"accepted_connected", "accepted_discovery"} - communication_paths):
        missing.append(f"communication:{path}/pair/success")

    resource_keys = {
        (str(row.get("image", "")), str(row.get("memory", "")))
        for row in data.get("resource_totals", [])
        if parse_float(row.get("used_bytes")) is not None
    }
    required_resources = {
        ("pulse", "flash"),
        ("pulse", "static_ram"),
        ("baseline", "flash"),
        ("baseline", "static_ram"),
    }
    for key in sorted(required_resources - resource_keys):
        missing.append("resource:" + "/".join(key))
    provenance_by_image: dict[str, list[dict[str, object]]] = {}
    for row in data.get("resource_provenance", []):
        provenance_by_image.setdefault(str(row.get("image", "")), []).append(row)
    for image in ("pulse", "baseline", "correctness"):
        rows = provenance_by_image.get(image, [])
        if len(rows) != 1:
            missing.append(
                f"resource_provenance:{image}:expected_one_row_found_{len(rows)}"
            )
            continue
        row = rows[0]
        if not _concrete_sha256(row.get("elf_sha256")):
            missing.append(f"resource_provenance:{image}:missing_elf_sha256")
        if image in {"pulse", "baseline"} and not _concrete_sha256(
            row.get("map_sha256")
        ):
            missing.append(f"resource_provenance:{image}:missing_map_sha256")
        if not _concrete_firmware_revision(row.get("firmware_revision")):
            missing.append(f"resource_provenance:{image}:missing_firmware_revision")

    release_artifacts = {
        (
            _concrete_sha256(row.get("artifact_sha256")),
            _concrete_sha256(row.get("fixture_hash")),
            _concrete_firmware_revision(row.get("firmware_revision")),
        )
        for row in data.get("run_metadata", [])
        if str(row.get("build_variant", "")) == "pulse_release"
    }
    valid_correctness = [
        row
        for row in data.get("correctness", [])
        if row_success(row) is True
        and (
            _concrete_sha256(row.get("artifact_sha256")),
            _concrete_sha256(row.get("fixture_hash")),
            _concrete_firmware_revision(row.get("firmware_revision")),
        )
        in release_artifacts
        and str(row.get("reference", "")) == "pytorch_cpu_float32"
    ]
    if not valid_correctness:
        missing.append("correctness:no_bound_pytorch_pass")

    if require_ngmo2:
        missing.extend(_ngmo2_build_context_binding_issues(data))
        ngmo2_power_summary = [
            row
            for row in data.get("power", [])
            if str(row.get("measurement_scope", "")) == "ngmo2_capture_envelope"
        ]
        ngmo2_power_events = [
            row
            for row in data.get("power_events", [])
            if str(row.get("measurement_scope", "")) == "ngmo2_capture_envelope"
        ]
        missing.extend(
            _summary_binding_issues(
                ngmo2_power_summary,
                ngmo2_power_events,
                (
                    *RESULT_CONTEXT_FIELDS,
                    "event_path",
                    "role",
                    "connection_state",
                    "remote_session_started",
                    "measurement_scope",
                    "baseline_source",
                    "baseline_method",
                    "outcome",
                ),
                "ngmo2_power",
                include_failed_metrics=True,
            )
        )
        ngmo_audit = data.get("ngmo2_capture_audit", [])
        if not ngmo_audit:
            missing.append("ngmo2_capture_audit:missing")
        elif any(str(row.get("status", "")) != "pass" for row in ngmo_audit):
            missing.append("ngmo2_capture_audit:not_all_pass")
        required_paths = {
            "local",
            "no_contact",
            "accepted_connected",
            "accepted_discovery",
        }
        required_metrics = {
            "energy_total_j",
            "energy_incremental_j",
            "average_power_w",
            "peak_power_w",
        }
        for path in sorted(required_paths):
            expected_role = "local" if path in {"local", "no_contact"} else "pair"
            found = {
                str(row.get("metric", ""))
                for row in data.get("power", [])
                if str(row.get("event_path", "")) == path
                and str(row.get("role", "")) == expected_role
                and str(row.get("measurement_scope", ""))
                == "ngmo2_capture_envelope"
                and str(row.get("outcome", "")) == "success"
                and parse_float(row.get("median")) is not None
            }
            if required_metrics - found:
                missing.append(
                    f"ngmo2_power:{path}:"
                    + ",".join(sorted(required_metrics - found))
                )

    if missing:
        raise ValueError(
            "Section-V figure coverage is incomplete; refusing a partial paper figure: "
            + "; ".join(missing)
        )


def _validate_section_v_coverage(
    data: dict[str, list[dict[str, object]]], *, profile: str = "instrumented"
) -> None:
    if profile == "instrumented":
        _validate_instrumented_coverage(data)
    elif profile == "ngmo2":
        _validate_board_or_ngmo2_coverage(data, require_ngmo2=True)
    elif profile == "boards-only":
        _validate_board_or_ngmo2_coverage(data, require_ngmo2=False)
    else:
        raise ValueError(
            f"unknown evidence profile {profile!r}; use ngmo2, boards-only, or instrumented"
        )


def _representative_event(
    data: dict[str, list[dict[str, object]]]
) -> dict[str, str] | None:
    pair_rows = [
        row
        for row in data["power_events"]
        if row.get("role") == "pair"
        and str(row.get("event_path", ""))
        in {"accepted_connected", "accepted_discovery"}
        and (str(row.get("event_index", "")) or str(row.get("trial_id", "")))
        and parse_float(row.get("energy_incremental_j")) is not None
        and is_post_warmup(row)
        and row_success(row) is not False
        and str(row.get("truncated_start", "0")).lower() not in {"1", "true", "yes"}
        and str(row.get("truncated_end", "0")).lower() not in {"1", "true", "yes"}
    ]
    if pair_rows:
        ordered = sorted(pair_rows, key=lambda row: float(parse_float(row.get("energy_incremental_j")) or 0.0))
        row = ordered[(len(ordered) - 1) // 2]
        identity = {
            field: str(row.get(field, ""))
            for field in (
                "run_id",
                "pair_id",
                "exchange_id",
                "event_index",
                "event_path",
            )
            if row.get(field, "") != ""
        }
        if all(field in identity for field in ("run_id", "pair_id", "exchange_id")):
            return identity
    candidates = [row for row in data["trace"] if is_post_warmup(row)]
    if candidates:
        row = sorted(
            candidates,
            key=lambda item: tuple(
                str(item.get(field, ""))
                for field in ("run_id", "pair_id", "board_id", "event_index")
            ),
        )[0]
        identity = {
            field: str(row.get(field, ""))
            for field in ("run_id", "pair_id", "board_id", "event_index", "event_path")
            if row.get(field, "") != ""
        }
        if all(field in identity for field in ("run_id", "pair_id", "board_id", "event_index")):
            return identity
    return None


def _plot_power(ax, data: dict[str, list[dict[str, object]]]) -> None:
    event_identifier = _representative_event(data)

    def selected(row: dict[str, object]) -> bool:
        return event_identifier is None or all(
            str(row.get(field, "")) == value
            for field, value in event_identifier.items()
        )

    trace = [row for row in data["trace"] if selected(row)]
    roles = [role for role in ("initiator", "responder", "local") if any(row.get("role") == role for row in trace)]
    if not trace or not roles:
        raise ValueError("power panel requires figure_power_trace.csv with a joined measured trial")
    selected_pair = next(
        (
            row
            for row in data["power_events"]
            if row.get("role") == "pair" and selected(row)
        ),
        None,
    )
    pair_start = parse_float(selected_pair.get("start_time_s")) if selected_pair else None
    origin = pair_start if pair_start is not None else min(
        float(parse_float(row.get("capture_time_s")) or 0.0) for row in trace
    )
    colors = {"initiator": "#0072B2", "responder": "#D55E00", "local": "#009E73"}
    for role in roles:
        rows = sorted(
            (row for row in trace if row.get("role") == role),
            key=lambda row: float(parse_float(row.get("capture_time_s")) or 0.0),
        )
        x = [(float(parse_float(row.get("capture_time_s")) or 0.0) - origin) * 1000.0 for row in rows]
        y = [float(parse_float(row.get("current_a")) or 0.0) * 1000.0 for row in rows]
        ax.plot(x, y, color=colors[role], linewidth=1.1, label=role.capitalize())
    stage_rows = [
        row
        for row in data["power_stages"]
        if selected(row)
    ]
    for role, line_style in (("initiator", ":"), ("responder", "--")):
        role_stages = sorted(
            (row for row in stage_rows if row.get("role") == role),
            key=lambda row: float(parse_float(row.get("start_time_s")) or 0.0),
        )
        for row in role_stages:
            boundary = parse_float(row.get("start_time_s"))
            if boundary is None:
                continue
            x_boundary = (boundary - origin) * 1000.0
            ax.axvline(x_boundary, color=colors[role], linestyle=line_style, linewidth=0.55, alpha=0.5)
            if role == "initiator":
                ax.text(
                    x_boundary,
                    0.02,
                    f"S{row.get('stage_code', '')}",
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    ha="right",
                    va="bottom",
                    color=colors[role],
                    fontsize=4.5,
                )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("(a) Synchronized current and envelope energy")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    energy_rows = [
        row
        for row in data["power"]
        if row.get("metric") == "energy_incremental_j"
        and (
            (
                str(row.get("event_path", "")),
                str(row.get("role", "")),
                str(row.get("connection_state", "")),
                str(row.get("measurement_scope", "")),
            )
            in PAPER_POWER_KEYS
            or (
                str(row.get("measurement_scope", ""))
                == "ngmo2_capture_envelope"
                and (
                    (
                        str(row.get("event_path", ""))
                        in {"local", "no_contact"}
                        and str(row.get("role", "")) == "local"
                    )
                    or (
                        str(row.get("event_path", ""))
                        in {"accepted_connected", "accepted_discovery"}
                        and str(row.get("role", "")) == "pair"
                    )
                )
            )
        )
        and parse_float(row.get("median")) is not None
    ]
    if energy_rows:
        energy_rows.sort(
            key=lambda row: (
                str(row.get("event_path", "")),
                str(row.get("role", "")),
                str(row.get("outcome", "")),
                str(row.get("remote_session_started", "")),
            )
        )
        inset = ax.inset_axes([0.59, 0.56, 0.39, 0.38])
        labels = [
            str(row.get("event_path") or "event").replace("_", " ")
            + f"\n{row.get('role')} | {row.get('connection_state')}"
            + (f" | {row.get('outcome')}" if row.get("outcome") else "")
            + (
                f" | remote={row.get('remote_session_started')}"
                if row.get("remote_session_started", "") != ""
                else ""
            )
            for row in energy_rows
        ]
        values = [float(parse_float(row.get("median")) or 0.0) * 1000.0 for row in energy_rows]
        q1 = [float(parse_float(row.get("q1")) or value / 1000.0) * 1000.0 for row, value in zip(energy_rows, values)]
        q3 = [float(parse_float(row.get("q3")) or value / 1000.0) * 1000.0 for row, value in zip(energy_rows, values)]
        p95 = [float(parse_float(row.get("p95")) or value / 1000.0) * 1000.0 for row, value in zip(energy_rows, values)]
        positions = list(range(len(values)))
        inset.bar(
            positions,
            values,
            yerr=[
                [max(0.0, value - lower) for value, lower in zip(values, q1)],
                [max(0.0, upper - value) for value, upper in zip(values, q3)],
            ],
            capsize=2,
            color="#56B4E9",
            width=0.7,
            label="median / IQR",
        )
        inset.scatter(positions, p95, marker="_", color="#D55E00", s=28, label="p95")
        inset.set_xticks(range(len(values)), labels, fontsize=5)
        inset.tick_params(axis="y", labelsize=5)
        inset.set_ylabel("Incremental event energy (mJ)", fontsize=6)
        inset.spines[["top", "right"]].set_visible(False)
        inset.legend(frameon=False, fontsize=4.5)
    annotation_lines: list[str] = []
    if selected_pair:
        selected_outcome = (
            "success"
            if row_success(selected_pair) is True
            else "failure"
            if row_success(selected_pair) is False
            else "unknown"
        )

        def aggregate_metric(metric: str) -> float | None:
            matches = [
                row
                for row in data["power"]
                if row.get("metric") == metric
                and row.get("role") == "pair"
                and row.get("measurement_scope")
                in {"pair_union", "ngmo2_capture_envelope"}
                and row.get("event_path") == selected_pair.get("event_path")
                and row.get("connection_state") == selected_pair.get("connection_state")
                and str(row.get("remote_session_started", ""))
                == str(selected_pair.get("remote_session_started", ""))
                and row.get("outcome") == selected_outcome
            ]
            return parse_float(matches[0].get("median")) if len(matches) == 1 else None

        total_energy = aggregate_metric("energy_total_j")
        incremental_energy = aggregate_metric("energy_incremental_j")
        average = aggregate_metric("average_power_w")
        peak = aggregate_metric("peak_power_w")
        if None not in (total_energy, incremental_energy, average, peak):
            annotation_lines.append(
                "aggregate pair median total/incremental: "
                f"{float(total_energy) * 1000:.2f}/{float(incremental_energy) * 1000:.2f} mJ"
            )
            annotation_lines.append(
                "aggregate pair median capture mean/max configured-bin-average: "
                f"{float(average) * 1000:.1f}/{float(peak) * 1000:.1f} mW"
            )
    if event_identifier:
        annotation_lines.append(
            "representative: "
            + ", ".join(f"{field}={value}" for field, value in event_identifier.items())
        )
    baseline_rows = [
        row
        for row in data["power"]
        if row.get("event_path") == "baseline" and parse_float(row.get("median")) is not None
    ]
    if baseline_rows:
        baseline_text = ", ".join(
            f"{row.get('role')}: {float(parse_float(row.get('median')) or 0.0) * 1000:.1f}"
            for row in baseline_rows
        )
        annotation_lines.append(f"baseline: {baseline_text} mW")
    if annotation_lines:
        ax.text(
            0.98,
            0.03,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.2,
        )


def _plot_communication(ax, data: dict[str, list[dict[str, object]]]) -> None:
    rows = [
        row
        for row in data.get("communication", [])
        if str(row.get("role", "")) == "pair"
        and str(row.get("outcome", "")) == "success"
        and str(row.get("measurement_scope", "")) == "firmware_att_value_boundary"
        and parse_float(row.get("median")) is not None
    ]
    paths = [
        path
        for path in ("accepted_connected", "accepted_discovery")
        if any(str(row.get("event_path", "")) == path for row in rows)
    ]
    if not paths:
        raise ValueError(
            "communication panel requires successful paired firmware ATT-value counters"
        )

    def value(path: str, metric: str) -> float:
        matches = [
            row
            for row in rows
            if str(row.get("event_path", "")) == path
            and str(row.get("metric", "")) == metric
        ]
        return float(parse_float(matches[0].get("median")) or 0.0) if len(matches) == 1 else 0.0

    payload = [value(path, "application_payload_bytes") for path in paths]
    framing = [value(path, "pulse_framing_bytes") for path in paths]
    positions = list(range(len(paths)))
    ax.bar(positions, payload, color="#0072B2", label="application payload")
    ax.bar(
        positions,
        framing,
        bottom=payload,
        color="#E69F00",
        label="PULSE framing",
    )
    for position, path, app_bytes, framing_bytes in zip(
        positions, paths, payload, framing
    ):
        fragments = value(path, "att_value_fragment_count")
        efficiency = value(path, "payload_efficiency")
        ax.text(
            position,
            app_bytes + framing_bytes,
            f"{int(round(fragments))} fragments\n{efficiency * 100:.1f}% payload",
            ha="center",
            va="bottom",
            fontsize=5.5,
        )
    ax.set_xticks(positions, [path.replace("_", " ") for path in paths])
    ax.set_ylabel("Sender-counted bytes")
    ax.set_title("Communication footprint")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=6)
    ax.text(
        0.01,
        0.01,
        "GATT attribute-value boundary; not over-air/link-layer traffic",
        transform=ax.transAxes,
        fontsize=5.2,
        va="bottom",
    )


def _plot_resources(
    ax,
    data: dict[str, list[dict[str, object]]],
    *,
    include_runtime: bool = True,
) -> None:
    rows = [row for row in data["resources"] if parse_float(row.get("bytes")) is not None]
    if not rows:
        raise ValueError("resource panel requires resource_breakdown.csv parsed from the exact linker map")
    keys = sorted({(str(row.get("image", "pulse")), str(row.get("memory", ""))) for row in rows})
    runtime_rows = [
        row for row in data["runtime_ram"] if parse_float(row.get("peak_ram_bytes")) is not None
    ] if include_runtime else []
    runtime_keys = [
        ("runtime", f"ram_{row.get('role') or row.get('event_path') or 'path'}_{index}")
        for index, row in enumerate(runtime_rows)
    ]
    runtime_by_key = {key: row for key, row in zip(runtime_keys, runtime_rows)}
    keys.extend(runtime_keys)
    categories = [
        "base_firmware",
        "encoder",
        "head",
        "mutable_head",
        "compute_workspace",
        "local_gradient",
        "remote_gradient",
        "tensor_wire_scratch",
        "peer_utility_table",
        "training_tensors",
        "protocol_buffers",
        "thread_stacks",
        "pulse_state",
        "linker_padding_or_unattributed",
        "runtime_peak",
    ]
    categories.extend(
        sorted(
            {
                str(row.get("category", ""))
                for row in rows
                if row.get("category") and row.get("category") not in categories
            }
        )
    )
    colors = {
        "base_firmware": "#999999",
        "encoder": "#0072B2",
        "head": "#56B4E9",
        "mutable_head": "#44AA99",
        "compute_workspace": "#117733",
        "local_gradient": "#88CCEE",
        "remote_gradient": "#CC6677",
        "tensor_wire_scratch": "#DDCC77",
        "peer_utility_table": "#AA4499",
        "training_tensors": "#009E73",
        "protocol_buffers": "#E69F00",
        "thread_stacks": "#F0E442",
        "pulse_state": "#D55E00",
        "linker_padding_or_unattributed": "#CC79A7",
        "runtime_peak": "#332288",
    }
    bottoms = [0.0] * len(keys)
    for category in categories:
        values = []
        for key in keys:
            if category == "runtime_peak" and key[0] == "runtime":
                values.append(float(parse_float(runtime_by_key[key].get("peak_ram_bytes")) or 0.0) / 1024.0)
            else:
                values.append(
                    sum(
                        float(parse_float(row.get("bytes")) or 0.0) / 1024.0
                        for row in rows
                        if (str(row.get("image", "pulse")), str(row.get("memory", ""))) == key
                        and row.get("category") == category
                    )
                )
        if not any(values):
            continue
        ax.bar(
            range(len(keys)),
            values,
            bottom=bottoms,
            label=category.replace("_", " "),
            color=colors.get(category),
        )
        bottoms = [left + value for left, value in zip(bottoms, values)]
    labels = []
    for image, memory in keys:
        if image == "runtime":
            runtime_row = runtime_by_key[(image, memory)]
            labels.append(f"peak\n{runtime_row.get('role') or runtime_row.get('event_path') or 'path'}")
        else:
            labels.append(f"{image}\n{'RAM' if memory == 'static_ram' else 'flash'}")
    ax.set_xticks(range(len(keys)), labels)
    totals = {
        (str(row.get("image", "")), str(row.get("memory", ""))): row
        for row in data["resource_totals"]
    }
    pulse_ram_capacity = parse_float(
        totals.get(("pulse", "static_ram"), {}).get("capacity_bytes")
    )
    for index, key in enumerate(keys):
        total = totals.get(key)
        used = parse_float(total.get("used_bytes")) if total else None
        capacity = parse_float(total.get("capacity_bytes")) if total else None
        if key[0] == "runtime":
            used = parse_float(runtime_by_key[key].get("peak_ram_bytes"))
            capacity = parse_float(runtime_by_key[key].get("ram_capacity_bytes")) or pulse_ram_capacity
        if used is not None and capacity is not None:
            ax.text(
                index,
                used / 1024.0,
                f"{used / 1024:.1f}/{capacity / 1024:.0f}",
                ha="center",
                va="bottom",
                fontsize=5,
            )
    ax.set_ylabel("Memory (KiB)")
    ax.set_title("(b) Computational resources")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper left")


def _plot_latency(ax, data: dict[str, list[dict[str, object]]]) -> None:
    rows = [row for row in data["latency"] if parse_float(row.get("median_ms")) is not None]
    if not rows:
        raise ValueError("latency panel requires event_summary.csv/stage_summary.csv with measured durations")
    # Preserve the exact end-to-end strata and compute/transport stages named
    # in Section V.  Avoid an arbitrary row-count truncation, which could hide
    # valid agreement, normalization, mixing, or forward/backward results.
    selected: list[dict[str, object]] = []
    for row in rows:
        source = str(row.get("source", ""))
        path = str(row.get("event_path", ""))
        role = str(row.get("role", ""))
        connection = str(row.get("connection_state", ""))
        stage = str(row.get("stage", ""))
        if source == "event":
            if (path, role, connection) in LATENCY_PLOT_EVENT_KEYS:
                selected.append(row)
        elif stage in SECTION_V_STAGE_REQUIREMENTS.get(
            (path, role, connection), set()
        ):
            selected.append(row)
    if selected:
        rows = selected
    rows.sort(
        key=lambda row: (
            0 if row.get("source") == "event" else 1,
            0 if row.get("connection_state") == "connected" else 1,
            LATENCY_STAGE_ORDER.get(str(row.get("stage", "")), 999),
            str(row.get("role", "")),
            str(row.get("event_path", "")),
        )
    )
    positions = list(range(len(rows)))
    medians = [float(parse_float(row.get("median_ms")) or 0.0) for row in rows]
    q1 = [float(parse_float(row.get("q1_ms")) or median) for row, median in zip(rows, medians)]
    q3 = [float(parse_float(row.get("q3_ms")) or median) for row, median in zip(rows, medians)]
    p95 = [float(parse_float(row.get("p95_ms")) or median) for row, median in zip(rows, medians)]
    labels = [
        str(row.get("stage") or row.get("event_path") or "event").replace("_", " ")
        + (f" ({row.get('role')})" if row.get("role") else "")
        + (
            f" [{str(row.get('event_path')).replace('_', ' ')}]"
            if row.get("source") == "stage" and row.get("event_path")
            else ""
        )
        + (f" [{row.get('connection_state')}]" if row.get("connection_state") else "")
        for row in rows
    ]
    ax.errorbar(
        medians,
        positions,
        xerr=[
            [median - lower for median, lower in zip(medians, q1)],
            [upper - median for median, upper in zip(medians, q3)],
        ],
        fmt="o",
        color="#0072B2",
        ecolor="#56B4E9",
        capsize=2,
        markersize=3,
        label="median / IQR",
    )
    ax.scatter(p95, positions, marker="|", color="#D55E00", s=45, label="p95")
    ax.set_yticks(positions, labels, fontsize=5.2)
    ax.invert_yaxis()
    ax.set_xlabel("Latency (ms)")
    if all(value > 0 for value in q1):
        ax.set_xscale("log")
    ax.set_title("(c) Execution time")
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    known_failure_rates = [
        parse_float(row.get("failure_rate"))
        for row in rows
        if row.get("source") == "event"
    ]
    known_failure_rates = [value for value in known_failure_rates if value is not None]
    if known_failure_rates:
        ax.text(
            0.99,
            0.99,
            f"event failure rate: {max(known_failure_rates) * 100:.1f}% max",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6,
        )


def render_figures(
    data: dict[str, list[dict[str, object]]],
    output_dir: Path,
    formats: Iterable[str],
    *,
    profile: str = "instrumented",
) -> list[Path]:
    _validate_section_v_coverage(data, profile=profile)
    try:
        import matplotlib

        # Rendering runs on lab/CI hosts without a desktop session. Select the
        # deterministic file backend before importing pyplot.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required only for PDF/PNG rendering. Figure-ready CSVs were written; "
            "install matplotlib or pass --formats none."
        ) from error

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if profile == "ngmo2":
        figure, axes = plt.subplots(1, 4, figsize=(13.6, 3.8), constrained_layout=True)
        _plot_power(axes[0], data)
        _plot_communication(axes[1], data)
        _plot_resources(axes[2], data, include_runtime=True)
        _plot_latency(axes[3], data)
    elif profile == "boards-only":
        figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), constrained_layout=True)
        _plot_communication(axes[0], data)
        _plot_resources(axes[1], data, include_runtime=True)
        _plot_latency(axes[2], data)
    else:
        figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), constrained_layout=True)
        _plot_power(axes[0], data)
        _plot_resources(axes[1], data)
        _plot_latency(axes[2], data)
    written: list[Path] = []
    for file_format in formats:
        normalized = file_format.strip().lower()
        if normalized in {"", "none"}:
            continue
        if normalized not in {"pdf", "png"}:
            raise ValueError(f"unsupported figure format {file_format!r}; use pdf, png, or none")
        path = output_dir / f"section_v_panels.{normalized}"
        figure.savefig(path, dpi=300 if normalized == "png" else None, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written
