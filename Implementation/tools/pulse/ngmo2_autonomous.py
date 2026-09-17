"""Proof-joined analysis for autonomous, markerless R&S NGMO2 captures.

The power capture and proof export are deliberately separate operations.  This
module accepts only captures whose later NVS records prove a structurally
coherent attempted event at the declared path and zero-based firmware sequence.
Successful and failed outcomes remain separate data strata.
NGMO2 dynamic samples are interval averages, so charge is a rectangular sum.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Mapping

from common import (
    PULSE_BUILD_CONTEXT_FIELDS,
    parse_bool,
    parse_float,
    read_csv_rows,
    sha256_file,
    write_csv_rows,
)
from serial_parser import _build_context_sha256


AUTONOMOUS_CAPTURE_PROFILE = "ngmo2_autonomous_one_event_per_power_cycle"
AUTONOMOUS_JOIN_STATUS = "ngmo2_autonomous_exact_proof_sequence"
MEASUREMENT_SCOPE = "ngmo2_capture_envelope"
PROOF_SCHEMA_VERSION = 2
UNILATERAL_PATHS = {"local", "no_contact"}
ACCEPTED_PATHS = {"accepted_connected", "accepted_discovery"}
VALID_PATHS = UNILATERAL_PATHS | ACCEPTED_PATHS
HEX32_RE = re.compile(r"[0-9a-fA-F]{8}")
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
BUILD_CONTEXT_DERIVED_FIELDS = {
    "source_file",
    "source_line",
    "record_index",
    "analysis_input_file",
    "analysis_input_sha256",
}


def _integer(row: Mapping[str, object], field: str) -> int | None:
    value = parse_float(row.get(field))
    if value is None or value != int(value):
        return None
    return int(value)


def _firmware_sequence(row: Mapping[str, object]) -> int | None:
    value = _integer(row, "firmware_sequence")
    if value is None:
        value = _integer(row, "sequence")
    return value


def _proof_key(row: Mapping[str, object]) -> tuple[str, str, int | None]:
    return (
        str(row.get("board_id", "")).strip(),
        str(row.get("path", "")).strip(),
        _firmware_sequence(row),
    )


def _connection_state(path: str) -> str:
    return "connected" if path in {"local", "accepted_connected"} else "disconnected"


def _input_digest(hashes: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def pulse_hash32(value: str) -> str:
    """Return the firmware's 32-bit FNV-1a fingerprint as eight lowercase hex digits."""

    result = 2166136261
    for byte in value.encode("utf-8"):
        result ^= byte
        result = (result * 16777619) & 0xFFFFFFFF
    return f"{result:08x}"


def _load_archived_build_context(
    build_context_csv: Path,
    expected_board_ids: set[str],
) -> tuple[dict[str, str], str]:
    """Load one immutable release contract from parsed PULSE_META rows."""

    rows = read_csv_rows(build_context_csv)
    release_rows = [
        row for row in rows if str(row.get("build_variant", "")) == "pulse_release"
    ]
    if not release_rows:
        raise ValueError(
            "archived build-context CSV has no build_variant=pulse_release PULSE_META row"
        )
    issues: list[str] = []
    observed_boards = {
        str(row.get("board_id", "")).strip() for row in release_rows
    }
    missing_boards = sorted(expected_board_ids - observed_boards)
    if missing_boards:
        issues.append("missing_board_context=" + ",".join(missing_boards))
    identities: set[tuple[str, ...]] = set()
    for position, row in enumerate(release_rows, 1):
        missing = [
            field
            for field in PULSE_BUILD_CONTEXT_FIELDS
            if not str(row.get(field, "")).strip()
        ]
        if missing:
            issues.append(f"row_{position}_missing=" + ",".join(missing))
            continue
        declared_hash = str(row.get("build_context_sha256", "")).strip().lower()
        unhashed_row = {
            key: value
            for key, value in row.items()
            if key not in BUILD_CONTEXT_DERIVED_FIELDS
        }
        calculated_hash = _build_context_sha256(unhashed_row)
        if not SHA256_RE.fullmatch(declared_hash) or declared_hash != calculated_hash:
            issues.append(f"row_{position}_build_context_sha256_mismatch")
        identities.add(
            tuple(str(row.get(field, "")) for field in PULSE_BUILD_CONTEXT_FIELDS)
        )
    if len(identities) != 1:
        issues.append(f"mixed_release_build_contexts={len(identities)}")
    if issues:
        raise ValueError("invalid archived build context: " + " | ".join(issues))
    context = {
        field: str(release_rows[0].get(field, ""))
        for field in PULSE_BUILD_CONTEXT_FIELDS
    }
    return context, sha256_file(build_context_csv)


def _proof_issues(
    proof: Mapping[str, object],
    *,
    board_id: str,
    path: str,
    firmware_sequence: int,
    expected_role: str,
    passive: bool,
) -> list[str]:
    issues: list[str] = []
    label = "responder" if passive else expected_role
    if str(proof.get("board_id", "")) != board_id:
        issues.append(f"{label}_board_id_mismatch")
    if str(proof.get("path", "")) != path:
        issues.append(f"{label}_path_mismatch")
    if _firmware_sequence(proof) != firmware_sequence:
        issues.append(f"{label}_firmware_sequence_mismatch")
    if _integer(proof, "schema") != PROOF_SCHEMA_VERSION:
        issues.append(f"{label}_proof_schema_not_{PROOF_SCHEMA_VERSION}")
    success = parse_bool(proof.get("success"))
    record_ready = parse_bool(proof.get("record_ready"))
    role = str(proof.get("role", ""))
    if success is None:
        issues.append(f"{label}_success_missing")
    # A failure before link-role resolution is still a valid attempted event,
    # but only if no event record exists.  Once a record exists, its role must
    # be the expected path role.
    if role != expected_role and not (
        success is False and record_ready is False and role == "unresolved"
    ):
        issues.append(f"{label}_proof_role_not_{expected_role}")
    errno_values = {
        field: _integer(proof, field)
        for field in ("event_errno", "infrastructure_errno", "result_release_errno")
    }
    if any(value is None for value in errno_values.values()):
        issues.append(f"{label}_errno_missing")
    failure_reason = _integer(proof, "failure_reason")
    if success is True:
        if failure_reason != 0:
            issues.append(f"{label}_successful_failure_reason_not_none")
        for field, value in errno_values.items():
            if value != 0:
                issues.append(f"{label}_successful_{field}_not_zero")
    elif success is False:
        if failure_reason is None:
            issues.append(f"{label}_failure_reason_missing")
        if failure_reason == 0 and all(value == 0 for value in errno_values.values()):
            issues.append(f"{label}_failure_has_no_error_evidence")
    if passive:
        if success is not True:
            issues.append(f"{label}_passive_proof_not_successful")
        if record_ready is not False:
            issues.append(f"{label}_passive_record_ready_not_zero")
    elif success is True and record_ready is not True:
        issues.append(f"{label}_successful_record_ready_not_one")
    elif success is False and record_ready is None:
        issues.append(f"{label}_failed_record_ready_missing")
    if parse_bool(proof.get("passive_expected")) is not passive:
        issues.append(
            f"{label}_passive_expected_not_{'one' if passive else 'zero'}"
        )
    if parse_bool(proof.get("warmup")) is None:
        issues.append(f"{label}_warmup_missing")
    if parse_bool(proof.get("remote_session_started")) is None:
        issues.append(f"{label}_remote_session_started_missing")
    if not HEX32_RE.fullmatch(str(proof.get("firmware_revision_hash", ""))):
        issues.append(f"{label}_firmware_revision_hash_invalid")
    if not HEX32_RE.fullmatch(str(proof.get("artifact_hash", ""))):
        issues.append(f"{label}_artifact_hash_invalid")
    duration_us = _integer(proof, "event_duration_us")
    if passive:
        if duration_us != 0:
            issues.append(f"{label}_passive_event_duration_not_zero")
    elif record_ready is True and (duration_us is None or duration_us <= 0):
        issues.append(f"{label}_recorded_event_duration_not_positive")
    elif record_ready is False and duration_us not in (0, None):
        issues.append(f"{label}_unrecorded_event_duration_not_zero")
    return issues


def _guard_statistics(
    currents: list[float],
    pre_indices: list[int],
    post_indices: list[int],
    *,
    resolution_a: float,
) -> dict[str, float]:
    pre = sum(currents[index] for index in pre_indices) / len(pre_indices)
    post = sum(currents[index] for index in post_indices) / len(post_indices)
    baseline = (pre + post) / 2.0
    drift = abs(post - pre) / max(abs(baseline), resolution_a, 1.0e-12)
    return {
        "pre_guard_current_a": pre,
        "post_guard_current_a": post,
        "baseline_current_a": baseline,
        "guard_drift_fraction": drift,
    }


def analyze_autonomous_captures(
    manifest_csv: Path,
    proofs_csv: Path,
    capture_status_csv: Path,
    output_dir: Path,
    *,
    build_context_csv: Path,
    max_guard_drift_fraction: float = 0.10,
    require_complete_block: bool = True,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Join autonomous captures to recovered proofs and integrate eligible rows.

    The pre-dispatch and post-completion regions contain only complete NGMO2
    sample intervals.  Their means receive equal weight when forming the idle
    offset. Both absolute and idle-subtracted energy are rectangular integrals
    over the complete capture. Firmware duration is retained as timing evidence
    and is never substituted for the energy denominator.
    """

    if not 0 <= max_guard_drift_fraction <= 1:
        raise ValueError("max guard drift fraction must be between 0 and 1")
    manifest = read_csv_rows(manifest_csv)
    proofs = read_csv_rows(proofs_csv)
    statuses = read_csv_rows(capture_status_csv)
    if not manifest:
        raise ValueError("autonomous NGMO2 manifest contains no captures")

    expected_board_ids = {
        str(entry.get(field, "")).strip()
        for entry in manifest
        for field in ("initiator_board_id", "responder_board_id")
        if str(entry.get(field, "")).strip()
    }
    build_context, build_context_source_sha256 = _load_archived_build_context(
        build_context_csv, expected_board_ids
    )

    input_hashes = {
        "manifest": sha256_file(manifest_csv),
        "proofs": sha256_file(proofs_csv),
        "capture_status": sha256_file(capture_status_csv),
        "build_context": build_context_source_sha256,
    }
    proof_groups: dict[tuple[str, str, int | None], list[dict[str, str]]] = {}
    for proof in proofs:
        proof_groups.setdefault(_proof_key(proof), []).append(proof)
    status_groups: dict[str, list[dict[str, str]]] = {}
    for status in statuses:
        status_groups.setdefault(str(status.get("board_id", "")).strip(), []).append(status)

    incomplete_blocks: set[tuple[str, str, str]] = set()
    incomplete_proof_sets: set[tuple[str, str, str, str]] = set()
    if require_complete_block:
        block_sequences: dict[tuple[str, str, str], list[int | None]] = {}
        block_boards: dict[tuple[str, str, str], set[str]] = {}
        for entry in manifest:
            block_key = (
                str(entry.get("run_id", "")).strip(),
                str(entry.get("pair_id", "")).strip(),
                str(entry.get("event_path", "")).strip(),
            )
            block_sequences.setdefault(block_key, []).append(_firmware_sequence(entry))
            block_boards.setdefault(block_key, set()).update(
                {
                    str(entry.get("initiator_board_id", "")).strip(),
                    str(entry.get("responder_board_id", "")).strip(),
                }
            )
        expected_sequences = set(range(105))
        for block_key, sequences in block_sequences.items():
            concrete = [value for value in sequences if value is not None]
            if len(concrete) != 105 or set(concrete) != expected_sequences:
                incomplete_blocks.add(block_key)
            path = block_key[2]
            for board_id in block_boards.get(block_key, set()):
                board_sequences = [
                    _firmware_sequence(proof)
                    for proof in proofs
                    if str(proof.get("board_id", "")).strip() == board_id
                    and str(proof.get("path", "")).strip() == path
                ]
                board_concrete = [
                    value for value in board_sequences if value is not None
                ]
                if (
                    not board_id
                    or len(board_concrete) != 105
                    or set(board_concrete) != expected_sequences
                ):
                    incomplete_proof_sets.add((*block_key, board_id))

    metrics: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    joins: list[dict[str, object]] = []
    seen_manifest_keys: set[tuple[str, str, str, int]] = set()

    for entry in manifest:
        capture_id = str(entry.get("capture_id", "")).strip()
        run_id = str(entry.get("run_id", "")).strip()
        pair_id = str(entry.get("pair_id", "")).strip()
        path = str(entry.get("event_path", "")).strip()
        sequence = _firmware_sequence(entry)
        issues: list[str] = []
        if not capture_id:
            issues.append("capture_id_missing")
        if not run_id:
            issues.append("run_id_missing")
        if not pair_id:
            issues.append("pair_id_missing")
        if path not in VALID_PATHS:
            issues.append("event_path_invalid")
        if sequence is None or sequence < 0:
            issues.append("firmware_sequence_invalid")
            sequence = -1
        manifest_key = (run_id, pair_id, path, sequence)
        if (run_id, pair_id, path) in incomplete_blocks:
            issues.append("capture_block_not_exact_firmware_sequences_0_through_104")
        if manifest_key in seen_manifest_keys:
            issues.append("duplicate_manifest_path_sequence")
        seen_manifest_keys.add(manifest_key)
        if str(entry.get("capture_profile", "")) != AUTONOMOUS_CAPTURE_PROFILE:
            issues.append("capture_profile_not_autonomous")
        if parse_bool(entry.get("debug_boards_disconnected_confirmed")) is not True:
            issues.append("debug_board_disconnect_not_confirmed")
        if parse_bool(entry.get("no_parallel_battery_confirmed")) is not True:
            issues.append("no_parallel_battery_not_confirmed")

        initiator_board = str(entry.get("initiator_board_id", "")).strip()
        responder_board = str(entry.get("responder_board_id", "")).strip()
        if not initiator_board or not responder_board or initiator_board == responder_board:
            issues.append("board_identity_invalid")
        block_key = (run_id, pair_id, path)
        if (*block_key, initiator_board) in incomplete_proof_sets:
            issues.append("initiator_proof_set_not_exact_firmware_sequences_0_through_104")
        if (*block_key, responder_board) in incomplete_proof_sets:
            issues.append("responder_proof_set_not_exact_firmware_sequences_0_through_104")
        for field in ("initiator_firmware_sha256", "responder_firmware_sha256"):
            if not SHA256_RE.fullmatch(str(entry.get(field, ""))):
                issues.append(f"{field}_invalid")
        firmware_revision = str(entry.get("firmware_revision", "")).strip()
        artifact_sha256 = str(entry.get("artifact_sha256", "")).strip().lower()
        if not firmware_revision:
            issues.append("firmware_revision_missing")
        if not SHA256_RE.fullmatch(artifact_sha256):
            issues.append("artifact_sha256_invalid")
        expected_revision_hash = (
            pulse_hash32(firmware_revision) if firmware_revision else ""
        )
        expected_artifact_hash = (
            pulse_hash32(artifact_sha256) if SHA256_RE.fullmatch(artifact_sha256) else ""
        )
        if str(entry.get("expected_firmware_revision_hash", "")).lower() != expected_revision_hash:
            issues.append("manifest_firmware_revision_fingerprint_mismatch")
        if str(entry.get("expected_artifact_hash", "")).lower() != expected_artifact_hash:
            issues.append("manifest_artifact_fingerprint_mismatch")
        if firmware_revision != build_context["firmware_revision"]:
            issues.append("manifest_firmware_revision_not_archived_build_context")
        if artifact_sha256 != build_context["artifact_sha256"].lower():
            issues.append("manifest_artifact_not_archived_build_context")
        initiator_channel = _integer(entry, "initiator_channel")
        role_voltages: dict[str, float] = {}
        if initiator_channel not in (1, 2):
            issues.append("initiator_channel_invalid")
        else:
            expected_channel_roles = (
                ("initiator", "responder")
                if initiator_channel == 1
                else ("responder", "initiator")
            )
            if str(entry.get("channel_1_role", "")) != expected_channel_roles[0]:
                issues.append("channel_1_role_mapping_mismatch")
            if str(entry.get("channel_2_role", "")) != expected_channel_roles[1]:
                issues.append("channel_2_role_mapping_mismatch")
        if str(entry.get("sense_mode", "")) != "local":
            issues.append("sense_mode_not_final_local_sense_protocol")
        for field in ("channel_1_lead_length_m", "channel_2_lead_length_m"):
            value = parse_float(entry.get(field))
            if value is None or value <= 0:
                issues.append(f"{field}_invalid")
        for field in ("channel_1_lead_gauge_awg", "channel_2_lead_gauge_awg"):
            value = parse_float(entry.get(field))
            if value is None or value <= 0:
                issues.append(f"{field}_invalid")
        for field in ("channel_1_cable_id", "channel_2_cable_id"):
            if not str(entry.get(field, "")).strip():
                issues.append(f"{field}_missing")
        if not str(entry.get("calibration_id", "")).strip():
            issues.append("calibration_id_missing")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(entry.get("calibration_date", ""))):
            issues.append("calibration_date_invalid")
        if parse_bool(entry.get("calibration_current_confirmed")) is not True:
            issues.append("calibration_current_not_confirmed")
        if parse_float(entry.get("output_impedance_ohm")) != 0:
            issues.append("output_impedance_not_zero")
        if parse_float(entry.get("output_impedance_programmed_ohm")) != 0:
            issues.append("output_impedance_programming_not_verified")
        if str(entry.get("common_output_coupling", "")).upper() != "OFF":
            issues.append("common_output_coupling_not_off")
        if str(entry.get("current_limit_type", "")).upper() != "LIMIT":
            issues.append("current_limit_type_not_limit")
        if (
            parse_bool(entry.get("current_limit_maximum_setting_readback_verified"))
            is not True
        ):
            issues.append("current_limit_maximum_setting_not_verified")
        for channel in (1, 2):
            for phase in ("pre", "post"):
                if parse_bool(
                    entry.get(f"channel_{channel}_current_limit_state_{phase}")
                ) is not False:
                    issues.append(
                        f"channel_{channel}_current_limit_state_{phase}_not_off"
                    )
        if parse_bool(entry.get("maximum_voltage_setting_readback_verified")) is not True:
            issues.append("maximum_voltage_setting_not_verified")

        for board_id in (initiator_board, responder_board):
            board_statuses = status_groups.get(board_id, [])
            completes = [
                row for row in board_statuses if str(row.get("state", "")) == "export_complete"
            ]
            errors = [row for row in board_statuses if str(row.get("error", ""))]
            if len(completes) != 1:
                issues.append(
                    f"{board_id or 'unknown'}_expected_one_export_complete_found_{len(completes)}"
                )
            if errors:
                issues.append(f"{board_id or 'unknown'}_proof_export_reported_error")

        initiator_expected_role = "local" if path in UNILATERAL_PATHS else "initiator"
        matched: dict[str, dict[str, str]] = {}
        initiator_candidates = proof_groups.get((initiator_board, path, sequence), [])
        if len(initiator_candidates) != 1:
            issues.append(
                f"initiator_expected_one_proof_found_{len(initiator_candidates)}"
            )
        else:
            primary_proof = initiator_candidates[0]
            matched["initiator"] = primary_proof
            issues.extend(
                _proof_issues(
                    primary_proof,
                    board_id=initiator_board,
                    path=path,
                    firmware_sequence=sequence,
                    expected_role=initiator_expected_role,
                    passive=False,
                )
            )

        primary = matched.get("initiator", {})
        if primary:
            expected_warmup = sequence < 5
            expected_trial = sequence if expected_warmup else sequence - 5
            expected_exchange = (
                sequence + 1
                if path == "accepted_connected"
                else 105 + sequence + 1
                if path == "accepted_discovery"
                else 0
            )
            if parse_bool(primary.get("warmup")) is not expected_warmup:
                issues.append("initiator_warmup_not_sequence_0_through_4")
            if _integer(primary, "trial_id") != expected_trial:
                issues.append("initiator_trial_id_not_expected_for_sequence")
            if _integer(primary, "exchange_id") != expected_exchange:
                issues.append("initiator_exchange_id_not_expected_for_path_sequence")
        remote_started = parse_bool(primary.get("remote_session_started"))
        responder_required = True
        responder_candidates = proof_groups.get((responder_board, path, sequence), [])
        if len(responder_candidates) > 1:
            issues.append(
                f"responder_expected_at_most_one_proof_found_{len(responder_candidates)}"
            )
        elif responder_required and not responder_candidates:
            issues.append("responder_required_proof_missing")
        elif len(responder_candidates) == 1:
            responder_proof = responder_candidates[0]
            matched["responder"] = responder_proof
            issues.extend(
                _proof_issues(
                    responder_proof,
                    board_id=responder_board,
                    path=path,
                    firmware_sequence=sequence,
                    expected_role="responder",
                    passive=path in UNILATERAL_PATHS,
                )
            )

        secondary = matched.get("responder", {})
        if primary and secondary:
            for field in (
                "trial_id",
                "exchange_id",
                "warmup",
                "firmware_revision_hash",
                "artifact_hash",
            ):
                if str(primary.get(field, "")) != str(secondary.get(field, "")):
                    issues.append(f"proof_pair_{field}_mismatch")
            if path in ACCEPTED_PATHS:
                if (_integer(primary, "exchange_id") or 0) <= 0:
                    issues.append("accepted_exchange_id_not_positive")
                if remote_started is True and parse_bool(
                    secondary.get("remote_session_started")
                ) is not True:
                    issues.append("responder_remote_session_not_started")
                if parse_bool(primary.get("success")) is True and parse_bool(
                    secondary.get("success")
                ) is not True:
                    issues.append("successful_initiator_responder_not_successful")
            elif parse_bool(primary.get("remote_session_started")) is not False:
                issues.append("unilateral_remote_session_started")
        if path in ACCEPTED_PATHS and parse_bool(primary.get("success")) is True:
            if remote_started is not True:
                issues.append("successful_accepted_remote_session_not_started")
        for logical_role, proof in matched.items():
            if str(proof.get("firmware_revision_hash", "")).lower() != expected_revision_hash:
                issues.append(f"{logical_role}_firmware_revision_fingerprint_mismatch")
            if str(proof.get("artifact_hash", "")).lower() != expected_artifact_hash:
                issues.append(f"{logical_role}_artifact_fingerprint_mismatch")

        raw_value = str(entry.get("canonical_csv", "")).strip()
        raw_path = Path(raw_value) if raw_value else Path(".")
        if raw_value and not raw_path.is_absolute():
            raw_path = manifest_csv.parent / raw_path
        raw_hash = str(entry.get("canonical_csv_sha256", "")).strip().lower()
        if not raw_value or not raw_path.is_file() or not SHA256_RE.fullmatch(raw_hash):
            issues.append("canonical_csv_not_bound")
            samples: list[dict[str, str]] = []
        elif sha256_file(raw_path) != raw_hash:
            issues.append("canonical_csv_hash_mismatch")
            samples = []
        else:
            samples = read_csv_rows(raw_path)

        interval = parse_float(entry.get("sample_interval_s"))
        sample_count = _integer(entry, "sample_count")
        voltage = parse_float(entry.get("voltage_v"))
        max_voltage_deviation = parse_float(
            entry.get("max_output_voltage_deviation_v")
        )
        resolution = parse_float(entry.get("current_resolution_a")) or 1.0e-5
        current_range = parse_float(entry.get("current_range_a"))
        current_range_limit = parse_float(entry.get("current_range_limit_a"))
        full_scale_deviation = parse_float(
            entry.get("current_full_scale_deviation_a")
        )
        current_limit = parse_float(entry.get("current_limit_a"))
        current_limit_margin = parse_float(
            entry.get("current_limit_rejection_margin_a")
        )
        maximum_safe_voltage = parse_float(entry.get("maximum_safe_voltage_v"))
        overvoltage_protection = parse_float(entry.get("overvoltage_protection_v"))
        dispatch_s = parse_float(entry.get("event_dispatch_in_capture_s"))
        deadline_s = parse_float(entry.get("event_completion_deadline_in_capture_s"))
        if interval is None or interval <= 0 or sample_count is None or sample_count <= 0:
            issues.append("sample_configuration_invalid")
        elif len(samples) != sample_count:
            issues.append("sample_count_mismatch")
        if voltage is None or voltage <= 0:
            issues.append("voltage_invalid")
        if current_range != 0.5 or str(entry.get("current_range_setting", "")).lower() != "medium":
            issues.append("dynamic_current_range_not_verified_0p5_a_medium")
        if current_range_limit != 0.510:
            issues.append("dynamic_current_range_limit_not_0p510_a")
        if full_scale_deviation != 0.001:
            issues.append("current_full_scale_deviation_not_0p001_a")
        if current_limit is None or current_limit <= 0:
            issues.append("current_limit_invalid")
        if (
            current_limit_margin is None
            or current_limit_margin <= 0
            or current_limit is None
            or current_limit_margin >= current_limit
        ):
            issues.append("current_limit_rejection_margin_invalid")
        if (
            voltage is None
            or maximum_safe_voltage is None
            or overvoltage_protection is None
            or not voltage < overvoltage_protection <= maximum_safe_voltage
        ):
            issues.append("overvoltage_protection_invalid")
        if max_voltage_deviation is None or max_voltage_deviation <= 0:
            issues.append("output_voltage_deviation_limit_invalid")
        output_readbacks: dict[int, tuple[float, float]] = {}
        for channel in (1, 2):
            pre = parse_float(entry.get(f"channel_{channel}_output_voltage_pre_v"))
            post = parse_float(entry.get(f"channel_{channel}_output_voltage_post_v"))
            if pre is None or post is None:
                issues.append(f"channel_{channel}_pre_post_voltage_readback_missing")
                continue
            output_readbacks[channel] = (pre, post)
            if voltage is not None and max_voltage_deviation is not None and (
                abs(pre - voltage) > max_voltage_deviation
                or abs(post - voltage) > max_voltage_deviation
            ):
                issues.append(f"channel_{channel}_voltage_readback_out_of_tolerance")
        if initiator_channel in (1, 2) and len(output_readbacks) == 2:
            responder_channel = 2 if initiator_channel == 1 else 1
            role_voltages = {
                "initiator": sum(output_readbacks[initiator_channel]) / 2.0,
                "responder": sum(output_readbacks[responder_channel]) / 2.0,
            }
        capture_duration = (interval or 0.0) * len(samples)
        if (
            dispatch_s is None
            or deadline_s is None
            or not 0 < dispatch_s < deadline_s < capture_duration
        ):
            issues.append("capture_schedule_invalid")

        currents: dict[str, list[float]] = {"initiator": [], "responder": []}
        if samples and interval is not None:
            for expected_index, sample in enumerate(samples):
                if _integer(sample, "sample_index") != expected_index:
                    issues.append("sample_indices_not_contiguous")
                    break
                row_interval = parse_float(sample.get("sample_interval_s"))
                if row_interval is None or abs(row_interval - interval) > max(1e-12, interval * 1e-9):
                    issues.append("raw_sample_interval_mismatch")
                    break
                expected_time = (expected_index + 0.5) * interval
                sample_time = parse_float(sample.get("sample_time_s"))
                if sample_time is None or abs(sample_time - expected_time) > max(1e-12, interval * 1e-6):
                    issues.append("raw_sample_time_mismatch")
                    break
                for role in currents:
                    current = parse_float(sample.get(f"{role}_current_a"))
                    if current is None:
                        issues.append(f"{role}_current_sample_invalid")
                        break
                    if current_range_limit is not None and abs(current) >= current_range_limit:
                        issues.append(f"{role}_dynamic_range_overload")
                        break
                    if (
                        current_limit is not None
                        and current_limit_margin is not None
                        and abs(current) >= current_limit - current_limit_margin
                    ):
                        issues.append(f"{role}_at_or_near_programmed_current_limit")
                        break
                    currents[role].append(current)
                if issues and (
                    issues[-1].endswith("current_sample_invalid")
                    or issues[-1].endswith("dynamic_range_overload")
                    or issues[-1].endswith("programmed_current_limit")
                ):
                    break

        pre_indices: list[int] = []
        post_indices: list[int] = []
        event_window_indices: list[int] = []
        if interval and dispatch_s is not None and deadline_s is not None:
            epsilon = interval * 1.0e-9
            for index in range(len(samples)):
                start = index * interval
                end = start + interval
                if end <= dispatch_s + epsilon:
                    pre_indices.append(index)
                elif start >= deadline_s - epsilon:
                    post_indices.append(index)
                else:
                    event_window_indices.append(index)
        if len(pre_indices) < 2:
            issues.append("insufficient_pre_dispatch_guard_samples")
        if len(post_indices) < 2:
            issues.append("insufficient_post_completion_guard_samples")
        if not event_window_indices:
            issues.append("declared_event_window_has_no_samples")

        event_duration_s = (
            (_integer(primary, "event_duration_us") or 0) / 1.0e6 if primary else 0.0
        )
        if (
            event_duration_s > 0
            and (dispatch_s is None
            or deadline_s is None
            or event_duration_s > deadline_s - dispatch_s + (interval or 0.0))
        ):
            issues.append("firmware_event_duration_outside_declared_bound")

        role_stats: dict[str, dict[str, float]] = {}
        if not issues:
            for role, values in currents.items():
                role_voltage = role_voltages[role]
                guard = _guard_statistics(
                    values, pre_indices, post_indices, resolution_a=resolution
                )
                if guard["guard_drift_fraction"] > max_guard_drift_fraction:
                    issues.append(
                        f"{role}_guard_drift_{guard['guard_drift_fraction']:.6g}"
                    )
                    continue
                charge_capture = sum(values) * interval
                charge_incremental = (
                    charge_capture - guard["baseline_current_a"] * capture_duration
                )
                energy_incremental = charge_incremental * role_voltage
                energy_total = charge_capture * role_voltage
                role_stats[role] = {
                    **guard,
                    "charge_capture_c": charge_capture,
                    "charge_incremental_c": charge_incremental,
                    "energy_incremental_j": energy_incremental,
                    "energy_total_j": energy_total,
                    "average_power_w": energy_total / capture_duration,
                    "peak_current_a": max(values),
                    "peak_power_w": max(values) * role_voltage,
                }

        status = "pass" if not issues else "fail"
        analysis_hashes = {**input_hashes, "canonical_csv": raw_hash}
        analysis_digest = _input_digest(analysis_hashes)
        audit_row: dict[str, object] = {
            **build_context,
            "capture_id": capture_id,
            "run_id": run_id,
            "pair_id": pair_id,
            "event_path": path,
            "firmware_sequence": sequence if sequence >= 0 else "",
            "audit_type": "ngmo2_autonomous_proof_join_full_capture",
            "status": status,
            "quality_valid": int(status == "pass"),
            "join_status": AUTONOMOUS_JOIN_STATUS if status == "pass" else "proof_join_failed",
            "issue_count": len(issues),
            "issues": " | ".join(issues),
            "canonical_csv": str(raw_path) if raw_value else "",
            "canonical_csv_sha256": raw_hash,
            "manifest_source_file": str(manifest_csv),
            "manifest_source_sha256": input_hashes["manifest"],
            "proofs_source_file": str(proofs_csv),
            "proofs_source_sha256": input_hashes["proofs"],
            "capture_status_source_file": str(capture_status_csv),
            "capture_status_source_sha256": input_hashes["capture_status"],
            "analysis_source_sha256": analysis_digest,
            "build_context_source_file": str(build_context_csv),
            "build_context_source_sha256": build_context_source_sha256,
        }
        audit.append(audit_row)
        if issues:
            continue

        warmup = parse_bool(primary.get("warmup")) is True
        attempt_success = parse_bool(primary.get("success")) is True
        common: dict[str, object] = {
            **build_context,
            "capture_id": capture_id,
            "run_id": run_id,
            "pair_id": pair_id,
            "firmware_sequence": sequence,
            "event_index": "",
            "trial_id": primary.get("trial_id", ""),
            "exchange_id": primary.get("exchange_id", ""),
            "event_path": path,
            "connection_state": _connection_state(path),
            "warmup": int(warmup),
            "post_warmup": int(not warmup),
            "remote_session_started": primary.get("remote_session_started", ""),
            "success": int(attempt_success),
            "outcome": "success" if attempt_success else "failure",
            "measurement_scope": MEASUREMENT_SCOPE,
            "capture_profile": "ngmo2_autonomous_proof_joined_full_capture",
            "duration_s": capture_duration,
            "firmware_event_duration_s": event_duration_s,
            "event_duration_us": primary.get("event_duration_us", ""),
            "capture_duration_s": capture_duration,
            "baseline_source": "same_capture_pre_dispatch_and_post_completion_guards",
            "baseline_method": "equal_weight_mean_of_interval_average_guard_means",
            "energy_integration_method": "rectangular_sum_of_interval_averages",
            "energy_scope": "absolute_and_idle_subtracted_full_capture_envelope",
            "supplied_energy_boundary": "ngmo2_output_includes_external_lead_loss",
            "average_power_denominator": "absolute_capture_energy_over_capture_duration",
            "peak_scope": "maximum_over_all_shared_clock_capture_bins",
            "pre_guard_sample_count": len(pre_indices),
            "post_guard_sample_count": len(post_indices),
            "nominal_sample_rate_hz": 1.0 / interval,
            "mean_voltage_v": "",
            "voltage_basis": entry.get("voltage_basis", ""),
            "join_status": AUTONOMOUS_JOIN_STATUS,
            "quality_valid": 1,
            "firmware_revision_hash": primary.get("firmware_revision_hash", ""),
            "artifact_hash": primary.get("artifact_hash", ""),
            "firmware_image_binding": (
                "manifest_capture_provenance_only_not_embedded_in_proof"
            ),
            "analysis_source_sha256": analysis_digest,
            "build_context_source_file": str(build_context_csv),
            "build_context_source_sha256": build_context_source_sha256,
            "source_file": str(raw_path),
            "source_sha256": raw_hash,
        }
        responder_endpoint_present = bool(secondary) and parse_bool(
            secondary.get("record_ready")
        ) is True
        initiator_endpoint_present = parse_bool(primary.get("record_ready")) is True
        output_roles = (
            (
                "initiator",
                initiator_expected_role,
                initiator_board,
                initiator_endpoint_present,
            ),
            (
                "responder",
                "responder" if responder_endpoint_present else "responder_background",
                responder_board,
                responder_endpoint_present,
            ),
        )
        role_rows: list[dict[str, object]] = []
        for channel_role, output_role, board_id, endpoint_present in output_roles:
            proof = matched.get(channel_role, {})
            row = {
                **common,
                **role_stats[channel_role],
                "board_id": board_id,
                "role": output_role,
                "logical_role": output_role,
                "power_channel_role": channel_role,
                "endpoint_event_present": int(endpoint_present),
                "endpoint_event_duration_us": proof.get("event_duration_us", ""),
                "baseline_power_w": role_stats[channel_role]["baseline_current_a"]
                * role_voltages[channel_role],
                "mean_voltage_v": role_voltages[channel_role],
            }
            if channel_role == "responder" and not endpoint_present:
                row["success"] = ""
                row["outcome"] = "background"
            else:
                endpoint_success = parse_bool(proof.get("success")) is True
                row["success"] = int(endpoint_success)
                row["outcome"] = "success" if endpoint_success else "failure"
            metrics.append(row)
            role_rows.append(row)
            if proof:
                joins.append(
                    {
                    **build_context,
                    "capture_id": capture_id,
                    "run_id": run_id,
                    "pair_id": pair_id,
                    "event_path": path,
                    "firmware_sequence": sequence,
                    "board_id": board_id,
                    "power_channel_role": channel_role,
                    "proof_role": proof.get("role", ""),
                    "trial_id": proof.get("trial_id", ""),
                    "exchange_id": proof.get("exchange_id", ""),
                    "warmup": proof.get("warmup", ""),
                    "record_ready": proof.get("record_ready", ""),
                    "passive_expected": proof.get("passive_expected", ""),
                    "success": proof.get("success", ""),
                    "event_duration_us": proof.get("event_duration_us", ""),
                    "firmware_revision_hash": proof.get("firmware_revision_hash", ""),
                    "artifact_hash": proof.get("artifact_hash", ""),
                    "proof_source_file": proof.get("source_file", ""),
                    "proof_source_line": proof.get("source_line", ""),
                    "join_status": AUTONOMOUS_JOIN_STATUS,
                    "analysis_source_sha256": analysis_digest,
                    "build_context_source_file": str(build_context_csv),
                    "build_context_source_sha256": build_context_source_sha256,
                    }
                )

        pair = {
            **common,
            "board_id": "pair",
            "role": "pair",
            "logical_role": "pair",
            "power_channel_role": "pair",
            "endpoint_event_present": 1,
            "energy_total_j": sum(float(row["energy_total_j"]) for row in role_rows),
            "energy_incremental_j": sum(
                float(row["energy_incremental_j"]) for row in role_rows
            ),
            "charge_capture_c": sum(float(row["charge_capture_c"]) for row in role_rows),
            "charge_incremental_c": sum(
                float(row["charge_incremental_c"]) for row in role_rows
            ),
            "baseline_current_a": sum(
                float(row["baseline_current_a"]) for row in role_rows
            ),
            "pre_guard_current_a": sum(
                float(row["pre_guard_current_a"]) for row in role_rows
            ),
            "post_guard_current_a": sum(
                float(row["post_guard_current_a"]) for row in role_rows
            ),
            "baseline_power_w": sum(float(row["baseline_power_w"]) for row in role_rows),
            "average_power_w": sum(float(row["average_power_w"]) for row in role_rows),
            "peak_current_a": max(
                left + right
                for left, right in zip(currents["initiator"], currents["responder"])
            ),
            "peak_power_w": max(
                left * role_voltages["initiator"]
                + right * role_voltages["responder"]
                for left, right in zip(currents["initiator"], currents["responder"])
            ),
            "guard_drift_fraction": max(
                float(row["guard_drift_fraction"]) for row in role_rows
            ),
        }
        metrics.append(pair)

        pre_set = set(pre_indices)
        post_set = set(post_indices)
        for index, sample in enumerate(samples):
            region = (
                "pre_dispatch_guard"
                if index in pre_set
                else "post_completion_guard"
                if index in post_set
                else "declared_event_completion_window"
            )
            for channel_role, output_role, board_id, _ in output_roles:
                current = currents[channel_role][index]
                traces.append(
                    {
                        **common,
                        "board_id": board_id,
                        "role": output_role,
                        "power_channel_role": channel_role,
                        "sample_index": index,
                        "capture_time_s": sample.get("sample_time_s", ""),
                        "event_time_s": sample.get("sample_time_s", ""),
                        "sample_interval_s": interval,
                        "schedule_region": region,
                        "current_a": current,
                        "baseline_current_a": role_stats[channel_role][
                            "baseline_current_a"
                        ],
                        "voltage_v": role_voltages[channel_role],
                        "power_w": current * role_voltages[channel_role],
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "event_power_metrics.csv", metrics)
    write_csv_rows(output_dir / "power_trace.csv", traces)
    write_csv_rows(output_dir / "ngmo2_capture_audit.csv", audit)
    write_csv_rows(output_dir / "autonomous_capture_join.csv", joins)
    metadata = [
        {
            **row,
            "capture_mode": "ngmo2_autonomous_proof_joined_full_capture",
            "role": "pair",
            "baseline_method": "equal_weight_mean_of_interval_average_guard_means",
            "energy_integration_method": "rectangular_sum_of_interval_averages",
        }
        for row in audit
    ]
    write_csv_rows(output_dir / "power_capture_metadata.csv", metadata)
    return metrics, traces, audit, joins


def require_valid_autonomous_audit(
    audit: Iterable[Mapping[str, object]],
) -> None:
    rows = list(audit)
    if not rows:
        raise ValueError("autonomous NGMO2 audit found no captures")
    failures = [row for row in rows if str(row.get("status", "")) != "pass"]
    if failures:
        raise ValueError(
            "autonomous NGMO2 proof/energy audit failed: "
            + " | ".join(str(row.get("issues", "")) for row in failures)
        )
