"""Parse the machine-readable records emitted by the PULSE firmware."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Iterable

from common import (
    PULSE_BUILD_CONTEXT_FIELDS,
    PULSE_EVENT_CONFIGURATION_FIELDS,
    normalized_name,
    write_csv_rows,
)


PREFIX_TO_FILE = {
    "PULSE_META": "run_metadata.csv",
    "PULSE_EVENT": "event_metrics.csv",
    "PULSE_STAGE": "stage_metrics.csv",
    "PULSE_CORRECTNESS": "correctness.csv",
    "PULSE_CAPTURE_PROOF": "capture_proofs.csv",
    "PULSE_CAPTURE_STATUS": "capture_status.csv",
}

PREFIX_RE = re.compile(
    r"(?<![A-Z0-9_])(PULSE_META|PULSE_EVENT|PULSE_STAGE|PULSE_CORRECTNESS|"
    r"PULSE_CAPTURE_PROOF|PULSE_CAPTURE_STATUS)\b(.*)$"
)

# Headerless positional records are supported for bench scripts that cannot
# conveniently format key=value fields. Key=value is the canonical, evolvable
# firmware format. A PULSE_<TYPE>,header,... record overrides these defaults.
POSITIONAL_SCHEMAS = {
    "PULSE_META": ["run_id", "board_id", "role", "key", "value"],
    "PULSE_EVENT": [
        "run_id",
        "board_id",
        "role",
        "trial_id",
        "event_path",
        "connection_state",
        "success",
        "duration_us",
        "status",
        "error_code",
        "application_tx_bytes",
        "application_rx_bytes",
        "link_tx_bytes",
        "link_rx_bytes",
    ],
    "PULSE_STAGE": [
        "run_id",
        "board_id",
        "role",
        "trial_id",
        "event_path",
        "stage_code",
        "stage",
        "duration_us",
        "cycles",
        "success",
    ],
    "PULSE_CORRECTNESS": [
        "run_id",
        "board_id",
        "role",
        "trial_id",
        "gradient_cosine_similarity",
        "max_abs_parameter_error",
        "success",
    ],
    # Autonomous records are emitted canonically as key=value. These compact
    # fallbacks keep the generic parser total if a lab wrapper strips keys.
    "PULSE_CAPTURE_PROOF": [
        "board_id",
        "schema",
        "path",
        "firmware_sequence",
        "trial_id",
        "exchange_id",
        "role",
        "success",
    ],
    "PULSE_CAPTURE_STATUS": ["board_id", "state", "error", "errno"],
}

PREFERRED_FIELDS = {
    "PULSE_META": [
        "run_id",
        "pair_id",
        "board_id",
        "role",
        "local_identity_address",
        "local_identity_address_type",
        "peer_identity_address",
        "peer_identity_address_type",
        "build_guard",
        "firmware_revision",
        "hardware_revision",
        "ncs_version",
        "zephyr_version",
        "compiler",
        "compiler_version",
        "optimization",
        "cpu_clock_hz",
        "cpu_hz",
        "wall_counter_hz",
        "duration_us_semantics",
        "tensor_dtype",
        "tensor_precision",
        "batch_size",
        "local_steps",
        "input_source",
        "fixture_hash",
        "artifact_sha256",
        "key",
        "value",
    ],
    "PULSE_EVENT": [
        "run_id",
        "pair_id",
        "board_id",
        "role",
        "trial_id",
        "event_index",
        "marker_event_index",
        "event_path",
        "connection_state",
        "connection_state_start",
        "post_warmup",
        "warmup",
        "success",
        "status",
        "error_code",
        "failure_reason",
        "remote_session_started",
        "event_start_counter",
        "event_end_counter",
        "cycles",
        "cycle_duration_us",
        "wall_start_counter",
        "wall_end_counter",
        "wall_duration_us",
        "duration_us",
        "capture_envelope_duration_us",
        "application_tx_bytes",
        "application_rx_bytes",
        "pulse_framing_tx_bytes",
        "pulse_framing_rx_bytes",
        "att_value_tx_bytes",
        "att_value_rx_bytes",
        "att_value_tx_fragments",
        "att_value_rx_fragments",
        "link_tx_bytes",
        "link_rx_bytes",
    ],
    "PULSE_STAGE": [
        "run_id",
        "pair_id",
        "board_id",
        "role",
        "trial_id",
        "event_index",
        "event_path",
        "connection_state",
        "post_warmup",
        "warmup",
        "remote_session_started",
        "stage_code",
        "stage",
        "start_counter",
        "end_counter",
        "cycles",
        "cycle_duration_us",
        "wall_start_counter",
        "wall_end_counter",
        "wall_duration_us",
        "duration_us",
        "success",
    ],
    "PULSE_CORRECTNESS": [
        "run_id",
        "pair_id",
        "board_id",
        "role",
        "trial_id",
        "build_variant",
        "firmware_revision",
        "model_id",
        "model_version",
        "encoder_hash",
        "fixture_hash",
        "fixture_sha256",
        "artifact_kind",
        "artifact_source",
        "artifact_sha256",
        "gradient_cosine_similarity",
        "max_abs_parameter_error",
        "local_gradient_cosine",
        "remote_gradient_cosine",
        "updated_head_max_abs_error",
        "pass",
        "reference_precision",
        "embedded_precision",
        "success",
    ],
    "PULSE_CAPTURE_PROOF": [
        "board_id",
        "schema",
        "path",
        "firmware_sequence",
        "sequence",
        "trial_id",
        "exchange_id",
        "role",
        "warmup",
        "record_ready",
        "passive_expected",
        "success",
        "failure_reason",
        "remote_session_started",
        "event_duration_us",
        "event_errno",
        "infrastructure_errno",
        "result_release_errno",
        "initial_head_hash",
        "result_head_hash",
        "firmware_revision_hash",
        "artifact_hash",
    ],
    "PULSE_CAPTURE_STATUS": [
        "board_id",
        "state",
        "error",
        "errno",
        "path",
        "firmware_sequence",
        "sequence",
        "read_result",
    ],
}

CONTEXT_FIELDS = {
    "run_id",
    "pair_id",
    "board_id",
    "role",
    *PULSE_BUILD_CONTEXT_FIELDS,
}

FIELD_ALIASES = {
    "connection_state_start": "connection_state",
    "compiler": "compiler_version",
    "cpu_clock_hz": "cpu_hz",
    "tensor_dtype": "tensor_precision",
    "input_mode": "input_source",
    "failure_reason": "error_code",
}

BUILD_HASH_EXCLUDED_FIELDS = {
    "build_context_sha256",
    "run_id",
    "pair_id",
    "board_id",
    "role",
    "local_identity_address",
    "local_identity_address_type",
    "peer_identity_address",
    "peer_identity_address_type",
    # RSSI is an observed environmental covariate, not an image/configuration
    # identity.  It remains in the metadata and per-event records.
    "initial_rssi_dbm",
    # Negotiated values are retained as observations and copied from each
    # PULSE_EVENT. Directional PHY values can legitimately be complementary
    # between initiator and responder, so they are not part of the image hash.
    "initial_att_mtu",
    "initial_data_length",
    "initial_tx_phy",
    "initial_rx_phy",
    "initial_connection_interval_units",
}


def _build_context_sha256(record: dict[str, str]) -> str:
    """Hash the complete non-identity PULSE_META contract, including new keys."""

    contract = {
        key: str(value)
        for key, value in record.items()
        if key not in BUILD_HASH_EXCLUDED_FIELDS and str(value) != ""
    }
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _payload_tokens(payload: str) -> list[str]:
    payload = payload.lstrip(" \t,:;")
    if not payload:
        return []
    if "," not in payload and "=" in payload:
        try:
            return shlex.split(payload)
        except ValueError:
            pass
    return [token.strip() for token in next(csv.reader([payload], skipinitialspace=True))]


def _looks_like_header(prefix: str, tokens: list[str]) -> bool:
    known = set(POSITIONAL_SCHEMAS[prefix]) | {
        "header",
        "columns",
        "schema",
        "timestamp_us",
        "timestamp_s",
        "start_us",
        "end_us",
    }
    normalized = [normalized_name(token) for token in tokens]
    return len(normalized) >= 2 and sum(token in known for token in normalized) >= 2


def _record_from_tokens(prefix: str, tokens: list[str], header: list[str] | None) -> dict[str, str]:
    if any("=" in token for token in tokens):
        record: dict[str, str] = {}
        extras: list[str] = []
        for token in tokens:
            if "=" not in token:
                if token:
                    extras.append(token)
                continue
            key, value = token.split("=", 1)
            key = normalized_name(key)
            if not key:
                extras.append(token)
            else:
                record[key] = value.strip()
        if extras:
            record["unparsed_tokens"] = " | ".join(extras)
        return record

    names = header or POSITIONAL_SCHEMAS[prefix]
    record = {}
    for index, value in enumerate(tokens):
        key = names[index] if index < len(names) else f"extra_{index - len(names) + 1}"
        record[normalized_name(key)] = value
    return record


def parse_serial_files(
    paths: Iterable[Path], output_dir: Path, initial_context: dict[str, str] | None = None
) -> dict[str, list[dict[str, str]]]:
    records: dict[str, list[dict[str, str]]] = {prefix: [] for prefix in PREFIX_TO_FILE}
    global_index = {prefix: 0 for prefix in PREFIX_TO_FILE}
    forced_context = dict(initial_context or {})

    for path in paths:
        headers: dict[str, list[str]] = {}
        context: dict[str, str] = dict(initial_context or {})
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                match = PREFIX_RE.search(line)
                if not match:
                    continue
                prefix = match.group(1)
                tokens = _payload_tokens(match.group(2))
                if not tokens:
                    continue

                first = normalized_name(tokens[0])
                if first in {"header", "columns", "schema"}:
                    headers[prefix] = [normalized_name(token) for token in tokens[1:]]
                    continue
                if not any("=" in token for token in tokens) and _looks_like_header(prefix, tokens):
                    headers[prefix] = [normalized_name(token) for token in tokens]
                    continue

                record = _record_from_tokens(prefix, tokens, headers.get(prefix))
                if prefix in {"PULSE_CAPTURE_PROOF", "PULSE_CAPTURE_STATUS"}:
                    if record.get("firmware_sequence", "") and not record.get("sequence", ""):
                        record["sequence"] = record["firmware_sequence"]
                    if record.get("sequence", "") and not record.get("firmware_sequence", ""):
                        record["firmware_sequence"] = record["sequence"]
                for source, target in FIELD_ALIASES.items():
                    if record.get(source, "") and not record.get(target, ""):
                        record[target] = record[source]
                for key, value in context.items():
                    record.setdefault(key, value)

                # Operator-supplied capture context is an explicit final
                # override, including for placeholder firmware values such as
                # pair_id=operator_required.  Apply it before hashing metadata
                # so an override cannot retain a stale build-context digest.
                record.update(forced_context)
                if prefix == "PULSE_META":
                    record["build_context_sha256"] = _build_context_sha256(record)
                    for key, value in record.items():
                        if key in CONTEXT_FIELDS and value:
                            context[key] = value
                    meta_key = normalized_name(record.get("key", ""))
                    if meta_key in CONTEXT_FIELDS and record.get("value", ""):
                        context[meta_key] = record["value"]

                global_index[prefix] += 1
                record["source_file"] = str(path)
                record["source_line"] = str(line_number)
                record["record_index"] = str(global_index[prefix])
                records[prefix].append(record)

    # Stage records intentionally remain compact on the UART.  Enrich their
    # session-state label from exactly one event on the same physical board
    # and monotonic event index; never fall back to trial order.
    events_by_identity: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for event in records["PULSE_EVENT"]:
        identity = tuple(
            str(event.get(field, "")).strip()
            for field in ("run_id", "pair_id", "board_id", "event_index")
        )
        if all(identity):
            events_by_identity.setdefault(identity, []).append(event)
    for stage in records["PULSE_STAGE"]:
        identity = tuple(
            str(stage.get(field, "")).strip()
            for field in ("run_id", "pair_id", "board_id", "event_index")
        )
        matches = events_by_identity.get(identity, []) if all(identity) else []
        if len(matches) != 1:
            continue
        event = matches[0]
        for field in (
            "exchange_id",
            "remote_session_started",
            *PULSE_EVENT_CONFIGURATION_FIELDS,
        ):
            if event.get(field, "") != "":
                stage.setdefault(field, event[field])

    output_dir.mkdir(parents=True, exist_ok=True)
    for prefix, filename in PREFIX_TO_FILE.items():
        fields = PREFERRED_FIELDS[prefix] + ["source_file", "source_line", "record_index"]
        # Passing preferred rather than a fixed schema retains unknown key=value
        # fields, which is essential for forward-compatible firmware records.
        write_csv_rows(output_dir / filename, records[prefix], preferred=fields)
    return records
