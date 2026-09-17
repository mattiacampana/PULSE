"""Audit firmware communication counters without claiming over-the-air bytes.

The PULSE firmware counts application payload and GATT attribute-value bytes at
the protocol endpoints.  These counters are useful without a BLE sniffer, but
they deliberately exclude ATT/L2CAP/LL headers, advertisements, control PDUs,
and retransmissions.  Accepted sessions are joined only by the exact
``run_id,pair_id,exchange_id`` identity emitted by both boards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from common import (
    PULSE_BUILD_CONTEXT_FIELDS,
    PULSE_EVENT_CONFIGURATION_FIELDS,
    is_post_warmup,
    normalized_name,
    parse_bool,
    parse_float,
    read_csv_rows,
    row_success,
    sha256_file,
    write_csv_rows,
)
from summarize import ACCEPTED_PATHS, accepted_session_integrity


COUNTER_ALIASES = {
    "application_tx_bytes": ("application_tx_bytes",),
    "application_rx_bytes": ("application_rx_bytes",),
    "pulse_framing_tx_bytes": (
        "pulse_framing_tx_bytes",
        "protocol_header_tx_bytes",
        "protocol_tx_bytes",
    ),
    "pulse_framing_rx_bytes": (
        "pulse_framing_rx_bytes",
        "protocol_header_rx_bytes",
        "protocol_rx_bytes",
    ),
    "att_value_tx_bytes": ("att_value_tx_bytes", "att_tx_bytes"),
    "att_value_rx_bytes": ("att_value_rx_bytes", "att_rx_bytes"),
    "att_value_tx_fragments": ("att_value_tx_fragments", "tx_chunks"),
    "att_value_rx_fragments": ("att_value_rx_fragments", "rx_chunks"),
}

COPY_FIELDS = (
    "run_id",
    "pair_id",
    "board_id",
    "trial_id",
    "exchange_id",
    "event_index",
    "event_path",
    "role",
    "connection_state",
    "post_warmup",
    "warmup",
    "remote_session_started",
    "success",
    "status",
    "error_code",
    *PULSE_BUILD_CONTEXT_FIELDS,
    *PULSE_EVENT_CONFIGURATION_FIELDS,
)


def _counter(row: Mapping[str, object], canonical: str) -> int | None:
    for field in COUNTER_ALIASES[canonical]:
        value = parse_float(row.get(field))
        if value is None:
            continue
        if value < 0 or not float(value).is_integer():
            return None
        return int(value)
    return None


def _counter_issues(row: Mapping[str, object], label: str) -> list[str]:
    issues: list[str] = []
    values = {name: _counter(row, name) for name in COUNTER_ALIASES}
    missing = [name for name, value in values.items() if value is None]
    if missing:
        return [f"{label}:missing_or_invalid_{','.join(missing)}"]
    if values["att_value_tx_bytes"] != (
        values["application_tx_bytes"] + values["pulse_framing_tx_bytes"]
    ):
        issues.append(f"{label}:tx_att_value_not_application_plus_pulse_framing")
    if values["att_value_rx_bytes"] != (
        values["application_rx_bytes"] + values["pulse_framing_rx_bytes"]
    ):
        issues.append(f"{label}:rx_att_value_not_application_plus_pulse_framing")
    if values["att_value_tx_bytes"] == 0 and values["att_value_tx_fragments"] != 0:
        issues.append(f"{label}:tx_fragments_without_att_value_bytes")
    if values["att_value_rx_bytes"] == 0 and values["att_value_rx_fragments"] != 0:
        issues.append(f"{label}:rx_fragments_without_att_value_bytes")
    return issues


def _directional_row(
    source: Mapping[str, object],
    *,
    event_source_file: str,
    event_source_sha256: str,
) -> dict[str, object]:
    application = int(_counter(source, "application_tx_bytes") or 0)
    framing = int(_counter(source, "pulse_framing_tx_bytes") or 0)
    att_value = int(_counter(source, "att_value_tx_bytes") or 0)
    fragments = int(_counter(source, "att_value_tx_fragments") or 0)
    row: dict[str, object] = {
        field: source.get(field, "") for field in COPY_FIELDS
    }
    row.update(
        {
            "logical_role": normalized_name(str(source.get("role", ""))),
            "direction": "tx",
            "measurement_scope": "firmware_att_value_boundary",
            "evidence_class": "firmware_counter_not_over_air",
            "application_payload_bytes": application,
            "pulse_framing_bytes": framing,
            "att_value_bytes": att_value,
            "att_value_fragment_count": fragments,
            "payload_efficiency": application / att_value if att_value else None,
            "quality_valid": 1,
            "join_status": "paired_firmware_exact_exchange_id",
            "event_source_file": event_source_file,
            "event_source_sha256": event_source_sha256,
        }
    )
    return row


def analyze_communication(
    events_csv: Path, output_dir: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Create sender-counted communication rows and an exact-pair audit."""

    event_rows = read_csv_rows(events_csv)
    source_sha256 = sha256_file(events_csv)
    initiators, responders, integrity_issues = accepted_session_integrity(event_rows)
    metrics: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []

    if integrity_issues:
        audit.append(
            {
                "audit_type": "accepted_session_integrity",
                "status": "fail",
                "issue_count": len(integrity_issues),
                "issues": " | ".join(integrity_issues),
                "event_source_file": str(events_csv),
                "event_source_sha256": source_sha256,
            }
        )

    for key, initiator in sorted(initiators.items()):
        run_id, pair_id, exchange_id = key
        responder = responders.get(key)
        remote_started = parse_bool(initiator.get("remote_session_started"))
        issues = _counter_issues(initiator, "initiator")
        if remote_started is True and responder is None:
            issues.append("remote_session_missing_responder")
        if responder is not None:
            issues.extend(_counter_issues(responder, "responder"))
            # A failed exchange may stop between endpoint callbacks, so its
            # submitted/accepted counters need not match.  Exact equality is
            # a completeness invariant only for a successful two-sided
            # transaction; failed rows still retain internally consistent
            # counters and remain in the failure denominator.
            if row_success(initiator) is True and row_success(responder) is True:
                comparisons = (
                    ("application_tx_bytes", "application_rx_bytes", "application_i_to_r"),
                    ("application_rx_bytes", "application_tx_bytes", "application_r_to_i"),
                    ("att_value_tx_bytes", "att_value_rx_bytes", "att_value_i_to_r"),
                    ("att_value_rx_bytes", "att_value_tx_bytes", "att_value_r_to_i"),
                    ("pulse_framing_tx_bytes", "pulse_framing_rx_bytes", "framing_i_to_r"),
                    ("pulse_framing_rx_bytes", "pulse_framing_tx_bytes", "framing_r_to_i"),
                    ("att_value_tx_fragments", "att_value_rx_fragments", "fragments_i_to_r"),
                    ("att_value_rx_fragments", "att_value_tx_fragments", "fragments_r_to_i"),
                )
                for initiator_field, responder_field, label in comparisons:
                    if _counter(initiator, initiator_field) != _counter(
                        responder, responder_field
                    ):
                        issues.append(f"endpoint_mismatch_{label}")

        status = "pass" if not issues else "fail"
        audit.append(
            {
                "audit_type": "firmware_counter_pair",
                "run_id": run_id,
                "pair_id": pair_id,
                "exchange_id": exchange_id,
                "event_path": initiator.get("event_path", ""),
                "trial_id": initiator.get("trial_id", ""),
                "post_warmup": int(is_post_warmup(initiator)),
                "remote_session_started": int(bool(remote_started)),
                "status": status,
                "issue_count": len(issues),
                "issues": " | ".join(issues),
                "measurement_scope": "firmware_att_value_boundary",
                "evidence_class": "firmware_counter_not_over_air",
                "event_source_file": str(events_csv),
                "event_source_sha256": source_sha256,
            }
        )

        init_metric = _directional_row(
            initiator,
            event_source_file=str(events_csv),
            event_source_sha256=source_sha256,
        )
        init_metric["quality_valid"] = int(status == "pass")
        metrics.append(init_metric)
        if responder is None:
            continue
        responder_metric = _directional_row(
            responder,
            event_source_file=str(events_csv),
            event_source_sha256=source_sha256,
        )
        responder_metric["quality_valid"] = int(status == "pass")
        metrics.append(responder_metric)

        app = int(init_metric["application_payload_bytes"]) + int(
            responder_metric["application_payload_bytes"]
        )
        framing = int(init_metric["pulse_framing_bytes"]) + int(
            responder_metric["pulse_framing_bytes"]
        )
        att_value = int(init_metric["att_value_bytes"]) + int(
            responder_metric["att_value_bytes"]
        )
        fragments = int(init_metric["att_value_fragment_count"]) + int(
            responder_metric["att_value_fragment_count"]
        )
        pair_row = dict(init_metric)
        pair_row.update(
            {
                "board_id": "pair",
                "role": "pair",
                "logical_role": "pair",
                "direction": "sender_sum_no_rx_double_count",
                "connection_state": initiator.get("connection_state", ""),
                "application_payload_bytes": app,
                "pulse_framing_bytes": framing,
                "att_value_bytes": att_value,
                "att_value_fragment_count": fragments,
                "payload_efficiency": app / att_value if att_value else None,
                "success": int(
                    row_success(initiator) is True and row_success(responder) is True
                ),
            }
        )
        metrics.append(pair_row)

    known_keys = set(initiators)
    for key, responder in sorted(responders.items()):
        if key in known_keys:
            continue
        # accepted_session_integrity already records the orphan.  Retain a
        # failing audit row so a caller cannot accidentally ignore it.
        audit.append(
            {
                "audit_type": "firmware_counter_pair",
                "run_id": key[0],
                "pair_id": key[1],
                "exchange_id": key[2],
                "event_path": responder.get("event_path", ""),
                "status": "fail",
                "issue_count": 1,
                "issues": "orphan_responder",
                "event_source_file": str(events_csv),
                "event_source_sha256": source_sha256,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "event_communication_metrics.csv",
        metrics,
        preferred=[
            "run_id",
            "pair_id",
            "board_id",
            "exchange_id",
            "event_index",
            "trial_id",
            "event_path",
            "role",
            "logical_role",
            "direction",
            "connection_state",
            "remote_session_started",
            "post_warmup",
            "warmup",
            "success",
            "measurement_scope",
            "evidence_class",
            "application_payload_bytes",
            "pulse_framing_bytes",
            "att_value_bytes",
            "att_value_fragment_count",
            "payload_efficiency",
            "quality_valid",
            "join_status",
            "event_source_file",
            "event_source_sha256",
            *PULSE_BUILD_CONTEXT_FIELDS,
            *PULSE_EVENT_CONFIGURATION_FIELDS,
        ],
    )
    write_csv_rows(
        output_dir / "communication_audit.csv",
        audit,
        preferred=[
            "audit_type",
            "run_id",
            "pair_id",
            "exchange_id",
            "event_path",
            "trial_id",
            "post_warmup",
            "remote_session_started",
            "status",
            "issue_count",
            "issues",
            "measurement_scope",
            "evidence_class",
            "event_source_file",
            "event_source_sha256",
        ],
    )
    return metrics, audit


def require_valid_communication_audit(
    audit: list[dict[str, object]],
) -> None:
    if not audit:
        raise ValueError("communication audit found no accepted-session records")
    failures = [row for row in audit if str(row.get("status", "")) != "pass"]
    if failures:
        details = " | ".join(str(row.get("issues", "")) for row in failures)
        raise ValueError("firmware communication-counter audit failed: " + details)
