"""Aggregate PULSE trials with robust descriptive statistics and audits."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

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
    summarize_values,
    write_csv_rows,
)

RESULT_CONTEXT_FIELDS = (
    *PULSE_BUILD_CONTEXT_FIELDS,
    *PULSE_EVENT_CONFIGURATION_FIELDS,
)


INPUT_SPECS = {
    "event_metrics.csv": {
        "output": "event_summary.csv",
        "groups": (
            *RESULT_CONTEXT_FIELDS,
            "event_path",
            "role",
            "connection_state",
            "remote_session_started",
        ),
    },
    "stage_metrics.csv": {
        "output": "stage_summary.csv",
        "groups": (
            *RESULT_CONTEXT_FIELDS,
            "event_path",
            "role",
            "connection_state",
            "remote_session_started",
            "stage",
            "stage_code",
        ),
    },
    "correctness.csv": {
        "output": "correctness_summary.csv",
        "groups": (
            *RESULT_CONTEXT_FIELDS,
            "event_path",
            "role",
            "artifact_sha256",
            "fixture_hash",
            "reference",
        ),
    },
    "event_power_metrics.csv": {
        "output": "power_summary.csv",
        "groups": (
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
        "include_failed_metrics": True,
    },
    "stage_power_metrics.csv": {
        "output": "stage_power_summary.csv",
        "groups": (
            *RESULT_CONTEXT_FIELDS,
            "event_path",
            "role",
            "connection_state",
            "remote_session_started",
            "measurement_scope",
            "baseline_source",
            "baseline_method",
            "outcome",
            "stage_sequence",
            "stage",
            "stage_code",
        ),
        "include_failed_metrics": True,
    },
    "power_capture_metadata.csv": {
        "output": "baseline_power_summary.csv",
        "groups": (
            *RESULT_CONTEXT_FIELDS,
            "role",
            "capture_mode",
            "baseline_method",
        ),
    },
    "event_link_metrics.csv": {
        "output": "link_summary.csv",
        "groups": (
            *RESULT_CONTEXT_FIELDS,
            "event_path",
            "role",
            "connection_state",
            "remote_session_started",
            "outcome",
        ),
        "include_failed_metrics": True,
    },
    "event_communication_metrics.csv": {
        "output": "communication_summary.csv",
        "groups": (
            *RESULT_CONTEXT_FIELDS,
            "event_path",
            "role",
            "direction",
            "connection_state",
            "remote_session_started",
            "measurement_scope",
            "evidence_class",
            "outcome",
        ),
        "include_failed_metrics": True,
    },
}

NON_METRIC_FIELDS = {
    *RESULT_CONTEXT_FIELDS,
    "run_id",
    "pair_id",
    "board_id",
    "trial_id",
    "role",
    "event_path",
    "connection_state",
    "post_warmup",
    "warmup",
    "success",
    "pass",
    "passed",
    "ok",
    "status",
    "outcome",
    "error_code",
    "timeout",
    "record_index",
    "source_line",
    "source_file",
    "source_sha256",
    "marker_event_index",
    "event_index",
    "exchange_id",
    "remote_session_started",
    "measurement_scope",
    "logical_role",
    "serial_event_present",
    "power_channel_role",
    "power_event_sequence",
    "stage_sequence",
    "stage_code",
    "stage",
    "baseline_method",
    "baseline_source",
    "baseline_metadata_file",
    "baseline_metadata_sha256",
    "baseline_capture_source_file",
    "artifact_sha256",
    "fixture_hash",
    "fixture_sha256",
    "reference",
    "capture_mode",
    "start_time_s",
    "end_time_s",
    "capture_time_s",
    "capture_start_s",
    "capture_end_s",
    "join_status",
    "truncated_start",
    "truncated_end",
    "capture_id",
    "coverage_count",
    "observed_packet_count",
    "crc_failure_count",
    "capture_gap_count",
    "boundary_ambiguous_packet_count",
    "pdu_byte_definition",
    "role_coverage_ok",
    "quality_valid",
    "quality_issues",
    "canonical_event_source",
    "canonical_event_sha256",
    "event_start_counter",
    "event_end_counter",
    "wall_start_counter",
    "wall_end_counter",
    "analysis_source_sha256",
    "direction",
    "evidence_class",
    "event_source_file",
    "event_source_sha256",
}


REQUIRED_INITIATOR_STRATA = (
    ("local", "local", "connected"),
    ("no_contact", "local", "disconnected"),
    ("accepted_connected", "initiator", "connected"),
    ("accepted_discovery", "initiator", "disconnected"),
)
ACCEPTED_PATHS = {"accepted_connected", "accepted_discovery"}
ACCEPTED_PHYSICAL_STATES = {
    ("accepted_connected", "initiator"): "connected",
    ("accepted_connected", "responder"): "connected",
    ("accepted_discovery", "initiator"): "disconnected",
    ("accepted_discovery", "responder"): "connected",
}
MISSING_IDENTIFIER_VALUES = {
    "",
    "unknown",
    "unset",
    "na",
    "n_a",
    "operator_required",
}


def _identifier(row: Mapping[str, object], field: str) -> str:
    return str(row.get(field, "")).strip()


def _identifier_present(value: str) -> bool:
    return value.lower() not in MISSING_IDENTIFIER_VALUES


def _warmup_label(row: Mapping[str, object]) -> bool | None:
    post = parse_bool(row.get("post_warmup"))
    if post is not None:
        return not post
    return parse_bool(row.get("warmup"))


def accepted_session_integrity(
    event_rows: Iterable[Mapping[str, object]],
) -> tuple[
    dict[tuple[str, str, str], Mapping[str, object]],
    dict[tuple[str, str, str], Mapping[str, object]],
    list[str],
]:
    """Index accepted sessions and report integrity errors without guessing joins.

    Accepted records are paired only by the firmware's exact
    ``run_id,pair_id,exchange_id`` identity.  ``event_index`` and ``trial_id``
    are deliberately not fallbacks because they can restart across runs.
    """

    initiator_groups: dict[
        tuple[str, str, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    responder_groups: dict[
        tuple[str, str, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    issues: list[str] = []

    for position, row in enumerate(event_rows, 1):
        record = _identifier(row, "record_index") or str(position)
        event_path = normalized_name(_identifier(row, "event_path"))
        role = normalized_name(_identifier(row, "role"))
        connection_state = normalized_name(_identifier(row, "connection_state"))

        missing_labels = []
        if not event_path:
            missing_labels.append("event_path")
        if not role:
            missing_labels.append("role")
        if not connection_state:
            missing_labels.append("connection_state")
        if row_success(row) is None:
            missing_labels.append("success")
        if _warmup_label(row) is None:
            missing_labels.append("warmup/post_warmup")
        if missing_labels:
            issues.append(f"record {record} missing/invalid labels: {','.join(missing_labels)}")

        if event_path not in ACCEPTED_PATHS:
            continue
        if role not in {"initiator", "responder"}:
            issues.append(
                f"record {record} accepted path has invalid role {role or '<missing>'}"
            )
            continue

        expected_state = ACCEPTED_PHYSICAL_STATES[(event_path, role)]
        if connection_state != expected_state:
            issues.append(
                f"record {record} has impossible accepted-session physical tuple "
                f"{event_path}/{role}/{connection_state or '<missing>'}; "
                f"expected connection_state={expected_state}"
            )

        key_values = tuple(_identifier(row, field) for field in ("run_id", "pair_id", "exchange_id"))
        missing_keys = [
            field
            for field, value in zip(("run_id", "pair_id", "exchange_id"), key_values)
            if not _identifier_present(value)
        ]
        if missing_keys:
            issues.append(
                f"record {record} accepted session missing key(s): {','.join(missing_keys)}"
            )
            continue

        if role == "initiator":
            remote_started = parse_bool(row.get("remote_session_started"))
            if remote_started is None:
                issues.append(
                    f"record {record} missing/invalid remote_session_started label"
                )
            elif not remote_started and row_success(row) is not False:
                issues.append(
                    f"record {record} has remote_session_started=0 without success=0"
                )
            initiator_groups[key_values].append(row)
        elif role == "responder":
            remote_started = parse_bool(row.get("remote_session_started"))
            if remote_started is not True:
                issues.append(
                    f"record {record} accepted responder requires "
                    "remote_session_started=1"
                )
            responder_groups[key_values].append(row)

    for key, rows in sorted(initiator_groups.items()):
        if len(rows) != 1:
            issues.append(f"duplicate initiator records for run/pair/exchange {key}: {len(rows)}")
    for key, rows in sorted(responder_groups.items()):
        if len(rows) != 1:
            issues.append(f"duplicate responder records for run/pair/exchange {key}: {len(rows)}")

    initiators = {key: rows[0] for key, rows in initiator_groups.items() if len(rows) == 1}
    responders = {key: rows[0] for key, rows in responder_groups.items() if len(rows) == 1}

    for key, responder in sorted(responders.items()):
        if key not in initiator_groups:
            issues.append(f"orphan responder record for run/pair/exchange {key}")
            continue
        if key not in initiators:
            continue
        initiator = initiators[key]
        if normalized_name(_identifier(initiator, "event_path")) != normalized_name(
            _identifier(responder, "event_path")
        ):
            issues.append(f"event_path mismatch for run/pair/exchange {key}")
        if _identifier(initiator, "trial_id") != _identifier(responder, "trial_id"):
            issues.append(f"trial_id mismatch for run/pair/exchange {key}")
        if _warmup_label(initiator) != _warmup_label(responder):
            issues.append(f"warmup label mismatch for run/pair/exchange {key}")

    for key, initiator in sorted(initiators.items()):
        remote_started = parse_bool(initiator.get("remote_session_started"))
        responder_count = len(responder_groups.get(key, []))
        if remote_started is True and responder_count != 1:
            issues.append(
                f"remote session requires exactly one responder for run/pair/exchange {key}; "
                f"found {responder_count}"
            )
        elif remote_started is False and responder_count != 0:
            issues.append(
                f"pre-session failure must not have a responder for run/pair/exchange {key}; "
                f"found {responder_count}"
            )

    return initiators, responders, issues

UNIT_SUFFIXES = {
    "_us": "us",
    "_ms": "ms",
    "_s": "s",
    "_hz": "Hz",
    "_bytes": "bytes",
    "_cycles": "cycles",
    "_j": "J",
    "_w": "W",
    "_a": "A",
    "_c": "C",
}


def metric_unit(metric: str) -> str:
    if metric == "cycles":
        return "cycles"
    for suffix, unit in UNIT_SUFFIXES.items():
        if metric.endswith(suffix):
            return unit
    if "cosine" in metric:
        return "ratio"
    return ""


def valid_metric_value(metric: str, value: float | None) -> bool:
    if value is None:
        return False
    nonnegative_suffixes = ("_bytes", "_cycles", "_us", "_ms", "_chunks")
    nonnegative_names = {"cycles", "retransmissions", "tx_chunks", "rx_chunks"}
    if metric.endswith(nonnegative_suffixes) or metric in nonnegative_names:
        return value >= 0
    return True


def infer_metrics(rows: list[dict[str, str]]) -> list[str]:
    fields = sorted({key for row in rows for key in row} - NON_METRIC_FIELDS)
    result = []
    for field in fields:
        nonempty = [row.get(field, "") for row in rows if row.get(field, "") != ""]
        if nonempty and all(parse_float(value) is not None for value in nonempty):
            result.append(field)
    return result


def quality_valid(row: Mapping[str, object]) -> bool:
    explicit_quality = parse_bool(row.get("quality_valid"))
    if explicit_quality is False:
        return False
    for field in ("truncated_start", "truncated_end"):
        value = str(row.get(field, "")).strip().lower()
        if value in {"1", "true", "yes"}:
            return False
    join_status = str(row.get("join_status", "")).strip().lower()
    if join_status and join_status not in {
        "matched",
        "paired_remote_session_exact_exchange_id",
        "initiator_only_pre_session_failure_exact_exchange_id",
        "paired_firmware_exact_exchange_id",
        "ngmo2_exact_ready_event_id",
        "ngmo2_autonomous_exact_proof_sequence",
    }:
        return False
    return True


def _source_digest_fields(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Bind an aggregate row to the exact set of hashed metric inputs."""

    digests = sorted(
        {
            str(row.get("analysis_source_sha256", "")).strip().lower()
            for row in rows
            if str(row.get("analysis_source_sha256", "")).strip()
        }
    )
    encoded = "\n".join(digests).encode("ascii")
    return {
        "analysis_source_count": len(digests),
        "analysis_source_sha256_set": ";".join(digests),
        "analysis_source_set_sha256": hashlib.sha256(encoded).hexdigest()
        if digests
        else "",
    }


def aggregate_rows(
    rows: Iterable[dict[str, str]],
    group_fields: tuple[str, ...],
    include_failed_metrics: bool = False,
) -> list[dict[str, object]]:
    post_warmup_rows = [row for row in rows if is_post_warmup(row)]
    metrics = infer_metrics([row for row in post_warmup_rows if quality_valid(row)])
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in post_warmup_rows:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)

    summaries: list[dict[str, object]] = []
    for group_key in sorted(groups):
        source_group_rows = groups[group_key]
        group_rows = [row for row in source_group_rows if quality_valid(row)]
        outcomes = [row_success(row) for row in group_rows]
        known_outcomes = [outcome for outcome in outcomes if outcome is not None]
        failures = sum(outcome is False for outcome in known_outcomes)
        successes = sum(outcome is True for outcome in known_outcomes)
        base: dict[str, object] = dict(zip(group_fields, group_key))
        base.update(
            {
                "total_count": len(group_rows),
                "source_count": len(source_group_rows),
                "quality_excluded_count": len(source_group_rows) - len(group_rows),
                "success_count": successes,
                "failure_count": failures,
                "unknown_outcome_count": len(group_rows) - len(known_outcomes),
                "failure_rate": failures / len(known_outcomes) if known_outcomes else None,
                "independent_pair_count": len(
                    {
                        row.get("pair_id", "")
                        for row in group_rows
                        if row.get("pair_id", "").strip().lower()
                        not in {"", "unknown", "unset", "na", "n_a", "operator_required"}
                    }
                ),
            }
        )
        base.update(_source_digest_fields(group_rows))
        emitted = 0
        for metric in metrics:
            # Failed protocol transactions contribute to failure probability but
            # not to the successful-path latency/energy distribution.
            values = [
                value
                for row in group_rows
                if include_failed_metrics or row_success(row) is not False
                for value in [parse_float(row.get(metric))]
                if valid_metric_value(metric, value)
            ]
            if not values:
                continue
            summary = dict(base)
            summary.update({"metric": metric, "unit": metric_unit(metric)})
            summary.update(summarize_values(values))
            summaries.append(summary)
            emitted += 1
        if emitted == 0:
            summary = dict(base)
            summary.update({"metric": "", "unit": "", "n": 0})
            summaries.append(summary)
    return summaries


def protocol_audit(
    event_rows: list[dict[str, str]], min_repetitions: int
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in event_rows:
        if is_post_warmup(row):
            key = tuple(
                normalized_name(row.get(field, ""))
                for field in ("event_path", "role", "connection_state")
            )
            groups[key].append(row)

    required = set(REQUIRED_INITIATOR_STRATA)
    ordered_keys = [*REQUIRED_INITIATOR_STRATA]
    ordered_keys.extend(sorted(set(groups).difference(required)))
    audit: list[dict[str, object]] = []
    for event_path, role, connection_state in ordered_keys:
        rows = groups.get((event_path, role, connection_state), [])
        outcomes = [row_success(row) for row in rows]
        known = [value for value in outcomes if value is not None]
        warmup_labeled = all(
            row.get("post_warmup", "") != "" or row.get("warmup", "") != "" for row in rows
        )
        pair_values = [row.get("pair_id", "").strip() for row in rows]
        run_values = [row.get("run_id", "").strip() for row in rows]
        board_values = [row.get("board_id", "").strip() for row in rows]
        valid_pairs = {
            value
            for value in pair_values
            if _identifier_present(value)
        }
        is_required = (event_path, role, connection_state) in required
        if not is_required:
            stratum_check = "not_required"
            repetition_check = "not_required"
        elif not rows:
            stratum_check = "missing"
            repetition_check = "fail"
        elif len(rows) < min_repetitions:
            stratum_check = "underfilled"
            repetition_check = "fail"
        else:
            stratum_check = "pass"
            repetition_check = "pass"
        audit.append(
            {
                "audit_type": "stratum",
                "event_path": event_path,
                "role": role,
                "connection_state": connection_state,
                "required_stratum": int(is_required),
                "stratum_check": stratum_check,
                "post_warmup_repetitions": len(rows),
                "minimum_required": min_repetitions if is_required else "",
                "repetition_check": repetition_check,
                "success_label_present": int(len(known) == len(rows)),
                "warmup_label_present": int(warmup_labeled),
                "pair_id_present": int(all(_identifier_present(value) for value in pair_values)),
                "run_id_present": int(all(_identifier_present(value) for value in run_values)),
                "board_id_present": int(all(_identifier_present(value) for value in board_values)),
                "independent_pair_count": len(valid_pairs),
                "failure_count": sum(value is False for value in known),
                "failure_rate": sum(value is False for value in known) / len(known) if known else None,
                "integrity_check": "pass",
                "integrity_issue_count": 0,
                "integrity_issues": "",
            }
        )

    post_warmup_rows = [row for row in event_rows if is_post_warmup(row)]
    campaign_identities = {
        tuple(_identifier(row, field) for field in ("run_id", "pair_id", "board_id"))
        for row in post_warmup_rows
        if (
            normalized_name(row.get("event_path", "")),
            normalized_name(row.get("role", "")),
            normalized_name(row.get("connection_state", "")),
        )
        in required
        and all(
            _identifier_present(_identifier(row, field))
            for field in ("run_id", "pair_id", "board_id")
        )
    }
    for run_id, pair_id, board_id in sorted(campaign_identities):
        for event_path, role, connection_state in REQUIRED_INITIATOR_STRATA:
            rows = [
                row
                for row in post_warmup_rows
                if _identifier(row, "run_id") == run_id
                and _identifier(row, "pair_id") == pair_id
                and _identifier(row, "board_id") == board_id
                and normalized_name(row.get("event_path", "")) == event_path
                and normalized_name(row.get("role", "")) == role
                and normalized_name(row.get("connection_state", "")) == connection_state
            ]
            outcomes = [row_success(row) for row in rows]
            known = [value for value in outcomes if value is not None]
            if not rows:
                stratum_check = "missing"
            elif len(rows) < min_repetitions:
                stratum_check = "underfilled"
            else:
                stratum_check = "pass"
            audit.append(
                {
                    "audit_type": "capture_stratum",
                    "run_id": run_id,
                    "pair_id": pair_id,
                    "board_id": board_id,
                    "event_path": event_path,
                    "role": role,
                    "connection_state": connection_state,
                    "required_stratum": 1,
                    "stratum_check": stratum_check,
                    "post_warmup_repetitions": len(rows),
                    "minimum_required": min_repetitions,
                    "repetition_check": "pass" if stratum_check == "pass" else "fail",
                    "success_label_present": int(len(known) == len(rows)),
                    "warmup_label_present": 1,
                    "run_id_present": 1,
                    "pair_id_present": 1,
                    "board_id_present": 1,
                    "independent_pair_count": 1,
                    "failure_count": sum(value is False for value in known),
                    "failure_rate": (
                        sum(value is False for value in known) / len(known)
                        if known
                        else None
                    ),
                    "integrity_check": "pass",
                    "integrity_issue_count": 0,
                    "integrity_issues": "",
                }
            )

    initiators, responders, integrity_issues = accepted_session_integrity(event_rows)
    audit.append(
        {
            "audit_type": "accepted_session_pairing",
            "event_path": "accepted_sessions",
            "role": "initiator_responder",
            "connection_state": "exact_run_pair_exchange",
            "required_stratum": 0,
            "stratum_check": "not_applicable",
            "post_warmup_repetitions": sum(
                is_post_warmup(row) for row in initiators.values()
            ),
            "minimum_required": "",
            "repetition_check": "not_applicable",
            "success_label_present": 1,
            "warmup_label_present": 1,
            "pair_id_present": 1,
            "run_id_present": 1,
            "board_id_present": 1,
            "independent_pair_count": len(
                {_identifier(row, "pair_id") for row in initiators.values()}
            ),
            "accepted_initiator_count": len(initiators),
            "accepted_responder_count": len(responders),
            "integrity_check": "pass" if not integrity_issues else "fail",
            "integrity_issue_count": len(integrity_issues),
            "integrity_issues": " | ".join(integrity_issues),
        }
    )
    return audit


def summarize_directory(
    input_dirs: Iterable[Path], output_dir: Path, min_repetitions: int
) -> tuple[list[Path], list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    all_event_rows: list[dict[str, str]] = []
    for filename, spec in INPUT_SPECS.items():
        rows: list[dict[str, str]] = []
        for input_dir in input_dirs:
            path = input_dir / filename
            if path.exists():
                source_sha256 = sha256_file(path)
                rows.extend(
                    {
                        **row,
                        "analysis_source_sha256": source_sha256,
                    }
                    for row in read_csv_rows(path)
                )
        if filename == "event_metrics.csv":
            all_event_rows.extend(rows)
        if not rows:
            continue
        if spec.get("include_failed_metrics"):
            for row in rows:
                success = row_success(row)
                row["outcome"] = (
                    "success" if success is True else "failure" if success is False else "unknown"
                )
        summaries = aggregate_rows(
            rows,
            spec["groups"],
            include_failed_metrics=bool(spec.get("include_failed_metrics")),
        )
        output_path = output_dir / str(spec["output"])
        write_csv_rows(
            output_path,
            summaries,
            preferred=[
                *spec["groups"],
                "metric",
                "unit",
                "n",
                "source_count",
                "total_count",
                "quality_excluded_count",
                "success_count",
                "failure_count",
                "unknown_outcome_count",
                "failure_rate",
                "independent_pair_count",
                "analysis_source_count",
                "analysis_source_sha256_set",
                "analysis_source_set_sha256",
                "mean",
                "median",
                "q1",
                "q3",
                "iqr",
                "p95",
                "min",
                "max",
            ],
        )
        written.append(output_path)

        pair_rows = [
            row
            for row in rows
            if _identifier_present(row.get("run_id", ""))
            and _identifier_present(row.get("pair_id", ""))
        ]
        if pair_rows:
            pair_group_fields = tuple(
                dict.fromkeys(("run_id", "pair_id", "board_id", *spec["groups"]))
            )
            pair_summaries = aggregate_rows(
                pair_rows,
                pair_group_fields,
                include_failed_metrics=bool(spec.get("include_failed_metrics")),
            )
            pair_output_path = output_dir / (
                Path(str(spec["output"])).stem + "_by_pair.csv"
            )
            write_csv_rows(
                pair_output_path,
                pair_summaries,
                preferred=[
                    *pair_group_fields,
                    "metric",
                    "unit",
                    "n",
                    "source_count",
                    "total_count",
                    "quality_excluded_count",
                    "success_count",
                    "failure_count",
                    "unknown_outcome_count",
                    "failure_rate",
                    "independent_pair_count",
                    "analysis_source_count",
                    "analysis_source_sha256_set",
                    "analysis_source_set_sha256",
                    "mean",
                    "median",
                    "q1",
                    "q3",
                    "iqr",
                    "p95",
                    "min",
                    "max",
                ],
            )
            written.append(pair_output_path)

    audit = protocol_audit(all_event_rows, min_repetitions)
    audit_path = output_dir / "protocol_audit.csv"
    write_csv_rows(
        audit_path,
        audit,
        preferred=[
            "audit_type",
            "event_path",
            "role",
            "connection_state",
            "run_id",
            "pair_id",
            "board_id",
            "required_stratum",
            "stratum_check",
            "post_warmup_repetitions",
            "minimum_required",
            "repetition_check",
            "success_label_present",
            "warmup_label_present",
            "pair_id_present",
            "run_id_present",
            "board_id_present",
            "independent_pair_count",
            "failure_count",
            "failure_rate",
            "accepted_initiator_count",
            "accepted_responder_count",
            "integrity_check",
            "integrity_issue_count",
            "integrity_issues",
        ],
    )
    written.append(audit_path)
    return written, audit
