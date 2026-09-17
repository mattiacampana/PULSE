"""Project measured per-event energy from a complete per-node event ledger.

The ledger carries observation duration, opportunity totals, and source
provenance so missing event classes are detected before energy is summed.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from common import normalized_name, normalized_row, parse_float, read_csv_rows, write_csv_rows


SECONDS_PER_DAY = 86400.0
STATISTICS = ("mean", "median", "q1", "q3", "p95")
ACCEPTED_PREFIX = "accepted_"
ALLOWED_COUNT_BASES = {"deployment_observed", "empirical_rate_model"}
ALLOWED_EVENT_TUPLES = {
    ("local", "local", "connected"),
    ("no_contact", "local", "disconnected"),
    ("accepted_connected", "initiator", "connected"),
    ("accepted_connected", "responder", "connected"),
    ("accepted_discovery", "initiator", "disconnected"),
    ("accepted_discovery", "responder", "connected"),
}
WILDCARD_TOKENS = {"any", "all", "wildcard", "na", "n_a", "not_applicable", "unknown"}
REQUIRED_FIELDS = (
    "scenario",
    "seed",
    "node_id",
    "network_node_count",
    "observation_duration_s",
    "event_path",
    "role",
    "connection_state",
    "outcome",
    "remote_session_started",
    "count_in_observation",
    "local_opportunities",
    "selection_opportunities",
    "count_basis",
    "source_artifact",
    "source_artifact_sha256",
)
REQUIRED_SEMANTIC_CATEGORIES = {
    "local_success",
    "local_failure",
    "no_contact_success",
    "no_contact_failure",
    *{
        f"{path}_{suffix}"
        for path in ("accepted_connected", "accepted_discovery")
        for suffix in (
            "initiator_success_remote_started",
            "initiator_failure_pre_session",
            "initiator_failure_remote_started",
            "responder_success",
            "responder_failure",
        )
    },
}
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


def _close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-9)


def _audit_row(
    check: str,
    passed: bool,
    *,
    scenario: str = "",
    seed: str = "",
    node_id: str = "",
    detail: str = "",
) -> dict[str, object]:
    return {
        "check": check,
        "pass": int(passed),
        "scenario": scenario,
        "seed": seed,
        "node_id": node_id,
        "detail": detail,
    }


def _write_audit(output_dir: Path, audit: Iterable[Mapping[str, object]]) -> Path:
    path = output_dir / "daily_energy_audit.csv"
    write_csv_rows(
        path,
        audit,
        fieldnames=("check", "pass", "scenario", "seed", "node_id", "detail"),
    )
    return path


def _raise_if_issues(
    output_dir: Path,
    audit: list[dict[str, object]],
    issues: list[str],
) -> None:
    audit_path = _write_audit(output_dir, audit)
    if issues:
        preview = "; ".join(issues[:8])
        if len(issues) > 8:
            preview += f"; and {len(issues) - 8} more"
        raise ValueError(f"daily energy ledger audit failed: {preview}; see {audit_path}")


def _remote_session_value(value: str) -> str | None:
    normalized = normalized_name(value)
    if normalized in {"0", "false", "no"}:
        return "0"
    if normalized in {"1", "true", "yes"}:
        return "1"
    if normalized in {"na", "n_a", "not_applicable"}:
        return "na"
    return None


def _semantic_category(
    event_path: str,
    role: str,
    outcome: str,
    remote_started: str,
) -> tuple[str | None, str | None]:
    if outcome not in {"success", "failure"}:
        return None, "outcome must be success or failure"

    if event_path == "local" and role == "local":
        if remote_started != "na":
            return None, "local rows require remote_session_started=na"
        return f"local_{outcome}", None

    if event_path == "no_contact" and role == "local":
        if remote_started != "na":
            return None, "no_contact rows require remote_session_started=na"
        return f"no_contact_{outcome}", None

    if event_path.startswith(ACCEPTED_PREFIX) and role == "initiator":
        if outcome == "success":
            if remote_started != "1":
                return None, "successful accepted initiator rows require remote_session_started=1"
            return f"{event_path}_initiator_success_remote_started", None
        if remote_started == "0":
            return f"{event_path}_initiator_failure_pre_session", None
        if remote_started == "1":
            return f"{event_path}_initiator_failure_remote_started", None
        return None, "failed accepted initiator rows require remote_session_started=0 or 1"

    if event_path.startswith(ACCEPTED_PREFIX) and role == "responder":
        if remote_started != "1":
            return None, "accepted responder rows require remote_session_started=1"
        return f"{event_path}_responder_{outcome}", None

    return None, (
        "unsupported event_path/role; use local/local, no_contact/local, or "
        "accepted_connected|accepted_discovery with initiator|responder, and retain the "
        "attempted path on failures"
    )


def _expected_power_scope(event_path: str) -> str:
    return "pair_union" if event_path.startswith(ACCEPTED_PREFIX) else "event_gate"


def _summary_remote_session_value(row: Mapping[str, object]) -> str | None:
    value = _remote_session_value(str(row.get("remote_session_started", "")))
    if value is not None:
        return value
    if (
        not str(row.get("remote_session_started", "")).strip()
        and normalized_name(str(row.get("event_path", ""))) in {"local", "no_contact"}
    ):
        return "na"
    return None


def _validated_count_rows(
    counts_csv: Path,
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit: list[dict[str, object]] = []
    issues: list[str] = []
    try:
        raw_rows = [normalized_row(row) for row in read_csv_rows(counts_csv)]
    except (OSError, ValueError) as error:
        detail = f"cannot read count ledger: {error}"
        audit.append(_audit_row("input_readable", False, detail=detail))
        _raise_if_issues(output_dir, audit, [detail])
        raise AssertionError("unreachable")
    audit.append(_audit_row("input_readable", True, detail=str(counts_csv)))

    if not raw_rows:
        issues.append("the ledger contains no event rows")
        audit.append(_audit_row("input_nonempty", False, detail=issues[-1]))
        _raise_if_issues(output_dir, audit, issues)
    audit.append(_audit_row("input_nonempty", True, detail=f"rows={len(raw_rows)}"))

    parsed: list[dict[str, object]] = []
    seen_keys: dict[tuple[str, ...], int] = {}
    source_hash_cache: dict[Path, str] = {}
    for line_number, row in enumerate(raw_rows, 2):
        missing = [field for field in REQUIRED_FIELDS if not str(row.get(field, "")).strip()]
        if missing:
            detail = f"{counts_csv}:{line_number}: missing required fields {missing}"
            issues.append(detail)
            audit.append(_audit_row("row_schema", False, detail=detail))
            continue

        scenario = str(row["scenario"]).strip()
        seed = str(row["seed"]).strip()
        node_id = str(row["node_id"]).strip()
        event_path = normalized_name(str(row["event_path"]))
        role = normalized_name(str(row["role"]))
        connection_state = normalized_name(str(row["connection_state"]))
        outcome = normalized_name(str(row["outcome"]))
        remote_started = _remote_session_value(str(row["remote_session_started"]))
        count_basis = normalized_name(str(row["count_basis"]))
        source_artifact = str(row["source_artifact"]).strip()
        source_hash = str(row["source_artifact_sha256"]).strip().lower()
        count = parse_float(row["count_in_observation"])
        duration = parse_float(row["observation_duration_s"])
        local_opportunities = parse_float(row["local_opportunities"])
        selection_opportunities = parse_float(row["selection_opportunities"])
        network_node_count_value = parse_float(row["network_node_count"])

        row_issues: list[str] = []
        if event_path in WILDCARD_TOKENS or role in WILDCARD_TOKENS:
            row_issues.append("event_path and role must be exact, not wildcard labels")
        if connection_state in WILDCARD_TOKENS or not connection_state:
            row_issues.append("connection_state must be an exact measured state, not a wildcard")
        if (event_path, role, connection_state) not in ALLOWED_EVENT_TUPLES:
            allowed = ", ".join("/".join(values) for values in sorted(ALLOWED_EVENT_TUPLES))
            row_issues.append(
                f"unsupported event_path/role/connection_state tuple; allowed: {allowed}"
            )
        if remote_started is None:
            row_issues.append("remote_session_started must be 0, 1, or na")
        if count is None or count < 0.0:
            row_issues.append("count_in_observation must be finite and non-negative")
        if duration is None or duration <= 0.0:
            row_issues.append("observation_duration_s must be finite and positive")
        if local_opportunities is None or local_opportunities < 0.0:
            row_issues.append("local_opportunities must be finite and non-negative")
        if selection_opportunities is None or selection_opportunities < 0.0:
            row_issues.append("selection_opportunities must be finite and non-negative")
        if (
            network_node_count_value is None
            or network_node_count_value <= 0.0
            or not float(network_node_count_value).is_integer()
        ):
            row_issues.append("network_node_count must be a positive integer")
        if count_basis not in ALLOWED_COUNT_BASES:
            row_issues.append(
                "count_basis must be deployment_observed or empirical_rate_model"
            )
        if (
            count_basis == "deployment_observed"
            and count is not None
            and not float(count).is_integer()
        ):
            row_issues.append("deployment_observed counts must be whole event counts")
        if not SHA256_RE.fullmatch(source_hash):
            row_issues.append("source_artifact_sha256 must be 64 hexadecimal characters")
        source_path = Path(source_artifact)
        if not source_path.is_absolute():
            source_path = counts_csv.parent / source_path
        source_path = source_path.resolve()
        if source_path not in source_hash_cache:
            try:
                source_hash_cache[source_path] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            except OSError as error:
                row_issues.append(f"cannot read source_artifact {source_path}: {error}")
        actual_source_hash = source_hash_cache.get(source_path)
        if actual_source_hash is not None and actual_source_hash != source_hash:
            row_issues.append(
                f"source_artifact_sha256 does not match {source_path}; actual={actual_source_hash}"
            )

        semantic: str | None = None
        if remote_started is not None:
            semantic, semantic_issue = _semantic_category(
                event_path, role, outcome, remote_started
            )
            if semantic_issue:
                row_issues.append(semantic_issue)

        if not row_issues and count is not None and duration is not None:
            calculated_rate = count * SECONDS_PER_DAY / duration
            supplied_rate_text = str(row.get("count_per_day", "")).strip()
            if supplied_rate_text:
                supplied_rate = parse_float(supplied_rate_text)
                if supplied_rate is None or not _close(supplied_rate, calculated_rate):
                    row_issues.append(
                        "count_per_day, when supplied, must equal "
                        "count_in_observation*86400/observation_duration_s"
                    )

        key = (
            scenario,
            seed,
            node_id,
            event_path,
            role,
            connection_state,
            outcome,
            remote_started or "",
        )
        if key in seen_keys:
            row_issues.append(
                f"duplicate event category; first seen at line {seen_keys[key]}"
            )
        else:
            seen_keys[key] = line_number

        if row_issues:
            detail = f"{counts_csv}:{line_number}: " + "; ".join(row_issues)
            issues.append(detail)
            audit.append(
                _audit_row(
                    "row_validation",
                    False,
                    scenario=scenario,
                    seed=seed,
                    node_id=node_id,
                    detail=detail,
                )
            )
            continue

        assert count is not None
        assert duration is not None
        assert local_opportunities is not None
        assert selection_opportunities is not None
        assert network_node_count_value is not None
        assert remote_started is not None
        assert semantic is not None
        parsed_row: dict[str, object] = dict(row)
        parsed_row.update(
            {
                "event_path": event_path,
                "role": role,
                "connection_state": connection_state,
                "outcome": outcome,
                "remote_session_started": remote_started,
                "count_basis": count_basis,
                "source_artifact_resolved": str(source_path),
                "source_artifact_sha256": source_hash,
                "count_in_observation": count,
                "observation_duration_s": duration,
                "network_node_count": int(network_node_count_value),
                "local_opportunities": local_opportunities,
                "selection_opportunities": selection_opportunities,
                "count_per_day": count * SECONDS_PER_DAY / duration,
                "semantic_category": semantic,
                "source_line": line_number,
            }
        )
        parsed.append(parsed_row)
        audit.append(
            _audit_row(
                "row_validation",
                True,
                scenario=scenario,
                seed=seed,
                node_id=node_id,
                detail=f"line={line_number}; category={semantic}",
            )
        )

    _raise_if_issues(output_dir, audit, issues)

    by_node: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    by_run: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in parsed:
        node_key = (str(row["scenario"]), str(row["seed"]), str(row["node_id"]))
        by_node[node_key].append(row)
        by_run[node_key[:2]].append(row)

    for (scenario, seed, node_id), rows in sorted(by_node.items()):
        categories = {str(row["semantic_category"]) for row in rows}
        missing_categories = sorted(REQUIRED_SEMANTIC_CATEGORIES - categories)
        coverage_ok = not missing_categories
        coverage_detail = (
            "all required categories present"
            if coverage_ok
            else "missing explicit categories: " + ", ".join(missing_categories)
        )
        audit.append(
            _audit_row(
                "semantic_coverage",
                coverage_ok,
                scenario=scenario,
                seed=seed,
                node_id=node_id,
                detail=coverage_detail,
            )
        )
        if not coverage_ok:
            issues.append(f"{scenario}/{seed}/{node_id}: {coverage_detail}")

        durations = {float(row["observation_duration_s"]) for row in rows}
        local_totals = {float(row["local_opportunities"]) for row in rows}
        selection_totals = {float(row["selection_opportunities"]) for row in rows}
        consistency_ok = len(durations) == len(local_totals) == len(selection_totals) == 1
        audit.append(
            _audit_row(
                "node_metadata_consistency",
                consistency_ok,
                scenario=scenario,
                seed=seed,
                node_id=node_id,
                detail=(
                    "duration and opportunity totals are consistent"
                    if consistency_ok
                    else "duration/local_opportunities/selection_opportunities vary between rows"
                ),
            )
        )
        if not consistency_ok:
            issues.append(
                f"{scenario}/{seed}/{node_id}: inconsistent duration or opportunity totals"
            )
            continue

        local_count = sum(
            float(row["count_in_observation"])
            for row in rows
            if str(row["semantic_category"]).startswith("local_")
        )
        declared_local = next(iter(local_totals))
        local_ok = _close(local_count, declared_local)
        audit.append(
            _audit_row(
                "local_opportunity_reconciliation",
                local_ok,
                scenario=scenario,
                seed=seed,
                node_id=node_id,
                detail=f"event_sum={local_count:g}; declared={declared_local:g}",
            )
        )
        if not local_ok:
            issues.append(f"{scenario}/{seed}/{node_id}: local opportunity count does not reconcile")

        decision_count = sum(
            float(row["count_in_observation"])
            for row in rows
            if row["event_path"] == "no_contact"
            or (
                str(row["event_path"]).startswith(ACCEPTED_PREFIX)
                and row["role"] == "initiator"
            )
        )
        declared_selection = next(iter(selection_totals))
        selection_ok = _close(decision_count, declared_selection)
        audit.append(
            _audit_row(
                "selection_opportunity_reconciliation",
                selection_ok,
                scenario=scenario,
                seed=seed,
                node_id=node_id,
                detail=f"decision_sum={decision_count:g}; declared={declared_selection:g}",
            )
        )
        if not selection_ok:
            issues.append(
                f"{scenario}/{seed}/{node_id}: selection opportunity count does not reconcile"
            )

    for (scenario, seed), rows in sorted(by_run.items()):
        node_ids = {str(row["node_id"]) for row in rows}
        declared_node_counts = {int(row["network_node_count"]) for row in rows}
        node_count_ok = (
            len(declared_node_counts) == 1
            and next(iter(declared_node_counts)) == len(node_ids)
        )
        audit.append(
            _audit_row(
                "network_node_coverage",
                node_count_ok,
                scenario=scenario,
                seed=seed,
                detail=f"observed_nodes={len(node_ids)}; declared={sorted(declared_node_counts)}",
            )
        )
        if not node_count_ok:
            issues.append(f"{scenario}/{seed}: per-node ledger does not cover the declared network")

        durations = {float(row["observation_duration_s"]) for row in rows}
        duration_ok = len(durations) == 1
        audit.append(
            _audit_row(
                "network_duration_consistency",
                duration_ok,
                scenario=scenario,
                seed=seed,
                detail=f"durations={sorted(durations)}",
            )
        )
        if not duration_ok:
            issues.append(f"{scenario}/{seed}: observation duration differs between nodes")

        for accepted_path in ("accepted_connected", "accepted_discovery"):
            remote_initiator_count = sum(
                float(row["count_in_observation"])
                for row in rows
                if row["event_path"] == accepted_path
                and row["role"] == "initiator"
                and row["remote_session_started"] == "1"
            )
            responder_count = sum(
                float(row["count_in_observation"])
                for row in rows
                if row["event_path"] == accepted_path and row["role"] == "responder"
            )
            responder_ok = _close(remote_initiator_count, responder_count)
            audit.append(
                _audit_row(
                    "responder_session_reconciliation",
                    responder_ok,
                    scenario=scenario,
                    seed=seed,
                    detail=(
                        f"path={accepted_path}; "
                        f"remote_started_initiators={remote_initiator_count:g}; "
                        f"responders={responder_count:g}"
                    ),
                )
            )
            if not responder_ok:
                issues.append(
                    f"{scenario}/{seed}/{accepted_path}: responder count does not match "
                    "remote-started initiators"
                )

    _raise_if_issues(output_dir, audit, issues)
    return parsed, audit


def project_daily_energy(
    counts_csv: Path, power_summary_csv: Path, output_dir: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    count_rows, audit = _validated_count_rows(counts_csv, output_dir)
    try:
        summary_rows = [normalized_row(row) for row in read_csv_rows(power_summary_csv)]
    except (OSError, ValueError) as error:
        detail = f"cannot read power summary: {error}"
        audit.append(_audit_row("power_summary_readable", False, detail=detail))
        _raise_if_issues(output_dir, audit, [detail])
        raise AssertionError("unreachable")
    audit.append(_audit_row("power_summary_readable", True, detail=str(power_summary_csv)))
    energy_rows = [
        row
        for row in summary_rows
        if normalized_name(row.get("metric", "")) == "energy_incremental_j"
    ]
    issues: list[str] = []
    if not energy_rows:
        detail = f"{power_summary_csv}: no energy_incremental_j summary rows"
        issues.append(detail)
        audit.append(_audit_row("power_summary_nonempty", False, detail=detail))
        _raise_if_issues(output_dir, audit, issues)
    audit.append(
        _audit_row("power_summary_nonempty", True, detail=f"rows={len(energy_rows)}")
    )

    contributions: list[dict[str, object]] = []
    for count_row in count_rows:
        expected_scope = _expected_power_scope(str(count_row["event_path"]))
        matches = []
        for energy_row in energy_rows:
            if normalized_name(energy_row.get("event_path", "")) != count_row["event_path"]:
                continue
            if normalized_name(energy_row.get("role", "")) != count_row["role"]:
                continue
            if normalized_name(energy_row.get("connection_state", "")) != count_row["connection_state"]:
                continue
            if normalized_name(energy_row.get("outcome", "")) != count_row["outcome"]:
                continue
            if _summary_remote_session_value(energy_row) != count_row["remote_session_started"]:
                continue
            if normalized_name(energy_row.get("measurement_scope", "")) != expected_scope:
                continue
            if normalized_name(energy_row.get("baseline_source", "")) not in {
                "baseline_metadata",
                "manual_numeric",
            }:
                continue
            matches.append(energy_row)

        count = float(count_row["count_in_observation"])
        match_ok = len(matches) == 1 or (count == 0.0 and len(matches) == 0)
        detail = (
            f"matches={len(matches)}; category={count_row['event_path']}/"
            f"{count_row['role']}/{count_row['connection_state']}/{count_row['outcome']}/"
            f"remote_session_started={count_row['remote_session_started']}/"
            f"measurement_scope={expected_scope}"
            "/baseline_source=baseline_metadata|manual_numeric"
        )
        audit.append(
            _audit_row(
                "power_category_match",
                match_ok,
                scenario=str(count_row["scenario"]),
                seed=str(count_row["seed"]),
                node_id=str(count_row["node_id"]),
                detail=detail,
            )
        )
        if not match_ok:
            issues.append(
                f"line {count_row['source_line']}: expected exactly one power summary for a "
                f"positive category (or zero/one for an explicit zero); {detail}"
            )
            continue

        energy = matches[0] if matches else None
        output: dict[str, object] = {
            key: value for key, value in count_row.items() if key != "source_line"
        }
        statistics_ok = True
        if energy is not None:
            sample_count = parse_float(energy.get("n"))
            if count > 0.0 and (sample_count is None or sample_count <= 0.0):
                statistics_ok = False
                issues.append(
                    f"line {count_row['source_line']}: matched power summary has no positive n"
                )
            for statistic in STATISTICS:
                per_event = parse_float(energy.get(statistic))
                if count > 0.0 and per_event is None:
                    statistics_ok = False
                    issues.append(
                        f"line {count_row['source_line']}: matched power summary is missing {statistic}"
                    )
                output[f"energy_{statistic}_j_per_event"] = per_event
                if statistic == "mean":
                    output["expected_incremental_energy_j_per_day"] = (
                        per_event * float(count_row["count_per_day"])
                        if per_event is not None
                        else 0.0
                    )
                else:
                    output[
                        "incremental_energy_plugin_sensitivity_using_event_"
                        f"{statistic}_j_per_day"
                    ] = (
                        per_event * float(count_row["count_per_day"])
                        if per_event is not None
                        else 0.0
                    )
            output["power_summary_n"] = energy.get("n", "")
            output["power_measurement_scope"] = energy.get("measurement_scope", "")
        else:
            for statistic in STATISTICS:
                output[f"energy_{statistic}_j_per_event"] = ""
                if statistic == "mean":
                    output["expected_incremental_energy_j_per_day"] = 0.0
                else:
                    output[
                        "incremental_energy_plugin_sensitivity_using_event_"
                        f"{statistic}_j_per_day"
                    ] = 0.0
            output["power_summary_n"] = 0
            output["power_measurement_scope"] = expected_scope
        audit.append(
            _audit_row(
                "power_statistics_complete",
                statistics_ok,
                scenario=str(count_row["scenario"]),
                seed=str(count_row["seed"]),
                node_id=str(count_row["node_id"]),
                detail=(
                    "mean and all four distribution statistics present"
                    if statistics_ok and energy is not None
                    else "explicit zero; no power statistic required"
                    if statistics_ok
                    else "positive category lacks n or a required statistic"
                ),
            )
        )
        output["counts_source_file"] = str(counts_csv)
        output["power_source_file"] = str(power_summary_csv)
        contributions.append(output)

    _raise_if_issues(output_dir, audit, issues)

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in contributions:
        grouped[(str(row["scenario"]), str(row["seed"]), str(row["node_id"]))].append(row)
    totals: list[dict[str, object]] = []
    for (scenario, seed, node_id), rows in sorted(grouped.items()):
        total: dict[str, object] = {
            "scenario": scenario,
            "seed": seed,
            "node_id": node_id,
            "observation_duration_s": rows[0]["observation_duration_s"],
            "event_categories": len(rows),
            "source_artifact_count": len(
                {str(row["source_artifact_sha256"]) for row in rows}
            ),
        }
        total["expected_incremental_energy_j_per_day"] = sum(
            float(row["expected_incremental_energy_j_per_day"]) for row in rows
        )
        for statistic in ("median", "q1", "q3", "p95"):
            field = (
                "incremental_energy_plugin_sensitivity_using_event_"
                f"{statistic}_j_per_day"
            )
            total[field] = sum(float(row[field]) for row in rows)
        total["note"] = (
            "expected value uses event rates times sample mean energy; plugin sensitivity "
            "columns sum rates times the named marginal event statistic and are neither "
            "daily-energy quantiles nor confidence bounds"
        )
        totals.append(total)

    write_csv_rows(
        output_dir / "daily_energy_contributions.csv",
        contributions,
        preferred=[
            "scenario",
            "seed",
            "node_id",
            "semantic_category",
            "event_path",
            "role",
            "connection_state",
            "outcome",
            "remote_session_started",
            "count_in_observation",
            "observation_duration_s",
            "count_per_day",
            "count_basis",
            "power_measurement_scope",
            "energy_mean_j_per_event",
            "expected_incremental_energy_j_per_day",
            "energy_median_j_per_event",
            "incremental_energy_plugin_sensitivity_using_event_median_j_per_day",
            "energy_q1_j_per_event",
            "incremental_energy_plugin_sensitivity_using_event_q1_j_per_day",
            "energy_q3_j_per_event",
            "incremental_energy_plugin_sensitivity_using_event_q3_j_per_day",
            "energy_p95_j_per_event",
            "incremental_energy_plugin_sensitivity_using_event_p95_j_per_day",
            "power_summary_n",
            "source_artifact",
            "source_artifact_resolved",
            "source_artifact_sha256",
            "counts_source_file",
            "power_source_file",
        ],
    )
    write_csv_rows(
        output_dir / "daily_energy_projection.csv",
        totals,
        preferred=[
            "scenario",
            "seed",
            "node_id",
            "observation_duration_s",
            "event_categories",
            "source_artifact_count",
            "expected_incremental_energy_j_per_day",
            "incremental_energy_plugin_sensitivity_using_event_median_j_per_day",
            "incremental_energy_plugin_sensitivity_using_event_q1_j_per_day",
            "incremental_energy_plugin_sensitivity_using_event_q3_j_per_day",
            "incremental_energy_plugin_sensitivity_using_event_p95_j_per_day",
            "note",
        ],
    )
    _write_audit(output_dir, audit)
    return contributions, totals
