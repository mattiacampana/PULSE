"""Project battery life from explicit state and incremental-event energy inputs.

The low, nominal, and high cases are scenario sensitivity assumptions.  They
are deliberately kept separate from the sampling uncertainty of measured
incremental event energy.  This module supplies no physical defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Mapping, Sequence


INPUT_SCHEMA = "senswear.pulse.battery_projection.input.v1"
OUTPUT_SCHEMA = "senswear.pulse.battery_projection.result.v1"
ASSUMPTION_NAMES = ("low", "nominal", "high")
SECONDS_PER_DAY = 86_400.0


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _required(mapping: Mapping[str, object], key: str, context: str) -> object:
    if key not in mapping:
        raise ValueError(f"{context} requires explicit {key}")
    return mapping[key]


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _positive_number(value: object, context: str) -> float:
    result = _number(value, context)
    if result <= 0.0:
        raise ValueError(f"{context} must be positive")
    return result


def _nonnegative_number(value: object, context: str) -> float:
    result = _number(value, context)
    if result < 0.0:
        raise ValueError(f"{context} must be nonnegative")
    return result


def _description(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value.strip()


def _named_mapping(value: object, context: str) -> dict[str, object]:
    raw = _mapping(value, context)
    result: dict[str, object] = {}
    for key, item in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{context} keys must be nonempty strings")
        name = key.strip()
        if name in result:
            raise ValueError(f"{context} contains duplicate normalized key {name!r}")
        result[name] = item
    if not result:
        raise ValueError(f"{context} must not be empty")
    return result


def _event_statistics(raw_samples: object) -> dict[str, dict[str, object]]:
    samples_by_event = _named_mapping(
        raw_samples,
        "incremental_event_energy_samples_j",
    )
    result: dict[str, dict[str, object]] = {}
    for event_name, raw_values in samples_by_event.items():
        context = f"incremental_event_energy_samples_j.{event_name}"
        if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
            raise ValueError(f"{context} must be a nonempty array")
        values = [_number(value, f"{context}[{index}]") for index, value in enumerate(raw_values)]
        if not values:
            raise ValueError(f"{context} must be a nonempty array")

        sample_standard_deviation = statistics.stdev(values) if len(values) >= 2 else None
        standard_error = (
            sample_standard_deviation / math.sqrt(len(values))
            if sample_standard_deviation is not None
            else None
        )
        result[event_name] = {
            "event": event_name,
            "sample_count": len(values),
            "arithmetic_mean_incremental_energy_j": statistics.fmean(values),
            "sample_standard_deviation_j": sample_standard_deviation,
            "standard_error_of_mean_j": standard_error,
        }
    return result


def _project_assumption(
    name: str,
    raw_assumption: object,
    event_statistics: Mapping[str, Mapping[str, object]],
    expected_state_names: set[str] | None,
) -> tuple[dict[str, object], set[str]]:
    context = f"assumptions.{name}"
    assumption = _mapping(raw_assumption, context)
    description = _description(_required(assumption, "description", context), f"{context}.description")

    battery = _mapping(_required(assumption, "battery", context), f"{context}.battery")
    capacity_mah = _positive_number(
        _required(battery, "capacity_mAh", f"{context}.battery"),
        f"{context}.battery.capacity_mAh",
    )
    nominal_voltage_v = _positive_number(
        _required(battery, "nominal_voltage_v", f"{context}.battery"),
        f"{context}.battery.nominal_voltage_v",
    )
    efficiency = _positive_number(
        _required(battery, "efficiency", f"{context}.battery"),
        f"{context}.battery.efficiency",
    )
    if efficiency > 1.0:
        raise ValueError(f"{context}.battery.efficiency must be at most 1")

    states = _named_mapping(_required(assumption, "states", context), f"{context}.states")
    state_names = set(states)
    if expected_state_names is not None and state_names != expected_state_names:
        raise ValueError(
            f"{context}.states must name exactly the same states as the other assumptions"
        )

    state_contributions: list[dict[str, object]] = []
    total_state_duration_s = 0.0
    state_energy_j_per_day = 0.0
    for state_name in sorted(states):
        state_context = f"{context}.states.{state_name}"
        state = _mapping(states[state_name], state_context)
        duration_s = _nonnegative_number(
            _required(state, "duration_s_per_day", state_context),
            f"{state_context}.duration_s_per_day",
        )
        idle_power_w = _nonnegative_number(
            _required(state, "idle_power_w", state_context),
            f"{state_context}.idle_power_w",
        )
        contribution_j = duration_s * idle_power_w
        total_state_duration_s += duration_s
        state_energy_j_per_day += contribution_j
        state_contributions.append(
            {
                "state": state_name,
                "duration_s_per_day": duration_s,
                "idle_power_w": idle_power_w,
                "energy_j_per_day": contribution_j,
            }
        )

    if not math.isclose(
        total_state_duration_s,
        SECONDS_PER_DAY,
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"{context}.states durations must sum to exactly {SECONDS_PER_DAY:g} seconds per day; "
            f"got {total_state_duration_s:g}"
        )

    raw_event_counts = _named_mapping(
        _required(assumption, "event_counts_per_day", context),
        f"{context}.event_counts_per_day",
    )
    if set(raw_event_counts) != set(event_statistics):
        raise ValueError(
            f"{context}.event_counts_per_day must name exactly the measured event-energy classes"
        )

    event_contributions: list[dict[str, object]] = []
    event_energy_j_per_day = 0.0
    sampling_variance_j2_per_day = 0.0
    sampling_uncertainty_available = True
    for event_name in sorted(raw_event_counts):
        count_per_day = _nonnegative_number(
            raw_event_counts[event_name],
            f"{context}.event_counts_per_day.{event_name}",
        )
        event_statistic = event_statistics[event_name]
        arithmetic_mean_j = float(event_statistic["arithmetic_mean_incremental_energy_j"])
        standard_error_j = event_statistic["standard_error_of_mean_j"]
        contribution_j = count_per_day * arithmetic_mean_j
        event_energy_j_per_day += contribution_j

        contribution_standard_error_j: float | None
        if count_per_day == 0.0:
            contribution_standard_error_j = 0.0
        elif standard_error_j is None:
            contribution_standard_error_j = None
            sampling_uncertainty_available = False
        else:
            contribution_standard_error_j = count_per_day * float(standard_error_j)
            sampling_variance_j2_per_day += contribution_standard_error_j**2

        event_contributions.append(
            {
                "event": event_name,
                "count_per_day": count_per_day,
                "arithmetic_mean_incremental_energy_j": arithmetic_mean_j,
                "incremental_energy_j_per_day": contribution_j,
                "sampling_standard_error_j_per_day": contribution_standard_error_j,
            }
        )

    total_energy_j_per_day = state_energy_j_per_day + event_energy_j_per_day
    if total_energy_j_per_day <= 0.0:
        raise ValueError(f"{context} produces nonpositive total daily energy")

    battery_energy_j = 3.6 * capacity_mah * nominal_voltage_v * efficiency
    battery_life_days = battery_energy_j / total_energy_j_per_day

    sampling_standard_error_j_per_day: float | None
    sampling_standard_error_life_days: float | None
    if sampling_uncertainty_available:
        sampling_standard_error_j_per_day = math.sqrt(sampling_variance_j2_per_day)
        sampling_standard_error_life_days = (
            battery_energy_j
            * sampling_standard_error_j_per_day
            / (total_energy_j_per_day**2)
        )
    else:
        sampling_standard_error_j_per_day = None
        sampling_standard_error_life_days = None

    return (
        {
            "assumption": name,
            "description": description,
            "capacity_mAh": capacity_mah,
            "nominal_voltage_v": nominal_voltage_v,
            "efficiency": efficiency,
            "battery_energy_j": battery_energy_j,
            "state_energy_j_per_day": state_energy_j_per_day,
            "incremental_event_energy_j_per_day": event_energy_j_per_day,
            "total_energy_j_per_day": total_energy_j_per_day,
            "battery_life_days": battery_life_days,
            "battery_life_hours": battery_life_days * 24.0,
            "event_sampling_standard_error_j_per_day": sampling_standard_error_j_per_day,
            "battery_life_sampling_standard_error_days": sampling_standard_error_life_days,
            "state_contributions": state_contributions,
            "event_contributions": event_contributions,
        },
        state_names,
    )


def project_battery_life(document: Mapping[str, object]) -> dict[str, object]:
    """Validate an explicit projection document and return a serializable report."""

    schema = _required(document, "schema", "input")
    if schema != INPUT_SCHEMA:
        raise ValueError(f"input.schema must equal {INPUT_SCHEMA!r}")

    event_statistics = _event_statistics(
        _required(document, "incremental_event_energy_samples_j", "input")
    )
    assumptions = _mapping(_required(document, "assumptions", "input"), "assumptions")
    if set(assumptions) != set(ASSUMPTION_NAMES):
        raise ValueError(
            "assumptions must contain exactly the explicit low, nominal, and high cases"
        )

    results: list[dict[str, object]] = []
    expected_state_names: set[str] | None = None
    for name in ASSUMPTION_NAMES:
        result, expected_state_names = _project_assumption(
            name,
            assumptions[name],
            event_statistics,
            expected_state_names,
        )
        results.append(result)

    life_by_name = {str(row["assumption"]): float(row["battery_life_days"]) for row in results}
    minimum_name = min(life_by_name, key=life_by_name.__getitem__)
    maximum_name = max(life_by_name, key=life_by_name.__getitem__)

    return {
        "schema": OUTPUT_SCHEMA,
        "equations": {
            "daily_energy": (
                "sum_state(duration_s_per_day * idle_power_w) + "
                "sum_event(count_per_day * arithmetic_mean_incremental_energy_j)"
            ),
            "battery_energy": "3.6 * capacity_mAh * nominal_voltage_v * efficiency",
            "battery_life_days": "battery_energy_j / total_energy_j_per_day",
        },
        "semantics": {
            "sensitivity": (
                "low, nominal, and high are explicit scenario assumptions; their span is not a "
                "confidence or uncertainty interval"
            ),
            "sampling_uncertainty": (
                "one standard error propagated from arithmetic-mean event-energy samples in "
                "quadrature, assuming independent event classes; it excludes uncertainty in "
                "state power, durations, event counts, capacity, voltage, and efficiency"
            ),
        },
        "event_energy_statistics": [event_statistics[name] for name in sorted(event_statistics)],
        "assumptions": results,
        "sensitivity_summary": {
            "nominal_life_days": life_by_name["nominal"],
            "minimum_life_days": life_by_name[minimum_name],
            "minimum_assumption": minimum_name,
            "maximum_life_days": life_by_name[maximum_name],
            "maximum_assumption": maximum_name,
            "span_days": life_by_name[maximum_name] - life_by_name[minimum_name],
            "semantics": "scenario sensitivity only; not sampling uncertainty",
        },
    }


def write_projection_json(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_projection_csv(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "assumption",
        "description",
        "capacity_mAh",
        "nominal_voltage_v",
        "efficiency",
        "battery_energy_j",
        "state_energy_j_per_day",
        "incremental_event_energy_j_per_day",
        "total_energy_j_per_day",
        "battery_life_days",
        "battery_life_hours",
        "event_sampling_standard_error_j_per_day",
        "battery_life_sampling_standard_error_days",
        "sensitivity_semantics",
        "sampling_uncertainty_semantics",
    )
    semantics = _mapping(report["semantics"], "report.semantics")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for raw_row in report["assumptions"]:  # type: ignore[index]
            row = dict(_mapping(raw_row, "report.assumptions row"))
            writer.writerow(
                {
                    **{field: row.get(field) for field in fieldnames if field in row},
                    "sensitivity_semantics": semantics["sensitivity"],
                    "sampling_uncertainty_semantics": semantics["sampling_uncertainty"],
                }
            )


def run_battery_projection(
    input_json: Path,
    output_json: Path,
    output_csv: Path | None = None,
) -> dict[str, object]:
    """Load, validate, project, and write one explicit battery-life study."""

    with input_json.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    report = project_battery_life(_mapping(document, "input"))
    write_projection_json(output_json, report)
    if output_csv is not None:
        write_projection_csv(output_csv, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project battery life from explicit low/nominal/high assumptions",
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_battery_projection(args.input_json, args.output_json, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
