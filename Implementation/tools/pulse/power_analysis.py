"""Decode GPIO markers and integrate synchronized power-analyzer captures."""

from __future__ import annotations

import bisect
import hashlib
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from common import (
    PULSE_BUILD_CONTEXT_FIELDS,
    PULSE_EVENT_CONFIGURATION_FIELDS,
    normalized_name,
    normalized_row,
    parse_bool,
    parse_float,
    read_csv_rows,
    sha256_file,
    write_csv_rows,
)
from summarize import ACCEPTED_PATHS, accepted_session_integrity


DEFAULT_STAGE_NAMES = {
    0: "event_unspecified",
    1: "selection",
    2: "encoder",
    3: "head_forward_backward",
    4: "serialization_deserialization",
    5: "radio_transfer",
    6: "agreement_normalization_mixing_utility",
    7: "ble_control_scan_connect",
}

TIME_ALIASES = ("time_s", "timestamp_s", "time", "seconds")
CURRENT_ALIASES = ("current_a", "current", "amps", "current_amp")
POWER_ALIASES = ("power_w", "power", "watts")
VOLTAGE_ALIASES = ("voltage_v", "voltage", "volts")
GATE_ALIASES = (
    "event_gate",
    "marker_event_gate",
    "gate",
    "pulse_gate",
    "gpio0",
    "p1_14",
    "gpio0_p1_14",
)
STAGE_BIT_ALIASES = {
    0: (
        "stage_bit0",
        "marker_bit0",
        "stage_0",
        "stage0",
        "stage_lsb",
        "gpio1",
        "p1_13",
        "gpio1_p1_13",
    ),
    1: (
        "stage_bit1",
        "marker_bit1",
        "stage_1",
        "stage1",
        "gpio2",
        "p1_12",
        "gpio2_p1_12",
    ),
    2: (
        "stage_bit2",
        "marker_bit2",
        "stage_2",
        "stage2",
        "stage_msb",
        "gpio3",
        "p1_11",
        "gpio3_p1_11",
    ),
}

EVENT_FIELDS_TO_COPY = (
    "run_id",
    "pair_id",
    "board_id",
    "trial_id",
    "exchange_id",
    "remote_session_started",
    "event_path",
    "connection_state",
    "post_warmup",
    "warmup",
    "success",
    "status",
    "error_code",
    "event_index",
    "marker_event_index",
    *PULSE_BUILD_CONTEXT_FIELDS,
    *PULSE_EVENT_CONFIGURATION_FIELDS,
)

MISSING_IDENTITY_VALUES = {
    "",
    "unknown",
    "unset",
    "na",
    "n_a",
    "operator_required",
}


@dataclass
class EventWindow:
    sequence: int
    first_high_index: int
    last_high_index: int
    start_s: float
    end_s: float
    truncated_start: bool
    truncated_end: bool


@dataclass
class PowerChannel:
    role: str
    source_file: str
    source_sha256: str
    times_s: list[float]
    currents_a: list[float]
    voltages_v: list[float]
    powers_w: list[float]
    gates: list[bool]
    stage_codes: list[int]
    baseline_power_w: float
    baseline_method: str
    diagnostic_gate_low_sample_median_power_w: float | None
    baseline_source: str
    baseline_metadata_file: str
    baseline_metadata_sha256: str
    baseline_capture_source_file: str


def _first_present(headers: set[str], role: str, aliases: Sequence[str], allow_generic: bool = True) -> str | None:
    role = normalized_name(role)
    candidates: list[str] = []
    if role and role != "auto":
        for alias in aliases:
            candidates.extend((f"{role}_{alias}", f"{alias}_{role}"))
    if allow_generic:
        candidates.extend(aliases)
    return next((candidate for candidate in candidates if candidate in headers), None)


def _parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        role, raw_path = spec.split("=", 1)
        role = normalized_name(role)
        if not role:
            raise ValueError(f"invalid power input role in {spec!r}")
        return role, Path(raw_path)
    return "auto", Path(spec)


def parse_baseline_specs(specs: Iterable[str]) -> tuple[float | None, dict[str, float]]:
    global_value: float | None = None
    per_role: dict[str, float] = {}
    for spec in specs:
        if "=" in spec:
            role, raw_value = spec.split("=", 1)
            value = parse_float(raw_value)
            if value is None or value < 0.0:
                raise ValueError(f"invalid baseline power {raw_value!r}")
            per_role[normalized_name(role)] = value
        else:
            value = parse_float(spec)
            if value is None or value < 0.0:
                raise ValueError(f"invalid baseline power {spec!r}")
            global_value = value
    return global_value, per_role


def load_baseline_metadata(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    rows = [normalized_row(row) for row in read_csv_rows(path)]
    bindings: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, 2):
        if normalized_name(row.get("capture_mode", "")) != "baseline_only":
            continue
        role = normalized_name(row.get("role", ""))
        value = parse_float(row.get("baseline_power_w"))
        method = normalized_name(row.get("baseline_method", ""))
        source_sha256 = str(row.get("source_sha256", "")).strip().lower()
        if role not in {"initiator", "responder"} or value is None or value < 0.0:
            raise ValueError(
                f"{path}:{line_number}: invalid baseline-only role or baseline_power_w"
            )
        if not method.startswith("matched_baseline_capture_integrated_mean"):
            raise ValueError(
                f"{path}:{line_number}: paper baseline must use integrated mean energy/time; "
                f"found baseline_method={row.get('baseline_method', '')!r}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise ValueError(
                f"{path}:{line_number}: baseline metadata lacks a concrete raw-power "
                "source_sha256; regenerate it from the archived capture"
            )
        if role in bindings:
            raise ValueError(f"{path}: duplicate baseline-only metadata row for {role}")
        bindings[role] = row
    if not bindings:
        raise ValueError(f"{path}: no baseline_only rows found")
    return bindings, hashlib.sha256(path.read_bytes()).hexdigest()


def _marker_high(value: object, threshold: float, active_low: bool) -> bool | None:
    numeric = parse_float(value)
    if numeric is not None:
        high = numeric >= threshold
    else:
        logical = parse_bool(value)
        if logical is None:
            return None
        high = logical
    return not high if active_low else high


def _roles_in_file(rows: list[dict[str, str]], requested_role: str) -> list[str]:
    headers = set(normalized_row(rows[0])) if rows else set()
    if requested_role != "auto":
        return [requested_role]
    roles = []
    for role in ("initiator", "responder"):
        if _first_present(headers, role, CURRENT_ALIASES, allow_generic=False) or _first_present(
            headers, role, POWER_ALIASES, allow_generic=False
        ):
            roles.append(role)
    if not roles:
        raise ValueError(
            "an unlabelled --power-input must contain initiator_current_a and/or "
            "responder_current_a; use ROLE=FILE for a generic current_a column"
        )
    return roles


def load_power_channels(
    specs: Iterable[str],
    voltage_v: float | None,
    baseline_specs: Iterable[str],
    marker_threshold: float = 0.5,
    active_low: bool = False,
    automatic_baseline_mode: str | None = None,
    baseline_bindings: Mapping[str, Mapping[str, str]] | None = None,
    baseline_metadata_file: str = "",
    baseline_metadata_sha256: str = "",
) -> list[PowerChannel]:
    if automatic_baseline_mode not in {None, "baseline_only", "diagnostic_intrace"}:
        raise ValueError(f"invalid automatic baseline mode {automatic_baseline_mode!r}")
    global_baseline, role_baselines = parse_baseline_specs(baseline_specs)
    channels: list[PowerChannel] = []
    for spec in specs:
        requested_role, path = _parse_input_spec(spec)
        rows = read_csv_rows(path)
        if len(rows) < 2:
            raise ValueError(f"{path}: power CSV requires at least two samples")
        normalized_rows = [normalized_row(row) for row in rows]
        headers = set(normalized_rows[0])
        time_column = _first_present(headers, "", TIME_ALIASES)
        if time_column is None:
            raise ValueError(f"{path}: expected a time_s column")

        for role in _roles_in_file(rows, requested_role):
            current_column = _first_present(headers, role, CURRENT_ALIASES)
            power_column = _first_present(headers, role, POWER_ALIASES)
            voltage_column = _first_present(headers, role, VOLTAGE_ALIASES)
            gate_column = _first_present(headers, role, GATE_ALIASES)
            bit_columns = {
                bit: _first_present(headers, role, aliases) for bit, aliases in STAGE_BIT_ALIASES.items()
            }
            if current_column is None and power_column is None:
                raise ValueError(f"{path}: no current_a or power_w column found for role {role!r}")
            if gate_column is None:
                raise ValueError(
                    f"{path}: no event-gate column found for role {role!r}; expected event_gate/GPIO0/P1.14"
                )
            missing_bits = [bit for bit, column in bit_columns.items() if column is None]
            if missing_bits:
                raise ValueError(
                    f"{path}: missing stage bit columns {missing_bits} for role {role!r}; "
                    "expected GPIO1/P1.13, GPIO2/P1.12, GPIO3/P1.11 LSB-first"
                )

            times: list[float] = []
            currents: list[float] = []
            voltages: list[float] = []
            powers: list[float] = []
            gates: list[bool] = []
            stages: list[int] = []
            for row_number, row in enumerate(normalized_rows, 2):
                time_s = parse_float(row.get(time_column))
                if time_s is None:
                    raise ValueError(f"{path}:{row_number}: invalid time value")
                sample_voltage = parse_float(row.get(voltage_column)) if voltage_column else voltage_v
                sample_current = parse_float(row.get(current_column)) if current_column else None
                sample_power = parse_float(row.get(power_column)) if power_column else None
                if sample_power is None:
                    if sample_current is None or sample_voltage is None:
                        raise ValueError(
                            f"{path}:{row_number}: current data require voltage_v in the CSV or --voltage-v"
                        )
                    sample_power = sample_current * sample_voltage
                if sample_voltage is None:
                    if sample_current is None or sample_current == 0:
                        raise ValueError(f"{path}:{row_number}: cannot derive voltage")
                    sample_voltage = sample_power / sample_current
                if sample_current is None:
                    sample_current = sample_power / sample_voltage
                gate_value = _marker_high(row.get(gate_column), marker_threshold, active_low)
                bit_values = [
                    _marker_high(row.get(bit_columns[bit]), marker_threshold, active_low)
                    for bit in range(3)
                ]
                if gate_value is None or any(value is None for value in bit_values):
                    raise ValueError(f"{path}:{row_number}: non-numeric GPIO marker")
                code = 0
                for bit, value in enumerate(bit_values):
                    code |= int(bool(value)) << bit
                times.append(time_s)
                currents.append(sample_current)
                voltages.append(sample_voltage)
                powers.append(sample_power)
                gates.append(gate_value)
                stages.append(code)

            if any(right <= left for left, right in zip(times, times[1:])):
                raise ValueError(f"{path}: time_s must be strictly increasing")
            binding = (baseline_bindings or {}).get(role)
            if baseline_bindings is not None and binding is None:
                raise ValueError(
                    f"{baseline_metadata_file}: matched baseline metadata has no {role} row"
                )
            baseline = (
                parse_float(binding.get("baseline_power_w"))
                if binding is not None
                else role_baselines.get(role, global_baseline)
            )
            low_samples = [power for power, gate in zip(powers, gates) if not gate]
            diagnostic_median = statistics.median(low_samples) if low_samples else None
            if baseline is None:
                if automatic_baseline_mode is None:
                    raise ValueError(
                        f"{path}: non-baseline analysis requires an explicit matched-baseline "
                        f"value for {role}; pass --baseline-power-w {role}=VALUE from a "
                        "separate --baseline-only capture, or use "
                        "--allow-gate-low-baseline-diagnostic for non-paper diagnostics"
                    )
                if not low_samples:
                    raise ValueError(
                        f"{path}: no gate-low samples for automatic baseline; pass --baseline-power-w {role}=VALUE"
                    )
                if automatic_baseline_mode == "baseline_only":
                    capture_duration_s = times[-1] - times[0]
                    baseline_energy_j = sum(
                        (right_t - left_t) * (left_p + right_p) / 2.0
                        for left_t, right_t, left_p, right_p in zip(
                            times, times[1:], powers, powers[1:]
                        )
                    )
                    baseline = baseline_energy_j / capture_duration_s
                    baseline_method = (
                        f"matched_baseline_capture_integrated_mean_n_{len(powers)}"
                    )
                else:
                    baseline = statistics.median(low_samples)
                    baseline_method = (
                        f"diagnostic_intrace_gate_low_median_n_{len(low_samples)}"
                    )
                baseline_source = (
                    "matched_baseline_capture"
                    if automatic_baseline_mode == "baseline_only"
                    else "diagnostic_intrace_gate_low"
                )
                bound_capture_source = str(path) if automatic_baseline_mode == "baseline_only" else ""
            elif binding is not None:
                baseline_method = "matched_baseline_metadata_integrated_mean"
                baseline_source = "baseline_metadata"
                bound_capture_source = str(binding.get("source_file", ""))
            else:
                baseline_method = "explicit_matched_baseline_manual"
                baseline_source = "manual_numeric"
                bound_capture_source = ""
            channels.append(
                PowerChannel(
                    role=role,
                    source_file=str(path),
                    source_sha256=sha256_file(path),
                    times_s=times,
                    currents_a=currents,
                    voltages_v=voltages,
                    powers_w=powers,
                    gates=gates,
                    stage_codes=stages,
                    baseline_power_w=baseline,
                    baseline_method=baseline_method,
                    diagnostic_gate_low_sample_median_power_w=diagnostic_median,
                    baseline_source=baseline_source,
                    baseline_metadata_file=baseline_metadata_file if binding is not None else "",
                    baseline_metadata_sha256=(
                        baseline_metadata_sha256 if binding is not None else ""
                    ),
                    baseline_capture_source_file=bound_capture_source,
                )
            )
    return channels


def find_event_windows(channel: PowerChannel) -> list[EventWindow]:
    windows: list[EventWindow] = []
    index = 0
    size = len(channel.gates)
    while index < size:
        if not channel.gates[index]:
            index += 1
            continue
        first = index
        while index + 1 < size and channel.gates[index + 1]:
            index += 1
        last = index
        truncated_start = first == 0
        truncated_end = last == size - 1
        start_s = channel.times_s[first] if truncated_start else (
            channel.times_s[first - 1] + channel.times_s[first]
        ) / 2.0
        end_s = channel.times_s[last] if truncated_end else (
            channel.times_s[last] + channel.times_s[last + 1]
        ) / 2.0
        windows.append(
            EventWindow(
                sequence=len(windows) + 1,
                first_high_index=first,
                last_high_index=last,
                start_s=start_s,
                end_s=end_s,
                truncated_start=truncated_start,
                truncated_end=truncated_end,
            )
        )
        index += 1
    return windows


def _interpolate(times: Sequence[float], values: Sequence[float], at_s: float) -> float:
    if at_s < times[0] or at_s > times[-1]:
        raise ValueError("integration window falls outside the power capture")
    index = bisect.bisect_left(times, at_s)
    if index < len(times) and times[index] == at_s:
        return values[index]
    if index == 0 or index == len(times):
        return values[min(index, len(values) - 1)]
    left_t, right_t = times[index - 1], times[index]
    fraction = (at_s - left_t) / (right_t - left_t)
    return values[index - 1] * (1.0 - fraction) + values[index] * fraction


def integrate_series(
    times: Sequence[float], values: Sequence[float], start_s: float, end_s: float
) -> tuple[float, float, float]:
    """Return integral, mean, and peak over a clipped piecewise-linear trace."""

    if end_s <= start_s:
        raise ValueError("integration window has non-positive duration")
    inner_start = bisect.bisect_right(times, start_s)
    inner_end = bisect.bisect_left(times, end_s)
    clipped_times = [start_s, *times[inner_start:inner_end], end_s]
    clipped_values = [
        _interpolate(times, values, start_s),
        *values[inner_start:inner_end],
        _interpolate(times, values, end_s),
    ]
    integral = sum(
        (right_t - left_t) * (left_v + right_v) / 2.0
        for left_t, right_t, left_v, right_v in zip(
            clipped_times, clipped_times[1:], clipped_values, clipped_values[1:]
        )
    )
    duration = end_s - start_s
    return integral, integral / duration, max(clipped_values)


def _stage_intervals(channel: PowerChannel, window: EventWindow) -> list[tuple[int, int, float, float]]:
    intervals: list[tuple[int, int, float, float]] = []
    first = window.first_high_index
    cursor = first
    sequence = 1
    while cursor <= window.last_high_index:
        code = channel.stage_codes[cursor]
        last = cursor
        while last + 1 <= window.last_high_index and channel.stage_codes[last + 1] == code:
            last += 1
        start_s = window.start_s if cursor == first else (
            channel.times_s[cursor - 1] + channel.times_s[cursor]
        ) / 2.0
        end_s = window.end_s if last == window.last_high_index else (
            channel.times_s[last] + channel.times_s[last + 1]
        ) / 2.0
        intervals.append((sequence, code, start_s, end_s))
        sequence += 1
        cursor = last + 1
    return intervals


def load_stage_names(metadata_csv: Path | None, overrides: Iterable[str]) -> dict[int, str]:
    names = dict(DEFAULT_STAGE_NAMES)
    if metadata_csv and metadata_csv.exists():
        for raw_row in read_csv_rows(metadata_csv):
            row = normalized_row(raw_row)
            for key, value in row.items():
                match = None
                for pattern in (r"marker_stage_(\d+)", r"stage_(\d+)_name"):
                    match = re.fullmatch(pattern, key)
                    if match:
                        break
                if match and value:
                    names[int(match.group(1))] = normalized_name(value)
            key = normalized_name(row.get("key", ""))
            if key.startswith("marker_stage_") and row.get("value", ""):
                try:
                    names[int(key.rsplit("_", 1)[1])] = normalized_name(row["value"])
                except ValueError:
                    pass
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"invalid stage mapping {override!r}; expected CODE=NAME")
        raw_code, value = override.split("=", 1)
        code = int(raw_code, 0)
        if not 0 <= code <= 7:
            raise ValueError(f"stage code must be in [0, 7], got {code}")
        names[code] = normalized_name(value)
    return names


def _board_ids_for_role(
    event_rows: list[dict[str, str]],
    metadata_rows: list[dict[str, str]],
    role: str,
) -> set[str]:
    """Return board IDs that the firmware assigned to a physical analyzer role."""

    board_ids: set[str] = set()
    for row in [*metadata_rows, *event_rows]:
        if normalized_name(row.get("role", "")) != role:
            continue
        board_id = normalized_name(row.get("board_id", ""))
        if board_id and board_id not in {"unknown", "na"}:
            board_ids.add(board_id)
    return board_ids


def _serial_candidates(
    event_rows: list[dict[str, str]],
    role: str,
    board_ids: set[str],
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for row in event_rows:
        event_role = normalized_name(row.get("role", ""))
        if event_role == role:
            candidates.append(row)
            continue

        # The initiator board executes the local and no-contact paths, but the
        # firmware correctly describes their algorithmic role as `local`.
        # Associate those records by immutable board identity; never treat an
        # arbitrary local record as belonging to the responder channel.
        if role != "initiator" or event_role != "local":
            continue
        event_path = normalized_name(row.get("event_path", ""))
        board_id = normalized_name(row.get("board_id", ""))
        if event_path in {"local", "no_contact"} and board_id in board_ids:
            candidates.append(row)
    return sorted(candidates, key=lambda row: int(parse_float(row.get("record_index")) or 0))


def join_windows_to_events(
    windows: list[EventWindow],
    event_rows: list[dict[str, str]],
    role: str,
    allow_unmatched: bool,
    board_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    candidates = _serial_candidates(event_rows, role, board_ids or set())
    if not candidates:
        if event_rows and not allow_unmatched:
            roles = sorted({row.get("role", "") for row in event_rows})
            raise ValueError(f"no PULSE_EVENT records match power role {role!r}; serial roles are {roles}")
        return [{} for _ in windows]

    identities: set[tuple[str, str, str]] = set()
    for row in candidates:
        identity = tuple(
            str(row.get(field, "")).strip()
            for field in ("run_id", "pair_id", "board_id")
        )
        if any(value.lower() in MISSING_IDENTITY_VALUES for value in identity):
            raise ValueError(
                f"{role}: every serial event used for a power join requires concrete "
                "run_id, pair_id, and board_id"
            )
        identities.add(identity)
    if len(identities) != 1:
        raise ValueError(
            f"{role}: a power channel may join exactly one run/pair/board identity; "
            f"found {sorted(identities)}"
        )

    indexed_rows: list[tuple[int, dict[str, str]]] = []
    for row in candidates:
        raw_index = row.get("marker_event_index") or row.get("event_index", "")
        value = parse_float(raw_index)
        if value is None or not value.is_integer() or value < 0:
            raise ValueError(
                f"{role}: every serial event requires a non-negative integer "
                "marker_event_index or event_index"
            )
        indexed_rows.append((int(value), row))
    indices = sorted(value for value, _ in indexed_rows)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{role}: serial event indices are not unique: {indices}")
    expected_indices = list(range(indices[0], indices[-1] + 1))
    if indices != expected_indices:
        raise ValueError(
            f"{role}: serial event indices must be contiguous within the capture; "
            f"observed {indices[0]}..{indices[-1]} with gaps"
        )
    ordered_candidates = [row for _, row in sorted(indexed_rows, key=lambda item: item[0])]
    joined = [
        ordered_candidates[index] if index < len(ordered_candidates) else {}
        for index in range(len(windows))
    ]

    unmatched_windows = sum(not row for row in joined)
    matched_ids = {id(row) for row in joined if row}
    unmatched_records = sum(id(row) not in matched_ids for row in candidates)
    if unmatched_windows or unmatched_records:
        raise ValueError(
            f"{role}: {len(windows)} GPIO event windows do not match {len(candidates)} serial events "
            f"({unmatched_windows} windows and {unmatched_records} records unmatched); "
            "fix capture alignment. --allow-unmatched never permits an ambiguous partial "
            "chronological join"
        )
    return joined


def _metric_row(
    channel: PowerChannel,
    window: EventWindow,
    event: Mapping[str, str],
) -> dict[str, object]:
    energy_j, average_w, peak_w = integrate_series(
        channel.times_s, channel.powers_w, window.start_s, window.end_s
    )
    charge_c, average_a, peak_a = integrate_series(
        channel.times_s, channel.currents_a, window.start_s, window.end_s
    )
    event_role = normalized_name(event.get("role", "")) or channel.role
    row: dict[str, object] = {
        "role": event_role,
        "logical_role": event_role,
        "measurement_scope": "event_gate",
        "serial_event_present": int(bool(event)),
        "power_channel_role": channel.role,
        "power_event_sequence": window.sequence,
        "start_time_s": window.start_s,
        "end_time_s": window.end_s,
        "duration_s": window.end_s - window.start_s,
        "energy_total_j": energy_j,
        "energy_incremental_j": energy_j
        - channel.baseline_power_w * (window.end_s - window.start_s),
        "charge_c": charge_c,
        "average_current_a": average_a,
        "peak_current_a": peak_a,
        "average_power_w": average_w,
        "peak_power_w": peak_w,
        "baseline_power_w": channel.baseline_power_w,
        "baseline_method": channel.baseline_method,
        "diagnostic_gate_low_sample_median_power_w": (
            channel.diagnostic_gate_low_sample_median_power_w
        ),
        "baseline_source": channel.baseline_source,
        "baseline_metadata_file": channel.baseline_metadata_file,
        "baseline_metadata_sha256": channel.baseline_metadata_sha256,
        "baseline_capture_source_file": channel.baseline_capture_source_file,
        "nominal_sample_rate_hz": 1.0
        / statistics.median(
            right - left for left, right in zip(channel.times_s, channel.times_s[1:])
        ),
        "mean_voltage_v": statistics.fmean(channel.voltages_v),
        "truncated_start": int(window.truncated_start),
        "truncated_end": int(window.truncated_end),
        "join_status": "matched" if event else "unmatched_power_window",
        "source_file": channel.source_file,
        "source_sha256": channel.source_sha256,
    }
    for field in EVENT_FIELDS_TO_COPY:
        if event.get(field, "") != "":
            row[field] = event[field]
    return row


def _capture_metadata(
    channel: PowerChannel,
    event_window_count: int,
    capture_mode: str,
    build_context: Mapping[str, str],
) -> dict[str, object]:
    intervals = [right - left for left, right in zip(channel.times_s, channel.times_s[1:])]
    median_interval = statistics.median(intervals)
    row: dict[str, object] = {
        "role": channel.role,
        "capture_mode": capture_mode,
        "event_window_count": event_window_count,
        "all_event_gates_low": int(not any(channel.gates)),
        "source_file": channel.source_file,
        "source_sha256": channel.source_sha256,
        "sample_count": len(channel.times_s),
        "capture_start_s": channel.times_s[0],
        "capture_end_s": channel.times_s[-1],
        "capture_duration_s": channel.times_s[-1] - channel.times_s[0],
        "nominal_sample_rate_hz": 1.0 / median_interval,
        "minimum_sample_interval_s": min(intervals),
        "maximum_sample_interval_s": max(intervals),
        "mean_voltage_v": statistics.fmean(channel.voltages_v),
        "minimum_voltage_v": min(channel.voltages_v),
        "maximum_voltage_v": max(channel.voltages_v),
        "baseline_power_w": channel.baseline_power_w,
        "baseline_method": channel.baseline_method,
        "diagnostic_gate_low_sample_median_power_w": (
            channel.diagnostic_gate_low_sample_median_power_w
        ),
        "baseline_source": channel.baseline_source,
        "baseline_metadata_file": channel.baseline_metadata_file,
        "baseline_metadata_sha256": channel.baseline_metadata_sha256,
        "baseline_capture_source_file": channel.baseline_capture_source_file,
    }
    for field in PULSE_BUILD_CONTEXT_FIELDS:
        if build_context.get(field, "") != "":
            row[field] = build_context[field]
    return row


def _stage_rows(
    channel: PowerChannel,
    window: EventWindow,
    event: Mapping[str, str],
    stage_names: Mapping[int, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sequence, code, start_s, end_s in _stage_intervals(channel, window):
        energy_j, average_w, peak_w = integrate_series(channel.times_s, channel.powers_w, start_s, end_s)
        event_role = normalized_name(event.get("role", "")) or channel.role
        row: dict[str, object] = {
            "role": event_role,
            "logical_role": event_role,
            "measurement_scope": "own_gate_stage",
            "serial_event_present": int(bool(event)),
            "power_channel_role": channel.role,
            "power_event_sequence": window.sequence,
            "stage_sequence": sequence,
            "stage_code": code,
            "stage": stage_names.get(code, f"stage_{code}"),
            "start_time_s": start_s,
            "end_time_s": end_s,
            "duration_s": end_s - start_s,
            "energy_total_j": energy_j,
            "energy_incremental_j": energy_j - channel.baseline_power_w * (end_s - start_s),
            "average_power_w": average_w,
            "peak_power_w": peak_w,
            "baseline_power_w": channel.baseline_power_w,
            "baseline_method": channel.baseline_method,
            "baseline_source": channel.baseline_source,
            "truncated_start": int(window.truncated_start),
            "truncated_end": int(window.truncated_end),
            "join_status": "matched" if event else "unmatched_power_window",
            "source_file": channel.source_file,
            "source_sha256": channel.source_sha256,
        }
        for field in EVENT_FIELDS_TO_COPY:
            if event.get(field, "") != "":
                row[field] = event[field]
        rows.append(row)
    return rows


def _trace_rows(
    channel: PowerChannel,
    window: EventWindow,
    event: Mapping[str, str],
    stage_names: Mapping[int, str],
    padding_s: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pair_origin = window.start_s
    first_index = bisect.bisect_left(channel.times_s, window.start_s - padding_s)
    last_index = bisect.bisect_right(channel.times_s, window.end_s + padding_s)
    for index in range(first_index, last_index):
        event_role = normalized_name(event.get("role", "")) or channel.role
        row: dict[str, object] = {
            "role": event_role,
            "power_channel_role": channel.role,
            "power_event_sequence": window.sequence,
            "capture_time_s": channel.times_s[index],
            "event_time_s": channel.times_s[index] - pair_origin,
            "current_a": channel.currents_a[index],
            "voltage_v": channel.voltages_v[index],
            "power_w": channel.powers_w[index],
            "event_gate": int(channel.gates[index]),
            "stage_code": channel.stage_codes[index],
            "stage": stage_names.get(channel.stage_codes[index], f"stage_{channel.stage_codes[index]}"),
            "source_file": channel.source_file,
            "source_sha256": channel.source_sha256,
        }
        for field in EVENT_FIELDS_TO_COPY:
            if event.get(field, "") != "":
                row[field] = event[field]
        rows.append(row)
    return rows


def _combined_trace_integral(
    left: PowerChannel, right: PowerChannel, start_s: float, end_s: float
) -> tuple[float, float, float]:
    if start_s < left.times_s[0] or start_s < right.times_s[0] or end_s > left.times_s[-1] or end_s > right.times_s[-1]:
        raise ValueError("paired channels do not both cover the complete pair-event interval")
    times = sorted(
        {start_s, end_s}
        | {value for value in left.times_s if start_s < value < end_s}
        | {value for value in right.times_s if start_s < value < end_s}
    )
    powers = [
        _interpolate(left.times_s, left.powers_w, value)
        + _interpolate(right.times_s, right.powers_w, value)
        for value in times
    ]
    energy = sum(
        (right_t - left_t) * (left_p + right_p) / 2.0
        for left_t, right_t, left_p, right_p in zip(times, times[1:], powers, powers[1:])
    )
    return energy, energy / (end_s - start_s), max(powers)


def _common_interval_metric_row(
    channel: PowerChannel,
    start_s: float,
    end_s: float,
    event: Mapping[str, str],
    *,
    role: str,
    measurement_scope: str,
    join_status: str,
    truncated_start: bool,
    truncated_end: bool,
    serial_event_present: bool,
) -> dict[str, object]:
    """Integrate one board over the exact system-level pair interval."""

    energy_j, average_w, peak_w = integrate_series(
        channel.times_s, channel.powers_w, start_s, end_s
    )
    charge_c, average_a, peak_a = integrate_series(
        channel.times_s, channel.currents_a, start_s, end_s
    )
    duration_s = end_s - start_s
    row: dict[str, object] = {
        "role": role,
        "logical_role": normalized_name(event.get("role", "")) or role,
        "measurement_scope": measurement_scope,
        "serial_event_present": int(serial_event_present),
        "power_channel_role": channel.role,
        "start_time_s": start_s,
        "end_time_s": end_s,
        "duration_s": duration_s,
        "energy_total_j": energy_j,
        "energy_incremental_j": energy_j - channel.baseline_power_w * duration_s,
        "charge_c": charge_c,
        "average_current_a": average_a,
        "peak_current_a": peak_a,
        "average_power_w": average_w,
        "peak_power_w": peak_w,
        "baseline_power_w": channel.baseline_power_w,
        "baseline_method": channel.baseline_method,
        "baseline_source": channel.baseline_source,
        "baseline_metadata_file": channel.baseline_metadata_file,
        "baseline_metadata_sha256": channel.baseline_metadata_sha256,
        "baseline_capture_source_file": channel.baseline_capture_source_file,
        "nominal_sample_rate_hz": 1.0
        / statistics.median(
            right - left for left, right in zip(channel.times_s, channel.times_s[1:])
        ),
        "mean_voltage_v": statistics.fmean(channel.voltages_v),
        "truncated_start": int(truncated_start),
        "truncated_end": int(truncated_end),
        "join_status": join_status,
        "source_file": channel.source_file,
        "source_sha256": channel.source_sha256,
    }
    for field in EVENT_FIELDS_TO_COPY:
        if event.get(field, "") != "":
            row[field] = event[field]
    return row


def pair_metric_rows(
    windows_by_role: Mapping[str, list[tuple[PowerChannel, EventWindow, Mapping[str, str]]]],
    event_rows: Sequence[Mapping[str, str]],
    channels: Sequence[PowerChannel],
) -> list[dict[str, object]]:
    serial_initiators, _, integrity_issues = accepted_session_integrity(event_rows)
    if integrity_issues:
        raise ValueError(
            "accepted-session integrity failure (not waived by --allow-unmatched): "
            + " | ".join(integrity_issues)
        )
    if not serial_initiators:
        return []

    def keyed_power_entries(
        entries: Sequence[tuple[PowerChannel, EventWindow, Mapping[str, str]]],
        role: str,
    ) -> dict[tuple[str, str, str], tuple[PowerChannel, EventWindow, Mapping[str, str]]]:
        result: dict[tuple[str, str, str], tuple[PowerChannel, EventWindow, Mapping[str, str]]] = {}
        for entry in entries:
            event = entry[2]
            if normalized_name(str(event.get("role", ""))) != role or normalized_name(
                str(event.get("event_path", ""))
            ) not in ACCEPTED_PATHS:
                continue
            key = tuple(
                str(event.get(field, "")).strip()
                for field in ("run_id", "pair_id", "exchange_id")
            )
            if not all(key):
                raise ValueError(
                    f"accepted {role} power event is missing run_id, pair_id, or exchange_id"
                )
            if key in result:
                raise ValueError(
                    f"duplicate {role} power event for run/pair/exchange {key}"
                )
            result[key] = entry
        return result

    initiators = keyed_power_entries(windows_by_role.get("initiator", []), "initiator")
    responders = keyed_power_entries(windows_by_role.get("responder", []), "responder")
    responder_channels = [channel for channel in channels if channel.role == "responder"]
    rows: list[dict[str, object]] = []
    for key, serial_initiator in sorted(serial_initiators.items()):
        run_id, pair_id, exchange_id = key
        if key not in initiators:
            raise ValueError(
                f"accepted initiator session {key} has no matched initiator power window"
            )
        init_channel, init_window, init_event = initiators[key]
        remote_started = parse_bool(serial_initiator.get("remote_session_started"))
        if remote_started is True:
            if key not in responders:
                raise ValueError(
                    f"remote session {key} has no matched responder power window"
                )
            resp_channel, resp_window, resp_event = responders[key]
            context_mismatches = [
                field
                for field in (
                    *PULSE_BUILD_CONTEXT_FIELDS,
                    "att_mtu",
                    "data_length",
                    "connection_interval_units",
                )
                if str(init_event.get(field, "")).strip()
                != str(resp_event.get(field, "")).strip()
            ]
            if str(init_event.get("tx_phy", "")).strip() != str(
                resp_event.get("rx_phy", "")
            ).strip():
                context_mismatches.append("initiator_tx_phy_vs_responder_rx_phy")
            if str(init_event.get("rx_phy", "")).strip() != str(
                resp_event.get("tx_phy", "")
            ).strip():
                context_mismatches.append("initiator_rx_phy_vs_responder_tx_phy")
            if context_mismatches:
                raise ValueError(
                    f"remote session {key} mixes initiator/responder build or link "
                    f"configuration fields: {context_mismatches}"
                )
            start_s = min(init_window.start_s, resp_window.start_s)
            end_s = max(init_window.end_s, resp_window.end_s)
            join_status = "paired_remote_session_exact_exchange_id"
            truncated_start = init_window.truncated_start or resp_window.truncated_start
            truncated_end = init_window.truncated_end or resp_window.truncated_end
        else:
            if key in responders:
                raise ValueError(
                    f"pre-session failure {key} unexpectedly has a responder power window"
                )
            covering_channels = [
                channel
                for channel in responder_channels
                if channel.times_s[0] <= init_window.start_s
                and channel.times_s[-1] >= init_window.end_s
            ]
            if len(covering_channels) != 1:
                raise ValueError(
                    f"pre-session failure {key} requires exactly one synchronized responder "
                    f"analyzer channel covering the initiator window; found {len(covering_channels)}"
                )
            resp_channel = covering_channels[0]
            resp_event = {}
            start_s = init_window.start_s
            end_s = init_window.end_s
            join_status = "initiator_only_pre_session_failure_exact_exchange_id"
            truncated_start = init_window.truncated_start
            truncated_end = init_window.truncated_end

        _, _, peak = _combined_trace_integral(
            init_channel, resp_channel, start_s, end_s
        )
        init_success = parse_bool(init_event.get("success"))
        resp_success = parse_bool(resp_event.get("success")) if remote_started else True
        init_common = _common_interval_metric_row(
            init_channel,
            start_s,
            end_s,
            init_event,
            role="initiator",
            measurement_scope="pair_union",
            join_status=join_status,
            truncated_start=truncated_start,
            truncated_end=truncated_end,
            serial_event_present=True,
        )
        if remote_started:
            resp_common = _common_interval_metric_row(
                resp_channel,
                start_s,
                end_s,
                resp_event,
                role="responder",
                measurement_scope="pair_union",
                join_status=join_status,
                truncated_start=truncated_start,
                truncated_end=truncated_end,
                serial_event_present=True,
            )
        else:
            # This board did not start a protocol session.  Retain its
            # synchronized background contribution so the pair sum is exact,
            # but keep it out of responder-event distributions.
            resp_common = _common_interval_metric_row(
                resp_channel,
                start_s,
                end_s,
                init_event,
                role="responder_idle_pair_union_diagnostic",
                measurement_scope="pair_union_background_diagnostic",
                join_status=join_status,
                truncated_start=truncated_start,
                truncated_end=truncated_end,
                serial_event_present=False,
            )
        rows.extend((init_common, resp_common))

        energy = float(init_common["energy_total_j"]) + float(
            resp_common["energy_total_j"]
        )
        incremental_energy = float(init_common["energy_incremental_j"]) + float(
            resp_common["energy_incremental_j"]
        )
        baseline = float(init_common["baseline_power_w"]) + float(
            resp_common["baseline_power_w"]
        )
        duration_s = end_s - start_s
        row: dict[str, object] = {
            "role": "pair",
            "logical_role": "pair",
            "measurement_scope": "pair_union",
            "serial_event_present": 1,
            "power_channel_role": "pair",
            "run_id": run_id,
            "pair_id": pair_id,
            "exchange_id": exchange_id,
            "remote_session_started": int(bool(remote_started)),
            "event_index": init_event.get("event_index", ""),
            "marker_event_index": init_event.get("marker_event_index", ""),
            "trial_id": init_event.get("trial_id", ""),
            "event_path": init_event.get("event_path", ""),
            "start_time_s": start_s,
            "end_time_s": end_s,
            "duration_s": duration_s,
            "energy_total_j": energy,
            "energy_incremental_j": incremental_energy,
            "charge_c": float(init_common["charge_c"]) + float(resp_common["charge_c"]),
            "average_current_a": float(init_common["average_current_a"])
            + float(resp_common["average_current_a"]),
            "peak_current_a": "",
            "average_power_w": energy / duration_s,
            "peak_power_w": peak,
            "baseline_power_w": baseline,
            "baseline_method": "sum_of_board_baselines",
            "baseline_source": (
                init_channel.baseline_source
                if init_channel.baseline_source == resp_channel.baseline_source
                else "mixed_board_baseline_sources"
            ),
            "baseline_metadata_file": (
                init_channel.baseline_metadata_file
                if init_channel.baseline_metadata_file
                == resp_channel.baseline_metadata_file
                else ""
            ),
            "baseline_metadata_sha256": (
                init_channel.baseline_metadata_sha256
                if init_channel.baseline_metadata_sha256
                == resp_channel.baseline_metadata_sha256
                else ""
            ),
            "baseline_capture_source_file": (
                f"{init_channel.baseline_capture_source_file} | "
                f"{resp_channel.baseline_capture_source_file}"
            ),
            "success": "" if init_success is None or resp_success is None else int(init_success and resp_success),
            "join_status": join_status,
            "truncated_start": int(truncated_start),
            "truncated_end": int(truncated_end),
            "source_file": f"{init_channel.source_file} | {resp_channel.source_file}",
            "source_sha256": (
                f"{init_channel.source_sha256};{resp_channel.source_sha256}"
            ),
        }
        for field in (
            "connection_state",
            "post_warmup",
            "warmup",
            "status",
            "error_code",
            *PULSE_BUILD_CONTEXT_FIELDS,
            *PULSE_EVENT_CONFIGURATION_FIELDS,
        ):
            value = init_event.get(field, resp_event.get(field, ""))
            if value != "":
                row[field] = value
        rows.append(row)

    unexpected_responder_power = sorted(set(responders).difference(serial_initiators))
    if unexpected_responder_power:
        raise ValueError(
            "orphan responder power window(s) for run/pair/exchange: "
            + ", ".join(str(key) for key in unexpected_responder_power)
        )
    return rows


def analyze_power(
    input_specs: Iterable[str],
    output_dir: Path,
    events_csv: Path | None,
    metadata_csv: Path | None,
    voltage_v: float | None,
    baseline_specs: Iterable[str],
    stage_overrides: Iterable[str],
    marker_threshold: float,
    active_low: bool,
    allow_unmatched: bool,
    trace_padding_ms: float,
    baseline_only: bool = False,
    allow_gate_low_baseline_diagnostic: bool = False,
    baseline_metadata_csv: Path | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    baseline_specs = list(baseline_specs)
    if baseline_only and baseline_specs:
        raise ValueError(
            "--baseline-only measures the all-gates-low baseline and must not be "
            "combined with --baseline-power-w"
        )
    if baseline_only and baseline_metadata_csv is not None:
        raise ValueError("--baseline-only cannot consume --baseline-metadata")
    if baseline_specs and baseline_metadata_csv is not None:
        raise ValueError(
            "choose --baseline-metadata or --baseline-power-w, not both"
        )
    if allow_gate_low_baseline_diagnostic and baseline_metadata_csv is not None:
        raise ValueError(
            "--allow-gate-low-baseline-diagnostic cannot be combined with --baseline-metadata"
        )
    baseline_bindings: dict[str, dict[str, str]] | None = None
    baseline_metadata_sha256 = ""
    if baseline_metadata_csv is not None:
        baseline_bindings, baseline_metadata_sha256 = load_baseline_metadata(
            baseline_metadata_csv
        )
    automatic_baseline_mode = (
        "baseline_only"
        if baseline_only
        else "diagnostic_intrace"
        if allow_gate_low_baseline_diagnostic
        else None
    )
    channels = load_power_channels(
        input_specs,
        voltage_v=voltage_v,
        baseline_specs=baseline_specs,
        marker_threshold=marker_threshold,
        active_low=active_low,
        automatic_baseline_mode=automatic_baseline_mode,
        baseline_bindings=baseline_bindings,
        baseline_metadata_file=str(baseline_metadata_csv or ""),
        baseline_metadata_sha256=baseline_metadata_sha256,
    )
    role_counts = {
        role: sum(channel.role == role for channel in channels)
        for role in {channel.role for channel in channels}
    }
    duplicate_roles = {role: count for role, count in role_counts.items() if count != 1}
    if duplicate_roles:
        raise ValueError(
            "power analysis requires exactly one synchronized channel per physical role; "
            f"found {duplicate_roles}"
        )
    event_rows = read_csv_rows(events_csv) if events_csv else []
    metadata_rows = read_csv_rows(metadata_csv) if metadata_csv and metadata_csv.exists() else []
    stage_names = load_stage_names(metadata_csv, stage_overrides)
    metrics: list[dict[str, object]] = []
    stages: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    capture_metadata: list[dict[str, object]] = []
    windows_by_role: dict[str, list[tuple[PowerChannel, EventWindow, Mapping[str, str]]]] = {}

    def build_context_for_role(role: str) -> dict[str, str]:
        """Return the unique PULSE_META build context for one analyzer channel."""

        role_rows = [
            row
            for row in metadata_rows
            if normalized_name(row.get("role", "")) == role
        ]
        context: dict[str, str] = {}
        for field in PULSE_BUILD_CONTEXT_FIELDS:
            values = {
                str(row.get(field, "")).strip()
                for row in role_rows
                if str(row.get(field, "")).strip()
            }
            if len(values) > 1:
                raise ValueError(
                    f"power metadata for role {role!r} has conflicting {field}: "
                    f"{sorted(values)}"
                )
            if values:
                context[field] = next(iter(values))
        return context

    accepted_initiators: dict[tuple[str, str, str], Mapping[str, object]] = {}
    accepted_responders: dict[tuple[str, str, str], Mapping[str, object]] = {}
    if event_rows and not baseline_only:
        accepted_initiators, accepted_responders, integrity_issues = accepted_session_integrity(
            event_rows
        )
        if integrity_issues:
            raise ValueError(
                "accepted-session integrity failure (not waived by --allow-unmatched): "
                + " | ".join(integrity_issues)
            )

    for channel in channels:
        windows = find_event_windows(channel)
        capture_mode = (
            "baseline_only"
            if baseline_only
            else "events_diagnostic_intrace_baseline"
            if channel.baseline_method.startswith("diagnostic_intrace")
            else "events_explicit_matched_baseline"
        )
        capture_metadata.append(
            _capture_metadata(
                channel,
                len(windows),
                capture_mode,
                build_context_for_role(channel.role),
            )
        )
        if baseline_only:
            if windows:
                raise ValueError(
                    f"{channel.source_file}: --baseline-only requires an all-gates-low capture "
                    f"for {channel.role}, but found {len(windows)} asserted event window(s)"
                )
            continue
        if not windows:
            valid_pre_session_only = (
                channel.role == "responder"
                and bool(accepted_initiators)
                and not accepted_responders
                and all(
                    parse_bool(row.get("remote_session_started")) is False
                    for row in accepted_initiators.values()
                )
            )
            if valid_pre_session_only:
                windows_by_role.setdefault(channel.role, [])
                continue
            raise ValueError(
                f"{channel.source_file}: no asserted event-gate window found for {channel.role}; "
                "use --baseline-only only for a deliberate all-gates-low baseline capture"
            )
        board_ids = _board_ids_for_role(event_rows, metadata_rows, channel.role)
        joined = join_windows_to_events(
            windows,
            event_rows,
            channel.role,
            allow_unmatched,
            board_ids=board_ids,
        )
        for window, event in zip(windows, joined):
            metric = _metric_row(channel, window, event)
            if (
                normalized_name(event.get("event_path", "")) in ACCEPTED_PATHS
                and normalized_name(event.get("role", "")) in {"initiator", "responder"}
            ):
                logical_role = normalized_name(event.get("role", ""))
                metric["role"] = f"{logical_role}_own_gate_diagnostic"
                metric["logical_role"] = logical_role
                metric["measurement_scope"] = "own_gate_diagnostic"
            metrics.append(metric)
            stages.extend(_stage_rows(channel, window, event, stage_names))
            traces.extend(
                _trace_rows(
                    channel,
                    window,
                    event,
                    stage_names,
                    padding_s=max(0.0, trace_padding_ms) / 1000.0,
                )
            )
            windows_by_role.setdefault(channel.role, []).append((channel, window, event))

    metrics.extend(pair_metric_rows(windows_by_role, event_rows, channels))
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        "run_id",
        "pair_id",
        "board_id",
        "trial_id",
        "exchange_id",
        "role",
        "logical_role",
        "measurement_scope",
        "serial_event_present",
        "power_channel_role",
        "remote_session_started",
        "event_path",
        "connection_state",
        "post_warmup",
        "warmup",
        "success",
        "status",
        "error_code",
        "event_index",
        "marker_event_index",
        "power_event_sequence",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "energy_total_j",
        "energy_incremental_j",
        "charge_c",
        "average_current_a",
        "peak_current_a",
        "average_power_w",
        "peak_power_w",
        "baseline_power_w",
        "baseline_method",
        "baseline_source",
        "baseline_metadata_file",
        "baseline_metadata_sha256",
        "baseline_capture_source_file",
        "nominal_sample_rate_hz",
        "mean_voltage_v",
        "join_status",
        "truncated_start",
        "truncated_end",
        "source_file",
        "source_sha256",
        *PULSE_BUILD_CONTEXT_FIELDS,
    ]
    write_csv_rows(output_dir / "event_power_metrics.csv", metrics, preferred=metric_fields)
    write_csv_rows(
        output_dir / "stage_power_metrics.csv",
        stages,
        preferred=metric_fields + ["stage_sequence", "stage_code", "stage"],
    )
    write_csv_rows(
        output_dir / "power_trace.csv",
        traces,
        preferred=[
            "run_id",
            "pair_id",
            "board_id",
            "trial_id",
            "exchange_id",
            "event_index",
            "marker_event_index",
            "role",
            "power_channel_role",
            "remote_session_started",
            "event_path",
            "connection_state",
            "post_warmup",
            "warmup",
            "success",
            "power_event_sequence",
            "capture_time_s",
            "event_time_s",
            "current_a",
            "voltage_v",
            "power_w",
            "event_gate",
            "stage_code",
            "stage",
            "source_file",
            "source_sha256",
            *PULSE_BUILD_CONTEXT_FIELDS,
        ],
    )
    write_csv_rows(
        output_dir / "power_capture_metadata.csv",
        capture_metadata,
        preferred=[
            "role",
            "capture_mode",
            "event_window_count",
            "all_event_gates_low",
            "source_file",
            "source_sha256",
            "sample_count",
            "capture_start_s",
            "capture_end_s",
            "capture_duration_s",
            "nominal_sample_rate_hz",
            "minimum_sample_interval_s",
            "maximum_sample_interval_s",
            "mean_voltage_v",
            "minimum_voltage_v",
            "maximum_voltage_v",
            "baseline_power_w",
            "baseline_method",
            "diagnostic_gate_low_sample_median_power_w",
            "baseline_source",
            "baseline_metadata_file",
            "baseline_metadata_sha256",
            "baseline_capture_source_file",
            *PULSE_BUILD_CONTEXT_FIELDS,
        ],
    )
    return metrics, stages, traces
