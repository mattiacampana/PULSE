"""Sequential Keysight 34465A capture and Figure 4a analysis for PULSE."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import time
from uuid import uuid4
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from common import read_csv_rows, sha256_file, summarize_values, write_csv_rows


SAMPLE_INTERVAL_S = 0.020
APERTURE_S = 0.019
SAMPLES_PER_TRIGGER = 225
PRETRIGGER_SAMPLES = 20
TRIGGER_COUNT_PER_ARM = 1
WARMUP_REPETITIONS = 5
MEASURED_REPETITIONS = 100
REPETITIONS_PER_PATH = WARMUP_REPETITIONS + MEASURED_REPETITIONS
STANDARD_MEMORY_READINGS = 50_000
PILOT_DURATION_S = 180.0
EXPECTED_TRIGGERS = {
    "initiator": 2 * REPETITIONS_PER_PATH,
    "responder": REPETITIONS_PER_PATH,
}
OVERLOAD_READING_A = 9.0e36
CURRENT_RANGE_PAIRS = ((0.01, 3), (0.1, 3))
MAX_STARTUP_SPURIOUS_TRIGGERS = 2
STARTUP_SPURIOUS_WINDOW_S = 15.0
WAIT_STATUS_INTERVAL_S = 15.0
STREAM_CHUNK_READINGS = 100
STREAM_MAX_REMOVE_READINGS = 1000
STREAM_EVENT_CONFIRM_READINGS = 3
STREAM_MIN_EVENT_PERIOD_S = 4.5
STREAM_FIRST_EVENT_GUARD_S = 20.0


def verify_keysight_idn(value: str) -> str:
    identity = value.strip()
    if "34465A" not in identity.upper():
        raise ValueError(f"expected a Keysight 34465A, received *IDN?={identity!r}")
    return identity


def validate_capture_settings(
    measured_role: str,
    sample_interval_s: float = SAMPLE_INTERVAL_S,
    aperture_s: float = APERTURE_S,
    samples_per_trigger: int = SAMPLES_PER_TRIGGER,
    pretrigger_samples: int = PRETRIGGER_SAMPLES,
    current_range_a: float = 0.01,
    current_terminal_a: int = 3,
    trigger_level_a: float = 0.005,
) -> int:
    if measured_role not in EXPECTED_TRIGGERS:
        raise ValueError("measured role must be initiator or responder")
    if not math.isfinite(sample_interval_s) or sample_interval_s <= 0.0:
        raise ValueError("sample interval must be finite and positive")
    if not math.isfinite(aperture_s) or aperture_s <= 0.0 or aperture_s >= sample_interval_s:
        raise ValueError("aperture must be positive and shorter than the sample interval")
    if samples_per_trigger <= 0:
        raise ValueError("samples per trigger must be positive")
    if pretrigger_samples <= 0 or pretrigger_samples >= samples_per_trigger:
        raise ValueError("pretrigger samples must be between zero and samples per trigger")
    if (current_range_a, current_terminal_a) not in CURRENT_RANGE_PAIRS:
        raise ValueError(
            "use the 10 mA or 100 mA fixed range on the 3 A terminal; "
            "the 10 A terminal forces the 10 A measurement range"
        )
    if not math.isfinite(trigger_level_a) or not 0.0 < trigger_level_a < current_range_a:
        raise ValueError("trigger level must be positive and below the fixed current range")
    expected_readings = EXPECTED_TRIGGERS[measured_role] * samples_per_trigger
    if expected_readings > STANDARD_MEMORY_READINGS:
        raise ValueError(
            f"capture needs {expected_readings} readings, exceeding the standard "
            f"{STANDARD_MEMORY_READINGS}-reading memory"
        )
    return expected_readings


def capture_configuration_commands(
    measured_role: str,
    *,
    sample_interval_s: float = SAMPLE_INTERVAL_S,
    aperture_s: float = APERTURE_S,
    samples_per_trigger: int = SAMPLES_PER_TRIGGER,
    pretrigger_samples: int = PRETRIGGER_SAMPLES,
    current_range_a: float = 0.01,
    current_terminal_a: int = 3,
    trigger_level_a: float = 0.005,
) -> list[str]:
    validate_capture_settings(
        measured_role,
        sample_interval_s,
        aperture_s,
        samples_per_trigger,
        pretrigger_samples,
        current_range_a,
        current_terminal_a,
        trigger_level_a,
    )
    return [
        "*RST",
        "*CLS",
        f"CONF:CURR:DC {current_range_a:.12g}",
        f"SENS:CURR:DC:TERM {current_terminal_a}",
        f"SENS:CURR:DC:APER {aperture_s:.12g}",
        "SENS:CURR:DC:ZERO:AUTO OFF",
        "SAMP:SOUR TIM",
        f"SAMP:TIM {sample_interval_s:.12g}",
        f"SAMP:COUN {samples_per_trigger}",
        f"SAMP:COUN:PRET {pretrigger_samples}",
        "TRIG:SOUR INT",
        f"TRIG:LEV {trigger_level_a:.12g}",
        "TRIG:SLOP POS",
        "TRIG:DEL:AUTO OFF",
        "TRIG:DEL 0",
        # The 34465A requires TRIG:COUN 1 whenever pretrigger sampling is enabled.
        # The host therefore fetches each completed record and rearms the DMM.
        f"TRIG:COUN {TRIGGER_COUNT_PER_ARM}",
        "FORM:DATA ASC",
    ]


def pilot_configuration_commands(
    *,
    duration_s: float = PILOT_DURATION_S,
    sample_interval_s: float = SAMPLE_INTERVAL_S,
    aperture_s: float = APERTURE_S,
    current_range_a: float = 0.01,
    current_terminal_a: int = 3,
) -> tuple[list[str], int]:
    """Configure a finite immediate-trigger trace without guessing an event level."""
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("pilot duration must be finite and positive")
    if not math.isfinite(sample_interval_s) or sample_interval_s <= 0.0:
        raise ValueError("sample interval must be finite and positive")
    if not math.isfinite(aperture_s) or not 0.0 < aperture_s < sample_interval_s:
        raise ValueError("aperture must be positive and shorter than the sample interval")
    if (current_range_a, current_terminal_a) not in CURRENT_RANGE_PAIRS:
        raise ValueError(
            "pilot requires the 10 mA or 100 mA range on the 3 A terminal"
        )
    count = round(duration_s / sample_interval_s)
    if count < 2 or count > STANDARD_MEMORY_READINGS:
        raise ValueError("pilot requires 2 to 50000 readings in standard DMM memory")
    return (
        [
            "*RST",
            "*CLS",
            f"CONF:CURR:DC {current_range_a:.12g}",
            f"SENS:CURR:DC:TERM {current_terminal_a}",
            f"SENS:CURR:DC:APER {aperture_s:.12g}",
            "SENS:CURR:DC:ZERO:AUTO OFF",
            "SAMP:SOUR TIM",
            f"SAMP:TIM {sample_interval_s:.12g}",
            f"SAMP:COUN {count}",
            "SAMP:COUN:PRET 0",
            "TRIG:SOUR IMM",
            "TRIG:DEL:AUTO OFF",
            "TRIG:DEL 0",
            "TRIG:COUN 1",
            "FORM:DATA ASC",
        ],
        count,
    )


def stream_configuration_commands(
    *,
    duration_s: float,
    sample_interval_s: float = SAMPLE_INTERVAL_S,
    aperture_s: float = APERTURE_S,
    current_range_a: float = 0.01,
    current_terminal_a: int = 3,
) -> tuple[list[str], int]:
    """Configure a continuous trace; the host drains memory while acquisition runs."""
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("stream duration must be finite and positive")
    if not math.isfinite(sample_interval_s) or sample_interval_s <= 0.0:
        raise ValueError("stream sample interval must be finite and positive")
    if not math.isfinite(aperture_s) or not 0.0 < aperture_s < sample_interval_s:
        raise ValueError("stream aperture must be shorter than the sample interval")
    if (current_range_a, current_terminal_a) not in CURRENT_RANGE_PAIRS:
        raise ValueError("stream requires the 10 mA or 100 mA range on the 3 A terminal")
    count = math.ceil(duration_s / sample_interval_s)
    if count < 2 or count > 1_000_000_000:
        raise ValueError("stream sample count must be between 2 and 1000000000")
    return (
        [
            "*RST",
            "*CLS",
            f"CONF:CURR:DC {current_range_a:.12g}",
            f"SENS:CURR:DC:TERM {current_terminal_a}",
            f"SENS:CURR:DC:APER {aperture_s:.12g}",
            "SENS:CURR:DC:ZERO:AUTO OFF",
            "SAMP:SOUR TIM",
            f"SAMP:TIM {sample_interval_s:.12g}",
            f"SAMP:COUN {count}",
            "SAMP:COUN:PRET 0",
            "TRIG:SOUR IMM",
            "TRIG:DEL:AUTO OFF",
            "TRIG:DEL 0",
            "TRIG:COUN 1",
            "FORM:DATA ASC",
        ],
        count,
    )


def _event_identity(measured_role: str, trigger_index: int) -> tuple[str, str, int]:
    if measured_role == "initiator":
        if not 0 <= trigger_index < EXPECTED_TRIGGERS[measured_role]:
            raise ValueError(f"initiator trigger index out of range: {trigger_index}")
        if trigger_index < REPETITIONS_PER_PATH:
            return "local", "local", trigger_index
        return "accepted_connected", "initiator", trigger_index - REPETITIONS_PER_PATH
    if measured_role == "responder":
        if not 0 <= trigger_index < EXPECTED_TRIGGERS[measured_role]:
            raise ValueError(f"responder trigger index out of range: {trigger_index}")
        return "accepted_connected", "responder", trigger_index
    raise ValueError(f"invalid measured role {measured_role!r}")


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "success"}


def _require_text(args: argparse.Namespace, names: Sequence[str]) -> None:
    missing = [
        "--" + name.replace("_", "-")
        for name in names
        if getattr(args, name, None) is None or not str(getattr(args, name)).strip()
    ]
    if missing:
        raise ValueError("missing required capture metadata: " + ", ".join(missing))


def _validate_digest(value: str, name: str) -> str:
    digest = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters")
    return digest


def _query_ascii_values(instrument: object, command: str) -> list[float]:
    method = getattr(instrument, "query_ascii_values", None)
    if callable(method):
        return [float(value) for value in method(command)]
    response = str(getattr(instrument, "query")(command))
    return [float(value) for value in response.strip().split(",") if value.strip()]


def _query_float(instrument: object, command: str) -> float:
    return float(str(getattr(instrument, "query")(command)).strip())


def _save_record_diagnostic(
    args: argparse.Namespace,
    trigger_index: int,
    attempt: int,
    reason: str,
    readings: Sequence[float],
    reported_count: int,
) -> Path:
    """Preserve an unusable record without presenting it as Figure 4a data."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"keysight_34465a_{args.measured_role}_trigger_{trigger_index:03d}_"
        f"attempt_{attempt:02d}_{uuid4().hex[:8]}_diagnostic.csv"
    )
    write_csv_rows(
        path,
        [
            {
                "run_id": args.run_id,
                "trigger_index": trigger_index,
                "arm_attempt": attempt,
                "reason": reason,
                "reported_reading_count": reported_count,
                "fetched_reading_count": len(readings),
                "expected_reading_count": args.samples_per_trigger,
                "pretrigger_samples_requested": args.pretrigger_samples,
                "trigger_level_a": args.trigger_level_a,
                "sample_index": index,
                "current_a": value,
            }
            for index, value in enumerate(readings)
        ],
        fieldnames=(
            "run_id",
            "trigger_index",
            "arm_attempt",
            "reason",
            "reported_reading_count",
            "fetched_reading_count",
            "expected_reading_count",
            "pretrigger_samples_requested",
            "trigger_level_a",
            "sample_index",
            "current_a",
        ),
    )
    return path


def _instrument_errors(instrument: object) -> list[str]:
    errors: list[str] = []
    for _ in range(32):
        response = str(getattr(instrument, "query")("SYST:ERR?")).strip()
        code = response.split(",", 1)[0].strip()
        if code in {"0", "+0"}:
            return errors
        errors.append(response)
    errors.append("error queue did not terminate after 32 entries")
    return errors


def _readbacks(instrument: object) -> dict[str, str]:
    commands = {
        "current_range_a_readback": "SENS:CURR:DC:RANG?",
        "current_terminal_a_readback": "SENS:CURR:DC:TERM?",
        "aperture_s_readback": "SENS:CURR:DC:APER?",
        "sample_interval_s_readback": "SAMP:TIM?",
        "samples_per_trigger_readback": "SAMP:COUN?",
        "pretrigger_samples_readback": "SAMP:COUN:PRET?",
        "trigger_count_readback": "TRIG:COUN?",
        "trigger_source_readback": "TRIG:SOUR?",
        "trigger_level_a_readback": "TRIG:LEV?",
        "trigger_slope_readback": "TRIG:SLOP?",
        "trigger_delay_s_readback": "TRIG:DEL?",
    }
    return {
        name: str(getattr(instrument, "query")(command)).strip()
        for name, command in commands.items()
    }


def _readback_close(readbacks: Mapping[str, str], name: str, expected: float) -> None:
    actual = float(readbacks[name])
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9):
        raise RuntimeError(f"instrument readback {name}={actual} does not match {expected}")


def _verify_readbacks(args: argparse.Namespace, readbacks: Mapping[str, str]) -> None:
    _readback_close(readbacks, "current_range_a_readback", args.current_range_a)
    _readback_close(readbacks, "current_terminal_a_readback", args.current_terminal_a)
    _readback_close(readbacks, "aperture_s_readback", args.aperture_s)
    _readback_close(readbacks, "sample_interval_s_readback", args.sample_interval_s)
    _readback_close(readbacks, "samples_per_trigger_readback", args.samples_per_trigger)
    _readback_close(readbacks, "pretrigger_samples_readback", args.pretrigger_samples)
    _readback_close(readbacks, "trigger_level_a_readback", args.trigger_level_a)
    _readback_close(readbacks, "trigger_count_readback", TRIGGER_COUNT_PER_ARM)
    if readbacks["trigger_source_readback"].strip().upper() not in {"INT", "INTERNAL"}:
        raise RuntimeError("instrument did not retain internal level triggering")
    if readbacks["trigger_slope_readback"].strip().upper() not in {"POS", "POSITIVE"}:
        raise RuntimeError("instrument did not retain positive-edge triggering")


def _capture_rows(
    readings: Sequence[float],
    args: argparse.Namespace,
    instrument_idn: str,
    readbacks: Mapping[str, str],
    *,
    stream_onsets: Sequence[int] | None = None,
    raw_invalid_indices: Sequence[int] = (),
) -> list[dict[str, object]]:
    expected = validate_capture_settings(
        args.measured_role,
        args.sample_interval_s,
        args.aperture_s,
        args.samples_per_trigger,
        args.pretrigger_samples,
        args.current_range_a,
        args.current_terminal_a,
        args.trigger_level_a,
    )
    if len(readings) != expected:
        raise RuntimeError(f"expected {expected} readings, received {len(readings)}")
    if any(not math.isfinite(value) or abs(value) >= OVERLOAD_READING_A for value in readings):
        raise RuntimeError("capture contains a non-finite or overload current reading")

    rows: list[dict[str, object]] = []
    common = {
        "capture_profile": (
            "keysight_34465a_continuous_stream_software_events"
            if stream_onsets is not None
            else "keysight_34465a_sequential_current_level_trigger"
        ),
        "run_id": args.run_id,
        "pair_id": args.pair_id,
        "board_id": args.board_id,
        "peer_board_id": args.peer_board_id,
        "measured_board_role": args.measured_role,
        "instrument_idn": instrument_idn,
        "visa_resource": args.resource,
        "supply_voltage_v": args.supply_voltage_v,
        "supply_voltage_source": args.supply_voltage_source,
        "source_current_limit_a": args.source_current_limit_a,
        "source_overvoltage_protection_v": args.source_overvoltage_protection_v,
        "maximum_safe_voltage_v": args.maximum_safe_voltage_v,
        "current_range_a": args.current_range_a,
        "current_terminal_a": args.current_terminal_a,
        "sample_interval_s": args.sample_interval_s,
        "aperture_s": args.aperture_s,
        "samples_per_trigger": args.samples_per_trigger,
        "pretrigger_samples": args.pretrigger_samples,
        "trigger_source": (
            "immediate_timed_stream_software_threshold"
            if stream_onsets is not None
            else "internal_positive_current_level"
        ),
        "trigger_signal": "measured_dc_supply_current",
        "trigger_level_a": args.trigger_level_a,
        "calibration_id": args.calibration_id,
        "calibration_date": args.calibration_date,
        "firmware_revision": args.firmware_revision,
        "firmware_image_sha256": args.firmware_image_sha256,
        "artifact_sha256": args.artifact_sha256,
        "outcome_join": "preflight_only_not_exact_power_trial_outcomes",
        "raw_invalid_reading_count": len(raw_invalid_indices),
        "raw_invalid_sample_indices": ";".join(str(index) for index in raw_invalid_indices),
        **readbacks,
    }
    for flat_index, current_a in enumerate(readings):
        trigger_index, sample_index = divmod(flat_index, args.samples_per_trigger)
        event_path, event_role, path_index = _event_identity(
            args.measured_role, trigger_index
        )
        warmup = path_index < WARMUP_REPETITIONS
        trial_id = path_index if warmup else path_index - WARMUP_REPETITIONS
        row = {
            **common,
            "trigger_index": trigger_index,
            "event_path": event_path,
            "event_role": event_role,
            "path_index": path_index,
            "warmup": int(warmup),
            "trial_id": trial_id,
            "sample_index": sample_index,
            "relative_time_s": (
                sample_index - args.pretrigger_samples
            ) * args.sample_interval_s,
            "current_a": current_a,
        }
        if stream_onsets is not None:
            onset = stream_onsets[trigger_index]
            row["source_sample_index"] = onset - args.pretrigger_samples + sample_index
            row["source_time_s"] = row["source_sample_index"] * args.sample_interval_s
        rows.append(row)
    return rows


def capture_keysight(instrument: object, args: argparse.Namespace) -> tuple[Path, Path]:
    expected = validate_capture_settings(
        args.measured_role,
        args.sample_interval_s,
        args.aperture_s,
        args.samples_per_trigger,
        args.pretrigger_samples,
        args.current_range_a,
        args.current_terminal_a,
        args.trigger_level_a,
    )
    instrument_idn = verify_keysight_idn(str(getattr(instrument, "query")("*IDN?")))
    for command in capture_configuration_commands(
        args.measured_role,
        sample_interval_s=args.sample_interval_s,
        aperture_s=args.aperture_s,
        samples_per_trigger=args.samples_per_trigger,
        pretrigger_samples=args.pretrigger_samples,
        current_range_a=args.current_range_a,
        current_terminal_a=args.current_terminal_a,
        trigger_level_a=args.trigger_level_a,
    ):
        getattr(instrument, "write")(command)
    getattr(instrument, "query")("*OPC?")
    errors = _instrument_errors(instrument)
    if errors:
        raise RuntimeError("Keysight configuration error: " + " | ".join(errors))
    readbacks = _readbacks(instrument)
    _verify_readbacks(args, readbacks)

    expected_triggers = EXPECTED_TRIGGERS[args.measured_role]
    print(
        f"Configured {expected_triggers} software-rearmed records "
        f"({expected} readings) for the {args.measured_role}; each record uses a "
        f"positive {args.trigger_level_a:.9g} A current-level trigger."
    )
    capture_started_at = time.monotonic()
    deadline = capture_started_at + args.capture_timeout_s
    readings: list[float] = []
    questionable = 0
    startup_spurious_triggers = 0
    for trigger_index in range(expected_triggers):
        while True:
            attempt = startup_spurious_triggers + 1 if trigger_index == 0 else 1
            getattr(instrument, "write")("INIT")
            # *OPC sets the Standard Event Register when the acquisition has
            # actually finished. DATA:POIN? alone cannot distinguish a short,
            # completed record from one still collecting pretrigger readings.
            getattr(instrument, "write")("*OPC")
            armed_at = time.monotonic()
            if trigger_index == 0:
                print(
                    "ARMED: keep both SensWear boards powered and below the trigger level. "
                    "Do not reset or change wiring during capture.",
                    flush=True,
                )

            last_status_at = armed_at
            record_count = 0
            while True:
                now = time.monotonic()
                if now >= deadline:
                    getattr(instrument, "write")("ABOR")
                    raise RuntimeError(
                        f"capture timed out after {trigger_index}/{expected_triggers} complete "
                        f"triggers and {max(record_count, 0)}/{args.samples_per_trigger} "
                        "readings in the active record"
                    )
                event_status = int(round(_query_float(instrument, "*ESR?")))
                if event_status & 0x3C:
                    errors = _instrument_errors(instrument)
                    raise RuntimeError(
                        "Keysight error while awaiting trigger: " + " | ".join(errors)
                    )
                record_count = int(round(_query_float(instrument, "DATA:POIN?")))
                if event_status & 1:
                    break
                if now - last_status_at >= WAIT_STATUS_INTERVAL_S:
                    print(
                        f"waiting for trigger {trigger_index + 1}/{expected_triggers}: "
                        f"{record_count} readings currently buffered",
                        flush=True,
                    )
                    last_status_at = now
                time.sleep(args.poll_interval_s)

            record_questionable = int(round(_query_float(instrument, "STAT:QUES:COND?")))
            questionable |= record_questionable
            record_readings = _query_ascii_values(instrument, "FETC?")
            errors = _instrument_errors(instrument)
            invalid_indices = [
                index
                for index, value in enumerate(record_readings)
                if not math.isfinite(value) or abs(value) >= OVERLOAD_READING_A
            ]
            posttrigger = record_readings[args.pretrigger_samples :]
            has_event = any(value >= args.trigger_level_a for value in posttrigger)
            reasons = []
            if record_questionable & (1 << 14):
                reasons.append("reading-memory overflow")
            if record_questionable & (1 << 1):
                reasons.append("current overload status")
            if (
                record_count != args.samples_per_trigger
                or len(record_readings) != args.samples_per_trigger
            ):
                reasons.append(
                    f"short record: meter reported {record_count}, fetched "
                    f"{len(record_readings)}/{args.samples_per_trigger} readings"
                )
            if invalid_indices:
                first = invalid_indices[0]
                reasons.append(
                    f"non-finite/overload current at sample {first}: "
                    f"{record_readings[first]:.9g} A"
                )
            if not has_event:
                reasons.append("no posttrigger reading reached the trigger level")
            if errors:
                reasons.append("instrument errors: " + " | ".join(errors))
            if reasons:
                reason = "; ".join(reasons)
                diagnostic_path = _save_record_diagnostic(
                    args, trigger_index, attempt, reason, record_readings, record_count
                )
                print(f"saved diagnostic {diagnostic_path}", flush=True)
                early_idle_only = (
                    trigger_index == 0
                    and armed_at - capture_started_at <= STARTUP_SPURIOUS_WINDOW_S
                    and not has_event
                    and not invalid_indices
                    and not errors
                    and not (record_questionable & ((1 << 1) | (1 << 14)))
                )
                if early_idle_only and startup_spurious_triggers < MAX_STARTUP_SPURIOUS_TRIGGERS:
                    startup_spurious_triggers += 1
                    print(
                        "early idle-only trigger; rearming first record "
                        f"({startup_spurious_triggers}/{MAX_STARTUP_SPURIOUS_TRIGGERS} retries)",
                        flush=True,
                    )
                    continue
                raise RuntimeError(
                    f"trigger {trigger_index} invalid: {reason}; see {diagnostic_path}"
                )
            break
        readings.extend(record_readings)
        print(
            f"progress: {len(readings)}/{expected} readings "
            f"({trigger_index + 1}/{expected_triggers} triggers)",
            flush=True,
        )

    rows = _capture_rows(readings, args, instrument_idn, readbacks)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / f"keysight_34465a_{args.measured_role}_samples.csv"
    write_csv_rows(samples_path, rows)
    manifest_path = output_dir / "keysight_34465a_capture_manifest.csv"
    manifest = {
        key: value
        for key, value in rows[0].items()
        if key not in {
            "trigger_index",
            "event_path",
            "event_role",
            "path_index",
            "warmup",
            "trial_id",
            "sample_index",
            "relative_time_s",
            "current_a",
        }
    }
    manifest.update(
        {
            "expected_trigger_count": EXPECTED_TRIGGERS[args.measured_role],
            "reading_count": len(readings),
            "questionable_condition": questionable,
            "memory_overflow": 0,
            "sample_csv": samples_path.name,
            "sample_csv_sha256": sha256_file(samples_path),
            "capture_status": "complete_count_and_instrument_audit_pass",
        }
    )
    write_csv_rows(manifest_path, [manifest])
    print(f"saved {samples_path}")
    print(f"saved {manifest_path}")
    return samples_path, manifest_path


def _stream_event_onsets(
    readings: Sequence[float], trigger_level_a: float, sample_interval_s: float
) -> list[int]:
    """Find sustained rising crossings without relying on the DMM level detector."""
    min_period = math.ceil(STREAM_MIN_EVENT_PERIOD_S / sample_interval_s)
    first_event = math.ceil(STREAM_FIRST_EVENT_GUARD_S / sample_interval_s)
    onsets: list[int] = []
    for index in range(len(readings) - STREAM_EVENT_CONFIRM_READINGS + 1):
        if index < first_event or (onsets and index - onsets[-1] < min_period):
            continue
        if (
            index > 0
            and (
                not _valid_current(readings[index - 1])
                or readings[index - 1] >= trigger_level_a
            )
        ):
            continue
        if all(
            _valid_current(value) and value >= trigger_level_a
            for value in readings[index : index + STREAM_EVENT_CONFIRM_READINGS]
        ):
            onsets.append(index)
    return onsets


def _valid_current(value: float) -> bool:
    return math.isfinite(value) and abs(value) < OVERLOAD_READING_A


def capture_keysight_stream(instrument: object, args: argparse.Namespace) -> tuple[Path, Path]:
    """Stream one continuous timed trace and extract paper windows in software."""
    expected_readings = validate_capture_settings(
        args.measured_role,
        args.sample_interval_s,
        args.aperture_s,
        args.samples_per_trigger,
        args.pretrigger_samples,
        args.current_range_a,
        args.current_terminal_a,
        args.trigger_level_a,
    )
    commands, stream_count = stream_configuration_commands(
        duration_s=args.capture_timeout_s,
        sample_interval_s=args.sample_interval_s,
        aperture_s=args.aperture_s,
        current_range_a=args.current_range_a,
        current_terminal_a=args.current_terminal_a,
    )
    output_dir = Path(args.output_dir)
    raw_path = output_dir / f"keysight_34465a_{args.measured_role}_stream_raw.csv"
    samples_path = output_dir / f"keysight_34465a_{args.measured_role}_samples.csv"
    manifest_path = output_dir / "keysight_34465a_capture_manifest.csv"
    for path in (raw_path, samples_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing capture file: {path}")
    instrument_idn = verify_keysight_idn(str(getattr(instrument, "query")("*IDN?")))
    for command in commands:
        getattr(instrument, "write")(command)
    getattr(instrument, "query")("*OPC?")
    errors = _instrument_errors(instrument)
    if errors:
        raise RuntimeError("Keysight stream configuration error: " + " | ".join(errors))
    readbacks = _readbacks(instrument)
    for name, value in (
        ("current_range_a_readback", args.current_range_a),
        ("current_terminal_a_readback", args.current_terminal_a),
        ("aperture_s_readback", args.aperture_s),
        ("sample_interval_s_readback", args.sample_interval_s),
        ("samples_per_trigger_readback", stream_count),
        ("pretrigger_samples_readback", 0),
        ("trigger_count_readback", 1),
    ):
        _readback_close(readbacks, name, value)
    if readbacks["trigger_source_readback"].upper() not in {"IMM", "IMMEDIATE"}:
        raise RuntimeError("DMM did not retain immediate triggering for streaming")

    expected_events = EXPECTED_TRIGGERS[args.measured_role]
    readings: list[float] = []
    onsets: list[int] = []
    raw_invalid_indices: list[int] = []
    questionable = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    with raw_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("run_id", "measured_role", "sample_index", "time_s", "current_a"))
        handle.flush()
        getattr(instrument, "write")("INIT")
        getattr(instrument, "write")("*OPC")
        started_at = time.monotonic()
        deadline = started_at + args.capture_timeout_s
        last_status_at = started_at
        print(
            f"STREAMING: collecting timed current from the {args.measured_role}; "
            f"raw samples are saved to {raw_path}. Keep both boards powered.",
            flush=True,
        )
        while True:
            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError(
                    f"stream timed out with {len(onsets)}/{expected_events} events; "
                    f"partial raw trace saved to {raw_path}"
                )
            event_status = int(round(_query_float(instrument, "*ESR?")))
            if event_status & 0x3C:
                errors = _instrument_errors(instrument)
                raise RuntimeError(
                    "Keysight streaming status error: " + " | ".join(errors)
                    + f"; partial raw trace saved to {raw_path}"
                )
            if event_status & 1:
                raise RuntimeError(
                    f"DMM stopped before {expected_events} events were captured; "
                    f"partial raw trace saved to {raw_path}"
                )
            available = int(round(_query_float(instrument, "DATA:POIN?")))
            if available < 0:
                raise RuntimeError(f"DMM reported negative available reading count: {available}")
            if available >= STREAM_CHUNK_READINGS or (
                onsets
                and len(onsets) == expected_events
                and len(readings) + available >= onsets[-1] + args.samples_per_trigger
                - args.pretrigger_samples
            ):
                take = min(available, STREAM_MAX_REMOVE_READINGS)
                chunk = _query_ascii_values(instrument, f"DATA:REMove? {take}")
                if len(chunk) != take:
                    raise RuntimeError(
                        f"DMM returned {len(chunk)}/{take} streaming readings; "
                        f"partial raw trace saved to {raw_path}"
                    )
                start_index = len(readings)
                readings.extend(chunk)
                writer.writerows(
                    (
                        args.run_id,
                        args.measured_role,
                        index,
                        format(index * args.sample_interval_s, ".12g"),
                        format(current, ".12g"),
                    )
                    for index, current in enumerate(chunk, start_index)
                )
                handle.flush()
                for index, current in enumerate(chunk, start_index):
                    if not _valid_current(current):
                        raw_invalid_indices.append(index)
                        if len(raw_invalid_indices) <= 3:
                            print(
                                f"WARNING: raw sample {index} is non-finite/overload "
                                f"({current:.9g} A); preserving its time slot and "
                                "continuing. Event windows will be checked at the end.",
                                flush=True,
                            )
                status = int(round(_query_float(instrument, "STAT:QUES:COND?")))
                questionable |= status
                if status & (1 << 14):
                    raise RuntimeError(
                        f"DMM reading-memory overflow ({status}); "
                        f"partial raw trace saved to {raw_path}"
                    )
                errors = _instrument_errors(instrument)
                if errors:
                    raise RuntimeError(
                        "Keysight streaming error: " + " | ".join(errors)
                        + f"; partial raw trace saved to {raw_path}"
                    )
                updated_onsets = _stream_event_onsets(
                    readings, args.trigger_level_a, args.sample_interval_s
                )
                if len(updated_onsets) > len(onsets):
                    onsets = updated_onsets
                    print(
                        f"stream progress: {len(onsets)}/{expected_events} events, "
                        f"{len(readings)} raw readings",
                        flush=True,
                    )
                if len(onsets) > expected_events:
                    raise RuntimeError(
                        f"stream detected {len(onsets)} events, expected {expected_events}; "
                        f"see {raw_path}"
                    )
                if (
                    len(onsets) == expected_events
                    and len(readings)
                    >= onsets[-1] + args.samples_per_trigger - args.pretrigger_samples
                ):
                    break
            else:
                if now - last_status_at >= WAIT_STATUS_INTERVAL_S:
                    print(
                        f"stream waiting: {len(onsets)}/{expected_events} events, "
                        f"{len(readings)} raw readings saved",
                        flush=True,
                    )
                    last_status_at = now
                time.sleep(args.poll_interval_s)
        getattr(instrument, "write")("ABOR")

    window_readings: list[float] = []
    invalid_index_set = set(raw_invalid_indices)
    for event_index, onset in enumerate(onsets):
        start = onset - args.pretrigger_samples
        end = start + args.samples_per_trigger
        if start < 0 or end > len(readings):
            raise RuntimeError(f"event at raw sample {onset} has an incomplete window")
        invalid_in_window = sorted(invalid_index_set.intersection(range(start, end)))
        if invalid_in_window:
            raise RuntimeError(
                f"event {event_index} window contains non-finite/overload raw "
                f"sample(s) {invalid_in_window}; no paper CSV written; see {raw_path}"
            )
        window_readings.extend(readings[start:end])
    if len(window_readings) != expected_readings:
        raise RuntimeError("software event windows have an incomplete reading count")
    rows = _capture_rows(
        window_readings, args, instrument_idn, readbacks,
        stream_onsets=onsets, raw_invalid_indices=raw_invalid_indices,
    )
    write_csv_rows(samples_path, rows)
    manifest = {
        key: value
        for key, value in rows[0].items()
        if key
        not in {
            "trigger_index", "event_path", "event_role", "path_index", "warmup",
            "trial_id", "sample_index", "relative_time_s", "current_a",
            "source_sample_index", "source_time_s",
        }
    }
    manifest.update(
        {
            "expected_trigger_count": expected_events,
            "reading_count": len(window_readings),
            "raw_reading_count": len(readings),
            "raw_csv": raw_path.name,
            "raw_csv_sha256": sha256_file(raw_path),
            "sample_csv": samples_path.name,
            "sample_csv_sha256": sha256_file(samples_path),
            "questionable_condition": questionable,
            "memory_overflow": 0,
            "capture_status": (
                "complete_with_out_of_window_invalid_raw_readings"
                if raw_invalid_indices
                else "complete_timed_stream_and_software_event_audit_pass"
            ),
        }
    )
    write_csv_rows(manifest_path, [manifest])
    print(f"saved {samples_path}", flush=True)
    print(f"saved {manifest_path}", flush=True)
    return samples_path, manifest_path


def pilot_keysight(instrument: object, args: argparse.Namespace) -> tuple[Path, Path]:
    """Record untriggered current so the event threshold can be chosen from evidence."""
    commands, expected = pilot_configuration_commands(
        duration_s=args.pilot_duration_s,
        sample_interval_s=args.sample_interval_s,
        aperture_s=args.aperture_s,
        current_range_a=args.current_range_a,
        current_terminal_a=args.current_terminal_a,
    )
    instrument_idn = verify_keysight_idn(str(getattr(instrument, "query")("*IDN?")))
    for command in commands:
        getattr(instrument, "write")(command)
    getattr(instrument, "query")("*OPC?")
    errors = _instrument_errors(instrument)
    if errors:
        raise RuntimeError("Keysight pilot configuration error: " + " | ".join(errors))
    readbacks = {
        "current_range_a": _query_float(instrument, "SENS:CURR:DC:RANG?"),
        "current_terminal_a": _query_float(instrument, "SENS:CURR:DC:TERM?"),
        "aperture_s": _query_float(instrument, "SENS:CURR:DC:APER?"),
        "sample_interval_s": _query_float(instrument, "SAMP:TIM?"),
        "sample_count": _query_float(instrument, "SAMP:COUN?"),
        "pretrigger_count": _query_float(instrument, "SAMP:COUN:PRET?"),
        "trigger_count": _query_float(instrument, "TRIG:COUN?"),
        "trigger_source": str(getattr(instrument, "query")("TRIG:SOUR?")).strip(),
    }
    for name, expected_value in (
        ("current_range_a", args.current_range_a),
        ("current_terminal_a", args.current_terminal_a),
        ("aperture_s", args.aperture_s),
        ("sample_interval_s", args.sample_interval_s),
        ("sample_count", expected),
        ("pretrigger_count", 0),
        ("trigger_count", 1),
    ):
        if not math.isclose(readbacks[name], expected_value, rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(f"pilot {name} readback {readbacks[name]} != {expected_value}")
    if readbacks["trigger_source"].upper() not in {"IMM", "IMMEDIATE"}:
        raise RuntimeError("pilot did not retain immediate triggering")

    getattr(instrument, "write")("INIT")
    print(
        f"PILOT STARTED: recording {expected} samples for "
        f"{expected * args.sample_interval_s:.1f} s. Keep both boards powered; "
        "do not reset during this diagnostic trace (not Figure 4a data).",
        flush=True,
    )
    deadline = time.monotonic() + expected * args.sample_interval_s + 60.0
    last_progress = -1
    while True:
        if time.monotonic() >= deadline:
            getattr(instrument, "write")("ABOR")
            raise RuntimeError("pilot capture timed out")
        count = int(round(_query_float(instrument, "DATA:POIN?")))
        progress = count // max(1, round(10.0 / args.sample_interval_s))
        if progress > last_progress:
            print(f"pilot progress: {count}/{expected} readings", flush=True)
            last_progress = progress
        if count >= expected:
            break
        time.sleep(max(0.5, args.poll_interval_s))

    questionable = int(round(_query_float(instrument, "STAT:QUES:COND?")))
    if questionable & ((1 << 1) | (1 << 14)):
        raise RuntimeError("pilot current overload or reading-memory overflow; discard trace")
    readings = _query_ascii_values(instrument, "FETC?")
    errors = _instrument_errors(instrument)
    if errors:
        raise RuntimeError("Keysight pilot acquisition error: " + " | ".join(errors))
    if len(readings) != expected:
        raise RuntimeError(
            f"pilot returned {len(readings)}/{expected} readings; discard trace"
        )
    invalid_indices = [
        index
        for index, value in enumerate(readings)
        if not math.isfinite(value) or abs(value) >= OVERLOAD_READING_A
    ]
    if invalid_indices:
        first = invalid_indices[0]
        raise RuntimeError(
            f"pilot has {len(invalid_indices)} non-finite/overload readings; "
            f"first at sample {first} ({first * args.sample_interval_s:.3f} s), "
            f"value {readings[first]!r} A. Discard trace."
        )

    output_dir = Path(args.output_dir)
    samples_path = output_dir / f"keysight_34465a_{args.measured_role}_pilot.csv"
    manifest_path = output_dir / "keysight_34465a_pilot_manifest.csv"
    if samples_path.exists() or manifest_path.exists():
        raise FileExistsError("pilot output already exists; choose a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        samples_path,
        [
            {
                "measured_role": args.measured_role,
                "board_id": args.board_id or "",
                "sample_index": index,
                "relative_time_s": index * args.sample_interval_s,
                "current_a": value,
            }
            for index, value in enumerate(readings)
        ],
    )
    late_readings = readings[round(60.0 / args.sample_interval_s) :]
    write_csv_rows(
        manifest_path,
        [
            {
                "capture_profile": "keysight_34465a_immediate_pilot_not_paper_data",
                "measured_role": args.measured_role,
                "board_id": args.board_id or "",
                "instrument_idn": instrument_idn,
                "visa_resource": args.resource,
                "supply_voltage_v": args.supply_voltage_v,
                "source_current_limit_a": args.source_current_limit_a,
                "source_overvoltage_protection_v": args.source_overvoltage_protection_v,
                "maximum_safe_voltage_v": args.maximum_safe_voltage_v,
                "current_range_a": args.current_range_a,
                "current_terminal_a": args.current_terminal_a,
                "current_range_a_readback": readbacks["current_range_a"],
                "current_terminal_a_readback": readbacks["current_terminal_a"],
                "sample_interval_s": args.sample_interval_s,
                "aperture_s": args.aperture_s,
                "reading_count": len(readings),
                "questionable_condition": questionable,
                "median_current_a": statistics.median(readings),
                "minimum_current_a": min(readings),
                "maximum_current_a": max(readings),
                "maximum_absolute_current_a": max(abs(value) for value in readings),
                "range_utilization_fraction": max(abs(value) for value in readings)
                / args.current_range_a,
                "maximum_current_after_60s_a": max(late_readings) if late_readings else "",
                "sample_csv": samples_path.name,
                "sample_csv_sha256": sha256_file(samples_path),
                "interpretation": "unclassified_trace_threshold_requires_idle_and_event_review",
            }
        ],
    )
    print(f"saved {samples_path}")
    print(f"saved {manifest_path}")
    print(
        f"pilot median {1000.0 * statistics.median(readings):.3f} mA; "
        f"range {1000.0 * args.current_range_a:.3f} mA; "
        f"minimum {1000.0 * min(readings):.3f} mA; "
        f"maximum {1000.0 * max(readings):.3f} mA. "
        "Do not use the maximum alone as an event trigger threshold."
    )
    if max(abs(value) for value in readings) >= 0.9 * args.current_range_a:
        next_step = (
            "repeat the pilot on the 100 mA range before paper capture"
            if args.current_range_a == 0.01
            else "stop and investigate before paper capture"
        )
        print(
            "WARNING: current reached at least 90% of the fixed range; "
            f"{next_step}.",
            flush=True,
        )
    return samples_path, manifest_path


def add_capture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resource")
    parser.add_argument("--list-resources", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="record a trigger-free diagnostic trace")
    parser.add_argument(
        "--capture-mode", choices=("stream", "level"), default="stream",
        help="stream timed samples (recommended) or use the DMM's level detector",
    )
    parser.add_argument("--pilot-duration-s", type=float, default=PILOT_DURATION_S)
    parser.add_argument("--visa-library")
    parser.add_argument("--visa-timeout-ms", type=int, default=180_000)
    parser.add_argument("--measured-role", choices=("initiator", "responder"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--pair-id")
    parser.add_argument("--board-id")
    parser.add_argument("--peer-board-id")
    parser.add_argument("--firmware-revision")
    parser.add_argument("--firmware-image-sha256")
    parser.add_argument("--artifact-sha256")
    parser.add_argument("--supply-voltage-v", type=float)
    parser.add_argument("--supply-voltage-source")
    parser.add_argument("--source-current-limit-a", type=float)
    parser.add_argument("--source-overvoltage-protection-v", type=float)
    parser.add_argument("--maximum-safe-voltage-v", type=float)
    parser.add_argument("--current-range-a", type=float, choices=(0.01, 0.1), default=0.01)
    parser.add_argument("--current-terminal-a", type=int, choices=(3,), default=3)
    parser.add_argument("--sample-interval-s", type=float, default=SAMPLE_INTERVAL_S)
    parser.add_argument("--aperture-s", type=float, default=APERTURE_S)
    parser.add_argument("--samples-per-trigger", type=int, default=SAMPLES_PER_TRIGGER)
    parser.add_argument("--pretrigger-samples", type=int, default=PRETRIGGER_SAMPLES)
    parser.add_argument("--trigger-level-a", type=float)
    parser.add_argument("--calibration-id")
    parser.add_argument("--calibration-date")
    parser.add_argument("--capture-timeout-s", type=float, default=1800.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.1)
    parser.add_argument("--confirm-current-fuse", action="store_true")
    parser.add_argument("--confirm-series-current-wiring", action="store_true")
    parser.add_argument("--confirm-measured-battery-removed", action="store_true")
    parser.add_argument("--confirm-no-usb-backpower", action="store_true")
    parser.add_argument("--confirm-peer-separately-powered", action="store_true")
    parser.add_argument("--confirm-trigger-level-tested", action="store_true")


def _open_visa(args: argparse.Namespace) -> tuple[object, object]:
    try:
        import pyvisa  # type: ignore
    except ImportError as error:
        raise RuntimeError("PyVISA is required for live Keysight control") from error
    manager = (
        pyvisa.ResourceManager(args.visa_library)
        if args.visa_library
        else pyvisa.ResourceManager()
    )
    instrument = manager.open_resource(args.resource)
    instrument.timeout = args.visa_timeout_ms
    if hasattr(instrument, "chunk_size"):
        instrument.chunk_size = max(int(instrument.chunk_size), 1024 * 1024)
    return manager, instrument


def run_capture(args: argparse.Namespace) -> int:
    if args.dry_run:
        if args.pilot:
            commands, _ = pilot_configuration_commands(
                duration_s=args.pilot_duration_s,
                sample_interval_s=args.sample_interval_s,
                aperture_s=args.aperture_s,
                current_range_a=args.current_range_a,
                current_terminal_a=args.current_terminal_a,
            )
            for command in commands:
                print(command)
            print("INIT")
            print("DATA:POIN?")
            print("FETC?")
            return 0
        if not args.measured_role:
            raise ValueError("--dry-run requires --measured-role")
        if args.trigger_level_a is None:
            raise ValueError("--dry-run for a paper capture requires --trigger-level-a")
        if args.capture_mode == "stream":
            commands, _ = stream_configuration_commands(
                duration_s=args.capture_timeout_s,
                sample_interval_s=args.sample_interval_s,
                aperture_s=args.aperture_s,
                current_range_a=args.current_range_a,
                current_terminal_a=args.current_terminal_a,
            )
        else:
            commands = capture_configuration_commands(
                args.measured_role,
                sample_interval_s=args.sample_interval_s,
                aperture_s=args.aperture_s,
                samples_per_trigger=args.samples_per_trigger,
                pretrigger_samples=args.pretrigger_samples,
                current_range_a=args.current_range_a,
                current_terminal_a=args.current_terminal_a,
                trigger_level_a=args.trigger_level_a,
            )
        for command in commands:
            print(command)
        print("INIT")
        if args.capture_mode == "stream":
            print("DATA:REMove? <available_count>")
            return 0
        print("*OPC")
        print("*ESR?")
        print("DATA:POIN?")
        print("FETC?")
        return 0
    if args.list_resources:
        try:
            import pyvisa  # type: ignore
        except ImportError as error:
            raise RuntimeError("PyVISA is required to list VISA resources") from error
        manager = (
            pyvisa.ResourceManager(args.visa_library)
            if args.visa_library
            else pyvisa.ResourceManager()
        )
        try:
            for resource in manager.list_resources():
                print(resource)
        finally:
            manager.close()
        return 0
    if not args.resource:
        raise ValueError("--resource is required for probe or capture")
    if args.probe:
        manager, instrument = _open_visa(args)
        try:
            print(verify_keysight_idn(str(instrument.query("*IDN?"))))
        finally:
            instrument.close()
            manager.close()
        return 0

    if args.pilot:
        _require_text(
            args,
            (
                "measured_role",
                "output_dir",
                "supply_voltage_v",
                "source_current_limit_a",
                "source_overvoltage_protection_v",
                "maximum_safe_voltage_v",
            ),
        )
        if args.supply_voltage_v <= 0.0 or args.source_current_limit_a <= 0.0:
            raise ValueError("pilot supply voltage and current limit must be positive")
        if not (
            args.supply_voltage_v
            < args.source_overvoltage_protection_v
            <= args.maximum_safe_voltage_v
        ):
            raise ValueError("pilot requires supply voltage < OVP <= maximum safe voltage")
        pilot_confirmations = (
            "confirm_current_fuse",
            "confirm_series_current_wiring",
            "confirm_measured_battery_removed",
            "confirm_no_usb_backpower",
            "confirm_peer_separately_powered",
        )
        missing = [
            "--" + name.replace("_", "-")
            for name in pilot_confirmations
            if not getattr(args, name)
        ]
        if missing:
            raise ValueError("missing pilot safety confirmation(s): " + ", ".join(missing))
        manager, instrument = _open_visa(args)
        try:
            pilot_keysight(instrument, args)
        except BaseException:
            try:
                instrument.write("ABOR")
            except BaseException:
                pass
            raise
        finally:
            instrument.close()
            manager.close()
        return 0

    _require_text(
        args,
        (
            "measured_role",
            "output_dir",
            "run_id",
            "pair_id",
            "board_id",
            "peer_board_id",
            "firmware_revision",
            "firmware_image_sha256",
            "artifact_sha256",
            "supply_voltage_v",
            "supply_voltage_source",
            "source_current_limit_a",
            "source_overvoltage_protection_v",
            "maximum_safe_voltage_v",
            "trigger_level_a",
            "calibration_id",
            "calibration_date",
        ),
    )
    args.firmware_image_sha256 = _validate_digest(
        args.firmware_image_sha256, "--firmware-image-sha256"
    )
    args.artifact_sha256 = _validate_digest(args.artifact_sha256, "--artifact-sha256")
    if args.supply_voltage_v <= 0.0 or args.source_current_limit_a <= 0.0:
        raise ValueError("supply voltage and source current limit must be positive")
    if not (
        args.supply_voltage_v
        < args.source_overvoltage_protection_v
        <= args.maximum_safe_voltage_v
    ):
        raise ValueError(
            "require supply voltage < source OVP <= the declared maximum safe voltage"
        )
    if not 0.0 < args.trigger_level_a < args.current_range_a:
        raise ValueError("trigger level must be positive and below the fixed current range")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.calibration_date):
        raise ValueError("calibration date must use YYYY-MM-DD")
    if args.capture_timeout_s <= 0.0 or args.poll_interval_s <= 0.0:
        raise ValueError("capture timeout and poll interval must be positive")
    confirmations = (
        "confirm_current_fuse",
        "confirm_series_current_wiring",
        "confirm_measured_battery_removed",
        "confirm_no_usb_backpower",
        "confirm_peer_separately_powered",
        "confirm_trigger_level_tested",
    )
    missing_confirmations = [
        "--" + name.replace("_", "-") for name in confirmations if not getattr(args, name)
    ]
    if missing_confirmations:
        raise ValueError("missing safety confirmation(s): " + ", ".join(missing_confirmations))
    manager, instrument = _open_visa(args)
    try:
        if args.capture_mode == "stream":
            capture_keysight_stream(instrument, args)
        else:
            capture_keysight(instrument, args)
    except BaseException:
        try:
            instrument.write("ABOR")
        except BaseException:
            pass
        raise
    finally:
        instrument.close()
        manager.close()
    return 0


def _float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid or missing {field!r} in Keysight CSV") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field!r} in Keysight CSV")
    return value


def _integer(row: Mapping[str, str], field: str) -> int:
    value = _float(row, field)
    if value != round(value):
        raise ValueError(f"non-integral {field!r} in Keysight CSV")
    return int(round(value))


def _optional_integer(row: Mapping[str, str], field: str, default: int = 0) -> int:
    return _integer(row, field) if str(row.get(field, "")).strip() else default


def _single(rows: Sequence[Mapping[str, str]], field: str) -> str:
    values = {str(row.get(field, "")).strip() for row in rows}
    if len(values) != 1 or not next(iter(values), ""):
        raise ValueError(f"expected one non-empty {field!r}, found {sorted(values)}")
    return values.pop()


def _validate_preflight(
    events_path: Path,
    metadata_path: Path,
    initiator_rows: Sequence[Mapping[str, str]],
    responder_rows: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    events = read_csv_rows(events_path)
    metadata = read_csv_rows(metadata_path)
    initiator_board = _single(initiator_rows, "board_id")
    responder_board = _single(responder_rows, "board_id")
    expected_roles = {initiator_board: "initiator", responder_board: "responder"}
    observed_roles = {
        row.get("board_id", ""): row.get("role", "")
        for row in metadata
        if row.get("board_id", "") in expected_roles
    }
    if observed_roles != expected_roles:
        raise ValueError(
            f"preflight role audit failed: expected {expected_roles}, observed {observed_roles}"
        )
    for capture_rows in (initiator_rows, responder_rows):
        capture_revision = _single(capture_rows, "firmware_revision")
        capture_artifact = _single(capture_rows, "artifact_sha256")
        board = _single(capture_rows, "board_id")
        matches = [row for row in metadata if row.get("board_id") == board]
        if len(matches) != 1:
            raise ValueError(f"preflight metadata must contain exactly one row for {board}")
        if matches[0].get("firmware_revision") != capture_revision:
            raise ValueError(f"preflight/capture firmware revision mismatch for {board}")
        if matches[0].get("artifact_sha256") != capture_artifact:
            raise ValueError(f"preflight/capture artifact mismatch for {board}")
        if matches[0].get("capture_pacing") != "keysight_current_level_trigger_fixed_period":
            raise ValueError(f"{board} preflight did not use the Keysight Figure 4a image")

    required = {
        ("local", "local"): MEASURED_REPETITIONS,
        ("accepted_connected", "initiator"): MEASURED_REPETITIONS,
        ("accepted_connected", "responder"): MEASURED_REPETITIONS,
    }
    counts: dict[tuple[str, str], int] = defaultdict(int)
    failures: dict[tuple[str, str], int] = defaultdict(int)
    for row in events:
        key = (row.get("event_path", ""), row.get("role", ""))
        if key not in required or _is_true(row.get("warmup", "")):
            continue
        counts[key] += 1
        if not _is_true(row.get("success", "")):
            failures[key] += 1
    for key, expected in required.items():
        if counts[key] != expected or failures[key] != 0:
            raise ValueError(
                f"preflight {key[0]}/{key[1]} requires {expected} successful measured "
                f"rows; found {counts[key]} rows and {failures[key]} failures"
            )
    return {
        "preflight_events_sha256": sha256_file(events_path),
        "preflight_metadata_sha256": sha256_file(metadata_path),
        "preflight_status": "same_image_100_successful_trials_per_required_role",
    }


def _validate_role_rows(rows: list[dict[str, str]], measured_role: str) -> None:
    if not rows:
        raise ValueError(f"{measured_role} Keysight CSV is empty")
    capture_profile = _single(rows, "capture_profile")
    if capture_profile not in {
        "keysight_34465a_sequential_current_level_trigger",
        "keysight_34465a_continuous_stream_software_events",
    }:
        raise ValueError(f"{measured_role} CSV has the wrong capture profile")
    if capture_profile == "keysight_34465a_continuous_stream_software_events":
        if _single(rows, "trigger_source") != "immediate_timed_stream_software_threshold":
            raise ValueError(f"{measured_role} stream has the wrong event-detection source")
        if _single(rows, "trigger_source_readback").upper() not in {"IMM", "IMMEDIATE"}:
            raise ValueError(f"{measured_role} DMM did not use immediate timed sampling")
        invalid_count = _optional_integer(rows[0], "raw_invalid_reading_count")
        invalid_indices_text = rows[0].get("raw_invalid_sample_indices", "")
        invalid_indices = (
            [int(value) for value in invalid_indices_text.split(";")]
            if invalid_indices_text
            else []
        )
        if invalid_count != len(invalid_indices) or invalid_indices != sorted(set(invalid_indices)):
            raise ValueError(f"{measured_role} raw invalid-reading audit is inconsistent")
        if any(
            _optional_integer(row, "raw_invalid_reading_count") != invalid_count
            or row.get("raw_invalid_sample_indices", "") != invalid_indices_text
            for row in rows
        ):
            raise ValueError(f"{measured_role} raw invalid-reading audit varies by row")
    if _single(rows, "measured_board_role") != measured_role:
        raise ValueError(f"expected {measured_role} capture rows")
    samples_per_trigger = _integer(rows[0], "samples_per_trigger")
    pretrigger = _integer(rows[0], "pretrigger_samples")
    interval = _float(rows[0], "sample_interval_s")
    expected_readings = validate_capture_settings(
        measured_role,
        interval,
        _float(rows[0], "aperture_s"),
        samples_per_trigger,
        pretrigger,
        _float(rows[0], "current_range_a"),
        _integer(rows[0], "current_terminal_a"),
        _float(rows[0], "trigger_level_a"),
    )
    if len(rows) != expected_readings:
        raise ValueError(
            f"{measured_role} CSV needs {expected_readings} rows, found {len(rows)}"
        )
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        current = _float(row, "current_a")
        if abs(current) >= OVERLOAD_READING_A:
            raise ValueError(f"{measured_role} capture contains an overload reading")
        grouped[_integer(row, "trigger_index")].append(row)
    if sorted(grouped) != list(range(EXPECTED_TRIGGERS[measured_role])):
        raise ValueError(f"{measured_role} trigger sequence is incomplete or non-contiguous")
    prior_onset = -1
    for trigger_index in sorted(grouped):
        trigger_rows = grouped[trigger_index]
        trigger_rows.sort(key=lambda row: _integer(row, "sample_index"))
        if len(trigger_rows) != samples_per_trigger:
            raise ValueError(f"trigger {trigger_index} has the wrong sample count")
        if [_integer(row, "sample_index") for row in trigger_rows] != list(
            range(samples_per_trigger)
        ):
            raise ValueError(f"trigger {trigger_index} sample sequence is invalid")
        if capture_profile == "keysight_34465a_continuous_stream_software_events":
            onset = _integer(trigger_rows[0], "source_sample_index") + pretrigger
            if onset * interval < STREAM_FIRST_EVENT_GUARD_S:
                raise ValueError(f"stream trigger {trigger_index} occurs during startup guard")
            if prior_onset >= 0 and (onset - prior_onset) * interval < STREAM_MIN_EVENT_PERIOD_S:
                raise ValueError(f"stream trigger {trigger_index} violates minimum event spacing")
            prior_onset = onset
        for sample_index, row in enumerate(trigger_rows):
            expected_time = (sample_index - pretrigger) * interval
            if not math.isclose(
                _float(row, "relative_time_s"), expected_time, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"trigger {trigger_index} has an invalid relative timebase")
            if capture_profile == "keysight_34465a_continuous_stream_software_events":
                source_index = _integer(row, "source_sample_index")
                if source_index in invalid_indices:
                    raise ValueError(
                        f"stream trigger {trigger_index} includes an invalid raw reading"
                    )
                if source_index != onset - pretrigger + sample_index or not math.isclose(
                    _float(row, "source_time_s"), source_index * interval,
                    rel_tol=0.0, abs_tol=1e-9,
                ):
                    raise ValueError(f"stream trigger {trigger_index} has an invalid source timebase")
            expected_path, expected_role, path_index = _event_identity(
                measured_role, trigger_index
            )
            if (
                row.get("event_path") != expected_path
                or row.get("event_role") != expected_role
                or _integer(row, "path_index") != path_index
            ):
                raise ValueError(f"trigger {trigger_index} has an invalid path/role label")


def _event_metrics(
    rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    first = rows[0]
    interval = _float(first, "sample_interval_s")
    voltage = _float(first, "supply_voltage_v")
    pre = [row for row in rows if _float(row, "relative_time_s") < 0.0]
    post = [row for row in rows if _float(row, "relative_time_s") >= 0.0]
    if len(pre) != _integer(first, "pretrigger_samples") or not post:
        raise ValueError("trigger record lacks the required pre/post samples")
    baseline_current = statistics.fmean(_float(row, "current_a") for row in pre)
    currents = [_float(row, "current_a") for row in post]
    energy_absolute = voltage * interval * sum(currents)
    energy_incremental = voltage * interval * sum(
        current - baseline_current for current in currents
    )
    metric = {
        "run_id": first.get("run_id", ""),
        "pair_id": first.get("pair_id", ""),
        "board_id": first.get("board_id", ""),
        "measured_board_role": first.get("measured_board_role", ""),
        "event_path": first.get("event_path", ""),
        "role": first.get("event_role", ""),
        "path_index": _integer(first, "path_index"),
        "trial_id": _integer(first, "trial_id"),
        "warmup": _integer(first, "warmup"),
        "supply_voltage_v": voltage,
        "baseline_current_a": baseline_current,
        "baseline_power_w": voltage * baseline_current,
        "window_duration_s": len(post) * interval,
        "energy_absolute_j": energy_absolute,
        "energy_incremental_j": energy_incremental,
        "average_power_w": energy_absolute / (len(post) * interval),
        "peak_sampled_power_w": voltage * max(currents),
        "peak_semantics": f"maximum_{_float(first, 'aperture_s'):.6g}s_aperture_reading",
        "outcome_evidence": first.get("outcome_join", ""),
        "firmware_revision": first.get("firmware_revision", ""),
        "firmware_image_sha256": first.get("firmware_image_sha256", ""),
        "artifact_sha256": first.get("artifact_sha256", ""),
    }
    trace = [
        {
            "event_path": first.get("event_path", ""),
            "role": first.get("event_role", ""),
            "trial_id": _integer(first, "trial_id"),
            "warmup": _integer(first, "warmup"),
            "relative_time_s": _float(row, "relative_time_s"),
            "current_a": _float(row, "current_a"),
            "baseline_current_a": baseline_current,
            "baseline_subtracted_current_a": _float(row, "current_a") - baseline_current,
        }
        for row in rows
    ]
    return metric, trace


def _summaries(metrics: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in metrics:
        if int(row.get("warmup", 0)):
            continue
        kind = "estimated" if row.get("role") == "pair_estimate" else "measured"
        metric_names = ("energy_incremental_j", "energy_absolute_j")
        if kind == "measured":
            metric_names += ("average_power_w", "peak_sampled_power_w")
        for metric_name in metric_names:
            value = row.get(metric_name, "")
            if value != "":
                groups[
                    (str(row["event_path"]), str(row["role"]), kind, metric_name)
                ].append(float(value))
    rows: list[dict[str, object]] = []
    for (event_path, role, kind, metric_name), values in groups.items():
        summary = summarize_values(values)
        rows.append(
            {
                "event_path": event_path,
                "role": role,
                "measurement_kind": kind,
                "metric": metric_name,
                **summary,
            }
        )
    return sorted(
        rows,
        key=lambda row: (str(row["event_path"]), str(row["role"]), str(row["metric"])),
    )


def _idle_summaries(metrics: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    board_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in metrics:
        if int(row.get("warmup", 0)) or row.get("role") == "pair_estimate":
            continue
        value = float(row["baseline_power_w"])
        board_id = str(row["board_id"])
        measured_role = str(row["measured_board_role"])
        groups[(board_id, str(row["event_path"]), str(row["role"]))].append(value)
        board_groups[(board_id, measured_role)].append(value)
    rows = [
        {
            "board_id": board_id,
            "event_path": event_path,
            "role": role,
            "metric": "same_record_pretrigger_baseline_power_w",
            **summarize_values(values),
        }
        for (board_id, event_path, role), values in sorted(groups.items())
    ]
    board_means = {
        key: statistics.fmean(values) for key, values in sorted(board_groups.items())
    }
    rows.extend(
        {
            "board_id": board_id,
            "event_path": "all_connected_pretriggers",
            "role": measured_role,
            "metric": "board_mean_same_record_pretrigger_baseline_power_w",
            **summarize_values(values),
        }
        for (board_id, measured_role), values in sorted(board_groups.items())
    )
    rows.append(
        {
            "board_id": "equal_weight_board_mean",
            "event_path": "all_connected_pretriggers",
            "role": "pair_boards",
            "metric": "idle_power_w",
            **summarize_values(list(board_means.values())),
        }
    )
    return rows


def _nearest_trial(rows: Sequence[Mapping[str, object]], value: float) -> int:
    nearest = min(
        rows, key=lambda row: abs(float(row["energy_incremental_j"]) - value)
    )
    return int(nearest["trial_id"])


def _render_figure(
    traces: Sequence[Mapping[str, object]],
    metrics: Sequence[Mapping[str, object]],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required for PDF/PNG rendering; analysis CSVs were written"
        ) from error

    figure, (trace_axis, energy_axis) = plt.subplots(
        2,
        1,
        figsize=(2.18, 2.18),
        gridspec_kw={"height_ratios": (1.25, 1.0), "hspace": 0.47},
    )
    colors = {"local": "#5B6470", "initiator": "#1976B9", "responder": "#D97706"}
    labels = {"local": "Local", "initiator": "Initiator", "responder": "Responder"}
    for role in ("local", "initiator", "responder"):
        selected = [row for row in traces if row["role"] == role]
        trace_axis.plot(
            [float(row["relative_time_s"]) for row in selected],
            [1000.0 * float(row["baseline_subtracted_current_a"]) for row in selected],
            color=colors[role],
            linewidth=0.85,
            label=labels[role],
        )
    trace_axis.axvline(0.0, color="#20252B", linewidth=0.55, linestyle="--")
    trace_axis.axhline(0.0, color="#A7ADB5", linewidth=0.45)
    trace_axis.set_ylabel(r"$\Delta I$ (mA)", fontsize=5.8)
    trace_axis.set_xlabel("Time from current trigger (s)", fontsize=5.8, labelpad=1.0)
    trace_axis.tick_params(labelsize=5.2, length=2.0, pad=1.0)
    trace_axis.legend(fontsize=4.8, frameon=False, ncol=3, loc="upper right", handlelength=1.3)
    trace_axis.spines[["top", "right"]].set_visible(False)

    categories = (
        ("local", "local", "Local"),
        ("accepted_connected", "initiator", "Init."),
        ("accepted_connected", "responder", "Resp."),
        ("accepted_connected", "pair_estimate", "Pair*"),
    )
    distributions = [
        [
            1000.0 * float(row["energy_incremental_j"])
            for row in metrics
            if not int(row.get("warmup", 0))
            and row["event_path"] == event_path
            and row["role"] == role
        ]
        for event_path, role, _ in categories
    ]
    boxes = energy_axis.boxplot(
        distributions,
        whis=(5, 95),
        showmeans=True,
        widths=0.55,
        patch_artist=True,
        meanprops={
            "marker": "D",
            "markersize": 2.2,
            "markerfacecolor": "white",
            "markeredgecolor": "#20252B",
        },
        flierprops={
            "marker": ".",
            "markersize": 1.5,
            "markerfacecolor": "#6B7280",
            "markeredgecolor": "none",
        },
        medianprops={"color": "white", "linewidth": 0.8},
        whiskerprops={"linewidth": 0.6},
        capprops={"linewidth": 0.6},
    )
    box_colors = ("#5B6470", "#1976B9", "#D97706", "#7D6BB3")
    for patch, color in zip(boxes["boxes"], box_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_linewidth(0.6)
    boxes["boxes"][-1].set_hatch("///")
    energy_axis.set_xticks(range(1, 5), [item[2] for item in categories])
    energy_axis.set_ylabel("Incremental energy (mJ)", fontsize=5.8)
    energy_axis.tick_params(labelsize=5.2, length=2.0, pad=1.0)
    energy_axis.text(
        0.99,
        0.98,
        "*sequential role sum",
        transform=energy_axis.transAxes,
        ha="right",
        va="top",
        fontsize=4.6,
        color="#5B6470",
    )
    energy_axis.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(left=0.24, right=0.98, top=0.98, bottom=0.13)
    for output_format in formats:
        if output_format not in {"pdf", "png"}:
            raise ValueError(f"unsupported Figure 4a format {output_format!r}")
        figure.savefig(
            output_dir / f"ondevice_power.{output_format}",
            dpi=300 if output_format == "png" else None,
        )
    plt.close(figure)


def analyze_keysight_captures(
    initiator_csv: Path,
    responder_csv: Path,
    preflight_events_csv: Path,
    preflight_metadata_csv: Path,
    output_dir: Path,
    formats: Sequence[str] = ("pdf", "png"),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    initiator_rows = read_csv_rows(initiator_csv)
    responder_rows = read_csv_rows(responder_csv)
    _validate_role_rows(initiator_rows, "initiator")
    _validate_role_rows(responder_rows, "responder")
    if _single(initiator_rows, "pair_id") != _single(responder_rows, "pair_id"):
        raise ValueError("initiator and responder captures have different pair IDs")
    if _single(initiator_rows, "firmware_revision") != _single(
        responder_rows, "firmware_revision"
    ):
        raise ValueError("initiator and responder captures use different firmware revisions")
    if _single(initiator_rows, "artifact_sha256") != _single(
        responder_rows, "artifact_sha256"
    ):
        raise ValueError("initiator and responder captures use different model artifacts")
    preflight = _validate_preflight(
        preflight_events_csv,
        preflight_metadata_csv,
        initiator_rows,
        responder_rows,
    )

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for rows in (initiator_rows, responder_rows):
        measured_role = _single(rows, "measured_board_role")
        for row in rows:
            grouped[(measured_role, _integer(row, "trigger_index"))].append(row)
    metrics: list[dict[str, object]] = []
    traces_by_key: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for _, trigger_rows in sorted(grouped.items()):
        trigger_rows.sort(key=lambda row: _integer(row, "sample_index"))
        metric, trace = _event_metrics(trigger_rows)
        metrics.append(metric)
        if not int(metric["warmup"]):
            traces_by_key[
                (str(metric["event_path"]), str(metric["role"]), int(metric["trial_id"]))
            ] = trace

    initiator_measured = {
        int(row["trial_id"]): row
        for row in metrics
        if not int(row["warmup"])
        and row["event_path"] == "accepted_connected"
        and row["role"] == "initiator"
    }
    responder_measured = {
        int(row["trial_id"]): row
        for row in metrics
        if not int(row["warmup"])
        and row["event_path"] == "accepted_connected"
        and row["role"] == "responder"
    }
    if set(initiator_measured) != set(range(MEASURED_REPETITIONS)) or set(
        responder_measured
    ) != set(range(MEASURED_REPETITIONS)):
        raise ValueError("accepted-connected measured trial IDs are incomplete")
    for trial_id in range(MEASURED_REPETITIONS):
        initiator = initiator_measured[trial_id]
        responder = responder_measured[trial_id]
        metrics.append(
            {
                "run_id": f"{initiator['run_id']}+{responder['run_id']}",
                "pair_id": initiator["pair_id"],
                "board_id": "non_simultaneous_role_sum",
                "event_path": "accepted_connected",
                "role": "pair_estimate",
                "path_index": trial_id + WARMUP_REPETITIONS,
                "trial_id": trial_id,
                "warmup": 0,
                "energy_incremental_j": float(initiator["energy_incremental_j"])
                + float(responder["energy_incremental_j"]),
                "energy_absolute_j": float(initiator["energy_absolute_j"])
                + float(responder["energy_absolute_j"]),
                "average_power_w": "",
                "peak_sampled_power_w": "",
                "pair_semantics": "matched_trial_id_sum_of_two_sequential_role_captures",
                "outcome_evidence": "same_image_preflight_only_not_same_encounter",
                "firmware_revision": initiator["firmware_revision"],
                "artifact_sha256": initiator["artifact_sha256"],
            }
        )

    measured_metrics = [row for row in metrics if not int(row.get("warmup", 0))]
    local_measured = [
        row for row in measured_metrics if row["event_path"] == "local" and row["role"] == "local"
    ]
    pair_measured = [row for row in measured_metrics if row["role"] == "pair_estimate"]
    local_trial = _nearest_trial(
        local_measured,
        float(
            summarize_values(
                [float(row["energy_incremental_j"]) for row in local_measured]
            )["median"]
        ),
    )
    pair_trial = _nearest_trial(
        pair_measured,
        float(
            summarize_values(
                [float(row["energy_incremental_j"]) for row in pair_measured]
            )["median"]
        ),
    )
    representative_traces = [
        *traces_by_key[("local", "local", local_trial)],
        *traces_by_key[("accepted_connected", "initiator", pair_trial)],
        *traces_by_key[("accepted_connected", "responder", pair_trial)],
    ]
    summaries = _summaries(metrics)
    idle_summaries = _idle_summaries(metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "keysight_event_energy.csv", metrics)
    write_csv_rows(output_dir / "keysight_energy_summary.csv", summaries)
    write_csv_rows(output_dir / "keysight_idle_summary.csv", idle_summaries)
    write_csv_rows(output_dir / "figure4a_trace.csv", representative_traces)
    audit = {
        "status": "pass_with_declared_sequential_pair_limitation",
        "initiator_sample_csv": str(initiator_csv),
        "initiator_sample_csv_sha256": sha256_file(initiator_csv),
        "responder_sample_csv": str(responder_csv),
        "responder_sample_csv_sha256": sha256_file(responder_csv),
        "initiator_trigger_count": EXPECTED_TRIGGERS["initiator"],
        "responder_trigger_count": EXPECTED_TRIGGERS["responder"],
        "measured_repetitions_per_category": MEASURED_REPETITIONS,
        "pair_energy_semantics": "matched_trial_id_sum_of_two_sequential_role_captures",
        "pair_peak_power": "not_reported_not_observable_with_one_dmm",
        "event_outcome_limitation": "preflight_same_image_not_exact_power_trial_join",
        "energy_window_semantics": "fixed_4.1s_post_trigger_baseline_subtracted_window",
        "trigger_alignment_uncertainty_s": SAMPLE_INTERVAL_S,
        "initiator_raw_invalid_reading_count": _optional_integer(
            initiator_rows[0], "raw_invalid_reading_count"
        ),
        "responder_raw_invalid_reading_count": _optional_integer(
            responder_rows[0], "raw_invalid_reading_count"
        ),
        "raw_invalid_reading_policy": "preserved_in_raw_excluded_from_all_energy_windows",
        **preflight,
    }
    write_csv_rows(output_dir / "keysight_capture_audit.csv", [audit])
    if formats:
        _render_figure(representative_traces, metrics, output_dir, formats)
    return metrics, summaries


def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initiator-csv", type=Path, required=True)
    parser.add_argument("--responder-csv", type=Path, required=True)
    parser.add_argument("--preflight-events-csv", type=Path, required=True)
    parser.add_argument("--preflight-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="comma-separated pdf,png, or none (figure-ready CSVs are always written)",
    )


def run_analysis(args: argparse.Namespace) -> int:
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    if formats == ["none"]:
        formats = []
    analyze_keysight_captures(
        args.initiator_csv,
        args.responder_csv,
        args.preflight_events_csv,
        args.preflight_metadata_csv,
        args.output_dir,
        formats,
    )
    return 0
