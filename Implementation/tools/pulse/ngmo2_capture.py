"""Safe R&S NGMO2 two-channel capture and envelope-energy analysis.

The primary controller mode is autonomous: the debug boards and FPCs remain
disconnected, the NGMO2 powers both targets, and one declared firmware event
schedule runs per power cycle.  One ``*TRG`` gives both current arrays a shared
sample clock.  This capture-only evidence is deliberately not labeled as an
exact firmware-record join.

Channel 1 (A) force/sense supplies and measures board 1 at its battery pads;
channel 2 (B) does the same for board 2.  Batteries, debug hardware/FPCs, and
series DMMs are absent from the measured circuit.

An optional UART-paced READY/GO/DUMP mode is retained for diagnostics, but is
valid only after physically isolating the debug-board CHG_AC/VBUS feed.  It is
not the primary power-measurement method.

PyVISA and pyserial are optional runtime dependencies.  They are imported only
for real hardware access; command planning, dry runs, import, and tests remain
dependency-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Protocol

from common import (
    PULSE_BUILD_CONTEXT_FIELDS,
    PULSE_EVENT_CONFIGURATION_FIELDS,
    parse_bool,
    parse_float,
    read_csv_rows,
    row_success,
    sha256_file,
    write_csv_rows,
)
from summarize import accepted_session_integrity


NGMO2_MANUAL_URL = (
    "https://scdn.rohde-schwarz.com/ur/pws/dl_downloads/dl_common_library/"
    "dl_manuals/dl_user_manual/NGMO_OperatingManual_en_04.pdf"
)
FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
CONTROL_PREFIX_RE = re.compile(r"(PULSE_READY|PULSE_EVENT|PULSE_STATUS)\b(.*)$")
SAFE_IDN_MANUFACTURERS = ("ROHDE", "SCHWARZ", "R&S")


class ScpiTransport(Protocol):
    def write(self, command: str) -> object: ...

    def query(self, command: str) -> str: ...

    def close(self) -> object: ...


def _pulse_hash32(value: str) -> str:
    """Match pulse_metrics_hash32 (32-bit FNV-1a) for proof provenance."""

    result = 2166136261
    for byte in value.encode("utf-8"):
        result ^= byte
        result = (result * 16777619) & 0xFFFFFFFF
    return f"{result:08x}"


def validate_capture_configuration(
    sample_interval_s: float, sample_count: int
) -> None:
    if not 1.0e-5 <= sample_interval_s <= 1.0:
        raise ValueError("NGMO2 sample interval must be between 1e-5 and 1 s")
    steps = sample_interval_s / 1.0e-5
    if abs(steps - round(steps)) > 1.0e-7:
        raise ValueError("NGMO2 sample interval must be an integer multiple of 10 us")
    if not 1 <= sample_count <= 5000:
        raise ValueError("NGMO2 sample count must be between 1 and 5000")


def capture_configuration_commands(
    sample_interval_s: float, sample_count: int
) -> list[str]:
    """Return the exact NGMO v4 manual command sequence for both channels."""

    validate_capture_configuration(sample_interval_s, sample_count)
    commands = ["FORMat:DATA ASCii"]
    for channel in (1, 2):
        prefix = f"SENSe{channel}"
        commands.extend(
            (
                f"{prefix}:FUNCtion AVERage",
                f"{prefix}:CURRent:RANGe MEDium",
                f"{prefix}:PULSe:CHANnel CURRent",
                f"{prefix}:PULSe:TRIGger:SOURce INT",
                f"{prefix}:PULSe:TRIGger:COUNt 1",
                f"{prefix}:PULSe:TRIGger:OFFSet 0",
                f"{prefix}:PULSe:SAMPle:LENGth {sample_count}",
                f"{prefix}:PULSe:SAMPle:INTerval {sample_interval_s:.9g}",
            )
        )
    return commands


def capture_configuration_queries() -> list[str]:
    queries: list[str] = []
    for channel in (1, 2):
        prefix = f"SENSe{channel}"
        queries.extend(
            (
                f"{prefix}:FUNCtion?",
                f"{prefix}:CURRent:RANGe?",
                f"{prefix}:PULSe:CHANnel?",
                f"{prefix}:PULSe:TRIGger:SOURce?",
                f"{prefix}:PULSe:TRIGger:COUNt?",
                f"{prefix}:PULSe:SAMPle:LENGth?",
                f"{prefix}:PULSe:SAMPle:INTerval?",
            )
        )
    return queries


def verify_ngmo2_idn(value: str) -> str:
    normalized = value.strip()
    upper = normalized.upper()
    if "NGMO2" not in upper or not any(name in upper for name in SAFE_IDN_MANUFACTURERS):
        raise ValueError(
            "refusing instrument control: *IDN? must identify a Rohde & Schwarz NGMO2; "
            f"received {normalized!r}"
        )
    return normalized


def parse_ascii_array(response: str, expected_count: int) -> list[float]:
    values = [float(token) for token in FLOAT_RE.findall(response)]
    if len(values) != expected_count:
        raise ValueError(
            f"NGMO2 returned {len(values)} samples; expected {expected_count}"
        )
    return values


def _query_number(instrument: ScpiTransport, command: str) -> float:
    value = parse_float(instrument.query(command))
    if value is None:
        raise ValueError(f"NGMO2 returned a non-numeric response to {command}")
    return value


def _system_error_is_clear(value: str) -> bool:
    match = re.match(r"\s*([+-]?\d+)", value)
    return bool(match and int(match.group(1)) == 0)


def configure_capture(
    instrument: ScpiTransport, sample_interval_s: float, sample_count: int
) -> list[str]:
    commands = capture_configuration_commands(sample_interval_s, sample_count)
    for command in commands:
        instrument.write(command)
    instrument.query("*OPC?")

    for channel in (1, 2):
        prefix = f"SENSe{channel}"
        text_expectations = {
            f"{prefix}:FUNCtion?": "AVER",
            f"{prefix}:CURRent:RANGe?": "MED",
            f"{prefix}:PULSe:CHANnel?": "CURR",
            f"{prefix}:PULSe:TRIGger:SOURce?": "INT",
        }
        for query, expected in text_expectations.items():
            response = instrument.query(query).strip().upper()
            if expected not in response:
                raise ValueError(
                    f"NGMO2 setting verification failed for {query}: {response!r}"
                )
        if int(round(_query_number(instrument, f"{prefix}:PULSe:TRIGger:COUNt?"))) != 1:
            raise ValueError(f"NGMO2 channel {channel} trigger count did not read back as 1")
        if int(round(_query_number(instrument, f"{prefix}:PULSe:SAMPle:LENGth?"))) != sample_count:
            raise ValueError(f"NGMO2 channel {channel} sample length readback mismatch")
        actual_interval = _query_number(
            instrument, f"{prefix}:PULSe:SAMPle:INTerval?"
        )
        if abs(actual_interval - sample_interval_s) > 5.1e-6:
            raise ValueError(f"NGMO2 channel {channel} sample interval readback mismatch")
    error = instrument.query("SYSTem:ERRor?")
    if not _system_error_is_clear(error):
        raise ValueError(f"NGMO2 reported an error after capture setup: {error.strip()}")
    return commands


def _output_off(instrument: ScpiTransport) -> None:
    for channel in (1, 2):
        try:
            instrument.write(f"OUTPut{channel} OFF")
        except Exception:
            pass


def _set_output_state(instrument: ScpiTransport, enabled: bool) -> None:
    state = "ON" if enabled else "OFF"
    try:
        for channel in (1, 2):
            instrument.write(f"OUTPut{channel} {state}")
        instrument.query("*OPC?")
        for channel in (1, 2):
            actual = instrument.query(f"OUTPut{channel}?").strip().upper()
            expected = {"1", "ON"} if enabled else {"0", "OFF"}
            if actual not in expected:
                raise ValueError(
                    f"NGMO2 channel {channel} output did not read back as {state}"
                )
    except Exception:
        _output_off(instrument)
        raise


def configure_outputs(
    instrument: ScpiTransport,
    voltage_v: float,
    current_limit_a: float,
    maximum_safe_voltage_v: float,
    *,
    overvoltage_protection_v: float,
    enable: bool,
    turn_on: bool = True,
) -> None:
    if voltage_v <= 0 or maximum_safe_voltage_v <= 0:
        raise ValueError("voltage and maximum-safe voltage must be positive")
    if voltage_v > maximum_safe_voltage_v:
        raise ValueError(
            f"requested {voltage_v:g} V exceeds the operator-declared safe maximum "
            f"of {maximum_safe_voltage_v:g} V"
        )
    if (
        not 1.5 <= overvoltage_protection_v <= 22.0
        or abs(overvoltage_protection_v * 10 - round(overvoltage_protection_v * 10))
        > 1.0e-9
        or not voltage_v < overvoltage_protection_v <= maximum_safe_voltage_v
    ):
        raise ValueError(
            "overvoltage protection must be a 0.1 V step in the NGMO2 1.5--22 V "
            "range, above the supply voltage, and at or below the operator-declared "
            "maximum-safe voltage"
        )
    if not 0 < current_limit_a <= 5.0:
        raise ValueError("NGMO2 current limit must be greater than 0 and at most 5 A")
    if enable:
        instrument.write("CONFIG:COMMon:OUTPut:ONOFf OFF")
        for channel in (1, 2):
            instrument.write(
                f"SOURce{channel}:VOLTage:MAXSetting {maximum_safe_voltage_v:.9g}"
            )
            instrument.write(
                f"SOURce{channel}:VOLTage:PROTection {overvoltage_protection_v:.9g}"
            )
            instrument.write(f"OUTPut{channel}:IMPedance 0")
            instrument.write(
                f"SOURce{channel}:CURRent:LIMit:MAXSetting {current_limit_a:.9g}"
            )
            instrument.write(f"SOURce{channel}:CURRent:LIMit:TYPE LIMIT")
            instrument.write(f"SOURce{channel}:VOLTage {voltage_v:.9g}")
            instrument.write(f"SOURce{channel}:CURRent {current_limit_a:.9g}")
        instrument.query("*OPC?")
    common_output = instrument.query("CONFIG:COMMon:OUTPut:ONOFf?").strip().upper()
    if common_output not in {"0", "OFF"}:
        raise ValueError("NGMO2 common output coupling is not OFF")
    for channel in (1, 2):
        actual_maximum = _query_number(
            instrument, f"SOURce{channel}:VOLTage:MAXSetting?"
        )
        actual_protection = _query_number(
            instrument, f"SOURce{channel}:VOLTage:PROTection?"
        )
        actual_impedance = _query_number(instrument, f"OUTPut{channel}:IMPedance?")
        actual_limit_type = instrument.query(
            f"SOURce{channel}:CURRent:LIMit:TYPE?"
        ).strip().upper()
        actual_limit_maximum = _query_number(
            instrument, f"SOURce{channel}:CURRent:LIMit:MAXSetting?"
        )
        actual_voltage = _query_number(instrument, f"SOURce{channel}:VOLTage?")
        actual_limit = _query_number(instrument, f"SOURce{channel}:CURRent?")
        if abs(actual_maximum - maximum_safe_voltage_v) > 0.002:
            raise ValueError(
                f"NGMO2 channel {channel} maximum-voltage setting mismatch"
            )
        if abs(actual_protection - overvoltage_protection_v) > 0.002:
            raise ValueError(
                f"NGMO2 channel {channel} overvoltage-protection mismatch"
            )
        if abs(actual_impedance) > 1.0e-9:
            raise ValueError(f"NGMO2 channel {channel} output impedance is not 0 ohm")
        if "LIMIT" not in actual_limit_type:
            raise ValueError(
                f"NGMO2 channel {channel} current-limit type is not LIMIT"
            )
        if abs(actual_limit_maximum - current_limit_a) > 0.002:
            raise ValueError(
                f"NGMO2 channel {channel} maximum current-limit setting mismatch"
            )
        if abs(actual_voltage - voltage_v) > 0.002:
            raise ValueError(f"NGMO2 channel {channel} voltage setpoint mismatch")
        if abs(actual_limit - current_limit_a) > 0.002:
            raise ValueError(f"NGMO2 channel {channel} current-limit mismatch")
    if enable:
        try:
            _set_output_state(instrument, turn_on)
        except Exception:
            _output_off(instrument)
            raise
    for channel in (1, 2):
        output_state = instrument.query(f"OUTPut{channel}?").strip().upper()
        expected_states = {"1", "ON"} if enable and turn_on else {"0", "OFF"}
        if output_state not in expected_states:
            raise ValueError(
                f"NGMO2 channel {channel} output state verification failed; "
                f"expected {'ON' if enable and turn_on else 'OFF'}"
            )


def current_limit_states(instrument: ScpiTransport) -> dict[int, bool]:
    """Read the documented current-limiting/trip state for both channels."""

    states: dict[int, bool] = {}
    for channel in (1, 2):
        response = instrument.query(
            f"SOURce{channel}:CURRent:LIMit:STATe?"
        ).strip().upper()
        if response in {"0", "OFF"}:
            states[channel] = False
        elif response in {"1", "ON"}:
            states[channel] = True
        else:
            raise ValueError(
                f"NGMO2 channel {channel} returned unknown current-limit state {response!r}"
            )
    return states


def measure_output_voltages(
    instrument: ScpiTransport,
    expected_voltage_v: float,
    maximum_deviation_v: float,
) -> dict[int, float]:
    """Read both channel output voltages and enforce an operator-set tolerance.

    The final protocol keeps the front-panel local-sense jumpers installed. These
    readings therefore represent the NGMO2 channel output, not the voltage at the
    board pads; external lead loss remains inside the supplied-energy boundary.
    """

    if maximum_deviation_v <= 0:
        raise ValueError("maximum output-voltage deviation must be positive")
    measured = {
        channel: _query_number(instrument, f"MEASure{channel}:VOLTage?")
        for channel in (1, 2)
    }
    for channel, value in measured.items():
        if abs(value - expected_voltage_v) > maximum_deviation_v:
            raise ValueError(
                f"NGMO2 channel {channel} output-voltage reading {value:g} V differs from "
                f"the setpoint by more than {maximum_deviation_v:g} V"
            )
    error = instrument.query("SYSTem:ERRor?")
    if not _system_error_is_clear(error):
        raise ValueError(
            f"NGMO2 reported an error after output-voltage queries: {error.strip()}"
        )
    return measured


def acquire_synchronized_arrays(
    instrument: ScpiTransport,
    sample_count: int,
    capture_timeout_s: float,
    *,
    poll_interval_s: float = 0.05,
    trigger_bracket: dict[str, float] | None = None,
) -> tuple[list[float], list[float]]:
    """Software-trigger both channels once and fetch their completed arrays."""

    if trigger_bracket is not None:
        trigger_bracket["before_monotonic_s"] = time.monotonic()
    instrument.write("*TRG")
    if trigger_bracket is not None:
        trigger_bracket["after_monotonic_s"] = time.monotonic()
    deadline = time.monotonic() + capture_timeout_s
    while True:
        states = [
            instrument.query(f"SENSe{channel}:PULSe:TRIGger:STATe?").strip().upper()
            for channel in (1, 2)
        ]
        if any("TIMEOUT" in state for state in states):
            raise RuntimeError(f"NGMO2 acquisition timed out: states={states}")
        if all("READY" in state for state in states):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"host timed out waiting for NGMO2 arrays: states={states}")
        time.sleep(poll_interval_s)
    channel_a = parse_ascii_array(instrument.query("FETCh1:ARRay?"), sample_count)
    channel_b = parse_ascii_array(instrument.query("FETCh2:ARRay?"), sample_count)
    error = instrument.query("SYSTem:ERRor?")
    if not _system_error_is_clear(error):
        raise RuntimeError(f"NGMO2 reported an acquisition error: {error.strip()}")
    return channel_a, channel_b


def _parse_control_line(line: str) -> tuple[str, dict[str, str]] | None:
    match = CONTROL_PREFIX_RE.search(line)
    if not match:
        return None
    fields: dict[str, str] = {}
    for token in match.group(2).lstrip(" ,:").split(","):
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key.strip()] = value.strip()
    return match.group(1), fields


class _BoardSerial:
    def __init__(self, serial_handle: object, log_path: Path):
        self.serial = serial_handle
        self.log_path = log_path
        self.log = log_path.open("x", encoding="utf-8", newline="")

    def close(self) -> None:
        self.log.close()
        self.serial.close()

    def command(self, text: str) -> None:
        self.serial.write((text + "\n").encode("ascii"))
        self.serial.flush()

    def line(self, deadline: float) -> str:
        while time.monotonic() < deadline:
            raw = self.serial.readline()
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            self.log.write(text + "\n")
            self.log.flush()
            return text
        raise TimeoutError(f"timed out waiting for firmware output in {self.log_path}")

    def ready_or_complete(self, timeout_s: float) -> dict[str, str] | None:
        deadline = time.monotonic() + timeout_s
        while True:
            parsed = _parse_control_line(self.line(deadline))
            if parsed is None:
                continue
            prefix, fields = parsed
            if prefix == "PULSE_READY":
                return fields
            if prefix == "PULSE_STATUS" and fields.get("state") in {
                "campaign_complete",
                "campaign_failed",
            }:
                if fields.get("state") == "campaign_failed":
                    raise RuntimeError(f"firmware campaign failed: {fields}")
                return None

    def event(
        self,
        expected: Mapping[str, str],
        timeout_s: float,
        *,
        match_event_index: bool = True,
        expected_role: str | None = None,
    ) -> dict[str, str]:
        deadline = time.monotonic() + timeout_s
        while True:
            parsed = _parse_control_line(self.line(deadline))
            if parsed is None or parsed[0] != "PULSE_EVENT":
                continue
            fields = parsed[1]
            # Responders decode their event_index from the session ID.  The
            # cross-board identity is exchange_id; event_index is only valid
            # for joining an initiator's READY line to its own EVENT line.
            identity_fields = (
                ("event_index", "exchange_id")
                if match_event_index
                else ("exchange_id",)
            )
            if all(
                not expected.get(field)
                or fields.get(field, "") == expected.get(field, "")
                for field in identity_fields
            ) and (expected_role is None or fields.get("role") == expected_role):
                return fields


def _open_board_serial(port: str, baud: int, log_path: Path) -> _BoardSerial:
    try:
        import serial  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "pyserial is required for live board UART pacing; install it in the "
            "capture environment"
        ) from error
    handle = serial.Serial(port=port, baudrate=baud, bytesize=8, parity="N", stopbits=1, timeout=0.1)
    return _BoardSerial(handle, log_path)


def _visa_termination(value: str | None) -> str | None:
    if value is None:
        return None
    return {"cr": "\r", "lf": "\n"}[value]


def open_visa_transport(args: argparse.Namespace) -> tuple[object, ScpiTransport]:
    try:
        import pyvisa  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "PyVISA plus a suitable VISA backend are required for live NGMO2 access"
        ) from error
    manager = pyvisa.ResourceManager(args.visa_library) if args.visa_library else pyvisa.ResourceManager()
    resource = str(args.resource)
    kwargs: dict[str, object] = {"timeout": int(args.visa_timeout_ms)}
    if resource.upper().startswith("ASRL"):
        required = {
            "visa_baud": args.visa_baud,
            "visa_data_bits": args.visa_data_bits,
            "visa_parity": args.visa_parity,
            "visa_stop_bits": args.visa_stop_bits,
            "visa_flow_control": args.visa_flow_control,
            "visa_termination": args.visa_termination,
        }
        missing = [name.replace("visa_", "--visa-").replace("_", "-") for name, value in required.items() if value is None]
        if missing:
            manager.close()
            raise ValueError(
                "ASRL settings are not safely assumed; provide " + ", ".join(missing)
            )
        kwargs.update(
            {
                "baud_rate": args.visa_baud,
                "data_bits": args.visa_data_bits,
                "parity": getattr(pyvisa.constants.Parity, args.visa_parity),
                "stop_bits": getattr(pyvisa.constants.StopBits, args.visa_stop_bits),
                "flow_control": getattr(pyvisa.constants.ControlFlow, args.visa_flow_control),
                "read_termination": _visa_termination(args.visa_termination),
                "write_termination": _visa_termination(args.visa_termination),
            }
        )
    instrument = manager.open_resource(resource, **kwargs)
    return manager, instrument


def _safe_capture_id(sequence: int, ready: Mapping[str, str]) -> str:
    identity = "_".join(
        str(ready.get(field, ""))
        for field in ("run_id", "pair_id", "event_index", "exchange_id")
    )
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", identity).strip("_")
    return f"c{sequence:04d}_{cleaned or 'unidentified'}"


def _write_raw_capture(
    path: Path,
    channel_1: list[float],
    channel_2: list[float],
    sample_interval_s: float,
    voltage_v: float,
    initiator_channel: int,
) -> None:
    if initiator_channel not in (1, 2):
        raise ValueError("initiator_channel must be 1 or 2")
    rows = [
        {
            "sample_index": index,
            "sample_time_s": (index + 0.5) * sample_interval_s,
            "sample_interval_s": sample_interval_s,
            "channel_1_current_a": current_1,
            "channel_2_current_a": current_2,
            "initiator_current_a": current_1
            if initiator_channel == 1
            else current_2,
            "responder_current_a": current_2
            if initiator_channel == 1
            else current_1,
            "initiator_voltage_v": voltage_v,
            "responder_voltage_v": voltage_v,
        }
        for index, (current_1, current_2) in enumerate(zip(channel_1, channel_2))
    ]
    write_csv_rows(
        path,
        rows,
        preferred=[
            "sample_index",
            "sample_time_s",
            "sample_interval_s",
            "channel_1_current_a",
            "channel_2_current_a",
            "initiator_current_a",
            "responder_current_a",
            "initiator_voltage_v",
            "responder_voltage_v",
        ],
    )


def capture_campaign(
    args: argparse.Namespace,
    instrument: ScpiTransport,
    initiator: _BoardSerial | None = None,
    responder: _BoardSerial | None = None,
) -> list[dict[str, object]]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ngmo2_capture_manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {manifest_path}")
    owns_initiator = initiator is None
    owns_responder = responder is None
    if initiator is None:
        initiator = _open_board_serial(
            args.initiator_port, args.board_baud, output_dir / "initiator_serial.log"
        )
    if responder is None:
        responder = _open_board_serial(
            args.responder_port, args.board_baud, output_dir / "responder_serial.log"
        )
    configuration = capture_configuration_commands(args.sample_interval_s, args.sample_count)
    configuration_sha256 = hashlib.sha256(
        json.dumps(configuration, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    rows: list[dict[str, object]] = []
    capture_duration_s = args.sample_interval_s * args.sample_count
    try:
        while args.capture_count is None or len(rows) < args.capture_count:
            ready = initiator.ready_or_complete(args.firmware_timeout_s)
            if ready is None:
                break
            sequence = len(rows) + 1
            capture_id = _safe_capture_id(sequence, ready)
            raw_path = output_dir / f"{capture_id}.csv"
            if raw_path.exists():
                raise FileExistsError(f"refusing to overwrite existing {raw_path}")
            triggered_at = datetime.now(timezone.utc).isoformat()
            instrument.write("*TRG")
            initiator.command("GO")
            deadline = time.monotonic() + capture_duration_s + args.instrument_timeout_margin_s
            while True:
                states = [
                    instrument.query(f"SENSe{channel}:PULSe:TRIGger:STATe?").strip().upper()
                    for channel in (1, 2)
                ]
                if any("TIMEOUT" in state for state in states):
                    raise RuntimeError(f"NGMO2 acquisition {capture_id} timed out: {states}")
                if all("READY" in state for state in states):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"host timed out waiting for NGMO2 acquisition {capture_id}: {states}"
                    )
                time.sleep(args.poll_interval_s)
            channel_a = parse_ascii_array(
                instrument.query("FETCh1:ARRay?"), args.sample_count
            )
            channel_b = parse_ascii_array(
                instrument.query("FETCh2:ARRay?"), args.sample_count
            )
            _write_raw_capture(
                raw_path,
                channel_a,
                channel_b,
                args.sample_interval_s,
                args.voltage_v,
                args.initiator_channel,
            )
            initiator.command("DUMP")
            initiator_event = initiator.event(ready, args.firmware_timeout_s)
            remote_started = parse_bool(initiator_event.get("remote_session_started"))
            responder_event: dict[str, str] = {}
            if remote_started is True:
                responder.command("DUMP")
                responder_event = responder.event(
                    initiator_event,
                    args.firmware_timeout_s,
                    match_event_index=False,
                    expected_role="responder",
                )
            channel_1_role = (
                "initiator" if args.initiator_channel == 1 else "responder"
            )
            channel_2_role = (
                "responder" if args.initiator_channel == 1 else "initiator"
            )
            row: dict[str, object] = {
                **ready,
                "capture_id": capture_id,
                "capture_profile": "ngmo2_uart_paced_envelope",
                "instrument_resource": args.resource,
                "instrument_idn": args.instrument_idn,
                "manual_url": NGMO2_MANUAL_URL,
                "trigger_command": "*TRG",
                "initiator_channel": args.initiator_channel,
                "channel_1_role": channel_1_role,
                "channel_2_role": channel_2_role,
                "channel_a_role": channel_1_role,
                "channel_b_role": channel_2_role,
                "sample_interval_s": args.sample_interval_s,
                "sample_count": args.sample_count,
                "capture_duration_s": capture_duration_s,
                "current_range_a": 0.5,
                "current_range_setting": "MEDium",
                "current_resolution_a": 1.0e-5,
                "current_full_scale_deviation_a": 1.0e-3,
                "voltage_v": args.voltage_v,
                "voltage_basis": "verified_programmed_setpoint_not_dynamic_voltage",
                "current_limit_a": args.current_limit_a,
                "trigger_host_utc": triggered_at,
                "capture_configuration_sha256": configuration_sha256,
                "canonical_csv": str(raw_path),
                "canonical_csv_sha256": sha256_file(raw_path),
                "initiator_event_success": initiator_event.get("success", ""),
                "remote_session_started": initiator_event.get("remote_session_started", ""),
                "capture_envelope_duration_us": initiator_event.get(
                    "capture_envelope_duration_us", ""
                ),
                "responder_event_present": int(bool(responder_event)),
                "responder_board_id": responder_event.get("board_id", ""),
                "responder_event_index": responder_event.get("event_index", ""),
                "responder_event_success": responder_event.get("success", ""),
                "uart_power_isolation_confirmed": 1,
                "no_parallel_battery_confirmed": 1,
            }
            rows.append(row)
            write_csv_rows(
                manifest_path,
                rows,
                preferred=[
                    "capture_id",
                    "run_id",
                    "pair_id",
                    "board_id",
                    "event_index",
                    "exchange_id",
                    "trial_id",
                    "event_path",
                    "role",
                    "warmup",
                    "capture_profile",
                    "instrument_resource",
                    "instrument_idn",
                    "trigger_command",
                    "initiator_channel",
                    "channel_1_role",
                    "channel_2_role",
                    "channel_a_role",
                    "channel_b_role",
                    "sample_interval_s",
                    "sample_count",
                    "capture_duration_s",
                    "current_range_a",
                    "current_range_setting",
                    "current_resolution_a",
                    "current_full_scale_deviation_a",
                    "voltage_v",
                    "voltage_basis",
                    "current_limit_a",
                    "capture_envelope_duration_us",
                    "initiator_event_success",
                    "remote_session_started",
                    "responder_event_present",
                    "responder_board_id",
                    "responder_event_index",
                    "responder_event_success",
                    "canonical_csv",
                    "canonical_csv_sha256",
                    "capture_configuration_sha256",
                    "trigger_host_utc",
                    "uart_power_isolation_confirmed",
                    "no_parallel_battery_confirmed",
                    "manual_url",
                ],
            )
    finally:
        if owns_initiator:
            initiator.close()
        if owns_responder:
            responder.close()
    return rows


def capture_autonomous_campaign(
    args: argparse.Namespace,
    instrument: ScpiTransport,
) -> list[dict[str, object]]:
    """Capture one operator-declared autonomous event schedule per power cycle.

    This mode has no live target transport.  It binds raw samples to the exact
    NGMO2 configuration and operator-declared schedule, but intentionally does
    not claim that a firmware event record was observed or joined.
    """

    if args.capture_count != 105:
        raise ValueError("the final autonomous protocol requires --capture-count 105")
    if args.firmware_sequence_start != 0:
        raise ValueError(
            "the final autonomous protocol requires --firmware-sequence-start 0"
        )
    if args.boot_schedule_uncertainty_s is None or args.boot_schedule_uncertainty_s <= 0:
        raise ValueError(
            "autonomous capture requires a positive preflight-bounded "
            "--boot-schedule-uncertainty-s"
        )
    for label, value in (
        ("initiator", args.initiator_firmware_sha256),
        ("responder", args.responder_firmware_sha256),
    ):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "")):
            raise ValueError(f"{label} firmware SHA-256 must be 64 hexadecimal digits")
    if not str(args.firmware_revision or "").strip():
        raise ValueError("autonomous capture requires --firmware-revision")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(args.artifact_sha256 or "")):
        raise ValueError("--artifact-sha256 must contain 64 hexadecimal digits")
    if args.autonomous_trigger_delay_s < 0 or args.autonomous_power_off_s < 0:
        raise ValueError("autonomous trigger and power-off delays must be non-negative")
    if args.autonomous_post_capture_s < 0:
        raise ValueError("--autonomous-post-capture-s must be non-negative")
    if args.max_output_voltage_deviation_v is None or args.max_output_voltage_deviation_v <= 0:
        raise ValueError(
            "autonomous capture requires a positive --max-output-voltage-deviation-v"
        )
    if not 0 <= args.autonomous_event_dispatch_s < args.autonomous_event_deadline_s:
        raise ValueError(
            "autonomous event dispatch/completion deadline must be ordered "
            "positive times relative to output enable"
        )
    capture_duration_s = args.sample_interval_s * args.sample_count
    event_dispatch_in_capture_s = (
        args.autonomous_event_dispatch_s - args.autonomous_trigger_delay_s
    )
    event_deadline_in_capture_s = (
        args.autonomous_event_deadline_s - args.autonomous_trigger_delay_s
    )
    if event_dispatch_in_capture_s < 0:
        raise ValueError(
            "the declared autonomous event dispatches before the NGMO2 trigger; "
            "reduce --autonomous-trigger-delay-s"
        )
    if event_deadline_in_capture_s > capture_duration_s:
        raise ValueError(
            "the declared autonomous event deadline is outside the NGMO2 buffer; "
            "increase sample interval/count or shorten the schedule"
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ngmo2_capture_manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {manifest_path}")
    configuration = capture_configuration_commands(
        args.sample_interval_s, args.sample_count
    )
    configuration_sha256 = hashlib.sha256(
        json.dumps(configuration, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    channel_1_role = "initiator" if args.initiator_channel == 1 else "responder"
    channel_2_role = "responder" if args.initiator_channel == 1 else "initiator"
    logical_role = (
        "local" if args.event_path in {"local", "no_contact"} else "initiator"
    )
    rows: list[dict[str, object]] = []

    for offset in range(args.capture_count):
        sequence = offset + 1
        firmware_sequence = args.firmware_sequence_start + offset
        event_index = args.event_index_start + offset
        expected_warmup = firmware_sequence < 5
        trial_id = firmware_sequence if expected_warmup else firmware_sequence - 5
        exchange_id = (
            firmware_sequence + 1
            if args.event_path == "accepted_connected"
            else 105 + firmware_sequence + 1
            if args.event_path == "accepted_discovery"
            else 0
        )
        ready = {
            "run_id": args.run_id,
            "pair_id": args.pair_id,
            "board_id": args.initiator_board_id,
            "event_index": str(event_index),
            "exchange_id": str(exchange_id),
        }
        capture_id = _safe_capture_id(sequence, ready)
        raw_path = output_dir / f"{capture_id}.csv"
        if raw_path.exists():
            raise FileExistsError(f"refusing to overwrite existing {raw_path}")
        try:
            power_anchor_monotonic = time.monotonic()
            power_anchor_utc = datetime.now(timezone.utc).isoformat()
            _set_output_state(instrument, True)
            outputs_verified_monotonic = time.monotonic()
            powered_at = datetime.now(timezone.utc).isoformat()
            output_voltage_pre = measure_output_voltages(
                instrument,
                args.voltage_v,
                args.max_output_voltage_deviation_v,
            )
            limit_state_pre = current_limit_states(instrument)
            if any(limit_state_pre.values()):
                raise RuntimeError("NGMO2 current limiting was active before acquisition")
            # A direct voltage query should not alter pulse acquisition, but
            # reapply and verify the current-array configuration fail-closed.
            configure_capture(instrument, args.sample_interval_s, args.sample_count)
            trigger_target = power_anchor_monotonic + args.autonomous_trigger_delay_s
            trigger_wait_s = trigger_target - time.monotonic()
            if args.autonomous_trigger_delay_s > 0 and trigger_wait_s < 0:
                raise RuntimeError(
                    "NGMO2 pre-trigger output/voltage/configuration checks overran the "
                    "absolute autonomous trigger deadline"
                )
            if trigger_wait_s > 0:
                time.sleep(trigger_wait_s)
            triggered_at = datetime.now(timezone.utc).isoformat()
            trigger_bracket: dict[str, float] = {}
            channel_1, channel_2 = acquire_synchronized_arrays(
                instrument,
                args.sample_count,
                capture_duration_s + args.instrument_timeout_margin_s,
                poll_interval_s=args.poll_interval_s,
                trigger_bracket=trigger_bracket,
            )
            _write_raw_capture(
                raw_path,
                channel_1,
                channel_2,
                args.sample_interval_s,
                args.voltage_v,
                args.initiator_channel,
            )
            if args.autonomous_post_capture_s:
                time.sleep(args.autonomous_post_capture_s)
            output_voltage_post = measure_output_voltages(
                instrument,
                args.voltage_v,
                args.max_output_voltage_deviation_v,
            )
            limit_state_post = current_limit_states(instrument)
            if any(limit_state_post.values()):
                raise RuntimeError("NGMO2 current limiting was active after acquisition")
        finally:
            _output_off(instrument)

        output_enable_verification_span_s = (
            outputs_verified_monotonic - power_anchor_monotonic
        )
        trigger_offset_earliest_s = (
            trigger_bracket["before_monotonic_s"] - power_anchor_monotonic
        )
        trigger_offset_latest_s = (
            trigger_bracket["after_monotonic_s"] - power_anchor_monotonic
        )
        conservative_dispatch_in_capture_s = (
            args.autonomous_event_dispatch_s
            - trigger_offset_latest_s
            - args.boot_schedule_uncertainty_s
        )
        conservative_deadline_in_capture_s = (
            args.autonomous_event_deadline_s
            + output_enable_verification_span_s
            + args.boot_schedule_uncertainty_s
            - trigger_offset_earliest_s
        )
        if conservative_dispatch_in_capture_s <= 0:
            raise RuntimeError(
                "actual trigger timing leaves no proven pre-dispatch quiet region"
            )
        if conservative_deadline_in_capture_s >= capture_duration_s:
            raise RuntimeError(
                "actual trigger/output-enable timing leaves no proven post-completion "
                "quiet region"
            )

        row: dict[str, object] = {
            **ready,
            "trial_id": trial_id,
            "capture_ordinal": sequence,
            "firmware_sequence": firmware_sequence,
            "event_path": args.event_path,
            "role": logical_role,
            "capture_id": capture_id,
            "capture_profile": "ngmo2_autonomous_one_event_per_power_cycle",
            "capture_evidence": "operator_declared_schedule_no_live_firmware_join",
            "join_status": "not_joined_capture_only_operator_schedule",
            "instrument_resource": args.resource,
            "instrument_idn": args.instrument_idn,
            "manual_url": NGMO2_MANUAL_URL,
            "trigger_command": "*TRG",
            "initiator_channel": args.initiator_channel,
            "channel_1_role": channel_1_role,
            "channel_2_role": channel_2_role,
            "channel_a_role": channel_1_role,
            "channel_b_role": channel_2_role,
            "initiator_board_id": args.initiator_board_id,
            "responder_board_id": args.responder_board_id,
            "initiator_firmware_sha256": args.initiator_firmware_sha256,
            "responder_firmware_sha256": args.responder_firmware_sha256,
            "firmware_revision": args.firmware_revision,
            "artifact_sha256": str(args.artifact_sha256).lower(),
            "expected_firmware_revision_hash": _pulse_hash32(args.firmware_revision),
            "expected_artifact_hash": _pulse_hash32(str(args.artifact_sha256).lower()),
            "sample_interval_s": args.sample_interval_s,
            "sample_count": args.sample_count,
            "capture_duration_s": capture_duration_s,
            "current_range_a": 0.5,
            "current_range_limit_a": 0.510,
            "current_range_setting": "MEDium",
            "current_resolution_a": 1.0e-5,
            "current_full_scale_deviation_a": 1.0e-3,
            "voltage_v": args.voltage_v,
            "voltage_basis": "local_sensed_ngmo2_output_pre_post_readback",
            "supplied_energy_boundary": "ngmo2_output_includes_external_lead_loss",
            "sense_mode": args.sense_mode,
            "channel_1_lead_length_m": args.channel_1_lead_length_m,
            "channel_2_lead_length_m": args.channel_2_lead_length_m,
            "channel_1_lead_gauge_awg": args.channel_1_lead_gauge_awg,
            "channel_2_lead_gauge_awg": args.channel_2_lead_gauge_awg,
            "channel_1_cable_id": args.channel_1_cable_id,
            "channel_2_cable_id": args.channel_2_cable_id,
            "calibration_id": args.calibration_id,
            "calibration_date": args.calibration_date,
            "calibration_current_confirmed": 1,
            "output_impedance_ohm": 0,
            "output_impedance_basis": "scpi_programmed_and_read_back",
            "maximum_safe_voltage_v": args.maximum_safe_voltage_v,
            "overvoltage_protection_v": args.overvoltage_protection_v,
            "output_impedance_programmed_ohm": 0,
            "common_output_coupling": "OFF",
            "current_limit_type": "LIMIT",
            "current_limit_maximum_setting_readback_verified": 1,
            "channel_1_current_limit_state_pre": int(limit_state_pre[1]),
            "channel_2_current_limit_state_pre": int(limit_state_pre[2]),
            "channel_1_current_limit_state_post": int(limit_state_post[1]),
            "channel_2_current_limit_state_post": int(limit_state_post[2]),
            "maximum_voltage_setting_readback_verified": 1,
            "max_output_voltage_deviation_v": args.max_output_voltage_deviation_v,
            "channel_1_output_voltage_pre_v": output_voltage_pre[1],
            "channel_2_output_voltage_pre_v": output_voltage_pre[2],
            "channel_1_output_voltage_post_v": output_voltage_post[1],
            "channel_2_output_voltage_post_v": output_voltage_post[2],
            "current_limit_a": args.current_limit_a,
            "current_limit_rejection_margin_a": args.current_limit_rejection_margin_a,
            "outputs_on_readback_host_utc": powered_at,
            "power_enable_anchor_host_utc": power_anchor_utc,
            "power_reference_semantics": (
                "monotonic_anchor_before_first_sequential_output_on_command"
            ),
            "output_enable_verification_span_s": output_enable_verification_span_s,
            "boot_schedule_uncertainty_s": args.boot_schedule_uncertainty_s,
            "trigger_host_utc": triggered_at,
            "trigger_offset_earliest_after_power_anchor_s": trigger_offset_earliest_s,
            "trigger_offset_latest_after_power_anchor_s": trigger_offset_latest_s,
            "trigger_command_host_bracket_s": (
                trigger_offset_latest_s - trigger_offset_earliest_s
            ),
            "autonomous_trigger_delay_s": args.autonomous_trigger_delay_s,
            "capture_window_start_after_power_on_s": trigger_offset_earliest_s,
            "capture_window_end_after_power_on_s": trigger_offset_latest_s
            + capture_duration_s,
            "event_dispatch_after_power_on_s": args.autonomous_event_dispatch_s,
            "event_completion_deadline_after_power_on_s": (
                args.autonomous_event_deadline_s
            ),
            "event_dispatch_in_capture_s": conservative_dispatch_in_capture_s,
            "event_completion_deadline_in_capture_s": conservative_deadline_in_capture_s,
            "guard_bound_semantics": (
                "earliest_dispatch_uses_latest_trigger;latest_completion_uses_"
                "earliest_trigger_plus_output_enable_verification_span;both_"
                "include_explicit_preflight_schedule_uncertainty"
            ),
            "autonomous_post_capture_s": args.autonomous_post_capture_s,
            "autonomous_power_off_s": args.autonomous_power_off_s,
            "debug_boards_disconnected_confirmed": 1,
            "no_parallel_battery_confirmed": 1,
            "capture_configuration_sha256": configuration_sha256,
            "visa_timeout_ms": args.visa_timeout_ms,
            "canonical_csv": str(raw_path),
            "canonical_csv_sha256": sha256_file(raw_path),
            "analysis_eligibility": (
                "requires_later_firmware_record_recovery_and_schedule_integrity_audit"
            ),
        }
        rows.append(row)
        write_csv_rows(manifest_path, rows)
        if sequence < args.capture_count and args.autonomous_power_off_s:
            time.sleep(args.autonomous_power_off_s)
    return rows


STATIC_CURRENT_RANGES = {
    "low": {
        "command": "LOW",
        "nominal_a": 0.005,
        "limit_a": 0.0051,
        "resolution_a": 1.0e-7,
        "full_scale_deviation_a": 1.0e-5,
    },
    "medium": {
        "command": "MEDium",
        "nominal_a": 0.5,
        "limit_a": 0.510,
        "resolution_a": 1.0e-5,
        "full_scale_deviation_a": 1.0e-3,
    },
}


def static_idle_configuration_commands(
    measure_interval_s: float,
    average_count: int,
    current_range: str,
) -> list[str]:
    """Return the documented NGMO2 static-current setup for one explicit range."""

    if measure_interval_s <= 0:
        raise ValueError("static idle measurement interval must be positive")
    if average_count <= 0:
        raise ValueError("static idle average count must be positive")
    if current_range not in STATIC_CURRENT_RANGES:
        raise ValueError("static idle current range must be 'low' or 'medium'")
    range_command = str(STATIC_CURRENT_RANGES[current_range]["command"])
    commands = ["FORMat:DATA ASCii"]
    for channel in (1, 2):
        prefix = f"SENSe{channel}"
        commands.extend(
            (
                f"{prefix}:FUNCtion AVERage",
                f"{prefix}:CURRent:RANGe {range_command}",
                f"{prefix}:MEASure:INTerval {measure_interval_s:.9g}",
                f"{prefix}:AVERage:COUNt {average_count}",
            )
        )
    return commands


def configure_static_idle(
    instrument: ScpiTransport,
    measure_interval_s: float,
    average_count: int,
    current_range: str,
) -> list[str]:
    commands = static_idle_configuration_commands(
        measure_interval_s, average_count, current_range
    )
    for command in commands:
        instrument.write(command)
    instrument.query("*OPC?")
    for channel in (1, 2):
        prefix = f"SENSe{channel}"
        if "AVER" not in instrument.query(f"{prefix}:FUNCtion?").strip().upper():
            raise ValueError(f"NGMO2 channel {channel} function is not AVERage")
        expected_range = str(STATIC_CURRENT_RANGES[current_range]["command"]).upper()
        actual_range = instrument.query(f"{prefix}:CURRent:RANGe?").strip().upper()
        if expected_range not in actual_range:
            raise ValueError(
                f"NGMO2 channel {channel} current range is not {expected_range}"
            )
        actual_interval = _query_number(
            instrument, f"{prefix}:MEASure:INTerval?"
        )
        if not abs(actual_interval - measure_interval_s) <= max(
            1.0e-9, measure_interval_s * 1.0e-6
        ):
            raise ValueError(
                f"NGMO2 channel {channel} static measurement interval mismatch"
            )
        actual_count = _query_number(instrument, f"{prefix}:AVERage:COUNt?")
        if int(round(actual_count)) != average_count:
            raise ValueError(f"NGMO2 channel {channel} average-count mismatch")
    error = instrument.query("SYSTem:ERRor?")
    if not _system_error_is_clear(error):
        raise ValueError(f"NGMO2 reported an error after idle setup: {error.strip()}")
    return commands


def capture_static_idle(
    args: argparse.Namespace,
    instrument: ScpiTransport,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Measure stable per-state idle current on an explicit static range."""

    if args.idle_samples <= 0:
        raise ValueError("--idle-samples must be positive")
    for label, value in (
        ("initiator", args.initiator_firmware_sha256),
        ("responder", args.responder_firmware_sha256),
    ):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "")):
            raise ValueError(f"{label} firmware SHA-256 must be 64 hexadecimal digits")
    if args.idle_settle_s is None or args.idle_settle_s < 0:
        raise ValueError("static idle capture requires non-negative --idle-settle-s")
    if not str(args.idle_state_evidence or "").strip():
        raise ValueError(
            "static idle capture requires --idle-state-evidence; channel state labels "
            "alone do not prove that firmware reached those states"
        )
    single_static_query_s = args.idle_measure_interval_s * args.idle_average_count
    estimated_current_sample_window_s = (
        2.0
        * args.idle_samples
        * single_static_query_s
    )
    # Include two pre- and two post-current output-voltage queries. Instrument
    # configuration/readback overhead is then checked against the actual host
    # monotonic deadline during the powered run.
    estimated_powered_measurement_window_s = (
        estimated_current_sample_window_s + 4.0 * single_static_query_s
    )
    if args.idle_event_dispatch_s is not None:
        if args.idle_event_dispatch_s <= 0:
            raise ValueError("--idle-event-dispatch-s must be positive")
        if (
            args.idle_settle_s + estimated_powered_measurement_window_s
            >= args.idle_event_dispatch_s
        ):
            raise ValueError(
                "static idle settle plus conservative sample window overlaps the "
                "declared scheduled event"
            )
    if args.max_output_voltage_deviation_v is None or args.max_output_voltage_deviation_v <= 0:
        raise ValueError(
            "static idle capture requires a positive --max-output-voltage-deviation-v"
        )
    if args.idle_current_range not in STATIC_CURRENT_RANGES:
        raise ValueError("static idle capture requires --idle-current-range low or medium")
    range_spec = STATIC_CURRENT_RANGES[args.idle_current_range]
    range_limit_a = float(range_spec["limit_a"])
    if (
        args.idle_overload_threshold_a is None
        or not 0 < args.idle_overload_threshold_a <= range_limit_a
    ):
        raise ValueError(
            "--idle-overload-threshold-a must be positive and no greater than "
            f"{range_limit_a:g} A for the selected range"
        )
    output_dir: Path = args.output_dir
    samples_path = output_dir / "ngmo2_idle_current.csv"
    summary_path = output_dir / "ngmo2_idle_summary.csv"
    metadata_path = output_dir / "ngmo2_idle_metadata.csv"
    existing = [
        path for path in (samples_path, summary_path, metadata_path) if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite idle outputs: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_roles = {
        1: "initiator" if args.initiator_channel == 1 else "responder",
        2: "responder" if args.initiator_channel == 1 else "initiator",
    }
    board_ids = {
        "initiator": args.initiator_board_id,
        "responder": args.responder_board_id,
    }
    states = {1: args.channel_1_state, 2: args.channel_2_state}
    samples: list[dict[str, object]] = []
    overloads: list[str] = []
    try:
        power_anchor_monotonic = time.monotonic()

        def require_idle_deadline_budget(remaining_measurement_s: float) -> None:
            if args.idle_event_dispatch_s is None:
                return
            projected = (
                time.monotonic()
                - power_anchor_monotonic
                + remaining_measurement_s
            )
            if projected >= args.idle_event_dispatch_s:
                raise RuntimeError(
                    "static measurement no longer has conservative budget before "
                    "the declared autonomous event dispatch deadline"
                )

        _set_output_state(instrument, True)
        powered_at = datetime.now(timezone.utc).isoformat()
        require_idle_deadline_budget(
            args.idle_settle_s + estimated_powered_measurement_window_s
        )
        output_voltage_pre = measure_output_voltages(
            instrument,
            args.voltage_v,
            args.max_output_voltage_deviation_v,
        )
        limit_state_pre = current_limit_states(instrument)
        if any(limit_state_pre.values()):
            raise RuntimeError("NGMO2 current limiting was active before idle samples")
        configure_static_idle(
            instrument,
            args.idle_measure_interval_s,
            args.idle_average_count,
            args.idle_current_range,
        )
        if args.idle_settle_s:
            time.sleep(args.idle_settle_s)
        for sample_index in range(args.idle_samples):
            for channel in (1, 2):
                remaining_current_queries = 2 * args.idle_samples - (
                    2 * sample_index + channel - 1
                )
                require_idle_deadline_budget(
                    (remaining_current_queries + 2) * single_static_query_s
                )
                measured_at = datetime.now(timezone.utc).isoformat()
                current = _query_number(instrument, f"MEASure{channel}:CURRent?")
                overloaded = abs(current) >= args.idle_overload_threshold_a
                if overloaded:
                    overloads.append(
                        f"channel {channel} sample {sample_index} current={current:.9g} A"
                    )
                role = channel_roles[channel]
                samples.append(
                    {
                        "run_id": args.run_id,
                        "pair_id": args.pair_id,
                        "sample_index": sample_index,
                        "channel": channel,
                        "role": role,
                        "board_id": board_ids[role],
                        "background_state": states[channel],
                        "current_a": current,
                        "power_w": "",
                        "voltage_v": "",
                        "measurement_scope": "ngmo2_static_idle_explicit_range",
                        "current_range_setting": range_spec["command"],
                        "current_range_limit_a": range_limit_a,
                        "sample_timing": "sequential_channel_queries_not_synchronized",
                        "overload": int(overloaded),
                        "measured_host_utc": measured_at,
                    }
                )
        require_idle_deadline_budget(2 * single_static_query_s)
        output_voltage_post = measure_output_voltages(
            instrument,
            args.voltage_v,
            args.max_output_voltage_deviation_v,
        )
        limit_state_post = current_limit_states(instrument)
        if any(limit_state_post.values()):
            raise RuntimeError("NGMO2 current limiting was active after idle samples")
        powered_measurement_elapsed_s = time.monotonic() - power_anchor_monotonic
        if (
            args.idle_event_dispatch_s is not None
            and powered_measurement_elapsed_s >= args.idle_event_dispatch_s
        ):
            raise RuntimeError(
                "actual static measurement reached the declared autonomous event "
                "dispatch deadline"
            )
    finally:
        _output_off(instrument)
    channel_voltage_means = {
        channel: (output_voltage_pre[channel] + output_voltage_post[channel]) / 2.0
        for channel in (1, 2)
    }
    for sample in samples:
        channel = int(sample["channel"])
        measured_voltage = channel_voltage_means[channel]
        sample["voltage_v"] = measured_voltage
        sample["power_w"] = float(sample["current_a"]) * measured_voltage
        sample["voltage_basis"] = "mean_pre_post_local_sensed_ngmo2_output_readback"
    write_csv_rows(samples_path, samples)

    summary: list[dict[str, object]] = []
    for channel in (1, 2):
        channel_samples = [
            float(row["current_a"])
            for row in samples
            if int(row["channel"]) == channel
        ]
        role = channel_roles[channel]
        summary.append(
            {
                "run_id": args.run_id,
                "pair_id": args.pair_id,
                "channel": channel,
                "role": role,
                "board_id": board_ids[role],
                "background_state": states[channel],
                "n": len(channel_samples),
                "mean_current_a": statistics.fmean(channel_samples),
                "median_current_a": statistics.median(channel_samples),
                "min_current_a": min(channel_samples),
                "max_current_a": max(channel_samples),
                "stdev_current_a": statistics.stdev(channel_samples)
                if len(channel_samples) > 1
                else 0.0,
                "mean_power_w": statistics.fmean(channel_samples)
                * channel_voltage_means[channel],
                "median_power_w": statistics.median(channel_samples)
                * channel_voltage_means[channel],
                "voltage_v": channel_voltage_means[channel],
                "voltage_basis": "mean_pre_post_local_sensed_ngmo2_output_readback",
                "measurement_scope": "ngmo2_static_idle_explicit_range",
                "overload_count": sum(
                    int(row["overload"])
                    for row in samples
                    if int(row["channel"]) == channel
                ),
            }
        )
    write_csv_rows(summary_path, summary)
    configuration = static_idle_configuration_commands(
        args.idle_measure_interval_s,
        args.idle_average_count,
        args.idle_current_range,
    )
    metadata = [
        {
            "run_id": args.run_id,
            "pair_id": args.pair_id,
            "instrument_resource": args.resource,
            "instrument_idn": args.instrument_idn,
            "initiator_firmware_sha256": args.initiator_firmware_sha256,
            "responder_firmware_sha256": args.responder_firmware_sha256,
            "manual_url": NGMO2_MANUAL_URL,
            "capture_profile": "ngmo2_static_idle_explicit_range",
            "current_range_setting": range_spec["command"],
            "current_range_nominal_a": range_spec["nominal_a"],
            "current_range_limit_a": range_limit_a,
            "current_resolution_a": range_spec["resolution_a"],
            "current_full_scale_deviation_a": range_spec[
                "full_scale_deviation_a"
            ],
            "overload_threshold_a": args.idle_overload_threshold_a,
            "measure_interval_s": args.idle_measure_interval_s,
            "average_count": args.idle_average_count,
            "samples_per_channel": args.idle_samples,
            "settle_s": args.idle_settle_s,
            "estimated_current_sample_window_s": estimated_current_sample_window_s,
            "estimated_powered_measurement_window_s": (
                estimated_powered_measurement_window_s
            ),
            "actual_powered_measurement_elapsed_s": powered_measurement_elapsed_s,
            "idle_event_dispatch_margin_s": (
                args.idle_event_dispatch_s - powered_measurement_elapsed_s
                if args.idle_event_dispatch_s is not None
                else ""
            ),
            "idle_event_dispatch_s": args.idle_event_dispatch_s
            if args.idle_event_dispatch_s is not None
            else "",
            "idle_state_evidence": args.idle_state_evidence,
            "state_semantics": "operator_preflight_evidence_not_inferred_from_labels",
            "programmed_voltage_v": args.voltage_v,
            "voltage_v": statistics.fmean(channel_voltage_means.values()),
            "voltage_basis": "local_sensed_ngmo2_output_pre_post_readback",
            "supplied_energy_boundary": "ngmo2_output_includes_external_lead_loss",
            "sense_mode": args.sense_mode,
            "channel_1_lead_length_m": args.channel_1_lead_length_m,
            "channel_2_lead_length_m": args.channel_2_lead_length_m,
            "channel_1_lead_gauge_awg": args.channel_1_lead_gauge_awg,
            "channel_2_lead_gauge_awg": args.channel_2_lead_gauge_awg,
            "channel_1_cable_id": args.channel_1_cable_id,
            "channel_2_cable_id": args.channel_2_cable_id,
            "calibration_id": args.calibration_id,
            "calibration_date": args.calibration_date,
            "calibration_current_confirmed": 1,
            "output_impedance_ohm": 0,
            "output_impedance_basis": "scpi_programmed_and_read_back",
            "maximum_safe_voltage_v": args.maximum_safe_voltage_v,
            "overvoltage_protection_v": args.overvoltage_protection_v,
            "output_impedance_programmed_ohm": 0,
            "common_output_coupling": "OFF",
            "current_limit_type": "LIMIT",
            "current_limit_maximum_setting_readback_verified": 1,
            "channel_1_current_limit_state_pre": int(limit_state_pre[1]),
            "channel_2_current_limit_state_pre": int(limit_state_pre[2]),
            "channel_1_current_limit_state_post": int(limit_state_post[1]),
            "channel_2_current_limit_state_post": int(limit_state_post[2]),
            "maximum_voltage_setting_readback_verified": 1,
            "max_output_voltage_deviation_v": args.max_output_voltage_deviation_v,
            "channel_1_output_voltage_pre_v": output_voltage_pre[1],
            "channel_2_output_voltage_pre_v": output_voltage_pre[2],
            "channel_1_output_voltage_post_v": output_voltage_post[1],
            "channel_2_output_voltage_post_v": output_voltage_post[2],
            "channel_1_mean_output_voltage_v": channel_voltage_means[1],
            "channel_2_mean_output_voltage_v": channel_voltage_means[2],
            "current_limit_a": args.current_limit_a,
            "current_limit_rejection_margin_a": args.current_limit_rejection_margin_a,
            "outputs_on_readback_host_utc": powered_at,
            "power_reference_semantics": (
                "host_timestamp_after_both_output_on_readbacks_channels_enabled_sequentially"
            ),
            "sample_timing": "sequential_channel_queries_not_synchronized",
            "configuration_commands_sha256": hashlib.sha256(
                json.dumps(configuration, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "samples_csv": str(samples_path),
            "samples_csv_sha256": sha256_file(samples_path),
            "summary_csv": str(summary_path),
            "summary_csv_sha256": sha256_file(summary_path),
            "debug_boards_disconnected_confirmed": 1,
            "no_parallel_battery_confirmed": 1,
            "status": "fail_overload" if overloads else "pass",
            "issues": " | ".join(overloads),
        }
    ]
    write_csv_rows(metadata_path, metadata)
    if overloads:
        raise ValueError(
            f"NGMO2 {range_spec['command']}-range idle capture overloaded; "
            "do not use it as a battery "
            f"baseline: {' | '.join(overloads[:4])}; see {metadata_path}"
        )
    return samples, summary


def _event_identity(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return tuple(
        str(row.get(field, "")).strip()
        for field in ("run_id", "pair_id", "board_id", "event_index")
    )  # type: ignore[return-value]


def _guard_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("quiet guard contains no samples")
    return sum(values) / len(values)


def analyze_envelope_captures(
    manifest_csv: Path,
    events_csv: Path,
    metadata_csv: Path | None,
    output_dir: Path,
    *,
    max_guard_drift_fraction: float = 0.10,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Integrate markerless NGMO2 captures using firmware envelope duration."""

    manifest = read_csv_rows(manifest_csv)
    events = read_csv_rows(events_csv)
    # The serial parser has already attached metadata context to each event.
    # Never use an unjoined metadata row as a fallback for event identity.
    if metadata_csv is not None and not metadata_csv.exists():
        raise FileNotFoundError(metadata_csv)
    event_groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for event in events:
        event_groups.setdefault(_event_identity(event), []).append(event)
    accepted_initiators, accepted_responders, accepted_issues = accepted_session_integrity(events)
    metrics: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []

    for entry in manifest:
        capture_id = str(entry.get("capture_id", ""))
        issues: list[str] = []
        raw_path = Path(str(entry.get("canonical_csv", "")))
        if not raw_path.is_absolute():
            raw_path = manifest_csv.parent / raw_path
        expected_hash = str(entry.get("canonical_csv_sha256", "")).lower()
        if not raw_path.is_file() or sha256_file(raw_path) != expected_hash:
            issues.append("canonical_csv_hash_not_bound")
            samples: list[dict[str, str]] = []
        else:
            samples = read_csv_rows(raw_path)
        identity = _event_identity(entry)
        initiator_matches = [
            row
            for row in event_groups.get(identity, [])
            if str(row.get("role", "")) in {"initiator", "local"}
        ]
        if len(initiator_matches) != 1:
            issues.append(f"expected_one_initiator_event_found_{len(initiator_matches)}")
            initiator: dict[str, str] = {}
        else:
            initiator = initiator_matches[0]
        if str(initiator.get("exchange_id", "")) != str(entry.get("exchange_id", "")):
            issues.append("exchange_id_not_bound")
        accepted_key = tuple(
            str(initiator.get(field, "")).strip()
            for field in ("run_id", "pair_id", "exchange_id")
        )
        responder_event = accepted_responders.get(accepted_key)
        if str(initiator.get("event_path", "")) in {
            "accepted_connected",
            "accepted_discovery",
        } and accepted_key not in accepted_initiators:
            issues.append("accepted_initiator_not_in_exact_session_index")
        if accepted_issues:
            issues.append("accepted_session_integrity_failed")
        interval = parse_float(entry.get("sample_interval_s"))
        expected_count = parse_float(entry.get("sample_count"))
        voltage = parse_float(entry.get("voltage_v"))
        active_us = parse_float(initiator.get("capture_envelope_duration_us"))
        guard_ms = parse_float(initiator.get("capture_quiet_guard_ms"))
        if interval is None or interval <= 0 or expected_count is None:
            issues.append("invalid_sample_configuration")
        if len(samples) != int(expected_count or 0):
            issues.append("sample_count_mismatch")
        if voltage is None or voltage <= 0:
            issues.append("invalid_voltage")
        if active_us is None or active_us <= 0:
            issues.append("missing_capture_envelope_duration_us")
        if guard_ms is None or guard_ms <= 0:
            issues.append("missing_capture_quiet_guard_ms")
        capture_duration = (interval or 0.0) * len(samples)
        active_s = (active_us or 0.0) / 1.0e6
        guard_s = (guard_ms or 0.0) / 1000.0
        if capture_duration + 1e-12 < active_s + 2.0 * guard_s:
            issues.append("capture_does_not_cover_pre_and_post_guards")
        guard_samples = int(guard_s / (interval or 1.0))
        if guard_samples < 2 or 2 * guard_samples > len(samples):
            issues.append("insufficient_quiet_guard_samples")

        per_role: dict[str, dict[str, float]] = {}
        if not issues:
            for role in ("initiator", "responder"):
                current_field = f"{role}_current_a"
                currents = [parse_float(sample.get(current_field)) for sample in samples]
                if any(value is None for value in currents):
                    issues.append(f"invalid_{role}_current_sample")
                    continue
                numeric = [float(value) for value in currents if value is not None]
                pre = _guard_mean(numeric[:guard_samples])
                post = _guard_mean(numeric[-guard_samples:])
                idle = (pre + post) / 2.0
                scale = max(abs(idle), 1.0e-12)
                drift = abs(post - pre) / scale
                if drift > max_guard_drift_fraction:
                    issues.append(f"{role}_guard_drift_{drift:.6g}")
                charge_capture = sum(numeric) * float(interval)
                charge_incremental = charge_capture - idle * capture_duration
                energy_incremental = charge_incremental * float(voltage)
                energy_total = energy_incremental + idle * float(voltage) * active_s
                per_role[role] = {
                    "baseline_current_a": idle,
                    "pre_guard_current_a": pre,
                    "post_guard_current_a": post,
                    "guard_drift_fraction": drift,
                    "charge_capture_c": charge_capture,
                    "charge_incremental_c": charge_incremental,
                    "energy_incremental_j": energy_incremental,
                    "energy_total_j": energy_total,
                    "average_power_w": energy_total / active_s,
                    "peak_current_a": max(numeric),
                    "peak_power_w": max(numeric) * float(voltage),
                }

        status = "pass" if not issues else "fail"
        audit.append(
            {
                "capture_id": capture_id,
                "run_id": entry.get("run_id", ""),
                "pair_id": entry.get("pair_id", ""),
                "event_index": entry.get("event_index", ""),
                "event_path": entry.get("event_path", ""),
                "audit_type": "ngmo2_markerless_envelope",
                "status": status,
                "issue_count": len(issues),
                "issues": " | ".join(issues),
                "canonical_csv": str(raw_path),
                "canonical_csv_sha256": expected_hash,
                "manifest_source_file": str(manifest_csv),
                "manifest_source_sha256": sha256_file(manifest_csv),
            }
        )
        if issues:
            continue

        role_rows: list[dict[str, object]] = []
        for role in ("initiator", "responder"):
            serial_event_present = role == "initiator" or responder_event is not None
            event = initiator if role == "initiator" else (responder_event or {})
            row: dict[str, object] = {
                field: event.get(field, "")
                for field in (
                    "run_id",
                    "pair_id",
                    "board_id",
                    "trial_id",
                    "exchange_id",
                    "event_index",
                    "event_path",
                    "connection_state",
                    "post_warmup",
                    "warmup",
                    "remote_session_started",
                    "success",
                    *PULSE_BUILD_CONTEXT_FIELDS,
                    *PULSE_EVENT_CONFIGURATION_FIELDS,
                )
            }
            row.update(per_role[role])
            row.update(
                {
                    "capture_id": capture_id,
                    "role": str(event.get("role", ""))
                    if serial_event_present
                    else "responder_idle_pair_envelope",
                    "logical_role": role,
                    "serial_event_present": int(serial_event_present),
                    "power_channel_role": role,
                    "measurement_scope": "ngmo2_capture_envelope",
                    "capture_profile": "ngmo2_uart_paced_envelope",
                    "duration_s": active_s,
                    "capture_duration_s": capture_duration,
                    "baseline_power_w": per_role[role]["baseline_current_a"] * float(voltage),
                    "baseline_source": "same_capture_event_offset_not_battery_idle",
                    "baseline_method": "mean_of_interval_averaged_pre_and_post_guard_current",
                    "nominal_sample_rate_hz": 1.0 / float(interval),
                    "mean_voltage_v": float(voltage),
                    "voltage_basis": entry.get("voltage_basis", ""),
                    "join_status": "ngmo2_exact_ready_event_id",
                    "quality_valid": 1,
                    "source_file": str(raw_path),
                    "source_sha256": expected_hash,
                }
            )
            role_rows.append(row)
            metrics.append(row)
        pair = dict(role_rows[0])
        pair.update(
            {
                "board_id": "pair",
                "role": "pair",
                "logical_role": "pair",
                "power_channel_role": "pair",
                "energy_total_j": sum(float(row["energy_total_j"]) for row in role_rows),
                "energy_incremental_j": sum(float(row["energy_incremental_j"]) for row in role_rows),
                "charge_capture_c": sum(float(row["charge_capture_c"]) for row in role_rows),
                "charge_incremental_c": sum(float(row["charge_incremental_c"]) for row in role_rows),
                "baseline_power_w": sum(float(row["baseline_power_w"]) for row in role_rows),
                "average_power_w": sum(float(row["average_power_w"]) for row in role_rows),
                "peak_current_a": "",
                "peak_power_w": max(
                    (
                        float(parse_float(sample.get("initiator_current_a")) or 0.0)
                        + float(parse_float(sample.get("responder_current_a")) or 0.0)
                    )
                    * float(voltage)
                    for sample in samples
                ),
            }
        )
        metrics.append(pair)
        for sample in samples:
            for role in ("initiator", "responder"):
                current = float(parse_float(sample.get(f"{role}_current_a")) or 0.0)
                traces.append(
                    {
                        "capture_id": capture_id,
                        "run_id": initiator.get("run_id", ""),
                        "pair_id": initiator.get("pair_id", ""),
                        "event_index": initiator.get("event_index", ""),
                        "exchange_id": initiator.get("exchange_id", ""),
                        "event_path": initiator.get("event_path", ""),
                        "role": role,
                        "power_channel_role": role,
                        "capture_time_s": sample.get("sample_time_s", ""),
                        "event_time_s": sample.get("sample_time_s", ""),
                        "current_a": current,
                        "voltage_v": voltage,
                        "power_w": current * float(voltage),
                        "source_file": str(raw_path),
                        "source_sha256": expected_hash,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "event_power_metrics.csv", metrics)
    write_csv_rows(output_dir / "power_trace.csv", traces)
    write_csv_rows(output_dir / "ngmo2_capture_audit.csv", audit)
    # The generic summarizer expects this filename.  Each row remains a
    # per-capture audit/provenance record; it is not a separate baseline run.
    metadata_rows = [
        {
            **row,
            "capture_mode": "ngmo2_uart_paced_envelope",
            "role": "pair",
            "baseline_method": "mean_of_interval_averaged_pre_and_post_guard_current",
        }
        for row in audit
    ]
    write_csv_rows(output_dir / "power_capture_metadata.csv", metadata_rows)
    return metrics, traces, audit


def require_valid_ngmo2_audit(audit: Iterable[Mapping[str, object]]) -> None:
    rows = list(audit)
    if not rows:
        raise ValueError("NGMO2 audit found no captures")
    failures = [row for row in rows if str(row.get("status", "")) != "pass"]
    if failures:
        raise ValueError(
            "NGMO2 markerless-envelope audit failed: "
            + " | ".join(str(row.get("issues", "")) for row in failures)
        )


def add_live_capture_arguments(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--autonomous",
        dest="capture_mode",
        action="store_const",
        const="autonomous",
        help=(
            "power-cycle and capture the boards with all debug hardware disconnected "
            "(default and primary measurement mode)"
        ),
    )
    mode.add_argument(
        "--uart-paced",
        dest="capture_mode",
        action="store_const",
        const="uart-paced",
        help=(
            "optional READY/GO/DUMP diagnostic mode; valid only with proven physical "
            "CHG_AC/VBUS isolation"
        ),
    )
    mode.add_argument(
        "--static-idle",
        dest="capture_mode",
        action="store_const",
        const="static-idle",
        help="measure per-state idle current on an explicit NGMO2 static range",
    )
    parser.set_defaults(capture_mode="autonomous")
    parser.add_argument(
        "--resource",
        help=(
            "PyVISA resource exposed by the connected interface; the NGMO2 manual "
            "documents RS-232 and GPIB, so USB adapters commonly appear as "
            "ASRL...::INSTR or GPIB...::INSTR (USB...::INSTR is also accepted if "
            "the interface exposes USBTMC)"
        ),
    )
    parser.add_argument("--list-resources", action="store_true")
    parser.add_argument("--probe", action="store_true", help="verify *IDN? only; never changes outputs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "print acquisition/configuration SCPI only, without importing PyVISA; "
            "output-safety programming and verification are intentionally omitted"
        ),
    )
    parser.add_argument("--visa-library")
    parser.add_argument("--visa-timeout-ms", type=int, default=180000)
    parser.add_argument("--visa-baud", type=int)
    parser.add_argument("--visa-data-bits", type=int, choices=(7, 8))
    parser.add_argument("--visa-parity", choices=("none", "odd", "even"))
    parser.add_argument("--visa-stop-bits", choices=("one", "two"))
    parser.add_argument("--visa-flow-control", choices=("none", "rts_cts", "xon_xoff"))
    parser.add_argument("--visa-termination", choices=("cr", "lf"))
    parser.add_argument("--initiator-port")
    parser.add_argument("--responder-port")
    parser.add_argument(
        "--initiator-channel",
        type=int,
        choices=(1, 2),
        help=(
            "NGMO2 channel physically wired to the initiator; required for capture "
            "and recorded so channel/cable assignments can be counterbalanced"
        ),
    )
    parser.add_argument("--board-baud", type=int, default=115200)
    parser.add_argument("--run-id")
    parser.add_argument("--pair-id")
    parser.add_argument("--initiator-board-id")
    parser.add_argument("--responder-board-id")
    parser.add_argument("--initiator-firmware-sha256")
    parser.add_argument("--responder-firmware-sha256")
    parser.add_argument(
        "--firmware-revision",
        help="exact SENSWEAR_FIRMWARE_REVISION string embedded in both autonomous images",
    )
    parser.add_argument(
        "--artifact-sha256",
        help="exact 64-hex deployed model artifact digest embedded in both images",
    )
    parser.add_argument(
        "--event-path",
        choices=("local", "no_contact", "accepted_connected", "accepted_discovery"),
    )
    parser.add_argument("--event-index-start", type=int, default=0)
    parser.add_argument("--trial-id-start", type=int, default=0)
    parser.add_argument("--exchange-id-start", type=int, default=0)
    parser.add_argument("--autonomous-trigger-delay-s", type=float, default=0.0)
    parser.add_argument(
        "--autonomous-event-dispatch-s",
        type=float,
        help="firmware's absolute event-dispatch uptime after output enable",
    )
    parser.add_argument(
        "--autonomous-event-deadline-s",
        type=float,
        help="validated worst-case event-completion uptime after output enable",
    )
    parser.add_argument("--autonomous-post-capture-s", type=float, default=15.0)
    parser.add_argument("--autonomous-power-off-s", type=float, default=2.0)
    parser.add_argument(
        "--boot-schedule-uncertainty-s",
        type=float,
        help="preflight upper bound for boot-to-scheduled-event timing uncertainty",
    )
    parser.add_argument("--channel-1-state")
    parser.add_argument("--channel-2-state")
    parser.add_argument("--idle-measure-interval-s", type=float, default=0.1)
    parser.add_argument("--idle-average-count", type=int, default=10)
    parser.add_argument("--idle-samples", type=int, default=30)
    parser.add_argument(
        "--idle-current-range",
        choices=("low", "medium"),
        help="required static range: low=5 mA or medium=0.5 A",
    )
    parser.add_argument(
        "--idle-settle-s",
        type=float,
        help="explicit settling time after output enable; no hardware-specific default",
    )
    parser.add_argument(
        "--idle-event-dispatch-s",
        type=float,
        help="optional earliest scheduled firmware event time after output enable",
    )
    parser.add_argument(
        "--idle-state-evidence",
        help="archived image/preflight identifier proving the declared idle states",
    )
    parser.add_argument("--idle-overload-threshold-a", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-interval-s", type=float, default=0.02)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--capture-count", type=int)
    parser.add_argument(
        "--firmware-sequence-start",
        type=int,
        help=(
            "zero-based next NVS sequence for the selected path; normally 0 for a "
            "fresh ledger, and explicit so resumed campaigns cannot be silently misjoined"
        ),
    )
    parser.add_argument("--firmware-timeout-s", type=float, default=120.0)
    parser.add_argument("--instrument-timeout-margin-s", type=float, default=10.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--voltage-v", type=float)
    parser.add_argument("--current-limit-a", type=float)
    parser.add_argument(
        "--current-limit-rejection-margin-a",
        type=float,
        help="reject samples within this explicit margin below the programmed current limit",
    )
    parser.add_argument("--maximum-safe-voltage-v", type=float)
    parser.add_argument(
        "--overvoltage-protection-v",
        type=float,
        help="explicit OVP threshold above --voltage-v and at/below the safe maximum",
    )
    parser.add_argument(
        "--max-output-voltage-deviation-v",
        type=float,
        help="operator-approved absolute tolerance for pre/post MEASure#:VOLTage?",
    )
    parser.add_argument(
        "--sense-mode",
        choices=("local",),
        help="final protocol requires the verified front-panel local-sense links",
    )
    parser.add_argument("--channel-1-lead-length-m", type=float)
    parser.add_argument("--channel-2-lead-length-m", type=float)
    parser.add_argument("--channel-1-lead-gauge-awg", type=float)
    parser.add_argument("--channel-2-lead-gauge-awg", type=float)
    parser.add_argument("--channel-1-cable-id")
    parser.add_argument("--channel-2-cable-id")
    parser.add_argument("--calibration-id")
    parser.add_argument("--calibration-date", help="NGMO2 calibration date, YYYY-MM-DD")
    parser.add_argument("--confirm-calibration-current", action="store_true")
    parser.add_argument(
        "--confirm-output-impedance-zero",
        action="store_true",
        help=(
            "confirm the operator intends 0-ohm output impedance; the tool also "
            "programs and reads back OUTPut#:IMPedance 0"
        ),
    )
    parser.add_argument("--enable-outputs", action="store_true")
    parser.add_argument(
        "--leave-outputs-on",
        action="store_true",
        help="leave both channels on after a successful campaign (default: turn both off)",
    )
    parser.add_argument("--confirm-uart-power-isolated", action="store_true")
    parser.add_argument("--confirm-debug-boards-disconnected", action="store_true")
    parser.add_argument("--confirm-no-parallel-battery", action="store_true")


def run_live_capture(args: argparse.Namespace) -> int:
    if args.dry_run:
        print("# acquisition/configuration SCPI only; no output-safety setup is emitted")
        print("*IDN?")
        if args.capture_mode == "static-idle":
            commands = static_idle_configuration_commands(
                args.idle_measure_interval_s,
                args.idle_average_count,
                args.idle_current_range,
            )
        else:
            commands = capture_configuration_commands(
                args.sample_interval_s, args.sample_count
            )
        for command in commands:
            print(command)
        if args.capture_mode == "static-idle":
            print("MEASure1:CURRent?")
            print("MEASure2:CURRent?")
        else:
            print("*TRG")
            print("FETCh1:ARRay?")
            print("FETCh2:ARRay?")
        return 0
    if args.list_resources:
        try:
            import pyvisa  # type: ignore
        except ImportError as error:
            raise RuntimeError("PyVISA is required to list VISA resources") from error
        manager = pyvisa.ResourceManager(args.visa_library) if args.visa_library else pyvisa.ResourceManager()
        try:
            for resource in manager.list_resources():
                print(resource)
        finally:
            manager.close()
        return 0
    if not args.resource:
        raise ValueError("--resource is required for probe or capture")
    manager, instrument = open_visa_transport(args)
    initiator_serial: _BoardSerial | None = None
    responder_serial: _BoardSerial | None = None
    ngmo2_verified = False
    try:
        args.instrument_idn = verify_ngmo2_idn(instrument.query("*IDN?"))
        ngmo2_verified = True
        print(args.instrument_idn)
        if args.probe:
            return 0
        required = {
            "--output-dir": args.output_dir,
            "--voltage-v": args.voltage_v,
            "--current-limit-a": args.current_limit_a,
            "--current-limit-rejection-margin-a": args.current_limit_rejection_margin_a,
            "--maximum-safe-voltage-v": args.maximum_safe_voltage_v,
            "--overvoltage-protection-v": args.overvoltage_protection_v,
            "--initiator-channel": args.initiator_channel,
            "--max-output-voltage-deviation-v": args.max_output_voltage_deviation_v,
            "--sense-mode": args.sense_mode,
            "--channel-1-lead-length-m": args.channel_1_lead_length_m,
            "--channel-2-lead-length-m": args.channel_2_lead_length_m,
            "--channel-1-lead-gauge-awg": args.channel_1_lead_gauge_awg,
            "--channel-2-lead-gauge-awg": args.channel_2_lead_gauge_awg,
            "--channel-1-cable-id": args.channel_1_cable_id,
            "--channel-2-cable-id": args.channel_2_cable_id,
            "--calibration-id": args.calibration_id,
            "--calibration-date": args.calibration_date,
        }
        missing = [
            name
            for name, value in required.items()
            if value is None or (isinstance(value, str) and not value.strip())
        ]
        if missing:
            raise ValueError("capture requires " + ", ".join(missing))
        if args.channel_1_lead_length_m <= 0 or args.channel_2_lead_length_m <= 0:
            raise ValueError("both channel lead lengths must be positive")
        if args.channel_1_lead_gauge_awg <= 0 or args.channel_2_lead_gauge_awg <= 0:
            raise ValueError("both channel lead gauges must be positive AWG values")
        if args.sense_mode != "local":
            raise ValueError(
                "the final protocol requires --sense-mode local with the NGMO2 "
                "front-panel local-sense jumpers installed"
            )
        if not 0 < args.current_limit_rejection_margin_a < args.current_limit_a:
            raise ValueError(
                "--current-limit-rejection-margin-a must be positive and below "
                "--current-limit-a"
            )
        try:
            datetime.strptime(args.calibration_date, "%Y-%m-%d")
        except ValueError as error:
            raise ValueError("--calibration-date must use YYYY-MM-DD") from error
        if not args.confirm_output_impedance_zero:
            raise ValueError(
                "refusing capture without --confirm-output-impedance-zero"
            )
        if not args.confirm_calibration_current:
            raise ValueError("refusing capture without --confirm-calibration-current")
        if not args.confirm_no_parallel_battery:
            raise ValueError(
                "refusing capture without --confirm-no-parallel-battery: remove/isolate the LiPo"
            )
        if not args.enable_outputs:
            raise ValueError(
                "measurement capture requires explicit --enable-outputs after wiring "
                "checks; list/probe/dry-run never enable outputs"
            )
        if args.capture_mode == "autonomous":
            autonomous_required = {
                "--run-id": args.run_id,
                "--pair-id": args.pair_id,
                "--initiator-board-id": args.initiator_board_id,
                "--responder-board-id": args.responder_board_id,
                "--initiator-firmware-sha256": args.initiator_firmware_sha256,
                "--responder-firmware-sha256": args.responder_firmware_sha256,
                "--firmware-revision": args.firmware_revision,
                "--artifact-sha256": args.artifact_sha256,
                "--event-path": args.event_path,
                "--autonomous-event-dispatch-s": args.autonomous_event_dispatch_s,
                "--autonomous-event-deadline-s": args.autonomous_event_deadline_s,
                "--capture-count": args.capture_count,
                "--firmware-sequence-start": args.firmware_sequence_start,
                "--boot-schedule-uncertainty-s": args.boot_schedule_uncertainty_s,
            }
            missing = [
                name
                for name, value in autonomous_required.items()
                if value is None or (isinstance(value, str) and not value.strip())
            ]
            if missing:
                raise ValueError("autonomous capture requires " + ", ".join(missing))
            if not args.confirm_debug_boards_disconnected:
                raise ValueError(
                    "refusing autonomous capture without "
                    "--confirm-debug-boards-disconnected"
                )
            if args.leave_outputs_on:
                raise ValueError(
                    "--leave-outputs-on is incompatible with one-event-per-power-cycle "
                    "autonomous capture"
                )
            required_timeout_ms = int(
                (
                    args.sample_interval_s * args.sample_count
                    + args.instrument_timeout_margin_s
                )
                * 1000
            )
            if args.visa_timeout_ms < required_timeout_ms:
                raise ValueError(
                    "--visa-timeout-ms must cover the dynamic capture plus the "
                    "instrument transfer margin"
                )
        elif args.capture_mode == "uart-paced":
            uart_required = {
                "--initiator-port": args.initiator_port,
                "--responder-port": args.responder_port,
            }
            missing = [
                name
                for name, value in uart_required.items()
                if value is None or (isinstance(value, str) and not value.strip())
            ]
            if missing:
                raise ValueError("UART-paced capture requires " + ", ".join(missing))
            if not args.confirm_uart_power_isolated:
                raise ValueError(
                    "refusing UART-paced capture without --confirm-uart-power-isolated: "
                    "debug-board USB VBUS must not bypass the NGMO2 current path"
                )
        else:
            idle_required = {
                "--run-id": args.run_id,
                "--pair-id": args.pair_id,
                "--initiator-board-id": args.initiator_board_id,
                "--responder-board-id": args.responder_board_id,
                "--initiator-firmware-sha256": args.initiator_firmware_sha256,
                "--responder-firmware-sha256": args.responder_firmware_sha256,
                "--channel-1-state": args.channel_1_state,
                "--channel-2-state": args.channel_2_state,
                "--idle-settle-s": args.idle_settle_s,
                "--idle-state-evidence": args.idle_state_evidence,
                "--idle-current-range": args.idle_current_range,
                "--idle-overload-threshold-a": args.idle_overload_threshold_a,
            }
            missing = [
                name
                for name, value in idle_required.items()
                if value is None or (isinstance(value, str) and not value.strip())
            ]
            if missing:
                raise ValueError("static-idle capture requires " + ", ".join(missing))
            if not args.confirm_debug_boards_disconnected:
                raise ValueError(
                    "refusing static-idle capture without "
                    "--confirm-debug-boards-disconnected"
                )
            if args.leave_outputs_on:
                raise ValueError("static-idle capture always turns both outputs off")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        existing_outputs = [
            path
            for path in (
                (
                    args.output_dir / "ngmo2_capture_manifest.csv",
                    args.output_dir / "initiator_serial.log",
                    args.output_dir / "responder_serial.log",
                )
                if args.capture_mode == "uart-paced"
                else (args.output_dir / "ngmo2_capture_manifest.csv",)
                if args.capture_mode == "autonomous"
                else (
                    args.output_dir / "ngmo2_idle_current.csv",
                    args.output_dir / "ngmo2_idle_summary.csv",
                    args.output_dir / "ngmo2_idle_metadata.csv",
                )
            )
            if path.exists()
        ]
        if existing_outputs:
            raise FileExistsError(
                "refusing to overwrite capture outputs: "
                + ", ".join(str(path) for path in existing_outputs)
            )
        _output_off(instrument)
        if args.capture_mode == "uart-paced":
            # Open both UARTs before applying power. With VBUS isolated this
            # retains boot PULSE_META/PULSE_READY instead of racing target boot.
            initiator_serial = _open_board_serial(
                args.initiator_port,
                args.board_baud,
                args.output_dir / "initiator_serial.log",
            )
            responder_serial = _open_board_serial(
                args.responder_port,
                args.board_baud,
                args.output_dir / "responder_serial.log",
            )
        configure_outputs(
            instrument,
            args.voltage_v,
            args.current_limit_a,
            args.maximum_safe_voltage_v,
            overvoltage_protection_v=args.overvoltage_protection_v,
            enable=args.enable_outputs,
            turn_on=args.capture_mode == "uart-paced",
        )
        if args.capture_mode == "autonomous":
            configure_capture(instrument, args.sample_interval_s, args.sample_count)
            capture_autonomous_campaign(args, instrument)
        elif args.capture_mode == "uart-paced":
            configure_capture(instrument, args.sample_interval_s, args.sample_count)
            capture_campaign(args, instrument, initiator_serial, responder_serial)
        else:
            configure_static_idle(
                instrument,
                args.idle_measure_interval_s,
                args.idle_average_count,
                args.idle_current_range,
            )
            capture_static_idle(args, instrument)
        if not args.leave_outputs_on:
            _output_off(instrument)
        return 0
    except (BaseException,):
        # Never send output-control commands to an instrument that did not
        # first pass the NGMO2 identity check. Once verified, fail closed on
        # every error and KeyboardInterrupt.
        if ngmo2_verified:
            _output_off(instrument)
        raise
    finally:
        if initiator_serial is not None:
            initiator_serial.close()
        if responder_serial is not None:
            responder_serial.close()
        instrument.close()
        manager.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_live_capture_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run_live_capture(args)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
