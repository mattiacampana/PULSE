from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

from common import PULSE_BUILD_CONTEXT_FIELDS, sha256_file, write_csv_rows  # noqa: E402
from communication_analysis import (  # noqa: E402
    analyze_communication,
    require_valid_communication_audit,
)
from ngmo2_capture import (  # noqa: E402
    _BoardSerial,
    _write_raw_capture,
    acquire_synchronized_arrays,
    analyze_envelope_captures,
    capture_autonomous_campaign,
    capture_static_idle,
    capture_configuration_commands,
    configure_capture,
    configure_outputs,
    configure_static_idle,
    parse_ascii_array,
    require_valid_ngmo2_audit,
    static_idle_configuration_commands,
    validate_capture_configuration,
    verify_ngmo2_idn,
)
from ngmo2_autonomous import (  # noqa: E402
    analyze_autonomous_captures,
    pulse_hash32,
    require_valid_autonomous_audit,
)
from figures import (  # noqa: E402
    LATENCY_PLOT_EVENT_KEYS,
    SECTION_V_STAGE_REQUIREMENTS,
    _validate_section_v_coverage,
)
from serial_parser import _build_context_sha256, parse_serial_files  # noqa: E402


TEST_FIRMWARE_REVISION = "1" * 40
TEST_ARTIFACT_SHA256 = "c" * 64


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def release_build_context(board_id: str, role: str) -> dict[str, object]:
    row: dict[str, object] = {
        field: f"test_{field}" for field in PULSE_BUILD_CONTEXT_FIELDS
    }
    row.update(
        {
            "run_id": "timing-run",
            "pair_id": "pair-auto",
            "board_id": board_id,
            "role": role,
            "firmware_revision": TEST_FIRMWARE_REVISION,
            "hardware_revision": "senswear-rev-a",
            "build_variant": "pulse_release",
            "build_guard": "final_capture_requirements_enforced",
            "artifact_sha256": TEST_ARTIFACT_SHA256,
            "fixture_hash": "d" * 64,
            "artifact_kind": "exported_pulse_npz",
            "encoder_hash": "e" * 64,
            "tensor_precision": "float32",
            "batch_size": "16",
            "local_steps": "2",
            "class_count": "6",
            "parameter_count": "4550",
        }
    )
    row["build_context_sha256"] = _build_context_sha256(
        {key: str(value) for key, value in row.items()}
    )
    return row


def write_release_build_context(path: Path) -> Path:
    write_csv_rows(
        path,
        [
            release_build_context("board-i", "initiator"),
            release_build_context("board-r", "responder"),
        ],
    )
    return path


def complete_ngmo2_figure_data() -> dict[str, list[dict[str, object]]]:
    context_rows = [
        release_build_context("board-i", "initiator"),
        release_build_context("board-r", "responder"),
    ]
    context = {
        field: context_rows[0][field] for field in PULSE_BUILD_CONTEXT_FIELDS
    }
    context_source = "9" * 64
    raw_source = "8" * 64
    analysis_source = "7" * 64
    run_metadata = [
        {**row, "analysis_input_sha256": context_source} for row in context_rows
    ]
    latency: list[dict[str, object]] = []
    for event_path, role, connection_state in LATENCY_PLOT_EVENT_KEYS:
        latency.append(
            {
                **context,
                "source": "event",
                "event_path": event_path,
                "role": role,
                "connection_state": connection_state,
                "median_ms": 1.0,
            }
        )
        for stage in SECTION_V_STAGE_REQUIREMENTS[
            (event_path, role, connection_state)
        ]:
            latency.append(
                {
                    **context,
                    "source": "stage",
                    "event_path": event_path,
                    "role": role,
                    "connection_state": connection_state,
                    "stage": stage,
                    "median_ms": 0.1,
                }
            )
    power_events: list[dict[str, object]] = []
    power: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for index, (event_path, role, connection_state) in enumerate(
        (
            ("local", "local", "connected"),
            ("no_contact", "local", "disconnected"),
            ("accepted_connected", "pair", "connected"),
            ("accepted_discovery", "pair", "disconnected"),
        ),
        1,
    ):
        # Capture IDs restart in each independently acquired path directory.
        # The figure join must therefore use the complete run/path/sequence key.
        capture_id = "c00000"
        run_id = f"run-{event_path}"
        raw = {
            **context,
            "run_id": run_id,
            "pair_id": "pair-auto",
            "capture_id": capture_id,
            "firmware_sequence": str(index - 1),
            "event_path": event_path,
            "role": role,
            "connection_state": connection_state,
            "remote_session_started": "1" if role == "pair" else "0",
            "measurement_scope": "ngmo2_capture_envelope",
            "baseline_source": "same_capture_pre_dispatch_and_post_completion_guards",
            "baseline_method": "equal_weight_mean_of_interval_average_guard_means",
            "outcome": "success",
            "warmup": "0",
            "success": "1",
            "energy_total_j": "0.02",
            "energy_incremental_j": "0.01",
            "average_power_w": "0.002",
            "peak_power_w": "0.004",
            "analysis_input_sha256": raw_source,
            "analysis_source_sha256": analysis_source,
            "build_context_source_sha256": context_source,
            "capture_profile": "ngmo2_autonomous_proof_joined_full_capture",
            "join_status": "ngmo2_autonomous_exact_proof_sequence",
        }
        power_events.append(raw)
        for metric in (
            "energy_total_j",
            "energy_incremental_j",
            "average_power_w",
            "peak_power_w",
        ):
            power.append(
                {
                    **context,
                    "event_path": event_path,
                    "role": role,
                    "connection_state": connection_state,
                    "remote_session_started": raw["remote_session_started"],
                    "measurement_scope": raw["measurement_scope"],
                    "baseline_source": raw["baseline_source"],
                    "baseline_method": raw["baseline_method"],
                    "outcome": "success",
                    "metric": metric,
                    "n": "1",
                    "median": raw[metric],
                    "analysis_source_sha256_set": raw_source,
                }
            )
        audits.append(
            {
                **context,
                "run_id": run_id,
                "pair_id": "pair-auto",
                "capture_id": capture_id,
                "firmware_sequence": str(index - 1),
                "event_path": event_path,
                "audit_type": "ngmo2_autonomous_proof_join_full_capture",
                "status": "pass",
                "analysis_source_sha256": analysis_source,
                "build_context_source_sha256": context_source,
            }
        )
    return {
        "protocol_audit": [
            {
                "required_stratum": "1",
                "stratum_check": "pass",
                "integrity_check": "pass",
            }
        ],
        "run_metadata": run_metadata,
        "latency": latency,
        "communication": [{**context, "metric": "att_value_bytes", "median": "1"}],
        "communication_audit": [{"status": "pass"}],
        "communication_metrics": [
            {
                "event_path": path,
                "role": "pair",
                "success": "1",
                "warmup": "0",
                "measurement_scope": "firmware_att_value_boundary",
                "evidence_class": "firmware_counter_not_over_air",
                "att_value_bytes": "1",
            }
            for path in ("accepted_connected", "accepted_discovery")
        ],
        "resource_totals": [
            {"image": image, "memory": memory, "used_bytes": "1"}
            for image in ("pulse", "baseline")
            for memory in ("flash", "static_ram")
        ],
        "resource_provenance": [
            {
                "image": image,
                "elf_sha256": "a" * 64,
                "map_sha256": "b" * 64 if image != "correctness" else "",
                "firmware_revision": TEST_FIRMWARE_REVISION,
            }
            for image in ("pulse", "baseline", "correctness")
        ],
        "correctness": [
            {
                "artifact_sha256": TEST_ARTIFACT_SHA256,
                "fixture_hash": "d" * 64,
                "firmware_revision": TEST_FIRMWARE_REVISION,
                "reference": "pytorch_cpu_float32",
                "pass": "1",
            }
        ],
        "power": power,
        "power_events": power_events,
        "ngmo2_capture_audit": audits,
    }


class FakeInstrument:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.queries: list[str] = []
        self.outputs = {1: False, 2: False}
        self.current_range = "MEDium"
        self.measure_interval = 0.1
        self.average_count = 10
        self.pulse_interval = 0.001
        self.pulse_length = 3
        self.measure_values = {1: 0.001, 2: 0.002}
        self.source_voltage = {1: 3.8, 2: 3.8}
        self.maximum_voltage = {1: 4.2, 2: 4.2}
        self.overvoltage_protection = {1: 4.1, 2: 4.1}
        self.output_impedance = {1: 0.0, 2: 0.0}
        self.current_limit_active = {1: False, 2: False}
        self.common_output = False
        self.current_limit = {1: 0.1, 2: 0.1}
        self.current_limit_maximum = {1: 0.1, 2: 0.1}

    def write(self, command: str) -> None:
        self.writes.append(command)
        match = __import__("re").fullmatch(r"OUTPut([12]) (ON|OFF)", command)
        if match:
            self.outputs[int(match.group(1))] = match.group(2) == "ON"
        if ":CURRent:RANGe LOW" in command:
            self.current_range = "LOW"
        if ":CURRent:RANGe MEDium" in command:
            self.current_range = "MEDium"
        if ":MEASure:INTerval " in command:
            self.measure_interval = float(command.rsplit(" ", 1)[1])
        if ":AVERage:COUNt " in command:
            self.average_count = int(command.rsplit(" ", 1)[1])
        if ":PULSe:SAMPle:INTerval " in command:
            self.pulse_interval = float(command.rsplit(" ", 1)[1])
        if ":PULSe:SAMPle:LENGth " in command:
            self.pulse_length = int(command.rsplit(" ", 1)[1])
        source_match = __import__("re").fullmatch(
            r"SOURce([12]):(VOLTage|CURRent) ([-+0-9.eE]+)", command
        )
        if source_match:
            target = (
                self.source_voltage
                if source_match.group(2) == "VOLTage"
                else self.current_limit
            )
            target[int(source_match.group(1))] = float(source_match.group(3))
        maximum_match = __import__("re").fullmatch(
            r"SOURce([12]):VOLTage:MAXSetting ([-+0-9.eE]+)", command
        )
        if maximum_match:
            self.maximum_voltage[int(maximum_match.group(1))] = float(
                maximum_match.group(2)
            )
        protection_match = __import__("re").fullmatch(
            r"SOURce([12]):VOLTage:PROTection ([-+0-9.eE]+)", command
        )
        if protection_match:
            self.overvoltage_protection[int(protection_match.group(1))] = float(
                protection_match.group(2)
            )
        limit_maximum_match = __import__("re").fullmatch(
            r"SOURce([12]):CURRent:LIMit:MAXSetting ([-+0-9.eE]+)", command
        )
        if limit_maximum_match:
            self.current_limit_maximum[int(limit_maximum_match.group(1))] = float(
                limit_maximum_match.group(2)
            )
        impedance_match = __import__("re").fullmatch(
            r"OUTPut([12]):IMPedance ([-+0-9.eE]+)", command
        )
        if impedance_match:
            self.output_impedance[int(impedance_match.group(1))] = float(
                impedance_match.group(2)
            )
        if command == "CONFIG:COMMon:OUTPut:ONOFf OFF":
            self.common_output = False

    def query(self, command: str) -> str:
        self.queries.append(command)
        if command in {"*OPC?"}:
            return "1"
        if command == "SYSTem:ERRor?":
            return '0,"No error"'
        if command == "CONFIG:COMMon:OUTPut:ONOFf?":
            return "ON" if self.common_output else "OFF"
        if command.endswith(":FUNCtion?"):
            return "AVERage"
        if command.endswith(":CURRent:RANGe?"):
            return self.current_range
        if command.endswith(":PULSe:CHANnel?"):
            return "CURRent"
        if command.endswith(":PULSe:TRIGger:SOURce?"):
            return "INTernal"
        if command.endswith(":PULSe:TRIGger:COUNt?"):
            return "1"
        if command.endswith(":PULSe:SAMPle:LENGth?"):
            return str(self.pulse_length)
        if command.endswith(":PULSe:SAMPle:INTerval?"):
            return str(self.pulse_interval)
        if command.endswith(":MEASure:INTerval?"):
            return str(self.measure_interval)
        if command.endswith(":AVERage:COUNt?"):
            return str(self.average_count)
        if command.endswith(":PULSe:TRIGger:STATe?"):
            return "READY"
        if command == "FETCh1:ARRay?":
            return "1e-3,2e-3,3e-3"
        if command == "FETCh2:ARRay?":
            return "4e-3,5e-3,6e-3"
        output_match = __import__("re").fullmatch(r"OUTPut([12])\?", command)
        if output_match:
            return "ON" if self.outputs[int(output_match.group(1))] else "OFF"
        measure_match = __import__("re").fullmatch(
            r"MEASure([12]):CURRent\?", command
        )
        if measure_match:
            return str(self.measure_values[int(measure_match.group(1))])
        voltage_measure_match = __import__("re").fullmatch(
            r"MEASure([12]):VOLTage\?", command
        )
        if voltage_measure_match:
            return str(self.source_voltage[int(voltage_measure_match.group(1))])
        maximum_match = __import__("re").fullmatch(
            r"SOURce([12]):VOLTage:MAXSetting\?", command
        )
        if maximum_match:
            return str(self.maximum_voltage[int(maximum_match.group(1))])
        protection_match = __import__("re").fullmatch(
            r"SOURce([12]):VOLTage:PROTection\?", command
        )
        if protection_match:
            return str(self.overvoltage_protection[int(protection_match.group(1))])
        impedance_match = __import__("re").fullmatch(
            r"OUTPut([12]):IMPedance\?", command
        )
        if impedance_match:
            return str(self.output_impedance[int(impedance_match.group(1))])
        limit_type_match = __import__("re").fullmatch(
            r"SOURce([12]):CURRent:LIMit:TYPE\?", command
        )
        if limit_type_match:
            return "LIMIT"
        limit_maximum_match = __import__("re").fullmatch(
            r"SOURce([12]):CURRent:LIMit:MAXSetting\?", command
        )
        if limit_maximum_match:
            return str(self.current_limit_maximum[int(limit_maximum_match.group(1))])
        limit_state_match = __import__("re").fullmatch(
            r"SOURce([12]):CURRent:LIMit:STATe\?", command
        )
        if limit_state_match:
            return (
                "ON"
                if self.current_limit_active[int(limit_state_match.group(1))]
                else "OFF"
            )
        source_match = __import__("re").fullmatch(
            r"SOURce([12]):(VOLTage|CURRent)\?", command
        )
        if source_match:
            source = (
                self.source_voltage
                if source_match.group(2) == "VOLTage"
                else self.current_limit
            )
            return str(source[int(source_match.group(1))])
        raise AssertionError(f"unexpected query: {command}")

    def close(self) -> None:
        return None


class FakeSerialHandle:
    def __init__(self, lines: list[str]) -> None:
        self.lines = [(line + "\n").encode("ascii") for line in lines]
        self.writes: list[bytes] = []
        self.closed = False

    def readline(self) -> bytes:
        return self.lines.pop(0) if self.lines else b""

    def write(self, value: bytes) -> None:
        self.writes.append(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def accepted_events() -> list[dict[str, object]]:
    common = {
        "run_id": "run-1",
        "pair_id": "pair-1",
        "trial_id": 0,
        "exchange_id": 7,
        "event_path": "accepted_connected",
        "connection_state": "connected",
        "warmup": 0,
        "post_warmup": 1,
        "success": 1,
        "remote_session_started": 1,
        "capture_quiet_guard_ms": 20,
    }
    return [
        {
            **common,
            "board_id": "board-i",
            "event_index": 10,
            "role": "initiator",
            "capture_envelope_duration_us": 60000,
            "application_tx_bytes": 100,
            "application_rx_bytes": 50,
            "protocol_header_tx_bytes": 4,
            "protocol_header_rx_bytes": 3,
            "att_tx_bytes": 104,
            "att_rx_bytes": 53,
            "tx_chunks": 2,
            "rx_chunks": 1,
        },
        {
            **common,
            "board_id": "board-r",
            "event_index": 20,
            "role": "responder",
            "application_tx_bytes": 50,
            "application_rx_bytes": 100,
            "protocol_header_tx_bytes": 3,
            "protocol_header_rx_bytes": 4,
            "att_tx_bytes": 53,
            "att_rx_bytes": 104,
            "tx_chunks": 1,
            "rx_chunks": 2,
        },
    ]


def autonomous_proof(
    board_id: str,
    role: str,
    *,
    success: int = 1,
    record_ready: int = 1,
    remote_session_started: int = 1,
    event_duration_us: int = 300000,
) -> dict[str, object]:
    return {
        "board_id": board_id,
        "schema": 2,
        "path": "accepted_connected",
        "firmware_sequence": 0,
        "sequence": 0,
        "trial_id": 0,
        "exchange_id": 1,
        "role": role,
        "warmup": 1,
        "record_ready": record_ready,
        "passive_expected": 0,
        "success": success,
        "failure_reason": 0 if success else 3,
        "remote_session_started": remote_session_started,
        "event_duration_us": event_duration_us,
        "event_errno": 0 if success else -110,
        "infrastructure_errno": 0 if success else -110,
        "result_release_errno": 0,
        "initial_head_hash": "11111111",
        "result_head_hash": "22222222",
        "firmware_revision_hash": pulse_hash32(TEST_FIRMWARE_REVISION),
        "artifact_hash": pulse_hash32(TEST_ARTIFACT_SHA256),
        "source_file": f"{board_id}.log",
        "source_line": 1,
    }


def autonomous_manifest(raw: Path) -> dict[str, object]:
    return {
        "capture_id": "auto-c1",
        "run_id": "run-auto",
        "pair_id": "pair-auto",
        "event_path": "accepted_connected",
        "firmware_sequence": 0,
        "capture_profile": "ngmo2_autonomous_one_event_per_power_cycle",
        "debug_boards_disconnected_confirmed": 1,
        "no_parallel_battery_confirmed": 1,
        "initiator_board_id": "board-i",
        "responder_board_id": "board-r",
        "initiator_firmware_sha256": "a" * 64,
        "responder_firmware_sha256": "b" * 64,
        "firmware_revision": TEST_FIRMWARE_REVISION,
        "artifact_sha256": TEST_ARTIFACT_SHA256,
        "expected_firmware_revision_hash": pulse_hash32(TEST_FIRMWARE_REVISION),
        "expected_artifact_hash": pulse_hash32(TEST_ARTIFACT_SHA256),
        "initiator_channel": 1,
        "channel_1_role": "initiator",
        "channel_2_role": "responder",
        "sample_interval_s": 0.1,
        "sample_count": 10,
        "current_range_a": 0.5,
        "current_range_limit_a": 0.510,
        "current_range_setting": "MEDium",
        "current_resolution_a": 1e-5,
        "current_full_scale_deviation_a": 0.001,
        "current_limit_a": 0.1,
        "current_limit_rejection_margin_a": 0.005,
        "maximum_safe_voltage_v": 4.2,
        "overvoltage_protection_v": 4.1,
        "voltage_v": 4.0,
        "voltage_basis": "local_sensed_ngmo2_output_pre_post_readback",
        "sense_mode": "local",
        "channel_1_lead_length_m": 0.5,
        "channel_2_lead_length_m": 0.5,
        "channel_1_lead_gauge_awg": 24,
        "channel_2_lead_gauge_awg": 24,
        "channel_1_cable_id": "cable-a",
        "channel_2_cable_id": "cable-b",
        "calibration_id": "CAL-001",
        "calibration_date": "2026-01-01",
        "calibration_current_confirmed": 1,
        "output_impedance_ohm": 0,
        "output_impedance_programmed_ohm": 0,
        "common_output_coupling": "OFF",
        "current_limit_type": "LIMIT",
        "current_limit_maximum_setting_readback_verified": 1,
        "channel_1_current_limit_state_pre": 0,
        "channel_2_current_limit_state_pre": 0,
        "channel_1_current_limit_state_post": 0,
        "channel_2_current_limit_state_post": 0,
        "maximum_voltage_setting_readback_verified": 1,
        "max_output_voltage_deviation_v": 0.05,
        "channel_1_output_voltage_pre_v": 4.0,
        "channel_2_output_voltage_pre_v": 4.0,
        "channel_1_output_voltage_post_v": 4.0,
        "channel_2_output_voltage_post_v": 4.0,
        "event_dispatch_in_capture_s": 0.2,
        "event_completion_deadline_in_capture_s": 0.8,
        "canonical_csv": str(raw),
        "canonical_csv_sha256": sha256_file(raw),
    }


class Ngmo2CommandTests(unittest.TestCase):
    def test_manual_command_dialect_and_bounds(self) -> None:
        commands = capture_configuration_commands(0.001, 3)
        self.assertEqual(commands[0], "FORMat:DATA ASCii")
        self.assertIn("SENSe1:FUNCtion AVERage", commands)
        self.assertIn("SENSe1:CURRent:RANGe MEDium", commands)
        self.assertIn("SENSe2:PULSe:SAMPle:LENGth 3", commands)
        self.assertIn("SENSe2:PULSe:SAMPle:INTerval 0.001", commands)
        with self.assertRaises(ValueError):
            validate_capture_configuration(0.000015, 3)
        with self.assertRaises(ValueError):
            validate_capture_configuration(0.001, 5001)

    def test_idn_and_ascii_array_are_fail_closed(self) -> None:
        self.assertIn("NGMO2", verify_ngmo2_idn("Rohde & Schwarz,NGMO2,1,4.0"))
        with self.assertRaises(ValueError):
            verify_ngmo2_idn("Acme,OtherSupply,1,1")
        self.assertEqual(parse_ascii_array("1e-3, 2e-3", 2), [0.001, 0.002])
        with self.assertRaises(ValueError):
            parse_ascii_array("1e-3", 2)

    def test_configuration_readback_and_common_trigger(self) -> None:
        instrument = FakeInstrument()
        configure_capture(instrument, 0.001, 3)
        self.assertEqual(
            instrument.writes,
            capture_configuration_commands(0.001, 3),
        )
        channel_1, channel_2 = acquire_synchronized_arrays(
            instrument, 3, 0.2, poll_interval_s=0
        )
        self.assertEqual(instrument.writes[-1], "*TRG")
        self.assertEqual(channel_1, [0.001, 0.002, 0.003])
        self.assertEqual(channel_2, [0.004, 0.005, 0.006])

    def test_output_programming_requires_safe_voltage_and_can_stay_off(self) -> None:
        instrument = FakeInstrument()
        configure_outputs(
            instrument,
            3.8,
            0.1,
            4.2,
            overvoltage_protection_v=4.1,
            enable=True,
            turn_on=False,
        )
        self.assertEqual(instrument.source_voltage, {1: 3.8, 2: 3.8})
        self.assertEqual(instrument.maximum_voltage, {1: 4.2, 2: 4.2})
        self.assertIn("SOURce1:VOLTage:MAXSetting 4.2", instrument.writes)
        self.assertIn("SOURce1:VOLTage:PROTection 4.1", instrument.writes)
        self.assertIn("SOURce1:CURRent:LIMit:MAXSetting 0.1", instrument.writes)
        self.assertIn("OUTPut1:IMPedance 0", instrument.writes)
        self.assertIn("CONFIG:COMMon:OUTPut:ONOFf OFF", instrument.writes)
        self.assertFalse(any(instrument.outputs.values()))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            configure_outputs(
                instrument,
                4.3,
                0.1,
                4.2,
                overvoltage_protection_v=4.1,
                enable=True,
            )

    def test_responder_uart_join_uses_exchange_not_initiator_event_index(self) -> None:
        line = (
            "PULSE_EVENT,board_id=board-r,event_index=20,trial_id=0,exchange_id=7,"
            "role=responder,event_path=accepted_connected,success=1"
        )
        with tempfile.TemporaryDirectory() as temporary:
            serial = FakeSerialHandle([line])
            board = _BoardSerial(serial, Path(temporary) / "responder.log")
            try:
                event = board.event(
                    {"event_index": "10", "exchange_id": "7"},
                    0.2,
                    match_event_index=False,
                    expected_role="responder",
                )
            finally:
                board.close()
        self.assertEqual(event["event_index"], "20")
        self.assertTrue(serial.closed)

    def test_physical_channel_mapping_is_explicit_in_canonical_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.csv"
            _write_raw_capture(path, [0.001], [0.009], 0.01, 3.8, 2)
            row = read_rows(path)[0]
        self.assertEqual(row["channel_1_current_a"], "0.001")
        self.assertEqual(row["channel_2_current_a"], "0.009")
        self.assertEqual(row["initiator_current_a"], "0.009")
        self.assertEqual(row["responder_current_a"], "0.001")

    def test_autonomous_capture_is_power_cycled_and_not_falsely_joined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = Namespace(
                output_dir=Path(temporary),
                capture_count=105,
                autonomous_trigger_delay_s=0.0,
                autonomous_power_off_s=0.0,
                autonomous_post_capture_s=0.0,
                autonomous_event_dispatch_s=0.1,
                autonomous_event_deadline_s=0.2,
                sample_interval_s=0.1,
                sample_count=3,
                instrument_timeout_margin_s=0.1,
                poll_interval_s=0.0,
                run_id="run-auto",
                pair_id="pair-auto",
                initiator_board_id="board-i",
                responder_board_id="board-r",
                initiator_firmware_sha256="a" * 64,
                responder_firmware_sha256="b" * 64,
                firmware_revision=TEST_FIRMWARE_REVISION,
                artifact_sha256=TEST_ARTIFACT_SHA256,
                event_path="accepted_connected",
                event_index_start=10,
                trial_id_start=20,
                exchange_id_start=30,
                firmware_sequence_start=0,
                boot_schedule_uncertainty_s=0.001,
                initiator_channel=2,
                resource="USB0::0x0AAD::0x0001::INSTR",
                instrument_idn="Rohde & Schwarz,NGMO2,1,4.0",
                voltage_v=3.8,
                current_limit_a=0.1,
                current_limit_rejection_margin_a=0.005,
                maximum_safe_voltage_v=4.2,
                overvoltage_protection_v=4.1,
                max_output_voltage_deviation_v=0.05,
                sense_mode="local",
                channel_1_lead_length_m=0.5,
                channel_2_lead_length_m=0.5,
                channel_1_lead_gauge_awg=24,
                channel_2_lead_gauge_awg=24,
                channel_1_cable_id="cable-a",
                channel_2_cable_id="cable-b",
                calibration_id="CAL-001",
                calibration_date="2026-01-01",
                visa_timeout_ms=180000,
            )
            instrument = FakeInstrument()
            rows = capture_autonomous_campaign(args, instrument)
            manifest = read_rows(Path(temporary) / "ngmo2_capture_manifest.csv")[0]
            raw = read_rows(Path(rows[0]["canonical_csv"]))[0]
        self.assertFalse(any(instrument.outputs.values()))
        self.assertEqual(manifest["capture_evidence"], "operator_declared_schedule_no_live_firmware_join")
        self.assertEqual(manifest["join_status"], "not_joined_capture_only_operator_schedule")
        self.assertEqual(manifest["firmware_sequence"], "0")
        self.assertEqual(manifest["sense_mode"], "local")
        self.assertEqual(manifest["channel_1_output_voltage_pre_v"], "3.8")
        earliest = float(manifest["trigger_offset_earliest_after_power_anchor_s"])
        latest = float(manifest["trigger_offset_latest_after_power_anchor_s"])
        self.assertGreaterEqual(earliest, 0)
        self.assertGreaterEqual(latest, earliest)
        self.assertEqual(raw["initiator_current_a"], "0.004")
        self.assertEqual(rows[-1]["firmware_sequence"], 104)

    def test_static_idle_uses_low_range_and_writes_state_bound_summary(self) -> None:
        commands = static_idle_configuration_commands(0.1, 10, "low")
        self.assertIn("SENSe1:CURRent:RANGe LOW", commands)
        self.assertIn("SENSe2:AVERage:COUNt 10", commands)
        with tempfile.TemporaryDirectory() as temporary:
            args = Namespace(
                output_dir=Path(temporary),
                idle_samples=2,
                idle_settle_s=0.0,
                idle_event_dispatch_s=None,
                idle_state_evidence="state-hold-image-sha256:abc",
                idle_overload_threshold_a=0.0051,
                idle_current_range="low",
                idle_measure_interval_s=0.1,
                idle_average_count=10,
                initiator_channel=1,
                initiator_board_id="board-i",
                responder_board_id="board-r",
                initiator_firmware_sha256="a" * 64,
                responder_firmware_sha256="b" * 64,
                channel_1_state="connected_idle",
                channel_2_state="advertising_idle",
                run_id="idle-1",
                pair_id="pair-1",
                resource="USB0::0x0AAD::0x0001::INSTR",
                instrument_idn="Rohde & Schwarz,NGMO2,1,4.0",
                voltage_v=3.8,
                current_limit_a=0.1,
                current_limit_rejection_margin_a=0.005,
                maximum_safe_voltage_v=4.2,
                overvoltage_protection_v=4.1,
                max_output_voltage_deviation_v=0.3,
                sense_mode="local",
                channel_1_lead_length_m=0.5,
                channel_2_lead_length_m=0.5,
                channel_1_lead_gauge_awg=24,
                channel_2_lead_gauge_awg=24,
                channel_1_cable_id="cable-a",
                channel_2_cable_id="cable-b",
                calibration_id="CAL-001",
                calibration_date="2026-01-01",
            )
            instrument = FakeInstrument()
            instrument.source_voltage = {1: 3.7, 2: 3.6}
            configure_static_idle(instrument, 0.1, 10, "low")
            samples, summary = capture_static_idle(args, instrument)
            metadata = read_rows(Path(temporary) / "ngmo2_idle_metadata.csv")[0]
        self.assertEqual(len(samples), 4)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["background_state"], "connected_idle")
        self.assertAlmostEqual(float(summary[0]["mean_power_w"]), 0.0037)
        self.assertAlmostEqual(float(summary[1]["mean_power_w"]), 0.0072)
        self.assertEqual(summary[0]["voltage_v"], 3.7)
        self.assertEqual(metadata["current_resolution_a"], "1e-07")
        self.assertEqual(metadata["status"], "pass")
        self.assertEqual(
            metadata["state_semantics"],
            "operator_preflight_evidence_not_inferred_from_labels",
        )
        self.assertFalse(any(instrument.outputs.values()))

        args.idle_event_dispatch_s = 0.1
        args.output_dir = Path(temporary) / "overlap"
        with self.assertRaisesRegex(ValueError, "overlaps"):
            capture_static_idle(args, FakeInstrument())

        medium_commands = static_idle_configuration_commands(0.1, 10, "medium")
        self.assertIn("SENSe1:CURRent:RANGe MEDium", medium_commands)


class Ngmo2AnalysisTests(unittest.TestCase):
    def test_ngmo2_figure_gate_requires_bound_archived_build_context(self) -> None:
        # Reused c00000 IDs across path directories are valid because the audit
        # binding includes run, pair, path, and firmware sequence.
        _validate_section_v_coverage(complete_ngmo2_figure_data(), profile="ngmo2")

        missing = complete_ngmo2_figure_data()
        missing["power_events"][0]["build_context_source_sha256"] = ""
        with self.assertRaisesRegex(ValueError, "source_not_included"):
            _validate_section_v_coverage(missing, profile="ngmo2")

        mismatched = complete_ngmo2_figure_data()
        mismatched["power_events"][0]["hardware_revision"] = "tampered-revision"
        with self.assertRaisesRegex(ValueError, "release_contract_mismatch"):
            _validate_section_v_coverage(mismatched, profile="ngmo2")

        diagnostic = complete_ngmo2_figure_data()
        for row in diagnostic["power_events"]:
            row["capture_profile"] = "ngmo2_uart_paced_capture_envelope"
        with self.assertRaisesRegex(ValueError, "not_autonomous_profile"):
            _validate_section_v_coverage(diagnostic, profile="ngmo2")

    def test_proof_export_parser_adds_zero_based_firmware_sequence(self) -> None:
        proof_line = (
            "PULSE_CAPTURE_PROOF,board_id=board-i,schema=2,"
            "path=accepted_connected,sequence=0,trial_id=0,exchange_id=1,"
            "role=initiator,warmup=0,record_ready=1,passive_expected=0,"
            "success=1,failure_reason=0,remote_session_started=1,"
            "event_duration_us=300000,event_errno=0,infrastructure_errno=0,"
            "result_release_errno=0,initial_head_hash=11111111,"
            "result_head_hash=22222222,firmware_revision_hash=33333333,"
            "artifact_hash=44444444\n"
            "PULSE_CAPTURE_STATUS,board_id=board-i,state=export_complete\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "proof.log"
            log.write_text(proof_line, encoding="utf-8")
            records = parse_serial_files([log], root / "parsed")
            proof = records["PULSE_CAPTURE_PROOF"][0]
            status = records["PULSE_CAPTURE_STATUS"][0]
            parsed = read_rows(root / "parsed" / "capture_proofs.csv")[0]
        self.assertEqual(proof["firmware_sequence"], "0")
        self.assertEqual(parsed["sequence"], "0")
        self.assertEqual(status["state"], "export_complete")

    def test_autonomous_proof_join_full_capture_rectangular_energy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "capture.csv"
            initiator = [0.001, 0.001, 0.006, 0.003, 0.003, 0.003, 0.003, 0.003, 0.001, 0.001]
            responder = [0.002, 0.002, 0.004, 0.007, 0.004, 0.004, 0.004, 0.004, 0.002, 0.002]
            _write_raw_capture(raw, initiator, responder, 0.1, 4.0, 1)
            manifest = root / "manifest.csv"
            proofs = root / "proofs.csv"
            statuses = root / "status.csv"
            write_csv_rows(manifest, [autonomous_manifest(raw)])
            write_csv_rows(
                proofs,
                [
                    autonomous_proof("board-i", "initiator"),
                    autonomous_proof("board-r", "responder"),
                ],
            )
            write_csv_rows(
                statuses,
                [
                    {"board_id": "board-i", "state": "export_complete"},
                    {"board_id": "board-r", "state": "export_complete"},
                ],
            )
            metrics, traces, audit, joins = analyze_autonomous_captures(
                manifest,
                proofs,
                statuses,
                root / "analysis",
                build_context_csv=write_release_build_context(
                    root / "build_context.csv"
                ),
                require_complete_block=False,
            )
            require_valid_autonomous_audit(audit)
            pair = next(row for row in metrics if row["role"] == "pair")
            initiator_row = next(row for row in metrics if row["role"] == "initiator")

        # NGMO2 samples are interval averages: E = V * dt * sum(I[n]).
        self.assertAlmostEqual(float(initiator_row["energy_total_j"]), 0.0100)
        self.assertAlmostEqual(float(initiator_row["energy_incremental_j"]), 0.0060)
        self.assertAlmostEqual(float(pair["energy_total_j"]), 0.0240)
        self.assertAlmostEqual(float(pair["energy_incremental_j"]), 0.0120)
        # Pair peak is max_n V*(I_i[n]+I_r[n]), not the sum of endpoint maxima.
        self.assertAlmostEqual(float(pair["peak_power_w"]), 0.04)
        self.assertAlmostEqual(float(pair["average_power_w"]), 0.0240)
        self.assertEqual(pair["average_power_denominator"], "absolute_capture_energy_over_capture_duration")
        self.assertEqual(len(traces), 20)
        self.assertEqual(len(joins), 2)

    def test_final_audit_rejects_incomplete_manifest_and_board_proof_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "capture.csv"
            _write_raw_capture(raw, [0.001] * 10, [0.002] * 10, 0.1, 4.0, 1)
            manifest = root / "manifest.csv"
            proofs = root / "proofs.csv"
            statuses = root / "status.csv"
            write_csv_rows(manifest, [autonomous_manifest(raw)])
            write_csv_rows(
                proofs,
                [
                    autonomous_proof("board-i", "initiator"),
                    autonomous_proof("board-r", "responder"),
                ],
            )
            write_csv_rows(
                statuses,
                [
                    {"board_id": "board-i", "state": "export_complete"},
                    {"board_id": "board-r", "state": "export_complete"},
                ],
            )
            metrics, _, audit, _ = analyze_autonomous_captures(
                manifest,
                proofs,
                statuses,
                root / "analysis",
                build_context_csv=write_release_build_context(
                    root / "build_context.csv"
                ),
            )
        self.assertFalse(metrics)
        self.assertIn(
            "capture_block_not_exact_firmware_sequences_0_through_104",
            audit[0]["issues"],
        )
        self.assertIn(
            "initiator_proof_set_not_exact_firmware_sequences_0_through_104",
            audit[0]["issues"],
        )
        self.assertIn(
            "responder_proof_set_not_exact_firmware_sequences_0_through_104",
            audit[0]["issues"],
        )

    def test_coherent_pre_session_failure_is_retained_with_unrecorded_responder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "capture.csv"
            _write_raw_capture(raw, [0.001] * 10, [0.002] * 10, 0.1, 4.0, 1)
            manifest = root / "manifest.csv"
            proofs = root / "proofs.csv"
            statuses = root / "status.csv"
            write_csv_rows(manifest, [autonomous_manifest(raw)])
            failed = autonomous_proof(
                "board-i",
                "unresolved",
                success=0,
                record_ready=0,
                remote_session_started=0,
                event_duration_us=0,
            )
            responder = autonomous_proof(
                "board-r",
                "unresolved",
                success=0,
                record_ready=0,
                remote_session_started=0,
                event_duration_us=0,
            )
            write_csv_rows(proofs, [failed, responder])
            write_csv_rows(
                statuses,
                [
                    {"board_id": "board-i", "state": "export_complete"},
                    {"board_id": "board-r", "state": "export_complete"},
                ],
            )
            metrics, _, audit, joins = analyze_autonomous_captures(
                manifest,
                proofs,
                statuses,
                root / "analysis",
                build_context_csv=write_release_build_context(
                    root / "build_context.csv"
                ),
                require_complete_block=False,
            )
            require_valid_autonomous_audit(audit)
        initiator_row = next(row for row in metrics if row["role"] == "initiator")
        responder_row = next(
            row for row in metrics if row["role"] == "responder_background"
        )
        self.assertEqual(initiator_row["outcome"], "failure")
        self.assertEqual(initiator_row["endpoint_event_present"], 0)
        self.assertEqual(responder_row["outcome"], "background")
        self.assertEqual(len(joins), 2)

    def test_autonomous_fingerprint_mismatch_is_quality_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "capture.csv"
            _write_raw_capture(raw, [0.001] * 10, [0.002] * 10, 0.1, 4.0, 1)
            manifest = root / "manifest.csv"
            proofs = root / "proofs.csv"
            statuses = root / "status.csv"
            write_csv_rows(manifest, [autonomous_manifest(raw)])
            bad = autonomous_proof("board-i", "initiator")
            bad["artifact_hash"] = "deadbeef"
            write_csv_rows(
                proofs, [bad, autonomous_proof("board-r", "responder")]
            )
            write_csv_rows(
                statuses,
                [
                    {"board_id": "board-i", "state": "export_complete"},
                    {"board_id": "board-r", "state": "export_complete"},
                ],
            )
            metrics, _, audit, _ = analyze_autonomous_captures(
                manifest,
                proofs,
                statuses,
                root / "analysis",
                build_context_csv=write_release_build_context(
                    root / "build_context.csv"
                ),
                require_complete_block=False,
            )
        self.assertFalse(metrics)
        self.assertIn("artifact_fingerprint_mismatch", audit[0]["issues"])
        with self.assertRaisesRegex(ValueError, "proof/energy audit failed"):
            require_valid_autonomous_audit(audit)

    def test_autonomous_range_overload_and_current_limiting_are_rejected(self) -> None:
        for bad_current, expected_issue in (
            (0.510, "dynamic_range_overload"),
            (0.095, "at_or_near_programmed_current_limit"),
        ):
            with self.subTest(bad_current=bad_current), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                raw = root / "capture.csv"
                initiator = [0.001] * 10
                initiator[4] = bad_current
                _write_raw_capture(raw, initiator, [0.002] * 10, 0.1, 4.0, 1)
                manifest = root / "manifest.csv"
                proofs = root / "proofs.csv"
                statuses = root / "status.csv"
                write_csv_rows(manifest, [autonomous_manifest(raw)])
                write_csv_rows(
                    proofs,
                    [
                        autonomous_proof("board-i", "initiator"),
                        autonomous_proof("board-r", "responder"),
                    ],
                )
                write_csv_rows(
                    statuses,
                    [
                        {"board_id": "board-i", "state": "export_complete"},
                        {"board_id": "board-r", "state": "export_complete"},
                    ],
                )
                metrics, _, audit, _ = analyze_autonomous_captures(
                    manifest,
                    proofs,
                    statuses,
                    root / "analysis",
                    build_context_csv=write_release_build_context(
                        root / "build_context.csv"
                    ),
                    require_complete_block=False,
                )
            self.assertFalse(metrics)
            self.assertIn(expected_issue, audit[0]["issues"])

    def test_rectangular_energy_pointwise_pair_peak_and_exact_responder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "capture.csv"
            initiator = [0.001, 0.001, 0.006, 0.003, 0.003, 0.003, 0.003, 0.003, 0.001, 0.001]
            responder = [0.002, 0.002, 0.004, 0.007, 0.004, 0.004, 0.004, 0.004, 0.002, 0.002]
            _write_raw_capture(raw, initiator, responder, 0.01, 3.8, 1)
            events_path = root / "event_metrics.csv"
            write_csv_rows(events_path, accepted_events())
            manifest_path = root / "ngmo2_capture_manifest.csv"
            write_csv_rows(
                manifest_path,
                [
                    {
                        "capture_id": "c1",
                        "run_id": "run-1",
                        "pair_id": "pair-1",
                        "board_id": "board-i",
                        "event_index": 10,
                        "exchange_id": 7,
                        "event_path": "accepted_connected",
                        "sample_interval_s": 0.01,
                        "sample_count": 10,
                        "voltage_v": 3.8,
                        "voltage_basis": "verified_programmed_setpoint_not_dynamic_voltage",
                        "canonical_csv": str(raw),
                        "canonical_csv_sha256": sha256_file(raw),
                    }
                ],
            )
            output = root / "analysis"
            metrics, traces, audit = analyze_envelope_captures(
                manifest_path, events_path, None, output
            )
            require_valid_ngmo2_audit(audit)

            pair = next(row for row in metrics if row.get("role") == "pair")
            responder_row = next(
                row
                for row in metrics
                if row.get("power_channel_role") == "responder"
                and row.get("role") == "responder"
            )
            self.assertAlmostEqual(float(pair["energy_incremental_j"]), 0.00114)
            self.assertAlmostEqual(float(pair["energy_total_j"]), 0.001824)
            # max_t(V*(I1+I2)) = 3.8*0.010, not 3.8*(0.006+0.007).
            self.assertAlmostEqual(float(pair["peak_power_w"]), 0.038)
            self.assertEqual(responder_row["board_id"], "board-r")
            self.assertEqual(str(responder_row["event_index"]), "20")
            self.assertEqual(len(traces), 20)
            self.assertEqual(read_rows(output / "ngmo2_capture_audit.csv")[0]["status"], "pass")

    def test_firmware_communication_counts_senders_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_path = root / "event_metrics.csv"
            write_csv_rows(events_path, accepted_events())
            metrics, audit = analyze_communication(events_path, root / "analysis")
            require_valid_communication_audit(audit)
            pair = next(row for row in metrics if row["role"] == "pair")
            self.assertEqual(pair["application_payload_bytes"], 150)
            self.assertEqual(pair["pulse_framing_bytes"], 7)
            self.assertEqual(pair["att_value_bytes"], 157)
            self.assertEqual(pair["att_value_fragment_count"], 3)
            self.assertEqual(pair["direction"], "sender_sum_no_rx_double_count")

    def test_communication_endpoint_mismatch_fails_audit(self) -> None:
        rows = accepted_events()
        rows[1]["application_rx_bytes"] = 99
        rows[1]["att_rx_bytes"] = 103
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_path = root / "event_metrics.csv"
            write_csv_rows(events_path, rows)
            _, audit = analyze_communication(events_path, root / "analysis")
            with self.assertRaisesRegex(ValueError, "endpoint_mismatch"):
                require_valid_communication_audit(audit)


if __name__ == "__main__":
    unittest.main()
