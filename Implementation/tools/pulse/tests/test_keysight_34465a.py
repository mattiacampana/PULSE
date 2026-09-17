from __future__ import annotations

import argparse
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

from keysight_34465a import (  # noqa: E402
    EXPECTED_TRIGGERS,
    _stream_event_onsets,
    _validate_role_rows,
    _event_identity,
    _event_metrics,
    capture_keysight,
    capture_keysight_stream,
    capture_configuration_commands,
    pilot_configuration_commands,
    pilot_keysight,
    stream_configuration_commands,
    validate_capture_settings,
    verify_keysight_idn,
)
from common import read_csv_rows  # noqa: E402
from pulse_tools import build_parser  # noqa: E402


def _capture_test_args(output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        measured_role="responder",
        sample_interval_s=0.02,
        aperture_s=0.019,
        samples_per_trigger=225,
        pretrigger_samples=20,
        current_range_a=0.01,
        current_terminal_a=3,
        trigger_level_a=0.005,
        capture_timeout_s=30.0,
        poll_interval_s=0.01,
        output_dir=output_dir,
        run_id="test-run",
        pair_id="p01",
        board_id="responder-board",
        peer_board_id="initiator-board",
        resource="USB::INSTR",
        supply_voltage_v=3.7,
        supply_voltage_source="test source",
        source_current_limit_a=0.1,
        source_overvoltage_protection_v=4.0,
        maximum_safe_voltage_v=4.2,
        calibration_id="CAL-001",
        calibration_date="2026-01-01",
        firmware_revision="revision",
        firmware_image_sha256="a" * 64,
        artifact_sha256="b" * 64,
    )


class CaptureInstrument:
    def __init__(self, records: list[list[float]]) -> None:
        self.records = records
        self.writes: list[str] = []

    def write(self, command: str) -> None:
        self.writes.append(command)

    def query(self, command: str) -> str:
        responses = {
            "*IDN?": "Keysight Technologies,34465A,MY12345678,A.03.03",
            "*OPC?": "1",
            "*ESR?": "1",
            "SYST:ERR?": '+0,"No error"',
            "SENS:CURR:DC:RANG?": "0.01",
            "SENS:CURR:DC:TERM?": "3",
            "SENS:CURR:DC:APER?": "0.019",
            "SAMP:TIM?": "0.02",
            "SAMP:COUN?": "225",
            "SAMP:COUN:PRET?": "20",
            "TRIG:COUN?": "1",
            "TRIG:SOUR?": "INT",
            "TRIG:LEV?": "0.005",
            "TRIG:SLOP?": "POS",
            "TRIG:DEL?": "0",
            "STAT:QUES:COND?": "0",
        }
        if command == "DATA:POIN?":
            return str(len(self.records[self.writes.count("INIT") - 1]))
        return responses[command]

    def query_ascii_values(self, command: str) -> list[float]:
        self.writes.append(command)
        return self.records[self.writes.count("INIT") - 1]


class StreamInstrument(CaptureInstrument):
    def __init__(self, readings: list[float]) -> None:
        super().__init__([])
        self.readings = readings
        self.produced = 0
        self.removed = 0

    def query(self, command: str) -> str:
        if command == "*ESR?":
            return "0"
        if command == "STAT:QUES:COND?" and any(
            value >= 9.0e36 for value in self.readings[: self.removed]
        ):
            return "2"
        if command == "DATA:POIN?":
            self.produced = min(len(self.readings), self.produced + 100)
            return str(self.produced - self.removed)
        if command == "SAMP:COUN?":
            return "6000"
        if command == "SAMP:COUN:PRET?":
            return "0"
        if command == "TRIG:SOUR?":
            return "IMM"
        return super().query(command)

    def query_ascii_values(self, command: str) -> list[float]:
        self.writes.append(command)
        count = int(command.split()[-1])
        result = self.readings[self.removed : self.removed + count]
        self.removed += count
        return result


class KeysightConfigurationTests(unittest.TestCase):
    def test_default_initiator_capture_fits_standard_memory(self) -> None:
        self.assertEqual(47_250, validate_capture_settings("initiator"))
        commands = capture_configuration_commands("initiator")
        self.assertIn("SAMP:TIM 0.02", commands)
        self.assertIn("SAMP:COUN 225", commands)
        self.assertIn("SAMP:COUN:PRET 20", commands)
        self.assertIn("TRIG:COUN 1", commands)
        self.assertNotIn("TRIG:COUN 210", commands)
        self.assertIn("TRIG:SOUR INT", commands)
        self.assertIn("TRIG:LEV 0.005", commands)
        self.assertIn("TRIG:SLOP POS", commands)
        self.assertIn("CONF:CURR:DC 0.01", commands)
        self.assertIn("SENS:CURR:DC:TERM 3", commands)

    def test_rejects_invalid_current_trigger_level(self) -> None:
        with self.assertRaisesRegex(ValueError, "trigger level"):
            validate_capture_settings("initiator", trigger_level_a=1.0)

    def test_pilot_uses_immediate_trigger_and_no_pretrigger(self) -> None:
        commands, count = pilot_configuration_commands()
        self.assertEqual(9000, count)
        self.assertIn("SAMP:COUN:PRET 0", commands)
        self.assertIn("TRIG:SOUR IMM", commands)
        self.assertIn("TRIG:COUN 1", commands)
        self.assertFalse(any(command.startswith("TRIG:LEV") for command in commands))
        self.assertIn("CONF:CURR:DC 0.01", commands)
        self.assertIn("SENS:CURR:DC:TERM 3", commands)

    def test_pilot_rejects_capture_larger_than_standard_memory(self) -> None:
        with self.assertRaisesRegex(ValueError, "standard DMM memory"):
            pilot_configuration_commands(duration_s=1200.0)

    def test_default_responder_capture_has_connected_triggers_only(self) -> None:
        self.assertEqual(23_625, validate_capture_settings("responder"))
        self.assertEqual(105, EXPECTED_TRIGGERS["responder"])
        self.assertEqual(("accepted_connected", "responder", 0), _event_identity("responder", 0))
        self.assertEqual(("local", "local", 104), _event_identity("initiator", 104))
        self.assertEqual(
            ("accepted_connected", "initiator", 0), _event_identity("initiator", 105)
        )

    def test_rejects_unsupported_range_terminal_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "10 A terminal forces"):
            validate_capture_settings("initiator", current_range_a=0.1, current_terminal_a=10)
        with self.assertRaisesRegex(ValueError, "10 A terminal forces"):
            validate_capture_settings("initiator", current_range_a=1.0, current_terminal_a=10)

    def test_allows_100ma_fallback_on_3a_terminal(self) -> None:
        commands = capture_configuration_commands(
            "initiator", current_range_a=0.1, current_terminal_a=3
        )
        self.assertIn("CONF:CURR:DC 0.1", commands)

    def test_idn_must_name_34465a(self) -> None:
        identity = "Keysight Technologies,34465A,MY12345678,A.03.03"
        self.assertEqual(identity, verify_keysight_idn(identity))
        with self.assertRaisesRegex(ValueError, "expected a Keysight 34465A"):
            verify_keysight_idn("Keysight Technologies,34461A,MY12345678,A.03.03")

    def test_capture_rearms_once_per_pretrigger_record(self) -> None:
        class FakeInstrument:
            def __init__(self) -> None:
                self.writes: list[str] = []

            def write(self, command: str) -> None:
                self.writes.append(command)

            def query(self, command: str) -> str:
                responses = {
                    "*IDN?": "Keysight Technologies,34465A,MY12345678,A.03.03",
                    "*OPC?": "1",
                    "*ESR?": "1",
                    "SYST:ERR?": '+0,"No error"',
                    "SENS:CURR:DC:RANG?": "0.01",
                    "SENS:CURR:DC:TERM?": "3",
                    "SENS:CURR:DC:APER?": "0.019",
                    "SAMP:TIM?": "0.02",
                    "SAMP:COUN?": "225",
                    "SAMP:COUN:PRET?": "20",
                    "TRIG:COUN?": "1",
                    "TRIG:SOUR?": "INT",
                    "TRIG:LEV?": "0.005",
                    "TRIG:SLOP?": "POS",
                    "TRIG:DEL?": "0",
                    "DATA:POIN?": "225",
                    "STAT:QUES:COND?": "0",
                }
                return responses[command]

            def query_ascii_values(self, command: str) -> list[float]:
                self.writes.append(command)
                return [0.001] * 20 + [0.006] * 205

        with tempfile.TemporaryDirectory() as temporary_directory:
            args = argparse.Namespace(
                measured_role="responder",
                sample_interval_s=0.02,
                aperture_s=0.019,
                samples_per_trigger=225,
                pretrigger_samples=20,
                current_range_a=0.01,
                current_terminal_a=3,
                trigger_level_a=0.005,
                capture_timeout_s=30.0,
                poll_interval_s=0.01,
                output_dir=Path(temporary_directory),
                run_id="test-run",
                pair_id="p01",
                board_id="responder-board",
                peer_board_id="initiator-board",
                resource="USB::INSTR",
                supply_voltage_v=3.7,
                supply_voltage_source="test source",
                source_current_limit_a=0.1,
                source_overvoltage_protection_v=4.0,
                maximum_safe_voltage_v=4.2,
                calibration_id="CAL-001",
                calibration_date="2026-01-01",
                firmware_revision="revision",
                firmware_image_sha256="a" * 64,
                artifact_sha256="b" * 64,
            )
            instrument = FakeInstrument()
            with patch.dict(EXPECTED_TRIGGERS, {"responder": 2}):
                samples_path, _ = capture_keysight(instrument, args)

            self.assertEqual(2, instrument.writes.count("INIT"))
            self.assertEqual(2, instrument.writes.count("*OPC"))
            self.assertEqual(2, instrument.writes.count("FETC?"))
            self.assertEqual(451, len(samples_path.read_text(encoding="utf-8").splitlines()))

    def test_early_short_idle_record_is_saved_and_rearmed(self) -> None:
        valid = [0.001] * 20 + [0.006] * 205
        instrument = CaptureInstrument([[0.001] * 206, valid, valid])
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with patch.dict(EXPECTED_TRIGGERS, {"responder": 2}):
                samples_path, _ = capture_keysight(
                    instrument, _capture_test_args(output_dir)
                )
            diagnostics = list(output_dir.glob("*diagnostic.csv"))
            self.assertEqual(3, instrument.writes.count("INIT"))
            self.assertEqual(3, instrument.writes.count("*OPC"))
            self.assertEqual(207, len(diagnostics[0].read_text(encoding="utf-8").splitlines()))
            self.assertEqual(1, len(diagnostics))
            self.assertEqual(451, len(samples_path.read_text(encoding="utf-8").splitlines()))

    def test_short_later_event_fails_with_diagnostic(self) -> None:
        valid = [0.001] * 20 + [0.006] * 205
        instrument = CaptureInstrument([valid, valid[:-19]])
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with patch.dict(EXPECTED_TRIGGERS, {"responder": 2}):
                with self.assertRaisesRegex(RuntimeError, "short record"):
                    capture_keysight(instrument, _capture_test_args(output_dir))
            self.assertEqual(2, instrument.writes.count("INIT"))
            self.assertEqual(1, len(list(output_dir.glob("*diagnostic.csv"))))
            self.assertFalse(list(output_dir.glob("*samples.csv")))

    def test_overload_record_fails_with_sample_index_and_diagnostic(self) -> None:
        overloaded = [0.001] * 19 + [9.9e37] + [0.006] * 205
        instrument = CaptureInstrument([overloaded])
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with patch.dict(EXPECTED_TRIGGERS, {"responder": 1}):
                with self.assertRaisesRegex(RuntimeError, "sample 19"):
                    capture_keysight(instrument, _capture_test_args(output_dir))
            self.assertEqual(1, len(list(output_dir.glob("*diagnostic.csv"))))
            self.assertFalse(list(output_dir.glob("*samples.csv")))

    def test_stream_drains_memory_and_extracts_complete_windows(self) -> None:
        commands, count = stream_configuration_commands(duration_s=120)
        self.assertEqual(6000, count)
        self.assertIn("TRIG:SOUR IMM", commands)
        self.assertIn("SAMP:COUN:PRET 0", commands)
        readings = [0.001] * 1800
        readings[500:503] = [0.006] * 3  # ignored during the startup guard
        readings[1200:1350] = [0.006] * 150
        readings[1500:1650] = [0.006] * 150
        self.assertEqual([1200, 1500], _stream_event_onsets(readings, 0.005, 0.02))
        instrument = StreamInstrument(readings)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            args = _capture_test_args(output_dir)
            args.capture_timeout_s = 120.0
            with patch.dict(EXPECTED_TRIGGERS, {"responder": 2}):
                samples_path, manifest_path = capture_keysight_stream(instrument, args)
                rows = read_csv_rows(samples_path)
                _validate_role_rows(rows, "responder")
            raw_rows = read_csv_rows(output_dir / "keysight_34465a_responder_stream_raw.csv")
            self.assertEqual(450, len(rows))
            self.assertEqual(1800, len(raw_rows))
            self.assertEqual("1180", rows[0]["source_sample_index"])
            self.assertEqual("1480", rows[225]["source_sample_index"])
            self.assertIn("continuous_stream", rows[0]["capture_profile"])
            self.assertIn("complete_timed_stream", manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(any(write.startswith("DATA:REMove?") for write in instrument.writes))
            self.assertIn("ABOR", instrument.writes)

    def test_stream_ignores_idle_overload_without_shifting_event_windows(self) -> None:
        readings = [0.001] * 1800
        readings[1100] = 9.9e37
        readings[1200:1350] = [0.006] * 150
        readings[1500:1650] = [0.006] * 150
        instrument = StreamInstrument(readings)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            args = _capture_test_args(output_dir)
            args.capture_timeout_s = 120.0
            with patch.dict(EXPECTED_TRIGGERS, {"responder": 2}):
                samples_path, manifest_path = capture_keysight_stream(instrument, args)
                rows = read_csv_rows(samples_path)
                _validate_role_rows(rows, "responder")
            raw_path = output_dir / "keysight_34465a_responder_stream_raw.csv"
            self.assertTrue(raw_path.exists())
            raw_rows = read_csv_rows(raw_path)
            self.assertEqual("9.9e+37", raw_rows[1100]["current_a"])
            self.assertEqual("1", rows[0]["raw_invalid_reading_count"])
            self.assertEqual("1100", rows[0]["raw_invalid_sample_indices"])
            self.assertEqual("1180", rows[0]["source_sample_index"])
            self.assertEqual("2", read_csv_rows(manifest_path)[0]["questionable_condition"])

    def test_stream_does_not_treat_invalid_reading_as_rising_crossing(self) -> None:
        readings = [0.001] * 1200
        readings[1100:1150] = [0.006] * 50
        readings[1120] = 9.9e37
        self.assertEqual([1100], _stream_event_onsets(readings, 0.005, 0.02))

    def test_stream_rejects_overload_inside_event_window_after_capture(self) -> None:
        readings = [0.001] * 1800
        readings[1200:1350] = [0.006] * 150
        readings[1210] = 9.9e37
        readings[1500:1650] = [0.006] * 150
        instrument = StreamInstrument(readings)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            args = _capture_test_args(output_dir)
            args.capture_timeout_s = 120.0
            with patch.dict(EXPECTED_TRIGGERS, {"responder": 2}):
                with self.assertRaisesRegex(RuntimeError, "event 0 window contains"):
                    capture_keysight_stream(instrument, args)
            raw_path = output_dir / "keysight_34465a_responder_stream_raw.csv"
            self.assertTrue(raw_path.exists())
            self.assertEqual(1801, len(raw_path.read_text(encoding="utf-8").splitlines()))
            self.assertFalse((output_dir / "keysight_34465a_responder_samples.csv").exists())

    def test_pilot_saves_unclassified_current_trace(self) -> None:
        class FakeInstrument:
            def __init__(self) -> None:
                self.writes: list[str] = []

            def write(self, command: str) -> None:
                self.writes.append(command)

            def query(self, command: str) -> str:
                responses = {
                    "*IDN?": "Keysight Technologies,34465A,MY12345678,A.03.03",
                    "*OPC?": "1",
                    "SYST:ERR?": '+0,"No error"',
                    "SENS:CURR:DC:RANG?": "0.01",
                    "SENS:CURR:DC:TERM?": "3",
                    "SENS:CURR:DC:APER?": "0.019",
                    "SAMP:TIM?": "0.02",
                    "SAMP:COUN?": "50",
                    "SAMP:COUN:PRET?": "0",
                    "TRIG:COUN?": "1",
                    "TRIG:SOUR?": "IMM",
                    "DATA:POIN?": "50",
                    "STAT:QUES:COND?": "0",
                }
                return responses[command]

            def query_ascii_values(self, command: str) -> list[float]:
                self.writes.append(command)
                return [0.004] * 50

        with tempfile.TemporaryDirectory() as temporary_directory:
            args = argparse.Namespace(
                measured_role="initiator",
                board_id="initiator-board",
                resource="USB::INSTR",
                output_dir=Path(temporary_directory),
                pilot_duration_s=1.0,
                sample_interval_s=0.02,
                aperture_s=0.019,
                current_range_a=0.01,
                current_terminal_a=3,
                poll_interval_s=0.1,
                supply_voltage_v=3.7,
                source_current_limit_a=0.1,
                source_overvoltage_protection_v=4.0,
                maximum_safe_voltage_v=4.2,
            )
            instrument = FakeInstrument()
            samples_path, manifest_path = pilot_keysight(instrument, args)
            self.assertIn("TRIG:SOUR IMM", instrument.writes)
            self.assertEqual(51, len(samples_path.read_text(encoding="utf-8").splitlines()))
            self.assertIn("not_paper_data", manifest_path.read_text(encoding="utf-8"))


class KeysightEnergyTests(unittest.TestCase):
    def test_fixed_window_energy_subtracts_same_record_pretrigger(self) -> None:
        rows = []
        currents = (0.001, 0.001, 0.003, 0.005)
        times = (-0.04, -0.02, 0.0, 0.02)
        for sample_index, (relative_time, current) in enumerate(zip(times, currents, strict=True)):
            rows.append(
                {
                    "run_id": "run-i",
                    "pair_id": "p01",
                    "board_id": "board-i",
                    "event_path": "local",
                    "event_role": "local",
                    "path_index": "5",
                    "trial_id": "0",
                    "warmup": "0",
                    "supply_voltage_v": "4.0",
                    "sample_interval_s": "0.02",
                    "aperture_s": "0.019",
                    "pretrigger_samples": "2",
                    "relative_time_s": str(relative_time),
                    "current_a": str(current),
                    "sample_index": str(sample_index),
                    "outcome_join": "preflight_only_not_exact_power_trial_outcomes",
                    "firmware_revision": "revision",
                    "firmware_image_sha256": "a" * 64,
                    "artifact_sha256": "b" * 64,
                }
            )
        metric, trace = _event_metrics(rows)
        self.assertTrue(math.isclose(float(metric["baseline_current_a"]), 0.001))
        self.assertTrue(math.isclose(float(metric["energy_absolute_j"]), 0.00064))
        self.assertTrue(math.isclose(float(metric["energy_incremental_j"]), 0.00048))
        self.assertEqual(4, len(trace))


class KeysightCommandLineTests(unittest.TestCase):
    def test_capture_cli_accepts_10ma_range(self) -> None:
        parser = build_parser()
        capture = parser.parse_args(
            [
                "keysight-capture",
                "--dry-run",
                "--measured-role",
                "initiator",
                "--current-range-a",
                "0.01",
                "--current-terminal-a",
                "3",
                "--trigger-level-a",
                "0.003",
            ]
        )
        self.assertEqual(0.01, capture.current_range_a)
        self.assertEqual(0.003, capture.trigger_level_a)

    def test_keysight_subcommands_are_registered(self) -> None:
        parser = build_parser()
        capture = parser.parse_args(
            ["keysight-capture", "--dry-run", "--measured-role", "initiator"]
        )
        self.assertEqual("keysight-capture", capture.command)
        figure = parser.parse_args(
            [
                "keysight-fig4a",
                "--initiator-csv",
                "initiator.csv",
                "--responder-csv",
                "responder.csv",
                "--preflight-events-csv",
                "events.csv",
                "--preflight-metadata-csv",
                "metadata.csv",
                "--output-dir",
                "results",
            ]
        )
        self.assertEqual("keysight-fig4a", figure.command)


if __name__ == "__main__":
    unittest.main()
