from __future__ import annotations

import csv
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(TOOL_DIR))

from power_analysis import (  # noqa: E402
    EventWindow,
    analyze_power,
    join_windows_to_events,
    load_baseline_metadata,
)
from pulse_tools import _check_audit  # noqa: E402
from energy_projection import project_daily_energy  # noqa: E402
from figures import (  # noqa: E402
    _capture_environment_issues,
    _representative_event,
    _validate_section_v_coverage,
    prepare_figure_csvs,
)
from resource_analysis import analyze_resources  # noqa: E402
from serial_parser import parse_serial_files  # noqa: E402
from summarize import (  # noqa: E402
    accepted_session_integrity,
    aggregate_rows,
    protocol_audit,
    summarize_directory,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


ENERGY_LEDGER_FIELDS = [
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
]


def complete_energy_ledger_rows() -> list[dict[str, object]]:
    base: dict[str, object] = {
        "scenario": "cambridge",
        "seed": "7",
        "node_id": "node_1",
        "network_node_count": 1,
        "observation_duration_s": 126.6 * 3600.0,
        "local_opportunities": 10,
        "selection_opportunities": 5,
        "count_basis": "empirical_rate_model",
        "source_artifact": "complete-count-manifest.json",
        "source_artifact_sha256": "a" * 64,
    }
    categories = [
        ("local", "local", "connected", "success", "na", 10),
        ("local", "local", "connected", "failure", "na", 0),
        ("no_contact", "local", "disconnected", "success", "na", 3),
        ("no_contact", "local", "disconnected", "failure", "na", 0),
        ("accepted_connected", "initiator", "connected", "success", "1", 1),
        ("accepted_connected", "initiator", "connected", "failure", "0", 0),
        ("accepted_connected", "initiator", "connected", "failure", "1", 0),
        ("accepted_connected", "responder", "connected", "success", "1", 1),
        ("accepted_connected", "responder", "connected", "failure", "1", 0),
        ("accepted_discovery", "initiator", "disconnected", "success", "1", 1),
        ("accepted_discovery", "initiator", "disconnected", "failure", "0", 0),
        ("accepted_discovery", "initiator", "disconnected", "failure", "1", 0),
        ("accepted_discovery", "responder", "connected", "success", "1", 1),
        ("accepted_discovery", "responder", "connected", "failure", "1", 0),
    ]
    return [
        {
            **base,
            "event_path": event_path,
            "role": role,
            "connection_state": connection_state,
            "outcome": outcome,
            "remote_session_started": remote_started,
            "count_in_observation": count,
        }
        for event_path, role, connection_state, outcome, remote_started, count in categories
    ]


def write_energy_ledger(path: Path, rows: list[dict[str, object]]) -> None:
    manifest_path = path.parent / "complete-count-manifest.json"
    manifest_path.write_text('{"fixture":"complete energy ledger"}\n', encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    prepared_rows = [
        {
            **row,
            "source_artifact": manifest_path.name,
            "source_artifact_sha256": manifest_hash,
        }
        for row in rows
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENERGY_LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(prepared_rows)


def write_energy_summary(path: Path, *, omit_p95: bool = False) -> None:
    fields = [
        "event_path",
        "role",
        "connection_state",
        "remote_session_started",
        "measurement_scope",
        "baseline_source",
        "outcome",
        "metric",
        "n",
        "mean",
        "median",
        "q1",
        "q3",
        "p95",
    ]
    categories = [
        ("local", "local", "connected", "na", "event_gate", "success"),
        ("no_contact", "local", "disconnected", "na", "event_gate", "success"),
        ("accepted_connected", "initiator", "connected", "1", "pair_union", "success"),
        ("accepted_connected", "responder", "connected", "1", "pair_union", "success"),
        ("accepted_discovery", "initiator", "disconnected", "1", "pair_union", "success"),
        ("accepted_discovery", "responder", "connected", "1", "pair_union", "success"),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event_path, role, connection_state, remote_started, scope, outcome in categories:
            writer.writerow(
                {
                    "event_path": event_path,
                    "role": role,
                    "connection_state": connection_state,
                    "remote_session_started": remote_started,
                    "measurement_scope": scope,
                    "baseline_source": "manual_numeric",
                    "outcome": outcome,
                    "metric": "energy_incremental_j",
                    "n": 100,
                    "mean": 0.0025,
                    "median": 0.002,
                    "q1": 0.001,
                    "q3": 0.003,
                    "p95": "" if omit_p95 and event_path == "local" else 0.004,
                }
            )


class SerialParserTests(unittest.TestCase):
    def test_key_value_and_headered_positional_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            records = parse_serial_files(
                [FIXTURES / "serial.log"], output, {"pair_id": "operator_override"}
            )
            self.assertEqual(2, len(records["PULSE_EVENT"]))
            self.assertEqual("initiator", records["PULSE_EVENT"][0]["role"])
            self.assertEqual("connected", records["PULSE_EVENT"][0]["connection_state"])
            self.assertEqual("responder", records["PULSE_EVENT"][1]["role"])
            self.assertEqual("operator_override", records["PULSE_STAGE"][0]["pair_id"])
            self.assertEqual("0.9999", records["PULSE_CORRECTNESS"][0]["gradient_cosine_similarity"])
            self.assertEqual(2, len(read_rows(output / "event_metrics.csv")))

    def test_headerless_positional_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            log = output / "positional.log"
            log.write_text(
                "PULSE_CORRECTNESS,run_x,board_x,local,9,0.998,0.0002,1\n",
                encoding="utf-8",
            )
            records = parse_serial_files([log], output)
            row = records["PULSE_CORRECTNESS"][0]
            self.assertEqual("9", row["trial_id"])
            self.assertEqual("0.998", row["gradient_cosine_similarity"])

    def test_build_contract_hash_changes_and_stage_inherits_event_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "contracts.log"
            log.write_text(
                "PULSE_META,run_id=run1,pair_id=pair1,board_id=board_a,role=initiator,"
                "build_variant=pulse_release,learning_rate=0.01,peer_limit=8\n"
                "PULSE_EVENT,event_index=1,trial_id=1,role=initiator,event_path=local,"
                "connection_state=connected,warmup=0,success=1,att_mtu=247,data_length=251,"
                "tx_phy=2,rx_phy=2,connection_interval_units=12\n"
                "PULSE_STAGE,event_index=1,trial_id=1,role=initiator,event_path=local,"
                "stage=encoder,warmup=0,success=1,duration_us=10\n"
                "PULSE_META,run_id=run1,pair_id=pair1,board_id=board_a,role=initiator,"
                "build_variant=pulse_release,learning_rate=0.02,peer_limit=8\n"
                "PULSE_EVENT,event_index=2,trial_id=2,role=initiator,event_path=local,"
                "connection_state=connected,warmup=0,success=1,att_mtu=247,data_length=251,"
                "tx_phy=2,rx_phy=2,connection_interval_units=12\n",
                encoding="utf-8",
            )
            records = parse_serial_files([log], root / "parsed")
            hashes = [row["build_context_sha256"] for row in records["PULSE_META"]]
            self.assertTrue(all(len(value) == 64 for value in hashes))
            self.assertNotEqual(hashes[0], hashes[1])
            self.assertEqual(hashes[0], records["PULSE_EVENT"][0]["build_context_sha256"])
            stage = records["PULSE_STAGE"][0]
            self.assertEqual("247", stage["att_mtu"])
            self.assertEqual("251", stage["data_length"])
            self.assertEqual("12", stage["connection_interval_units"])


class PowerAnalysisTests(unittest.TestCase):
    def test_marker_decode_integration_and_trial_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            parse_serial_files([FIXTURES / "serial.log"], output)
            metrics, stages, traces = analyze_power(
                input_specs=[str(FIXTURES / "power_combined.csv")],
                output_dir=output,
                events_csv=output / "event_metrics.csv",
                metadata_csv=output / "run_metadata.csv",
                voltage_v=2.0,
                baseline_specs=["initiator=0.02", "responder=0.04"],
                stage_overrides=[],
                marker_threshold=0.5,
                active_low=False,
                allow_unmatched=False,
                trace_padding_ms=0.0,
            )
            self.assertEqual(
                {
                    "initiator",
                    "responder",
                    "pair",
                    "initiator_own_gate_diagnostic",
                    "responder_own_gate_diagnostic",
                },
                {row["role"] for row in metrics},
            )
            initiator = next(
                row
                for row in metrics
                if row["role"] == "initiator" and row["measurement_scope"] == "pair_union"
            )
            responder = next(
                row
                for row in metrics
                if row["role"] == "responder" and row["measurement_scope"] == "pair_union"
            )
            pair = next(row for row in metrics if row["role"] == "pair")
            self.assertTrue(math.isclose(float(initiator["energy_incremental_j"]), 0.0495, abs_tol=1e-9))
            self.assertTrue(math.isclose(float(responder["energy_incremental_j"]), 0.0675, abs_tol=1e-9))
            self.assertEqual(initiator["start_time_s"], responder["start_time_s"])
            self.assertEqual(initiator["end_time_s"], responder["end_time_s"])
            self.assertEqual("1", str(pair["trial_id"]))
            self.assertEqual("41", str(pair["exchange_id"]))
            self.assertEqual("paired_remote_session_exact_exchange_id", pair["join_status"])
            self.assertEqual("accepted_connected", pair["event_path"])
            self.assertTrue(math.isclose(float(pair["energy_total_j"]), 0.135, abs_tol=1e-9))
            self.assertTrue(
                math.isclose(
                    float(pair["energy_total_j"]),
                    float(initiator["energy_total_j"])
                    + float(responder["energy_total_j"]),
                    abs_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    float(pair["energy_incremental_j"]),
                    float(initiator["energy_incremental_j"])
                    + float(responder["energy_incremental_j"]),
                    abs_tol=1e-12,
                )
            )
            self.assertTrue(math.isclose(float(pair["peak_power_w"]), 0.6, abs_tol=1e-9))
            self.assertEqual({1, 3, 4, 5}, {int(row["stage_code"]) for row in stages})
            self.assertGreater(len(traces), 0)

    def test_local_events_join_only_to_board_mapped_initiator_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            events = output / "event_metrics.csv"
            metadata = output / "run_metadata.csv"
            power = output / "power.csv"
            with events.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "run_id", "pair_id", "board_id", "role", "trial_id",
                        "event_index", "event_path", "connection_state", "warmup",
                        "success", "record_index",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "run_id": "run1", "pair_id": "pair1", "board_id": "board_a",
                            "role": "local", "trial_id": 0, "event_index": 0,
                            "event_path": "local", "connection_state": "na", "warmup": 0,
                            "success": 1, "record_index": 1,
                        },
                        {
                            "run_id": "run1", "pair_id": "pair1", "board_id": "board_a",
                            "role": "local", "trial_id": 0, "event_index": 1,
                            "event_path": "no_contact", "connection_state": "disconnected",
                            "warmup": 0, "success": 1, "record_index": 2,
                        },
                        {
                            "run_id": "run1", "pair_id": "pair1", "board_id": "board_b",
                            "role": "responder", "trial_id": 0, "event_index": 2,
                            "event_path": "responder", "connection_state": "connected",
                            "warmup": 0, "success": 1, "record_index": 3,
                        },
                    ]
                )
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["board_id", "role"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"board_id": "board_a", "role": "initiator"},
                        {"board_id": "board_b", "role": "responder"},
                    ]
                )
            power.write_text(
                "time_s,initiator_current_a,responder_current_a,"
                "initiator_gpio0,initiator_gpio1,initiator_gpio2,initiator_gpio3,"
                "responder_gpio0,responder_gpio1,responder_gpio2,responder_gpio3\n"
                "0.0,0.01,0.02,0,0,0,0,0,0,0,0\n"
                "0.1,0.10,0.02,1,1,0,0,0,0,0,0\n"
                "0.2,0.01,0.02,0,0,0,0,0,0,0,0\n"
                "0.3,0.10,0.02,1,1,0,0,0,0,0,0\n"
                "0.4,0.01,0.02,0,0,0,0,0,0,0,0\n"
                "0.5,0.01,0.20,0,0,0,0,1,1,0,1\n"
                "0.6,0.01,0.02,0,0,0,0,0,0,0,0\n",
                encoding="utf-8",
            )

            metrics, _, _ = analyze_power(
                input_specs=[str(power)],
                output_dir=output,
                events_csv=events,
                metadata_csv=metadata,
                voltage_v=2.0,
                baseline_specs=["initiator=0.02", "responder=0.04"],
                stage_overrides=[],
                marker_threshold=0.5,
                active_low=False,
                allow_unmatched=False,
                trace_padding_ms=0.0,
            )

            self.assertEqual(3, len(metrics))
            local_rows = [row for row in metrics if row["role"] == "local"]
            self.assertEqual({"local", "no_contact"}, {row["event_path"] for row in local_rows})
            self.assertEqual({"board_a"}, {row["board_id"] for row in local_rows})
            self.assertEqual({"initiator"}, {row["power_channel_role"] for row in local_rows})
            responder = next(row for row in metrics if row["role"] == "responder")
            self.assertEqual("board_b", responder["board_id"])
            self.assertEqual("responder", responder["power_channel_role"])

    def test_baseline_only_accepts_all_gates_low_and_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            power = output / "baseline.csv"
            power.write_text(
                "time_s,current_a,event_gate,stage_bit0,stage_bit1,stage_bit2\n"
                "0.0,0.010,0,0,0,0\n"
                "0.1,0.012,0,0,0,0\n"
                "0.2,0.010,0,0,0,0\n",
                encoding="utf-8",
            )

            metrics, stages, traces = analyze_power(
                input_specs=[f"initiator={power}"],
                output_dir=output,
                events_csv=None,
                metadata_csv=None,
                voltage_v=2.0,
                baseline_specs=[],
                stage_overrides=[],
                marker_threshold=0.5,
                active_low=False,
                allow_unmatched=False,
                trace_padding_ms=0.0,
                baseline_only=True,
            )

            self.assertEqual(([], [], []), (metrics, stages, traces))
            capture = read_rows(output / "power_capture_metadata.csv")
            self.assertEqual(1, len(capture))
            self.assertEqual("baseline_only", capture[0]["capture_mode"])
            self.assertEqual("0", capture[0]["event_window_count"])
            self.assertEqual("1", capture[0]["all_event_gates_low"])
            self.assertTrue(math.isclose(float(capture[0]["baseline_power_w"]), 0.022))
            self.assertTrue(
                math.isclose(
                    float(capture[0]["diagnostic_gate_low_sample_median_power_w"]),
                    0.02,
                )
            )
            self.assertTrue(
                capture[0]["baseline_method"].startswith(
                    "matched_baseline_capture_integrated_mean"
                )
            )
            self.assertEqual(
                hashlib.sha256(power.read_bytes()).hexdigest(),
                capture[0]["source_sha256"],
            )
            self.assertEqual([], read_rows(output / "event_power_metrics.csv"))

    def test_legacy_baseline_metadata_without_raw_capture_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.csv"
            path.write_text(
                "role,capture_mode,baseline_power_w,baseline_method,source_file\n"
                "initiator,baseline_only,0.02,"
                "matched_baseline_capture_integrated_mean_energy_over_duration,baseline.csv\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "concrete raw-power source_sha256"):
                load_baseline_metadata(path)

    def test_all_gates_low_requires_explicit_baseline_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            power = output / "missing-events.csv"
            power.write_text(
                "time_s,current_a,event_gate,stage_bit0,stage_bit1,stage_bit2\n"
                "0.0,0.010,0,0,0,0\n"
                "0.1,0.010,0,0,0,0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "use --baseline-only"):
                analyze_power(
                    input_specs=[f"initiator={power}"],
                    output_dir=output,
                    events_csv=None,
                    metadata_csv=None,
                    voltage_v=2.0,
                    baseline_specs=["initiator=0.02"],
                    stage_overrides=[],
                    marker_threshold=0.5,
                    active_low=False,
                    allow_unmatched=False,
                    trace_padding_ms=0.0,
                )

    def test_baseline_only_rejects_asserted_event_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            power = output / "not-a-baseline.csv"
            power.write_text(
                "time_s,current_a,event_gate,stage_bit0,stage_bit1,stage_bit2\n"
                "0.0,0.010,0,0,0,0\n"
                "0.1,0.020,1,1,0,0\n"
                "0.2,0.010,0,0,0,0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "all-gates-low"):
                analyze_power(
                    input_specs=[f"initiator={power}"],
                    output_dir=output,
                    events_csv=None,
                    metadata_csv=None,
                    voltage_v=2.0,
                    baseline_specs=[],
                    stage_overrides=[],
                    marker_threshold=0.5,
                    active_low=False,
                    allow_unmatched=False,
                    trace_padding_ms=0.0,
                    baseline_only=True,
                )

    def test_pre_session_failure_integrates_both_synchronized_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            events = output / "event_metrics.csv"
            power = output / "power.csv"
            with events.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "run_id", "pair_id", "board_id", "role", "trial_id",
                        "event_index", "exchange_id", "remote_session_started",
                        "event_path", "connection_state", "warmup", "success",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "run_id": "run1", "pair_id": "pair1", "board_id": "board_a",
                        "role": "initiator", "trial_id": 7, "event_index": 7,
                        "exchange_id": 107, "remote_session_started": 0,
                        "event_path": "accepted_discovery", "connection_state": "disconnected",
                        "warmup": 0, "success": 0,
                    }
                )
            power.write_text(
                "time_s,initiator_current_a,responder_current_a,"
                "initiator_gpio0,initiator_gpio1,initiator_gpio2,initiator_gpio3,"
                "responder_gpio0,responder_gpio1,responder_gpio2,responder_gpio3\n"
                "0.0,0.01,0.02,0,0,0,0,0,0,0,0\n"
                "0.1,0.10,0.03,1,1,1,1,0,0,0,0\n"
                "0.2,0.01,0.02,0,0,0,0,0,0,0,0\n",
                encoding="utf-8",
            )

            metrics, _, _ = analyze_power(
                input_specs=[str(power)],
                output_dir=output,
                events_csv=events,
                metadata_csv=None,
                voltage_v=2.0,
                baseline_specs=["initiator=0.02", "responder=0.04"],
                stage_overrides=[],
                marker_threshold=0.5,
                active_low=False,
                allow_unmatched=False,
                trace_padding_ms=0.0,
            )

            self.assertEqual(4, len(metrics))
            initiator = next(row for row in metrics if row["role"] == "initiator")
            responder_idle = next(
                row
                for row in metrics
                if row["role"] == "responder_idle_pair_union_diagnostic"
            )
            pair = next(row for row in metrics if row["role"] == "pair")
            self.assertEqual(0, pair["success"])
            self.assertEqual("107", str(pair["exchange_id"]))
            self.assertEqual(
                "initiator_only_pre_session_failure_exact_exchange_id",
                pair["join_status"],
            )
            self.assertGreater(float(pair["energy_total_j"]), float(initiator["energy_total_j"]))
            self.assertTrue(
                math.isclose(
                    float(pair["energy_total_j"]),
                    float(initiator["energy_total_j"])
                    + float(responder_idle["energy_total_j"]),
                    abs_tol=1e-12,
                )
            )

    def test_remote_session_missing_responder_power_is_not_waived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            events = output / "event_metrics.csv"
            power = output / "initiator.csv"
            with events.open("w", encoding="utf-8", newline="") as handle:
                fields = [
                    "run_id", "pair_id", "board_id", "role", "trial_id", "event_index", "exchange_id",
                    "remote_session_started", "event_path", "connection_state", "warmup", "success",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "run_id": "run1", "pair_id": "pair1", "board_id": "board_a", "role": "initiator",
                            "trial_id": 1, "event_index": 1, "exchange_id": 9,
                            "remote_session_started": 1, "event_path": "accepted_connected",
                            "connection_state": "connected", "warmup": 0, "success": 0,
                        },
                        {
                            "run_id": "run1", "pair_id": "pair1", "board_id": "board_b", "role": "responder",
                            "trial_id": 1, "event_index": 1, "exchange_id": 9,
                            "remote_session_started": 1,
                            "event_path": "accepted_connected", "connection_state": "connected",
                            "warmup": 0, "success": 0,
                        },
                    ]
                )
            power.write_text(
                "time_s,current_a,event_gate,stage_bit0,stage_bit1,stage_bit2\n"
                "0.0,0.01,0,0,0,0\n"
                "0.1,0.10,1,1,0,0\n"
                "0.2,0.01,0,0,0,0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "no matched responder power window"):
                analyze_power(
                    input_specs=[f"initiator={power}"],
                    output_dir=output,
                    events_csv=events,
                    metadata_csv=None,
                    voltage_v=2.0,
                    baseline_specs=["initiator=0.02"],
                    stage_overrides=[],
                    marker_threshold=0.5,
                    active_low=False,
                    allow_unmatched=True,
                    trace_padding_ms=0.0,
                )

    def test_power_join_requires_one_identity_and_contiguous_indices(self) -> None:
        windows = [
            EventWindow(index + 1, index, index, float(index), float(index + 1), False, False)
            for index in range(2)
        ]
        base = {
            "run_id": "run1",
            "pair_id": "pair1",
            "board_id": "board_a",
            "role": "initiator",
            "event_path": "accepted_connected",
            "connection_state": "connected",
        }
        cases = {
            "exactly one run/pair/board": [
                {**base, "event_index": "0"},
                {**base, "run_id": "run2", "event_index": "1"},
            ],
            "must be contiguous": [
                {**base, "event_index": "0"},
                {**base, "event_index": "2"},
            ],
        }
        for expected, rows in cases.items():
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                join_windows_to_events(windows, rows, "initiator", False)

    def test_pair_truncation_propagates_to_common_interval_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            parse_serial_files([FIXTURES / "serial.log"], output)
            power = output / "truncated.csv"
            power.write_text(
                "time_s,initiator_current_a,responder_current_a,"
                "initiator_gpio0,initiator_gpio1,initiator_gpio2,initiator_gpio3,"
                "responder_gpio0,responder_gpio1,responder_gpio2,responder_gpio3\n"
                "0.0,0.10,0.02,1,1,0,0,0,0,0,0\n"
                "0.1,0.10,0.20,1,1,0,0,1,1,0,1\n"
                "0.2,0.01,0.02,0,0,0,0,0,0,0,0\n",
                encoding="utf-8",
            )
            metrics, _, _ = analyze_power(
                input_specs=[str(power)],
                output_dir=output,
                events_csv=output / "event_metrics.csv",
                metadata_csv=output / "run_metadata.csv",
                voltage_v=2.0,
                baseline_specs=["initiator=0.02", "responder=0.04"],
                stage_overrides=[],
                marker_threshold=0.5,
                active_low=False,
                allow_unmatched=False,
                trace_padding_ms=0.0,
            )
            paper_rows = [
                row
                for row in metrics
                if row.get("measurement_scope") == "pair_union"
            ]
            self.assertEqual({"initiator", "responder", "pair"}, {row["role"] for row in paper_rows})
            self.assertTrue(all(int(row["truncated_start"]) == 1 for row in paper_rows))

    def test_baseline_metadata_binds_integrated_mean_to_event_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_dir = root / "baseline"
            event_dir = root / "event"
            baseline_power = root / "baseline.csv"
            baseline_power.write_text(
                "time_s,current_a,event_gate,stage_bit0,stage_bit1,stage_bit2\n"
                "0.0,0.010,0,0,0,0\n"
                "0.1,0.012,0,0,0,0\n"
                "0.2,0.010,0,0,0,0\n",
                encoding="utf-8",
            )
            analyze_power(
                input_specs=[f"initiator={baseline_power}"],
                output_dir=baseline_dir,
                events_csv=None,
                metadata_csv=None,
                voltage_v=2.0,
                baseline_specs=[],
                stage_overrides=[],
                marker_threshold=0.5,
                active_low=False,
                allow_unmatched=False,
                trace_padding_ms=0.0,
                baseline_only=True,
            )
            events = root / "events.csv"
            with events.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "run_id", "pair_id", "board_id", "role", "trial_id",
                        "event_index", "event_path", "connection_state", "warmup", "success",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "run_id": "run1", "pair_id": "pair1", "board_id": "board_a",
                        "role": "initiator", "trial_id": 0, "event_index": 0,
                        "event_path": "local", "connection_state": "connected",
                        "warmup": 0, "success": 1,
                    }
                )
            event_power = root / "event.csv"
            event_power.write_text(
                "time_s,current_a,event_gate,stage_bit0,stage_bit1,stage_bit2\n"
                "0.0,0.010,0,0,0,0\n"
                "0.1,0.050,1,1,0,0\n"
                "0.2,0.010,0,0,0,0\n",
                encoding="utf-8",
            )
            metrics, _, _ = analyze_power(
                input_specs=[f"initiator={event_power}"],
                output_dir=event_dir,
                events_csv=events,
                metadata_csv=None,
                voltage_v=2.0,
                baseline_specs=[],
                stage_overrides=[],
                marker_threshold=0.5,
                active_low=False,
                allow_unmatched=False,
                trace_padding_ms=0.0,
                baseline_metadata_csv=baseline_dir / "power_capture_metadata.csv",
            )
            self.assertEqual("baseline_metadata", metrics[0]["baseline_source"])
            self.assertTrue(math.isclose(float(metrics[0]["baseline_power_w"]), 0.022))
            self.assertEqual(
                hashlib.sha256(
                    (baseline_dir / "power_capture_metadata.csv").read_bytes()
                ).hexdigest(),
                metrics[0]["baseline_metadata_sha256"],
            )


class SummaryAndResourceTests(unittest.TestCase):
    def test_section_v_requires_every_accepted_power_role_and_emitted_stage(self) -> None:
        data: dict[str, list[dict[str, object]]] = {
            "protocol_audit": [],
            "latency": [],
            "power": [],
            "power_capture_metadata": [],
            "link": [],
            "resource_totals": [],
            "resource_provenance": [],
            "runtime_ram": [],
            "run_metadata": [],
            "correctness": [],
            "power_events": [],
            "trace": [],
        }
        with self.assertRaises(ValueError) as raised:
            _validate_section_v_coverage(data)
        message = str(raised.exception)
        self.assertIn(
            "power:accepted_connected/initiator/connected/pair_union", message
        )
        self.assertIn(
            "power:accepted_discovery/responder/connected/pair_union", message
        )
        self.assertIn("energy_total_j", message)
        self.assertIn(
            "stage:accepted_connected/initiator/connected/head_serialization",
            message,
        )
        self.assertIn(
            "stage:accepted_discovery/responder/connected/gradient_serialization",
            message,
        )
        self.assertIn("link_provenance:missing_link_audit", message)
        self.assertIn("link_provenance:missing_capture_metadata", message)
        self.assertIn("stage:local/local/connected/local_step_2_update", message)

    def test_capture_environment_requires_concrete_bound_lab_settings(self) -> None:
        complete = {
            "supply_voltage_v": "3.8",
            "mean_voltage_v": "3.8",
            "analyzer_sample_rate_hz": "100000",
            "nominal_sample_rate_hz": "100000",
            "analyzer_filter": "none",
            "device_separation_m": "0.5",
        }
        self.assertEqual([], _capture_environment_issues(complete, "capture"))
        incomplete = {**complete, "analyzer_filter": "external_required"}
        self.assertIn(
            "capture_environment:capture:missing_analyzer_filter",
            _capture_environment_issues(incomplete, "capture"),
        )
    def test_resource_provenance_hashes_and_binds_uart_constants_to_elf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "out"
            revision = "a" * 40
            artifact = "b" * 64
            elf = root / "zephyr.elf"
            elf.write_bytes(b"ELF-fixture\0" + revision.encode() + b"\0" + artifact.encode())
            baseline_elf = root / "baseline.elf"
            baseline_elf.write_bytes(b"ELF-baseline\0" + revision.encode())
            metadata = root / "run_metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "firmware_revision",
                        "build_variant",
                        "artifact_sha256",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "firmware_revision": revision,
                        "build_variant": "correctness",
                        "artifact_sha256": artifact,
                    }
                )
            baseline_metadata = root / "baseline_metadata.csv"
            with baseline_metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "firmware_revision",
                        "build_variant",
                        "artifact_sha256",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "firmware_revision": revision,
                        "build_variant": "matched_baseline",
                        "artifact_sha256": "none",
                    }
                )
            correctness = root / "correctness.csv"
            with correctness.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["firmware_revision", "artifact_sha256", "pass"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "firmware_revision": revision,
                        "artifact_sha256": artifact,
                        "pass": 1,
                    }
                )

            analyze_resources(
                map_specs=[
                    f"correctness={FIXTURES / 'sample_zephyr.map'}",
                    f"baseline={FIXTURES / 'sample_zephyr.map'}",
                ],
                nm_specs=[],
                runtime_paths=[],
                output_dir=output,
                category_rules=None,
                elf_specs=[f"correctness={elf}", f"baseline={baseline_elf}"],
                metadata_specs=[
                    f"correctness={metadata}",
                    f"baseline={baseline_metadata}",
                ],
                correctness_specs=[f"correctness={correctness}"],
            )
            provenance = read_rows(output / "resource_provenance.csv")
            self.assertEqual(2, len(provenance))
            by_image = {row["image"]: row for row in provenance}
            self.assertEqual(hashlib.sha256(elf.read_bytes()).hexdigest(), by_image["correctness"]["elf_sha256"])
            self.assertEqual(revision, by_image["correctness"]["firmware_revision"])
            self.assertEqual("1", by_image["correctness"]["firmware_revision_embedded_in_elf"])
            self.assertEqual("1", by_image["correctness"]["artifact_sha256_embedded_in_elf"])
            self.assertEqual(revision, by_image["baseline"]["firmware_revision"])
            self.assertEqual("", by_image["baseline"]["artifact_sha256"])
            self.assertEqual("1", by_image["baseline"]["firmware_revision_embedded_in_elf"])
            totals = read_rows(output / "resource_summary.csv")
            self.assertTrue(all(row["source_sha256"] for row in totals))

            elf.write_bytes(b"different ELF")
            with self.assertRaisesRegex(ValueError, "revision .* is not embedded"):
                analyze_resources(
                    map_specs=[],
                    nm_specs=[],
                    runtime_paths=[],
                    output_dir=output,
                    category_rules=None,
                    elf_specs=[f"correctness={elf}"],
                    metadata_specs=[f"correctness={metadata}"],
                    correctness_specs=[f"correctness={correctness}"],
                )

    def test_figure_ram_breakdown_uses_layout_metadata_without_changing_total(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            output = Path(temporary) / "figures"
            data.mkdir()
            with (data / "resource_breakdown.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["image", "memory", "category", "bytes", "source"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"image": "pulse", "memory": "static_ram", "category": "pulse_state", "bytes": 22000, "source": "linker_map"},
                        {"image": "pulse", "memory": "static_ram", "category": "training_tensors", "bytes": 51000, "source": "linker_map"},
                        {"image": "pulse", "memory": "static_ram", "category": "protocol_buffers", "bytes": 42000, "source": "linker_map"},
                    ]
                )
            with (data / "run_metadata.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                fields = [
                    "build_variant", "mutable_head_bytes", "compute_workspace_bytes",
                    "gradient_buffer_bytes_each", "gradient_buffer_count", "received_head_bytes",
                    "utility_table_bytes", "peer_entry_bytes", "peer_limit",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "build_variant": "pulse_release", "mutable_head_bytes": 18200,
                        "compute_workspace_bytes": 13616, "gradient_buffer_bytes_each": 18200,
                        "gradient_buffer_count": 2, "received_head_bytes": 18200,
                        "utility_table_bytes": 208, "peer_entry_bytes": 24, "peer_limit": 8,
                    }
                )

            result = prepare_figure_csvs([data], output)
            original_total = 22000 + 51000 + 42000
            self.assertEqual(original_total, sum(int(row["bytes"]) for row in result["resources"]))
            by_category = {row["category"]: row for row in result["resources"]}
            self.assertEqual(18200, by_category["mutable_head"]["bytes"])
            self.assertEqual(13616, by_category["compute_workspace"]["bytes"])
            self.assertEqual(18200, by_category["local_gradient"]["bytes"])
            self.assertEqual(18200, by_category["remote_gradient"]["bytes"])
            self.assertEqual(18200, by_category["tensor_wire_scratch"]["bytes"])
            self.assertEqual(208, by_category["peer_utility_table"]["bytes"])
            self.assertEqual(24, by_category["peer_utility_table"]["peer_entry_bytes"])
            with self.assertRaisesRegex(
                ValueError,
                "validated_runtime_ram:.*correctness:no_pass_1",
            ):
                prepare_figure_csvs([data], output, require_complete=True)

    def test_representative_trace_uses_full_session_identity(self) -> None:
        data: dict[str, list[dict[str, object]]] = {
            "power_events": [
                {
                    "run_id": "run_a", "pair_id": "pair_a", "exchange_id": "41",
                    "event_index": "7", "event_path": "accepted_connected", "role": "pair",
                    "warmup": "0", "success": "1", "truncated_start": "0",
                    "truncated_end": "0", "energy_incremental_j": "0.001",
                },
                {
                    "run_id": "run_b", "pair_id": "pair_b", "exchange_id": "41",
                    "event_index": "7", "event_path": "accepted_connected", "role": "pair",
                    "warmup": "0", "success": "1", "truncated_start": "0",
                    "truncated_end": "0", "energy_incremental_j": "0.002",
                },
            ],
            "trace": [],
        }
        self.assertEqual(
            {
                "run_id": "run_a",
                "pair_id": "pair_a",
                "exchange_id": "41",
                "event_index": "7",
                "event_path": "accepted_connected",
            },
            _representative_event(data),
        )

    def test_warmup_rows_are_excluded(self) -> None:
        rows = [
            {"event_path": "local", "role": "local", "warmup": "1", "success": "1", "duration_us": "10"},
            {"event_path": "local", "role": "local", "warmup": "0", "success": "1", "duration_us": "20"},
        ]
        summary = aggregate_rows(rows, ("event_path", "role"))
        duration = next(row for row in summary if row["metric"] == "duration_us")
        self.assertEqual(1, duration["total_count"])
        self.assertEqual(20.0, duration["median"])

    def test_summary_never_pools_distinct_build_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "event_metrics.csv"
            fields = [
                "build_context_sha256",
                "learning_rate",
                "event_path",
                "role",
                "connection_state",
                "remote_session_started",
                "warmup",
                "success",
                "duration_us",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "build_context_sha256": "a" * 64,
                            "learning_rate": "0.01",
                            "event_path": "local",
                            "role": "local",
                            "connection_state": "connected",
                            "remote_session_started": "0",
                            "warmup": "0",
                            "success": "1",
                            "duration_us": "10",
                        },
                        {
                            "build_context_sha256": "b" * 64,
                            "learning_rate": "0.02",
                            "event_path": "local",
                            "role": "local",
                            "connection_state": "connected",
                            "remote_session_started": "0",
                            "warmup": "0",
                            "success": "1",
                            "duration_us": "30",
                        },
                    ]
                )
            summarize_directory([root], root / "summary", min_repetitions=1)
            rows = [
                row
                for row in read_rows(root / "summary" / "event_summary.csv")
                if row["metric"] == "duration_us"
            ]
            self.assertEqual(2, len(rows))
            self.assertEqual({"10", "30"}, {row["median"] for row in rows})
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                {source_hash},
                {row["analysis_source_sha256_set"] for row in rows},
            )

    def test_failed_attempt_energy_is_retained_when_outcome_stratified(self) -> None:
        rows = [
            {
                "event_path": "accepted_discovery",
                "role": "initiator",
                "outcome": "failure",
                "warmup": "0",
                "success": "0",
                "energy_incremental_j": "0.003",
            }
        ]
        summary = aggregate_rows(
            rows,
            ("event_path", "role", "outcome"),
            include_failed_metrics=True,
        )
        energy = next(row for row in summary if row["metric"] == "energy_incremental_j")
        self.assertEqual(0.003, energy["median"])

    def test_stage_power_summary_does_not_pool_repeated_stage_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "stage_power_metrics.csv"
            fields = [
                "run_id", "pair_id", "board_id", "event_path", "role",
                "connection_state", "remote_session_started", "measurement_scope",
                "baseline_source", "baseline_method", "warmup", "success",
                "stage_sequence", "stage", "stage_code", "energy_incremental_j",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for sequence, energy in ((1, 0.001), (3, 0.003)):
                    writer.writerow(
                        {
                            "run_id": "run1", "pair_id": "pair1", "board_id": "board_a",
                            "event_path": "accepted_connected", "role": "initiator",
                            "connection_state": "connected", "remote_session_started": "1",
                            "measurement_scope": "own_gate_stage",
                            "baseline_source": "baseline_metadata",
                            "baseline_method": "matched_baseline_metadata_integrated_mean",
                            "warmup": "0", "success": "1", "stage_sequence": sequence,
                            "stage": "radio_transfer", "stage_code": 5,
                            "energy_incremental_j": energy,
                        }
                    )
            summarize_directory([root], root, min_repetitions=1)
            rows = [
                row
                for row in read_rows(root / "stage_power_summary.csv")
                if row["metric"] == "energy_incremental_j"
            ]
            self.assertEqual({"1", "3"}, {row["stage_sequence"] for row in rows})
            self.assertEqual({"0.001", "0.003"}, {row["median"] for row in rows})
            self.assertTrue((root / "stage_power_summary_by_pair.csv").exists())

    def test_summary_and_protocol_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            parse_serial_files([FIXTURES / "serial.log"], output)
            _, audit = summarize_directory([output], output, min_repetitions=1)
            self.assertTrue(audit)
            accepted = next(
                row
                for row in audit
                if row["event_path"] == "accepted_connected" and row["role"] == "initiator"
            )
            self.assertEqual("pass", accepted["repetition_check"])
            self.assertEqual(
                3,
                sum(
                    row.get("audit_type") == "stratum"
                    and row.get("stratum_check") == "missing"
                    for row in audit
                ),
            )
            summary = read_rows(output / "event_summary.csv")
            duration = [row for row in summary if row["metric"] == "duration_us"]
            self.assertEqual(2, len(duration))
            self.assertTrue(all(row["p95"] for row in duration))

    def test_required_stratum_absence_cannot_be_waived(self) -> None:
        rows = [
            {
                "run_id": "run1", "pair_id": "pair1", "board_id": "board_a", "role": "local",
                "event_path": "local", "connection_state": "connected",
                "warmup": "0", "success": "1",
            }
        ]
        audit = protocol_audit(rows, min_repetitions=1)
        missing = {
            (row["event_path"], row["role"], row["connection_state"])
            for row in audit
            if row.get("stratum_check") == "missing"
        }
        self.assertEqual(
            {
                ("no_contact", "local", "disconnected"),
                ("accepted_connected", "initiator", "connected"),
                ("accepted_discovery", "initiator", "disconnected"),
            },
            missing,
        )
        with self.assertRaisesRegex(ValueError, "cannot waive missing configurations"):
            _check_audit(audit, allow_underfilled=True)

    def test_underfilled_present_strata_can_be_waived(self) -> None:
        rows = [
            {
                "run_id": "run1", "pair_id": "pair1", "board_id": "board_a", "role": "local",
                "event_path": path, "connection_state": state,
                "warmup": "0", "success": "1",
            }
            for path, state in (("local", "connected"), ("no_contact", "disconnected"))
        ]
        for exchange_id, path, state in (
            (1, "accepted_connected", "connected"),
            (2, "accepted_discovery", "disconnected"),
        ):
            rows.extend(
                [
                    {
                        "run_id": "run1", "pair_id": "pair1", "board_id": "board_a", "role": "initiator",
                        "trial_id": "0", "exchange_id": str(exchange_id),
                        "remote_session_started": "1", "event_path": path,
                        "connection_state": state, "warmup": "0", "success": "1",
                    },
                    {
                        "run_id": "run1", "pair_id": "pair1", "board_id": "board_b", "role": "responder",
                        "trial_id": "0", "exchange_id": str(exchange_id),
                        "remote_session_started": "1",
                        "event_path": path, "connection_state": "connected",
                        "warmup": "0", "success": "1",
                    },
                ]
            )
        audit = protocol_audit(rows, min_repetitions=2)
        _check_audit(audit, allow_underfilled=True)

    def test_repetition_minimum_is_not_satisfied_by_pooling_pairs(self) -> None:
        rows: list[dict[str, str]] = []
        for pair_number in (1, 2):
            pair_id = f"pair{pair_number}"
            run_id = f"run{pair_number}"
            initiator_board = f"board_{pair_number}_a"
            responder_board = f"board_{pair_number}_b"
            rows.extend(
                [
                    {
                        "run_id": run_id, "pair_id": pair_id, "board_id": initiator_board,
                        "role": "local", "event_path": "local",
                        "connection_state": "connected", "warmup": "0", "success": "1",
                    },
                    {
                        "run_id": run_id, "pair_id": pair_id, "board_id": initiator_board,
                        "role": "local", "event_path": "no_contact",
                        "connection_state": "disconnected", "warmup": "0", "success": "1",
                    },
                ]
            )
            for exchange_id, path, state in (
                (pair_number * 10 + 1, "accepted_connected", "connected"),
                (pair_number * 10 + 2, "accepted_discovery", "disconnected"),
            ):
                rows.extend(
                    [
                        {
                            "run_id": run_id, "pair_id": pair_id,
                            "board_id": initiator_board, "role": "initiator",
                            "trial_id": "0", "exchange_id": str(exchange_id),
                            "remote_session_started": "1", "event_path": path,
                            "connection_state": state, "warmup": "0", "success": "1",
                        },
                        {
                            "run_id": run_id, "pair_id": pair_id,
                            "board_id": responder_board, "role": "responder",
                            "trial_id": "0", "exchange_id": str(exchange_id),
                            "remote_session_started": "1", "event_path": path,
                            "connection_state": "connected", "warmup": "0", "success": "1",
                        },
                    ]
                )
        audit = protocol_audit(rows, min_repetitions=2)
        pooled = [
            row
            for row in audit
            if row.get("audit_type") == "stratum" and row.get("required_stratum") == 1
        ]
        captures = [
            row
            for row in audit
            if row.get("audit_type") == "capture_stratum"
        ]
        self.assertTrue(all(row["stratum_check"] == "pass" for row in pooled))
        self.assertTrue(all(row["stratum_check"] == "underfilled" for row in captures))
        with self.assertRaisesRegex(ValueError, "repetition audit failed"):
            _check_audit(audit, allow_underfilled=False)

    def test_exact_exchange_pair_integrity_rejects_bad_records(self) -> None:
        base_initiator = {
            "run_id": "run1", "pair_id": "pair1", "board_id": "board_a", "role": "initiator",
            "trial_id": "4", "exchange_id": "44", "remote_session_started": "1",
            "event_path": "accepted_connected", "connection_state": "connected",
            "warmup": "0", "success": "1",
        }
        base_responder = {
            "run_id": "run1", "pair_id": "pair1", "board_id": "board_b", "role": "responder",
            "trial_id": "4", "exchange_id": "44", "remote_session_started": "1",
            "event_path": "accepted_connected",
            "connection_state": "connected", "warmup": "0", "success": "1",
        }

        initiators, responders, issues = accepted_session_integrity(
            [base_initiator, base_responder]
        )
        self.assertEqual(1, len(initiators))
        self.assertEqual(1, len(responders))
        self.assertFalse(issues)

        cases = {
            "duplicate": [base_initiator, base_responder, dict(base_responder)],
            "orphan": [base_responder],
            "missing_remote_label": [
                {key: value for key, value in base_initiator.items() if key != "remote_session_started"}
            ],
            "missing_key": [{**base_initiator, "exchange_id": ""}],
            "pre_session_with_responder": [
                {**base_initiator, "remote_session_started": "0", "success": "0"},
                base_responder,
            ],
            "impossible_physical_tuple": [
                {**base_initiator, "connection_state": "disconnected"},
                base_responder,
            ],
            "responder_remote_flag_zero": [
                base_initiator,
                {**base_responder, "remote_session_started": "0"},
            ],
        }
        for name, records in cases.items():
            with self.subTest(name=name):
                _, _, case_issues = accepted_session_integrity(records)
                self.assertTrue(case_issues)

        duplicate_audit = protocol_audit(cases["duplicate"], min_repetitions=1)
        with self.assertRaisesRegex(ValueError, "protocol integrity audit failed"):
            _check_audit(duplicate_audit, allow_underfilled=True)

    def test_per_node_daily_energy_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            power_summary = output / "power_summary.csv"
            write_energy_summary(power_summary)
            counts = output / "counts.csv"
            ledger_rows = complete_energy_ledger_rows()
            write_energy_ledger(counts, ledger_rows)
            contributions, totals = project_daily_energy(counts, power_summary, output)

            self.assertEqual(1, len(totals))
            self.assertEqual("7", totals[0]["seed"])
            self.assertEqual(14, len(contributions))
            expected_daily_events = 17 * 86400.0 / (126.6 * 3600.0)
            self.assertTrue(
                math.isclose(
                    float(totals[0]["expected_incremental_energy_j_per_day"]),
                    expected_daily_events * 0.0025,
                )
            )
            self.assertTrue(
                math.isclose(
                    float(
                        totals[0][
                            "incremental_energy_plugin_sensitivity_using_event_median_j_per_day"
                        ]
                    ),
                    expected_daily_events * 0.002,
                )
            )
            audit = read_rows(output / "daily_energy_audit.csv")
            self.assertTrue(audit)
            self.assertTrue(all(row["pass"] == "1" for row in audit))

    def test_daily_energy_projection_rejects_incomplete_or_duplicate_ledger(self) -> None:
        cases: dict[str, list[dict[str, object]]] = {}
        incomplete = complete_energy_ledger_rows()
        cases["missing explicit categories"] = incomplete[:-1]
        duplicate = complete_energy_ledger_rows()
        cases["duplicate event category"] = duplicate + [dict(duplicate[0])]

        for expected_error, rows in cases.items():
            with self.subTest(expected_error=expected_error), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                counts = output / "counts.csv"
                power_summary = output / "power_summary.csv"
                write_energy_ledger(counts, rows)
                write_energy_summary(power_summary)
                with self.assertRaisesRegex(ValueError, expected_error):
                    project_daily_energy(counts, power_summary, output)
                self.assertTrue((output / "daily_energy_audit.csv").exists())

    def test_daily_energy_projection_rejects_empty_or_wildcard_ledger(self) -> None:
        cases: dict[str, list[dict[str, object]]] = {
            "contains no event rows": [],
            "missing required fields": [
                {**row, "outcome": ""} if index == 0 else row
                for index, row in enumerate(complete_energy_ledger_rows())
            ],
        }
        for expected_error, rows in cases.items():
            with self.subTest(expected_error=expected_error), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                counts = output / "counts.csv"
                power_summary = output / "power_summary.csv"
                write_energy_ledger(counts, rows)
                write_energy_summary(power_summary)
                with self.assertRaisesRegex(ValueError, expected_error):
                    project_daily_energy(counts, power_summary, output)
                self.assertTrue((output / "daily_energy_audit.csv").exists())

    def test_daily_energy_projection_requires_exact_physical_strata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = complete_energy_ledger_rows()
            discovery_responder = next(
                row
                for row in rows
                if row["event_path"] == "accepted_discovery"
                and row["role"] == "responder"
                and row["outcome"] == "success"
            )
            discovery_responder["connection_state"] = "disconnected"
            counts = output / "counts.csv"
            power_summary = output / "power_summary.csv"
            write_energy_ledger(counts, rows)
            write_energy_summary(power_summary)
            with self.assertRaisesRegex(ValueError, "unsupported event_path/role/connection_state"):
                project_daily_energy(counts, power_summary, output)

    def test_daily_energy_projection_matches_remote_session_state_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            counts = output / "counts.csv"
            power_summary = output / "power_summary.csv"
            write_energy_ledger(counts, complete_energy_ledger_rows())
            write_energy_summary(power_summary)
            rows = read_rows(power_summary)
            target = next(
                row
                for row in rows
                if row["event_path"] == "accepted_connected"
                and row["role"] == "initiator"
            )
            target["remote_session_started"] = "0"
            with power_summary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(
                ValueError, "remote_session_started=1.*matches=0|matches=0.*remote_session_started=1"
            ):
                project_daily_energy(counts, power_summary, output)

    def test_daily_energy_projection_reconciles_opportunities_and_sessions(self) -> None:
        cases = {
            "selection opportunity count": ("selection_opportunities", 6),
            "responder count": ("count_in_observation", 0),
        }
        for expected_error, (field, value) in cases.items():
            with self.subTest(expected_error=expected_error), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                rows = complete_energy_ledger_rows()
                if field == "selection_opportunities":
                    for row in rows:
                        row[field] = value
                else:
                    responder_success = next(
                        row
                        for row in rows
                        if row["role"] == "responder" and row["outcome"] == "success"
                    )
                    responder_success[field] = value
                counts = output / "counts.csv"
                power_summary = output / "power_summary.csv"
                write_energy_ledger(counts, rows)
                write_energy_summary(power_summary)
                with self.assertRaisesRegex(ValueError, expected_error):
                    project_daily_energy(counts, power_summary, output)

    def test_daily_energy_projection_reconciles_each_accepted_path_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = complete_energy_ledger_rows()
            for row in rows:
                if row["role"] != "responder" or row["outcome"] != "success":
                    continue
                if row["event_path"] == "accepted_connected":
                    row["count_in_observation"] = 0
                elif row["event_path"] == "accepted_discovery":
                    row["count_in_observation"] = 2
            counts = output / "counts.csv"
            power_summary = output / "power_summary.csv"
            write_energy_ledger(counts, rows)
            write_energy_summary(power_summary)
            with self.assertRaisesRegex(ValueError, "accepted_connected.*responder count"):
                project_daily_energy(counts, power_summary, output)

    def test_daily_energy_projection_requires_complete_power_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            counts = output / "counts.csv"
            power_summary = output / "power_summary.csv"
            write_energy_ledger(counts, complete_energy_ledger_rows())
            write_energy_summary(power_summary, omit_p95=True)
            with self.assertRaisesRegex(ValueError, "missing p95"):
                project_daily_energy(counts, power_summary, output)

    def test_daily_energy_projection_verifies_count_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            counts = output / "counts.csv"
            power_summary = output / "power_summary.csv"
            write_energy_ledger(counts, complete_energy_ledger_rows())
            write_energy_summary(power_summary)
            (output / "complete-count-manifest.json").write_text(
                '{"fixture":"tampered"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "source_artifact_sha256 does not match"):
                project_daily_energy(counts, power_summary, output)

    def test_map_and_nm_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summaries, breakdown, runtime = analyze_resources(
                map_specs=[f"pulse={FIXTURES / 'sample.map'}"],
                nm_specs=[f"pulse={FIXTURES / 'sample.nm'}"],
                runtime_paths=[],
                output_dir=output,
                category_rules=None,
            )
            self.assertFalse(runtime)
            self.assertEqual({"flash", "static_ram"}, {row["memory"] for row in summaries})
            self.assertIn("encoder", {row["category"] for row in breakdown})
            symbols = read_rows(output / "symbol_sizes.csv")
            self.assertEqual(3, len(symbols))

    def test_real_zephyr_map_shape_uses_usage_symbols_and_ignores_debug(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summaries, breakdown, _ = analyze_resources(
                map_specs=[f"pulse={FIXTURES / 'sample_zephyr.map'}"],
                nm_specs=[],
                runtime_paths=[],
                output_dir=output,
                category_rules=None,
            )
            by_memory = {row["memory"]: row for row in summaries}
            self.assertEqual(0x64, by_memory["flash"]["used_bytes"])
            self.assertEqual(0x40, by_memory["static_ram"]["used_bytes"])
            self.assertEqual("zephyr_linker_usage_symbol", by_memory["flash"]["measurement"])
            self.assertLessEqual(by_memory["flash"]["used_bytes"], by_memory["flash"]["capacity_bytes"])
            details = read_rows(output / "map_contributions.csv")
            self.assertFalse(any(row["section"].startswith(".debug") for row in details))
            pulse_flash = sum(
                int(row["bytes"])
                for row in breakdown
                if row["memory"] == "flash" and row["category"] == "pulse_state"
            )
            self.assertEqual(0x10, pulse_flash)

    def test_firmware_runtime_watermark_aliases_do_not_double_count_static_ram(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runtime_input = output / "event_metrics.csv"
            with runtime_input.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "event_path",
                        "role",
                        "thread_stack_peak_bytes",
                        "system_heap_peak_bytes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "event_path": "local",
                        "role": "local",
                        "thread_stack_peak_bytes": 20,
                        "system_heap_peak_bytes": 12,
                    }
                )

            summaries, _, runtime = analyze_resources(
                map_specs=[f"pulse={FIXTURES / 'sample_zephyr.map'}"],
                nm_specs=[],
                runtime_paths=[runtime_input],
                output_dir=output,
                category_rules=None,
            )
            static_ram = next(
                row["used_bytes"] for row in summaries if row["memory"] == "static_ram"
            )
            self.assertEqual(1, len(runtime))
            row = runtime[0]
            self.assertEqual(20.0, row["peak_stack_bytes"])
            self.assertEqual(12.0, row["peak_heap_bytes"])
            self.assertEqual(32.0, row["stack_heap_watermark_upper_bound_bytes"])
            self.assertEqual(static_ram, row["peak_ram_bytes"])
            self.assertEqual(
                "linker_reserved_ram_includes_stack_heap", row["calculation_method"]
            )

    def test_explicit_nonoverlapping_dynamic_is_added_but_pool_watermarks_are_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runtime_input = output / "runtime.csv"
            with runtime_input.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "static_ram_bytes",
                        "thread_stack_peak_bytes",
                        "system_heap_peak_bytes",
                        "additional_dynamic_bytes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "static_ram_bytes": 1000,
                        "thread_stack_peak_bytes": 100,
                        "system_heap_peak_bytes": 200,
                        "additional_dynamic_bytes": 30,
                    }
                )

            _, _, runtime = analyze_resources(
                map_specs=[],
                nm_specs=[],
                runtime_paths=[runtime_input],
                output_dir=output,
                category_rules=None,
            )
            row = runtime[0]
            self.assertEqual(1030.0, row["peak_ram_bytes"])
            self.assertEqual(300.0, row["stack_heap_watermark_upper_bound_bytes"])
            self.assertEqual(
                "static_plus_explicit_nonoverlapping_dynamic", row["calculation_method"]
            )

    def test_nm_only_is_labelled_as_lower_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summaries, breakdown, _ = analyze_resources(
                map_specs=[],
                nm_specs=[f"pulse={FIXTURES / 'sample.nm'}"],
                runtime_paths=[],
                output_dir=output,
                category_rules=None,
            )
            self.assertTrue(summaries)
            self.assertTrue(
                all(row["measurement"] == "sum_nm_symbols_lower_bound" for row in summaries)
            )
            self.assertTrue(all(row["source"] == "nm_lower_bound" for row in breakdown))


if __name__ == "__main__":
    unittest.main()
