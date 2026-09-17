from __future__ import annotations

import csv
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

from battery_projection import (  # noqa: E402
    INPUT_SCHEMA,
    main as battery_projection_main,
    project_battery_life,
)
from pulse_tools import main as pulse_tools_main  # noqa: E402


def _assumption(
    description: str,
    *,
    idle_power_w: float,
    active_power_w: float,
    local_count: float,
    exchange_count: float,
) -> dict[str, object]:
    return {
        "description": description,
        "battery": {
            "capacity_mAh": 100.0,
            "nominal_voltage_v": 3.7,
            "efficiency": 0.8,
        },
        "states": {
            "idle": {
                "duration_s_per_day": 82_800.0,
                "idle_power_w": idle_power_w,
            },
            "active_baseline": {
                "duration_s_per_day": 3_600.0,
                "idle_power_w": active_power_w,
            },
        },
        "event_counts_per_day": {
            "local": local_count,
            "exchange": exchange_count,
        },
    }


def valid_document() -> dict[str, object]:
    return {
        "schema": INPUT_SCHEMA,
        "incremental_event_energy_samples_j": {
            "local": [1.0, 1.0, 10.0],
            "exchange": [2.0, 4.0],
        },
        "assumptions": {
            "low": _assumption(
                "Lower daily activity and state power",
                idle_power_w=0.001,
                active_power_w=0.002,
                local_count=2.0,
                exchange_count=1.0,
            ),
            "nominal": _assumption(
                "Nominal measured-use assumptions",
                idle_power_w=0.0012,
                active_power_w=0.0024,
                local_count=3.0,
                exchange_count=2.0,
            ),
            "high": _assumption(
                "Higher daily activity and state power",
                idle_power_w=0.0015,
                active_power_w=0.003,
                local_count=5.0,
                exchange_count=4.0,
            ),
        },
    }


class BatteryProjectionTests(unittest.TestCase):
    def test_exact_equations_use_arithmetic_event_mean(self) -> None:
        report = project_battery_life(valid_document())
        by_name = {row["assumption"]: row for row in report["assumptions"]}
        low = by_name["low"]

        # State energy is 82,800*0.001 + 3,600*0.002 = 90 J/day.
        self.assertAlmostEqual(low["state_energy_j_per_day"], 90.0)
        # Arithmetic means are 4 J and 3 J, despite the local median being 1 J.
        self.assertAlmostEqual(low["incremental_event_energy_j_per_day"], 11.0)
        self.assertAlmostEqual(low["total_energy_j_per_day"], 101.0)

        expected_battery_j = 3.6 * 100.0 * 3.7 * 0.8
        self.assertAlmostEqual(low["battery_energy_j"], expected_battery_j)
        self.assertAlmostEqual(low["battery_life_days"], expected_battery_j / 101.0)

        statistics_by_event = {
            row["event"]: row for row in report["event_energy_statistics"]
        }
        self.assertAlmostEqual(
            statistics_by_event["local"]["arithmetic_mean_incremental_energy_j"],
            4.0,
        )
        self.assertAlmostEqual(
            statistics_by_event["exchange"]["arithmetic_mean_incremental_energy_j"],
            3.0,
        )

    def test_sampling_standard_error_is_separate_from_sensitivity(self) -> None:
        report = project_battery_life(valid_document())
        low = next(row for row in report["assumptions"] if row["assumption"] == "low")

        # local SEM=3 J and exchange SEM=1 J; counts are 2 and 1.
        expected_daily_standard_error = math.sqrt((2.0 * 3.0) ** 2 + (1.0 * 1.0) ** 2)
        self.assertAlmostEqual(
            low["event_sampling_standard_error_j_per_day"],
            expected_daily_standard_error,
        )
        expected_life_standard_error = (
            low["battery_energy_j"]
            * expected_daily_standard_error
            / low["total_energy_j_per_day"] ** 2
        )
        self.assertAlmostEqual(
            low["battery_life_sampling_standard_error_days"],
            expected_life_standard_error,
        )
        self.assertIn("not a confidence", report["semantics"]["sensitivity"])
        self.assertIn(
            "scenario sensitivity only",
            report["sensitivity_summary"]["semantics"],
        )
        self.assertNotEqual(
            report["sensitivity_summary"]["span_days"],
            low["battery_life_sampling_standard_error_days"],
        )

    def test_single_event_sample_marks_sampling_uncertainty_unavailable(self) -> None:
        document = valid_document()
        document["incremental_event_energy_samples_j"]["exchange"] = [3.0]
        report = project_battery_life(document)
        for row in report["assumptions"]:
            self.assertIsNone(row["event_sampling_standard_error_j_per_day"])
            self.assertIsNone(row["battery_life_sampling_standard_error_days"])

    def test_requires_all_three_explicit_assumptions(self) -> None:
        document = valid_document()
        del document["assumptions"]["high"]
        with self.assertRaisesRegex(ValueError, "exactly the explicit low, nominal, and high"):
            project_battery_life(document)

    def test_requires_explicit_battery_inputs_without_defaults(self) -> None:
        for field in ("capacity_mAh", "nominal_voltage_v", "efficiency"):
            with self.subTest(field=field):
                document = valid_document()
                del document["assumptions"]["nominal"]["battery"][field]
                with self.assertRaisesRegex(ValueError, f"requires explicit {field}"):
                    project_battery_life(document)

        document = valid_document()
        document["assumptions"]["nominal"]["battery"]["efficiency"] = 1.01
        with self.assertRaisesRegex(ValueError, "efficiency must be at most 1"):
            project_battery_life(document)

    def test_requires_a_complete_day_and_complete_event_ledger(self) -> None:
        document = valid_document()
        document["assumptions"]["low"]["states"]["idle"]["duration_s_per_day"] -= 1.0
        with self.assertRaisesRegex(ValueError, "sum to exactly 86400"):
            project_battery_life(document)

        document = valid_document()
        del document["assumptions"]["high"]["event_counts_per_day"]["exchange"]
        with self.assertRaisesRegex(ValueError, "exactly the measured event-energy classes"):
            project_battery_life(document)

    def test_cli_writes_json_and_csv_with_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.json"
            output_json = root / "projection.json"
            output_csv = root / "projection.csv"
            input_path.write_text(json.dumps(valid_document()), encoding="utf-8")

            self.assertEqual(
                battery_projection_main(
                    [
                        "--input-json",
                        str(input_path),
                        "--output-json",
                        str(output_json),
                        "--output-csv",
                        str(output_csv),
                    ]
                ),
                0,
            )

            parsed = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(parsed["schema"], "senswear.pulse.battery_projection.result.v1")
            self.assertEqual(len(parsed["assumptions"]), 3)

            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["assumption"] for row in rows], ["low", "nominal", "high"])
            self.assertTrue(all("not a confidence" in row["sensitivity_semantics"] for row in rows))

    def test_pulse_tools_cli_subcommand_writes_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.json"
            output_json = root / "projection.json"
            output_csv = root / "projection.csv"
            input_path.write_text(json.dumps(valid_document()), encoding="utf-8")

            result = pulse_tools_main(
                [
                    "project-battery-life",
                    "--input-json",
                    str(input_path),
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue(output_json.is_file())
            self.assertTrue(output_csv.is_file())
            parsed = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(
                [row["assumption"] for row in parsed["assumptions"]],
                ["low", "nominal", "high"],
            )

    def test_pulse_tools_cli_does_not_default_missing_efficiency(self) -> None:
        document = valid_document()
        del document["assumptions"]["nominal"]["battery"]["efficiency"]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.json"
            output_json = root / "projection.json"
            input_path.write_text(json.dumps(document), encoding="utf-8")
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                result = pulse_tools_main(
                    [
                        "project-battery-life",
                        "--input-json",
                        str(input_path),
                        "--output-json",
                        str(output_json),
                    ]
                )

            self.assertEqual(result, 2)
            self.assertIn("requires explicit efficiency", stderr.getvalue())
            self.assertFalse(output_json.exists())


if __name__ == "__main__":
    unittest.main()
