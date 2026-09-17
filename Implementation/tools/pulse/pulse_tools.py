#!/usr/bin/env python3
"""Command-line entry point for the SensWear PULSE evaluation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from battery_projection import run_battery_projection
from energy_projection import project_daily_energy
from communication_analysis import analyze_communication, require_valid_communication_audit
from figures import prepare_figure_csvs, render_figures
from keysight_34465a import (
    add_analysis_arguments as add_keysight_analysis_arguments,
    add_capture_arguments as add_keysight_capture_arguments,
    run_analysis as run_keysight_analysis,
    run_capture as run_keysight_capture,
)
from link_analysis import analyze_link, require_valid_link_audit
from power_analysis import analyze_power
from ngmo2_capture import (
    add_live_capture_arguments,
    analyze_envelope_captures,
    require_valid_ngmo2_audit,
    run_live_capture,
)
from ngmo2_autonomous import (
    analyze_autonomous_captures,
    require_valid_autonomous_audit,
)
from resource_analysis import analyze_resources
from serial_parser import parse_serial_files
from summarize import summarize_directory


def _path(value: str) -> Path:
    return Path(value)


def _context_values(specs: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid context {spec!r}; expected KEY=VALUE")
        key, value = spec.split("=", 1)
        key = key.strip().lower().replace("-", "_")
        if not key or not value:
            raise ValueError(f"invalid context {spec!r}; expected non-empty KEY=VALUE")
        values[key] = value
    return values


def _add_power_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--input",
        dest="power_inputs",
        action="append",
        required=required,
        default=[],
        metavar="[ROLE=]CSV",
        help=(
            "power-analyzer CSV; repeat for channels. Use initiator=file.csv and "
            "responder=file.csv for generic current_a columns, or pass one combined "
            "CSV with role-prefixed columns"
        ),
    )
    parser.add_argument("--events-csv", type=_path, help="parsed event_metrics.csv used to assign trial/path")
    parser.add_argument("--metadata-csv", type=_path, help="parsed run_metadata.csv containing marker_stage_N names")
    parser.add_argument("--voltage-v", type=float, help="fixed analyzer supply voltage when CSV has no voltage_v")
    parser.add_argument(
        "--baseline-power-w",
        action="append",
        default=[],
        metavar="[ROLE=]W",
        help=(
            "manual matched-baseline power; repeat per role. Prefer --baseline-metadata so "
            "the value is cryptographically bound to an archived baseline-only analysis"
        ),
    )
    parser.add_argument(
        "--baseline-metadata",
        type=_path,
        help=(
            "power_capture_metadata.csv from a separate --baseline-only analysis; loads the "
            "integrated-mean baseline per role and records its SHA-256 provenance"
        ),
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help=(
            "analyze a deliberate all-gates-low matched-baseline capture; emit capture/baseline "
            "metadata without requiring event windows"
        ),
    )
    parser.add_argument(
        "--allow-gate-low-baseline-diagnostic",
        action="store_true",
        help=(
            "diagnostic only: estimate baseline from gate-low samples in the event trace; "
            "paper-facing non-baseline analysis requires --baseline-metadata (preferred) or "
            "--baseline-power-w from a separate matched --baseline-only capture"
        ),
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="CODE=NAME",
        help="override a 3-bit stage name (metadata overrides defaults; CLI overrides metadata)",
    )
    parser.add_argument("--marker-threshold", type=float, default=0.5)
    parser.add_argument(
        "--trace-padding-ms",
        type=float,
        default=5.0,
        help="gate-low context retained before/after each sample-level trace (default: 5 ms)",
    )
    parser.add_argument("--active-low", action="store_true", help="decode all four GPIO markers as active-low")
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help=(
            "retain wholly unlabelled GPIO windows for diagnostics; never permits ambiguous "
            "partial chronological joins or non-contiguous serial indices"
        ),
    )


def _add_resource_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--map",
        dest="map_specs",
        action="append",
        default=[],
        metavar="[IMAGE=]FILE",
        help="Zephyr/GNU linker map; use pulse=... and baseline=... to calculate the delta",
    )
    parser.add_argument(
        "--nm",
        dest="nm_specs",
        action="append",
        default=[],
        metavar="[IMAGE=]FILE",
        help="saved `nm --print-size --size-sort --radix=x` output",
    )
    parser.add_argument(
        "--runtime-watermarks",
        action="append",
        type=_path,
        default=[],
        metavar="CSV",
        help="runtime peak RAM/stack/heap measurements; repeat as needed",
    )
    parser.add_argument(
        "--elf",
        dest="elf_specs",
        action="append",
        default=[],
        metavar="[IMAGE=]FILE",
        help=(
            "exact ELF paired with a resource image; its SHA-256 is recorded and UART "
            "revision/artifact constants are verified when --image-metadata is supplied"
        ),
    )
    parser.add_argument(
        "--image-metadata",
        dest="image_metadata_specs",
        action="append",
        default=[],
        metavar="[IMAGE=]RUN_METADATA_CSV",
        help="parsed UART run_metadata.csv to bind firmware revision/artifact to an ELF",
    )
    parser.add_argument(
        "--correctness-input",
        dest="correctness_specs",
        action="append",
        default=[],
        metavar="[IMAGE=]CORRECTNESS_CSV",
        help="exact correctness.csv to hash and bind to its metadata revision and ELF",
    )
    parser.add_argument(
        "--category-rules",
        type=_path,
        help="optional JSON mapping resource category names to regex lists",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse, validate, aggregate, and plot real SensWear PULSE measurements. "
            "The tools never synthesize missing hardware results."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serial = subparsers.add_parser("serial", help="parse PULSE_* firmware records")
    serial.add_argument("logs", nargs="+", type=_path)
    serial.add_argument("--output-dir", type=_path, required=True)
    serial.add_argument(
        "--set",
        dest="context_specs",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="attach capture context absent from firmware records, e.g. pair_id=p01",
    )

    power = subparsers.add_parser("power", help="decode GPIO markers and integrate power")
    _add_power_arguments(power, required=True)
    power.add_argument("--output-dir", type=_path, required=True)

    ngmo_capture = subparsers.add_parser(
        "ngmo2-capture",
        help=(
            "capture NGMO2 arrays autonomously (primary), static idle, or with "
            "optional UART pacing"
        ),
    )
    add_live_capture_arguments(ngmo_capture)

    ngmo_power = subparsers.add_parser(
        "ngmo2-power",
        help=(
            "integrate optional UART-paced NGMO2 envelopes against their quiet guards"
        ),
    )
    ngmo_power.add_argument("--manifest", type=_path, required=True)
    ngmo_power.add_argument("--events-csv", type=_path, required=True)
    ngmo_power.add_argument("--metadata-csv", type=_path)
    ngmo_power.add_argument("--output-dir", type=_path, required=True)
    ngmo_power.add_argument("--max-guard-drift-fraction", type=float, default=0.10)

    keysight_capture = subparsers.add_parser(
        "keysight-capture",
        help="capture one PULSE role sequentially with a Keysight 34465A",
    )
    add_keysight_capture_arguments(keysight_capture)

    keysight_figure = subparsers.add_parser(
        "keysight-fig4a",
        help="audit sequential Keysight role captures and render Figure 4a",
    )
    add_keysight_analysis_arguments(keysight_figure)

    autonomous_power = subparsers.add_parser(
        "ngmo2-autonomous-power",
        help=(
            "join autonomous NGMO2 captures to recovered NVS proofs and integrate "
            "the full synchronized capture"
        ),
    )
    autonomous_power.add_argument("--manifest", type=_path, required=True)
    autonomous_power.add_argument("--proofs-csv", type=_path, required=True)
    autonomous_power.add_argument("--capture-status-csv", type=_path, required=True)
    autonomous_power.add_argument(
        "--build-context-csv",
        type=_path,
        required=True,
        help=(
            "archived run_metadata.csv from both measured boards; its release "
            "contract is hash-checked and bound to every power row"
        ),
    )
    autonomous_power.add_argument("--output-dir", type=_path, required=True)
    autonomous_power.add_argument(
        "--max-guard-drift-fraction", type=float, default=0.10
    )

    communication = subparsers.add_parser(
        "communication",
        help="audit firmware ATT-value counters and create sender-counted byte metrics",
    )
    communication.add_argument("--events-csv", type=_path, required=True)
    communication.add_argument("--output-dir", type=_path, required=True)

    link = subparsers.add_parser(
        "link", help="validate canonical BLE-sniffer PDUs and account link-layer bytes"
    )
    link.add_argument(
        "--input",
        dest="link_inputs",
        action="append",
        type=_path,
        required=True,
        metavar="CSV",
        help="canonical PDU CSV; repeat for independently synchronized captures",
    )
    link.add_argument(
        "--event-windows-csv",
        type=_path,
        required=True,
        help="event_power_metrics.csv providing canonical GPIO-derived event windows",
    )
    link.add_argument(
        "--metadata-csv",
        type=_path,
        required=True,
        help="run_metadata.csv providing complementary logical-role BLE identities",
    )
    link.add_argument(
        "--raw-pcap",
        action="append",
        required=True,
        metavar="CAPTURE_ID=PATH",
        help="untouched PCAP/PCAPNG provenance to hash; one exact file per capture_id",
    )
    synchronization = link.add_mutually_exclusive_group(required=True)
    synchronization.add_argument(
        "--shared-timebase",
        action="store_true",
        help="packet timestamps already use the event/power analyzer clock",
    )
    synchronization.add_argument(
        "--sync-anchors",
        type=_path,
        help="CSV with >=2 bracketing affine clock anchors per capture_id",
    )
    link.add_argument(
        "--time-uncertainty-s",
        type=float,
        help="explicit worst-case timestamp uncertainty for --shared-timebase",
    )
    link.add_argument("--output-dir", type=_path, required=True)

    summarize = subparsers.add_parser("summarize", help="calculate median/IQR/p95 and failure rates")
    summarize.add_argument("--input-dir", action="append", type=_path, required=True)
    summarize.add_argument("--output-dir", type=_path, required=True)
    summarize.add_argument("--min-repetitions", type=int, default=100)
    summarize.add_argument(
        "--allow-underfilled",
        action="store_true",
        help="do not fail when a post-warm-up event stratum has fewer than the required trials",
    )

    resources = subparsers.add_parser("resources", help="summarize map/nm and runtime RAM measurements")
    _add_resource_arguments(resources)
    resources.add_argument("--output-dir", type=_path, required=True)

    projection = subparsers.add_parser(
        "project-energy",
        help="audit a complete per-node event ledger and combine it with measured event energy",
    )
    projection.add_argument(
        "--counts",
        type=_path,
        required=True,
        help=(
            "complete per-node ledger with observation duration, explicit zero categories, "
            "opportunity totals, and source provenance"
        ),
    )
    projection.add_argument("--power-summary", type=_path, required=True)
    projection.add_argument("--output-dir", type=_path, required=True)

    battery_projection = subparsers.add_parser(
        "project-battery-life",
        help=(
            "project battery life from explicit low/nominal/high state, battery, "
            "and measured event-energy assumptions"
        ),
    )
    battery_projection.add_argument(
        "--input-json",
        type=_path,
        required=True,
        help=(
            "complete projection input; capacity, voltage, efficiency, state durations/powers, "
            "event counts, and event-energy samples have no defaults"
        ),
    )
    battery_projection.add_argument("--output-json", type=_path, required=True)
    battery_projection.add_argument("--output-csv", type=_path)

    figures = subparsers.add_parser("figures", help="write figure tables and optional PDF/PNG")
    figures.add_argument("--data-dir", action="append", type=_path, required=True)
    figures.add_argument("--output-dir", type=_path, required=True)
    figures.add_argument(
        "--profile",
        choices=("ngmo2", "boards-only", "instrumented"),
        default="ngmo2",
        help="evidence profile (default: ngmo2 power plus firmware communication counters)",
    )
    figures.add_argument(
        "--formats",
        default="pdf,png",
        help="comma-separated pdf/png, or none to produce only figure-ready CSVs",
    )

    pipeline = subparsers.add_parser("all", help="run the available stages into one output directory")
    pipeline.add_argument("--serial-log", action="append", type=_path, required=True)
    pipeline.add_argument(
        "--set",
        dest="context_specs",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="attach capture context absent from firmware records",
    )
    _add_power_arguments(pipeline, required=False)
    pipeline.add_argument(
        "--profile",
        choices=("ngmo2", "boards-only", "instrumented"),
        default="ngmo2",
    )
    pipeline.add_argument(
        "--ngmo2-manifest",
        type=_path,
        help="ngmo2_capture_manifest.csv produced by ngmo2-capture",
    )
    pipeline.add_argument(
        "--ngmo2-proofs-csv",
        type=_path,
        help="capture_proofs.csv recovered after autonomous power measurements",
    )
    pipeline.add_argument(
        "--ngmo2-capture-status-csv",
        type=_path,
        help="capture_status.csv proving both autonomous NVS exports completed",
    )
    pipeline.add_argument(
        "--ngmo2-build-context-csv",
        type=_path,
        help="archived two-board run_metadata.csv for autonomous power provenance",
    )
    pipeline.add_argument("--max-guard-drift-fraction", type=float, default=0.10)
    _add_resource_arguments(pipeline)
    pipeline.add_argument("--output-dir", type=_path, required=True)
    pipeline.add_argument("--min-repetitions", type=int, default=100)
    pipeline.add_argument("--allow-underfilled", action="store_true")
    pipeline.add_argument(
        "--event-counts",
        type=_path,
        help=(
            "optional complete, auditable per-node event ledger for the incremental "
            "daily-energy projection"
        ),
    )
    pipeline.add_argument("--formats", default="pdf,png")
    return parser


def _run_power(args: argparse.Namespace, output_dir: Path, defaults: bool = False) -> None:
    events_csv = args.events_csv
    metadata_csv = args.metadata_csv
    if defaults:
        events_csv = events_csv or output_dir / "event_metrics.csv"
        metadata_csv = metadata_csv or output_dir / "run_metadata.csv"
    analyze_power(
        input_specs=args.power_inputs,
        output_dir=output_dir,
        events_csv=events_csv,
        metadata_csv=metadata_csv,
        voltage_v=args.voltage_v,
        baseline_specs=args.baseline_power_w,
        stage_overrides=args.stage,
        marker_threshold=args.marker_threshold,
        active_low=args.active_low,
        allow_unmatched=args.allow_unmatched,
        trace_padding_ms=args.trace_padding_ms,
        baseline_only=args.baseline_only,
        allow_gate_low_baseline_diagnostic=args.allow_gate_low_baseline_diagnostic,
        baseline_metadata_csv=args.baseline_metadata,
    )


def _check_audit(audit: list[dict[str, object]], allow_underfilled: bool) -> None:
    if not audit:
        raise ValueError("protocol audit found no post-warm-up PULSE_EVENT records")
    missing_success = [row for row in audit if not int(row.get("success_label_present", 0))]
    missing_warmup = [row for row in audit if not int(row.get("warmup_label_present", 0))]
    missing_pair = [row for row in audit if not int(row.get("pair_id_present", 0))]
    missing_run = [row for row in audit if not int(row.get("run_id_present", 0))]
    missing_board = [row for row in audit if not int(row.get("board_id_present", 0))]
    if missing_success:
        raise ValueError(
            "protocol audit found PULSE_EVENT rows without explicit success/pass labels; "
            "failure probability would be undefined"
        )
    if missing_warmup:
        raise ValueError(
            "protocol audit found PULSE_EVENT rows without warmup/post_warmup labels; "
            "post-warm-up repetitions cannot be verified"
        )
    if missing_pair:
        raise ValueError(
            "protocol audit found a missing/placeholder pair_id; set the physical pair ID in "
            "firmware metadata or parse with --set pair_id=..."
        )
    if missing_run:
        raise ValueError(
            "protocol audit found a missing/placeholder run_id; every physical capture "
            "must retain its unique run identity"
        )
    if missing_board:
        raise ValueError(
            "protocol audit found a missing/placeholder board_id; per-board repetition "
            "coverage cannot be verified"
        )
    integrity_failures = [row for row in audit if row.get("integrity_check") == "fail"]
    if integrity_failures:
        details = " | ".join(
            str(row.get("integrity_issues", "")) for row in integrity_failures
        )
        raise ValueError(
            "protocol integrity audit failed; --allow-underfilled cannot waive missing/duplicate/"
            "orphan session records: " + details
        )
    missing_required = [
        row
        for row in audit
        if int(row.get("required_stratum", 0)) and row.get("stratum_check") == "missing"
    ]
    if missing_required:
        descriptions = ", ".join(
            f"{row.get('event_path')}/{row.get('role')}/{row.get('connection_state')}"
            for row in missing_required
        )
        raise ValueError(
            "required post-warm-up event strata are absent; --allow-underfilled cannot waive "
            "missing configurations: " + descriptions
        )
    underfilled = [
        row
        for row in audit
        if int(row.get("required_stratum", 0)) and row.get("stratum_check") == "underfilled"
    ]
    if underfilled and not allow_underfilled:
        descriptions = ", ".join(
            f"{row.get('event_path')}/{row.get('role')}/{row.get('connection_state')}: "
            f"{row.get('post_warmup_repetitions')}"
            for row in underfilled
        )
        raise ValueError(
            "post-warm-up repetition audit failed (required minimum is recorded in protocol_audit.csv): "
            + descriptions
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "serial":
            parse_serial_files(args.logs, args.output_dir, _context_values(args.context_specs))
        elif args.command == "power":
            _run_power(args, args.output_dir)
        elif args.command == "ngmo2-capture":
            return run_live_capture(args)
        elif args.command == "ngmo2-power":
            _, _, audit = analyze_envelope_captures(
                args.manifest,
                args.events_csv,
                args.metadata_csv,
                args.output_dir,
                max_guard_drift_fraction=args.max_guard_drift_fraction,
            )
            require_valid_ngmo2_audit(audit)
        elif args.command == "keysight-capture":
            return run_keysight_capture(args)
        elif args.command == "keysight-fig4a":
            return run_keysight_analysis(args)
        elif args.command == "ngmo2-autonomous-power":
            _, _, audit, _ = analyze_autonomous_captures(
                args.manifest,
                args.proofs_csv,
                args.capture_status_csv,
                args.output_dir,
                build_context_csv=args.build_context_csv,
                max_guard_drift_fraction=args.max_guard_drift_fraction,
            )
            require_valid_autonomous_audit(audit)
        elif args.command == "communication":
            _, audit = analyze_communication(args.events_csv, args.output_dir)
            require_valid_communication_audit(audit)
        elif args.command == "link":
            _, _, _, audit = analyze_link(
                args.link_inputs,
                args.event_windows_csv,
                args.metadata_csv,
                args.output_dir,
                args.raw_pcap,
                shared_timebase=args.shared_timebase,
                time_uncertainty_s=args.time_uncertainty_s,
                sync_anchors_csv=args.sync_anchors,
            )
            require_valid_link_audit(audit)
        elif args.command == "summarize":
            _, audit = summarize_directory(args.input_dir, args.output_dir, args.min_repetitions)
            _check_audit(audit, args.allow_underfilled)
        elif args.command == "resources":
            if not any(
                (
                    args.map_specs,
                    args.nm_specs,
                    args.runtime_watermarks,
                    args.elf_specs,
                    args.image_metadata_specs,
                    args.correctness_specs,
                )
            ):
                raise ValueError(
                    "resources requires at least one map, nm, runtime, ELF, metadata, "
                    "or correctness input"
                )
            analyze_resources(
                args.map_specs,
                args.nm_specs,
                args.runtime_watermarks,
                args.output_dir,
                args.category_rules,
                args.elf_specs,
                args.image_metadata_specs,
                args.correctness_specs,
            )
        elif args.command == "project-energy":
            project_daily_energy(args.counts, args.power_summary, args.output_dir)
        elif args.command == "project-battery-life":
            run_battery_projection(args.input_json, args.output_json, args.output_csv)
        elif args.command == "figures":
            data = prepare_figure_csvs(
                args.data_dir,
                args.output_dir,
                require_complete=True,
                profile=args.profile,
            )
            formats = [item.strip() for item in args.formats.split(",")]
            if formats != ["none"]:
                render_figures(data, args.output_dir, formats, profile=args.profile)
        elif args.command == "all":
            args.output_dir.mkdir(parents=True, exist_ok=True)
            parse_serial_files(
                args.serial_log, args.output_dir, _context_values(args.context_specs)
            )
            _, communication_audit = analyze_communication(
                args.output_dir / "event_metrics.csv", args.output_dir
            )
            require_valid_communication_audit(communication_audit)
            if args.profile == "ngmo2":
                if args.ngmo2_manifest is None:
                    raise ValueError("--profile ngmo2 requires --ngmo2-manifest")
                autonomous_inputs = (
                    args.ngmo2_proofs_csv,
                    args.ngmo2_capture_status_csv,
                    args.ngmo2_build_context_csv,
                )
                if any(autonomous_inputs) and not all(autonomous_inputs):
                    raise ValueError(
                        "autonomous NGMO2 analysis requires --ngmo2-proofs-csv, "
                        "--ngmo2-capture-status-csv, and --ngmo2-build-context-csv"
                    )
                if all(autonomous_inputs):
                    _, _, ngmo2_audit, _ = analyze_autonomous_captures(
                        args.ngmo2_manifest,
                        args.ngmo2_proofs_csv,
                        args.ngmo2_capture_status_csv,
                        args.output_dir,
                        build_context_csv=args.ngmo2_build_context_csv,
                        max_guard_drift_fraction=args.max_guard_drift_fraction,
                    )
                    require_valid_autonomous_audit(ngmo2_audit)
                else:
                    _, _, ngmo2_audit = analyze_envelope_captures(
                        args.ngmo2_manifest,
                        args.output_dir / "event_metrics.csv",
                        args.output_dir / "run_metadata.csv",
                        args.output_dir,
                        max_guard_drift_fraction=args.max_guard_drift_fraction,
                    )
                    require_valid_ngmo2_audit(ngmo2_audit)
            elif args.profile == "instrumented" and args.power_inputs:
                _run_power(args, args.output_dir, defaults=True)
            if any(
                (
                    args.map_specs,
                    args.nm_specs,
                    args.runtime_watermarks,
                    args.elf_specs,
                    args.image_metadata_specs,
                    args.correctness_specs,
                )
            ):
                analyze_resources(
                    args.map_specs,
                    args.nm_specs,
                    args.runtime_watermarks,
                    args.output_dir,
                    args.category_rules,
                    args.elf_specs,
                    args.image_metadata_specs,
                    args.correctness_specs,
                )
            _, audit = summarize_directory([args.output_dir], args.output_dir, args.min_repetitions)
            _check_audit(audit, args.allow_underfilled)
            if args.event_counts:
                project_daily_energy(
                    args.event_counts, args.output_dir / "power_summary.csv", args.output_dir
                )
            data = prepare_figure_csvs(
                [args.output_dir],
                args.output_dir,
                require_complete=True,
                profile=args.profile,
            )
            formats = [item.strip() for item in args.formats.split(",")]
            if formats != ["none"]:
                render_figures(data, args.output_dir, formats, profile=args.profile)
        else:
            parser.error(f"unknown command {args.command}")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
