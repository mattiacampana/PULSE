#!/usr/bin/env python3
"""Process a local Cambridge/Haggle Experiment 2 archive for DeepRAP.

The output CSV files use the simulator format: ``time,node1,node2``.
Only contacts between experiment iMotes are retained; sightings of external
Bluetooth devices are discarded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tarfile
from pathlib import Path


CONTACTS_NAME = "contacts.Exp2.dat"
CAMBRIDGE_NODES = set(range(1, 13))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mobility/haggle"),
        help="Destination root (default: data/mobility/haggle).",
    )
    parser.add_argument(
        "--snapshot-step",
        type=int,
        default=120,
        help="Seconds between contact opportunities (default: 120).",
    )
    parser.add_argument(
        "--keep-absolute-time",
        action="store_true",
        help="Do not shift the first output timestamp to zero.",
    )
    return parser.parse_args()


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a tar archive while rejecting paths outside the destination."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination, filter="data")


def find_one(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename} under {root}, found {len(matches)}.")
    return matches[0]


def read_intervals(path: Path, allowed_nodes: set[int]) -> list[tuple[int, int, int, int]]:
    """Read Haggle rows: observer, observed, first time, last time, ..."""
    intervals: list[tuple[int, int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 4:
                raise ValueError(f"Malformed row {line_number} in {path}: {line.rstrip()}")
            try:
                observer, observed, start, end = map(int, fields[:4])
            except ValueError as error:
                raise ValueError(f"Non-integer row {line_number} in {path}: {line.rstrip()}") from error
            if observer == observed or observer not in allowed_nodes or observed not in allowed_nodes:
                continue
            if end < start:
                raise ValueError(f"End precedes start at row {line_number} in {path}.")
            left, right = sorted((observer, observed))
            intervals.append((start, end, left, right))
    return intervals


def intervals_to_snapshots(
    intervals: list[tuple[int, int, int, int]], step: int, normalize_time: bool
) -> list[tuple[int, str, str]]:
    """Expand intervals onto a regular grid and merge asymmetric sightings."""
    if step <= 0:
        raise ValueError("--snapshot-step must be positive.")
    snapshots: set[tuple[int, int, int]] = set()
    for start, end, left, right in intervals:
        first = (start // step) * step
        last = (end // step) * step
        for timestamp in range(first, last + 1, step):
            snapshots.add((timestamp, left, right))
    ordered = sorted(snapshots)
    origin = ordered[0][0] if ordered and normalize_time else 0
    return [(time - origin, f"trace_{left}", f"trace_{right}") for time, left, right in ordered]


def write_outputs(
    output: Path,
    events: list[tuple[int, str, str]],
    source: Path,
    step: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "node1", "node2"))
        writer.writerows(events)

    nodes = sorted({node for _, left, right in events for node in (left, right)})
    summary = {
        "dataset": "cambridge_haggle_experiment_2",
        "source_file": str(source),
        "allowed_internal_nodes": [f"trace_{node}" for node in sorted(CAMBRIDGE_NODES)],
        "snapshot_step_seconds": step,
        "num_nodes_observed": len(nodes),
        "nodes_observed": nodes,
        "num_contact_events": len(events),
        "num_unique_pairs": len({(left, right) for _, left, right in events}),
        "duration_seconds": events[-1][0] if events else 0,
        "time_normalized_to_zero": bool(events and events[0][0] == 0),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def prepare(args: argparse.Namespace) -> Path:
    archive_value = os.environ.get("HAGGLE_ARCHIVE")
    if not archive_value:
        raise RuntimeError(
            "HAGGLE_ARCHIVE is not set. Set it to the local "
            "imote-trace2.tar.gz path."
        )

    archive = Path(archive_value).expanduser().resolve()
    extracted = args.output_dir / "raw" / "cambridge"

    if not archive.is_file():
        raise RuntimeError(
            f"Cambridge archive not found: {archive}\n"
            "Set HAGGLE_ARCHIVE to the correct imote-trace2.tar.gz path."
        )

    print(f"Using Cambridge archive: {archive}", flush=True)

    if not extracted.exists():
        print(f"Extracting archive into: {extracted}", flush=True)
        safe_extract(archive, extracted)
    else:
        print(f"Using existing extracted files: {extracted}", flush=True)

    contacts = find_one(extracted, CONTACTS_NAME)
    intervals = read_intervals(contacts, CAMBRIDGE_NODES)
    events = intervals_to_snapshots(intervals, args.snapshot_step, not args.keep_absolute_time)

    output = args.output_dir / "processed" / "cambridge.csv"
    write_outputs(output, events, contacts, args.snapshot_step)
    print(f"Wrote {len(events):,} events to {output}", flush=True)
    return output


def main() -> None:
    args = parse_args()
    prepare(args)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, tarfile.TarError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None