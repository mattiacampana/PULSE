"""Incremental, append-only logs for simulation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


class SimulationLogger:
    """Write the simulator's CSV logs row by row.

    CSV column layouts are deliberately fixed.  This makes it safe to append
    to logs produced by earlier seeds without rereading or rewriting them.
    Values specific to an algorithm are serialised in the ``details`` column.
    Every write is flushed immediately, so completed work remains available if
    a long run is interrupted.
    """

    _SCHEMAS: dict[str, tuple[str, ...]] = {
        "metrics": (
            "seed", "time", "node", "split", "accuracy", "macro_f1",
        ),
        "training": (
            "seed", "time", "node", "loss", "sgd_steps",
            "total_sgd_steps", "details",
        ),
        "contacts": (
            "seed", "time", "requester_id", "peer_id", "details",
        )
    }

    def __init__(self, output_dir: Path, seed: int) -> None:
        self.output_dir = output_dir
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, Any] = {}
        self._writers: dict[str, csv.DictWriter] = {}
        self._row_counts = {name: 0 for name in self._SCHEMAS}

        for name, fieldnames in self._SCHEMAS.items():
            path = self.output_dir / f"{name}.csv"
            has_content = path.exists() and path.stat().st_size > 0
            if has_content:
                with path.open(newline="", encoding="utf-8") as handle:
                    existing_fields = next(csv.reader(handle), [])
                if tuple(existing_fields) != fieldnames:
                    raise ValueError(
                        f"Existing log {path} has incompatible columns: "
                        f"{existing_fields}. Expected {list(fieldnames)}."
                    )
            handle = path.open("a", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not has_content:
                writer.writeheader()
                handle.flush()
            self._files[name] = handle
            self._writers[name] = writer

    def log(self, name: str, row: Mapping[str, Any]) -> None:
        """Append one row to a named log and flush it to disk."""
        if name not in self._SCHEMAS:
            raise ValueError(f"Unknown log name: {name}.")

        fields = self._SCHEMAS[name]
        values = dict(row)
        if "seed" in fields:
            supplied_seed = values.pop("seed", self.seed)
            if supplied_seed != self.seed:
                raise ValueError(
                    f"Log row seed ({supplied_seed}) does not match logger seed ({self.seed})."
                )
            values["seed"] = self.seed

        extras = {key: values.pop(key) for key in list(values) if key not in fields}
        if "details" in fields:
            explicit_details = values.get("details")
            if explicit_details is not None and not isinstance(explicit_details, Mapping):
                raise TypeError("The 'details' log value must be a mapping or None.")
            details = dict(explicit_details or {})
            details.update(extras)
            values["details"] = json.dumps(details, default=str, sort_keys=True) if details else ""
        elif extras:
            raise ValueError(f"Unexpected fields for {name}.csv: {sorted(extras)}.")

        self._writers[name].writerow({field: values.get(field) for field in fields})
        self._files[name].flush()
        self._row_counts[name] += 1

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        """Write the end-of-run JSON summary for this seed."""
        path = self.output_dir / f"summary_seed_{self.seed}.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    def row_count(self, name: str) -> int:
        """Return rows written by this logger instance, excluding earlier runs."""
        return self._row_counts[name]

    def close(self) -> None:
        """Close all open files. Safe to call more than once."""
        for handle in self._files.values():
            if not handle.closed:
                handle.close()
