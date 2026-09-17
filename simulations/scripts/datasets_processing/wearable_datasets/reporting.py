"""Small, dependency-free console reporting for dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ConsoleReporter:
    """Emit concise, consistently formatted pipeline messages."""

    verbose: bool = False
    quiet: bool = False

    def section(self, dataset: str, title: str) -> None:
        if not self.quiet:
            print(f"\n[{dataset}] {title}")

    def info(self, dataset: str, stage: str, message: str) -> None:
        if not self.quiet:
            print(f"[{dataset}] {stage:<10} {message}")

    def detail(self, dataset: str, stage: str, message: str) -> None:
        if self.verbose and not self.quiet:
            print(f"[{dataset}] {stage:<10} {message}")


REPORTER = ConsoleReporter()


def configure_reporting(*, verbose: bool = False, quiet: bool = False) -> None:
    """Configure the process-wide reporter used by adapters and helpers."""
    REPORTER.verbose = verbose
    REPORTER.quiet = quiet


def human_size(size: int) -> str:
    """Format a byte count for human-facing summaries."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def describe_paths(paths: Iterable[Path]) -> str:
    """Return a compact comma-separated artifact summary."""
    return ", ".join(f"{path.name} ({human_size(path.stat().st_size)})" for path in paths if path.exists())
