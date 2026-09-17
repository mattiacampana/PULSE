#!/usr/bin/env python3
"""Create the compact Fig. 4b resource table and vector PDF."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


def _read_rows(path: Path) -> list[dict[str, str]]:
	with path.open("r", encoding="utf-8-sig", newline="") as stream:
		return list(csv.DictReader(stream))


def _one(rows: list[dict[str, str]], image: str, memory: str) -> dict[str, str]:
	matches = [row for row in rows if row.get("image") == image and row.get("memory") == memory]
	if len(matches) != 1:
		raise ValueError(f"expected one {image}/{memory} row, found {len(matches)}")
	return matches[0]


def _unique(rows: list[dict[str, str]], field: str) -> str:
	values = {row.get(field, "").strip() for row in rows if row.get(field, "").strip()}
	if len(values) != 1:
		raise ValueError(f"expected one non-empty {field}, found {sorted(values)}")
	return values.pop()


def _number(row: dict[str, str], field: str) -> int:
	return int(float(row[field]))


def _kib(value: int) -> float:
	return value / 1024.0


def _register_paper_fonts() -> tuple[str, str]:
	kpsewhich = shutil.which("kpsewhich")
	if kpsewhich is None:
		return "Times-Roman", "Times-Bold"
	registered: list[str] = []
	for alias, stem in (("PaperTimes", "utmr8a"), ("PaperTimes-Bold", "utmb8a")):
		paths = []
		for suffix in ("afm", "pfb"):
			result = subprocess.run(
				[kpsewhich, f"{stem}.{suffix}"],
				check=True,
				capture_output=True,
				text=True,
			)
			paths.append(result.stdout.strip())
		face = pdfmetrics.EmbeddedType1Face(*paths)
		pdfmetrics.registerTypeFace(face)
		pdfmetrics.registerFont(pdfmetrics.Font(alias, face.name, "WinAnsiEncoding"))
		registered.append(alias)
	return registered[0], registered[1]


def _write_figure_csv(
	path: Path,
	resources: list[dict[str, str]],
	metadata: list[dict[str, str]],
) -> list[dict[str, object]]:
	common = {
		"head_serialized_bytes": int(_unique(metadata, "head_serialized_bytes")),
		"gradient_serialized_bytes": int(_unique(metadata, "gradient_buffer_bytes_each")),
		"peer_entry_bytes": int(_unique(metadata, "peer_entry_bytes")),
		"peer_limit": int(_unique(metadata, "peer_limit")),
		"utility_table_bytes": int(_unique(metadata, "utility_table_bytes")),
		"firmware_revision": _unique(metadata, "firmware_revision"),
		"artifact_sha256": _unique(metadata, "artifact_sha256"),
	}
	rows: list[dict[str, object]] = []
	for resource, memory in (("Flash", "flash"), ("Static RAM", "static_ram")):
		baseline = _one(resources, "baseline", memory)
		pulse = _one(resources, "pulse", memory)
		baseline_used = _number(baseline, "used_bytes")
		pulse_used = _number(pulse, "used_bytes")
		capacity = _number(pulse, "capacity_bytes")
		increment = pulse_used - baseline_used
		free = capacity - pulse_used
		rows.append(
			{
				"resource": resource,
				"baseline_used_bytes": baseline_used,
				"pulse_used_bytes": pulse_used,
				"incremental_bytes": increment,
				"capacity_bytes": capacity,
				"pulse_free_bytes": free,
				"baseline_used_kib": f"{_kib(baseline_used):.1f}",
				"pulse_used_kib": f"{_kib(pulse_used):.1f}",
				"incremental_kib": f"{_kib(increment):.1f}",
				"capacity_kib": f"{_kib(capacity):.1f}",
				"pulse_free_kib": f"{_kib(free):.1f}",
				"baseline_utilization_pct": f"{100.0 * baseline_used / capacity:.1f}",
				"pulse_utilization_pct": f"{100.0 * pulse_used / capacity:.1f}",
				"pulse_map_sha256": pulse["source_sha256"],
				"baseline_map_sha256": baseline["source_sha256"],
				**common,
			}
		)

	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as stream:
		writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)
	return rows


def _draw_resource_figure(path: Path, rows: list[dict[str, object]]) -> None:
	# Match the square canvas and nominal type size of Fig. 4c so both panels
	# render at the same height and text scale in a 0.32\textwidth subfigure.
	page_width = 156.892
	page_height = 157.316
	bar_x = 34.0
	bar_width = 118.0
	bar_height = 15.0
	colors = {
		"ink": (0.10, 0.13, 0.18),
		"muted": (0.38, 0.43, 0.50),
		"baseline": (0.36, 0.40, 0.46),
		"increment": (0.10, 0.45, 0.78),
		"unused": (0.91, 0.93, 0.95),
		"border": (0.76, 0.80, 0.84),
	}
	roman_font, bold_font = _register_paper_fonts()

	path.parent.mkdir(parents=True, exist_ok=True)
	pdf = canvas.Canvas(
		str(path),
		pagesize=(page_width, page_height),
		pageCompression=1,
		initialFontName=roman_font,
		initialFontSize=6.2,
	)
	pdf.setTitle("PULSE static firmware footprint")
	pdf.setAuthor("SensWear")

	for row, y in zip(rows, (113.0, 61.0), strict=True):
		capacity = int(row["capacity_bytes"])
		baseline = int(row["baseline_used_bytes"])
		increment = int(row["incremental_bytes"])
		pulse = int(row["pulse_used_bytes"])
		baseline_width = bar_width * baseline / capacity
		increment_width = bar_width * increment / capacity

		pdf.setFillColorRGB(*colors["ink"])
		pdf.setFont(bold_font, 6.2)
		pdf.drawString(3, y + 4.5, str(row["resource"]))
		pdf.setFont(roman_font, 6.2)
		pdf.drawString(
			bar_x,
			y + 20,
			f"PULSE {float(row['pulse_used_kib']):,.1f} KiB ({float(row['pulse_utilization_pct']):.1f}%)",
		)
		pdf.setFillColorRGB(*colors["muted"])
		pdf.drawRightString(bar_x + bar_width, y + 20, f"capacity {float(row['capacity_kib']):,.0f} KiB")

		pdf.setFillColorRGB(*colors["unused"])
		pdf.setStrokeColorRGB(*colors["border"])
		pdf.roundRect(bar_x, y, bar_width, bar_height, 2.2, fill=1, stroke=1)
		pdf.setFillColorRGB(*colors["baseline"])
		pdf.rect(bar_x, y, baseline_width, bar_height, fill=1, stroke=0)
		pdf.setFillColorRGB(*colors["increment"])
		pdf.rect(bar_x + baseline_width, y, increment_width, bar_height, fill=1, stroke=0)

		pdf.setFillColorRGB(1, 1, 1)
		pdf.setFont(bold_font, 5.8)
		pdf.drawCentredString(
			bar_x + baseline_width / 2,
			y + 5.7,
			f"{float(row['baseline_used_kib']):.1f}",
		)
		pdf.drawCentredString(
			bar_x + baseline_width + increment_width / 2,
			y + 5.7,
			f"+{float(row['incremental_kib']):.1f}",
		)
		pdf.setFillColorRGB(*colors["muted"])
		pdf.setFont(roman_font, 5.8)
		pdf.drawRightString(
			bar_x + bar_width,
			y - 8,
			f"{float(row['pulse_free_kib']):,.1f} KiB free",
		)

	legend_y = 28.0
	pdf.setFillColorRGB(*colors["baseline"])
	pdf.rect(34, legend_y, 6, 6, fill=1, stroke=0)
	pdf.setFillColorRGB(*colors["ink"])
	pdf.setFont(roman_font, 5.8)
	pdf.drawString(43, legend_y + 0.5, "Matched baseline")
	pdf.setFillColorRGB(*colors["increment"])
	pdf.rect(97, legend_y, 6, 6, fill=1, stroke=0)
	pdf.setFillColorRGB(*colors["ink"])
	pdf.drawString(106, legend_y + 0.5, "PULSE increment")

	pdf.setStrokeColorRGB(*colors["border"])
	pdf.line(3, 18, 153, 18)
	pdf.setFillColorRGB(*colors["muted"])
	pdf.setFont(roman_font, 5.8)
	pdf.drawCentredString(
		page_width / 2,
		6,
		f"Head/gradient: {int(rows[0]['head_serialized_bytes']):,} B each   |   Peer entry: {int(rows[0]['peer_entry_bytes'])} B",
	)
	pdf.showPage()
	pdf.save()


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--resource-summary", type=Path, required=True)
	parser.add_argument("--metadata", type=Path, required=True)
	parser.add_argument("--output-csv", type=Path, required=True)
	parser.add_argument("--output-pdf", type=Path, required=True)
	args = parser.parse_args()
	resources = _read_rows(args.resource_summary)
	metadata = _read_rows(args.metadata)
	rows = _write_figure_csv(args.output_csv, resources, metadata)
	_draw_resource_figure(args.output_pdf, rows)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
