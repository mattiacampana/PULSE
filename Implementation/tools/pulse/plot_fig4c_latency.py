#!/usr/bin/env python3
"""Create the compact Fig. 4c latency PDF using paper-matched typography."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


PLOT_ROWS = [
	("Scan (I)", "accepted_discovery", "initiator", "ble_scan", "discovery"),
	("Connect (I)", "accepted_discovery", "initiator", "ble_connect", "discovery"),
	("Select (I)", "accepted_connected", "initiator", "peer_selection", "connected"),
	("Head ser. (I)", "accepted_connected", "initiator", "head_serialization", "connected"),
	("Head transfer (I)", "accepted_connected", "initiator", "head_transfer", "connected"),
	("Encoder (I)", "accepted_connected", "initiator", "initiator_encoder", "connected"),
	("Head F/B (I)", "accepted_connected", "initiator", "initiator_head_forward_backward", "connected"),
	("Encoder (R)", "accepted_connected", "responder", "responder_encoder", "connected"),
	("Head F/B (R)", "accepted_connected", "responder", "responder_head_forward_backward", "connected"),
	("Grad. ser. (R)", "accepted_connected", "responder", "gradient_serialization", "connected"),
	("Grad. transfer (I)", "accepted_connected", "initiator", "gradient_transfer", "connected"),
	("Agreement (I)", "accepted_connected", "initiator", "agreement", "connected"),
	("Normalize (I)", "accepted_connected", "initiator", "normalization", "connected"),
	("Mix (I)", "accepted_connected", "initiator", "mixing", "connected"),
	("Connected E2E", "accepted_connected", "initiator", "end_to_end", "connected"),
	("Discovery E2E", "accepted_discovery", "initiator", "end_to_end", "discovery"),
	("Local update", "local", "local", "end_to_end", "local"),
]


def _register_paper_font() -> str:
	kpsewhich = shutil.which("kpsewhich")
	if kpsewhich is None:
		return "Times-Roman"
	paths = []
	for suffix in ("afm", "pfb"):
		result = subprocess.run(
			[kpsewhich, f"utmr8a.{suffix}"],
			check=True,
			capture_output=True,
			text=True,
		)
		paths.append(result.stdout.strip())
	face = pdfmetrics.EmbeddedType1Face(*paths)
	pdfmetrics.registerTypeFace(face)
	font_name = "PaperTimes"
	pdfmetrics.registerFont(pdfmetrics.Font(font_name, face.name, "WinAnsiEncoding"))
	return font_name


def _load_samples(path: Path) -> dict[tuple[str, str, str], dict[str, list[object]]]:
	grouped: defaultdict[tuple[str, str, str], dict[str, list[object]]] = defaultdict(
		lambda: {"values": [], "trial_ids": []}
	)
	with path.open(newline="", encoding="utf-8-sig") as stream:
		for row in csv.DictReader(stream):
			key = (row["event_path"], row["role"], row["stage"])
			grouped[key]["values"].append(float(row["duration_ms"]))
			grouped[key]["trial_ids"].append(row["trial_id"])
	return dict(grouped)


def _validate(data: dict[tuple[str, str, str], dict[str, list[object]]]) -> None:
	for _label, event_path, role, stage, _series in PLOT_ROWS:
		key = (event_path, role, stage)
		if key not in data:
			raise ValueError(f"missing required latency row: {key}")
		values = [float(value) for value in data[key]["values"]]
		trial_ids = data[key]["trial_ids"]
		if not 96 <= len(values) <= 100:
			raise ValueError(f"invalid sample count for {key}: {len(values)}")
		if len(set(trial_ids)) != len(values):
			raise ValueError(f"duplicate trial IDs for {key}")
		if any(value <= 0 for value in values):
			raise ValueError(f"non-positive latency for {key}")


def _percentile(values: list[float], percentile: float) -> float:
	ordered = sorted(values)
	position = (len(ordered) - 1) * percentile
	lower = math.floor(position)
	upper = math.ceil(position)
	if lower == upper:
		return ordered[lower]
	fraction = position - lower
	return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _draw_diamond(pdf: canvas.Canvas, x: float, y: float, radius: float) -> None:
	path = pdf.beginPath()
	path.moveTo(x, y + radius)
	path.lineTo(x + radius, y)
	path.lineTo(x, y - radius)
	path.lineTo(x - radius, y)
	path.close()
	pdf.drawPath(path, fill=1, stroke=1)


def _draw_latency_pdf(
	output_path: Path,
	data: dict[tuple[str, str, str], dict[str, list[object]]],
) -> None:
	page_width = 156.892
	page_height = 157.316
	plot_left = 62.0
	plot_right = 153.0
	plot_bottom = 28.0
	plot_top = 122.0
	x_min = math.log10(0.012)
	x_max = math.log10(5000.0)
	series_colors = {
		"connected": ((0.12, 0.47, 0.71), (0.76, 0.86, 0.94)),
		"discovery": ((1.00, 0.50, 0.05), (1.00, 0.87, 0.72)),
		"local": ((0.00, 0.62, 0.45), (0.72, 0.90, 0.84)),
	}
	roman_font = _register_paper_font()

	def x_position(value: float) -> float:
		log_value = math.log10(value)
		return plot_left + (log_value - x_min) / (x_max - x_min) * (plot_right - plot_left)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	pdf = canvas.Canvas(
		str(output_path),
		pagesize=(page_width, page_height),
		pageCompression=1,
		initialFontName=roman_font,
		initialFontSize=5.7,
	)
	pdf.setTitle("PULSE on-device latency")
	pdf.setAuthor("SensWear")

	legend_items = (("Connected", "connected", 62.0), ("Discovery", "discovery", 101.0), ("Local", "local", 134.0))
	for label, series, x in legend_items:
		dark, light = series_colors[series]
		pdf.setFillColorRGB(*light)
		pdf.setStrokeColorRGB(*dark)
		pdf.rect(x, 148.0, 5.0, 4.5, fill=1, stroke=1)
		pdf.setFillColorRGB(0.12, 0.12, 0.12)
		pdf.setFont(roman_font, 5.7)
		pdf.drawString(x + 6.5, 148.0, label)

	pdf.setFillColorRGB(0.20, 0.20, 0.20)
	pdf.setFont(roman_font, 5.2)
	pdf.drawCentredString((plot_left + plot_right) / 2, 140.5, "Box = IQR; line = median; diamond = mean")
	pdf.drawCentredString((plot_left + plot_right) / 2, 134.5, "Whiskers = p5-p95; dots = outside whiskers")

	for tick in (0.1, 1.0, 10.0, 100.0, 1000.0):
		x = x_position(tick)
		pdf.setStrokeColorRGB(0.88, 0.88, 0.88)
		pdf.setLineWidth(0.45)
		pdf.line(x, plot_bottom, x, plot_top)
		pdf.setFillColorRGB(0.12, 0.12, 0.12)
		pdf.setFont(roman_font, 5.8)
		label = "1k" if tick == 1000.0 else f"{tick:g}"
		pdf.drawCentredString(x, 19.0, label)

	row_spacing = (plot_top - plot_bottom) / (len(PLOT_ROWS) - 1)
	for index, (label, event_path, role, stage, series) in enumerate(PLOT_ROWS):
		y = plot_top - index * row_spacing
		values = [float(value) for value in data[(event_path, role, stage)]["values"]]
		p5 = _percentile(values, 0.05)
		q1 = _percentile(values, 0.25)
		median = _percentile(values, 0.50)
		q3 = _percentile(values, 0.75)
		p95 = _percentile(values, 0.95)
		mean = sum(values) / len(values)
		dark, light = series_colors[series]

		pdf.setFillColorRGB(0.08, 0.08, 0.08)
		pdf.setFont(roman_font, 5.7)
		pdf.drawRightString(plot_left - 3.0, y - 1.8, label)
		pdf.setStrokeColorRGB(*dark)
		pdf.setLineWidth(0.55)
		pdf.line(x_position(p5), y, x_position(p95), y)
		pdf.line(x_position(p5), y - 1.3, x_position(p5), y + 1.3)
		pdf.line(x_position(p95), y - 1.3, x_position(p95), y + 1.3)
		pdf.setFillColorRGB(*light)
		box_left = x_position(q1)
		box_right = x_position(q3)
		pdf.rect(box_left, y - 1.6, max(box_right - box_left, 0.35), 3.2, fill=1, stroke=1)
		pdf.setStrokeColorRGB(0.12, 0.12, 0.12)
		pdf.setLineWidth(0.65)
		pdf.line(x_position(median), y - 1.8, x_position(median), y + 1.8)
		pdf.setFillColorRGB(*dark)
		pdf.setStrokeColorRGB(1, 1, 1)
		_draw_diamond(pdf, x_position(mean), y, 1.45)
		pdf.setFillColorRGB(*light)
		pdf.setStrokeColorRGB(*dark)
		for value in values:
			if value < p5 or value > p95:
				pdf.circle(x_position(value), y, 0.34, fill=1, stroke=0)

	pdf.setStrokeColorRGB(0.70, 0.70, 0.70)
	pdf.setLineWidth(0.55)
	pdf.line(plot_left, plot_bottom - 2.0, plot_right, plot_bottom - 2.0)
	pdf.setFillColorRGB(0.08, 0.08, 0.08)
	pdf.setFont(roman_font, 6.4)
	pdf.drawCentredString((plot_left + plot_right) / 2, 7.0, "Latency (ms; log scale)")
	pdf.showPage()
	pdf.save()


def _render_png(pdf_path: Path, output_prefix: Path) -> None:
	pdftoppm = shutil.which("pdftoppm")
	if pdftoppm is None:
		return
	subprocess.run(
		[pdftoppm, "-png", "-r", "600", "-singlefile", str(pdf_path), str(output_prefix)],
		check=True,
	)


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", type=Path, required=True)
	parser.add_argument("--output-prefix", type=Path, required=True)
	args = parser.parse_args()
	data = _load_samples(args.input)
	_validate(data)
	pdf_path = args.output_prefix.with_suffix(".pdf")
	_draw_latency_pdf(pdf_path, data)
	_render_png(pdf_path, args.output_prefix)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
