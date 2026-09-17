"""Audit the available Keysight streams and plot only defensible power evidence.

The initiator has a complete capture manifest and extracted windows that can be
checked against its raw stream. The responder stream has neither a manifest nor
unaltered provenance; it is plotted separately as a diagnostic, never integrated
as a paper energy result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from keysight_34465a import _stream_event_onsets


ROOT = Path(__file__).resolve().parents[2]
INIT_DIR = ROOT / "capture/p01/keysight-initiator-r03"
RESP_DIR = ROOT / "capture/p01/keysight-responder-r01"
OUT_DIR = ROOT / "results/fig4a"
INIT_RAW = INIT_DIR / "keysight_34465a_initiator_stream_raw.csv"
INIT_SAMPLES = INIT_DIR / "keysight_34465a_initiator_samples.csv"
INIT_MANIFEST = INIT_DIR / "keysight_34465a_capture_manifest.csv"
RESP_RAW = RESP_DIR / "keysight_34465a_responder_stream_raw.csv"
VOLTAGE_V = 3.7
INTERVAL_S = 0.02
PRE_SAMPLES = 20
POST_SAMPLES = 205
WARMUPS = 5


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_raw(path: Path, expected_role: str) -> np.ndarray:
    values = []
    with path.open(newline="", encoding="utf-8") as source:
        for expected_index, row in enumerate(csv.DictReader(source)):
            if row["measured_role"] != expected_role:
                raise ValueError(f"role mismatch at {path}:{expected_index + 2}")
            if int(row["sample_index"]) != expected_index:
                raise ValueError(f"noncontiguous sample index at {path}:{expected_index + 2}")
            if not math.isclose(float(row["time_s"]), expected_index * INTERVAL_S, abs_tol=1e-7):
                raise ValueError(f"unexpected sample time at {path}:{expected_index + 2}")
            values.append(float(row["current_a"]))
    return np.asarray(values, dtype=float)


def load_initiator(raw: np.ndarray) -> dict[str, list[dict[str, object]]]:
    with INIT_MANIFEST.open(newline="", encoding="utf-8") as source:
        manifest = next(csv.DictReader(source))
    if manifest["capture_status"] != "complete_timed_stream_and_software_event_audit_pass":
        raise ValueError("initiator manifest does not report a complete capture")
    if digest(INIT_RAW) != manifest["raw_csv_sha256"]:
        raise ValueError("initiator raw SHA-256 differs from capture manifest")
    if digest(INIT_SAMPLES) != manifest["sample_csv_sha256"]:
        raise ValueError("initiator extracted-window SHA-256 differs from capture manifest")
    if len(raw) != int(manifest["raw_reading_count"]):
        raise ValueError("initiator raw reading count differs from manifest")
    if not math.isclose(float(manifest["supply_voltage_v"]), VOLTAGE_V):
        raise ValueError("unexpected supply voltage")
    if not math.isclose(float(manifest["sample_interval_s"]), INTERVAL_S):
        raise ValueError("unexpected sample interval")
    if int(manifest["pretrigger_samples"]) != PRE_SAMPLES:
        raise ValueError("unexpected pretrigger length")
    if int(manifest["samples_per_trigger"]) != PRE_SAMPLES + POST_SAMPLES:
        raise ValueError("unexpected event window length")

    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    with INIT_SAMPLES.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            groups[int(row["trigger_index"])].append(row)
    if len(groups) != 210 or sum(map(len, groups.values())) != 210 * 225:
        raise ValueError("initiator event count or window sizes are incomplete")

    paths: dict[str, list[dict[str, object]]] = {"local": [], "accepted_connected": []}
    for trigger_index in range(210):
        rows = sorted(groups[trigger_index], key=lambda row: int(row["sample_index"]))
        if len(rows) != 225:
            raise ValueError(f"event {trigger_index} is incomplete")
        path = "local" if trigger_index < 105 else "accepted_connected"
        role = "local" if path == "local" else "initiator"
        values = np.empty(225, dtype=float)
        for index, row in enumerate(rows):
            raw_index = int(row["source_sample_index"])
            value = float(row["current_a"])
            if int(row["sample_index"]) != index or row["event_path"] != path:
                raise ValueError(f"event metadata mismatch at trigger {trigger_index}")
            if row["event_role"] != role or raw_index >= len(raw):
                raise ValueError(f"raw index or role mismatch at trigger {trigger_index}")
            if value != raw[raw_index]:
                raise ValueError(f"extracted value differs from raw stream at {raw_index}")
            values[index] = value
        if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1):
            raise ValueError(f"invalid DMM reading inside event {trigger_index}")
        baseline_a = float(np.mean(values[:PRE_SAMPLES]))
        post = values[PRE_SAMPLES:]
        paths[path].append(
            {
                "trigger_index": trigger_index,
                "baseline_a": baseline_a,
                "energy_mj": 1000 * VOLTAGE_V * INTERVAL_S * float(np.sum(post - baseline_a)),
                "values_a": values,
            }
        )
    if any(len(events) != 105 for events in paths.values()):
        raise ValueError("initiator does not have 105 events per path")
    return paths


def describe(events: list[dict[str, object]]) -> dict[str, float | int]:
    measured = events[WARMUPS:]
    energies = np.asarray([event["energy_mj"] for event in measured], dtype=float)
    baselines = np.asarray([event["baseline_a"] for event in measured], dtype=float)
    return {
        "n": len(measured),
        "median_energy_mj": float(np.median(energies)),
        "p25_energy_mj": float(np.percentile(energies, 25)),
        "p75_energy_mj": float(np.percentile(energies, 75)),
        "p95_energy_mj": float(np.percentile(energies, 95)),
        "mean_energy_mj": float(np.mean(energies)),
        "median_preevent_power_mw": float(np.median(baselines) * VOLTAGE_V * 1000),
    }


def representative(events: list[dict[str, object]]) -> dict[str, object]:
    measured = events[WARMUPS:]
    median = float(np.median([event["energy_mj"] for event in measured]))
    return min(measured, key=lambda event: abs(float(event["energy_mj"]) - median))


def plot_initiator(paths: dict[str, list[dict[str, object]]]) -> None:
    fig, (trace_ax, energy_ax) = plt.subplots(
        2, 1, figsize=(2.23, 2.55), gridspec_kw={"height_ratios": (1.13, 1), "hspace": 0.45}
    )
    time_s = (np.arange(225) - PRE_SAMPLES) * INTERVAL_S
    colors = {"local": "#176da8", "accepted_connected": "#d17815"}
    labels = {"local": "Local update", "accepted_connected": "Connected-path initiator"}
    for path in paths:
        event = representative(paths[path])
        delta_ma = 1000 * (event["values_a"] - event["baseline_a"])
        trace_ax.plot(time_s, delta_ma, linewidth=0.9, color=colors[path], label=labels[path])
    trace_ax.axvline(0, color="#60666d", linewidth=0.55, linestyle="--")
    trace_ax.axhline(0, color="#adb3b9", linewidth=0.45)
    trace_ax.set_xlim(-0.4, 4.08)
    trace_ax.set_ylabel(r"$\Delta I$ (mA)", fontsize=6)
    trace_ax.set_xlabel("Time from onset (s)", fontsize=6, labelpad=1)
    trace_ax.tick_params(labelsize=5.4, length=2, pad=1)
    trace_ax.legend(loc="upper right", frameon=False, fontsize=5.1, handlelength=1.2)
    trace_ax.spines[["top", "right"]].set_visible(False)

    groups = [[float(event["energy_mj"]) for event in paths[path][WARMUPS:]] for path in paths]
    box = energy_ax.boxplot(
        groups, whis=(5, 95), showmeans=True, patch_artist=True, widths=0.45,
        meanprops={"marker": "D", "markersize": 2.1, "markerfacecolor": "white",
                   "markeredgecolor": "#30343b"},
        flierprops={"marker": ".", "markersize": 1.5, "markerfacecolor": "#60666d",
                    "markeredgecolor": "none"},
        medianprops={"color": "white", "linewidth": 0.9},
        whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6},
    )
    for patch, path in zip(box["boxes"], paths):
        patch.set_facecolor(colors[path])
        patch.set_linewidth(0.6)
    energy_ax.set_xticks((1, 2), ("Local\nn=100", "Connected initiator\nn=100"))
    energy_ax.set_ylabel("Energy (mJ)", fontsize=6)
    energy_ax.tick_params(labelsize=5.4, length=2, pad=1)
    energy_ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.98, bottom=0.20)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"ondevice_power_initiator_partial.{suffix}", dpi=350)
    plt.close(fig)


def plot_responder_diagnostic(raw: np.ndarray) -> dict[str, int | str]:
    exact_repeats = int(np.count_nonzero(raw[1:] == raw[:-1]))
    invalid_indices = np.flatnonzero(~np.isfinite(raw) | (np.abs(raw) > 1))
    invalid = len(invalid_indices)
    onsets = _stream_event_onsets(raw, 0.0035, INTERVAL_S)
    affected_windows = sum(
        bool(np.any((invalid_indices >= onset - PRE_SAMPLES) &
                    (invalid_indices < onset + POST_SAMPLES))) for onset in onsets
    )
    seconds = len(raw) // 50
    blocks = raw[: seconds * 50].reshape(seconds, 50)
    valid_blocks = np.where(np.isfinite(blocks) & (np.abs(blocks) < 1), blocks, np.nan)
    lo = np.nanmin(valid_blocks, axis=1) * 1000
    hi = np.nanmax(valid_blocks, axis=1) * 1000
    x = np.arange(seconds)
    fig, ax = plt.subplots(figsize=(6.8, 2.1))
    ax.fill_between(x, lo, hi, color="#8c9198", linewidth=0, alpha=0.8)
    ax.scatter(invalid_indices * INTERVAL_S, np.full(invalid, 0.7), s=6,
               marker="|", color="#b93c3c")
    ax.set(xlabel="Elapsed capture time (s)", ylabel="Recorded current (mA)",
           title="Responder stream diagnostic - not a validated power trial")
    ax.text(0.02, 0.93, "Red ticks: invalid readings. No manifest or energy estimate.",
            transform=ax.transAxes, va="top", fontsize=7, bbox={"facecolor": "white", "alpha": 0.85,
                                                                "edgecolor": "none"})
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"responder_stream_diagnostic.{suffix}", dpi=250)
    plt.close(fig)
    return {"raw_readings": len(raw), "sha256": digest(RESP_RAW),
            "adjacent_exact_repeats": exact_repeats, "invalid_values": invalid,
            "candidate_onsets": len(onsets), "invalid_event_windows": affected_windows,
            "manifest_present": False}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    initiator_raw = load_raw(INIT_RAW, "initiator")
    responder_raw = load_raw(RESP_RAW, "responder")
    paths = load_initiator(initiator_raw)
    plot_initiator(paths)
    responder_diagnostic = plot_responder_diagnostic(responder_raw)
    summary = {
        "status": "partial_initiator_only_not_complete_two_role_figure",
        "initiator_raw_sha256": digest(INIT_RAW),
        "initiator_samples_sha256": digest(INIT_SAMPLES),
        "source_voltage_v": VOLTAGE_V,
        "sampling_rate_hz": 1 / INTERVAL_S,
        "pre_event_s": PRE_SAMPLES * INTERVAL_S,
        "post_event_s": POST_SAMPLES * INTERVAL_S,
        "local": describe(paths["local"]),
        "accepted_connected_initiator": describe(paths["accepted_connected"]),
        "responder_diagnostic": responder_diagnostic,
        "limitations": [
            "initiator event-path labels follow capture schedule; exact power-trial UART outcomes unavailable",
            "responder stream contains invalid DMM readings and has no completed capture manifest",
            "the two role runs were sequential and cannot yield simultaneous pair power",
            "power-capture firmware revision differs from archived latency/resource release revision",
            "source-side current includes DMM series burden",
        ],
    }
    (OUT_DIR / "fig4a_partial_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
