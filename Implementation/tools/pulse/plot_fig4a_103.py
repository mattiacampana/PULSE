"""Render a descriptive Figure 4a from the available sequential DMM captures.

This is deliberately separate from the final-capture analysis pipeline. The
responder input was edited in place and contains repeated-value fill, so this
plot labels that limitation rather than treating it as a verified paper result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keysight_34465a import _stream_event_onsets


ROOT = Path(__file__).resolve().parents[2]
INITIATOR_CSV = ROOT / "capture/p01/keysight-initiator-r03/keysight_34465a_initiator_samples.csv"
RESPONDER_CSV = ROOT / "capture/p01/keysight-responder-r01/keysight_34465a_responder_stream_raw.csv"
OUTPUT_DIR = ROOT / "results/fig4a"
SAMPLE_INTERVAL_S = 0.02
PRETRIGGER_SAMPLES = 20
SAMPLES_PER_EVENT = 225
WARMUP_EVENTS = 5
SUPPLY_V = 3.7
TRIGGER_A = 0.0035


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_record(currents: list[float], role: str, ordinal: int) -> dict[str, object]:
    if len(currents) != SAMPLES_PER_EVENT:
        raise ValueError(f"{role} event {ordinal} has {len(currents)} samples")
    baseline = statistics.fmean(currents[:PRETRIGGER_SAMPLES])
    post = currents[PRETRIGGER_SAMPLES:]
    return {
        "role": role,
        "ordinal": ordinal,
        "warmup": ordinal < WARMUP_EVENTS,
        "energy_mj": 1000 * SUPPLY_V * SAMPLE_INTERVAL_S * sum(i - baseline for i in post),
        "time_s": [(i - PRETRIGGER_SAMPLES) * SAMPLE_INTERVAL_S for i in range(len(currents))],
        "delta_current_ma": [1000 * (i - baseline) for i in currents],
    }


def initiator_events() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    with INITIATOR_CSV.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            groups[int(row["trigger_index"])].append(row)
    local: list[dict[str, object]] = []
    initiator: list[dict[str, object]] = []
    for rows in (groups[index] for index in sorted(groups)):
        rows.sort(key=lambda row: int(row["sample_index"]))
        role = rows[0]["event_role"]
        if role not in {"local", "initiator"}:
            raise ValueError(f"unexpected initiator-file event role: {role}")
        target = local if role == "local" else initiator
        target.append(event_record([float(row["current_a"]) for row in rows], role, len(target)))
    if len(local) != 105 or len(initiator) != 105:
        raise ValueError("initiator file does not contain 105 events per path")
    return local, initiator


def responder_events() -> tuple[list[dict[str, object]], dict[str, object]]:
    with RESPONDER_CSV.open(newline="", encoding="utf-8") as source:
        currents = [float(row["current_a"]) for row in csv.DictReader(source)]
    candidates = _stream_event_onsets(currents, TRIGGER_A, SAMPLE_INTERVAL_S)
    if len(candidates) != 105:
        raise ValueError(f"expected 105 current-file onset candidates; found {len(candidates)}")
    # The two additional detections in the edited CSV each include an exact
    # previous-sample copy in the detector's three-reading confirmation span.
    retained = [
        onset
        for onset in candidates
        if not any(currents[index] == currents[index - 1] for index in range(onset, onset + 3))
    ]
    if len(retained) != 103:
        raise ValueError(f"expected 103 retained responder onsets; found {len(retained)}")
    events: list[dict[str, object]] = []
    for ordinal, onset in enumerate(retained):
        start = onset - PRETRIGGER_SAMPLES
        end = start + SAMPLES_PER_EVENT
        if start < 0 or end > len(currents):
            raise ValueError(f"responder event {ordinal} has an incomplete window")
        events.append(event_record(currents[start:end], "responder", ordinal))
    provenance = {
        "raw_readings": len(currents),
        "candidate_onsets": len(candidates),
        "excluded_onset_indices": [onset for onset in candidates if onset not in retained],
        "retained_onsets": len(retained),
        "adjacent_exact_repeats": sum(
            currents[index] == currents[index - 1] for index in range(1, len(currents))
        ),
    }
    return events, provenance


def measured(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [event for event in events if not event["warmup"]]


def representative(events: list[dict[str, object]]) -> dict[str, object]:
    center = statistics.median(float(event["energy_mj"]) for event in events)
    return min(events, key=lambda event: abs(float(event["energy_mj"]) - center))


def render(initiator: list[dict[str, object]], responder: list[dict[str, object]],
           responder_provenance: dict[str, object]) -> None:
    initiator_m, responder_m = map(measured, (initiator, responder))
    groups = [
        [float(event["energy_mj"]) for event in initiator_m],
        [float(event["energy_mj"]) for event in responder_m],
    ]

    figure, (trace_ax, energy_ax) = plt.subplots(
        2, 1, figsize=(2.18, 2.45), gridspec_kw={"height_ratios": (1.2, 1.0), "hspace": 0.55}
    )
    colors = ("#176da8", "#d17815")
    for events, label, color in zip(
        (initiator_m, responder_m), ("Initiator", "Responder"), colors
    ):
        event = representative(events)
        trace_ax.plot(event["time_s"], event["delta_current_ma"], label=label,
                      color=color, linewidth=0.8)
    trace_ax.axvline(0, color="#30343b", linestyle="--", linewidth=0.5)
    trace_ax.axhline(0, color="#a8adb4", linewidth=0.45)
    trace_ax.set_xlim(-0.4, 4.08)
    trace_ax.set_ylabel(r"$\Delta I$ (mA)", fontsize=5.8)
    trace_ax.set_xlabel("Time from onset (s)", fontsize=5.8, labelpad=1)
    trace_ax.legend(loc="upper right", ncol=1, frameon=False, fontsize=5.0,
                    handlelength=1.2, columnspacing=0.8)
    trace_ax.tick_params(labelsize=5.1, length=2, pad=1)
    trace_ax.spines[["top", "right"]].set_visible(False)

    boxes = energy_ax.boxplot(
        groups, whis=(5, 95), showmeans=True, widths=0.42, patch_artist=True,
        meanprops={"marker": "D", "markersize": 2.1, "markerfacecolor": "white",
                   "markeredgecolor": "#30343b"},
        flierprops={"marker": ".", "markersize": 1.5, "markeredgecolor": "none",
                    "markerfacecolor": "#59636f"},
        medianprops={"color": "white", "linewidth": 0.8},
        whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_linewidth(0.6)
    energy_ax.set_xticks((1, 2), ("Initiator\nn=100", "Responder\nn=98"))
    energy_ax.set_ylabel("Incremental energy (mJ)", fontsize=5.8)
    energy_ax.tick_params(labelsize=5.1, length=2, pad=1)
    energy_ax.spines[["top", "right"]].set_visible(False)
    figure.text(0.10, 0.115, "Init.: 105 events; Resp.: 103 retained events.", fontsize=4.7)
    figure.text(0.10, 0.082, "Five warm-ups excluded from each distribution.", fontsize=4.7)
    figure.text(0.10, 0.049, "Sequential captures; responder has repeated-value fill.", fontsize=4.7)
    figure.subplots_adjust(left=0.24, right=0.98, top=0.98, bottom=0.23)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "ondevice_power_103_exploratory.pdf", metadata={
        "Title": "Figure 4a - initiator and responder current and energy",
        "Subject": "Responder source includes repeated-value fill; sequential role captures",
    })
    plt.close(figure)

    provenance = {
        "figure_status": "exploratory_not_validated_final_capture",
        "initiator_source": str(INITIATOR_CSV.relative_to(ROOT)),
        "initiator_sha256": sha256(INITIATOR_CSV),
        "responder_source": str(RESPONDER_CSV.relative_to(ROOT)),
        "responder_sha256": sha256(RESPONDER_CSV),
        "supply_voltage_v": SUPPLY_V,
        "sample_interval_s": SAMPLE_INTERVAL_S,
        "window_posttrigger_s": (SAMPLES_PER_EVENT - PRETRIGGER_SAMPLES) * SAMPLE_INTERVAL_S,
        "baseline": "mean of 20 pre-onset readings",
        "warmup_excluded_per_path": WARMUP_EVENTS,
        "recorded_counts": {"initiator": len(initiator), "responder": len(responder)},
        "measured_counts": {"initiator": len(initiator_m), "responder": len(responder_m)},
        "trace_semantics": "independently selected median-energy event per role, onset-aligned; not synchronized",
        "median_incremental_energy_mj": {
            "initiator": statistics.median(groups[0]),
            "responder": statistics.median(groups[1]),
        },
    }
    provenance.update(responder_provenance)
    (OUTPUT_DIR / "ondevice_power_103_exploratory.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    _, initiator_role_events = initiator_events()
    responder_role_events, responder_provenance = responder_events()
    render(initiator_role_events, responder_role_events, responder_provenance)
