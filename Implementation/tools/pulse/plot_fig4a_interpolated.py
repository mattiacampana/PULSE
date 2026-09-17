"""Figure 4a from intact initiator data and an explicitly imputed responder stream.

Only invalid DMM readings are replaced, in memory, by linear interpolation
between the nearest valid readings. Original sample indices/times and raw CSVs
are never changed. The responder is exploratory: no capture manifest or UART
outcome join exists, and the two roles were recorded in separate runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from keysight_34465a import OVERLOAD_READING_A, _stream_event_onsets
from plot_fig4a_partial import (
    INIT_RAW,
    INIT_SAMPLES,
    INTERVAL_S,
    OUT_DIR,
    POST_SAMPLES,
    PRE_SAMPLES,
    RESP_RAW,
    VOLTAGE_V,
    WARMUPS,
    describe,
    digest,
    load_initiator,
    load_raw,
)


EXPECTED_RESPONDER_SHA256 = "1f1f27127677ba555888164ba639e812a438275612c5e18f5adad64121edbe3e"
TRIGGER_LEVEL_A = 0.0035
EXPECTED_EVENTS = 105
OUTPUT_STEM = "ondevice_power_interpolated_roles"


def event_energy(values: np.ndarray, onset: int) -> tuple[float, float]:
    start = onset - PRE_SAMPLES
    end = onset + POST_SAMPLES
    if start < 0 or end > len(values):
        raise ValueError(f"event at {onset} is outside the saved stream")
    baseline_a = float(np.mean(values[start:onset]))
    energy_mj = 1000 * VOLTAGE_V * INTERVAL_S * float(
        np.sum(values[onset:end] - baseline_a)
    )
    return baseline_a, energy_mj


def repair_responder(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    invalid = ~np.isfinite(raw) | (np.abs(raw) >= OVERLOAD_READING_A)
    if invalid[0] or invalid[-1]:
        raise ValueError("cannot interpolate an invalid endpoint from two neighbors")
    indices = np.arange(len(raw))
    repaired = raw.copy()
    repaired[invalid] = np.interp(indices[invalid], indices[~invalid], raw[~invalid])
    if not np.array_equal(repaired[~invalid], raw[~invalid]):
        raise AssertionError("valid raw readings changed")
    if not np.all(np.isfinite(repaired)):
        raise AssertionError("interpolation left invalid values")
    return repaired, invalid


def responder_events(
    values: np.ndarray, invalid: np.ndarray
) -> list[dict[str, object]]:
    onsets = _stream_event_onsets(values, TRIGGER_LEVEL_A, INTERVAL_S)
    if len(onsets) != EXPECTED_EVENTS:
        raise ValueError(f"expected {EXPECTED_EVENTS} onsets after interpolation, got {len(onsets)}")
    events: list[dict[str, object]] = []
    for ordinal, onset in enumerate(onsets):
        baseline_a, energy_mj = event_energy(values, onset)
        start, end = onset - PRE_SAMPLES, onset + POST_SAMPLES
        events.append(
            {
                "ordinal": ordinal,
                "onset_sample": onset,
                "warmup": ordinal < WARMUPS,
                "imputed_samples_in_window": int(np.count_nonzero(invalid[start:end])),
                "baseline_a": baseline_a,
                "energy_mj": energy_mj,
                "values_a": values[start:end],
            }
        )
    return events


def fill_sensitivity(raw: np.ndarray, invalid: np.ndarray) -> dict[str, object]:
    indices = np.arange(len(raw))
    previous_valid = np.maximum.accumulate(np.where(~invalid, indices, -1))
    next_valid = np.minimum.accumulate(
        np.where(~invalid, indices, len(raw))[::-1]
    )[::-1]
    variants = {
        "previous_neighbor": raw[previous_valid],
        "next_neighbor": raw[next_valid],
    }
    result: dict[str, object] = {}
    for label, values in variants.items():
        onsets = _stream_event_onsets(values, TRIGGER_LEVEL_A, INTERVAL_S)
        result[label] = {
            "candidate_onsets": len(onsets),
            "median_energy_mj": float(
                np.median([event_energy(values, onset)[1] for onset in onsets[WARMUPS:]])
            ),
        }
    return result


def representative(events: list[dict[str, object]], target_median: float) -> dict[str, object]:
    return min(events, key=lambda event: abs(float(event["energy_mj"]) - target_median))


def render(
    initiator: dict[str, list[dict[str, object]]], responder: list[dict[str, object]]
) -> None:
    measured_responder = responder[WARMUPS:]
    clean_responder = [
        event for event in measured_responder if event["imputed_samples_in_window"] == 0
    ]
    if not clean_responder:
        raise ValueError("no unimputed responder trace available")
    responder_center = float(np.median([event["energy_mj"] for event in measured_responder]))
    trace_events = [
        representative(initiator["local"][WARMUPS:], describe(initiator["local"])["median_energy_mj"]),
        representative(
            initiator["accepted_connected"][WARMUPS:],
            describe(initiator["accepted_connected"])["median_energy_mj"],
        ),
        representative(clean_responder, responder_center),
    ]
    groups = [
        [float(event["energy_mj"]) for event in initiator["local"][WARMUPS:]],
        [float(event["energy_mj"]) for event in initiator["accepted_connected"][WARMUPS:]],
        [float(event["energy_mj"]) for event in measured_responder],
    ]
    colors = ("#176da8", "#d17815", "#29785d")
    labels = ("Local", "Initiator", "Responder")
    figure = plt.figure(figsize=(3.25, 2.36))
    outer = figure.add_gridspec(2, 1, height_ratios=(1.12, 1), hspace=0.53)
    trace_ax = figure.add_subplot(outer[0])
    energy_axes = [figure.add_subplot(grid) for grid in outer[1].subgridspec(3, 1, hspace=0.62)]
    time_s = (np.arange(PRE_SAMPLES + POST_SAMPLES) - PRE_SAMPLES) * INTERVAL_S
    for event, color, label in zip(trace_events, colors, labels):
        delta_ma = 1000 * (event["values_a"] - event["baseline_a"])
        trace_ax.plot(time_s, delta_ma, linewidth=0.9, color=color, label=label)
    trace_ax.axvline(0, color="#60666d", linewidth=0.5, linestyle="--")
    trace_ax.axhline(0, color="#adb3b9", linewidth=0.4)
    trace_ax.set_xlim(-0.4, 4.08)
    trace_ax.set_ylabel(r"$\Delta I$ (mA)", fontsize=7)
    trace_ax.set_xlabel("Time from detected onset (s)", fontsize=7, labelpad=1)
    trace_ax.tick_params(labelsize=6.1, length=2, pad=1)
    trace_ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
        frameon=False, fontsize=5.8, handlelength=1.2,
    )
    trace_ax.spines[["top", "right"]].set_visible(False)

    for values, color, label, energy_ax in zip(groups, colors, labels, energy_axes):
        boxes = energy_ax.boxplot(
            [values], vert=False, positions=(0,), widths=0.42, whis=(5, 95),
            showmeans=False, patch_artist=True,
            flierprops={"marker": ".", "markersize": 1.1,
                        "markerfacecolor": "#60666d", "markeredgecolor": "none"},
            medianprops={"color": "white", "linewidth": 0.8},
            whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6},
        )
        boxes["boxes"][0].set_facecolor(color)
        boxes["boxes"][0].set_linewidth(0.6)
        if label == "Responder":
            boxes["boxes"][0].set_hatch("///")
        spread = max(values) - min(values)
        energy_ax.set_xlim(min(values) - 0.08 * spread, max(values) + 0.08 * spread)
        energy_ax.set_ylim(-0.42, 0.42)
        energy_ax.set_yticks(())
        energy_ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        energy_ax.tick_params(axis="x", labelsize=5.6, length=1.5, pad=0.5)
        energy_ax.text(-0.025, 0.5, label, transform=energy_ax.transAxes,
                       fontsize=6.0, ha="right", va="center")
        energy_ax.spines[["top", "right", "left"]].set_visible(False)
    energy_axes[-1].set_xlabel("Energy (mJ; separate scales)", fontsize=6.4, labelpad=1)
    figure.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.19)
    for suffix in ("pdf", "png"):
        figure.savefig(OUT_DIR / f"{OUTPUT_STEM}.{suffix}", dpi=350)
    plt.close(figure)


def main() -> None:
    if digest(RESP_RAW) != EXPECTED_RESPONDER_SHA256:
        raise ValueError("responder stream changed since this analysis was specified")
    initiator_raw = load_raw(INIT_RAW, "initiator")
    responder_raw = load_raw(RESP_RAW, "responder")
    initiator = load_initiator(initiator_raw)
    repaired, invalid = repair_responder(responder_raw)
    responder = responder_events(repaired, invalid)
    render(initiator, responder)
    measured = responder[WARMUPS:]
    summary = {
        "status": "exploratory_two_role_figure_with_interpolated_responder",
        "raw_initiator_sha256": digest(INIT_RAW),
        "raw_responder_sha256": digest(RESP_RAW),
        "initiator_samples_sha256": digest(INIT_SAMPLES),
        "raw_unchanged": True,
        "responder_imputation": "linear interpolation between nearest valid prior and next raw reading at fixed 20-ms time slots",
        "invalid_responder_readings": int(np.count_nonzero(invalid)),
        "responder_raw_readings": len(responder_raw),
        "candidate_responder_onsets_after_imputation": len(responder),
        "warmups_excluded_per_path": WARMUPS,
        "measured_responder_windows_with_imputation": sum(
            event["imputed_samples_in_window"] > 0 for event in measured
        ),
        "imputed_slots_inside_measured_responder_windows": sum(
            int(event["imputed_samples_in_window"]) for event in measured
        ),
        "source_voltage_v": VOLTAGE_V,
        "source_voltage_status": "verified in initiator manifest; assumed equal for responder without responder manifest",
        "sample_interval_s": INTERVAL_S,
        "pre_event_s": PRE_SAMPLES * INTERVAL_S,
        "post_event_s": POST_SAMPLES * INTERVAL_S,
        "local": describe(initiator["local"]),
        "initiator": describe(initiator["accepted_connected"]),
        "responder_candidate": {
            **describe(responder),
            "clean_candidate_median_energy_mj": float(np.median([
                event["energy_mj"] for event in measured
                if event["imputed_samples_in_window"] == 0
            ])),
            "imputed_candidate_median_energy_mj": float(np.median([
                event["energy_mj"] for event in measured
                if event["imputed_samples_in_window"] > 0
            ])),
        },
        "responder_fill_sensitivity": fill_sensitivity(responder_raw, invalid),
        "responder_events": [
            {key: value for key, value in event.items() if key != "values_a"}
            for event in responder
        ],
        "limitations": [
            "responder readings and some onset detections depend on interpolation",
            "no completed responder capture manifest or exact UART outcome join",
            "sequential recordings are not synchronized and do not support pair peak or pair energy",
            "responder supply voltage is assumed equal to 3.7 V rather than documented in a responder manifest",
            "power firmware revision differs from archived timing and resource build",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fig4a_interpolated_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "responder_events"}, indent=2))


if __name__ == "__main__":
    main()
