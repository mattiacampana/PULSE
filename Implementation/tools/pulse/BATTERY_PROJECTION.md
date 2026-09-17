# Battery-life projection

`battery_projection.py` computes a daily energy budget and battery-life projection from explicitly
supplied assumptions. It does not provide battery, power, workload, or efficiency defaults and does
not infer energy from firmware timing or byte counters.

The calculation is

\[
E_{\mathrm{day}}=
\sum_s t_s P_s +
\sum_e N_e\,\overline{E}_{e,\mathrm{inc}},
\]

\[
E_{\mathrm{bat}}=3.6\,C_{\mathrm{mAh}}V_{\mathrm{nom}}\eta,
\qquad
L_{\mathrm{days}}=E_{\mathrm{bat}}/E_{\mathrm{day}}.
\]

The state durations must sum to exactly 86,400 seconds. Event energy is incremental over the
state-specific baseline, so the state energy covers the full day and the incremental event terms are
then added without counting the baseline twice.

## Required input

The input is JSON with schema `senswear.pulse.battery_projection.input.v1`. It must contain all of
the following:

- Nonempty measured incremental-energy sample arrays in joules for every event class.
- Exactly three named sensitivity cases: `low`, `nominal`, and `high`.
- A nonempty description of what each sensitivity case means.
- Explicit battery capacity in mAh, nominal voltage in V, and usable-energy efficiency in `(0,1]`
  for every case.
- A complete 24-hour state ledger. Every state requires duration in seconds per day and its
  state-specific idle power in W.
- An explicit daily count for every measured event class in every case, including zeros.

The required structure is:

```text
{
  "schema": "senswear.pulse.battery_projection.input.v1",
  "incremental_event_energy_samples_j": {
    "<event class>": [<measured joules>, ...]
  },
  "assumptions": {
    "low": {
      "description": "<what is low in this sensitivity case>",
      "battery": {
        "capacity_mAh": <required>,
        "nominal_voltage_v": <required>,
        "efficiency": <required>
      },
      "states": {
        "<state>": {
          "duration_s_per_day": <required>,
          "idle_power_w": <required>
        }
      },
      "event_counts_per_day": {
        "<event class>": <required>
      }
    },
    "nominal": { <same complete structure> },
    "high": { <same complete structure> }
  }
}
```

The three cases must use identical state and event names. The names `low` and `high` do not impose
an ordering: the output records which explicit case produces the shortest and longest life.

## Run

From `tools/pulse`:

```powershell
python .\pulse_tools.py project-battery-life `
  --input-json .\battery_projection_input.json `
  --output-json .\battery_projection_result.json `
  --output-csv .\battery_projection_result.csv
```

The module also remains directly executable with the same three path arguments, omitting the
`project-battery-life` subcommand.

The JSON contains the state and event contributions, measured event-energy statistics, all three
scenario results, and the sensitivity envelope. The CSV contains one summary row per scenario.

## Sensitivity is not sampling uncertainty

The span across `low`, `nominal`, and `high` is scenario sensitivity. It is not a confidence
interval and must not be described as measurement uncertainty.

For each event class, the projection uses the arithmetic mean of the supplied incremental-energy
samples. It separately reports the sample standard deviation and standard error of that mean. The
daily event-energy standard error combines event-class contributions in quadrature, assuming the
measured event classes are independent. Battery-life standard error uses first-order propagation
through `L=E_bat/E_day`.

This sampling calculation excludes uncertainty in state power, state durations, event counts,
battery capacity, nominal voltage, and efficiency. Those quantities belong in the explicit
sensitivity cases or require a separate uncertainty model. If a used event class has fewer than two
samples, sampling uncertainty is reported as unavailable rather than as zero.

## Interpretation limits

- Use arithmetic mean incremental event energy for the expected daily budget. Median, quartile, or
  p95 plug-in calculations are sensitivity analyses, not expected energy and not confidence bounds.
- Do not substitute CPU cycles, UART byte counts, Bluetooth application payload, a fuel-gauge state
  of charge estimate, or datasheet typical current for measured event energy without labeling the
  result as a separate modeled assumption.
- Do not claim link-layer or radio energy from application/GATT-value counters.
- Battery capacity and efficiency depend on load, temperature, aging, cutoff voltage, and the actual
  cell. The three explicit cases should document the evidence used for each value.
