# PULSE on-device experiment tools

This directory turns **measured** SensWear firmware records, NGMO2 or sequential Keysight current
captures, and release-image linker artifacts into the tables required by Section V of
`PERCOM2026/main.tex`. The scripts never fill a missing measurement with an estimate or a synthetic
number.

The analysis path is Python 3.10+ and standard-library only. `matplotlib` is optional and is used
only to render PDF/PNG; every plotting input is also exported as a figure-ready CSV. NumPy is not
required. Live NGMO2 control additionally needs PyVISA plus a separately installed, approved
R&S/IVI VISA runtime (PyVISA itself is not an instrument driver):

```powershell
python -m pip install pyvisa
```

## Figure 4a with one Keysight 34465A

Use `docs/KEYSIGHT_34465A_FIG4A.md` for the exact wiring, firmware overlay,
same-image UART preflight, two sequential role captures, safety gates, and
manuscript limitations. The two commands added for that workflow are:

```powershell
python .\pulse_tools.py keysight-capture --help
python .\pulse_tools.py keysight-fig4a --help
```

`keysight-capture` continuously streams timed 34465A current readings and
extracts fixed-size event windows using a pilot-tested threshold in software,
without GPIO access. The DMM's internal level-trigger mode remains available
only with `--capture-mode level` for diagnosis. `keysight-fig4a`
requires complete initiator and
responder CSVs plus the preflight event/metadata CSVs. It writes the audited
per-event energy tables, representative event-aligned traces, and
`ondevice_power.pdf/.png`. The pair value is explicitly a sum of sequential
role measurements; the tool never fabricates a simultaneous pair peak.

## Primary NGMO2 workflow (debug boards disconnected)

Power and UART measurements are separate campaigns. During every NGMO2 current capture, remove both
LiPo cells and disconnect both debug boards **and their FPC cables**. The autonomous firmware runs
one configured event per power cycle and writes its proof to internal nonvolatile storage only after
the acquisition. Reattach a debug board later, with the analyzer outputs off, to export that proof.
Do not describe an operator-declared capture ID as an exact firmware join until the recovered proof
passes the schedule/provenance audit.

The NGMO2 is the only supply and current-measurement instrument: connect channel 1/A `FORCE+` and
`FORCE-` to board 1 `PACK+` and `PACK-`, respectively, and connect channel 2/B identically to board
2. Verify terminal and pad labels rather than inferring polarity from cable color. Keep the
front-panel local-sense jumpers installed; this final protocol does not attach remote-sense leads.
The reported supplied energy therefore includes the loss in the separately recorded output
leads. No series DMM, Keithley, Keysight, battery, debug board, or debug FPC belongs in the measured
circuit. Assign the logical initiator with `--initiator-channel`, record cable IDs/length/gauge, then
swap that channel/cable assignment in counterbalanced blocks. Record lead length as the total
`FORCE+` plus `FORCE-` loop length for each channel. During accepted-path trials, power only the
intended pair and disable or remove every other compatible SensWear board from the RF test area.

Build a separate autonomous image for each event path. From the firmware repository root, append
the autonomous and one path overlay to the same operator-filled lab configuration used for the
release image (replace the path overlay for the other three paths):

```powershell
$ngmoAutonomous = (Resolve-Path .\config\pulse_ngmo2_autonomous.conf).Path
$ngmoPath = (Resolve-Path .\config\pulse_ngmo2_path_accepted_connected.conf).Path
$ngmoExtra = "$labConfig;$ngmoAutonomous;$ngmoPath"

cmake --preset senswear_pulse --fresh `
  "-DPULSE_FINAL_CAPTURE=ON" `
  "-DPULSE_ARTIFACT_NPZ=$artifact" `
  "-DEXTRA_CONF_FILE=$ngmoExtra"
cmake --build --preset senswear_pulse
```

Archive the exact image hash before flashing. After all power captures, rebuild with
`config/pulse_ngmo2_export.conf` appended instead of the autonomous no-console overlay, and recover
the stored proofs only with NGMO2 outputs off. Never use the export image for measured current.

Discover or probe the VISA resource exposed by the USB adapter without changing either output.
The NGMO2 manual documents RS-232 and GPIB, so the normal resource is `ASRL...::INSTR` for a
USB-to-serial adapter or `GPIB...::INSTR` for a USB-to-GPIB controller. A bridge may instead expose
`USB...::INSTR`:

```powershell
python .\pulse_tools.py ngmo2-capture --list-resources
python .\pulse_tools.py ngmo2-capture `
  --resource '<ASRL...::INSTR, GPIB...::INSTR, or exposed USB...::INSTR>' --probe
```

If the USB connection enumerates as `ASRL...::INSTR`, copy every serial setting from the NGMO2
front-panel menu; the tool intentionally assumes none of them:

```powershell
python .\pulse_tools.py ngmo2-capture `
  --resource '<ASRL...::INSTR>' --probe `
  --visa-baud '<menu value>' --visa-data-bits '<7 or 8>' `
  --visa-parity '<none, odd, or even>' --visa-stop-bits '<one or two>' `
  --visa-flow-control '<none, rts_cts, or xon_xoff>' `
  --visa-termination '<cr or lf>'
```

The primary dynamic capture uses the NGMO2 0.5 A range and synchronized interval-average arrays.
The defaults `--sample-interval-s 0.02 --sample-count 5000` make a 100 s buffer. With firmware
dispatch at 45 s after power-on, this example triggers at 25 s, requires completion by 65 s,
captures until approximately 125 s, then keeps power on for another 15 s so the 135 s proof-
persistence deadline can complete:

```powershell
python .\pulse_tools.py ngmo2-capture --autonomous `
  --resource '<verified VISA resource>' `
  --output-dir .\capture\accepted-connected-r001 `
  --run-id 'accepted-connected-r001' --pair-id 'p01' `
  --initiator-board-id '<board A ID>' --responder-board-id '<board B ID>' `
  --initiator-firmware-sha256 '<exact image SHA-256>' `
  --responder-firmware-sha256 '<exact image SHA-256>' `
  --firmware-revision '<exact embedded revision>' `
  --artifact-sha256 '<exact deployed model artifact SHA-256>' `
  --initiator-channel 1 --event-path accepted_connected --capture-count 105 `
  --firmware-sequence-start 0 `
  --autonomous-trigger-delay-s 25 `
  --autonomous-event-dispatch-s 45 `
  --autonomous-event-deadline-s 65 `
  --boot-schedule-uncertainty-s $preflightBootScheduleBoundS `
  --autonomous-post-capture-s 15 `
  --voltage-v $approvedCellEquivalentVoltage `
  --current-limit-a $approvedCurrentLimit `
  --current-limit-rejection-margin-a $approvedLimitMargin `
  --maximum-safe-voltage-v $cellMaximumVoltage `
  --overvoltage-protection-v $approvedOvpThreshold `
  --max-output-voltage-deviation-v $approvedVoltageTolerance `
  --sense-mode local `
  --channel-1-cable-id 'cable-01' --channel-2-cable-id 'cable-02' `
  --channel-1-lead-length-m $channel1LeadLengthM `
  --channel-2-lead-length-m $channel2LeadLengthM `
  --channel-1-lead-gauge-awg $channel1Awg `
  --channel-2-lead-gauge-awg $channel2Awg `
  --calibration-id $ngmoCalibrationId --calibration-date $ngmoCalibrationDate `
  --confirm-calibration-current `
  --confirm-output-impedance-zero `
  --confirm-debug-boards-disconnected --confirm-no-parallel-battery `
  --enable-outputs
```

The controller requests OFF after each capture; the operator must verify both channels OFF on the
front panel after every run, error, or USB loss before touching leads. Supply voltage/current limits are intentionally mandatory;
the tool has no battery-specific default. The OVP threshold must be a documented 0.1 V step in
the 1.5--22 V range, above the programmed voltage, and no greater than the declared safe maximum.
The tool programs and reads back `VOLTage:MAXSetting`, `VOLTage:PROTection`, zero output impedance,
independent channel control, and both the current-limit safety cap and `LIMIT` mode. Each cycle is
anchored before the first
sequential output-on command, waits only until the absolute trigger deadline, and records the
actual trigger bracket and output-verification span. Those values produce conservative, per-capture
pre-dispatch and post-completion guard bounds. The manifest remains
`not_joined_capture_only_operator_schedule` until recovered firmware proofs are joined.

With analyzer outputs off, attach one debug board at a time, flash the export-only image, and save
the complete UART output. Parse both logs and perform the proof join:

```powershell
python .\pulse_tools.py serial .\export\board-1.log .\export\board-2.log `
  --output-dir .\recovered-proofs

python .\pulse_tools.py ngmo2-autonomous-power `
  --manifest .\capture\accepted-connected-r001\ngmo2_capture_manifest.csv `
  --proofs-csv .\recovered-proofs\capture_proofs.csv `
  --capture-status-csv .\recovered-proofs\capture_status.csv `
  --build-context-csv .\recovered-proofs\run_metadata.csv `
  --output-dir .\results\accepted-connected-r001
```

The export-only firmware emits a complete `PULSE_META` record before the stored proofs without
starting sensing, BLE, protocol, or model work. Its role and initial link-observation fields are
therefore explicitly unresolved/unavailable. `--build-context-csv` is the two-board
`run_metadata.csv` parsed from those matching exporter logs. The analyzer recomputes its context
digest, requires both board identities, binds the
capture manifest's firmware revision and model artifact digest to it, and copies that complete
contract into every accepted power and audit row. Preserve that exact CSV and include its directory
when assembling figures.

The parser preserves both `firmware_sequence` and the legacy `sequence` alias; both are zero-based
within an event path. The join key is path + firmware sequence + board identity/role, never the
operator-declared `event_index`. It also binds the proof's FNV-1a firmware/artifact fingerprints to
the exact revision and artifact SHA-256 recorded at capture time. Coherent failed attempts remain
in the `outcome=failure` stratum. For accepted attempts, a responder event proof is mandatory once
the initiator says a remote session started. The final audit requires both boards to export exactly
sequences 0--104 for every path, including failures before a remote session starts. The full firmware
image SHA-256 values are capture-time provenance; the recovered proof directly binds only the
embedded firmware-revision and artifact FNV-1a fingerprints. Missing, duplicate, corrupt, incoherent, or
fingerprint-mismatched proofs produce no energy row.

NGMO2 current array values are interval averages. Absolute supplied energy is
`Vmean * dt * sum(I[n])`; incremental full-capture energy subtracts the equal-weight mean of the
pre-dispatch and post-completion guard currents from every interval. `average_power_w` is absolute
capture energy divided by full capture duration. Endpoint peaks use all returned bins, and pair
peak is the maximum simultaneous `V1*I1[n] + V2*I2[n]`, never a sum of independent maxima.

For the battery-life background, select `low` (5 mA) or `medium` (0.5 A) explicitly after a pilot,
using the lowest range that does not overload on BLE spikes. Capture each
connected/disconnected/advertising state explicitly;
the two channel queries are sequential and are not labeled synchronized:

```powershell
python .\pulse_tools.py ngmo2-capture --static-idle `
  --resource '<verified VISA resource>' --output-dir .\capture\idle-connected `
  --run-id 'idle-connected-r001' --pair-id 'p01' `
  --initiator-board-id '<board A ID>' --responder-board-id '<board B ID>' `
  --initiator-firmware-sha256 '<64 hex digits>' `
  --responder-firmware-sha256 '<64 hex digits>' `
  --initiator-channel 1 --channel-1-state connected_idle `
  --channel-2-state connected_idle --idle-samples 30 `
  --idle-current-range medium --idle-overload-threshold-a 0.510 `
  --idle-settle-s 10 --idle-measure-interval-s 0.02 --idle-average-count 5 `
  --idle-state-evidence '<state-hold image/preflight log hash>' `
  --voltage-v $approvedCellEquivalentVoltage `
  --current-limit-a $approvedCurrentLimit `
  --current-limit-rejection-margin-a $approvedLimitMargin `
  --maximum-safe-voltage-v $cellMaximumVoltage `
  --overvoltage-protection-v $approvedOvpThreshold `
  --max-output-voltage-deviation-v $approvedVoltageTolerance `
  --sense-mode local `
  --channel-1-cable-id 'cable-01' --channel-2-cable-id 'cable-02' `
  --channel-1-lead-length-m $channel1LeadLengthM `
  --channel-2-lead-length-m $channel2LeadLengthM `
  --channel-1-lead-gauge-awg $channel1Awg `
  --channel-2-lead-gauge-awg $channel2Awg `
  --calibration-id $ngmoCalibrationId --calibration-date $ngmoCalibrationDate `
  --confirm-calibration-current `
  --confirm-output-impedance-zero `
  --confirm-debug-boards-disconnected --confirm-no-parallel-battery `
  --enable-outputs
```

Prefer a dedicated, proofed no-event state-hold image for static idle. If an autonomous image is
used, also pass its earliest `--idle-event-dispatch-s`; the tool conservatively budgets two
sequential channels times interval times average count times sample count and fails before enabling
outputs when settling plus sampling would overlap the event. State names alone are never treated as
evidence that firmware reached the requested states. Static power uses each channel's mean pre/post
local-sense NGMO2 output-voltage readback, not the programmed setpoint. The output records the chosen
range's limit, resolution, and full-scale deviation. For battery projection, copy `mean_power_w`
from the matching board/state row of `ngmo2_idle_summary.csv` into that state's `idle_power_w` field;
copy measured `energy_incremental_j` values from `event_power_metrics.csv` into the corresponding
event-class sample array. Preserve the run, board, state, and path provenance when assembling JSON.

The optional `--uart-paced` READY/GO/DUMP mode is diagnostic only. It requires proven physical
isolation of the debug-board VBUS-to-`CHG_AC` path; ordinary attached debug boards are not power-
measurement-transparent. Run latency, software-stage timing, correctness, runtime watermarks, and
firmware communication counters as a separate UART campaign. A phone BLE scanner may be used for
bring-up, but it is not evidence for link-layer bytes, packets, retransmissions, or PCAP claims.

## Legacy optional GPIO/sniffer workflow (not used for the NGMO2-only experiment)

The commands below document the optional legacy GPIO/power and hardware-sniffer profile. They are
not required by the primary NGMO2/no-sniffer experiment above.

Run commands from `firmware/tools/pulse` or invoke `pulse_tools.py` by its full path.

```powershell
# Replace these examples with the exact lab settings for this capture block.
$supplyVoltageV = 3.8
$analyzerSampleRateHz = 100000
$analyzerFilter = 'none'
$deviceSeparationM = 0.5

python .\pulse_tools.py serial board_a.log board_b.log `
  --set pair_id=p01 --set run_id=release-r01 `
  --set "supply_voltage_v=$supplyVoltageV" `
  --set "analyzer_sample_rate_hz=$analyzerSampleRateHz" `
  --set "analyzer_filter=$analyzerFilter" `
  --set "device_separation_m=$deviceSeparationM" `
  --output-dir results
python .\pulse_tools.py serial baseline_board_a.log baseline_board_b.log `
  --set pair_id=p01 --set run_id=baseline-r01 `
  --set "supply_voltage_v=$supplyVoltageV" `
  --set "analyzer_sample_rate_hz=$analyzerSampleRateHz" `
  --set "analyzer_filter=$analyzerFilter" `
  --set "device_separation_m=$deviceSeparationM" `
  --output-dir baseline-results
python .\pulse_tools.py serial correctness_board_a.log correctness_board_b.log `
  --set pair_id=p01 --set run_id=correctness-r01 --output-dir correctness-results

python .\pulse_tools.py power `
  --input initiator=.\capture\baseline-initiator.csv `
  --input responder=.\capture\baseline-responder.csv `
  --metadata-csv .\baseline-results\run_metadata.csv `
  --baseline-only --voltage-v $supplyVoltageV `
  --output-dir baseline-results

python .\pulse_tools.py power `
  --input .\capture\pair.csv `
  --events-csv .\results\event_metrics.csv `
  --metadata-csv .\results\run_metadata.csv `
  --baseline-metadata .\baseline-results\power_capture_metadata.csv `
  --voltage-v $supplyVoltageV `
  --output-dir results

python .\pulse_tools.py link `
  --input .\capture\canonical-pdus.csv `
  --event-windows-csv .\results\event_power_metrics.csv `
  --metadata-csv .\results\run_metadata.csv `
  --raw-pcap cap01=.\capture\untouched.pcapng `
  --shared-timebase --time-uncertainty-s 0.000005 `
  --output-dir results

python .\pulse_tools.py resources `
  --map pulse=..\..\build\pulse\release\zephyr\zephyr.map `
  --map baseline=..\..\build\pulse\baseline\zephyr\zephyr.map `
  --elf pulse=..\..\build\pulse\release\zephyr\zephyr.elf `
  --elf baseline=..\..\build\pulse\baseline\zephyr\zephyr.elf `
  --elf correctness=..\..\build\pulse\correctness\zephyr\zephyr.elf `
  --image-metadata pulse=.\results\run_metadata.csv `
  --image-metadata baseline=.\baseline-results\run_metadata.csv `
  --image-metadata correctness=.\correctness-results\run_metadata.csv `
  --correctness-input correctness=.\correctness-results\correctness.csv `
  --nm pulse=.\capture\pulse.nm `
  --nm baseline=.\capture\baseline.nm `
  --runtime-watermarks .\capture\runtime_ram.csv `
  --output-dir results

python .\pulse_tools.py summarize --input-dir results --output-dir results
python .\pulse_tools.py project-energy `
  --counts .\capture\per_node_event_counts.csv `
  --power-summary .\results\power_summary.csv `
  --output-dir results
python .\pulse_tools.py figures `
  --profile instrumented `
  --data-dir pooled-results --data-dir results --data-dir baseline-results `
  --data-dir correctness-results --output-dir figures --formats pdf,png
```

For the primary NGMO2 profile, first pool the four analyzed power-path directories with the UART
timing/communication results, then pass both the pooled summaries and every raw evidence directory.
Each recovered-proof directory below contains the exact exporter-derived `run_metadata.csv` supplied
to its corresponding `ngmo2-autonomous-power` invocation; `timing-results` contains the separate UART
evidence:

```powershell
python .\pulse_tools.py summarize `
  --input-dir .\timing-results `
  --input-dir .\results\local-r001 `
  --input-dir .\results\no-contact-r001 `
  --input-dir .\results\accepted-connected-r001 `
  --input-dir .\results\accepted-discovery-r001 `
  --output-dir .\results\ngmo2-pooled

python .\pulse_tools.py figures `
  --profile ngmo2 `
  --data-dir .\results\ngmo2-pooled `
  --data-dir .\timing-results `
  --data-dir .\recovered-proofs\local-r001 `
  --data-dir .\recovered-proofs\no-contact-r001 `
  --data-dir .\recovered-proofs\accepted-connected-r001 `
  --data-dir .\recovered-proofs\accepted-discovery-r001 `
  --data-dir .\results\local-r001 `
  --data-dir .\results\no-contact-r001 `
  --data-dir .\results\accepted-connected-r001 `
  --data-dir .\results\accepted-discovery-r001 `
  --data-dir .\resource-results `
  --data-dir .\correctness-results `
  --output-dir .\figures\ngmo2 --formats pdf,png
```

The NGMO2 figure gate rejects missing or altered build-context archives, mismatched copied context,
unbound summaries, failed capture audits, and diagnostic UART-paced power data.

Use `--formats none` when matplotlib is not installed. The `figures` command still writes the
figure-ready CSVs. `python pulse_tools.py all ...` combines the same stages; see `--help` for its
arguments.

The summarizer always materializes and checks these four required initiator-side strata:
`local/local/connected`, `no_contact/local/disconnected`,
`accepted_connected/initiator/connected`, and
`accepted_discovery/initiator/disconnected`. It returns a non-zero status if any is absent or has
fewer than 100 post-warm-up attempts **within any individual
`run_id,pair_id,initiator-board` capture**, if a required label/key is missing, or if
accepted-session pairing is inconsistent. Two pairs with 50 trials each never satisfy a 100-trial
minimum. `protocol_audit.csv` is written first. `--allow-underfilled` can waive only
a nonzero pilot sample below the requested repetition count; it cannot waive an absent stratum,
missing labels/keys, duplicates, or orphan/missing responder records.

## Firmware serial contract

The parser finds these prefixes anywhere in a log line, so Zephyr timestamps and log-level text may
precede them:

- `PULSE_META` -> `run_metadata.csv`
- `PULSE_EVENT` -> `event_metrics.csv`
- `PULSE_STAGE` -> `stage_metrics.csv`
- `PULSE_CORRECTNESS` -> `correctness.csv`

The canonical payload is comma-separated `key=value`. Unknown keys are retained as new CSV columns,
which keeps the wire log schema evolvable.

```text
PULSE_META,run_id=2026-09-03-p01,pair_id=p01,board_id=105012345,role=initiator,cpu_hz=128000000
PULSE_EVENT,event_index=27,trial_id=17,exchange_id=117,remote_session_started=1,role=initiator,event_path=accepted_connected,connection_state_start=connected,warmup=0,success=1,duration_us=28431
PULSE_STAGE,event_index=27,warmup=0,trial_id=17,role=initiator,event_path=accepted_connected,stage=agreement,duration_us=913,cycles=116864
PULSE_CORRECTNESS,board_id=105012345,pass=1,local_gradient_cosine=0.99987,remote_gradient_cosine=0.99991,updated_head_max_abs_error=1.7e-5
```

Metadata context (`run_id`, `pair_id`, `board_id`, role, toolchain/model settings) is carried into
later records in the same input file when a field is omitted. Emit a new complete `PULSE_META` row
whenever that context changes.

If a campaign identifier is intentionally external to the firmware build, attach it to every parsed
row with repeatable `--set KEY=VALUE` arguments. These are explicit final overrides. In particular,
use `--set pair_id=p01` when the image still reports `pair_id=operator_required`; do not infer pair
independence from board filenames.

Headered positional CSV is also accepted:

```text
PULSE_EVENT,header,event_index,trial_id,role,event_path,connection_state_start,warmup,success,duration_us
PULSE_EVENT,27,17,initiator,accepted_connected,connected,0,1,28431
```

A header line without the literal `header` is recognized when at least two known field names are
present. Headerless positional records use the documented lists in `serial_parser.py`; they are less
safe than key/value records and should not be used for the final campaign.

Recommended required fields are:

- Every event/stage record: `run_id`, `pair_id`, `board_id`, `role`, and `trial_id` (directly or by
  metadata context).
- Event: monotonic `event_index` (or `marker_event_index`), `event_path`, `connection_state_start`,
  `warmup`, explicit `success`, `failure_reason`, duration, and byte counters.
- Accepted events: a non-placeholder `exchange_id` on both roles and an explicit initiator
  `remote_session_started=0|1`; every responder must explicitly emit `1`. The exact
  `(run_id,pair_id,exchange_id)` tuple is the only accepted
  session-pair key. A responder record is required exactly once when that flag is 1 and prohibited
  when it is 0.
- Stage: matching `event_index`/`warmup`, unambiguous `stage`, `duration_us`, and `cycles`. The
  parser propagates `remote_session_started` from exactly one same-run/pair/board/event record.
  `duration_us`/`wall_duration_us` are GRTC wall time; `cycles` and `cycle_duration_us` retain the
  elapsed DWT counter diagnostic and are not exclusive thread CPU time.
- Correctness: local/remote gradient cosine similarity, updated-head maximum absolute parameter
  error, explicit pass/fail, fixture hash, reference identity, tensor precision, and concrete
  `firmware_revision` inherited from `PULSE_META`.

`rejected` and `skipped` are successful executions of those decision paths, not transport failures.
Emit `success=0` for a genuine timeout/error; an explicit success field takes precedence over
`status` during aggregation. Keep `event_path` equal to the attempted path when it fails (for example,
`accepted_discovery,success=0,failure_reason=timeout`) so its failure-rate denominator remains valid;
use a generic `failed` path only for a deliberately separate failure microbenchmark.

## Power analyzer and GPIO contract

The four daughter-header markers are sampled by the analyzer alongside current:

| Function | Daughter GPIO | nRF54L15 pin | Encoding |
|---|---:|---:|---|
| Event gate | GPIO0 | P1.14 | high for exactly one measured event |
| Stage bit 0 | GPIO1 | P1.13 | least-significant bit |
| Stage bit 1 | GPIO2 | P1.12 | middle bit |
| Stage bit 2 | GPIO3 | P1.11 | most-significant bit |

The resulting 3-bit values are:

| Code | Firmware meaning |
|---:|---|
| 0 | event/unspecified |
| 1 | selection |
| 2 | encoder |
| 3 | head forward/backward |
| 4 | serialization/deserialization |
| 5 | radio transfer |
| 6 | agreement/normalization/mixing/utility |
| 7 | BLE control/scan/connect |

The firmware repeats this table as `marker_stage_0` ... `marker_stage_7` in `PULSE_META`; metadata is
authoritative, and `--stage CODE=NAME` is the final override. Event type, connection state, role, and
trial number are deliberately **not** encoded in the GPIOs. They come from `PULSE_EVENT` and are
joined to marker windows only when one concrete `run_id,pair_id,board_id` has unique contiguous
`event_index`/`marker_event_index` values and the window count is exact. There is no chronological
fallback. Pass parsed `run_metadata.csv` for board-role mapping. The
initiator board's `role=local` local/no-contact events are associated by immutable `board_id`; output
rows keep their logical `role=local` and record the physical analyzer source as
`power_channel_role=initiator`. This mapping is never applied to the responder channel.

The preferred synchronized two-channel analyzer CSV is:

```text
time_s,initiator_current_a,responder_current_a,initiator_gpio0,initiator_gpio1,initiator_gpio2,initiator_gpio3,responder_gpio0,responder_gpio1,responder_gpio2,responder_gpio3
```

`initiator_event_gate`, `initiator_stage_bit0`, etc. are equivalent names. `p1_14`, `p1_13`,
`p1_12`, and `p1_11` are accepted for a single channel. A varying `voltage_v` (or role-prefixed
voltage) may be included. Otherwise pass the analyzer's fixed supply using `--voltage-v`.

Separate synchronized files are accepted as well:

```powershell
python .\pulse_tools.py power `
  --input initiator=.\capture\initiator.csv `
  --input responder=.\capture\responder.csv `
  --events-csv .\results\event_metrics.csv `
  --metadata-csv .\results\run_metadata.csv `
  --voltage-v 3.8 --output-dir results
```

Treat one `power` invocation as one synchronized capture block. For repeated analyzer exports, write
each block to its own result directory and pass every raw block directory to repeatable
`summarize --input-dir`, writing pooled summaries to a separate directory. For plotting, pass that
pooled directory plus the raw directory containing the chosen representative trace. Do not feed both
per-block and pooled summary CSVs to `figures`, which would duplicate aggregate rows. This preserves
block-specific baselines and avoids ambiguous event ordering.

Generic columns in a role-labelled file are `time_s,current_a,event_gate,stage_bit0,stage_bit1,stage_bit2`.
A single unlabelled file must use role-prefixed current and marker columns. Markers are active-high by
default; `--active-low` applies to all four marker signals.

For each gate window, the analyzer tool writes:

- `event_power_metrics.csv`: duration, charge, average/peak current and power, total energy, and
  baseline-subtracted energy;
- `stage_power_metrics.csv`: the same power statistics for every contiguous stage-code interval;
- `power_trace.csv`: joined sample-level current/power/stage data for the trace panel;
- `power_capture_metadata.csv`: observed sample rate, time span, voltage range, and baseline method
  for each analyzer channel.

Across repeated capture blocks, `summarize` writes `baseline_power_summary.csv`; its
`baseline_power_w` row supplies the measured idle/sensing baseline statistic for the discussion.

Gate and stage transitions are placed halfway between adjacent digital samples. Current/power is
linearly interpolated at the resulting boundary and trapezoid-integrated. A window touching the
capture edge is flagged `truncated_start`/`truncated_end`; do not use truncated events in final data.
The sample-level trace retains 5 ms of gate-low context on each side by default; change this only for
plot readability with `--trace-padding-ms` (it does not change any integration window).

Paper-facing event analysis requires a separate matched all-gates-low capture. Run it with
`--baseline-only`, then pass its archived `power_capture_metadata.csv` as `--baseline-metadata`.
The analyzer loads each role's value, records the metadata SHA-256 and raw-capture source, and rejects
a missing/duplicate role. Manual `--baseline-power-w initiator=...` values remain available but are
marked `baseline_source=manual_numeric`; final figure coverage accepts them only when the same role
and value occur in an included baseline-only capture. `--allow-gate-low-baseline-diagnostic` is an
explicitly non-paper override that uses the event trace's gate-low sample median. Incremental energy
is exactly

```text
E_incremental = integral(P(t), t0..t1) - P_baseline * (t1 - t0).
```

For a matched-baseline image that deliberately emits no event gates, `--baseline-only` calculates
the authoritative baseline as trapezoidal energy over the complete capture divided by capture
duration. This retains periodic sensing/BLE spikes. The instantaneous sample median is emitted only
as `diagnostic_gate_low_sample_median_power_w`. The command writes
`capture_mode=baseline_only`, capture properties, and the integrated-mean method to
`power_capture_metadata.csv`, while leaving event/stage/trace outputs empty. It rejects any asserted
gate. Event captures are separately labeled `events_explicit_matched_baseline` or
`events_diagnostic_intrace_baseline`.

For synchronized initiator/responder channels, accepted events are paired only by exact
`run_id,pair_id,exchange_id`. When `remote_session_started=1`, the paper-facing `role=initiator` and
`role=responder` rows are each integrated over the **same union** of both GPIO windows; their total
and incremental energies therefore sum exactly to the `role=pair,measurement_scope=pair_union` row.
Each board's original own-gate integration remains available only as
`role=<role>_own_gate_diagnostic,measurement_scope=own_gate_diagnostic`. Stage-power rows are
explicitly `measurement_scope=own_gate_stage` and retain `stage_sequence`, so repeated occurrences
of one marker code are not pooled. When an accepted
attempt fails before the remote session starts, the responder correctly has no event record or gate;
the tool emits an `initiator_only_pre_session_failure_exact_exchange_id` system row only if the
responder analyzer channel covers the complete initiator window, and integrates both channels over
that window. Missing power for a started remote session, duplicates, and responder orphans are hard
errors even with `--allow-unmatched`.

The command always rejects partial joins, duplicate/gapped indices, mixed run/pair/board identities,
or a different number of serial events and GPIO windows. `--allow-unmatched` can retain a wholly
unlabelled window for diagnosis; it never authorizes an ambiguous partial chronological join.

## BLE sniffer and canonical PDU contract

`pulse_tools.py link` accounts measured link traffic but deliberately does **not** parse PCAP or a
vendor-specific sniffer export. Preserve the untouched `.pcap`/`.pcapng`, inspect it with the
appropriate decoder, filter it to PDUs attributable to the captured physical pair, and export one
row per on-air BLE LL PDU using this exact canonical header:

```text
capture_id,run_id,pair_id,packet_index,timestamp_s,capture_start_s,capture_end_s,direction,transmitter_address,transmitter_address_type,receiver_address,receiver_address_type,pdu_kind,llid,length_bytes,crc_ok,gap_before,is_retransmission,connection_epoch,sn,nesn,payload_sha256
```

The fields have intentionally strict meanings:

- `packet_index` is unique and strictly increasing in capture order; timestamps may be equal but
  may not move backwards. `capture_start_s` and `capture_end_s` are the full acquisition bounds in
  the sniffer clock and are repeated unchanged for every row.
- `direction` is exactly `initiator_to_responder` or `responder_to_initiator`. Transmitter and
  receiver addresses are lowercase colon-separated 48-bit BLE identities, with type `public`,
  `random`, `random_static`, `random_resolvable`, or `random_non_resolvable`. Data PDUs require both
  identities. Undirected advertising leaves receiver address/type empty. The tool checks these
  identities against the complementary roles in both boards' `PULSE_META`; a USB port or sniffer
  display label is never a role source.
- `pdu_kind` is `data` or `advertising`. A data row has `llid=1|2|3`, a nonempty connection epoch,
  and `sn`/`nesn` in `{0,1}`. Advertising rows leave those four fields empty.
- `length_bytes` is the decoded BLE Link Layer `Length` field. The reported traffic definition is
  exactly `2 + length_bytes`: the two-byte LL header plus its payload. It excludes preamble, access
  address, and CRC, and is not interchangeable with firmware ATT/application counters.
- `payload_sha256` is SHA-256 of the exact `Length` payload bytes (including a MIC when it is part of
  that length), excluding the mutable two-byte header. For each connection epoch and direction,
  `is_retransmission=1` is accepted only when SN, LLID, length, and payload hash match the latest
  CRC-valid unique PDU. A repeated advertisement is a separate transmission, not an ARQ retry.
- `crc_ok` and `gap_before` are mandatory booleans. Set `gap_before=1` when the sniffer/exporter
  reports possible packet loss since the preceding retained target-pair row. A CRC failure or gap
  overlapping an event invalidates that event rather than silently undercounting traffic.

Pass the original capture separately; it is hashed byte-for-byte into provenance metadata but never
parsed:

```powershell
python .\pulse_tools.py link `
  --input .\capture\canonical-pdus.csv `
  --event-windows-csv .\results\event_power_metrics.csv `
  --metadata-csv .\results\run_metadata.csv `
  --raw-pcap cap01=.\capture\untouched.pcapng `
  --shared-timebase --time-uncertainty-s 0.000005 `
  --output-dir .\results
```

Use shared time only when sniffer timestamps already use the analyzer/event clock; the explicit
uncertainty is a worst-case bound in seconds. Otherwise supply a synchronization CSV:

```text
capture_id,sniffer_time_s,event_time_s,uncertainty_s
cap01,12.004820,0.000000,0.000005
cap01,3721.114910,3709.091620,0.000005
```

and replace the shared-time options with `--sync-anchors sync-anchors.csv`. Every capture needs at
least two distinct anchors that bracket its complete declared time range. The tool fits an affine
map, rejects extrapolation or an anchor residual larger than its stated uncertainty, and propagates
the conservative residual-plus-anchor uncertainty into packet/event-boundary checks.

Canonical event windows come only from `event_power_metrics.csv`. Accepted attempts use its exact
`role=pair` union window and `(run_id,pair_id,exchange_id)` key. The event is paired back to the
initiator/responder records with the same integrity routine as power analysis. A
`remote_session_started=0` failure must have the explicit initiator-only pair join and no responder
record; it is still retained and can include discovery/advertising traffic. Local/no-contact
windows use their exact exchange identity. Overlapping, duplicate, truncated, missing, or multiply
covered windows fail the audit. A packet whose uncertainty interval touches a window boundary is
left unassigned as ambiguous.

The command writes `link_packet_metrics.csv`, `event_link_metrics.csv`,
`link_capture_metadata.csv`, and `link_audit.csv`, then exits nonzero if any semantic audit row
fails. Capture metadata hashes the canonical CSV, raw PCAP, event-window CSV, run metadata, and
affine-anchor CSV when used. Event output includes total/unique/retransmitted LL-PDU bytes and
packet counts, directional bytes, retry rates, all quality counters, and `quality_valid`. Never
remove invalid rows to make an audit pass; correct or repeat the source capture.

## Exact two-board measurement procedure

The following is the minimum repeatable procedure for each independent physical board pair.

1. **Freeze both images.** Build the release PULSE image and the sensing+BLE baseline from the same
   commit, NCS/Zephyr toolchain, optimization level, board revision, clock, tensor precision, model,
   batch size 16, and `R=2`. Preserve `zephyr.map`, ELF SHA-256, `.config`, devicetree, and the full
   `PULSE_META` record. Do not compare a debug PULSE image with a release baseline.
2. **Prepare exactly two boards.** Label their immutable hardware IDs A and B and record the
   `pair_id`. Use the same IMU input windows and initial weights for all configurations. Record
   whether windows are live or replayed; replayed inputs demonstrate compute feasibility only.
3. **Make power safe and comparable.** Disconnect/isolate each battery and charger path according to
   the platform's approved lab procedure. Power both battery inputs from isolated analyzer channels
   at the same recorded voltage. Share analyzer digital ground only as required by the instrument.
   Never connect analyzer outputs together. Verify wiring against the schematic before energizing.
   The benchmark build disables the BQ25180, BQ27427, and M95P drivers: fuel-gauge status, if needed,
   belongs to a separate battery-attached preflight, and the measured model/replay bytes reside in
   internal flash rather than the external M95P.
4. **Wire and verify markers.** Connect P1.14/P1.13/P1.12/P1.11 from each board to separate digital
   inputs. Capture a slow marker self-test and confirm gate polarity plus all codes 0--7 before a
   current run. Use one analyzer clock for both current channels and eight digital inputs; if two
   analyzers are unavoidable, feed both a common hardware sync pulse and verify residual skew.
5. **Freeze radio conditions.** Record PHY, ATT MTU, data length, connection interval, TX power,
   security state, board separation/orientation, RSSI, and the connection state at trial start. Keep
   the RF setup unchanged within a block. Use a BLE sniffer concurrently if link-layer bytes are to
   be reported.
6. **Choose adequate sampling.** Record analyzer voltage and actual sample rate. The final rate must
   yield at least ten current samples in the shortest reported stage after any instrument filtering;
   otherwise report only a coarser combined stage. Do not upsample a low-rate trace to claim stage
   resolution.
7. **Control instrumentation overhead.** Disable shell, visual indicators, and unrelated UART logs
   in both compared images. Buffer the compact `PULSE_*` record and transmit it only after the event
   gate falls (or dump it after the capture). Keep marker and record instrumentation identical in
   PULSE and baseline builds. Confirm no UART activity lies inside a measured gate.
8. **Use a fixed warm-up.** After boot and BLE stabilization, execute exactly the configured five
   warm-up attempts (`CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS=5`) for the current path and connection
   state. If a different count is preregistered, freeze it for every compared image and record it in
   metadata. Emit warm-up event records with `warmup=1`, then
   emit measured records with `warmup=0`; do not merely delete inconvenient early observations.
   Start analyzer acquisition before the first warm-up so all GPIO windows join one-to-one.
9. **Acquire at least 100 post-warm-up attempts per required initiator stratum.** The mandatory
   strata are the four exact combinations listed above. Measure configured timeout/failure attempts
   in their attempted stratum. Each accepted attempt emits an initiator row. It emits exactly one
   responder row only after the remote session started; a pre-session discovery/connect failure has
   none but still yields a two-channel system-power row. Automatic identity ordering keeps logical
   roles fixed for a complete boot. Counterbalance placement and cabling only between complete
   blocks, using a new `run_id`, while retaining unique monotonic `event_index` within each boot (the
   reported `trial_id` may restart at zero after warm-up). Run more
   than 100 when a capture is rejected; never silently replace a failed protocol attempt, because it
   contributes to failure rate.
10. **Bracket with baseline.** Include at least 5 s of gate-low steady sensing+connected baseline
    before and after every trial block. Repeat the same workload with the no-PULSE baseline image.
    Investigate drift rather than pooling blocks whose baseline changes materially.
11. **Measure correctness separately but exactly.** Replay byte-identical input tensors and initial
    weights in embedded and Python reference implementations. Preserve exported tensors, model hash,
    precision, and quantization/scaling. Emit cosine similarity of the complete gradient vectors and
    maximum absolute error of the updated parameters.
12. **Measure memory under maximum overlap.** Obtain static flash/RAM from the exact release maps and
    retain the runtime diagnostics for local, initiator, and responder paths. The firmware's event
    workqueue stack watermark is cumulative since boot and excludes Bluetooth RX, UART, system, and
    driver threads; its system-heap peak is reset per event. Do not add either value to linker RAM,
    which already reserves those pools, or present the event-worker watermark as whole-system peak
    RAM. A separate defensible measurement is required for that stronger claim.
13. **Parse and audit before plotting.** Run `serial`, `power`, `link`, `resources`, then
    `summarize`. Resolve every failed row in `protocol_audit.csv` and `link_audit.csv`, every
    unmatched marker, unknown outcome, truncated capture, and missing warm-up label. Only then run
    `figures`.

Repeat the complete procedure for every physical pair claimed as independent in the paper. One pair
with 100 trials provides 100 repeated measurements, not 100 independent board pairs.

## Statistics and figure outputs

Only post-warm-up, successfully joined, non-truncated rows enter distributions. Invalid analyzer rows
remain visible through `source_count` and `quality_excluded_count`. Failed transactions enter
`failure_rate` and are excluded from successful-path latency distributions. Power summaries retain
their measured cost in separate `outcome=success|failure` strata so failed-attempt energy can enter
the daily projection. Every summary is tidy/long-form and includes `n`, total/success/failure/unknown
counts, failure rate, mean, median, Q1, Q3, IQR, p95, minimum, and maximum. Event, stage,
power, stage-power, and link summaries retain `remote_session_started`; stage-power summaries also
retain `stage_sequence`. Every pooled summary has a corresponding `*_summary_by_pair.csv` table
grouped by run, pair, and board for sensitivity analysis. Pooled plotting is permitted only after
every capture-level required stratum passes `protocol_audit.csv`.
Quantiles use linear interpolation at index `(n-1)p` (R-7/NumPy default).

Firmware `duration_us`/`wall_duration_us` use Zephyr's 64-bit cycle-counter API backed by the nRF54
GRTC SYSCOUNTER, so they are the authoritative sleep-inclusive wall intervals. Firmware `cycles` and
`cycle_duration_us` are elapsed 32-bit DWT-counter intervals that include interrupts and time
preempted by higher-priority work; they are not exclusive thread CPU cycles. Radio and control stages
intentionally measure wall-clock waits. Cross-check wall duration against GPIO/analyzer edges. The
configured timeout is at most 20 seconds, leaving headroom below the DWT wrap at 128 MHz.

The figure command writes:

- `figure_power_trace.csv`, `figure_power_events.csv`, `figure_power_stages.csv`, and
  `figure_power.csv` for the synchronized traces, GPIO-derived boundaries, and event
  energy/average/peak power panel;
- `figure_resources.csv`, `figure_resource_totals.csv`, and `figure_runtime_ram.csv` for static and
  peak-memory bars; `figure_resource_provenance.csv` binds those rows to content-addressed maps,
  ELFs, UART metadata, and correctness input;
- `figure_latency.csv` for stage and end-to-end median/IQR/p95 plus failure rates;
- `figure_link.csv` for total, unique, retransmitted, and directional LL-PDU byte/count statistics;
- `section_v_panels.pdf` and `section_v_panels.png` when matplotlib is available.

The representative trace is selected by the measured pair event nearest the median incremental
energy using its complete run/pair/exchange/event/path identity, so equal event indices in another
run cannot leak into the trace. Energy bars show median/IQR and p95; average/peak annotations come
from aggregate pair summaries, not the representative event. Failure annotations use event-level
rates only. The energy/power table and panel retain separate success/failure strata and require
successful local, no-contact, accepted-initiator, accepted-responder, and exact pair-union rows.
Every required row has incremental energy plus average and peak power; accepted pair unions also
require total two-board energy. Figure generation fails if any required latency, power, baseline,
link, resource, trace, or protocol-audit coverage is missing, and verifies that event-analysis
baseline values exactly match the included baseline-only capture. It also requires every stage the
firmware emits for the applicable successful path: both local steps; no-contact scan, selection and
local update; accepted selection, head/gradient serialization and transfer, both endpoint compute
stages, agreement, normalization, mixing and utility update; and scan/connect for discovery.
Figure tables retain all trials for traceability.

For the discussion's incremental daily-energy projection, provide a **complete per-node ledger**.
The input is not a free-form table of rates. Its mandatory columns are:

```text
scenario,seed,node_id,network_node_count,observation_duration_s,event_path,role,connection_state,outcome,remote_session_started,count_in_observation,local_opportunities,selection_opportunities,count_basis,source_artifact,source_artifact_sha256
```

Only the physical strata emitted by the release campaign are accepted:
`local/local/connected`, `no_contact/local/disconnected`,
`accepted_connected/initiator/connected`, `accepted_connected/responder/connected`,
`accepted_discovery/initiator/disconnected`, and
`accepted_discovery/responder/connected`. In particular, the responder is already connected when
its discovery-path event starts. Wildcards and `na` connection states are rejected.
`remote_session_started` is `na` for local/no-contact rows, `1` for every responder and successful
initiator row, and either `0` or `1` for failed initiator rows. `count_basis` is exactly one of:

- `deployment_observed`: a complete observation of all devices in the declared network; or
- `empirical_rate_model`: logical opportunity counts combined with measured firmware
  success/failure and remote-session-start branching rates.

`source_artifact` names a readable immutable count manifest and `source_artifact_sha256` is its
64-digit SHA-256, which the tool verifies before projecting. That manifest should hash the
simulator/deployment logs, configuration, node mapping, and
the procedure used for any empirical-rate split. `count_in_observation` may be fractional only for
an explicitly documented empirical-rate model; deployment-observed counts must be whole events.
The tool derives
`count_per_day=count_in_observation*86400/observation_duration_s`; if a `count_per_day` column is
also supplied, it is treated only as a cross-check.

Every `(scenario,seed,node_id)` must contain explicit rows, including zero-count rows, for these
semantic categories:

1. local success and local failure;
2. no-contact success and no-contact failure;
3. for `accepted_connected`, initiator success with `remote_session_started=1`, initiator failure
   with `remote_session_started=0` and `1`, and responder success and failure;
4. the same five rows for `accepted_discovery`.

Both accepted paths require the complete five-row outcome/session lattice even when one path has
zero projected events. This prevents a connected success row and discovery failure rows from
collectively hiding missing categories.

Keep the attempted `accepted_*` path on failures. Do not relabel one as a generic `failed` path.
For each node, local success+failure must equal `local_opportunities`; no-contact outcomes plus all
accepted initiator outcomes must equal `selection_opportunities`. Across the complete declared
network, responder events must equal accepted initiator events with
`remote_session_started=1`. `network_node_count` makes a silently truncated node subset fail.

The present PULSE simulator artifacts do **not** by themselves meet this contract. `training.csv`
contains scheduled local work and `contacts.csv` contains accepted directional attempts, but a
PULSE no-contact choice returns no log row, transport failures are not simulated, and the current
runner does not write its realized node assignment. Therefore, do not infer missing rows as zero or
claim that raw `contacts.csv` is a complete count source. Use a separately validated complete
decision export, or a complete deployment observation, then preserve its manifest and hash.

`project-energy` writes `daily_energy_audit.csv` before returning or failing. It rejects empty or
incomplete ledgers, duplicate categories, inconsistent durations/opportunity totals, missing nodes,
unbalanced responder sessions, wildcard labels, and missing energy statistics for positive counts.
An explicit zero row may have no corresponding power row because its contribution is exactly zero.
Successful output includes `daily_energy_contributions.csv` and one
`daily_energy_projection.csv` row per scenario/seed/node; seeds are never silently pooled. The
expected daily incremental energy is the event rate times the **sample mean** event energy. Columns
named `plugin_sensitivity_using_event_{median,q1,q3,p95}` are deliberately labeled plug-in
sensitivity calculations: they sum rates times a marginal event statistic and are neither daily
energy quantiles nor confidence bounds. Energy rows match the ledger exactly on path, physical role,
connection state, outcome, `remote_session_started`, and measurement scope.
Idle/sensing baseline energy is intentionally excluded from this *incremental* projection.

### Full battery-life projection

`project-battery-life` combines a complete 24-hour state-power ledger with arithmetic-mean
incremental event energy and an explicit usable battery-energy model:

```powershell
python .\pulse_tools.py project-battery-life `
  --input-json .\battery_projection_input.json `
  --output-json .\battery_projection_result.json `
  --output-csv .\battery_projection_result.csv
```

The input must provide exactly `low`, `nominal`, and `high` sensitivity cases. Every case requires
an explicit description, battery capacity, nominal voltage, usable-energy efficiency, state
durations and state-specific idle powers, and daily event counts. There are no physical defaults.
State durations must sum to 86,400 seconds, and every counted event class must have measured
incremental-energy samples. Scenario spread is labeled sensitivity, while standard error propagated
from the arithmetic event-energy means is reported separately and is not a confidence interval.
See [`BATTERY_PROJECTION.md`](BATTERY_PROJECTION.md) for the input contract, equations, and
interpretation limits.

## Linker maps, `nm`, and RAM watermarks

Save symbol sizes without allowing locale-dependent formatting:

```powershell
arm-zephyr-eabi-nm.exe --print-size --size-sort --radix=x .\zephyr.elf |
  Set-Content -Encoding ascii .\pulse.nm
```

The map parser uses memory-region capacities and non-overlapping output sections for totals. It
classifies object contributions into base firmware, encoder, head, training tensors, protocol
buffers, thread stacks, and PULSE state using conservative filename/symbol regexes. Review
`map_contributions.csv`; for a final paper, supply project-specific rules:

```json
{
  "encoder": ["libexact_encoder\\.a", "encoder_model"],
  "head": ["pulse_head"],
  "training_tensors": ["training_workspace", "gradient"],
  "protocol_buffers": ["pulse_wire", "gatt"],
  "pulse_state": ["pulse_peer", "utility_table"]
}
```

Pass that file with `--category-rules categories.json`. Initialized `.data` counts in both static RAM
and the flash load image. Linker padding/unattributed bytes are explicit rather than silently assigned
to a component. Always cross-check `resource_summary.csv` against Zephyr's own RAM/ROM report;
vendor linker-script layouts can require adjusted rules.

For paper output, pass the exact archived `zephyr.elf` and `run_metadata.csv` for `pulse`,
`baseline`, and `correctness`, plus the exact `correctness.csv`. The resource command
writes `resource_provenance.csv` with a SHA-256 for every supplied map, `nm` file, ELF, metadata CSV,
correctness CSV, and runtime-watermark source. It rejects an ELF that does not contain the full Git
revision reported by UART and, for model-bearing images, the artifact digest. Figure coverage then
requires the release, matched-baseline, and correctness images to name the same revision and binds
each metadata CSV by hash, directly tying the parsed records and resource rows to those images.

When only `nm` output is available, the tool emits a clearly labelled symbol-sum lower bound with no
capacity/free-space claim. It counts initialized data in RAM and its flash load image. Use the linker
map for the final totals because `nm` does not account for section padding, alignment, or all linker
metadata.

Runtime watermark CSV accepts `event_path`, `role`, and preferably a direct `peak_ram_bytes`. It can
also derive the peak from `ram_capacity_bytes - minimum_free_ram_bytes`, or from static RAM plus an
explicitly non-overlapping `additional_dynamic_bytes`. `peak_stack_bytes`, `peak_heap_bytes`, and
`other_dynamic_bytes` are retained as component measurements; firmware-native
`thread_stack_peak_bytes` and `system_heap_peak_bytes` are accepted aliases. Stack and heap peaks
are not blindly added because Zephyr's stack and system-heap pools are already reserved in
`.bss`/`.noinit`, and independent peaks need not be simultaneous. When event rows contain only
those watermarks, `peak_ram_bytes` is the linker-reserved RAM footprint and
`stack_heap_watermark_upper_bound_bytes` is emitted as a diagnostic. The output always records the
calculation method.

For final Section-V figures, every required local/no-contact and accepted initiator/responder path
must have a positive peak from `reported_peak_ram_bytes`, `capacity_minus_minimum_free_ram`, or
`static_plus_explicit_nonoverlapping_dynamic`. The linker-reserved fallback is useful context but is
not accepted as a measured runtime peak. Figure coverage also requires a raw
`PULSE_CORRECTNESS,pass=1,reference=pytorch_cpu_float32` row whose 64-digit artifact and fixture
hashes exactly match a `pulse_release/exported_pulse_npz` metadata row; include the parsed
correctness result directory as a separate `--data-dir`.

## Important limitations

- The scripts cannot produce hardware results without real current, marker, timing, correctness, and
  memory measurements. The fixture data under `tests/fixtures` is solely a parser test and must never
  be copied into the paper.
- Application request/response bytes can come from firmware counters. **Link-layer traffic requires
  a synchronized BLE sniffer capture plus a canonical PDU export that passes `link_audit.csv`.** ATT
  payload size is not link-layer traffic; the link importer hashes but does not parse the PCAP and
  never fabricates PHY/MAC overhead. `ll_pdu_bytes` means the two-byte BLE data-channel LL header
  plus the octet count encoded by its Length field. It includes empty/control/data PDUs, MIC bytes
  when present in Length, and retransmissions; it excludes preamble, access address, CRC, and IFS.
- GPIO timing is quantized by analyzer sampling and the real edge lies somewhere between samples.
  Half-sample boundary placement is explicit; stages too short for the configured rate must be
  combined or omitted.
- Stage code 3 combines head forward/backward, and code 6 combines agreement, normalization, mixing,
  and utility update. Separate latency values for those sub-operations must come from `PULSE_STAGE`
  cycle-counter records, not from the 3-bit marker alone.
- The gate-low sample median is diagnostic only. Paper results use a separate all-gates-low matched
  capture's integrated mean power, bound through `--baseline-metadata` (or manually cross-checked by
  exact role/value during figure coverage).
- Pair average/peak power is valid only when both analyzer channels share a time base and cover the
  entire union window. Merely adding per-board peaks is invalid.
- A map file describes static allocation, not live heap/stack high-water marks, external flash use,
  model-storage wear, battery lifetime, thermal behavior, or RF reliability.
- Discovery/scanning and failed/skipped attempts must be measured as their own paths. Exchange-only
  energy cannot support a daily battery-life claim. A daily projection must multiply per-role event
  energies by measured per-node counts; do not multiply by aggregate network contacts. This tool
  reports only incremental event energy, so a separate full battery-life claim must also add the
  measured idle/sensing energy over the same observation period.

## Tests

```powershell
python -m unittest discover -s .\tests -v
```

The tiny fixtures exercise key/value and positional serial records, autonomous 105-sequence proof
completeness for both boards, rectangular integration of NGMO2 interval averages, simultaneous pair
bin maxima, explicit static current ranges, warm-up filtering, LSB-first GPIO decoding, legacy
trapezoidal integration, exact exchange-ID pairing, failure-energy stratification, canonical link-PDU
accounting and clock audits, robust summaries, daily projection, and map/`nm` parsing.
