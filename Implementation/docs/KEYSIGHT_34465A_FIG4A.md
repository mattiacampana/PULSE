# Figure 4a with one Keysight 34465A

This is the operator procedure for Figure 4a. It replaces the two-channel
NGMO2 procedure when only one Keysight 34465A is available. The two SensWear
boards run the same firmware and retain automatic identity-based role election,
but current is measured in two separate runs:

1. measure the initiator while the responder is powered independently;
2. move the meter and measure the responder while the initiator is powered
   independently.

The result contains measured local, initiator, and responder energy. Complete-
pair energy is the sum of matching trial IDs from the two sequential role runs.
It is an estimate, not a simultaneous pair measurement. Do not report a pair
peak power or call the two traces synchronized.

## 1. What this protocol measures

The firmware overlay `config/pulse_keysight_34465a_fig4a.conf` runs only:

- 5 warm-ups plus 100 measured local updates;
- 5 warm-ups plus 100 measured accepted-connected exchanges.

The initiator produces 210 identifiable current events: 105 local events
followed by 105 accepted-connected events. The responder produces 105 events,
all during accepted-connected exchanges. Event starts are at least 5 s apart,
and the first event starts 60 s after boot so timed recording can start at idle.

The DMM records a continuous, immediately triggered 50-readings/s current stream
with a 19 ms aperture. The host drains reading memory during acquisition, so
the 50,000-reading standard memory does not limit the roughly 19-minute run.
Sustained upward crossings of the pilot-tested current threshold are found in
software. For each event the tool extracts 225 consecutive readings: 20 before
the crossing (0.4 s) and 205 from the crossing onward (4.1 s). Incremental source-side
energy is

```text
E_inc = V_supply * 0.020 s * sum(I_post[n] - mean(I_pre))
```

over the fixed 4.1 s post-trigger window. The idle tail is deliberately retained
and baseline-subtracted. This is not a GPIO-delimited interval.

## 2. Required equipment and non-negotiable limits

- one Keysight 34465A with a current calibration in date;
- a current-limited DC source set to the approved cell-equivalent voltage for
  the measured board; the 34465A measures current but does not source power;
- one charged cell or a second safe supply for the unmeasured peer;
- the two boards and normal flashing/debug hardware for preflight only.

Use the fixed 10 mA DC range through the 3 A current input for the paper
capture. The valid initiator pilot observed about 1.15 mA idle and repeated
peaks near 4.5 mA. The 10 A input
**forces the actual 10 A measurement range** even if a range query reports a
different value; the earlier 1 A/10 A configuration was invalid. The 10 mA
range has less than 0.027 V full-scale burden. A single `9.9e37` reading is an
invalid DMM reading, not evidence that the board drew that current. The stream
preserves its raw time slot, ignores it for event detection, and continues on
the 10 mA range. Only if repeated valid readings approach the range limit or
independent evidence confirms a real overload should the range be reassessed;
the 100 mA range can have up to 0.27 V full-scale burden. Verify that the board
still receives a suitable operating voltage and completes its BLE protocol. Report
source-side supplied energy and the DMM's series-loss limitation in the paper.

Warm the DMM for at least 90 minutes before paper measurements. Check that its
current-input fuse is intact. Never connect a current input directly across a
voltage source.

No GPIO or external-trigger wiring is used. This protocol does not measure
firmware phase boundaries.

## 3. Build and flash the one common image

Complete the final-capture prerequisites in `docs/PULSE_MINIMAL_MEASUREMENT.md`,
including a clean committed source tree, the real exported model artifact, and
the filled hardware-revision lab configuration. Then run from the firmware
repository root:

```powershell
$artifact = (Resolve-Path '<real exported artifact.npz>').Path
$labConfig = (Resolve-Path '<filled pulse_lab.conf outside this repository>').Path
$keysightConfig = (Resolve-Path .\config\pulse_keysight_34465a_fig4a.conf).Path
$extraConfig = "$labConfig;$keysightConfig"

cmake --preset senswear_pulse --fresh `
  "-DPULSE_FINAL_CAPTURE=ON" `
  "-DPULSE_ARTIFACT_NPZ=$artifact" `
  "-DEXTRA_CONF_FILE=$extraConfig"
cmake --build --preset senswear_pulse
```

Archive `zephyr.elf`, `zephyr.hex`, `zephyr.map`, `.config`, the generated model
manifest, and their hashes. Obtain the exact flashed-image hash:

```powershell
$imageSha = (Get-FileHash .\build\pulse\release\zephyr\zephyr.hex -Algorithm SHA256).Hash.ToLower()
$imageSha
```

Flash this exact image to both boards with the repository's nRF54L15 OpenOCD
RRAM loader, selecting each live CMSIS-DAP probe explicitly. The board does not
register a `pyocd` west runner. Confirm each command prints `verified ... bytes`
and no `Error:` line before proceeding to the next board:

```powershell
$hex = (Resolve-Path .\build\pulse\release\zephyr\zephyr.hex).Path.Replace('\', '/')
$interface = (Resolve-Path .\config\openocd-cmsis-dap.cfg).Path
$target = (Resolve-Path .\config\openocd-nrf54l15.cfg).Path
foreach ($probeId in @('<RESPONDER_PROBE_UID>', '<INITIATOR_PROBE_UID>')) {
  & .\config\scripts\openocd-with-env.cmd `
    -c "adapter serial $probeId" -f $interface -f $target `
    -c "init; reset halt; echo [nrf54l-load {$hex}]; echo [verify_image {$hex}]; reset run; shutdown"
  if ($LASTEXITCODE -ne 0) { throw "OpenOCD failed for $probeId" }
}
```

## 4. Run one UART preflight before current measurement

Keep both debug boards attached for this step only. Use 921600-8-N-1, the
`current-speed` recorded in `build/pulse/release/zephyr/zephyr.dts`, start one
raw logger for each board, and reset both boards. Wait for the initiator to print
`state=campaign_complete`.

After `protocol_errno=0`, the initiator is intentionally quiet for about 60 s.
The first `PULSE_EVENT` should then appear; if none appears within two minutes,
stop and diagnose the firmware rather than accepting a partial preflight. The
responder stays quiet through the initiator's 105 local trials (about nine more
minutes). The whole two-path campaign takes at least about 18 minutes.

Parse the two complete logs:

```powershell
python .\tools\pulse\pulse_tools.py serial `
  .\capture\p01\preflight-board-a.log `
  .\capture\p01\preflight-board-b.log `
  --set pair_id=p01 --set run_id=p01-keysight-preflight-r01 `
  --output-dir .\results\p01\keysight-preflight
```

Open `results/p01/keysight-preflight/run_metadata.csv` and record:

- the immutable board ID assigned `role=initiator`;
- the immutable board ID assigned `role=responder`;
- `firmware_revision` and `artifact_sha256`;
- `capture_pacing=keysight_current_level_trigger_fixed_period`;
- `capture_min_event_period_ms=5000`;
- `marker_available=0` and `marker_event_gate=disabled`.

Do not proceed unless `event_metrics.csv` has exactly 100 successful,
non-warm-up rows for local/local, accepted_connected/initiator, and
accepted_connected/responder, with zero failures.

## 5. Wire the measured board

Turn the DC source off before changing any lead or current terminal. Do not
switch the meter's current path while the board is powered.

Current path, in series:

```text
DC source +  ->  Keysight 3 A current input
Keysight LO  ->  measured board battery-positive input
DC source -  ->  measured board ground/battery-negative input
```

Remove the measured board's cell. Disconnect its debugger, debug FPC, USB, and
all other possible supply paths. Set the approved source voltage, OVP, and
current limit before enabling it. Record how the source voltage was verified;
the DMM cannot measure voltage and current simultaneously.

Power the peer from its own cell or independent safe source. Its debugger and
USB must also be absent. During each run only the intended two-board pair may be
advertising nearby.

Leave the Keysight external-trigger BNC and all SensWear GPIOs disconnected.

## 6. Connect to and check the DMM

Install PyVISA once if needed, along with the approved Keysight/IVI VISA
runtime:

```powershell
python -m pip install pyvisa
```

Discover and probe the instrument without changing its state:

```powershell
python .\tools\pulse\pulse_tools.py keysight-capture --list-resources
python .\tools\pulse\pulse_tools.py keysight-capture `
  --resource '<USB...::INSTR or TCPIP...::INSTR>' --probe
```

The probe must identify model `34465A`. Before the paper runs, make one
trigger-free pilot trace at 50 readings/s. With the measured board's cell and
USB/debug supply removed, wire the meter in series as above. Set the measured
board's supply to 3.7 V, 0.1 A current limit, and 4.0 V OVP, and power the peer
independently. Power-cycle both boards, then run the pilot promptly while they
are in the 60 s startup delay:

```powershell
python .\tools\pulse\pulse_tools.py keysight-capture `
  --pilot --pilot-duration-s 180 `
  --resource 'USB0::0x2A8D::0x0101::MY59028205::INSTR' `
  --measured-role initiator --board-id f5ea5fa8fcfa5209 `
  --current-range-a 0.01 --current-terminal-a 3 `
  --output-dir .\capture\p01\keysight-initiator-pilot-r01 `
  --supply-voltage-v 3.7 --source-current-limit-a 0.1 `
  --source-overvoltage-protection-v 4.0 --maximum-safe-voltage-v 4.2 `
  --confirm-current-fuse --confirm-series-current-wiring `
  --confirm-measured-battery-removed --confirm-no-usb-backpower `
  --confirm-peer-separately-powered
```

The command resets and reconfigures the DMM, so selecting `10 mADC` on its
front panel alone does not set the capture range. The explicit range and
terminal arguments above reproduce that setting and verify its readbacks.
Use a new output directory for each retry; the tool never overwrites a pilot.
Do not reset or rewire after `PILOT STARTED`. The DMM records
9,000 untriggered samples over 180 s, including boot and the beginning of the
initiator's local block. This is diagnostic-only data. Its printed maximum is
not, by itself, a valid threshold: inspect the saved CSV for settled idle
excursions versus the repeated event onsets. Choose a current trigger level
above all idle/noise excursions and below every event-onset plateau. Verify
that current returns below it between events. A threshold that has not been
tested must not be used for a paper run. For the responder, use a 750 s pilot
(37,500 readings, within standard memory) so the trace spans the initiator's
local block and the start of the connected block; do not infer its event
threshold from an initiator trace.

If a pilot fails its sample-integrity check, rerun it with a fresh output
directory using the current script, which distinguishes a short `FETC?`
response from a non-finite/overload value and reports the first bad sample's
time and value. Do not infer an over-range current from the old combined error
message alone. Increase the range if the new result confirms overload or a
near-range peak.

Before choosing a threshold, confirm that the DMM reports **positive** current
consistent with the independently observed supply current, with neither a range
warning nor a value above 9 mA anywhere in the pilot. The previous trace on
the 10 A terminal was about -45 mA while the supply showed about 4 mA; changing
the range alone does not resolve a reversed or mismatched current path. Also
confirm both UART logs resolve their roles and `link_errno=0`; the observed
responder `role=unresolved,link_errno=-11` means that run contained no valid
connected PULSE exchanges. If any of these checks fails, fix the wiring or
board-to-board link and repeat the pilot, not the paper capture.

To inspect the exact commands without touching the DMM:

```powershell
python .\tools\pulse\pulse_tools.py keysight-capture `
  --dry-run --measured-role initiator --trigger-level-a '<tested amperes>'
```

The capture uses timed digitizing without the DMM's internal level detector.
The 34465A's fixed-range hardware detector completed three 206-reading records
while measured current stayed near 1.15 mA, below the tested 3.5 mA level;
those records are diagnostics, not paper data. The capture command fails on
unsupported SCPI commands or mismatched readbacks.

## 7. Capture run 1: measured initiator

Wire the board identified as initiator into the DMM current path. Power the
responder independently. Power-cycle/reset both boards together, then start
this command during the firmware's 60 s boot delay:

```powershell
python .\tools\pulse\pulse_tools.py keysight-capture `
  --resource '<verified VISA resource>' `
  --measured-role initiator --capture-mode stream `
  --current-range-a 0.01 --current-terminal-a 3 `
  --output-dir .\capture\p01\keysight-initiator-r01 `
  --run-id p01-keysight-initiator-r01 --pair-id p01 `
  --board-id '<INITIATOR_BOARD_ID>' --peer-board-id '<RESPONDER_BOARD_ID>' `
  --firmware-revision '<PULSE_META firmware_revision>' `
  --firmware-image-sha256 $imageSha `
  --artifact-sha256 '<PULSE_META artifact_sha256>' `
  --supply-voltage-v '<verified source voltage>' `
  --supply-voltage-source '<source model and verification record>' `
  --source-current-limit-a '<approved current limit>' `
  --source-overvoltage-protection-v '<approved OVP>' `
  --maximum-safe-voltage-v '<board maximum safe voltage>' `
  --trigger-level-a '<tested initiator threshold in amperes>' `
  --calibration-id '<DMM calibration ID>' --calibration-date '<YYYY-MM-DD>' `
  --confirm-current-fuse --confirm-series-current-wiring `
  --confirm-measured-battery-removed --confirm-no-usb-backpower `
  --confirm-peer-separately-powered --confirm-trigger-level-tested
```

Power-cycle/reset both boards first, then start the command within about 20 s,
well before the first event. The tool verifies the timed-sampling readbacks and
prints `STREAMING`. Raw readings are saved continuously to `*stream_raw.csv`;
event progress is printed as crossings are found. Do not reset or interact with
either board after `STREAMING`. Wait until it saves 47,250 extracted event-window
readings from 210 events. The expected duration after reset is about 19 minutes.

An isolated non-finite or `9.9e37` raw reading does not stop the stream or count
as an event. The script preserves its sample index and time, prints a warning,
and completes the paper sample CSV if every selected event window contains only
valid readings. The manifest records how many invalid raw readings were outside
the event windows. An incomplete run retains the raw trace for diagnosis but
does not write a completed paper sample CSV. Repeat in a new output directory
if an invalid reading falls within a selected event window, or the script
reports a timeout, a DMM error, a memory overflow, fewer than 210 events, or a
changed connection. The first 20 s of the stream are
excluded from event detection to avoid startup transients; start the command
promptly after resetting both boards so this guard ends before the first event.

## 8. Capture run 2: measured responder

Turn the source off. Move the series current path to the board identified as
responder. Power the initiator independently.
Do not change firmware, DMM settings, source voltage, board separation, or the
RF environment.

Run the same command with these substitutions:

```powershell
python .\tools\pulse\pulse_tools.py keysight-capture `
  --resource '<same VISA resource>' `
  --measured-role responder --capture-mode stream `
  --current-range-a 0.01 --current-terminal-a 3 `
  --output-dir .\capture\p01\keysight-responder-r01 `
  --run-id p01-keysight-responder-r01 --pair-id p01 `
  --board-id '<RESPONDER_BOARD_ID>' --peer-board-id '<INITIATOR_BOARD_ID>' `
  --firmware-revision '<same firmware_revision>' `
  --firmware-image-sha256 $imageSha `
  --artifact-sha256 '<same artifact_sha256>' `
  --supply-voltage-v '<same verified source voltage>' `
  --supply-voltage-source '<source model and verification record>' `
  --source-current-limit-a '<same approved current limit>' `
  --source-overvoltage-protection-v '<same approved OVP>' `
  --maximum-safe-voltage-v '<board maximum safe voltage>' `
  --trigger-level-a '<tested responder threshold in amperes>' `
  --calibration-id '<same calibration ID>' --calibration-date '<YYYY-MM-DD>' `
  --confirm-current-fuse --confirm-series-current-wiring `
  --confirm-measured-battery-removed --confirm-no-usb-backpower `
  --confirm-peer-separately-powered --confirm-trigger-level-tested
```

Power-cycle/reset both boards, then reach `STREAMING` within about 20 s. The
responder current remains below its tested event threshold during the
initiator-only local block, so software extracts 23,625 readings from the
responder's 105 accepted-connected events. The raw trace includes the earlier
idle period. The total wait is still about 19 minutes.

## 9. Audit and generate Figure 4a

Run:

```powershell
python .\tools\pulse\pulse_tools.py keysight-fig4a `
  --initiator-csv .\capture\p01\keysight-initiator-r01\keysight_34465a_initiator_samples.csv `
  --responder-csv .\capture\p01\keysight-responder-r01\keysight_34465a_responder_samples.csv `
  --preflight-events-csv .\results\p01\keysight-preflight\event_metrics.csv `
  --preflight-metadata-csv .\results\p01\keysight-preflight\run_metadata.csv `
  --output-dir .\results\p01\fig4a
```

Do not use the figure unless
`results/p01/fig4a/keysight_capture_audit.csv` says
`pass_with_declared_sequential_pair_limitation`. Preserve:

```text
capture/p01/keysight-initiator-r01/
  keysight_34465a_initiator_samples.csv
  keysight_34465a_initiator_stream_raw.csv
  keysight_34465a_capture_manifest.csv
capture/p01/keysight-responder-r01/
  keysight_34465a_responder_samples.csv
  keysight_34465a_responder_stream_raw.csv
  keysight_34465a_capture_manifest.csv
results/p01/keysight-preflight/
  event_metrics.csv
  run_metadata.csv
results/p01/fig4a/
  keysight_capture_audit.csv
  keysight_event_energy.csv
  keysight_energy_summary.csv
  keysight_idle_summary.csv
  figure4a_trace.csv
  ondevice_power.pdf
  ondevice_power.png
```

The trace panel uses the local trial nearest the local median and the accepted-
connected trial ID nearest the median sequential role sum. Initiator and
responder traces are aligned to sustained positive current crossings identified
in each role's continuous trace;
they were not recorded during the same encounter.

## 10. Required manuscript wording changes

Replace "synchronized initiator and responder current traces" with
"sequential, event-aligned initiator and responder current traces." Remove the
claim of GPIO-derived phase boundaries and state that alignment uses pilot-tested
current thresholds applied in software to continuously sampled current.

State all of the following:

- one physical board pair, 5 warm-ups and 100 measured trials per category;
- Keysight 34465A, fixed 10 mA range through the 3 A input (or 100 mA if
  validated pilots require it), 50 readings/s,
  19 ms aperture;
- 0.4 s same-stream pre-event baseline and fixed 4.1 s post-event window;
- no GPIO access; role-specific current thresholds were validated in pilot
  traces and recorded in the capture manifests;
- supply voltage and the fact that energy is source-side and includes DMM
  burden (specified below 0.027 V full-scale on 10 mA or below 0.27 V
  full-scale on 100 mA);
- local, initiator, and responder distributions are measured separately;
- pair energy is the matched-trial sum of sequential role measurements;
- no pair peak power is reported; per-role peak is only the maximum 19 ms
  aperture reading, not instantaneous peak power;
- exact power-trial outcomes are not joined to UART; the same-image 100-trial
  preflight had zero failures.

Do not use these captures for daily battery-life, scanning/discovery, failed-
attempt, or simultaneous pair-power claims.
