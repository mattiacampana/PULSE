# R&S NGMO2-only measurement procedure

This is the primary power-measurement procedure for the two-board SensWear
PULSE experiment. During a recorded power run, the bottom R&S NGMO2 is the only
source and current-measurement instrument. It powers and measures both SensWear
main boards while being controlled from the PC through the laboratory's
USB-to-instrument adapter and its VISA resource.

The following are absent from the measured circuit for every dynamic and static
capture:

- both batteries;
- both debug boards and both FPC cables;
- all GPIO/event-trigger wires;
- the upper Keithley and Keysight DMMs and all of their leads;
- phone connections and any other USB connection to either SensWear board.

The debug boards are used only before a block for flashing/UART preflight and
after a block for proof recovery, always with the NGMO2 outputs off. The upper
meters may be used for a separate, explicitly documented bench check, but must
then be completely removed before recorded captures. They are not series meters
or voltage monitors in this experiment.

## 1. Measurement boundary and wiring

Keep both NGMO2 outputs off while making or changing any connection.
After every run, exception, PC failure, or USB/VISA loss, verify on the NGMO2
front panel that both channel outputs are OFF before touching a lead or
reconnecting a debug board. A software OFF request cannot prove the state after
communications are lost.

| NGMO2 channel | SensWear connection |
| --- | --- |
| channel 1 / A `FORCE+` | positive power-input/battery pad of one main board |
| channel 1 / A `FORCE-` | matching negative power-input/battery pad of that board |
| channel 2 / B `FORCE+` | positive power-input/battery pad of the other main board |
| channel 2 / B `FORCE-` | matching negative power-input/battery pad of that board |

Use the battery-pack-side negative connection named by the board documentation,
not an arbitrary ground point; the board contains low-side fuel-gauge sensing.
Never join the two NGMO2 outputs, cross a sense lead between channels, or connect
an output across the two pads as though it were an ammeter.

The NGMO2 breakout has duplicated colored sockets, silver links, and separate
black DVM sockets. Do not infer a socket function solely from color or physical
position:

- verify the breakout and instrument labels against the NGMO2 manual;
- use the designated `FORCE+` and `FORCE-` sockets and leave the verified
  local-sense jumpers/links in their required positions;
- leave the black DVM sockets unused during every recorded capture.

This final protocol is two-wire/local-sense only: do not remove the local links
or attach remote-sense leads. Record both output-lead sets' IDs, lengths, and
gauges. The pre/post channel voltage readbacks are local to the analyzer output,
not independent Kelvin measurements at the board pads, so the supplied-energy
boundary includes external lead loss.

Before enabling either output, complete these gates:

1. Physically remove/disconnect both cells. Never parallel a cell with an NGMO2
   voltage-source output.
2. Remove both debug boards and the complete FPC cables, not merely their USB
   cables.
3. Remove every lead from the Keithley/Keysight instruments and every GPIO or
   logic-trigger lead.
4. Verify positive and negative pad identity, polarity, cable integrity, and
   absence of shorts with all sources off.
5. Obtain the requested supply voltage and maximum-safe voltage from the exact
   cell, board, and power-path documentation. The software intentionally has no
   battery-specific voltage default.
6. Choose a current limit that is below the safe limit of the board, contacts,
   and leads but above the validated peak demand. Determine it in a separate
   pilot; do not copy an unverified value from this guide.
7. Verify that disabling channel A powers down only its board and disabling
   channel B powers down only its board. Any board that remains powered indicates
   cross-powering or another supply path and invalidates the setup.
8. Confirm that the NGMO2 is within calibration and that neither channel enters
   current-limit operation or range overload in the pilot.
9. For accepted-path trials, power only the intended pair. Turn off or remove
   every other compatible SensWear board from the RF test area so discovery
   cannot silently select a third board.
10. Program and read back independent-output mode, 0-ohm output impedance,
   current-limit mode, the maximum settable voltage and current, and OVP before
   enabling either channel. The capture controller performs these checks and
   fails closed.

## 2. Fixed autonomous timeline

Each power cycle executes at most one configured firmware event. The release
overlays and host command implement this schedule:

| Approximate time relative to verified output enable | Required action |
| ---: | --- |
| +0 s | controller enables and reads back both NGMO2 outputs; boards boot |
| +25 s | controller sends one common `*TRG`; the 100 s dynamic acquisition begins |
| boot +45 s | initiator dispatches the one configured event at its absolute uptime deadline |
| boot +65 s | preflight-validated worst-case completion deadline |
| about +125 s | 5000 samples at 20 ms are complete on both channels |
| boot +135 s | both boards persist their compact outcome proofs to internal NVS |
| no earlier than about +140 s | controller requests both outputs off; operator verifies OFF at the front panel |
| +2 s or later | next power cycle may begin |

The dynamic current arrays therefore cover approximately +25 s through +125 s.
The NVS write is deliberately outside those arrays. The
`--autonomous-post-capture-s 15` hold keeps the boards powered long enough for
the +135 s write and its bounded retry window; `--autonomous-power-off-s 2` is
the off-time between cycles. Never use `--leave-outputs-on` for this campaign.

The common `*TRG` aligns the two NGMO2 acquisition channels only. The outputs
are enabled sequentially and the two board clocks and boots are independent.
Consequently, `*TRG` is not a firmware trigger and does not identify the event
onset. A separate UART preflight must show that link/setup work finishes before
boot +45 s and that every path, including cleanup, finishes by boot +65 s. If
that bound fails, stop and revise the preregistered schedule before collecting
paper data.

The accepted-discovery build executes the complete
disconnect -> discover -> exchange -> disconnect sequence. Thus its pre- and
post-event guards are both disconnected. The accepted-connected build retains
the connection through both guards.

## 3. Build the four autonomous images

Run from the firmware repository root. Start from a committed, clean final
worktree because `PULSE_FINAL_CAPTURE=ON` rejects dirty or placeholder builds.
Use the same exported deployment artifact and operator-filled lab configuration
for all four images.

| Firmware path | Overlay | CLI `--event-path` | Accepted exchange-ID start |
| --- | --- | --- | ---: |
| local update | `config/pulse_ngmo2_path_local.conf` | `local` | not used |
| no-contact fallback | `config/pulse_ngmo2_path_no_contact.conf` | `no_contact` | not used |
| accepted, connected | `config/pulse_ngmo2_path_accepted_connected.conf` | `accepted_connected` | 1 |
| accepted, discovery | `config/pulse_ngmo2_path_accepted_discovery.conf` | `accepted_discovery` | 106 |

Build each path separately by changing only `$pathOverlay`:

```powershell
$artifact = (Resolve-Path '<exported PULSE deployment artifact .npz>').Path
$labConfig = (Resolve-Path '<operator-filled pulse_lab.conf>').Path
$autonomous = (Resolve-Path .\config\pulse_ngmo2_autonomous.conf).Path
$pathOverlay = (Resolve-Path .\config\pulse_ngmo2_path_accepted_connected.conf).Path
$extra = "$labConfig;$autonomous;$pathOverlay"

cmake --preset senswear_pulse --fresh `
  "-DPULSE_FINAL_CAPTURE=ON" `
  "-DPULSE_ARTIFACT_NPZ=$artifact" `
  "-DEXTRA_CONF_FILE=$extra"
cmake --build --preset senswear_pulse
```

Before building the next path, archive at least
`build/pulse/release/zephyr/zephyr.elf`, `zephyr.hex`, `zephyr.map`,
`zephyr/.config`, `zephyr/zephyr.dts`, the generated fixture manifest, and their
SHA-256 hashes under a path-specific directory. The host capture requires the
full 64-hex-digit firmware SHA-256; use the archived ELF hash consistently:

```powershell
$firmwareSha256 = (Get-FileHash -Algorithm SHA256 `
  .\build\pulse\release\zephyr\zephyr.elf).Hash.ToLowerInvariant()
```

With the debug boards attached, flash the same path image to both SensWear
boards while selecting each probe explicitly:

```powershell
west flash -d build\pulse\release --runner pyocd --dev-id '<BOARD_A_PROBE_UID>'
west flash -d build\pulse\release --runner pyocd --dev-id '<BOARD_B_PROBE_UID>'
```

Use a separate UART preflight to record both immutable board IDs and discover
which physical board becomes the fixed initiator. Role selection follows the
boards' identities; do not swap or relabel logical roles to manufacture a
counterbalance. After preflight, power down and completely remove both debug
boards and FPC cables before attaching the NGMO2.

The autonomous NVS ledger must be empty for that path before its first capture.
If it contains earlier records, export and archive them first, then use the
board-approved erase procedure and reflash. Never erase a board whose proofs
have not been recovered.

## 4. Discover and verify the USB instrument

The real capture path needs PyVISA and a working R&S/IVI VISA backend. Discover
the actual resource string instead of guessing it:

```powershell
python -m pip install pyvisa
```

Install or select the laboratory's approved R&S/IVI VISA runtime separately;
PyVISA is the Python API and is not itself the instrument driver.

```powershell
python .\tools\pulse\pulse_tools.py ngmo2-capture --list-resources
python .\tools\pulse\pulse_tools.py ngmo2-capture `
  --resource '<resource returned above>' --probe
```

`--probe` only sends `*IDN?`. The controller refuses output control unless the
reply identifies a Rohde & Schwarz NGMO2. The NGMO2 manual documents RS-232 and
IEEE-488/GPIB; a USB cable or adapter may therefore appear as either
`ASRL...::INSTR` or `GPIB...::INSTR` rather than native USBTMC. If it appears as
`ASRL`, copy every serial setting from the NGMO2 menu instead of assuming one:

```powershell
python .\tools\pulse\pulse_tools.py ngmo2-capture `
  --resource '<ASRL...::INSTR>' --probe `
  --visa-baud '<300--38400 menu value>' --visa-data-bits '<7 or 8>' `
  --visa-parity '<none, odd, or even>' --visa-stop-bits '<one or two>' `
  --visa-flow-control '<none, rts_cts, or xon_xoff>' `
  --visa-termination '<cr or lf>'
```

Set `--visa-timeout-ms` to the value validated for two 5000-value ASCII array
transfers; the primary command below uses a conservative explicit value.

Use `--dry-run` to review the array-acquisition, trigger, and fetch SCPI commands
without importing PyVISA or changing an output. It deliberately does not print
or validate the voltage, current-limit, OVP, impedance, or output-control
commands and is not a substitute for the live probe and pilot:

```powershell
python .\tools\pulse\pulse_tools.py ngmo2-capture --autonomous --dry-run `
  --sample-interval-s 0.02 --sample-count 5000
```

## 5. Run one dynamic path block

Replace every angle-bracket placeholder. The following command executes 105
power cycles: five warm-ups followed by 100 measured repetitions. Use a new,
empty output directory for every path and every channel/cable assignment.

```powershell
$visaResource = '<verified VISA resource>'
$voltageV = '<approved cell-equivalent voltage>'
$maximumSafeVoltageV = '<documented maximum safe voltage>'
$overvoltageProtectionV = '<approved 0.1-V-step OVP threshold>'
$currentLimitA = '<pilot-validated current limit>'
$currentLimitMarginA = '<preregistered rejection margin below current limit>'
$voltageToleranceV = '<approved output-readback tolerance>'
$bootScheduleUncertaintyS = '<UART-preflight upper bound>'
$runId = 'accepted-connected-map1-r001'
$pairId = 'p01'
$initiatorBoardId = '<immutable initiator board ID>'
$responderBoardId = '<immutable responder board ID>'
$firmwareSha256 = '<archived 64-hex-digit ELF SHA-256>'

python .\tools\pulse\pulse_tools.py ngmo2-capture --autonomous `
  --resource $visaResource `
  --visa-timeout-ms 180000 `
  --output-dir ".\capture\$runId" `
  --run-id $runId --pair-id $pairId `
  --initiator-board-id $initiatorBoardId `
  --responder-board-id $responderBoardId `
  --initiator-firmware-sha256 $firmwareSha256 `
  --responder-firmware-sha256 $firmwareSha256 `
  --firmware-revision '<exact embedded Git revision>' `
  --artifact-sha256 '<exact deployed artifact SHA-256>' `
  --initiator-channel 1 `
  --event-path accepted_connected `
  --capture-count 105 --firmware-sequence-start 0 `
  --sample-interval-s 0.02 --sample-count 5000 `
  --autonomous-trigger-delay-s 25 `
  --autonomous-event-dispatch-s 45 `
  --autonomous-event-deadline-s 65 `
  --boot-schedule-uncertainty-s $bootScheduleUncertaintyS `
  --autonomous-post-capture-s 15 `
  --autonomous-power-off-s 2 `
  --voltage-v $voltageV `
  --current-limit-a $currentLimitA `
  --current-limit-rejection-margin-a $currentLimitMarginA `
  --maximum-safe-voltage-v $maximumSafeVoltageV `
  --overvoltage-protection-v $overvoltageProtectionV `
  --max-output-voltage-deviation-v $voltageToleranceV `
  --sense-mode local `
  --channel-1-cable-id '<cable-set-1 ID>' `
  --channel-2-cable-id '<cable-set-2 ID>' `
  --channel-1-lead-length-m '<total FORCE+ plus FORCE- loop length in metres>' `
  --channel-2-lead-length-m '<total FORCE+ plus FORCE- loop length in metres>' `
  --channel-1-lead-gauge-awg '<AWG>' --channel-2-lead-gauge-awg '<AWG>' `
  --calibration-id '<certificate/inventory ID>' `
  --calibration-date '<YYYY-MM-DD>' `
  --confirm-calibration-current --confirm-output-impedance-zero `
  --confirm-debug-boards-disconnected `
  --confirm-no-parallel-battery `
  --enable-outputs
```

For the other builds, change `--event-path`, the run/output name, and the path
overlay used to build the firmware. The controller derives the accepted-path
exchange IDs from path and `firmware_sequence`; it does not use an
operator-chosen exchange ID as a join key. All paths are associated by the
zero-based `firmware_sequence` and the later proof `sequence`, never by
`event_index`.

The controller performs these safety actions automatically:

- verifies the voltage and current-limit readbacks on both channels;
- programs and verifies independent channel control, 0-ohm output impedance,
  current-limit mode, OVP, and maximum settable voltage/current;
- fixes both dynamic channels to the 0.5 A (`MEDium`) range;
- verifies interval, length, trigger source, and range configuration;
- issues one `*TRG` for both arrays;
- writes one raw two-channel CSV per power cycle and updates
  `ngmo2_capture_manifest.csv` without overwriting an earlier campaign;
- requests both outputs off after every cycle and on any caught error or
  interruption; the operator then verifies both OFF indicators at the front
  panel, including after a host crash or communications loss.

The analysis rejects an array at the documented 510-mA range boundary or within
the declared margin of the programmed current limit. The pre/post current-limit
condition queries are supporting checks, not proof that a shorter transient
never entered limiting; retain pilot evidence and state this limitation.

The first five firmware sequences, 0--4, are warm-ups. Sequences 5--104 are the
100 measured repetitions. Do not delete timeouts or failures; retain them in
the denominator and analyze them in separate outcome strata.

## 6. Counterbalance only the analyzer assignment

The physical boards retain their identity-selected logical roles throughout.
To evaluate channel/cable bias, finish and export one complete 105-cycle block,
turn both outputs off, and then swap the complete channel/cable assemblies:

- the fixed initiator moves from channel 1/A to channel 2/B;
- the fixed responder moves from channel 2/B to channel 1/A;
- the complete `FORCE+`/`FORCE-` lead set moves together;
- `--initiator-channel` changes from `1` to `2`;
- use a new run ID and output directory.

For an equal counterbalance, repeat the complete path block after this swap. The
final protocol writes exactly 105 proof records per board and path. Recover and
archive both boards' first-block proofs before deliberately clearing the ledger
and reflashing the same path image for the second assignment. This is channel/cable
counterbalancing, not logical-role counterbalancing, and it does not remove the
one-pair board/role confounding.

## 7. Recover the post-capture proofs

Proof recovery occurs after a complete block, never during current capture.

1. Confirm both NGMO2 outputs are off at the instrument and disconnect the
   FORCE leads from the board being handled.
2. Reattach that board's debug FPC and debug board.
3. Build the exporter by replacing the autonomous no-console overlay with the
   existing export overlay. Preserve the selected path overlay and artifact:

```powershell
$export = (Resolve-Path .\config\pulse_ngmo2_export.conf).Path
$extra = "$labConfig;$export;$pathOverlay"

cmake --preset senswear_pulse --fresh `
  "-DPULSE_FINAL_CAPTURE=ON" `
  "-DPULSE_ARTIFACT_NPZ=$artifact" `
  "-DEXTRA_CONF_FILE=$extra"
cmake --build --preset senswear_pulse
```

4. Flash this exporter without a recover, mass erase, or other operation that
   erases the storage partition. If the programmer cannot guarantee
   storage-preserving programming, stop and validate the process on a disposable
   pilot record first.
5. Open the board UART at 115200 baud, save the raw output to a board-specific
   log, and reset the board. The exporter first prints the complete
   compile-time `PULSE_META` contract, then the `PULSE_CAPTURE_PROOF` records,
   and ends with
   `PULSE_CAPTURE_STATUS,...,state=export_complete` before holding.
   Because this branch deliberately does not start Bluetooth, its metadata role
   and initial link-observation fields are unresolved/unavailable; they are not
   substitutes for the separate UART timing campaign's per-event link fields.
6. Repeat for the second board.
7. Parse both complete logs together. This also creates the two-board archived
   `run_metadata.csv` release contract required by the power analysis:

```powershell
python .\tools\pulse\pulse_tools.py serial `
  .\export\board-a.log .\export\board-b.log `
  --output-dir ".\recovered-proofs\$runId"
```

8. Join the raw NGMO2 block to the recovered proofs and the hash-checked build
   context:

```powershell
python .\tools\pulse\pulse_tools.py ngmo2-autonomous-power `
  --manifest ".\capture\$runId\ngmo2_capture_manifest.csv" `
  --proofs-csv ".\recovered-proofs\$runId\capture_proofs.csv" `
  --capture-status-csv ".\recovered-proofs\$runId\capture_status.csv" `
  --build-context-csv ".\recovered-proofs\$runId\run_metadata.csv" `
  --output-dir ".\results\$runId"
```

Before accepting the power block, verify for each board and path:

- exactly 105 valid, unique, contiguous `sequence` values from 0 through 104
  on **each** board, including a responder proof for a pre-session failure;
- `warmup=1` for sequences 0--4 and `warmup=0` for 5--104;
- expected path, fixed role, board ID, firmware revision fingerprint, and
  artifact fingerprint;
- `record_ready`, `success`, failure reason, event error, infrastructure error,
  event duration, and result hashes;
- matching exchange IDs between the fixed initiator and responder for accepted
  exchanges;
- one `export_complete` status and no invalid-record/storage errors.

Associate an autonomous raw capture only with the proof having the same path
and `sequence` as the manifest `firmware_sequence`. This is a post-run schedule and
outcome audit, not timestamp synchronization. It must not be used to place an
event boundary in the trace or to join a power trial to the separate UART
timing/communication campaign.

The proof's 32-bit fingerprints bind the embedded Git revision and deployed
artifact digest. The full ELF SHA-256 is capture-manifest provenance, not a
cryptographic field recovered from the board; do not claim that the proof alone
distinguishes differently configured ELFs built from the same revision and
artifact. Preserve the archived `.config`, ELF, hashes, and block log as one
provenance bundle.

## 8. Static idle measurements

Battery-life background power comes from separate static measurements, not from
the dynamic guards. The physical circuit remains NGMO2-only: batteries,
debug boards/FPC cables, upper DMMs, GPIO leads, and phone links remain absent.

Run a static block before the first dynamic power cycle for that path, or only
after its completed proofs have been exported, archived, and the NVS ledger has
been deliberately cleared before reflashing. Once all 105 records for a path
exist, the autonomous image holds before normal sensing/BLE setup; that exhausted
image is not evidence for a declared idle state.

Use the appropriate path image and measure only after its intended pre-event
idle state has stabilized in a separately timed UART preflight. The static
   command must declare `--idle-event-dispatch-s 45`, finish, and request both
   outputs off before the boot +45 s event dispatch. The operator verifies OFF
   at the front panel. The controller checks both a
conservative estimate before power-on and the actual powered elapsed time. If
either check fails, discard the capture and use a separately archived no-event
state-hold configuration; never accept an idle series that contains an event or
NVS write.

Run each declared state into a new directory. This connected-idle example assumes
the fixed initiator is physically on channel 1:

```powershell
python .\tools\pulse\pulse_tools.py ngmo2-capture --static-idle `
  --resource $visaResource `
  --visa-timeout-ms 60000 `
  --output-dir .\capture\idle-connected-map1-r001 `
  --run-id 'idle-connected-map1-r001' --pair-id $pairId `
  --initiator-board-id $initiatorBoardId `
  --responder-board-id $responderBoardId `
  --initiator-firmware-sha256 $firmwareSha256 `
  --responder-firmware-sha256 $firmwareSha256 `
  --initiator-channel 1 `
  --channel-1-state connected_idle `
  --channel-2-state connected_idle `
  --idle-measure-interval-s 0.02 `
  --idle-average-count 5 --idle-samples 30 `
  --idle-current-range medium --idle-overload-threshold-a 0.510 `
  --idle-settle-s 10 --idle-event-dispatch-s 45 `
  --idle-state-evidence '<state/preflight log and image hash>' `
  --voltage-v $voltageV `
  --current-limit-a $currentLimitA `
  --current-limit-rejection-margin-a $currentLimitMarginA `
  --maximum-safe-voltage-v $maximumSafeVoltageV `
  --overvoltage-protection-v $overvoltageProtectionV `
  --max-output-voltage-deviation-v $voltageToleranceV `
  --sense-mode local `
  --channel-1-cable-id '<cable-set-1 ID>' `
  --channel-2-cable-id '<cable-set-2 ID>' `
  --channel-1-lead-length-m '<total FORCE+ plus FORCE- loop length in metres>' `
  --channel-2-lead-length-m '<total FORCE+ plus FORCE- loop length in metres>' `
  --channel-1-lead-gauge-awg '<AWG>' --channel-2-lead-gauge-awg '<AWG>' `
  --calibration-id '<certificate/inventory ID>' `
  --calibration-date '<YYYY-MM-DD>' `
  --confirm-calibration-current --confirm-output-impedance-zero `
  --confirm-debug-boards-disconnected `
  --confirm-no-parallel-battery `
  --enable-outputs
```

For disconnected paths, label only the stationary states actually demonstrated
by the state evidence, mapped to channel 1/2 by the physical wiring. Do not call
the initiator `scanning` unless a dedicated state-hold image proves continuous
scanning. The energy of finite scan/discovery attempts is already captured by
the no-contact or discovery event and must not be counted again as idle power.
Repeat for every state used in the battery-residency model and for both
channel/cable assignments.

The static range is mandatory. Use `low` (5 mA, 5.1-mA valid boundary) only for
a pilot-proven quiet state; BLE radio-active states should normally begin on
`medium` (0.5 A, 510-mA valid boundary). The command records sequential
per-channel queries, not synchronized pair-current data, and rejects readings at
the declared boundary. An overload, state transition, missed pre-event deadline,
current-limit condition, or proof/NVS write invalidates the idle capture; use
the less sensitive documented range rather than accepting clipping.

## 9. Energy and battery calculations

The dynamic samples are interval means. For board `b`, capture `k`, interval
`dt = 0.02 s`, and all `N = 5000` bins, define the same-range guard current as
the average of the pre- and post-guard means:

```text
I_guard[b,k] = (mean(I_pre[b,k]) + mean(I_post[b,k])) / 2
```

Reject the capture if the two guards are not in the same declared state or fail
the preregistered drift/stationarity check. Let `V[b,k]` be the mean of that
channel's pre/post output-voltage readbacks, then integrate the entire 100 s
array:

```text
E_total[b,k] = V[b,k] * dt * sum_n(I[b,k,n])
E_inc[b,k]   = V[b,k] * dt * sum_n(I[b,k,n] - I_guard[b,k])
```

Do not apply trapezoidal endpoint weighting: every returned NGMO2 value is the
mean current within its configured 20 ms interval. Quiet padding cancels in
`E_inc`, so no visually selected firmware boundary is needed.

For an accepted exchange captured under the common NGMO2 trigger:

```text
E_inc_pair[k] = E_inc[A,k] + E_inc[B,k]
P_pair_bin_max[k] = max_n(V[A,k]*I[A,k,n] + V[B,k]*I[B,k,n])
```

Never add two independent per-channel maxima. Report this quantity as the
maximum 20 ms interval-average power, not instantaneous or peak-transient
power.

For static idle state `s`:

```text
P_idle[b,s] = V[b,s] * mean(I_idle[b,s])
```

The controller has already written this value as `mean_power_w` in
`ngmo2_idle_summary.csv`; use the row matching the exact board and state as the
battery input's `idle_power_w`. Use the measured `energy_incremental_j` samples
from `event_power_metrics.csv` for the matching path/role/outcome event class.
Do not pool different physical boards or silently substitute a dynamic guard for
a missing static state measurement.

Expected daily energy uses arithmetic means of the measured event strata:

```text
E_day[b] = sum_s(T[b,s] * P_idle[b,s])
           + sum_k(N_events[b,k] * mean(E_inc[b,k]))
```

with `sum_s T[b,s] = 86400 s`. For capacity `C_mAh`, nominal cell voltage
`V_nominal`, and explicit usable-energy fraction `eta`:

```text
E_battery_J = 3.6 * C_mAh * V_nominal * eta
life_days = E_battery_J / E_day
```

Use capacity and voltage from the exact cell data sheet. Report low, nominal,
and high workload/residency/usable-fraction cases as sensitivity scenarios.
Substituting an IQR endpoint or p95 energy is a separately labelled stress case,
not the expected-value calculation and not a confidence interval.

## 10. Claim boundaries

After the proof and quality audits pass, this setup supports claims about:

- per-board NGMO2 current arrays on the stated range and interval;
- full-capture total and guard-subtracted incremental energy;
- same-acquisition two-board energy and maximum 20 ms pair-average power;
- range-qualified static idle power for explicitly declared board states;
- attempt, success, timeout, and failure strata from the recovered NVS proofs;
- assumption-labelled battery-life projections.

It does not support claims about:

- instantaneous current or transient peak power below the 20 ms bin;
- energy of individual software, BLE, inference, or training stages;
- exact event onset in the NGMO2 trace or microsecond board-to-board ordering;
- trial-level energy/latency or energy/communication correlation;
- over-the-air BLE bytes, packets, advertisements, PHY/MAC headers,
  retransmissions, airtime, or goodput;
- energy attributable to the debug boards, NVS proof write, upper DMMs, or a
  hardware BLE sniffer;
- measured discharge lifetime or population-level device variation.

Run latency, stage timing, correctness, runtime-memory diagnostics, and
application/GATT counters in separate debug-board UART campaigns. A phone BLE
scanner may be used only as a bring-up sanity check; it is not a hardware
link-layer sniffer and must remain disconnected during recorded power trials.
