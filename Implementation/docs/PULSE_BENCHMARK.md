# PULSE Section V benchmark runbook

This is the operator-facing procedure for the two-board SensWear PULSE experiment. It covers the
real-artifact build, synchronized capture, audit, and production of the Section V figure inputs.
The detailed serial, power-integration, statistics, and CSV schemas remain in
[`tools/pulse/README.md`](../tools/pulse/README.md).

> **No board was flashed and no hardware measurement was made while implementing this code.** The
> current build trees use the supplied deploy encoder/data and provenance-bound seed-7 common head,
> but were compiled with `PULSE_FINAL_CAPTURE=OFF` while the implementation worktree is uncommitted.
> They are functional-validation images, not experimental results. Final figures still require real
> two-board serial/current/GPIO captures, and real linker/runtime artifacts.

> **Encoder-freezing contract.** The supplied PULSE tree now calls
> `encoder.requires_grad_(not config.model.freeze_encoder)`, so the checked-in
> `freeze_encoder: true` disables encoder gradients as intended. Controlled simulation results made
> before this correction must be regenerated before comparison with the frozen, head-only firmware.

> **Remote-gradient normalization contract.** The executable PULSE repository is authoritative:
> `scale = sqrt(local_norm_sq / max(remote_norm_sq, epsilon))`. The fixture generator, firmware, and
> correctness reference use the same reduction. The epsilon floor is applied to the squared remote
> norm before the square root, which defines the behavior for a near-zero remote gradient.

> **Keep the paper wording aligned with the measured image.** The compact UART records are retained
> but emitted only after the GPIO event gate falls; the DWT counter is an active-core interval that
> includes interrupts/preemption, not pure kernel compute time; and the fuel-gauge driver is disabled
> in the matched release/baseline images. A responder success ends at CRC-valid gradient-receipt ACK,
> before the initiator's decode/mix result is known, so a successful *pair* transaction must require a
> joined successful row from both boards. The automatic within-boot path order is fixed; use complete
> blocks, bracketing baselines, multiple physical pairs, and explicitly report any residual order
> limitation.

## 1. Prerequisites and identifiers

- Load the NCS/Zephyr and CMake environment described in [`SETUP.md`](../SETUP.md). Use the same
  commit, NCS/Zephyr SDK, compiler, optimization, board revision, supply, and RF arrangement for the
  PULSE and baseline builds.
- Install Python 3, NumPy, PyTorch, PyYAML, and h5py for artifact export. PyTorch is also required for the
  final correctness reference. Matplotlib is optional and is needed only to render PDF/PNG; the
  analysis always writes figure-ready CSVs.
- Obtain the pretrained PULSE HAR encoder and processed HAR data, including the per-window
  `node_ids`. The upstream simulator does not train or save a common head: it creates the same
  deterministic fresh `init_id="common"` head for every node from the simulation seed. The exporter
  can reproduce that exact head from the archived PULSE source/config/seed. Alternatively, supply an
  explicitly captured personalized-head snapshot and identify where in the simulation it came from.
- Prepare exactly two SensWear `nRF54L15/cpuapp` boards with no daughter shield, two independently
  selectable debug probes, two lossless UART capture paths, and a synchronized two-channel power
  analyzer with eight digital inputs. A synchronized BLE sniffer is mandatory if link-layer bytes
  or packets will be reported.
- Assign a non-personal `pair_id` such as `p01` and a unique capture `run_id`, for example
  `20260904-p01-release-r01`. One pair with 100 repetitions is one physical pair, not 100
  independent pairs.

Create capture directories before starting any logger:

```powershell
New-Item -ItemType Directory -Force artifacts, capture\p01, results\p01 | Out-Null
```

## 2. Export a real, frozen artifact

In legacy mode, the input data archive must contain `x:[N,3,128]`, integer `y:[N]` with labels 0--5, and
`node_ids:[N]`. Use pseudonymous owner IDs. The exporter selects four batches of 16. The first three
batches (two scheduled steps and the initiator-local gradient) must belong to one owner; the fourth
must belong to one distinct responder owner. Prefer an explicit, archived `replay_indices:[4,16]`
array so every build replays the same 64 real windows. The checked-in PULSE processed archives are
already normalized, so the default is to verify their per-window/per-channel normalization, not to
apply it again.

This fixed replay exercises full 16-sample batches only; it does not characterize a shorter tail
batch that may occur in the simulator. The current reproducibility image also embeds all four replay
batches on both boards, including both pseudonymous owners. The BLE protocol still transmits only a
head and a gradient, but this image does **not** demonstrate at-rest separation of private samples.
Qualify any privacy claim accordingly; a storage-isolation claim requires role-specific artifacts or
a device-private data path and a separate correctness binding.

The supplied `PULSE/deploy` bundle is the authoritative HAR deployment input. Its frozen encoder
contains Conv1d--BatchNorm blocks, whereas the embedded inference kernel uses biased Conv1d layers.
The exporter loads the exact architecture from the supplied `hhar_device_setup.py`, validates the
checkpoint strictly, and uses PyTorch's evaluation fusion to fold each BatchNorm into its preceding
convolution. It verifies the original and folded encoders on the exact selected 64 windows, including
embedding/logit tolerances and identical predictions, and records both source and transformed hashes.

For the Section-V artifact, use the canonical seed-7 PULSE `init_id=common` head and bind its
hyperparameters and 70/15/15 split to `community_hhar_pulse.yaml`. The YAML's old encoder/data paths
are not silently treated as matches: provenance identifies both as explicit deploy-bundle overrides.
Run from the firmware repository root. Keep the output outside this Git worktree, because
`PULSE_FINAL_CAPTURE=ON` rejects every dirty or untracked firmware path:

```powershell
$experimentRoot = (Resolve-Path '..\Academic Collaboration').Path
$pulseRoot = Join-Path $experimentRoot 'PULSE'
$deployBundle = Join-Path $pulseRoot 'deploy'
$artifactDir = Join-Path $experimentRoot '_pulse_capture_inputs'
New-Item -ItemType Directory -Force $artifactDir | Out-Null
$artifact = Join-Path $artifactDir 'pulse_hhar_seed7_a_b.npz'

python tools\export_pulse_artifact.py `
  --deploy-bundle $deployBundle `
  --deploy-head-policy pulse-common `
  --pulse-source-root $pulseRoot `
  --pulse-config (Join-Path $pulseRoot 'configs\community_hhar_pulse.yaml') `
  --simulation-seed 7 `
  --initiator-node a `
  --responder-node b `
  --deploy-source-split simulation `
  --deploy-train-fraction 0.70 `
  --deploy-val-fraction 0.15 `
  --normalization verify `
  --replay-split-id community_hhar_seed7_train `
  --replay-event-id fixed_microbenchmark_a_b_train_prefix_v1 `
  --validate-only

python tools\export_pulse_artifact.py `
  --deploy-bundle $deployBundle `
  --deploy-head-policy pulse-common `
  --pulse-source-root $pulseRoot `
  --pulse-config (Join-Path $pulseRoot 'configs\community_hhar_pulse.yaml') `
  --simulation-seed 7 `
  --initiator-node a `
  --responder-node b `
  --deploy-source-split simulation `
  --deploy-train-fraction 0.70 `
  --deploy-val-fraction 0.15 `
  --normalization verify `
  --replay-split-id community_hhar_seed7_train `
  --replay-event-id fixed_microbenchmark_a_b_train_prefix_v1 `
  --output $artifact
```

This reproduces `PULSE.data.make_node_splits`: one NumPy generator seeded with 7 shuffles every
lexicographically sorted owner, after which the first 48 training indices for `a` and first 16 for
`b` become the four replay roles. This is a deterministic microbenchmark replay, not an observed
contact event. For the supplied bundle, require folded-encoder digest
`f89d70f80dbf093633acc1b23fdfbf987b7c38bf0c26ec8fcbe4019416af56aa` and common-head
fingerprint `2d278705fb367b02492d532d2211ada467f54b5b31996d324834531ba48010c8`
in validation/provenance output.

The explicit alternative `--deploy-head-policy node-seeded --deploy-head-seed 42` reconstructs the
physical setup script's node-`a`, position-zero head. A separately archived exact head can instead be
passed with `--head-checkpoint`. These policies are mutually exclusive; neither is an implicit
fallback. The canonical Section-V command above uses `pulse-common` so both peers begin from the same
controlled-evaluation head.

The legacy NPZ/checkpoint mode remains available for other controlled artifacts. It checks that the
selected seed and Section-V hyperparameters occur in the supplied YAML, that the YAML-declared
encoder checkpoint has the same SHA-256 as `--checkpoint`, and that the supplied
`models.py`/`utils/commons.py` code produces the independently reconstructed common head.

If a combined checkpoint contains an explicitly saved head, omit the three canonical-seed flags. If
the encoder and exact head snapshot are separate, use `--head-checkpoint`. Neither path may invent an
implicit random head. Use `--normalization apply` only for genuinely raw, unnormalized windows.
Preserve the exported NPZ and its generated sibling `.provenance.json`, along with hashes of every
input:

```powershell
Get-FileHash -Algorithm SHA256 `
  $artifact, `
  ($artifact -replace '\.npz$','.provenance.json'), `
  (Join-Path $deployBundle 'uci_har_encoder.pt'), `
  (Join-Path $deployBundle 'data.h5'), `
  (Join-Path $deployBundle 'manifest.csv'), `
  (Join-Path $deployBundle 'hhar_device_setup.py'), `
  (Join-Path $pulseRoot 'configs\community_hhar_pulse.yaml')
```

The exporter rejects missing tensors, wrong shapes/dtypes, non-finite values, invalid labels,
ambiguous state-dict prefixes, duplicate/out-of-range replay indices, mixed initiator ownership, and
a responder batch owned by the initiator. `--allow-owner-role-mismatch` is diagnostic only; the
firmware fixture generator rejects such an artifact. Do not bypass these checks.

## 3. Configure and build all three presets

Run these commands from the firmware repository root. `PULSE_ARTIFACT_NPZ` is required for the
release and correctness images used in the final experiment. The baseline intentionally contains
no model and therefore does not consume the artifact.

```powershell
$artifact = (Resolve-Path $artifact).Path
$labConfig = (Resolve-Path C:\path\to\pulse_lab.conf).Path
$pythonWithTorch = (Get-Command python).Source

cmake --preset senswear_pulse --fresh `
  "-DPULSE_FINAL_CAPTURE=ON" `
  "-DPULSE_ARTIFACT_NPZ=$artifact" `
  "-DEXTRA_CONF_FILE=$labConfig"
cmake --build --preset senswear_pulse

cmake --preset senswear_pulse_baseline --fresh `
  "-DPULSE_FINAL_CAPTURE=ON" `
  "-DEXTRA_CONF_FILE=$labConfig"
cmake --build --preset senswear_pulse_baseline

cmake --preset senswear_pulse_correctness --fresh `
  "-DPULSE_FINAL_CAPTURE=ON" `
  "-DPULSE_ARTIFACT_NPZ=$artifact" `
  "-DPULSE_FIXTURE_PYTHON:FILEPATH=$pythonWithTorch" `
  "-DEXTRA_CONF_FILE=$labConfig"
cmake --build --preset senswear_pulse_correctness
```

`PULSE_FIXTURE_PYTHON` selects only the fixture/golden-vector generator; do not replace Zephyr's
own `Python3_EXECUTABLE`. The correctness configure now fails unless that interpreter imports
PyTorch, so a NumPy fallback cannot accidentally be reported as the simulation reference.
`PULSE_FINAL_CAPTURE=ON` also rejects a missing real artifact, a placeholder hardware revision,
an unknown Git revision, or any dirty/untracked worktree content. These provenance checks run both
at configure time and before every normal application build; changing the commit, artifact bytes, or
worktree after configuration therefore stops the build. Development/synthetic builds remain possible
only without that guard and identify themselves as `build_guard=development_validation_only`.

Start from [`config/pulse_lab.conf.example`](../config/pulse_lab.conf.example) and record the exact
PCB/assembly revision in `CONFIG_SENSWEAR_PULSE_HARDWARE_REVISION`. The default
`operator_required` is deliberately not a board-target guess and is unacceptable in a final
capture.

The presets have distinct purposes:

| Preset | Build directory | Use |
| --- | --- | --- |
| `senswear_pulse` | `build/pulse/release` | Release timing, power, and resource image |
| `senswear_pulse_baseline` | `build/pulse/baseline` | Matched sensing+BLE, no-PULSE baseline |
| `senswear_pulse_correctness` | `build/pulse/correctness` | PyTorch-vector validation only |

The correctness image embeds large golden vectors. Never use its flash, RAM, latency, or power as
the release result. Check that release/correctness manifests say `exported_pulse_npz` and that the
correctness reference says `pytorch_cpu_float32`:

```powershell
Get-Content build\pulse\release\generated\pulse\pulse_fixture_manifest.json
Get-Content build\pulse\correctness\generated\pulse\pulse_fixture_manifest.json
```

If either manifest says `deterministic_synthetic`, stop: that image is a synthetic replay build and
is not eligible for a paper figure or quantitative claim.

Archive, for every image, `zephyr.elf`, `zephyr.hex`, `zephyr.map`, `zephyr/.config`,
`zephyr/zephyr.dts`, the generated fixture manifest, the ELF SHA-256, the Git revision, and toolchain
versions. `PULSE_META` must later match those archived artifacts. The current real-artifact,
development-validation build gives the following linker sanity values; because
`PULSE_FINAL_CAPTURE=OFF`, they are not measurements to copy into the paper:

| Image | Linker flash (bytes) | Static RAM (bytes) |
| --- | ---: | ---: |
| Release | 574,416 | 153,304 |
| Baseline | 358,728 | 59,224 |
| Release minus baseline | 215,688 | 94,080 |
| Correctness | 632,856 | 231,320 |

Recompute the table from the final real-artifact maps. The linker usage symbols are authoritative;
the category split is conservative because the mutable 18,200-byte head is a field inside the
composite `.pulse.state` object and a map alone cannot split structure members.

`CONFIG_SENSWEAR_PULSE_PAIR_ID` defaults to `operator_required`. A pair-specific image can set it at
configure time, but the preferred one-binary campaign leaves the image unchanged and attaches both
`pair_id` and unique `run_id` during serial import with `--set` (Section 8).

The release overlay retains the BHI360 sensing stream and BLE link in both release and baseline,
but disables the unused M95P, charger, fuel-gauge, and daughter-board drivers. The frozen model and
replay windows are linked into internal flash; this image neither loads them from M95P nor measures
M95P energy. Because the battery/charger path is isolated and the BQ27427 driver is absent, it also
cannot provide in-run fuel-gauge status. If battery health must be checked, do that as a separately
archived, battery-attached preflight (for example with the `test_bq27427` image), then restore and
verify the isolated analyzer setup. Record these boundaries in the paper rather than attributing
the disabled components to the measured execution path.

## 4. Flashing status and operator commands

The commands below are instructions for the lab operator; they were **not** executed as part of this
implementation. Program the same release binary on both boards. With two probes attached, always
select the probe explicitly so the wrong board is not overwritten:

```powershell
west flash -d build\pulse\release --runner pyocd --dev-id <BOARD_A_PROBE_UID>
west flash -d build\pulse\release --runner pyocd --dev-id <BOARD_B_PROBE_UID>
```

Use `build\pulse\baseline` for the matched baseline block and `build\pulse\correctness` for the
separate correctness block. Follow the board's approved recovery/programming procedure if the local
probe runner differs. Record the programmed ELF hash for each board before capture.

## 5. Power, marker, UART, and sniffer wiring

Disconnect or isolate each battery/charger path according to the board schematic and approved lab
procedure. Power the two boards from separate, isolated analyzer channels at the same recorded
voltage. Never join analyzer outputs. Add only the common digital reference required by the
instrument, and check for UART/analyzer ground loops before energizing the boards.

The benchmark owns the otherwise-unused daughter connector pins below. Do not fit a daughter shield
or connect any driven peripheral to these pins during the campaign.

| Signal | Daughter signal | MCU pin | Meaning |
| --- | --- | --- | --- |
| Event gate | GPIO0 | P1.14 | Active high for one complete event |
| Stage bit 0 | GPIO1 | P1.13 | Least-significant code bit |
| Stage bit 1 | GPIO2 | P1.12 | Middle code bit |
| Stage bit 2 | GPIO3 | P1.11 | Most-significant code bit |

The LSB-first stage codes are: 0 event/unspecified, 1 selection, 2 encoder, 3 head
forward/backward, 4 serialization/deserialization, 5 radio transfer, 6
agreement/normalization/mixing/utility, and 7 BLE control/scan/connect. Verify active-high polarity
and all relevant codes with the analyzer before accepting a power block.

Use one analyzer clock for both current channels and all eight digital lines. If two instruments are
unavoidable, drive both from a common hardware synchronization signal and quantify residual skew.
The final sampling rate must provide at least ten samples in the shortest reported stage after
instrument filtering; otherwise merge that stage into a coarser interval.

UART30 is 921600 baud, 8 data bits, no parity, one stop bit. The MCU pins are P0.03 TX and P0.04 RX;
use the actual board connector and voltage levels from the schematic. Capture each board in a
separate terminal. The following dependency-free PowerShell logger can be pasted into each terminal
after changing `$portName` and `$logPath`:

```powershell
$portName = 'COM7'
$logPath = (Join-Path (Get-Location) 'capture\p01\board-a.log')
$serial = [IO.Ports.SerialPort]::new(
  $portName, 921600, [IO.Ports.Parity]::None, 8, [IO.Ports.StopBits]::One)
$serial.NewLine = "`n"
$writer = [IO.StreamWriter]::new($logPath, $false, [Text.UTF8Encoding]::new($false))
$writer.AutoFlush = $true
$serial.Open()
try {
  while ($true) {
    $line = $serial.ReadLine().TrimEnd("`r")
    $writer.WriteLine($line)
    $line
  }
} finally {
  $serial.Close()
  $writer.Dispose()
}
```

Start both UART loggers, the power/GPIO acquisition, and the BLE sniffer before resetting either
board. Preserve raw logs and analyzer exports unchanged. Firmware byte counters describe the PULSE
application/ATT-value transfer. Client TX counters mean successfully submitted to the Bluetooth
host and server RX counters mean accepted by the GATT callback; neither proves over-air delivery.
They do **not** include link-layer headers, retransmissions, connection events, or other PHY/MAC
traffic; only the synchronized sniffer capture can support those claims.

## 6. Exact two-board automatic campaign

Use the default `CONFIG_SENSWEAR_PULSE_ROLE_OVERRIDE=0`. Both boards initially advertise and scan.
After they see one another, the board with the lexicographically lower canonical 48-bit identity
address becomes the logical initiator and the other becomes the responder. The mapping is stable
for that pair and is printed in `PULSE_META`/`PULSE_STATUS`; never infer it from USB port names.

1. With capture already running, power or reset both boards close together. Do not start acquisition
   after the first gate edge.
2. Require one `PULSE_META` line per board and a status line with `marker_errno=0`,
   `sensing_errno=0`, `link_errno=0`, and a resolved role. Then require `protocol_errno=0`. The link
   setup negotiates the benchmark connection before the automatic campaign starts.
3. The firmware waits `CONFIG_SENSWEAR_PULSE_START_DELAY_MS=5000`. Keep this gate-low interval in
   the analyzer trace.
4. The initiator automatically runs, in order, `local`, `no_contact`, `accepted_connected`, and
   `accepted_discovery`. Each path contains exactly five recorded warm-ups followed by 100 measured
   attempts with the checked-in configuration. `local` produces a connected `role=local` record.
   Before the no-contact block, the initiator disconnects once; every no-contact gate starts in the
   disconnected state, scans until it observes the known responder, rejects that reachable peer by
   the controlled UCB branch, and performs the one-step local fallback without connecting. The
   public policy setup represents one prior signed observation: `s=1,n=1,U=-0.2` for no-contact and
   `s=1,n=1,U=+0.2` for accepted exchange, keeping both branches reachable through the actual
   UCB/EMA APIs. These
   attempts also retain `role=local`. The firmware reconnects outside a gate before the
   `accepted_connected` block. Each accepted attempt produces an initiator record. It produces exactly one
   responder record when `remote_session_started=1`; a discovery/connect failure before the remote
   session begins has `remote_session_started=0` and correctly has no responder record or gate.
   Accepted-discovery trials disconnect first, and the initiator gate includes scan/connect.
   In those trials, `ble_scan` ends when the known responder advertisement is observed;
   `ble_connect` then spans connection establishment **and** PULSE GATT service/characteristic
   discovery. It is therefore an end-to-end setup interval, not controller connection latency alone.
5. Warm-up rows and their GPIO windows are retained with `warmup=1`; measured rows use `warmup=0`.
   `event_index` is monotonic across the run: 0--104 local, 105--209 no-contact, 210--314 connected
   exchange, and 315--419 discovery exchange. Measured `trial_id` restarts at 0 after each path's
   warm-up. GPIO windows are aligned to each board's monotonic `event_index`; initiator/responder
   protocol records are paired only by exact `run_id,pair_id,exchange_id`, never by `trial_id`.
   After each successful remote session, both peers close their measured gates before the initiator
   sends a `RESULT_RELEASE` control handshake. On a measured failure after `HEAD_BEGIN`, the
   initiator first sends `CANCEL` inside its gate. The responder keeps STATUS at `PROCESSING` until
   its worker has stopped, closes its own gate, and then publishes `CANCELLED`; observing that status
   proves both gates are low before `RESULT_RELEASE`. If cancellation cannot be proven by the event
   deadline, the initiator disconnects and sends no release. `RESULT_RELEASE` is excluded from both
   peers' protocol counters and never extends either GPIO interval. Only release (or the bounded
   disconnect fallback) frees the responder record. The responder freezes that completed event
   once, keeps UART silent until coordination is safe, and prints the original `success`,
   `failure_reason`, and `exchange_id`. When the measured exchange had already failed, a secondary
   cancel/release failure does not abort the remaining repetitions. A release failure after a
   successful exchange remains a capture-integrity error and aborts the campaign; it does not
   rewrite or extend the successful measured event.
6. Keep all capture systems running until the initiator prints `state=campaign_complete`, the final
   responder record has arrived, and at least five seconds of gate-low postamble has been acquired.
7. A natural timeout/disconnect remains in its attempted path with `success=0` and a failure reason.
   It counts in the failure-rate denominator. Never delete it or silently rerun only the failed
   trial. If an entire boot/capture is invalid, archive it as rejected and restart a new complete
   block with a new `run_id`. In particular, an accepted-discovery scan, connect, negotiation, or
   GATT-discovery failure before `HEAD_BEGIN` is quiesced after its row and the campaign proceeds to
   the next disconnected-start repetition; it is not followed by a mandatory out-of-gate reconnect.

Every repetition intentionally reloads the same initial head and reconstructs the controlled peer
state so paths can be compared as independent microbenchmark trials. With two boards, only one peer
occupies the configured table. These measurements therefore characterize the stated event paths,
not long-lived personalization dynamics, accumulated utility history, or multi-peer table scaling;
those remain properties of the controlled simulator or require a separate deployment experiment.

Automatic roles cannot be swapped at trial 50: identity ordering deliberately makes them stable.
For placement/cable counterbalancing, finish a complete block, stop capture, rotate the boards'
physical positions and analyzer/UART cables, then reset both boards for a new complete block. Rename
analyzer columns to the logical roles printed by the firmware. The primary one-image experiment must
not use fixed-role overrides; use multiple physical pairs, or separately preregistered override
builds, if a logical-role crossover is scientifically required.

## 7. Baseline and correctness blocks

Bracket the release run with matched-baseline captures when practical. Flash the same baseline image
to both boards, start synchronized current/GPIO/UART capture, reset both boards, and wait for
`state=matched_baseline_idle,connection=retained`. Acquire at least five seconds of stationary
connected+sensing current per role. No event gates are expected in this image. Repeat after the
release block and investigate drift rather than pooling incompatible baselines.

Set these four values to the measured/configured lab values and reuse them unchanged for the
baseline and release records from this block. The final figure gate rejects these sentinel strings,
so leaving one unfilled cannot silently produce a paper figure:

```powershell
$supplyVoltageV = 'REPLACE_WITH_NUMERIC_VOLTS'
$analyzerSampleRateHz = 'REPLACE_WITH_NUMERIC_HZ'
$analyzerFilter = 'REPLACE_WITH_EXACT_SETTING_OR_none'
$deviceSeparationM = 'REPLACE_WITH_NUMERIC_METRES'
```

Parse the baseline UART capture as its own build/capture identity; its `PULSE_META` record is later
used to bind the archived baseline ELF to the resource delta:

```powershell
python tools\pulse\pulse_tools.py serial `
  capture\p01\baseline-board-a.log capture\p01\baseline-board-b.log `
  --set pair_id=p01 `
  --set run_id=20260904-p01-baseline-pre `
  --set "supply_voltage_v=$supplyVoltageV" `
  --set "analyzer_sample_rate_hz=$analyzerSampleRateHz" `
  --set "analyzer_filter=$analyzerFilter" `
  --set "device_separation_m=$deviceSeparationM" `
  --output-dir results\p01\baseline-pre
```

For correctness, flash the same real-artifact correctness image to both boards and capture UART from
boot. Accept only `PULSE_CORRECTNESS` rows with `pass=1`, the expected artifact/fixture hash, and
`reference=pytorch_cpu_float32`. Preserve all component errors and cosine similarities. Stop this
separate validation after the required records if desired; do not mix it into release statistics.

Parse that capture independently:

```powershell
python tools\pulse\pulse_tools.py serial `
  capture\p01\correctness-board-a.log capture\p01\correctness-board-b.log `
  --set pair_id=p01 `
  --set run_id=20260904-p01-correctness-r01 `
  --output-dir results\p01\correctness-r01
```

Review `correctness.csv` and retain `run_metadata.csv` beside it. A parser-produced row is not proof
of correctness unless its firmware record, hashes, reference identity, and pass criteria are valid.

## 8. Export and analyze the real capture

Export analyzer data without resampling. For one synchronized two-channel CSV, use this header
(a varying role-prefixed voltage column may be added):

```text
time_s,initiator_current_a,responder_current_a,initiator_gpio0,initiator_gpio1,initiator_gpio2,initiator_gpio3,responder_gpio0,responder_gpio1,responder_gpio2,responder_gpio3
```

If roles were unknown at acquisition time, retain the immutable raw export, make a documented
analysis copy, and rename physical A/B columns according to `PULSE_META`. Separate role-labelled
files may instead use `time_s,current_a,event_gate,stage_bit0,stage_bit1,stage_bit2`.

Parse both release UART logs and attach the external campaign identifiers:

```powershell
$raw = 'results\p01\release-r01'
python tools\pulse\pulse_tools.py serial `
  capture\p01\board-a.log capture\p01\board-b.log `
  --set pair_id=p01 `
  --set run_id=20260904-p01-release-r01 `
  --set "supply_voltage_v=$supplyVoltageV" `
  --set "analyzer_sample_rate_hz=$analyzerSampleRateHz" `
  --set "analyzer_filter=$analyzerFilter" `
  --set "device_separation_m=$deviceSeparationM" `
  --output-dir $raw
```

This writes `run_metadata.csv`, `event_metrics.csv`, `stage_metrics.csv`, and `correctness.csv`.

First extract the matched baseline estimates from its all-gate-low capture:

```powershell
python tools\pulse\pulse_tools.py power `
  --input initiator=capture\p01\baseline-initiator.csv `
  --input responder=capture\p01\baseline-responder.csv `
  --metadata-csv results\p01\baseline-pre\run_metadata.csv `
  --baseline-only `
  --voltage-v $supplyVoltageV `
  --output-dir results\p01\baseline-pre
```

`--baseline-only` requires every event gate to remain low and records
`capture_mode=baseline_only`, `event_window_count=0`, and `all_event_gates_low=1`. Its authoritative
baseline is trapezoidal energy divided by the complete capture duration, so periodic sensing/BLE
spikes remain represented; the instantaneous sample median is diagnostic only. Compare pre/post
captures and preregister the selected archive. Then bind the release analysis directly to that
metadata file (including its SHA-256 and raw-capture source):

```powershell
python tools\pulse\pulse_tools.py power `
  --input capture\p01\release-pair.csv `
  --events-csv "$raw\event_metrics.csv" `
  --metadata-csv "$raw\run_metadata.csv" `
  --voltage-v $supplyVoltageV `
  --baseline-metadata results\p01\baseline-pre\power_capture_metadata.csv `
  --output-dir $raw
```

The power command writes `event_power_metrics.csv`, `stage_power_metrics.csv`, `power_trace.csv`,
and `power_capture_metadata.csv`. It requires one run/pair/board identity with unique contiguous
event indices and an exact serial-event/GPIO-window count; `--allow-unmatched` cannot waive a
partial chronological join. Exclude any edge-truncated capture.
The diagnostic flag never waives accepted-session integrity: duplicates, responder orphans, a
missing responder event/gate after `remote_session_started=1`, and missing exact session keys remain
hard errors. For `remote_session_started=0`, it emits an initiator-only failure system row only when
both synchronized analyzer channels cover the complete initiator gate, integrating both channels
over that interval. For a started session, paper-facing initiator and responder rows are both
integrated over the same union of their GPIO windows and sum exactly to the pair row. Their
own-gate integrations remain distinctly labeled diagnostics; stage-power summaries also retain
`stage_sequence` so repeated marker phases are not pooled.
The analyzer channel remains identified by `power_channel_role`. Local and no-contact firmware rows
retain `role=local`, but are joined to the initiator power channel only when their immutable
`board_id` matches the board mapped to `role=initiator` in `run_metadata.csv`; they are never joined
to the responder channel.

Export the synchronized BLE capture to the strict canonical PDU CSV described in
`tools/pulse/README.md`; do not pass a PCAP directly to the analysis code. Keep the untouched PCAP
as the provenance artifact. A shared analyzer/sniffer clock requires an explicit worst-case timing
uncertainty:

```powershell
python tools\pulse\pulse_tools.py link `
  --input capture\p01\canonical-pdus.csv `
  --event-windows-csv "$raw\event_power_metrics.csv" `
  --metadata-csv "$raw\run_metadata.csv" `
  --raw-pcap cap01=capture\p01\untouched.pcapng `
  --shared-timebase --time-uncertainty-s <MEASURED_BOUND_S> `
  --output-dir $raw
```

For a separate sniffer clock, replace the last synchronization options with
`--sync-anchors capture\p01\sniffer-sync.csv`. Supply at least two uncertainty-labelled affine
anchors per `capture_id`, bracketing the entire capture; extrapolation is rejected. The canonical
rows carry the transmitter/receiver BLE identity and address type. They are checked against
`local_identity_address`, `local_identity_address_type`, `peer_identity_address`, and
`peer_identity_address_type` from both boards' `PULSE_META`, so logical directions are not inferred
from USB or sniffer labels.

`link_layer_bytes` is the two-byte BLE data-channel LL header plus the number of octets encoded by
its `Length` field for every sniffer-observed CRC-valid PDU in the event. It excludes preamble,
access address, CRC, and inter-frame spacing. The importer separately records unique and retransmitted bytes
and validates data-channel retry labels using connection epoch, direction, SN, LLID, length, and a
SHA-256 of the exact `Length` payload bytes. Known capture gaps, CRC failures, uncertain boundaries,
partial coverage, duplicate indices/windows, invalid role identities, and inconsistent accepted
session records fail `link_audit.csv`. The command writes `link_packet_metrics.csv`,
`event_link_metrics.csv`, `link_capture_metadata.csv`, and `link_audit.csv` before returning a
failure status. A pre-session discovery failure remains an exact initiator-only system event; it is
not dropped merely because no responder event exists.

Save symbol listings and analyze the exact final maps. A runtime watermark CSV is required for a
peak-RAM claim; do not add stack/heap figures to static RAM if those reservations already appear in
`.bss`/`.noinit`.

```powershell
arm-zephyr-eabi-nm.exe --print-size --size-sort --radix=x `
  build\pulse\release\zephyr\zephyr.elf |
  Set-Content -Encoding ascii capture\p01\pulse.nm
arm-zephyr-eabi-nm.exe --print-size --size-sort --radix=x `
  build\pulse\baseline\zephyr\zephyr.elf |
  Set-Content -Encoding ascii capture\p01\baseline.nm

$pooled = 'results\p01\pooled'
python tools\pulse\pulse_tools.py resources `
  --map pulse=build\pulse\release\zephyr\zephyr.map `
  --map baseline=build\pulse\baseline\zephyr\zephyr.map `
  --map correctness=build\pulse\correctness\zephyr\zephyr.map `
  --elf pulse=build\pulse\release\zephyr\zephyr.elf `
  --elf baseline=build\pulse\baseline\zephyr\zephyr.elf `
  --elf correctness=build\pulse\correctness\zephyr\zephyr.elf `
  --image-metadata "pulse=$raw\run_metadata.csv" `
  --image-metadata baseline=results\p01\baseline-pre\run_metadata.csv `
  --image-metadata correctness=results\p01\correctness-r01\run_metadata.csv `
  --correctness-input correctness=results\p01\correctness-r01\correctness.csv `
  --nm pulse=capture\p01\pulse.nm `
  --nm baseline=capture\p01\baseline.nm `
  --runtime-watermarks capture\p01\runtime_ram.csv `
  --output-dir $pooled
```

If no defensible runtime peak measurement exists, omit `--runtime-watermarks` and report static RAM
only. `resource_summary.csv` contains linker-authoritative totals and baseline deltas;
`resource_breakdown.csv` is an approximate object/section classification that must be reviewed.

Aggregate one or more complete blocks. Do not use `--allow-underfilled` for final results:

```powershell
python tools\pulse\pulse_tools.py summarize `
  --input-dir results\p01\release-r01 `
  --input-dir results\p01\baseline-pre `
  --input-dir results\p01\correctness-r01 `
  --output-dir $pooled
```

For additional complete blocks, repeat `--input-dir results\p01\release-r02`, and so on. The audit
always emits rows for `local/local/connected`, `no_contact/local/disconnected`,
`accepted_connected/initiator/connected`, and
`accepted_discovery/initiator/disconnected`, including an explicit zero-count failure when a
configuration is absent. The repetition minimum is checked independently for every
`run_id,pair_id,initiator-board` capture, so two 50-trial pairs cannot pass a 100-trial requirement.
It also audits exact accepted-session pairing. `--allow-underfilled` may
waive only a present pilot stratum below the repetition threshold; it cannot waive an absent stratum
or an integrity failure. Require every applicable row in `protocol_audit.csv` to pass. The summary
outputs include `event_summary.csv`,
`stage_summary.csv`, `correctness_summary.csv`, `power_summary.csv`,
`stage_power_summary.csv`, `baseline_power_summary.csv`, and `link_summary.csv`, with post-warm-up
mean, median, Q1, Q3, IQR, p95, counts, and failure rates. Corresponding
`*_summary_by_pair.csv` outputs retain run/pair/board sensitivity strata; all event, stage, power,
stage-power, and link summaries retain `remote_session_started`.

Firmware `duration_us`/`wall_duration_us` use Zephyr's 64-bit cycle-counter API backed by the nRF54
GRTC SYSCOUNTER and are the authoritative wall latency, including sleep. `cycles`/`cycle_duration_us` are elapsed DWT intervals,
not exclusive thread CPU time: higher-priority Bluetooth work, interrupts, and preemption remain
inside an interval, while sleep behavior differs from the wall clock. Radio and control stages are
deliberately wall-clock waits. Use the wall duration and GPIO/analyzer edges as the primary latency
cross-check, and do not describe DWT values as compute-only cycles. The configured timeout is at
most 20 seconds, leaving headroom below a 32-bit DWT wrap at 128 MHz.

Finally generate the three Section V panels and their traceable figure tables. Supply the pooled
directory for summaries/resources and each raw block for measured traces:

```powershell
python tools\pulse\pulse_tools.py figures `
  --data-dir $pooled `
  --data-dir results\p01\release-r01 `
  --data-dir results\p01\baseline-pre `
  --data-dir results\p01\correctness-r01 `
  --output-dir results\p01\figures `
  --formats pdf,png
```

This produces `section_v_panels.pdf`, `section_v_panels.png`, and the `figure_power_*`,
`figure_resources*`, `figure_runtime_ram.csv`, `figure_latency.csv`, and `figure_link.csv` source
tables. Final figure generation rejects missing required latency, power, baseline, link, resource,
validated runtime-peak RAM, passing same-artifact/fixture PyTorch correctness, trace, or
protocol-audit strata, and verifies that every event baseline exactly matches an included
baseline-only capture. Energy bars show median/IQR and p95; average/peak annotations are aggregate
pair statistics, and the representative trace is joined by full run/pair/exchange/event identity. Use
`--formats none` if Matplotlib is unavailable; the figure CSVs are still generated.

For the discussion's incremental daily-energy projection, prepare the strict per-device ledger
documented in `tools/pulse/README.md`. It must include scenario, seed, every node in the declared
network, observation duration, exact measured path/role/connection/outcome labels,
`remote_session_started`, counts before daily normalization, local and selection-opportunity totals,
and a readable hashed source manifest. The accepted physical tuples are
`local/local/connected`, `no_contact/local/disconnected`,
`accepted_connected/initiator/connected`, `accepted_connected/responder/connected`,
`accepted_discovery/initiator/disconnected`, and
`accepted_discovery/responder/connected`. Include explicit zero rows for local and no-contact
failures and, for both accepted paths, initiator pre-session and post-session failures plus
responder failures. Then run:

```powershell
python tools\pulse\pulse_tools.py project-energy `
  --counts capture\p01\per_node_event_counts.csv `
  --power-summary "$pooled\power_summary.csv" `
  --output-dir results\p01\projection
```

Require every row in `daily_energy_audit.csv` to pass. The current PULSE `contacts.csv` is not a
complete ledger: no-contact decisions have no row, transport failures are not simulated, and the
realized random node assignment is not emitted. Never fill those absences with assumed zeros. Use a
validated complete simulator decision export combined with measured firmware branching rates
(`count_basis=empirical_rate_model`), or a complete deployment observation
(`count_basis=deployment_observed`). The other outputs are `daily_energy_contributions.csv` and one
`daily_energy_projection.csv` row per scenario/seed/node. The expected value uses event rates times
sample mean event energy. The median/Q1/Q3/p95 columns are explicitly named plug-in sensitivity
calculations; they are neither daily-energy quantiles nor confidence bounds. They are incremental
PULSE energy only; do not multiply by aggregate network
contacts when the table requires per-node counts, and do not infer link-layer traffic from
application payload bytes.

## 9. Final acceptance checklist

- Both real-artifact manifests say `exported_pulse_npz`; the correctness reference says
  `pytorch_cpu_float32`.
- Both boards report the same pair, firmware, model, fixture, and artifact hashes, with complementary
  resolved roles and zero marker/sensing/link/protocol initialization errors.
- Every captured gate joins exactly one same-board serial event by monotonic `event_index`; every
  accepted initiator/responder record joins by exact `run_id,pair_id,exchange_id`; warm-ups remain
  labelled and are excluded only by the analysis filter.
- Every required path/role/connection-state stratum has at least 100 post-warm-up attempts, including
  recorded natural failures, and `protocol_audit.csv` has no failure.
- Supply voltage, baseline method, analyzer sample rate/filtering, clock synchronization, RF setup,
  board position, RSSI/link parameters, and physical-pair count are archived.
- Release/baseline maps and runtime measurements come from the flashed ELF hashes. Correctness image
  resources are not substituted for release resources.
- BLE link-layer claims have a synchronized sniffer/PCAP source, archived SHA-256, canonical export,
  and passing `link_audit.csv`. Otherwise report only the firmware's explicitly labelled
  application-transfer counters.
- Synthetic fixtures, test CSVs, parser replays, and the sanity totals in this document are never
  copied into Section V as hardware results.
