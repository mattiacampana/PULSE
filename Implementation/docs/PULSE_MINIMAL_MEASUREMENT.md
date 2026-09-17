# PULSE minimal two-device measurement plan

This is the shortest defensible route from two SensWear boards to the three
missing on-device panels in the PULSE manuscript. It deliberately narrows the
hardware claim to what can be established with two flashed boards, one Keysight
34465A used sequentially, and UART capture. It does not attempt a battery-life
study, a BLE link-layer study, or a multi-pair hardware study.

Use `docs/KEYSIGHT_34465A_FIG4A.md` as the detailed Figure 4a safety and command
reference. Use this file as the overall three-panel checklist.

## 1. Narrow the paper before measuring

Use the following three panels:

| Panel | Report | Evidence |
| --- | --- | --- |
| Fig. 4a | Sequential event-aligned current traces, idle power, local-update energy, measured initiator/responder energy, and an explicitly estimated pair role sum | Two Keysight 34465A role runs |
| Fig. 4b | Baseline versus PULSE flash and static RAM, with device capacities | Final baseline and release linker maps |
| Fig. 4c | Local-update and accepted-connected initiator/responder/end-to-end latency, with median, IQR, p95, and failure rate | One two-board UART campaign |

Delete or defer these claims unless a later experiment is explicitly added:

- incremental daily energy or battery lifetime;
- no-contact and discovery-path energy;
- whole-system peak RAM or free-RAM claims;
- BLE link-layer bytes, packets, or retransmissions;
- independent-board-pair replication beyond the one physical pair;
- GPIO-derived sub-stage boundaries in the power trace.

The current simulator logs do not contain a complete per-node ledger for a
daily-energy projection. The firmware's event-worker stack and heap diagnostics
are not a whole-system peak-RAM measurement. Omitting those two claims is more
accurate than deriving them from incomplete data.

Two results are constants of the implemented six-class FP32 head and do not need
a current experiment:

- head payload: `4550 * 4 = 18,200` bytes;
- gradient payload: `4550 * 4 = 18,200` bytes.

Read the peer-entry size from the final firmware `PULSE_META` output. Do not
infer it from source layout because compiler padding is part of the result.

## 2. Do not collect paper data yet if a preflight gate fails

All of the following must be true:

1. Regenerate the controlled simulation results after the upstream
   `freeze_encoder` correction. Remote-gradient normalization is pinned to the
   executable repository rule:
   `sqrt(local_norm_sq / max(remote_norm_sq, epsilon))`.
2. Commit the intended firmware and analysis changes. A final capture build is
   intentionally rejected when the worktree is dirty or contains untracked
   experiment code.
3. Run the host tests:

   ```powershell
   python -m unittest discover -s .\tools\pulse\tests -v
   ```

4. Produce the real deployment artifact described in
   `docs/PULSE_BENCHMARK.md`; both final manifests must say
   `artifact_kind=exported_pulse_npz`, never `deterministic_synthetic`.
5. Copy `config/pulse_lab.conf.example` outside the repository and replace its
   hardware-revision placeholder.
6. Record the one pair ID, the two immutable board IDs, both probe IDs, both COM
   ports, the approved supply voltage, maximum safe voltage, OVP, current limit,
   NGMO2 VISA resource, device separation, and exact toolchain versions.

With only two boards, write "one physical board pair with repeated trials" in
the paper. Do not call repeated trials independent board pairs.

## 3. Campaign A - latency and firmware metadata

This campaign uses UART; it is not a power measurement.

1. Build the normal final release image, without an NGMO2 autonomous overlay:

   ```powershell
   $artifact = (Resolve-Path '<real exported artifact.npz>').Path
   $labConfig = (Resolve-Path '<filled pulse_lab.conf>').Path

   cmake --preset senswear_pulse --fresh `
     "-DPULSE_FINAL_CAPTURE=ON" `
     "-DPULSE_ARTIFACT_NPZ=$artifact" `
     "-DEXTRA_CONF_FILE=$labConfig"
   cmake --build --preset senswear_pulse
   ```

2. Flash the exact same `build/pulse/release` image to both boards, selecting
   each probe explicitly.
3. Connect both debug boards and start one 921600-8-N-1 logger per COM port.
   Save the raw files as `latency-board-a.log` and `latency-board-b.log`.
4. Reset both boards close together. The firmware automatically runs five
   warm-ups and 100 measured repetitions of every path.
5. Stop only after the initiator prints `state=campaign_complete` and the final
   responder record has appeared.
6. Parse the two files:

   ```powershell
   python .\tools\pulse\pulse_tools.py serial `
     .\capture\p01\latency-board-a.log `
     .\capture\p01\latency-board-b.log `
     --set pair_id=p01 --set run_id=p01-latency-r01 `
     --output-dir .\results\p01\latency
   ```

Keep only the `local` and `accepted_connected` rows for the minimal Fig. 4c.
The discovery rows are useful as optional secondary results and cost no extra
lab work, but they are not required by the reduced figure.

## 4. Campaign B - current and energy

Use the complete step-by-step procedure in
`docs/KEYSIGHT_34465A_FIG4A.md`. It builds one reduced image, performs one
same-image UART preflight, measures the automatic initiator and responder in
two sequential Keysight 34465A runs, audits both fixed-size captures, and
generates `ondevice_power.pdf` plus its source CSVs.

The sequential setup cannot establish synchronized pair current or pair peak
power. Report the initiator and responder distributions as separately measured
and the matched-trial energy sum as an explicit non-simultaneous pair estimate.

## 5. Campaign C - resource and correctness results

No additional current measurement is needed.

1. Build the matched baseline and correctness images from the same commit,
   toolchain, lab configuration, and exported artifact used for the release:

   ```powershell
   cmake --preset senswear_pulse_baseline --fresh `
     "-DPULSE_FINAL_CAPTURE=ON" `
     "-DEXTRA_CONF_FILE=$labConfig"
   cmake --build --preset senswear_pulse_baseline

   $pythonWithTorch = (Get-Command python).Source
   cmake --preset senswear_pulse_correctness --fresh `
     "-DPULSE_FINAL_CAPTURE=ON" `
     "-DPULSE_ARTIFACT_NPZ=$artifact" `
     "-DPULSE_FIXTURE_PYTHON:FILEPATH=$pythonWithTorch" `
     "-DEXTRA_CONF_FILE=$labConfig"
   cmake --build --preset senswear_pulse_correctness
   ```

2. Generate Fig. 4b from the release and baseline linker-map totals. Report
   flash, static RAM, release-minus-baseline deltas, capacities, and remaining
   static space. Label the result "static RAM"; do not label it peak RAM.
3. Flash the correctness image to both boards, capture UART, and require a
   `PULSE_CORRECTNESS,pass=1,reference=pytorch_cpu_float32` result with the same
   artifact and fixture hashes as the release.
4. Take gradient cosine similarity and maximum mixed-head parameter error from
   the parsed `correctness.csv`.

## 6. Files to preserve and return for figure generation

Do not rename or edit raw captures. Collect these under one experiment folder:

```text
p01/
  artifact/
    pulse_*.npz
    pulse_*.provenance.json
  latency/
    latency-board-a.log
    latency-board-b.log
  power-local/
    ngmo2_capture_manifest.csv
    raw capture CSVs
    board-a-export.log
    board-b-export.log
  power-accepted-connected/
    ngmo2_capture_manifest.csv
    raw capture CSVs
    board-a-export.log
    board-b-export.log
  images/
    release/      # ELF, HEX, map, .config, manifest, hashes
    baseline/     # ELF, HEX, map, .config, hashes
    correctness/  # ELF, HEX, map, .config, manifest, hashes
  notes.txt       # voltage, limits, analyzer settings, board IDs, distance, anomalies
```

Once these files exist, the remaining work is offline: parse, audit, produce the
three panels, and replace only the retained manuscript placeholders. Never fill
a missing result with the development-build sanity values in
`docs/PULSE_BENCHMARK.md`.

## 7. Minimal manuscript substitutions

After successful measurements, the retained TBD values are:

- one physical board pair and the number of repeated trials;
- analyzer output voltage;
- idle power;
- local-update energy;
- accepted-connected initiator, responder, and pair energy;
- PULSE-minus-baseline flash and static-RAM deltas;
- remaining static flash and RAM;
- 18,200 bytes per serialized head and gradient;
- compiler-reported peer-entry size;
- accepted-connected median/p95 latency and failure rate;
- embedded/reference gradient cosine similarity and maximum parameter error.

Remove the daily-energy TBD, the discovery-latency TBD, and all peak-RAM TBDs
unless their additional measurements are actually performed.
