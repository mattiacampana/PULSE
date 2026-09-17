# Figure 4a partial capture audit (2026-09-17)

This records the earlier read-only audit and the later, explicitly requested
exploratory interpolation analysis. The raw capture files were not changed.

## Exploratory interpolation analysis (supersedes the plotting decision below)

At the user's request, `tools/pulse/plot_fig4a_interpolated.py` replaces only
the 85 invalid responder readings in an in-memory copy by linear interpolation
between neighboring valid readings on the existing 20-ms grid. It leaves every
valid raw value, sample index, and raw file unchanged. The repaired stream has
105 candidate onsets; after excluding five warm-ups, 100 candidate windows
are plotted. Fifty of those windows include 80 interpolated readings. The
resulting responder median is 6.87 mJ (IQR 2.15--20.65 mJ) **if** the
responder was supplied at 3.7 V, which its absent manifest cannot verify.
The previous- and next-neighbor fill alternatives give medians of 6.87 and
7.42 mJ. Full counts, per-event values, input hashes, and assumptions are in
`fig4a_interpolated_audit.json`.

This is a labeled exploratory responder distribution, not recovered DMM
readings or a validated exchange result. The invalid readings might conceal
peaks that interpolation cannot reconstruct. Missing responder provenance and
UART outcome joins, and nonsimultaneous role captures, still preclude a
verified responder result or pair-energy sum. The older decisions below
describe the stricter no-imputation analysis and are retained as history.

## Latest responder stream (2026-09-17 15:20 local)

The file at `capture/p01/keysight-responder-r01/keysight_34465a_responder_stream_raw.csv`
was replaced after the earlier audits below. Its current SHA-256 is
`1f1f27127677ba555888164ba639e812a438275612c5e18f5adad64121edbe3e`.
It contains 58,264 indexed readings at 50 readings/s (1,165.28 s), including
85 `9.9e37` invalid DMM readings. The reported failure at raw sample 29,956
occurs at 599.12 s inside detected event 1's 225-reading window; it cannot be
used as measured current or silently filled.

With the current 0.0035-A software threshold, the saved stream yields 104
candidate onsets, not the required 105. One interval between detected onsets
is 11.0 s, versus a median 5.4 s, consistent with a missed detection. Fifty-one
of the 104 detected event windows include at least one invalid reading; after
excluding the first five detected windows as warm-ups, only 50 windows are
clean. Because onset numbering may shift after a missed detection, this is not
a verified 50-trial subset. No completed responder sample CSV or capture
manifest exists. Valid readings peak at 4.69 mA, so the sentinel alone does not
establish that steady current exceeded the configured 10-mA range. This new
file also cannot support the paper's responder or pair energy result.

The earlier hashes and counts below are retained only as history of the files
that previously occupied the same path. Do not use them for the current stream.

## Update after the responder CSV changed

The responder CSV was modified in place after the initial audit. At
2026-09-17 14:19:54 local time it still contained exactly 65,034 time slots,
but its SHA-256 changed from the original value below to
`a4f7ac34dc9a06f49ae452a0a5665f9b2c5ff7095fb1a560183ac82e5abc82cc`.
The 79 original `9.9e37` values are no longer present. At least 74 revised
samples are exact copies of the immediately preceding sample, including the
previously observed invalid positions 7,657, 27,615, 29,696, 29,741, 30,005,
30,150, 30,287, and 31,128. This is a fill operation, not recovered DMM
measurements. The other five original invalid positions cannot be fully
reconstructed from the overwritten file.

The detector now reports 105 candidate onsets at a 0.0035 A threshold. Those
extra detections depend on the substituted values and do not establish 105
fully observed events. The original 103-onset and invalid-window results below
describe the CSV *before* this in-place modification. Neither version supports
an unqualified 103- or 105-event responder energy distribution for the paper.

## Available files

- Initiator: `capture/p01/keysight-initiator-r03/` contains a raw stream, an
  extracted 210-event sample CSV, and a completed capture manifest.
- Responder: `capture/p01/keysight-responder-r01/` contains only
  `keysight_34465a_responder_stream_raw.csv`. There is no extracted sample CSV
  or capture manifest for this run.
- No same-image, two-board UART preflight CSVs were found under `capture/p01/`
  or `results/p01/`.

## Responder raw-stream audit

The raw file has 65,034 current readings at 50 readings/s (1,300.68 s). Its
SHA-256 is `c27aa51bfa5f960725beb1adc5f7ec76500dd8a0e560e6cd4c63fc052c0f82c0`.
There are 79 isolated `9.9e37` readings. The capture code classifies these as
invalid/overload values. They cannot be used as measured current or silently
interpolated for paper energy or peak-power estimates.

Using the repository's `_stream_event_onsets` detector with its 20 ms sample
interval and the stated 225-reading window (20 pre-event, 205 post-event):

| Offline detection level | Candidate onsets | Windows containing invalid readings | Clean windows |
| --- | ---: | ---: | ---: |
| 0.0035 A | 103 | 50 | 53 |
| 0.0020 A | 105 | 43 | 62 |

The 0.0020 A level was examined offline; this does **not** establish that it was
pilot-tested or valid for the paper run. If the first five detected windows are
treated as warm-ups, at most 60 of its remaining 100 windows are free of invalid
readings. Missed or extra onsets can also shift inferred trial IDs, so clean
window counts are not a verified matched-trial set.

## Provenance check

The completed initiator manifest records firmware revision
`1164e3356e3d6c0e54f8ac46522ba65a1cf8fb15` and HEX SHA-256
`68df553482b6ae0cb09d00968c05493a634bc19596cc096ffe95dbabf73f937d`.
Those values differ from the archived FIFO-guard release revision
`6bd1144c107318e896efb1e39a6c833ebcf558f3` and HEX SHA-256
`582e93a57462e4daf52d0b04588a82809a6c021d07d7fa43db61f7310191f391`.
The responder raw CSV contains no flashed-image identity. Therefore the
available files do not establish that both measured roles used the same image.
The initiator manifest's calibration date is `2020-05-29`; verify the current
calibration's validity against the instrument record.

## Decision

Do not run `keysight-fig4a` on these files or insert a numerical Figure 4a in
the manuscript. The missing responder values, incomplete capture audit,
unverified role pairing, and absent same-image UART preflight prevent the
required 100-trial, per-role and complete-pair energy claims.

Repeat a responder pilot to establish a threshold and determine whether the
10 mA range actually overloads. If so, validate the 100 mA range through the
3 A terminal and repeat **both** role captures at that range. Save complete
manifests and a same-image two-board UART preflight before generating the paper
figure. Use new output directories; preserve these raw files unchanged.
