# PULSE wearable dataset preparation

This package downloads or imports, preprocesses, validates, and stores wearable
datasets in the exact tensor representation expected by lightweight PULSE
encoders. The simulator only has to load an input, run the encoder, and attach
its small predictive MLP. Filtering, resampling, synchronization, windowing,
padding, deterministic feature extraction, and label encoding happen offline.

The pipeline supports:

| Dataset | Model-ready encoder input | Users | Default model |
|---|---|---|---|
| UCI HAR | `imu [3, 128]` | 30 | tiny 1-D DS-CNN |
| HHAR | `imu [3, 128]` | 9 | tiny 1-D DS-CNN |
| WESAD | ACC, BVP, EDA, and temperature branches | 15 | multibranch tiny CNN |
| SisFall | six-channel waist IMU windows | 38 | tiny causal TCN / DS-CNN |
| RingAuth | padded six-channel IMU gesture and validity mask | 21 | tiny DS-CNN |
| PADS | synchronized left/right six-channel IMU windows | 469 | shared dual-wrist DS-CNN |
| Audio-IMU Cough | log spectrum and aligned accelerometer | 13 | audio DS-CNN + IMU CNN |
| Pulse Transit Time PPG | eleven physiological features | 22 | tiny feature MLP |

DREAMT and My Seizure Gauge are intentionally out of scope because their access
flows require user registration or a data-use agreement.

## Installation

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

The installed command is `prepare-wearable-datasets`. The repository-level
`prepare_datasets.py` entry point is equivalent.

## Quick start

Prepare the complete UCI HAR and HHAR datasets without assigning roles:

```bash
prepare-wearable-datasets --dataset uci_har
prepare-wearable-datasets --dataset hhar
```

This is the intended cross-dataset DeepRAP setup: the experiment can use the
entire UCI HAR output for encoder pretraining and the entire HHAR output for
opportunistic nodes. Empty split parameters never silently invent roles.

Prepare a dataset with disjoint subject groups:

```bash
prepare-wearable-datasets \
  --dataset sisfall \
  --num-pretrain-users 20 \
  --num-validation-users 8 \
  --num-simulation-users 10 \
  --split-seed 42
```

For multiple datasets, copy and edit
[`configs/datasets.example.yaml`](configs/datasets.example.yaml), then run:

```bash
prepare-wearable-datasets --config configs/datasets.example.yaml
```

List stable dataset identifiers with:

```bash
prepare-wearable-datasets --list-datasets
```

## Split semantics

There are exactly two modes.

### Full-dataset mode

If `pretrain_users`, `validation_users`, and `simulation_users` are all absent,
the output contains every valid subject and `splits.json` reports
`"split_mode": "full"`. The preprocessing pipeline does not decide whether the
dataset is a pretraining or simulation dataset.

### Subject-split mode

Specifying one or more counts enables deterministic subject allocation before
output materialization. The groups are disjoint. Omitted counts mean zero;
`simulation_users: remaining` assigns all subjects left after pretraining and
validation. A request larger than the number of real subjects fails instead of
duplicating users.

One real subject remains one simulated user. Multiple devices belonging to one
person (two PADS watches, or multiple WESAD sensors) never become independent
users.

## Output contract

Each processed dataset directory contains:

```text
processed/<dataset>/
├── data.h5
├── manifest.csv
├── dataset.json
├── splits.json
└── preprocessing_report.json
```

`data.h5` contains fixed-shape arrays:

```text
/inputs/<branch>   [samples, channels-or-features, time]
/labels            [samples]
```

`manifest.csv` maps every HDF5 row to `subject_id`, `session_id`, `device_id`,
timestamp, encoded label, assigned split, and JSON metadata. `dataset.json`
documents the task, branches, channels, rates, normalization, input shapes,
and expected 64-dimensional encoder embedding. The simulator-facing contract
is therefore:

```python
with h5py.File("data/processed/uci_har/data.h5") as dataset:
    inputs = dataset["inputs/imu"][:]       # [N, 3, 128]
    labels = dataset["labels"][:]

embedding = encoder(inputs)                 # [N, 64]
prediction = simulator_mlp(embedding)
```

No model head is stored by this package. The MLP belongs to the simulator.

## Normalization and leakage

Raw time-series branches use deterministic per-window, per-channel z-scores.
This matches the legacy UCI HAR/HHAR behavior and does not consume another
subject's statistics. Physiological PTT features remain in documented physical
or dimensionless units. Audio spectra are log-compressed and normalized.

Subject selection always occurs over stable `subject_id` values. Windows from
one subject cannot cross pretraining, validation, and simulation groups. Data
augmentation is deliberately excluded because it is stochastic and belongs in
the training loop.

## Dataset acquisition

The pipeline automatically downloads UCI HAR, HHAR, and WESAD from UCI.
The remaining adapters consume official files placed under their raw directory;
this avoids unstable links, hidden consent steps, and accidental credential
handling.

### SisFall

Extract the official SisFall subject folders into `data/raw/sisfall`. Files must
retain names such as `D01_SA01_R01.txt` or `F01_SA01_R01.txt`. The adapter
converts the first accelerometer to g, the gyroscope to rad/s, and creates binary
ADL/fall windows. The original project host has historically been unreliable,
so absence produces a precise local-source error.

### RingAuth

Download `data.7z.001`, `data.7z.002`, and `data.7z.003` from the Oxford
University Research Archive into `data/raw/ringauth`. If CSV files are not
already extracted, the adapter invokes `7z`/`7zz` on part 001. Preserve original
paths because they encode user, device, and gesture context. CSV column aliases
for accelerometer and gyroscope axes are resolved case-insensitively.

### PADS

Mirror the public PhysioNet release without credentials:

```bash
wget -r -N -c -np \
  https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/movement/ \
  -P data/raw/pads
```

The adapter pairs left/right wrist text records, discards the time column,
keeps accelerometer and gyroscope channels, aligns both wrists to their shortest
common interval, and emits paired windows. Keep the official path and side
naming intact.

### Audio-IMU Cough

Download the public Dryad package and extract it under
`data/raw/audio_imu_cough`. A WAV recording must have a same-stem CSV or
`<stem>_imu.csv` companion. The code resamples audio to 16 kHz, computes a
64-bin log spectrum using a 25 ms window and 10 ms hop, and aligns a 100 Hz
three-axis IMU tensor. Filenames containing `cough` are positive unless they
contain `noncough`; adjust the adapter if the release's annotation table uses a
different convention.

### Pulse Transit Time PPG

Mirror the public PhysioNet dataset into `data/raw/pulse_transit_time`, retaining
CSV records with ECG, PPG/plethysmogram, and optional accelerometer columns.
If the downloaded release uses another container, convert records to CSV while
preserving sampling rate and channel names. The adapter detects cardiac and
pulse peaks and stores heart rate, RR variability, pulse arrival statistics,
signal statistics, motion, and a peak-match quality score. `source_rate_hz`
must match the converted files.

## Adapter configuration

All adapters accept `window_seconds`, `stride_seconds`, and `target_rate_hz`
when relevant. Dataset-specific keys remain in `options` and are documented in
the example YAML. Important defaults are:

| Dataset | Window | Stride | Rate |
|---|---:|---:|---:|
| UCI HAR / HHAR | 2.56 s | 2.56 s | 50 Hz |
| WESAD | 60 s | 60 s | native per branch |
| SisFall | 2.56 s | 1.28 s | 200 Hz |
| RingAuth | 3 s padded event | event-based | 100 Hz |
| PADS | 2.56 s | 2.56 s | 100 Hz |
| Audio-IMU Cough | 2 s | event-based | 16 kHz / 100 Hz |
| Pulse Transit Time | 10 s | 10 s | configured source rate |

Changing an output-affecting option requires `--force-preprocess`. Existing
completed outputs are otherwise reused. `--force-download` refreshes sources
only for adapters with automatic acquisition.

## Adding an adapter

Create a module in `wearable_datasets/adapters`, subclass `DatasetAdapter`, and
decorate the class with `@register`. Implement:

```python
class NewAdapter(DatasetAdapter):
    name = "new_dataset"
    descriptor = {"task_type": "multiclass", "embedding_dim": 64}

    def acquire(self, raw_dir, force):
        ...

    def preprocess(self, source, config):
        return [Sample(inputs={"imu": tensor}, label=0, subject_id="S01")]
```

Import the module from `wearable_datasets/adapters/__init__.py`. The shared
writer verifies fixed branch names and shapes, rejects NaN/infinity, allocates
subjects, and writes the full contract.

## Validation and tests

Run:

```bash
pytest -q
python prepare_datasets.py --list-datasets
```

Tests cover full mode, reproducible and disjoint splits, impossible requests,
per-channel normalization, HDF5/manifest/schema materialization, non-finite
rejection, and adapter registration. They use synthetic data and never download
large datasets.

For a prepared real dataset, inspect `preprocessing_report.json`, confirm the
subject and label counts, and load one batch from every HDF5 input branch before
starting a long training or simulation run.
