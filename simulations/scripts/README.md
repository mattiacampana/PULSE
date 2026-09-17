# Dataset preparation and encoder pretraining

This directory contains the reproducible data pipeline used by the first
Opportunistic Learning simulator experiments. The source domain is UCI HAR;
the target domain is HHAR. A simulated node is always an **HHAR user**. Device
identifiers are used only while forming windows, so that a window never mixes
recordings from different devices.

Run the commands below from the repository root (`opportunistic_simulator/`).
The `PYTHONPATH=..` prefix lets the scripts import the project model definitions
from its parent directory.

## 0. Generate a synthetic mobility trace

`generate_mobility.py` produces the simulator input format directly:
`time,node1,node2`. It intentionally generates contacts rather than physical
coordinates, because the simulator models *when* an exchange is possible—not
radio range, BLE duration, or energy cost.

```bash
python scripts/generate_mobility.py \
  --model edge_markov \
  --num-nodes 12 \
  --duration 86400 --time-step 20 \
  --contact-probability 0.10 --persistence 0.80 \
  --seed 7 \
  --output data/mobility/edge_markov_seed7.csv
```

Two models are currently available:

| Model | Main parameters | Use |
|---|---|---|
| `edge_markov` | `--contact-probability`, `--persistence` | Controlled baseline: sets the average density of pairwise contacts and their temporal persistence. |
| `community` | `--num-communities`, `--intra-contact-probability`, `--inter-contact-probability` | Social/routine-like scenario with repeated contacts inside communities and rarer cross-community encounters. |

For example, the community scenario can be generated with:

```bash
python scripts/generate_mobility.py \
  --model community --num-nodes 12 --num-communities 3 \
  --duration 86400 --time-step 20 \
  --intra-contact-probability 0.35 --inter-contact-probability 0.02 \
  --seed 7 --output data/mobility/community_seed7.csv
```

The script also writes a neighbouring `.summary.json` with its parameters,
number of contact events, number of distinct contacting pairs and, for the
community model, the generated community label of each trace identity.

Trace IDs have the form `trace_0`, `trace_1`, … and are deliberately separate
from HHAR user IDs. Configure `mobility.node_assignment: random` in the
simulation YAML to assign these identities to the selected data nodes with the
experiment seed. The resulting mapping is saved in each run as
`node_assignment.csv`.

## 1. Download and preprocess the datasets

```bash
PYTHONPATH=.. python scripts/prepare_har_datasets.py \
  --dataset all \
  --raw-dir data/raw \
  --output-dir data/processed \
  --window-size 128 --stride 128
```

`prepare_har_datasets.py` downloads the official UCI HAR and HHAR archives when
they are not already present, extracts them under `data/raw/`, and creates:

| File | Purpose |
|---|---|
| `uci_har.npz` | Source windows for encoder pretraining. It contains `x`, `y`, `subject_ids`, `split`, `class_names`, `mean`, and `std`. |
| `hhar_by_user.npz` | Target windows for the simulator. It contains `x`, `y`, `node_ids`, and `class_names`. |

Both `x` arrays have shape `[num_windows, 3, 128]`. UCI HAR supplies its
predefined windows. HHAR is segmented separately by user, device, and activity,
using 128 samples per window. The statistics `mean` and `std` are estimated
from the UCI HAR **training split only** and are applied to both datasets. HHAR
labels are therefore never used to fit preprocessing parameters.

Useful options:

- `--dataset uci_har` prepares just UCI HAR. `--dataset hhar` also ensures UCI
  HAR is prepared first, because it needs its normalization statistics.
- `--stride 64` creates 50% overlapping HHAR windows; `128` creates
  non-overlapping windows.
- `--max-gap-ms VALUE` discards an HHAR window if it contains a timestamp gap
  larger than `VALUE` milliseconds.
- `--stats-dir PATH` saves summaries separately from the `.npz` archives.
- `--force-preprocess` rebuilds the `.npz` archives and all summaries.
- `--force-download` downloads and extracts the official archives again.

### Generated statistics

The script writes the following files to `--stats-dir` (by default,
`data/processed/`):

| File | Contents |
|---|---|
| `uci_har_dataset_summary.json` | Dataset-level counts, classes, normalization, and predefined split size. |
| `uci_har_class_summary.csv` | Number of windows per activity, including UCI HAR train/test counts. |
| `uci_har_node_summary.csv` | Per-subject distribution. UCI subjects are included for source-data diagnostics only, not as simulator nodes. |
| `hhar_dataset_summary.json` | Target dataset/windowing configuration and number of simulated user nodes. |
| `hhar_class_summary.csv` | Number of HHAR windows per activity. |
| `hhar_node_summary.csv` | One row per simulated user: windows, number of original devices, per-class counts, missing classes, class entropy, and imbalance ratio. |

`class_entropy` is the Shannon entropy of the user’s observed class
distribution (larger means more balanced). `imbalance_ratio` is the largest
class count divided by the smallest non-zero class count. These summaries are
pre-split properties of the generated data and should be kept alongside the
simulation logs to interpret which users benefit most from collaboration.

## 2. Pretrain the shared encoder

After preparation, train the CNN-1D encoder on UCI HAR:

```bash
PYTHONPATH=.. python scripts/pretrain_encoder.py \
  --data data/processed/uci_har.npz \
  --output data/checkpoints/uci_encoder.pt \
  --num-classes 6 \
  --embedding-dim 64 \
  --input-channels 3 \
  --epochs 20 \
  --batch-size 64
```

The script trains a `CNN1DEncoder` plus a temporary linear classifier using the
source arrays `x` and `y`. It saves only the encoder weights to the `.pt` file:
the simulator creates a separate personalized classification head for every
HHAR user. Configure this checkpoint under `pretraining.checkpoint` and leave
`pretraining.freeze_encoder: true` for the intended initial protocol—no HHAR
label is used to adapt the encoder.

The checkpoint must match `model.input_channels` and `model.embedding_dim` in
the simulator configuration.

## Manual HHAR-only preprocessing

`preprocess_hhar.py` is retained for a manually downloaded
`Phones_accelerometer.csv`. For the standard pipeline, prefer
`prepare_har_datasets.py`, which guarantees source-consistent normalization.
