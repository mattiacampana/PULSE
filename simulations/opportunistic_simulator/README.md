# PULSE: Personalized Opportunistic Learning for Wearable Devices

## Model-ready HDF5 simulation data

The simulator reads a directory produced by
`scripts/datasets_processing/prepare_datasets.py`. It must contain `data.h5`,
`manifest.csv`, and `dataset.json`. Set `data.source_split: simulation` so only
subjects reserved for simulation become nodes. Each subject is then divided
locally into train, validation, and test windows.

`data.allow_full_fallback` defaults to `false`. Set it to `true` only for an
ex-novo simulation corpus intentionally prepared without subject roles. The
fallback selects `full`; it never mixes pretraining or validation subjects.

The simulation corpus and checkpoint are independent:

```yaml
data:
  loading_mode: cache_embeddings
  cache_batch_size: 1024
  cache_num_workers: 4
  cache_pin_memory: true
  cache_prefetch_factor: 2
  path: ../data/processed/hhar
  source_split: simulation
  allow_full_fallback: false

model:
  encoder: auto
  embedding_dim: auto
  num_classes: auto

pretraining:
  checkpoint: ../data/checkpoints/uci_har_encoder.pt
  freeze_encoder: true
```

Cross-dataset use is accepted when the encoder architecture and input contract
match, as for UCI-HAR and HHAR. The classifier uses the class count of the
simulation dataset. Incompatible architectures or tensor shapes fail at setup.

With `data.loading_mode: cache_embeddings`, the frozen encoder processes every
selected window once during setup. Compact embeddings are retained in CPU RAM,
and the HDF5 file and encoder are no longer used during the simulation timeline.
This mode requires `pretraining.freeze_encoder: true`. Use `lazy_hdf5` for an
encoder that must remain trainable. `data.cache_batch_size` controls only the
one-time precomputation batch and can be reduced if setup exhausts GPU memory.
Caching performs one globally ordered scan of the selected HDF5 rows instead of
one scan per node and split. `cache_num_workers` controls parallel HDF5 readers,
`cache_prefetch_factor` queues batches per worker, and `cache_pin_memory` enables
asynchronous transfers to CUDA. Start with 4 workers; more workers can be slower
on a single mechanical disk or network filesystem. The resulting node datasets
are zero-copy views over one shared embedding matrix in CPU RAM.

`training.evaluation_batch_size` is independent from the SGD `batch_size`.
Using a large value such as `1024` accelerates validation and test passes without
changing local optimization or metrics; reduce it only if evaluation exhausts
GPU memory.

This repository contains an event-driven PyTorch simulator for studying
**personalized opportunistic collaborative learning** on resource-constrained
wearable devices. PULSE is designed for settings in which users carry private,
heterogeneous data and can collaborate only during intermittent encounters.

One simulated node represents one **user**. Measurements from that user's
multiple devices remain in the same local data partition. The simulator models
the availability of peer encounters and the learning and communication actions
triggered by them; it does not aim to emulate BLE duration, battery consumption,
RAM use, or wall-clock execution on a physical device.

## Overview

All methods use the same deployment model: a CNN encoder pretrained on a source
dataset and frozen during deployment, followed by a small personalized MLP
classifier. The hidden layers of the classifier can be configured through
`model.classifier_hidden_dims` (for example, `[64]` gives a `64 -> 64 -> 6`
classifier for six activities).

The simulator replays a mobility trace with rows of the form
`time,node1,node2`. At each encounter, it snapshots both outgoing messages
before applying either endpoint's update, making the exchange bidirectional and
independent of call order. Periodic local training gives nodes a common local
adaptation mechanism, while each algorithm determines how an encounter is used.

## Implemented methods

- **Local** (`local_only`): personalized local training with no peer
  communication.
- **OppAVG** (`oppavg`): an opportunistic counterpart of FedAvg. At an
  encounter, a node merges its prediction head with the heads of all currently
  reachable peers.
- **OpportunisticFL / OppFL** (`opportunistic_fl`): implementation of Lee et
  al.'s Opportunistic Federated Learning (PerCom 2021). A learner sends its
  current classifier to a peer and receives a gradient computed on the peer's
  private minibatch; the received signal is combined with the local gradient
  following the method's aggregation rule.
- **OPTIMIST** (`optimist`): implementation of Romero et al.'s OPTIMIST
  (PerCom Workshops 2025). Compatible nodes exchange active classifier
  subnetworks and train only the received subnetwork, following its original
  sequential-training protocol.
- **PULSE** (`pulse`): the proposed method. At each discovery event, it uses a
  utility-guided UCB policy to decide whether a reachable peer is worth
  contacting. A contacted peer evaluates a temporary copy of the local head on
  a private batch and returns only the resulting gradient. PULSE mixes that
  signal with the local gradient only when they agree; when the *no-contact*
  action is chosen, it performs the corresponding purely local update.

PULSE keeps, for each peer, a signed utility estimate
\(\mathcal{U}_i[j]\), a selection count \(n_{ij}\), and the total number of
selection opportunities \(s_i\). These quantities correspond directly to the
peer-selection policy described in the paper.

## Fair compute budgets

The default fixed-budget mode performs the same scheduled local SGD steps for
all nodes. For a compute-matched comparison, set
`training.budget_mode: contact_matched`. The simulator first applies
OPTIMIST's one-contact-per-node matching to the trace, derives a per-node SGD
cap from the resulting matched contacts, and enforces the same cap for every
method. The realised caps and steps are included in the per-seed summary.

OPTIMIST is deliberately different in how it consumes this budget: it trains
only after a matched contact, as specified by its original protocol. PULSE and
the other methods use their own encounter logic while respecting the same
per-node cap.

## Data and mobility inputs

The target dataset file (`hhar_by_user.npz`) must contain:

- `x`: floating-point windows with shape `[num_windows, 3, 128]`;
- `y`: integer activity labels;
- `node_ids`: the user identifier associated with every window.

The source pretraining dataset (`uci_har.npz`) contains `x` and `y` with the
same input shape. A mobility CSV must contain `time`, `node1`, and `node2`.

With `mobility.node_assignment: identity`, trace identifiers must match the
selected data-node identifiers. With `random`, a distinct trace identity is
assigned to each selected user using the experiment seed; the realised mapping
is saved in `node_assignment.csv`.

### Prepare the datasets

The data preparation scripts download and preprocess UCI HAR and HHAR. See
[scripts/README.md](scripts/README.md) for all options.

```bash
PYTHONPATH=.. python scripts/prepare_har_datasets.py \
  --dataset all \
  --raw-dir data/raw \
  --output-dir data/processed \
  --window-size 128 --stride 128
```

HHAR windows are created separately for each user, device, and activity so
that recording boundaries are never crossed; the simulation node remains the
user. Both datasets use per-window, per-axis z-score normalization, which can
be calculated independently at inference time.

## Installation and execution

```bash
git clone <repository-url>
cd opportunistic_simulator
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pretrain the encoder once:

```bash
PYTHONPATH=.. python scripts/pretrain_encoder.py \
  --data data/processed/uci_har.npz \
  --output configs/checkpoints/uci_encoder.pt \
  --num-classes 6 --embedding-dim 64 --input-channels 3
```

Then run an experiment by selecting a YAML configuration:

```bash
PYTHONPATH=.. python run.py --config configs/pulse.yaml
```

Set the configuration's algorithm field to one of `local_only`, `oppavg`,
`opportunistic_fl`, `optimist`, or `pulse`. Paths in a YAML file are resolved
relative to that file. For OPTIMIST, use exactly one classifier hidden layer
and choose `num_subnetworks` no greater than the width of that layer.

## Outputs

Each run writes the following files to the configured output directory:

- `metrics.csv`: per-user test accuracy and macro-F1 over simulated time;
- `training.csv`: local-training losses and SGD-step information;
- `contacts.csv`: directed contact outcomes and algorithm-specific metadata;
- `node_assignment.csv`: the seeded mapping between trace identities and data
  nodes when random assignment is used;
- `summary_seed_<seed>.json`: the configuration, budget information, and final
  mean metrics for the seed.

All CSV logs include a `seed` column. Existing logs are preserved and new runs
are appended, allowing results from multiple seeds to be aggregated directly.
During execution, the simulator prints setup milestones and periodic updates;
adjust their cadence with `logging.progress_interval`.

## Extending the simulator

Create a module in `nodes/`, subclass `Node`, and implement the protected
message hooks `_build_contact_message()` and `_apply_contact_message()`. The
shared simulator continues to provide trace replay, local training, evaluation,
budget accounting, and logging; only the algorithm-specific encounter logic
needs to be added. Register the new class in `run.py` and add a dedicated YAML
configuration.

## Reproducibility

Set the experiment seed in the YAML configuration. The seed controls model
initialization, data-related sampling, and random node assignment. Report
aggregated accuracy and macro-F1 across multiple seeds, and retain the emitted
per-seed summaries together with the configuration files used to produce them.
