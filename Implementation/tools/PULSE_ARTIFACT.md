# PULSE firmware artifact

`export_pulse_artifact.py` converts a supplied PULSE encoder, an exact head, and
real HAR windows into a versioned, self-describing NPZ. It never substitutes
synthetic data. The upstream pretraining script saves only the encoder; the head
must therefore be either a separately captured state dictionary or the canonical
deterministic `init_id="common"` head reproduced from an explicitly identified
PULSE source tree, configuration, and simulation seed.

## Export

```console
python tools/export_pulse_artifact.py \
  --checkpoint path/to/combined_checkpoint.pt \
  --data path/to/processed_hhar.npz \
  --output build/pulse_artifact.npz
```

For an encoder-only checkpoint, either add
`--head-checkpoint path/to/exact_head.pt`, or bind the canonical initial head to
the controlled evaluation source:

```console
python tools/export_pulse_artifact.py --checkpoint uci_encoder.pt \
  --pulse-source-root path/to/PULSE \
  --pulse-config path/to/PULSE/configs/community_hhar_pulse.yaml \
  --simulation-seed 7 --data processed_hhar.npz \
  --output build/pulse_artifact.npz
```

A dry schema and input check performs
all loading, shape, dtype, label, finiteness, normalization, and flatten-order
validation without writing files:

```console
python tools/export_pulse_artifact.py --checkpoint encoder.pt \
  --head-checkpoint common_head.pt --data processed_hhar.npz --validate-only
```

Run `python tools/export_pulse_artifact.py --help` for state-dict prefix and replay
selection options. By default the first 64 input samples become four consecutive
batches. `--indices-key replay_indices` instead accepts an integer `[4,16]` (or
64-element) index array from the input archive. The data archive must contain
`x:[N,3,128]`, integer `y:[N]`, and `node_ids:[N]`, with every label in `[0,5]`.
The first three replay batches must belong to one initiator owner and the fourth
to one distinct responder owner. `--allow-owner-role-mismatch` exists only for
diagnostic artifacts; such an artifact is rejected by the fixture generator.
Optional `--replay-split-id` and `--replay-event-id` values are recorded as
operator-supplied provenance.

Processed PULSE archives are already normalized, so the default is
`--normalization verify`. It checks the PULSE preprocessing operation applied
separately to every channel of every window:

```text
(x - mean(x, time)) / max(std(x, time), 1e-6)
```

Use explicit `--normalization apply` only when exporting raw windows.

## NPZ schema `senswear.pulse.artifact.v1`

All weights and samples are detached, contiguous CPU `float32` arrays.

| Key | Shape | Dtype |
| --- | ---: | --- |
| `encoder.features.0.weight` | `[32,3,5]` | float32 |
| `encoder.features.0.bias` | `[32]` | float32 |
| `encoder.features.3.weight` | `[64,32,5]` | float32 |
| `encoder.features.3.bias` | `[64]` | float32 |
| `encoder.projection.weight` | `[64,64]` | float32 |
| `encoder.projection.bias` | `[64]` | float32 |
| `head.feature_layers.0.weight` | `[64,64]` | float32 |
| `head.feature_layers.0.bias` | `[64]` | float32 |
| `head.output_layer.weight` | `[6,64]` | float32 |
| `head.output_layer.bias` | `[6]` | float32 |
| `head.flat` | `[4550]` | float32 |
| `replay.x` | `[4,16,3,128]` | float32 |
| `replay.y` | `[4,16]` | uint8 |
| `replay.source_indices` | `[4,16]` | int64 |
| `replay.owner_ids` | `[4,16]` | Unicode string |

`head.flat` is bit-for-bit concatenated in canonical `W1,b1,W2,b2` order. The
archive also includes scalar schema and payload hashes plus the four replay-role
names. A sibling `.provenance.json` records source-file SHA-256 values, exact
state-dict name resolution, source dtypes, replay indices and owners, owner-role
validation, operator-supplied split/event identifiers, normalization, tool
versions, the canonical payload SHA-256, and the final NPZ file SHA-256.
