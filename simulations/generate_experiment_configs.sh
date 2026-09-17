#!/usr/bin/env bash
set -euo pipefail

# Generate the full DeepRAP experiment matrix:
#   5 algorithms x 5 simulation datasets x 13 contact traces = 325 YAML files.
#
# Run this script from DeepRAP/opportunistic_simulator. By default, the
# simulator root is the current working directory. Override it with
# --project-root if needed.

usage() {
    cat <<'EOF'
Usage: generate_experiment_configs.sh [OPTIONS]

Options:
  --project-root PATH  Simulator root (default: current working directory)
  --output-dir PATH    YAML output directory (default: PROJECT_ROOT/configs/generated)
  --force              Archive and replace an existing generated-config directory
  --dry-run            Validate inputs and print the planned file count only
  -h, --help           Show this help
EOF
}

PROJECT_ROOT="$PWD"
OUTPUT_DIR=""
FORCE=false
DRY_RUN=false

while (($#)); do
    case "$1" in
        --project-root)
            [[ $# -ge 2 ]] || { echo "ERROR: --project-root requires a path." >&2; exit 2; }
            PROJECT_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "ERROR: --output-dir requires a path." >&2; exit 2; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd)"
DATA_DIR="$(cd -- "$PROJECT_ROOT/../data" && pwd)"
PROCESSED_DIR="$DATA_DIR/processed"
CHECKPOINT_DIR="$DATA_DIR/checkpoints"
MOBILITY_DIR="$DATA_DIR/mobility"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/configs/generated"
elif [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
fi

ALGORITHMS=(local optimist oppavg oppfl pulse)
DATASETS=(hhar sisfall wesad cifar10 fashion_mnist)
TRACES=(
    cambridge
    community_dense_seed7
    community_dense_seed42
    community_dense_seed123
    community_sparse_seed7
    community_sparse_seed42
    community_sparse_seed123
    edge_markov_dense_seed7
    edge_markov_dense_seed42
    edge_markov_dense_seed123
    edge_markov_sparse_seed7
    edge_markov_sparse_seed42
    edge_markov_sparse_seed123
)

# HHAR nodes use an encoder pretrained on UCI-HAR. The other experiments use
# encoders pretrained on their own dataset.
declare -A CHECKPOINT_BY_DATASET=(
    [hhar]="uci_har_encoder.pt"
    [sisfall]="sisfall_encoder.pt"
    [wesad]="wesad_encoder.pt"
    [cifar10]="cifar10_encoder.pt"
    [fashion_mnist]="fmnist_encoder.pt"
)

failures=0
require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "ERROR: required directory not found: $1" >&2
        failures=$((failures + 1))
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required file not found: $1" >&2
        failures=$((failures + 1))
    fi
}

for dataset in "${DATASETS[@]}"; do
    require_dir "$PROCESSED_DIR/$dataset"
    require_file "$CHECKPOINT_DIR/${CHECKPOINT_BY_DATASET[$dataset]}"
done

for trace in "${TRACES[@]}"; do
    require_file "$MOBILITY_DIR/$trace.csv"
done

if ((failures > 0)); then
    echo "Generation aborted: $failures required input(s) are missing." >&2
    exit 1
fi

expected_count=$((${#ALGORITHMS[@]} * ${#DATASETS[@]} * ${#TRACES[@]}))

if [[ "$DRY_RUN" == true ]]; then
    echo "Inputs valid. Would generate $expected_count YAML files in $OUTPUT_DIR"
    exit 0
fi

if [[ -e "$OUTPUT_DIR" ]]; then
    if [[ "$FORCE" != true ]]; then
        echo "ERROR: output directory already exists: $OUTPUT_DIR" >&2
        echo "Use --force to replace it." >&2
        exit 1
    fi

    # Protect against accidentally moving a broad or unexpected directory.
    case "$OUTPUT_DIR" in
        /|"$PROJECT_ROOT"|"$PROJECT_ROOT/configs")
            echo "ERROR: refusing to archive unsafe output directory: $OUTPUT_DIR" >&2
            exit 1
            ;;
    esac

    backup_path="${OUTPUT_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
    if [[ -e "$backup_path" ]]; then
        echo "ERROR: backup path already exists: $backup_path" >&2
        exit 1
    fi
    mv -- "$OUTPUT_DIR" "$backup_path"
    echo "Archived previous configs to: $backup_path"
fi

mkdir -p -- "$OUTPUT_DIR"

# Paths in each YAML are relative to that YAML's directory. This keeps the
# generated configs portable if the whole DeepRAP project is moved.
relative_path() {
    python3 - "$1" "$OUTPUT_DIR" <<'PY'
import os
import sys

print(os.path.relpath(sys.argv[1], start=sys.argv[2]))
PY
}

generated=0

for dataset in "${DATASETS[@]}"; do
    data_path="$(relative_path "$PROCESSED_DIR/$dataset")"
    checkpoint_path="$(relative_path "$CHECKPOINT_DIR/${CHECKPOINT_BY_DATASET[$dataset]}")"

    for trace in "${TRACES[@]}"; do
        mobility_path="$(relative_path "$MOBILITY_DIR/$trace.csv")"

        for algorithm in "${ALGORITHMS[@]}"; do
            config_path="$OUTPUT_DIR/${dataset}__${trace}__${algorithm}.yaml"
            result_path="$(relative_path "$PROJECT_ROOT/results/$dataset/$trace/$algorithm")"

            algorithm_options=""
            case "$algorithm" in
                oppfl)
                    algorithm_options=$'\nopportunistic_fl:\n  similarity_threshold: 0.0\n  lambda_weight: 1.0\n  encounter_rounds: 2\n  aggregation: momentum\n'
                    ;;
                pulse)
                    algorithm_options=$'\npulse:\n  alpha_max: 0.5\n  utility_momentum: 0.2\n  initial_utility: 0.0\n  ucb_exploration: 0.25\n  norm_epsilon: 1.0e-12\n'
                    ;;
            esac

            cat >"$config_path" <<EOF
# Generated by generate_experiment_configs.sh; do not edit manually.
seeds: [7, 129, 56, 83]

mobility:
  path: $mobility_path
  node_assignment: random

data:
  path: $data_path
  source_split: simulation
  allow_full_fallback: false
  loading_mode: lazy_hdf5
  cache_batch_size: 1024
  cache_num_workers: 4
  cache_pin_memory: true
  cache_prefetch_factor: 2
  num_nodes: null
  node_selection: random
  train_fraction: 0.70
  val_fraction: 0.15
  test_fraction: 0.15

model:
  encoder: auto
  input_channels: null
  embedding_dim: auto
  num_classes: auto
  classifier_hidden_dims: [64]

pretraining:
  checkpoint: $checkpoint_path
  freeze_encoder: false

training:
  interval: 300
  local_steps: 2
  batch_size: 16
  evaluation_batch_size: 1024
  learning_rate: 0.01

algorithm:
  name: $algorithm
${algorithm_options}
evaluation:
  interval: 900

logging:
  progress_interval: 900
  output_dir: $result_path
EOF
            generated=$((generated + 1))
        done
    done
done

if ((generated != expected_count)); then
    echo "ERROR: generated $generated files; expected $expected_count." >&2
    exit 1
fi

# Parse every generated file when PyYAML is available.
if python3 -c 'import yaml' >/dev/null 2>&1; then
    python3 - "$OUTPUT_DIR" "$expected_count" <<'PY'
from pathlib import Path
import sys

import yaml

output_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
paths = sorted(output_dir.glob("*.yaml"))

if len(paths) != expected:
    raise SystemExit(f"Found {len(paths)} YAML files; expected {expected}.")

required = {
    "seeds", "mobility", "data", "model", "pretraining", "training",
    "algorithm", "evaluation", "logging",
}

for path in paths:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    missing = required - config.keys()
    if missing:
        raise SystemExit(f"{path.name}: missing sections {sorted(missing)}")

print(f"Validated {len(paths)} YAML files with PyYAML.")
PY
else
    echo "NOTE: PyYAML is unavailable; skipped YAML parsing validation."
fi

echo "Generated $generated experiment configs in: $OUTPUT_DIR"
