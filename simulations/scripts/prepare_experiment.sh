#!/usr/bin/env bash
# Build the complete input pipeline for one Opportunistic Learning experiment.
# Run from anywhere: bash scripts/prepare_experiment.sh

set -euo pipefail

# Main experiment settings.
SEED=7
NUM_NODES=9
MOBILITY_MODEL="community"  # community | edge_markov

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

DATA_DIR="data/processed"
CHECKPOINT="data/checkpoints/uci_encoder.pt"


MOBILITY_TRACE="data/mobility/${MOBILITY_MODEL}_seed${SEED}.csv"

echo "[1/3] Generating ${MOBILITY_MODEL} (strong community) mobility trace..."
python3 scripts/generate_mobility.py \
    --model "${MOBILITY_MODEL}" \
    --num-nodes "${NUM_NODES}" \
    --duration 86400 \
    --time-step 20 \
    --num-communities 3 \
    --intra-contact-probability 0.35 \
    --inter-contact-probability 0.02 \
    --seed "${SEED}" \
    --output "${MOBILITY_TRACE}"

echo "[1/3] Generating ${MOBILITY_MODEL} (weak community) mobility trace..."
python3 scripts/generate_mobility.py \
    --model "${MOBILITY_MODEL}" \
    --num-nodes "${NUM_NODES}" \
    --duration 86400 \
    --time-step 20 \
    --num-communities 3 \
    --intra-contact-probability 0.35 \
    --inter-contact-probability 0.02 \
    --seed "${SEED}" \
    --output "${MOBILITY_TRACE}"

echo "[2/3] Downloading and preparing UCI HAR and HHAR..."
python3 scripts/prepare_har_datasets.py \
    --dataset all \
    --raw-dir data/raw \
    --output-dir "${DATA_DIR}" \
    --window-size 128 \
    --stride 128

echo "[3/3] Pretraining the encoder on UCI HAR..."
PYTHONPATH="${PROJECT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}" \
python3 scripts/pretrain_encoder.py \
    --data "${DATA_DIR}/uci_har.npz" \
    --output "${CHECKPOINT}" \
    --num-classes 6 \
    --embedding-dim 64 \
    --input-channels 3 \
    --epochs 20 \
    --batch-size 64

echo "Done. Trace: ${MOBILITY_TRACE} | checkpoint: ${CHECKPOINT}"
