#!/usr/bin/env bash
# Generate synthetic mobility traces and prepare the real Cambridge trace.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_NODES="${NUM_NODES:-10}"
DURATION="${DURATION:-86400}"
TIME_STEP="${TIME_STEP:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data/mobility/synthetic}"
HAGGLE_OUTPUT_DIR="${HAGGLE_OUTPUT_DIR:-${PROJECT_DIR}/data/mobility/haggle}"
HAGGLE_SNAPSHOT_STEP="${HAGGLE_SNAPSHOT_STEP:-120}"
HAGGLE_FORCE_DOWNLOAD="${HAGGLE_FORCE_DOWNLOAD:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SEEDS=(7 42 123)

# Dense Community and Edge-Markov scenarios use the strong-persistence/bias
# parameters; their sparse counterparts reduce the contact opportunity rate.
COMMUNITY_STRONG_INTRA=0.35
COMMUNITY_STRONG_INTER=0.02

# Sparse Community traces preserve the Strong/Weak community bias while
# substantially reducing the number of contact opportunities.
COMMUNITY_SPARSE_STRONG_INTRA=0.070
COMMUNITY_SPARSE_STRONG_INTER=0.004

# Edge-Markov is homogeneous; the dense and sparse variants differ in their
# stationary contact probability while retaining strong temporal persistence.
EDGE_CONTACT_PROBABILITY=0.1025
EDGE_STRONG_PERSISTENCE=0.90

# Sparse Edge-Markov traces use a lower stationary contact probability while
# retaining the same Strong/Weak persistence levels.
EDGE_SPARSE_CONTACT_PROBABILITY=0.02

log() {
    printf '[mobility] %s\n' "$*"
}

validate_setup() {
    local synthetic_generator="${PROJECT_DIR}/scripts/generate_mobility.py"
    local haggle_preparer="${PROJECT_DIR}/scripts/prepare_haggle_traces.py"

    log "Checking the required scripts and Python interpreter..."

    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
        printf 'Error: Python interpreter not found: %s\n' "${PYTHON_BIN}" >&2
        exit 1
    }

    [[ -f "${synthetic_generator}" ]] || {
        printf 'Error: synthetic trace generator not found: %s\n' "${synthetic_generator}" >&2
        exit 1
    }

    [[ -f "${haggle_preparer}" ]] || {
        printf 'Error: Cambridge trace preparer not found: %s\n' "${haggle_preparer}" >&2
        exit 1
    }

    if [[ "${HAGGLE_FORCE_DOWNLOAD}" != "0" && "${HAGGLE_FORCE_DOWNLOAD}" != "1" ]]; then
        printf 'Error: HAGGLE_FORCE_DOWNLOAD must be 0 or 1.\n' >&2
        exit 1
    fi
}

print_configuration() {
    log "Configuration:"
    log "  Python interpreter: ${PYTHON_BIN}"
    log "  Synthetic nodes: ${NUM_NODES}"
    log "  Synthetic duration: ${DURATION} seconds"
    log "  Synthetic time step: ${TIME_STEP} seconds"
    log "  Synthetic seeds: ${SEEDS[*]}"
    log "  Synthetic output: ${OUTPUT_DIR}"
    log "  Sparse Community probabilities: ${COMMUNITY_SPARSE_STRONG_INTRA}/${COMMUNITY_SPARSE_STRONG_INTER}"
    log "  Sparse Edge-Markov contact probability: ${EDGE_SPARSE_CONTACT_PROBABILITY}"
    log "  Cambridge snapshot step: ${HAGGLE_SNAPSHOT_STEP} seconds"
    log "  Cambridge output: ${HAGGLE_OUTPUT_DIR}"
}

generate_community_trace() {
    local strength="$1"
    local intra_probability="$2"
    local inter_probability="$3"
    local seed="$4"
    local output="${OUTPUT_DIR}/community_${strength}_seed${seed}.csv"

    log "Generating Community ${strength} trace (seed=${seed}, intra=${intra_probability}, inter=${inter_probability})..."

    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/generate_mobility.py" \
        --model community \
        --num-nodes "${NUM_NODES}" \
        --duration "${DURATION}" \
        --time-step "${TIME_STEP}" \
        --num-communities 3 \
        --intra-contact-probability "${intra_probability}" \
        --inter-contact-probability "${inter_probability}" \
        --seed "${seed}" \
        --output "${output}"

    log "Created ${output}"
}

generate_edge_markov_trace() {
    local scenario="$1"
    local contact_probability="$2"
    local persistence="$3"
    local seed="$4"
    local output="${OUTPUT_DIR}/edge_markov_${scenario}_seed${seed}.csv"

    log "Generating Edge-Markov ${scenario} trace (seed=${seed}, contact_probability=${contact_probability}, persistence=${persistence})..."

    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/generate_mobility.py" \
        --model edge_markov \
        --num-nodes "${NUM_NODES}" \
        --duration "${DURATION}" \
        --time-step "${TIME_STEP}" \
        --contact-probability "${contact_probability}" \
        --persistence "${persistence}" \
        --seed "${seed}" \
        --output "${output}"

    log "Created ${output}"
}

prepare_cambridge_trace() {
    local command=(
        "${PYTHON_BIN}"
        "${PROJECT_DIR}/scripts/prepare_haggle_traces.py"
        --output-dir "${HAGGLE_OUTPUT_DIR}"
        --snapshot-step "${HAGGLE_SNAPSHOT_STEP}"
    )

    if [[ "${HAGGLE_FORCE_DOWNLOAD}" == "1" ]]; then
        command+=(--force-download)
    fi

    log "Preparing the real Cambridge/Haggle Experiment 2 trace..."
    log "Only contacts among the 12 internal iMotes will be retained."
    "${command[@]}"
    log "Created ${HAGGLE_OUTPUT_DIR}/processed/cambridge.csv"
    log "Created ${HAGGLE_OUTPUT_DIR}/processed/cambridge.summary.json"
}

validate_setup
print_configuration

log "Creating the synthetic output directory..."
mkdir -p "${OUTPUT_DIR}"

log "Starting synthetic trace generation (12 traces in total)..."

for seed in "${SEEDS[@]}"; do
    log "Processing seed ${seed}..."

    generate_community_trace \
        dense "${COMMUNITY_STRONG_INTRA}" "${COMMUNITY_STRONG_INTER}" "${seed}"
    generate_community_trace \
        sparse "${COMMUNITY_SPARSE_STRONG_INTRA}" "${COMMUNITY_SPARSE_STRONG_INTER}" "${seed}"
    generate_edge_markov_trace \
        dense "${EDGE_CONTACT_PROBABILITY}" "${EDGE_STRONG_PERSISTENCE}" "${seed}"
    generate_edge_markov_trace \
        sparse "${EDGE_SPARSE_CONTACT_PROBABILITY}" "${EDGE_STRONG_PERSISTENCE}" "${seed}"
done

log "Synthetic trace generation completed: 12 CSV files plus their summaries."

prepare_cambridge_trace

log "All mobility traces are ready."
log "Synthetic traces: ${OUTPUT_DIR}"
log "Real Cambridge trace: ${HAGGLE_OUTPUT_DIR}/processed/cambridge.csv"
