#!/usr/bin/env bash
set -uo pipefail

# Run every generated DeepRAP configuration with bounded parallelism.
# Each YAML file is one job; the seeds declared inside it are handled by run.py.

usage() {
    cat <<'EOF'
Usage: run_generated_experiments.sh [OPTIONS]

Options:
  --configs-dir PATH  Config directory (default: ./configs/generated)
  --logs-dir PATH     Parent log directory (default: ./logs/experiment_batches)
  --python COMMAND    Python executable (default: python3)
  -j, --jobs N        Maximum concurrent jobs (default: N_PARALLEL_TASK or 4)
  -h, --help          Show this help

Examples:
  N_PARALLEL_TASK=4 ./run_generated_experiments.sh
  ./run_generated_experiments.sh --jobs 2
EOF
}

SIMULATOR_DIR="$PWD"
CONFIGS_DIR="$SIMULATOR_DIR/configs/generated"
LOGS_PARENT="$SIMULATOR_DIR/logs/experiment_batches"
PYTHON_COMMAND="python3"
MAX_PARALLEL="${N_PARALLEL_TASK:-4}"

while (($#)); do
    case "$1" in
        --configs-dir)
            [[ $# -ge 2 ]] || { echo "ERROR: --configs-dir requires a path." >&2; exit 2; }
            CONFIGS_DIR="$2"
            shift 2
            ;;
        --logs-dir)
            [[ $# -ge 2 ]] || { echo "ERROR: --logs-dir requires a path." >&2; exit 2; }
            LOGS_PARENT="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "ERROR: --python requires a command." >&2; exit 2; }
            PYTHON_COMMAND="$2"
            shift 2
            ;;
        -j|--jobs)
            [[ $# -ge 2 ]] || { echo "ERROR: --jobs requires a number." >&2; exit 2; }
            MAX_PARALLEL="$2"
            shift 2
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

if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: N_PARALLEL_TASK/--jobs must be a positive integer." >&2
    exit 2
fi

if [[ ! -f "$SIMULATOR_DIR/run.py" ]]; then
    echo "ERROR: run.py not found in $SIMULATOR_DIR" >&2
    echo "Run this script from DeepRAP/opportunistic_simulator." >&2
    exit 1
fi

if [[ ! -d "$CONFIGS_DIR" ]]; then
    echo "ERROR: config directory not found: $CONFIGS_DIR" >&2
    exit 1
fi

if ! command -v "$PYTHON_COMMAND" >/dev/null 2>&1; then
    echo "ERROR: Python command not found: $PYTHON_COMMAND" >&2
    exit 1
fi

mapfile -d '' CONFIGS < <(
    find "$CONFIGS_DIR" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) \
        -print0 | sort -z
)

TOTAL=${#CONFIGS[@]}
if ((TOTAL == 0)); then
    echo "ERROR: no YAML files found in $CONFIGS_DIR" >&2
    exit 1
fi

BATCH_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOGS_PARENT/$BATCH_ID"
mkdir -p -- "$LOG_DIR"
STATUS_FILE="$LOG_DIR/status.tsv"
printf 'status\texit_code\tconfig\tlog\n' >"$STATUS_FILE"

declare -A PID_TO_CONFIG=()
declare -A PID_TO_LOG=()

launched=0
completed=0
succeeded=0
failed=0

print_progress() {
    local active=$((launched - completed))
    local missing=$((TOTAL - completed))
    printf '[progress] completed=%d/%d | succeeded=%d | failed=%d | running=%d | remaining=%d\n' \
        "$completed" "$TOTAL" "$succeeded" "$failed" "$active" "$missing"
}

stop_children() {
    local pid
    trap - INT TERM
    echo
    echo "Interrupt received; stopping active jobs ..." >&2
    for pid in "${!PID_TO_CONFIG[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    print_progress
    echo "Partial status: $STATUS_FILE" >&2
    exit 130
}

trap stop_children INT TERM

echo "Experiment batch: $BATCH_ID"
echo "Configs:          $CONFIGS_DIR"
echo "Jobs:             $TOTAL"
echo "Max parallel:     $MAX_PARALLEL"
echo "Logs:             $LOG_DIR"
print_progress

while ((completed < TOTAL)); do
    while ((launched < TOTAL && launched - completed < MAX_PARALLEL)); do
        config="${CONFIGS[$launched]}"
        config_name="$(basename -- "$config")"
        log_path="$LOG_DIR/${config_name%.*}.log"

        echo "[start $((launched + 1))/$TOTAL] $config_name"
        (
            printf 'Command: %q run.py --config %q\n\n' "$PYTHON_COMMAND" "$config"
            "$PYTHON_COMMAND" run.py --config "$config"
        ) >"$log_path" 2>&1 &

        pid=$!
        PID_TO_CONFIG["$pid"]="$config"
        PID_TO_LOG["$pid"]="$log_path"
        launched=$((launched + 1))
    done

    # Bash versions before 5.1 do not support `wait -p`. Poll the shell's
    # running-job list to identify a completed PID, then collect its status.
    finished_pid=""
    while [[ -z "$finished_pid" ]]; do
        mapfile -t running_pids < <(jobs -pr)

        for candidate_pid in "${!PID_TO_CONFIG[@]}"; do
            candidate_running=false

            for running_pid in "${running_pids[@]}"; do
                if [[ "$candidate_pid" == "$running_pid" ]]; then
                    candidate_running=true
                    break
                fi
            done

            if [[ "$candidate_running" == false ]]; then
                finished_pid="$candidate_pid"
                break
            fi
        done

        if [[ -z "$finished_pid" ]]; then
            sleep 1
        fi
    done

    wait "$finished_pid"
    exit_code=$?

    if [[ -z "$finished_pid" || -z "${PID_TO_CONFIG[$finished_pid]+present}" ]]; then
        echo "ERROR: could not identify the completed child process." >&2
        exit 1
    fi

    finished_config="${PID_TO_CONFIG[$finished_pid]}"
    finished_log="${PID_TO_LOG[$finished_pid]}"
    finished_name="$(basename -- "$finished_config")"

    if ((exit_code == 0)); then
        status="SUCCEEDED"
        succeeded=$((succeeded + 1))
    else
        status="FAILED"
        failed=$((failed + 1))
    fi

    completed=$((completed + 1))
    printf '%s\t%d\t%s\t%s\n' \
        "$status" "$exit_code" "$finished_config" "$finished_log" >>"$STATUS_FILE"
    printf '[done %d/%d] %s: %s (exit=%d)\n' \
        "$completed" "$TOTAL" "$status" "$finished_name" "$exit_code"

    unset 'PID_TO_CONFIG[$finished_pid]'
    unset 'PID_TO_LOG[$finished_pid]'
    print_progress
done

echo
echo "Batch finished."
print_progress
echo "Status file: $STATUS_FILE"

if ((failed > 0)); then
    echo "One or more jobs failed. Inspect their logs in $LOG_DIR" >&2
    exit 1
fi
