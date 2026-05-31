#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${1:-${EVAL_CKPT_PATH_DIR:-}}"
TOTAL_CHUNKS="${2:-1}"
IDX_START="${3:-0}"
GPU_LIST="${4:-0}"  # e.g., "0,1,2,3"
LOG_DIR="${5:-$EVAL_DIR/logs_rxr}"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 MODEL_PATH [TOTAL_CHUNKS] [IDX_START] [GPU_LIST] [LOG_DIR]"
    echo "Example: $0 ./ckpt/exp_20260211-120000 4 0 \"0,1,2,3\""
    exit 1
fi

# Activate evaluation environment (default: navila-eval) when conda is available.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-navila-eval}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    set +u
    if ! conda activate "$CONDA_ENV_NAME"; then
        set -u
        echo "Error: failed to activate conda env: $CONDA_ENV_NAME"
        exit 1
    fi
    set -u
else
    echo "Warning: conda not found, using current Python environment."
fi

mkdir -p "$LOG_DIR"

cd "$EVAL_DIR"
IFS=, read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}
declare -a PIDS=()

for IDX in $(seq 0 $((CHUNKS - 1))); do
    CHUNK_IDX=$((IDX + IDX_START))
    LOG_FILE="$LOG_DIR/rxr_chunk_${CHUNK_IDX}.log"

    echo "[RxR] Total Chunks: $TOTAL_CHUNKS, Local Chunks: $CHUNKS, Chunk Index: $CHUNK_IDX, GPU: ${GPULIST[$IDX]}"

    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python run.py \
        --exp-config vlnce_baselines/config/rxr_baselines/navila.yaml \
        --run-type eval \
        --num-chunks "$TOTAL_CHUNKS" \
        --chunk-idx "$CHUNK_IDX" \
        EVAL_CKPT_PATH_DIR "$MODEL_PATH" \
        > "$LOG_FILE" 2>&1 &
    PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILED=1
    fi
done

if [[ "$FAILED" -ne 0 ]]; then
    echo "Error: one or more RxR chunks failed. Check logs in: $LOG_DIR"
    exit 1
fi

echo "All RxR chunks finished. Logs saved to: $LOG_DIR"
