#!/bin/bash

# MindCraft Evaluation Script
# This script runs inference and evaluation separately for MindCraft cognitive tasks

# Array to track all background PIDs
declare -a ALL_PIDS=()

# Cleanup function to kill all child processes
cleanup() {
    echo ""
    echo "========================================="
    echo "⚠️  Caught interrupt signal (Ctrl+C). Cleaning up..."
    echo "========================================="
    
    if [ ${#ALL_PIDS[@]} -gt 0 ]; then
        echo "Terminating ${#ALL_PIDS[@]} background processes..."
        for pid in "${ALL_PIDS[@]}"; do
            if ps -p $pid > /dev/null 2>&1; then
                echo "  Killing process $pid and its children..."
                # Kill the process group to ensure all child processes are terminated
                pkill -P $pid 2>/dev/null || true
                kill $pid 2>/dev/null || true
            fi
        done
        
        # Wait a bit for graceful termination
        sleep 2
        
        # Force kill any remaining processes
        for pid in "${ALL_PIDS[@]}"; do
            if ps -p $pid > /dev/null 2>&1; then
                echo "  Force killing process $pid..."
                kill -9 $pid 2>/dev/null || true
            fi
        done
        
        echo "✓ All processes terminated."
    fi
    
    echo ""
    echo "Cleanup completed. Exiting..."
    exit 130  # Exit code 130 = terminated by Ctrl+C
}

# Set up signal traps
trap cleanup SIGINT SIGTERM

# Ensure we're in the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

MODEL_PATH=${1:-${MODEL_PATH:-./ckpt/navila-8b-8f-sft/checkpoint-4200}}
DATASET_PATH=${2:-${DATASET_PATH:-./data/dataset/mindcraft}}
OUTPUT_DIR=${3:-eval_out/mindcraft}
GPU_LIST=${4:-"5,6"}  # GPU list as a string (e.g., "0,1,2,3")

echo "========================================="
echo "MindCraft Evaluation Pipeline"
echo "========================================="
echo "Model Path: $MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "GPU List: $GPU_LIST"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/inference_results"
mkdir -p "$OUTPUT_DIR/metrics_results"

# Activate conda environment
echo "Activating navila-eval conda environment..."
CONDA_ENV_NAME=${CONDA_ENV_NAME:-navila-eval}
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
else
    echo "Warning: conda not found, using current Python environment."
fi

# Step 1: Run inference to generate predictions
echo ""
echo "Step 1/2: Running inference on MindCraft dataset..."
echo "========================================="

IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}

# Store PIDs
PIDS=()

for IDX in $(seq 0 $((CHUNKS-1))); do
    CHUNK_IDX=$IDX
    GPU=${GPULIST[$IDX]}
    echo "Launching inference on GPU $GPU (Chunk $CHUNK_IDX/$CHUNKS)..."
    
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/vlnce_baselines/mindcraft_inference.py \
        --model-path "$MODEL_PATH" \
        --dataset-path "$DATASET_PATH" \
        --output-dir "$OUTPUT_DIR/inference_results" \
        --num-chunks $CHUNKS \
        --chunk-idx $CHUNK_IDX \
        > "$OUTPUT_DIR/inference_chunk_$CHUNK_IDX.log" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    ALL_PIDS+=($PID)
    echo "Started process with PID: $PID"
done

# Wait for all inference jobs to complete
echo ""
echo "Waiting for all inference jobs to complete..."
echo "  (Press Ctrl+C to cancel)"
for PID in "${PIDS[@]}"; do
    echo "Waiting for PID $PID..."
    if wait $PID 2>/dev/null; then
        echo "  ✓ Process $PID completed successfully"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ] && [ $EXIT_CODE -ne 130 ]; then
            echo "  ⚠️  Process $PID exited with code $EXIT_CODE"
        fi
    fi
done

echo "Inference completed for all chunks."

# Check if inference results exist
if [ ! -f "$OUTPUT_DIR/inference_results/predictions_chunk_0.json" ]; then
    echo "Error: Inference results not found. Check logs in $OUTPUT_DIR/inference_chunk_*.log"
    echo "Printing last 50 lines of chunk 0 log:"
    tail -n 50 "$OUTPUT_DIR/inference_chunk_0.log"
    exit 1
fi

# Step 2: Evaluate predictions and compute metrics
echo ""
echo "Step 2/2: Computing evaluation metrics..."
echo "========================================="

python evaluation/vlnce_baselines/mindcraft_metrics.py \
    --predictions-dir "$OUTPUT_DIR/inference_results" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR/metrics_results" \
    --metrics all

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR/metrics_results"
    echo "========================================="
    
    # Clear trap on successful exit
    trap - SIGINT SIGTERM
else
    echo "Error: Metrics computation failed."
    
    # Clear trap before exit
    trap - SIGINT SIGTERM
    exit 1
fi
