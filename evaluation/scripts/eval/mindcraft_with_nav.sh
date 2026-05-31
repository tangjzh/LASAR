#!/bin/bash

# MindCraft Complete Evaluation Pipeline with R2R Navigation
# This script:
# 1. Runs R2R navigation evaluation first
# 2. Extracts navigation success labels from R2R results
# 3. Runs MindCraft cognitive query inference
# 4. Computes all MindCraft metrics including GCA and SR@WA

set -e  # Exit on error

# Array to track all background PIDs
declare -a ALL_PIDS=()

# Cleanup function to kill all child processes
cleanup() {
    echo ""
    echo "================================================================================"
    echo "⚠️  Caught interrupt signal (Ctrl+C). Cleaning up..."
    echo "================================================================================"
    
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

# Set up signal traps (only for SIGINT and SIGTERM, not EXIT during normal completion)
trap cleanup SIGINT SIGTERM

# Ensure we're in the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH=${1:-${MODEL_PATH:-./ckpt/navila-llama3-8b-8f}}
DATASET_PATH=${2:-${DATASET_PATH:-./data/dataset/mindcraft}}
OUTPUT_DIR=${3:-eval_out/navila_mindcraft_with_nav}
GPU_LIST=${4:-"2,3"}

echo "================================================================================"
echo "MindCraft Complete Evaluation Pipeline with R2R Navigation"
echo "================================================================================"
echo "Model Path: $MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "GPU List: $GPU_LIST"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/r2r_eval"
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
echo ""

# ============================================================================
# STEP 1: Run R2R Navigation Evaluation
# ============================================================================
echo "================================================================================"
echo "STEP 1/4: Running R2R Navigation Evaluation"
echo "================================================================================"
echo "This will evaluate the model's navigation performance on R2R dataset."
echo "Results will be used to compute GCA and SR@WA metrics."
echo ""

# Parse GPU list
IFS=',' read -ra GPULIST <<< "$GPU_LIST"
NUM_GPUS=${#GPULIST[@]}

# R2R evaluation parameters
TOTAL_CHUNKS=8  # Adjust based on your dataset size
IDX_START=0

echo "Running R2R evaluation with $NUM_GPUS GPUs..."
echo "Total chunks: $TOTAL_CHUNKS, Starting index: $IDX_START"
echo ""

# Create R2R logs directory (use absolute path)
R2R_LOG_DIR="$PROJECT_ROOT/$OUTPUT_DIR/r2r_eval/logs"
mkdir -p "$R2R_LOG_DIR"

echo "R2R logs will be saved to: $R2R_LOG_DIR"
echo ""

# Run R2R evaluation using the existing r2r.sh approach
# We'll run it inline here to have better control
cd evaluation

R2R_PIDS=()
for IDX in $(seq 0 $((NUM_GPUS-1))); do
    CHUNK_IDX=$((IDX + IDX_START))
    GPU=${GPULIST[$IDX]}
    echo "  [R2R] Launching evaluation on GPU $GPU (Chunk $CHUNK_IDX/$TOTAL_CHUNKS)..."
    
    # Use absolute paths for logs
    LOG_FILE="$R2R_LOG_DIR/chunk_$CHUNK_IDX.log"
    
    CUDA_VISIBLE_DEVICES=$GPU python run.py \
        --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml \
        --run-type eval \
        --num-chunks $TOTAL_CHUNKS \
        --chunk-idx $CHUNK_IDX \
        EVAL_CKPT_PATH_DIR "$MODEL_PATH" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    R2R_PIDS+=($PID)
    ALL_PIDS+=($PID)
    echo "  [R2R] Started chunk $CHUNK_IDX with PID $PID (log: $LOG_FILE)"
done

echo ""
echo "Waiting for R2R navigation evaluation to complete..."
echo "  (Press Ctrl+C to cancel)"

# Wait for all R2R processes
for PID in "${R2R_PIDS[@]}"; do
    if wait $PID 2>/dev/null; then
        echo "  ✓ R2R process $PID completed successfully"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ] && [ $EXIT_CODE -ne 130 ]; then
            echo "  ⚠️  R2R process $PID exited with code $EXIT_CODE"
        fi
    fi
done

echo "R2R evaluation completed!"
echo ""

# Return to project root
cd "$PROJECT_ROOT"

# ============================================================================
# STEP 2: Extract Navigation Success Labels from R2R Results
# ============================================================================
echo "================================================================================"
echo "STEP 2/4: Extracting Navigation Success Labels from R2R Results"
echo "================================================================================"

# Find the most recent R2R evaluation results
# R2R saves chunk files like val_unseen_8-0.json in evaluation/eval_out/checkpoint-*/VLN-CE-v1/val_unseen/
echo "Searching for R2R evaluation results..."
echo "  Looking in: evaluation/eval_out/"
echo ""

# Find R2R result directory (contains val_unseen_*.json files)
R2R_RESULTS_DIR=$(find evaluation/eval_out -type d -name "val_unseen" 2>/dev/null | head -1)

if [ -z "$R2R_RESULTS_DIR" ]; then
    echo "⚠️  Warning: R2R evaluation results directory not found!"
    echo ""
    echo "   Searched: evaluation/eval_out/"
    echo "   Looking for: val_unseen/ directory"
    echo ""
    echo "   Available directories:"
    find evaluation/eval_out -type d -maxdepth 3 2>/dev/null | head -10
    echo ""
    echo "   R2R logs (check for errors):"
    ls -lh "$R2R_LOG_DIR/" 2>/dev/null || echo "   No logs found"
    echo ""
    echo "   Will use default navigation success assumption instead."
    NAV_RESULTS_FILE=""
else
    echo "✓ Found R2R results directory: $R2R_RESULTS_DIR"
    
    # Count JSON files
    R2R_JSON_COUNT=$(ls "$R2R_RESULTS_DIR"/*.json 2>/dev/null | wc -l)
    echo "  Found $R2R_JSON_COUNT JSON files"
    echo ""
    
    # Extract navigation success labels from chunk files
    NAV_RESULTS_FILE="$OUTPUT_DIR/navigation_success_from_r2r.json"
    
    echo "Extracting navigation success labels from R2R chunks..."
    python evaluation/analyze_mindcraft_r2r_mapping.py \
        --mindcraft-path "$DATASET_PATH" \
        --r2r-results-dir "$R2R_RESULTS_DIR" \
        --output "$NAV_RESULTS_FILE"
    
    if [ -f "$NAV_RESULTS_FILE" ]; then
        echo ""
        echo "✓ Navigation success labels extracted to: $NAV_RESULTS_FILE"
    else
        echo "⚠️  Warning: Failed to extract navigation labels."
        NAV_RESULTS_FILE=""
    fi
fi

echo ""

# ============================================================================
# STEP 3: Run MindCraft Cognitive Query Inference
# ============================================================================
echo "================================================================================"
echo "STEP 3/4: Running MindCraft Cognitive Query Inference"
echo "================================================================================"
echo "This will generate answers to cognitive queries for each episode."
echo ""

# Store PIDs for inference jobs
MC_PIDS=()

for IDX in $(seq 0 $((NUM_GPUS-1))); do
    CHUNK_IDX=$IDX
    GPU=${GPULIST[$IDX]}
    echo "  [MindCraft] Launching inference on GPU $GPU (Chunk $CHUNK_IDX/$NUM_GPUS)..."
    
    # Use absolute path for log
    MC_LOG_FILE="$PROJECT_ROOT/$OUTPUT_DIR/inference_chunk_$CHUNK_IDX.log"
    
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/vlnce_baselines/mindcraft_inference.py \
        --model-path "$MODEL_PATH" \
        --dataset-path "$DATASET_PATH" \
        --output-dir "$OUTPUT_DIR/inference_results" \
        --num-chunks $NUM_GPUS \
        --chunk-idx $CHUNK_IDX \
        > "$MC_LOG_FILE" 2>&1 &
    
    PID=$!
    MC_PIDS+=($PID)
    ALL_PIDS+=($PID)
    echo "  [MindCraft] Started chunk $CHUNK_IDX with PID $PID (log: $MC_LOG_FILE)"
done

echo ""
echo "Waiting for all MindCraft inference jobs to complete..."
echo "  (Press Ctrl+C to cancel)"

# Wait for all inference jobs
for PID in "${MC_PIDS[@]}"; do
    if wait $PID 2>/dev/null; then
        echo "  ✓ MindCraft process $PID completed successfully"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ] && [ $EXIT_CODE -ne 130 ]; then
            echo "  ⚠️  MindCraft process $PID exited with code $EXIT_CODE"
        fi
    fi
done

echo "MindCraft inference completed!"
echo ""

# Verify inference results exist
if [ ! -f "$OUTPUT_DIR/inference_results/predictions_chunk_0.json" ]; then
    echo "❌ Error: MindCraft inference results not found!"
    echo "   Check logs: $OUTPUT_DIR/inference_chunk_*.log"
    echo ""
    echo "Last 30 lines of chunk 0 log:"
    tail -n 30 "$OUTPUT_DIR/inference_chunk_0.log"
    exit 1
fi

echo "✓ Inference results verified"
echo ""

# ============================================================================
# STEP 4: Compute MindCraft Evaluation Metrics
# ============================================================================
echo "================================================================================"
echo "STEP 4/4: Computing MindCraft Evaluation Metrics"
echo "================================================================================"

if [ -n "$NAV_RESULTS_FILE" ] && [ -f "$NAV_RESULTS_FILE" ]; then
    echo "Using navigation results from R2R evaluation:"
    echo "  $NAV_RESULTS_FILE"
    echo ""
    echo "Will compute all 4 metrics: QA-Acc, CMC, GCA, SR@WA"
    echo ""
    
    python evaluation/vlnce_baselines/mindcraft_metrics.py \
        --predictions-dir "$OUTPUT_DIR/inference_results" \
        --dataset-path "$DATASET_PATH" \
        --output-dir "$OUTPUT_DIR/metrics_results" \
        --navigation-results "$NAV_RESULTS_FILE" \
        --metrics all
else
    echo "⚠️  No navigation results available."
    echo "   Will compute only: QA-Acc, CMC"
    echo "   (GCA and SR@WA require navigation success labels)"
    echo ""
    
    python evaluation/vlnce_baselines/mindcraft_metrics.py \
        --predictions-dir "$OUTPUT_DIR/inference_results" \
        --dataset-path "$DATASET_PATH" \
        --output-dir "$OUTPUT_DIR/metrics_results" \
        --metrics qa-acc,cmc
fi

# Check if metrics computation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✅ EVALUATION COMPLETED SUCCESSFULLY!"
    echo "================================================================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/metrics_results/metrics_results.json"
    echo ""
    
    if [ -n "$NAV_RESULTS_FILE" ] && [ -f "$NAV_RESULTS_FILE" ]; then
        echo "Navigation labels: $NAV_RESULTS_FILE"
        echo ""
        echo "📊 Computed metrics:"
        echo "   • QA-Acc: Overall Reasoning Accuracy"
        echo "   • CMC: Cognitive Map Consistency"
        echo "   • GCA: Goal-Conditioned Accuracy (from R2R navigation)"
        echo "   • SR@WA: Success Rate with Wrong Answers (from R2R navigation)"
    else
        echo "📊 Computed metrics:"
        echo "   • QA-Acc: Overall Reasoning Accuracy"
        echo "   • CMC: Cognitive Map Consistency"
        echo ""
        echo "ℹ️  To compute GCA and SR@WA, R2R navigation results are needed."
    fi
    
    echo ""
    echo "View results:"
    echo "  cat $OUTPUT_DIR/metrics_results/metrics_results.json"
    echo ""
    echo "R2R logs: $OUTPUT_DIR/r2r_eval/logs/"
    echo "MindCraft logs: $OUTPUT_DIR/inference_chunk_*.log"
    echo "================================================================================"
    
    # Clear trap on successful exit
    trap - SIGINT SIGTERM
else
    echo ""
    echo "================================================================================"
    echo "❌ ERROR: Metrics computation failed!"
    echo "================================================================================"
    echo "Check the output above for error details."
    
    # Clear trap before exit
    trap - SIGINT SIGTERM
    exit 1
fi
