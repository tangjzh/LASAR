#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
CURRENT_RANK="${CURRENT_RANK:-0}"
n_node="${N_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

MODEL_PATH="${MODEL_PATH:-./ckpt/navila-llama3-8b-8f}"
DATA_MIXTURE="${DATA_MIXTURE:-r2r+rxr}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT="${OUTPUT:-./ckpt/exp_${RUN_ID}}"

SEED="${SEED:-10}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-8}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"

EXP_NAME="${EXP_NAME:-${RUN_ID}}"
ENABLE_GPU_IDLE_WATCH="${ENABLE_GPU_IDLE_WATCH:-0}"
GPU_IDLE_MAX_UTILIZATION="${GPU_IDLE_MAX_UTILIZATION:-10}"
GPU_IDLE_MAX_MEMORY_USED_MB="${GPU_IDLE_MAX_MEMORY_USED_MB:-2048}"
GPU_IDLE_MIN_MEMORY_FREE_MB="${GPU_IDLE_MIN_MEMORY_FREE_MB:-}"
GPU_IDLE_INTERVAL_SECONDS="${GPU_IDLE_INTERVAL_SECONDS:-30}"
GPU_IDLE_CONSECUTIVE_CHECKS="${GPU_IDLE_CONSECUTIVE_CHECKS:-2}"
GPU_IDLE_TIMEOUT_SECONDS="${GPU_IDLE_TIMEOUT_SECONDS:-0}"
ENABLE_START_NOTIFY="${ENABLE_START_NOTIFY:-1}"

BEACON_DIR="${BEACON_DIR:-$ROOT_DIR/beacon}"
BEACON_NOTIFY_SCRIPT="$BEACON_DIR/notify_feishu_flow.py"
BEACON_GPU_WATCH_SCRIPT="$BEACON_DIR/watch_gpu_idle.py"

GLOBAL_BATCH_SIZE=$((GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))

gpu_snapshot="unavailable"
if command -v nvidia-smi >/dev/null 2>&1; then
  if gpu_raw=$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null); then
    gpu_snapshot=$(echo "$gpu_raw" | awk 'NF { if (count++) printf "; "; printf "%s", $0 } END { if (count == 0) printf "unavailable" }')
  fi
fi

if [[ "$ENABLE_GPU_IDLE_WATCH" == "1" ]]; then
  if [[ -f "$BEACON_GPU_WATCH_SCRIPT" ]]; then
    WATCH_ARGS=(
      --exp-name "$EXP_NAME"
      --gpu-ids "$CUDA_VISIBLE_DEVICES"
      --required-gpu-count "$GPUS_PER_NODE"
      --max-utilization "$GPU_IDLE_MAX_UTILIZATION"
      --max-memory-used-mb "$GPU_IDLE_MAX_MEMORY_USED_MB"
      --interval-seconds "$GPU_IDLE_INTERVAL_SECONDS"
      --consecutive-checks "$GPU_IDLE_CONSECUTIVE_CHECKS"
      --timeout-seconds "$GPU_IDLE_TIMEOUT_SECONDS"
      --status info
      --status-color blue
      --message "gpu-idle gate passed; ready to launch training"
    )
    if [[ -n "$GPU_IDLE_MIN_MEMORY_FREE_MB" ]]; then
      WATCH_ARGS+=(--min-memory-free-mb "$GPU_IDLE_MIN_MEMORY_FREE_MB")
    fi
    echo "[sft_8frames] waiting for idle GPUs via beacon/watch_gpu_idle.py ..."
    python3 "$BEACON_GPU_WATCH_SCRIPT" "${WATCH_ARGS[@]}"
  else
    echo "[sft_8frames] ENABLE_GPU_IDLE_WATCH=1 but script not found: $BEACON_GPU_WATCH_SCRIPT"
    exit 1
  fi
fi

if [[ "$ENABLE_START_NOTIFY" == "1" && -f "$BEACON_NOTIFY_SCRIPT" ]]; then
  START_MESSAGE="**run_id**: \`$RUN_ID\`

**data_mixture**: \`$DATA_MIXTURE\`

**gpu_list**: \`$CUDA_VISIBLE_DEVICES\`

**global_batch**: \`$GLOBAL_BATCH_SIZE\` ($GPUS_PER_NODE x $PER_DEVICE_TRAIN_BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS)

**model_path**: \`$MODEL_PATH\`

**output**: \`$OUTPUT\`

**gpu_snapshot**: $gpu_snapshot"
  python3 "$BEACON_NOTIFY_SCRIPT" \
    --status started \
    --status-color blue \
    --exp-name "$EXP_NAME" \
    --message "$START_MESSAGE" || true
fi

echo "[sft_8frames] RUN_ID=${RUN_ID}"
echo "[sft_8frames] OUTPUT=${OUTPUT}"
echo "[sft_8frames] DATA_MIXTURE=${DATA_MIXTURE}"
echo "[sft_8frames] GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} (${GPUS_PER_NODE} x ${PER_DEVICE_TRAIN_BATCH_SIZE} x ${GRADIENT_ACCUMULATION_STEPS})"

torchrun --nnodes="$n_node" --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" \
    --master_addr "$MASTER_ADDR" --node_rank="$CURRENT_RANK" \
    llava/train/train_mem.py \
    --longvila_sampler True \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "$MODEL_PATH" \
    --version llama_3 \
    --seed "$SEED" \
    --data_mixture "$DATA_MIXTURE" \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames "$NUM_VIDEO_FRAMES" \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir "$OUTPUT" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --do_eval False \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --fps 0.0 \
    --save_total_limit 1 \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps "$LOGGING_STEPS" \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_lasar_injection "${USE_LASAR_INJECTION:-False}" \
    --tune_lasar_modules "${TUNE_LASAR_MODULES:-False}"
