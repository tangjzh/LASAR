#!/bin/bash
# LASAR Training Script
# 从 LlavaLlama checkpoint 开始训练 LASAR 模型
# 
# 关键点：
# 1. 只使用 --model_name_or_path（加载模型权重）
# 2. 不使用 --resume_from_checkpoint（避免 optimizer 状态冲突）
# 3. 新的 optimizer 会自动创建，包含所有参数

export CUDA_VISIBLE_DEVICES=6,7
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
MASTER_ADDR=localhost
MASTER_PORT=29501  # 改个端口避免冲突
CURRENT_RANK=0
n_node=1
GPUS_PER_NODE=2
OUTPUT="./ckpt/lasar-8b-mindcraft"

echo "🚀 Starting LASAR training from LlavaLlama checkpoint..."
echo "   Base model: ./ckpt/navila-llama3-8b-8f"
echo "   Output: $OUTPUT"
echo ""

torchrun --nnodes=$n_node --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --longvila_sampler True \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./ckpt/navila-llama3-8b-8f \
    --version llama_3 \
    --seed 10 \
    --data_mixture r2r+rxr+mindcraft \
    --vision_tower google/siglip-so400m-patch14-384+facebook/VGGT-1B \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT \
    --num_train_epochs 1 \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 10 \
    --do_eval False \
    --save_strategy "steps" \
    --save_steps 500 \
    --fps 0.0 \
    --save_total_limit 2 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb

echo ""
echo "✅ Training script completed!"

