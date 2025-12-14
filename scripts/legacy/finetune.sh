#!/bin/bash
#
# Fine-tune BGE-M3 model for LCA dataset
#
# Usage:
#   Linux (GPU):  bash scripts/finetune.sh
#   Linux (多卡): bash scripts/finetune.sh <num_gpus>
#   Mac (CPU):    bash scripts/finetune.sh mac
#   nohup bash scripts/finetune.sh 2 > finetune.log 2>&1 &
#

set -e

# ============================================================
# Configuration
# ============================================================

# Model - 使用本地 BGE-M3 模型
MODEL_NAME="data/model/BAAI--bge-m3"

# Paths (export TRAIN_DATA_PATH to override; if training_with_hard_neg.json exists it is used automatically)
TRAIN_DATA_BASE="data/ft_data/training.json"
TRAIN_DATA_HARD="data/ft_data/training_with_hard_neg.json"
if [ -n "$TRAIN_DATA_PATH" ]; then
    TRAIN_DATA="$TRAIN_DATA_PATH"
elif [ -f "$TRAIN_DATA_HARD" ]; then
    TRAIN_DATA="$TRAIN_DATA_HARD"
    echo "[Info] Detected $TRAIN_DATA_HARD, using it for fine-tuning."
else
    TRAIN_DATA="$TRAIN_DATA_BASE"
fi
OUTPUT_DIR="data/output/lca-bge-m3-finetuned"
CACHE_DIR="data/cache/model"
DATA_CACHE_DIR="data/cache/data"
DEEPSPEED_CONFIG="config/ds_stage0.json"

# Training parameters
NUM_EPOCHS=2
LEARNING_RATE=1e-5
PER_DEVICE_BATCH_SIZE=16
TRAIN_GROUP_SIZE=8
QUERY_MAX_LEN=64
PASSAGE_MAX_LEN=1024
TEMPERATURE=0.02
WARMUP_RATIO=0.1
SAVE_STEPS=1000
LOGGING_STEPS=10

# Query instruction (BGE-M3 不需要 instruction)
QUERY_INSTRUCTION=""

# ============================================================
# Detect platform and set mode
# ============================================================

MODE=${1:-"auto"}

# Auto-detect platform
if [ "$MODE" = "auto" ]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        MODE="mac"
    else
        MODE="1"  # Default to single GPU on Linux
    fi
fi

# ============================================================
# Run Training
# ============================================================

echo "============================================================"
echo "LCA Embedding Model Fine-tuning (BGE-M3)"
echo "============================================================"
echo "Model:          $MODEL_NAME"
echo "Train data:     $TRAIN_DATA"
echo "Output dir:     $OUTPUT_DIR"
echo "Mode:           $MODE"
echo "Epochs:         $NUM_EPOCHS"
echo "Learning rate:  $LEARNING_RATE"
echo "============================================================"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$DATA_CACHE_DIR"

if [ "$MODE" = "mac" ]; then
    # Mac 模式: 使用 torchrun 单进程, 不使用 DeepSpeed
    echo "Running in Mac mode (CPU/MPS, no DeepSpeed)..."
    echo "Warning: Training on Mac will be slow. Consider using a Linux server with GPU."
    echo ""

    torchrun --nproc_per_node 1 --standalone \
        -m FlagEmbedding.finetune.embedder.encoder_only.base \
        --model_name_or_path "$MODEL_NAME" \
        --cache_dir "$CACHE_DIR" \
        --train_data "$TRAIN_DATA" \
        --cache_path "$DATA_CACHE_DIR" \
        --train_group_size "$TRAIN_GROUP_SIZE" \
        --query_max_len "$QUERY_MAX_LEN" \
        --passage_max_len "$PASSAGE_MAX_LEN" \
        --pad_to_multiple_of 8 \
        --query_instruction_for_retrieval "$QUERY_INSTRUCTION" \
        --query_instruction_format '{}{}' \
        --knowledge_distillation False \
        --output_dir "$OUTPUT_DIR" \
        --overwrite_output_dir \
        --learning_rate "$LEARNING_RATE" \
        --num_train_epochs "$NUM_EPOCHS" \
        --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
        --dataloader_drop_last True \
        --warmup_ratio "$WARMUP_RATIO" \
        --ddp_find_unused_parameters True \
        --logging_steps "$LOGGING_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --temperature "$TEMPERATURE" \
        --sentence_pooling_method cls \
        --normalize_embeddings True \
        --kd_loss_type kl_div
else
    # Linux GPU 模式: 使用 DeepSpeed + torchrun
    NUM_GPUS="$MODE"
    echo "Running in Linux GPU mode ($NUM_GPUS GPU(s))..."
    echo "Batch size:     $PER_DEVICE_BATCH_SIZE x $NUM_GPUS"
    echo ""

    torchrun --nproc_per_node "$NUM_GPUS" \
        -m FlagEmbedding.finetune.embedder.encoder_only.base \
        --model_name_or_path "$MODEL_NAME" \
        --cache_dir "$CACHE_DIR" \
        --train_data "$TRAIN_DATA" \
        --cache_path "$DATA_CACHE_DIR" \
        --train_group_size "$TRAIN_GROUP_SIZE" \
        --query_max_len "$QUERY_MAX_LEN" \
        --passage_max_len "$PASSAGE_MAX_LEN" \
        --pad_to_multiple_of 8 \
        --query_instruction_for_retrieval "$QUERY_INSTRUCTION" \
        --query_instruction_format '{}{}' \
        --knowledge_distillation False \
        --output_dir "$OUTPUT_DIR" \
        --overwrite_output_dir \
        --learning_rate "$LEARNING_RATE" \
        --bf16 \
        --num_train_epochs "$NUM_EPOCHS" \
        --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
        --dataloader_drop_last True \
        --warmup_ratio "$WARMUP_RATIO" \
        --gradient_checkpointing \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --logging_steps "$LOGGING_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --negatives_cross_device \
        --temperature "$TEMPERATURE" \
        --sentence_pooling_method cls \
        --normalize_embeddings True \
        --kd_loss_type kl_div \
        --weight_decay 0.01
fi

echo ""
echo "============================================================"
echo "Fine-tuning completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "============================================================"
