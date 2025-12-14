#!/bin/bash
#
# Fine-tune Qwen3-Embedding using Sentence Transformers
#
# Usage:
#   Single GPU:   bash scripts/finetune_st.sh
#   Multi-GPU:    bash scripts/finetune_st.sh <num_gpus>
#   Mac (CPU):    bash scripts/finetune_st.sh mac
#   Background:   nohup bash scripts/finetune_st.sh 2 > finetune_st.log 2>&1 &
#

set -e

# ============================================================
# Configuration
# ============================================================

# Model - 使用本地 Qwen3-Embedding 模型
MODEL_NAME="data/model/Qwen--Qwen3-Embedding-0.6B"

# Paths (export TRAIN_DATA_PATH to override; prefer training_with_hard_neg.json if available)
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
TRAIN_DATA="$TRAIN_DATA_BASE"
OUTPUT_DIR="data/output/lca-qwen3-embedding"
CACHE_DIR="data/cache/model"

# Training parameters
NUM_EPOCHS=2
LEARNING_RATE=1e-5
PER_DEVICE_BATCH_SIZE=8
MAX_SEQ_LENGTH=1024
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
LOGGING_STEPS=10
SAVE_STEPS=1000

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
echo "LCA Embedding Model Fine-tuning (Sentence Transformers)"
echo "============================================================"
echo "Model:          $MODEL_NAME"
echo "Train data:     $TRAIN_DATA"
echo "Output dir:     $OUTPUT_DIR"
echo "Mode:           $MODE"
echo "Epochs:         $NUM_EPOCHS"
echo "Learning rate:  $LEARNING_RATE"
echo "Batch size:     $PER_DEVICE_BATCH_SIZE"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo "============================================================"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

if [ "$MODE" = "mac" ]; then
    # Mac 模式: 使用单进程, 不使用分布式
    echo "Running in Mac mode (CPU/MPS)..."
    echo "Warning: Training on Mac will be slow. Consider using a Linux server with GPU."
    echo ""

    python scripts/finetune_st.py \
        --model_name "$MODEL_NAME" \
        --train_data "$TRAIN_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --cache_dir "$CACHE_DIR" \
        --epochs "$NUM_EPOCHS" \
        --batch_size "$PER_DEVICE_BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --warmup_ratio "$WARMUP_RATIO" \
        --weight_decay "$WEIGHT_DECAY" \
        --logging_steps "$LOGGING_STEPS" \
        --save_steps "$SAVE_STEPS"
else
    # Linux GPU 模式: 使用 accelerate
    NUM_GPUS="$MODE"
    echo "Running in Linux GPU mode ($NUM_GPUS GPU(s))..."
    echo ""

    if [ "$NUM_GPUS" = "1" ]; then
        # Single GPU - 直接运行
        python scripts/finetune_st.py \
            --model_name "$MODEL_NAME" \
            --train_data "$TRAIN_DATA" \
            --output_dir "$OUTPUT_DIR" \
            --cache_dir "$CACHE_DIR" \
            --epochs "$NUM_EPOCHS" \
            --batch_size "$PER_DEVICE_BATCH_SIZE" \
            --lr "$LEARNING_RATE" \
            --max_seq_length "$MAX_SEQ_LENGTH" \
            --warmup_ratio "$WARMUP_RATIO" \
            --weight_decay "$WEIGHT_DECAY" \
            --logging_steps "$LOGGING_STEPS" \
            --save_steps "$SAVE_STEPS" \
            --bf16
    else
        # Multi-GPU - 使用 accelerate
        accelerate launch --num_processes "$NUM_GPUS" \
            scripts/finetune_st.py \
            --model_name "$MODEL_NAME" \
            --train_data "$TRAIN_DATA" \
            --output_dir "$OUTPUT_DIR" \
            --cache_dir "$CACHE_DIR" \
            --epochs "$NUM_EPOCHS" \
            --batch_size "$PER_DEVICE_BATCH_SIZE" \
            --lr "$LEARNING_RATE" \
            --max_seq_length "$MAX_SEQ_LENGTH" \
            --warmup_ratio "$WARMUP_RATIO" \
            --weight_decay "$WEIGHT_DECAY" \
            --logging_steps "$LOGGING_STEPS" \
            --save_steps "$SAVE_STEPS" \
            --bf16
    fi
fi

echo ""
echo "============================================================"
echo "Fine-tuning completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "============================================================"

# Run evaluation if training was successful
echo ""
echo "Running evaluation..."
python scripts/evaluate_st.py \
    --raw_model "$MODEL_NAME" \
    --finetuned_path "$OUTPUT_DIR"
