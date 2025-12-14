#!/usr/bin/env python3
"""
Fine-tune Qwen3-Embedding using Sentence Transformers

This script fine-tunes Qwen3-Embedding-0.6B on the LCA dataset using
Sentence Transformers v3 with MultipleNegativesRankingLoss.

Usage:
    # Single GPU
    python scripts/finetune_st.py

    # Multi-GPU with accelerate
    accelerate launch scripts/finetune_st.py

    # Custom parameters
    python scripts/finetune_st.py --batch_size 32 --epochs 3 --lr 2e-5
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL_NAME = "data/model/Qwen--Qwen3-Embedding-0.6B"
DEFAULT_TRAIN_DATA_BASE = "data/ft_data/training.json"
DEFAULT_TRAIN_DATA_HARD = "data/ft_data/training_with_hard_neg.json"
if Path(DEFAULT_TRAIN_DATA_HARD).exists():
    DEFAULT_TRAIN_DATA = DEFAULT_TRAIN_DATA_HARD
else:
    DEFAULT_TRAIN_DATA = DEFAULT_TRAIN_DATA_BASE
DEFAULT_TRAIN_DATA = DEFAULT_TRAIN_DATA_BASE
DEFAULT_OUTPUT_DIR = "data/output/lca-qwen3-st-finetuned"
DEFAULT_CACHE_DIR = "data/cache/model"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-Embedding with Sentence Transformers"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Cache directory for model files",
    )

    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        default=DEFAULT_TRAIN_DATA,
        help="Path to training data (JSON format)",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=1024, help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Per-device training batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bf16 precision when supported (default: enabled)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing (default: off for multi-GPU stability)",
    )

    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save checkpoint every N steps"
    )

    return parser.parse_args()


def bf16_is_supported() -> bool:
    """Return True if the current CUDA device can run bf16 kernels."""
    if not torch.cuda.is_available():
        return False

    if hasattr(torch.cuda, "is_bf16_supported"):
        return torch.cuda.is_bf16_supported()

    try:
        major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return major >= 8


def _get_auto_model(model: SentenceTransformer):
    if not len(model):
        return None
    first_module = model[0]
    return getattr(first_module, "auto_model", None)


def configure_model_gradient_checkpointing(model: SentenceTransformer, enabled: bool) -> None:
    """Ensure the underlying HF auto model honors our gradient checkpointing flag."""
    auto_model = _get_auto_model(model)
    if auto_model is None:
        return

    if enabled:
        if hasattr(auto_model, "gradient_checkpointing_enable"):
            auto_model.gradient_checkpointing_enable()
    else:
        if hasattr(auto_model, "gradient_checkpointing_disable"):
            auto_model.gradient_checkpointing_disable()

    if hasattr(auto_model, "config"):
        auto_model.config.gradient_checkpointing = enabled


def enforce_model_torch_dtype(model: SentenceTransformer, dtype: torch.dtype) -> None:
    """
    Make sure all modules (and the HF config) are aligned to the requested dtype.
    This keeps the saved safetensors files in the same precision we trained with.
    """
    if dtype is None:
        return

    auto_model = _get_auto_model(model)
    if auto_model is None:
        return

    model.to(dtype=dtype)

    if hasattr(auto_model, "config"):
        dtype_str = str(dtype).replace("torch.", "")
        auto_model.config.torch_dtype = dtype_str


def force_single_gpu(training_args: SentenceTransformerTrainingArguments) -> None:
    """
    HF Trainer 会在同一进程可见多块 GPU 时自动切到 DataParallel。
    我们的单机脚本只期望使用单 GPU（多卡通过 accelerate 调用），
    因此这里在 WORLD_SIZE==1 的情况下把 n_gpu 强制成 1，避免 DP 带来的奇怪错误。
    """
    if not torch.cuda.is_available():
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        return

    # cached_property, 访问一次即可完成设备初始化
    _ = training_args._setup_devices
    if getattr(training_args, "_n_gpu", 0) > 1:
        print(
            "[Info] Detected multiple visible GPUs but single-process training. "
            "Disabling DataParallel to keep the run on one GPU."
        )
        training_args._n_gpu = 1


def load_training_data(data_path: str) -> Dataset:
    """
    Load training data from JSON file and convert to Sentence Transformers format.

    Expected input format (per line):
    {
        "query": "...",
        "pos": ["positive passage 1", ...],
        "neg": ["negative passage 1", ...],
        ...
    }

    Output format for MultipleNegativesRankingLoss:
    - anchor: query text
    - positive: positive passage text
    (negatives are handled via in-batch negatives)
    """
    print(f"\nLoading training data from {data_path}...")

    # Load JSON lines
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    print(f"  Loaded {len(data)} samples")

    # Convert to Sentence Transformers format
    # For MultipleNegativesRankingLoss, we need (anchor, positive) pairs
    # The loss function handles in-batch negatives automatically
    anchors = []
    positives = []

    for item in data:
        query = item["query"]
        # Use all positive passages (typically just one)
        for pos in item["pos"]:
            anchors.append(query)
            positives.append(pos)

    print(f"  Created {len(anchors)} training pairs")

    # Create dataset
    dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})

    return dataset


def main():
    args = parse_args()

    if args.bf16:
        if not torch.cuda.is_available():
            print("[Warning] BF16 precision requested but CUDA is not available. Falling back to fp32.")
            args.bf16 = False
        elif not bf16_is_supported():
            device_name = torch.cuda.get_device_name(0)
            print(
                f"[Warning] BF16 precision requested but GPU '{device_name}' does not fully support BF16. "
                "Falling back to fp32."
            )
            args.bf16 = False

    print("=" * 60)
    print("LCA Embedding Fine-tuning (Sentence Transformers)")
    print("=" * 60)
    print(f"Model:          {args.model_name}")
    print(f"Train data:     {args.train_data}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.lr}")
    print(f"Max seq length: {args.max_seq_length}")
    precision_str = "bf16" if args.bf16 else "fp32"
    print(f"Precision:      {precision_str}")
    gc_str = "enabled" if args.gradient_checkpointing else "disabled"
    print(f"Grad ckpt:      {gc_str}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model_kwargs = {}
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = SentenceTransformer(
        args.model_name,
        cache_folder=args.cache_dir,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )
    configure_model_gradient_checkpointing(model, args.gradient_checkpointing)
    if args.bf16:
        enforce_model_torch_dtype(model, torch.bfloat16)

    # Set max sequence length
    model.max_seq_length = args.max_seq_length
    print(f"  Max sequence length: {model.max_seq_length}")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Load training data
    train_dataset = load_training_data(args.train_data)

    # Define loss function
    # MultipleNegativesRankingLoss uses in-batch negatives:
    # For each (anchor, positive) pair, all other positives in the batch
    # are treated as negatives
    print("\nSetting up loss function...")
    loss = MultipleNegativesRankingLoss(model)

    # Define training arguments
    print("\nConfiguring training arguments...")
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        # Training parameters
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        # Precision
        bf16=args.bf16,
        # Gradient checkpointing (saves memory)
        gradient_checkpointing=args.gradient_checkpointing,
        # Batch sampler - important for contrastive learning
        # NO_DUPLICATES ensures no duplicate passages in same batch
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        # Saving
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        # Other settings
        dataloader_drop_last=True,
        seed=42,
        # Disable evaluation during training (we'll evaluate separately)
        eval_strategy="no",
    )
    force_single_gpu(training_args)

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save final model
    print("\n" + "=" * 60)
    print("Saving final model...")
    print("=" * 60)

    if args.bf16:
        enforce_model_torch_dtype(model, torch.bfloat16)
    model.save(args.output_dir)

    print(f"\nModel saved to: {args.output_dir}")
    print("\n" + "=" * 60)
    print("Fine-tuning completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
