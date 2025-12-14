#!/usr/bin/env python3
"""
Upload a local fine-tuned model folder to the Hugging Face Hub.

Example:
    uv run python scripts/upload_hf_model.py \\
        --repo-id your-name/lca-bge-m3-ft \\
        --token hf_xxx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

try:
    from huggingface_hub import HfApi, upload_folder
except ImportError as exc:  # pragma: no cover - informative failure is enough
    raise SystemExit(
        "huggingface_hub is required. Install it via `pip install huggingface_hub`."
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT_DIR / "data" / "output" / "lca-bge-m3-finetuned"
REQUIRED_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
WEIGHT_SUFFIXES = (".bin", ".safetensors")


def validate_model_dir(model_dir: Path) -> list[str]:
    """Return a list of missing required files in the model directory."""
    missing = [name for name in REQUIRED_FILES if not (model_dir / name).exists()]
    if not any(model_dir.glob(f"*{suffix}") for suffix in WEIGHT_SUFFIXES):
        missing.append("*.bin or *.safetensors weights")
    return missing


def normalize_patterns(patterns: Iterable[str] | None) -> list[str] | None:
    if not patterns:
        return None
    flattened: list[str] = []
    for pattern in patterns:
        if "," in pattern:
            flattened.extend(p.strip() for p in pattern.split(",") if p.strip())
        elif pattern.strip():
            flattened.append(pattern.strip())
    return flattened or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local fine-tuned model to Hugging Face Hub.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to the folder that was produced by fine-tuning (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target repository on Hugging Face Hub, e.g. your-name/lca-bge-m3-ft",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token. If omitted, uses the cached login from `huggingface-cli login`.",
    )
    parser.add_argument("--commit-message", default="Upload fine-tuned model", help="Commit message for the upload")
    parser.add_argument("--private", action="store_true", help="Create the repo as private (only used on first upload)")
    parser.add_argument(
        "--skip-create",
        action="store_true",
        help="Assume the repo already exists and skip the create_repo call.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip checking for required config/tokenizer/weight files before upload.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help="Glob pattern(s) to selectively upload (can be repeated or comma separated).",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        dest="ignore_patterns",
        help="Glob pattern(s) to exclude from the upload (can be repeated or comma separated).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be uploaded without contacting the Hub.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        raise SystemExit(f"模型目录不存在: {model_dir}")

    if not args.skip_validation:
        missing = validate_model_dir(model_dir)
        if missing:
            missing_str = ", ".join(missing)
            raise SystemExit(f"模型目录缺少必要文件: {missing_str}")

    allow_patterns = normalize_patterns(args.allow_patterns)
    ignore_patterns = normalize_patterns(args.ignore_patterns)

    print("=" * 60)
    print("Hugging Face 上传配置")
    print("=" * 60)
    print(f"模型目录: {model_dir}")
    print(f"目标仓库: {args.repo_id}")
    print(f"提交说明: {args.commit_message}")
    print(f"私有仓库: {'是' if args.private else '否'} (仅在创建仓库时生效)")
    if allow_patterns:
        print(f"包含文件: {allow_patterns}")
    if ignore_patterns:
        print(f"排除文件: {ignore_patterns}")
    if args.dry_run:
        print("\nDry-run 模式，不会执行任何上传。")
        return

    api = HfApi(token=args.token)
    if not args.skip_create:
        print("\n检查/创建仓库...")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
            token=args.token,
        )
    else:
        print("\n跳过仓库创建步骤 (skip-create)")

    print("开始上传文件到 Hugging Face Hub ...")
    commit_info = upload_folder(
        repo_id=args.repo_id,
        folder_path=str(model_dir),
        repo_type="model",
        token=args.token,
        commit_message=args.commit_message,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    print("\n上传完成!")
    print(f"仓库: {commit_info.repo_id}")
    print(f"提交: {commit_info.commit_hash}")
    if getattr(commit_info, "commit_url", None):
        print(f"详情: {commit_info.commit_url}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("上传已取消 (Ctrl+C)")
