#!/usr/bin/env python3
"""
下载 Hugging Face 模型脚本
支持断点续传、网络重试、镜像源切换
"""

import os
import sys
import time
import argparse
from pathlib import Path

# 模型配置
MODEL_ID = "BAAI/bge-m3"
MODEL_BASE_DIR = Path(__file__).parent.parent / "data" / "model"

# 镜像源列表
HF_MIRRORS = [
    None,  # 官方源
    "https://modelscope.cn/models",  # 阿里 ModelScope (dashscope)
]

# ModelScope 模型映射 (HF模型ID -> ModelScope模型ID)
MODELSCOPE_MAPPING = {
    "BAAI/bge-m3": "BAAI/bge-m3",
}


def download_with_huggingface_hub(
    model_id: str,
    save_dir: Path,
    max_retries: int = 3,
    use_mirror: bool = True,
) -> bool:
    """使用 huggingface_hub 下载模型"""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (
            HfHubHTTPError,
            RepositoryNotFoundError,
            EntryNotFoundError,
        )
    except ImportError:
        print("正在安装 huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (
            HfHubHTTPError,
            RepositoryNotFoundError,
            EntryNotFoundError,
        )

    mirrors = HF_MIRRORS if use_mirror else [None]

    for mirror in mirrors:
        mirror_name = mirror if mirror else "官方源"
        print(f"\n尝试使用镜像源: {mirror_name}")

        # 设置环境变量
        if mirror:
            os.environ["HF_ENDPOINT"] = mirror
        elif "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]

        for attempt in range(max_retries):
            try:
                print(f"下载尝试 {attempt + 1}/{max_retries}...")

                local_dir = snapshot_download(
                    repo_id=model_id,
                    local_dir=save_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,  # 支持断点续传
                )

                print(f"\n模型下载成功!")
                print(f"保存路径: {local_dir}")
                return True

            except RepositoryNotFoundError:
                print(f"错误: 模型仓库 '{model_id}' 不存在")
                return False

            except EntryNotFoundError as e:
                print(f"错误: 文件不存在 - {e}")
                return False

            except HfHubHTTPError as e:
                print(f"HTTP 错误: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

            except Exception as e:
                print(f"下载错误: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

        print(f"镜像源 {mirror_name} 失败，尝试下一个...")

    return False


def download_with_modelscope(
    model_id: str,
    save_dir: Path,
    max_retries: int = 3,
) -> bool:
    """使用 ModelScope (阿里云/dashscope) 下载模型"""
    try:
        from modelscope import snapshot_download as ms_snapshot_download
    except ImportError:
        print("正在安装 modelscope...")
        os.system(f"{sys.executable} -m pip install modelscope -q")
        try:
            from modelscope import snapshot_download as ms_snapshot_download
        except ImportError:
            print("modelscope 安装失败")
            return False

    # 获取 ModelScope 对应的模型 ID
    ms_model_id = MODELSCOPE_MAPPING.get(model_id, model_id)
    print(f"ModelScope 模型 ID: {ms_model_id}")

    for attempt in range(max_retries):
        try:
            print(f"\n使用 ModelScope 下载，尝试 {attempt + 1}/{max_retries}...")

            local_dir = ms_snapshot_download(
                model_id=ms_model_id,
                local_dir=str(save_dir),
            )

            print(f"\n模型下载成功!")
            print(f"保存路径: {local_dir}")
            return True

        except Exception as e:
            print(f"错误: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

    return False


def verify_download(save_dir: Path) -> bool:
    """验证模型文件是否完整"""
    required_files = [
        "config.json",
        "tokenizer.json",
    ]

    # 检查必需文件
    for file in required_files:
        if not (save_dir / file).exists():
            print(f"缺少文件: {file}")
            return False

    # 检查模型权重文件 (可能是 .bin 或 .safetensors)
    has_weights = any(
        (save_dir / f).exists()
        for f in save_dir.iterdir()
        if f.suffix in [".bin", ".safetensors"]
    )

    if not has_weights:
        print("缺少模型权重文件 (.bin 或 .safetensors)")
        return False

    print("模型文件验证通过!")
    return True


def main():
    parser = argparse.ArgumentParser(description="下载 Hugging Face 模型")
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"模型 ID (默认: {MODEL_ID})",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help=f"保存目录 (默认: {MODEL_BASE_DIR}/{{model_id}})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="最大重试次数 (默认: 3)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="不使用镜像源",
    )
    parser.add_argument(
        "--use-modelscope",
        action="store_true",
        help="使用 ModelScope (阿里云/dashscope) 下载",
    )

    args = parser.parse_args()

    # 如果没有指定保存目录，使用 model/{model_id}
    # 将 model_id 中的 / 替换为 -- (与 Hugging Face 缓存命名一致)
    if args.save_dir is None:
        safe_model_name = args.model.replace("/", "--")
        args.save_dir = MODEL_BASE_DIR / safe_model_name

    print("=" * 50)
    print("Hugging Face 模型下载器")
    print("=" * 50)
    print(f"模型: {args.model}")
    print(f"保存路径: {args.save_dir}")
    print(f"最大重试次数: {args.retries}")
    print("=" * 50)

    # 创建保存目录
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # 下载模型
    success = False

    if args.use_modelscope:
        # 直接使用 ModelScope
        success = download_with_modelscope(
            args.model,
            args.save_dir,
            args.retries,
        )
    else:
        success = download_with_huggingface_hub(
            args.model,
            args.save_dir,
            args.retries,
            use_mirror=not args.no_mirror,
        )

        # 如果 huggingface_hub 失败，尝试 ModelScope
        if not success:
            print("\n\nhuggingface_hub 下载失败，尝试使用 ModelScope (阿里云)...")
            success = download_with_modelscope(
                args.model,
                args.save_dir,
                args.retries,
            )

    if success:
        # 验证下载
        if verify_download(args.save_dir):
            print("\n下载完成!")
            print(f"模型保存在: {args.save_dir}")

            # 列出文件
            print("\n文件列表:")
            for f in sorted(args.save_dir.iterdir()):
                size = f.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                print(f"  {f.name}: {size_str}")

            return 0
        else:
            print("\n警告: 模型文件可能不完整，请重新下载")
            return 1
    else:
        print("\n下载失败!")
        print("建议:")
        print("  1. 检查网络连接")
        print("  2. 尝试使用 VPN")
        print("  3. 手动下载: https://huggingface.co/BAAI/bge-m3")
        return 1


if __name__ == "__main__":
    sys.exit(main())
