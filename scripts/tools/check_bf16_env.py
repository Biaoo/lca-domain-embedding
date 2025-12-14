#!/usr/bin/env python3
"""
Quick probe for bf16 readiness in the current training environment.

It reports torch/CUDA versions, inspects every visible GPU, and optionally
runs a tiny bf16 matmul per device to verify that kernels execute without
raising runtime errors. Use it before launching fine-tuning to confirm that
bf16 checkpoints will stay in bf16 precision.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the current environment supports bf16 training."
    )
    parser.add_argument(
        "--skip-kernel",
        action="store_true",
        help="Skip running an actual bf16 matmul (uses metadata only).",
    )
    return parser.parse_args()


def torch_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {
        "torch": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", "unknown") or "not built with CUDA",
    }
    driver = None
    if torch.cuda.is_available():
        try:
            driver = torch.cuda.driver_version()
        except Exception:
            driver = None
    versions["cuda_driver"] = str(driver) if driver is not None else "unavailable"
    return versions


def check_metadata_support(device_id: int) -> bool:
    """
    Use torch's built-in helpers (or fall back to capability >= 8) to infer bf16 support.
    """
    with torch.cuda.device(device_id):
        if hasattr(torch.cuda, "is_bf16_supported"):
            try:
                return bool(torch.cuda.is_bf16_supported())
            except TypeError:
                # Older PyTorch expected an optional device index.
                return bool(torch.cuda.is_bf16_supported(torch.cuda.current_device()))
        major, _ = torch.cuda.get_device_capability(device_id)
        return major >= 8


def run_bf16_kernel(device_id: int) -> Tuple[bool, str]:
    """
    Launch a small bf16 matmul to ensure kernels actually execute.
    """
    try:
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
            a = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
            b = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
            _ = torch.matmul(a, b)
        return True, ""
    except Exception as exc:  # pragma: no cover - best effort diagnostic
        return False, str(exc)


def describe_device(device_id: int, skip_kernel: bool) -> Dict[str, object]:
    props = torch.cuda.get_device_properties(device_id)
    name = torch.cuda.get_device_name(device_id)
    total_mem_gb = props.total_memory / (1024**3)

    metadata_ok = check_metadata_support(device_id)
    kernel_ok = None
    kernel_error = ""
    if not skip_kernel:
        kernel_ok, kernel_error = run_bf16_kernel(device_id)

    report: Dict[str, object] = {
        "id": device_id,
        "name": name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(total_mem_gb, 2),
        "metadata_support": metadata_ok,
    }
    if skip_kernel:
        report["kernel_check"] = "skipped"
    else:
        report["kernel_support"] = kernel_ok
        if not kernel_ok:
            report["kernel_error"] = kernel_error
    return report


def main() -> None:
    args = parse_args()

    versions = torch_versions()
    print("=== Environment ====================================================")
    print(f"PyTorch:      {versions['torch']}")
    print(f"CUDA runtime: {versions['cuda_runtime']}")
    print(f"CUDA driver:  {versions['cuda_driver']}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print("====================================================================\n")

    if not torch.cuda.is_available():
        print("CUDA is not available; bf16 training on GPU is not possible here.")
        sys.exit(0)

    device_count = torch.cuda.device_count()
    print(f"Detected {device_count} CUDA device(s).\n")

    all_metadata_ok = True
    all_kernel_ok = True
    for idx in range(device_count):
        info = describe_device(idx, skip_kernel=args.skip_kernel)
        print(f"[cuda:{info['id']}] {info['name']} (capability {info['capability']})")
        print(f"  Memory:            {info['total_memory_gb']} GiB")
        print(f"  Metadata bf16 ok:  {info['metadata_support']}")
        if args.skip_kernel:
            print("  Kernel test:       skipped")
        else:
            print(f"  Kernel bf16 ok:    {info['kernel_support']}")
            if not info["kernel_support"]:
                print(f"    Error: {info['kernel_error']}")
        print("")

        all_metadata_ok = all_metadata_ok and bool(info["metadata_support"])
        if not args.skip_kernel:
            all_kernel_ok = all_kernel_ok and bool(info["kernel_support"])

    if all_metadata_ok and (args.skip_kernel or all_kernel_ok):
        print("bf16 looks good on all visible GPUs ✅")
    else:
        print("bf16 is NOT fully available on every GPU ❌")
        print("Investigate driver/runtime versions above or run with different devices.")


if __name__ == "__main__":
    main()
