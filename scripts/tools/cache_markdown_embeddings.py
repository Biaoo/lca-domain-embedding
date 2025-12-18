#!/usr/bin/env python3
"""
Pre-compute and cache embeddings for a directory of markdown documents, so Embedding Atlas
can later load a dataset with pre-computed vectors (via --vector) without re-embedding.

Designed for:
  data/supabase_exports_markdown/{flows,processs}/*.md

Cache layout:
  {output_dir}/{alias}/
    - embeddings.npy      (float32, shape [N, D])
    - ids.json            (list[str], length N; id == file stem)
    - meta.json           (run metadata)

Example:
  .venv/bin/python scripts/tools/cache_markdown_embeddings.py \
    --input_dir data/supabase_exports_markdown \
    --subset both \
    --text_mode plain \
    --model ft=local:data/output/lca-qwen3-embedding \
    --device cuda:0 \
    --output_dir data/embed_cache/supabase_markdown \
    --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from src.embed.cache import ModelSpec, normalize_embeddings, parse_model_spec
except ModuleNotFoundError:
    # Some environments may have a different `src` module installed; fall back to importing
    # from the source root directly.
    from embed.cache import ModelSpec, normalize_embeddings, parse_model_spec


DEFAULT_INPUT_DIR = "data/supabase_exports_markdown"
DEFAULT_OUTPUT_DIR = "data/embed_cache/supabase_markdown"
DEFAULT_BATCH_SIZE = 64
DEFAULT_DEVICE = "cpu"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"

FIELD_RE = re.compile(r"^\*\*(.+?):\*\*\s*(.+?)\s*$")


def _iter_markdown_files(input_dir: Path, subset: str) -> List[Path]:
    roots: List[Path] = []
    if subset in {"flows", "both"}:
        roots.append(input_dir / "flows")
    if subset in {"processs", "both"}:
        roots.append(input_dir / "processs")
    if not roots:
        roots = [input_dir]

    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("*.md")))
    return files


def _read_text_capped(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if max_chars <= 0:
            return f.read()
        return f.read(max_chars)


def _strip_markdown(text: str) -> str:
    # Remove fenced code blocks
    out_lines: List[str] = []
    in_fence = False
    for line in text.splitlines():
        s = line.rstrip("\n")
        if s.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        out_lines.append(s)
    text = "\n".join(out_lines)

    text = text.replace("`", "")
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _get_text(path: Path, text_mode: str, max_chars: int) -> str:
    raw = _read_text_capped(path, max_chars=max_chars)
    return raw if text_mode == "markdown" else _strip_markdown(raw)


def _encode_local(
    model: "SentenceTransformer",
    texts: List[str],
    *,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    if torch is not None:
        with torch.inference_mode():
            emb = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )
    else:
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
    return np.asarray(emb, dtype=np.float32)


def _encode_openrouter_batch(
    model_id: str,
    texts: List[str],
    *,
    url: str,
    api_key: str,
    timeout: int,
    retries: int,
    retry_backoff: float,
    normalize: bool,
) -> np.ndarray:
    if not api_key:
        raise ValueError("OpenRouter API key is required for openrouter provider")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "input": texts,
        "encoding_format": "float",
    }
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            rows = sorted(data["data"], key=lambda x: x.get("index", 0))
            emb = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
            if normalize:
                emb = normalize_embeddings(emb)
            return emb
        except Exception as e:
            last_err = e
            if attempt == retries:
                raise
            time.sleep(retry_backoff * attempt)
    assert last_err is not None
    raise last_err


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cache embeddings for a markdown directory (for Embedding Atlas --vector)"
    )
    p.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    p.add_argument(
        "--subset",
        choices=["flows", "processs", "both"],
        default="both",
        help="Which subfolders to include under input_dir.",
    )
    p.add_argument(
        "--text_mode",
        choices=["markdown", "plain"],
        default="plain",
        help="Use raw markdown or stripped plain text for embedding.",
    )
    p.add_argument("--max_chars", type=int, default=20000, help="0 means no cap.")
    p.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    p.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec alias=provider:value. provider: local/openrouter. Repeatable.",
    )
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2 normalize embeddings (recommended).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing caches.")
    p.add_argument(
        "--openrouter_api_key",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env).",
    )
    p.add_argument(
        "--openrouter_url",
        type=str,
        default=DEFAULT_OPENROUTER_URL,
        help="OpenRouter embeddings endpoint URL.",
    )
    p.add_argument("--request_timeout", type=int, default=60)
    p.add_argument("--request_retries", type=int, default=3)
    p.add_argument("--retry_backoff", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    if args.openrouter_api_key is None:
        args.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

    paths = _iter_markdown_files(input_dir, subset=args.subset)
    if not paths:
        raise RuntimeError(f"No markdown files found under {input_dir} (subset={args.subset})")
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    ids = [p.stem for p in paths]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [parse_model_spec(s) for s in args.model]

    for spec in specs:
        model_dir = output_dir / spec.alias
        model_dir.mkdir(parents=True, exist_ok=True)
        emb_path = model_dir / "embeddings.npy"
        ids_path = model_dir / "ids.json"
        meta_path = model_dir / "meta.json"

        if (emb_path.exists() or ids_path.exists()) and not args.overwrite:
            raise FileExistsError(
                f"Cache already exists for alias '{spec.alias}' at {model_dir}. "
                f"Pass --overwrite to overwrite."
            )

        local_model = None
        if spec.provider == "local":
            from sentence_transformers import SentenceTransformer

            local_model = SentenceTransformer(
                spec.value, trust_remote_code=True, device=args.device
            )

        # Stream embeddings to a .npy file (no need to hold full matrix in RAM).
        mm = None
        dim = None
        offset = 0
        for start in tqdm(
            range(0, len(paths), args.batch_size),
            desc=f"Embedding {spec.alias}",
        ):
            batch_paths = paths[start : start + args.batch_size]
            batch_texts = [_get_text(p, text_mode=args.text_mode, max_chars=args.max_chars) for p in batch_paths]
            if spec.provider == "local":
                emb = _encode_local(
                    local_model,
                    batch_texts,
                    batch_size=args.batch_size,
                    normalize=args.normalize,
                )
            else:
                emb = _encode_openrouter_batch(
                    spec.value,
                    batch_texts,
                    url=args.openrouter_url,
                    api_key=args.openrouter_api_key,
                    timeout=args.request_timeout,
                    retries=args.request_retries,
                    retry_backoff=args.retry_backoff,
                    normalize=args.normalize,
                )

            if mm is None:
                dim = int(emb.shape[1])
                from numpy.lib.format import open_memmap

                mm = open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(len(paths), dim))

            mm[offset : offset + emb.shape[0]] = emb.astype(np.float32)
            offset += emb.shape[0]

        if mm is None or dim is None:
            raise RuntimeError("No embeddings were produced (empty input?)")
        if offset != len(paths):
            raise RuntimeError(f"Embedding count mismatch: wrote {offset}, expected {len(paths)}")

        ids_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
        meta = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(input_dir),
            "subset": args.subset,
            "text_mode": args.text_mode,
            "max_chars": args.max_chars,
            "count": len(ids),
            "embedding_dim": dim,
            "normalize": args.normalize,
            "batch_size": args.batch_size,
            "device": args.device,
            "model": {
                "alias": spec.alias,
                "provider": spec.provider,
                "value": spec.value,
            },
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved cache: {spec.alias} -> {model_dir} (N={len(ids)}, D={dim})")


if __name__ == "__main__":
    main()
