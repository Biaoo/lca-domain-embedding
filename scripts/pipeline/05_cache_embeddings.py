"""
Cache embeddings for multiple models (local SentenceTransformers or OpenRouter)
so they can be reused across evaluation runs.

Usage examples:

  # Local + OpenRouter models
  python scripts/cache_embeddings.py \
    --data_dir data/ft_data \
    --output_dir data/eval_cache \
    --model raw=local:data/model/Qwen--Qwen3-Embedding-0.6B \
    --model qwen8b=openrouter:qwen/qwen3-embedding-8b \
    --batch_size 64

  # Local only, store under alias "bge"
  python scripts/cache_embeddings.py \
    --model bge=local:data/model/BAAI--bge-m3

Model spec format:
  alias=provider:value
    - provider: "local" for SentenceTransformer checkpoints
                "openrouter" for OpenRouter embeddings endpoint
    - value:    local path or model id for the provider

Embeddings are stored under {output_dir}/{alias}/ with npy + id jsons + meta.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from src.embed.cache import (
    ModelSpec,
    embed_local,
    embed_openrouter,
    load_eval_texts,
    normalize_embeddings,
    parse_model_spec,
    save_embeddings,
)


DEFAULT_DATA_DIR = "data/ft_data"
DEFAULT_OUTPUT_DIR = "data/eval_cache"
DEFAULT_BATCH_SIZE = 64
DEFAULT_DEVICE = "cuda:0"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"


def process_model(
    spec: ModelSpec,
    queries: List[Tuple[str, str]],
    corpus: List[Tuple[str, str]],
    args: argparse.Namespace,
    output_dir: Path,
):
    print(f"\n{'='*60}\nEncoding model: {spec.alias} ({spec.provider})\n{'='*60}")
    texts_queries = [text for _, text in queries]
    texts_corpus = [text for _, text in corpus]

    if spec.provider == "local":
        q_emb = embed_local(
            spec.value, texts_queries, args.batch_size, args.device, args.normalize
        )
        c_emb = embed_local(
            spec.value, texts_corpus, args.batch_size, args.device, args.normalize
        )
    else:
        q_emb = embed_openrouter(
            spec.value,
            texts_queries,
            args.batch_size,
            api_key=args.openrouter_api_key,
            url=args.openrouter_url,
            referrer=args.openrouter_site,
            site_title=args.openrouter_title,
            timeout=args.request_timeout,
            normalize=args.normalize,
            retries=args.request_retries,
            retry_backoff=args.retry_backoff,
        )
        c_emb = embed_openrouter(
            spec.value,
            texts_corpus,
            args.batch_size,
            api_key=args.openrouter_api_key,
            url=args.openrouter_url,
            referrer=args.openrouter_site,
            site_title=args.openrouter_title,
            timeout=args.request_timeout,
            normalize=args.normalize,
            retries=args.request_retries,
            retry_backoff=args.retry_backoff,
        )

    if q_emb.shape[1] != c_emb.shape[1]:
        raise ValueError(
            f"Embedding dims mismatch for {spec.alias}: queries {q_emb.shape} vs corpus {c_emb.shape}"
        )

    save_embeddings(
        spec,
        output_dir,
        queries,
        corpus,
        q_emb,
        c_emb,
        args.normalize,
        args.data_dir,
        args.batch_size,
        args.device,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cache embeddings for multiple models to reuse during evaluation"
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec alias=provider:value. Repeat for multiple models.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing test_queries.jsonl and corpus.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write cached embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for encoding or OpenRouter requests",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device for local SentenceTransformer models",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2 normalize embeddings (recommended)",
    )
    parser.add_argument(
        "--openrouter_api_key",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env)",
    )
    parser.add_argument(
        "--openrouter_url",
        type=str,
        default=DEFAULT_OPENROUTER_URL,
        help="OpenRouter embeddings endpoint URL",
    )
    parser.add_argument(
        "--openrouter_site",
        type=str,
        default="",
        help="Optional site URL for OpenRouter rankings (HTTP-Referer header)",
    )
    parser.add_argument(
        "--openrouter_title",
        type=str,
        default="",
        help="Optional site title for OpenRouter rankings (X-Title header)",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=60,
        help="Timeout seconds for OpenRouter requests",
    )
    parser.add_argument(
        "--request_retries",
        type=int,
        default=3,
        help="Retries for OpenRouter requests",
    )
    parser.add_argument(
        "--retry_backoff",
        type=float,
        default=2.0,
        help="Base backoff seconds between retries (multiplied by attempt)",
    )

    args = parser.parse_args()
    if args.openrouter_api_key is None:
        args.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

    model_specs = [parse_model_spec(spec) for spec in args.model]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries, corpus = load_eval_texts(args.data_dir)

    for spec in model_specs:
        process_model(spec, queries, corpus, args, output_dir)

    print(f"\nAll embeddings cached to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
