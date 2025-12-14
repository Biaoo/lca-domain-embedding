#!/usr/bin/env python3
"""
Evaluate cached embeddings for multiple models (baseline or fine-tuned).

This script loads precomputed embeddings (produced by cache_embeddings.py)
and computes IR metrics across all requested models without re-encoding text.

Cache layout per model alias:
  {cache_dir}/{alias}/
    - queries_embeddings.npy
    - corpus_embeddings.npy
    - queries_ids.json
    - corpus_ids.json
    - meta.json (optional)

Examples:
  # Evaluate all cached models under data/eval_cache
  python scripts/evaluate_cached_embeddings.py --cache_dir data/eval_cache

  # Evaluate selected aliases and save results
  python scripts/evaluate_cached_embeddings.py \
    --cache_dir data/eval_cache \
    --data_dir data/ft_data \
    --model raw qwen8b bge \
    --output_path data/output/cached_eval.json
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import faiss  # type: ignore
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from src.eval.metrics import (
    evaluate_metrics,
    evaluate_mrr,
    format_results,
    maybe_normalize,
)

# Defaults
DEFAULT_DATA_DIR = "data/ft_data"
DEFAULT_CACHE_DIR = "data/eval_cache"
DEFAULT_K_VALUES = [1, 5, 10, 50, 100]
DEFAULT_TOP_K = 100
SEARCH_BATCH_SIZE = 32


@dataclass
class CachedEmbeddings:
    alias: str
    queries_ids: List[str]
    corpus_ids: List[str]
    queries_emb: np.ndarray
    corpus_emb: np.ndarray
    meta: Dict


def load_qrels(data_dir: str) -> Dict[str, Dict[str, int]]:
    """Load test_qrels.jsonl into a dict[qid][docid] = relevance."""
    qrels_ds = load_dataset("json", data_files=f"{data_dir}/test_qrels.jsonl")["train"]
    qrels: Dict[str, Dict[str, int]] = {}
    for item in qrels_ds:
        qid = item["qid"]
        docid = item["docid"]
        rel = item.get("relevance", 1)
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][docid] = rel
    print(f"Loaded qrels: {len(qrels)} queries with relevance labels")
    return qrels


def load_cached_embeddings(model_dir: Path) -> CachedEmbeddings:
    """Load embeddings and metadata from a cache directory."""
    alias = model_dir.name
    queries_emb = np.load(model_dir / "queries_embeddings.npy")
    corpus_emb = np.load(model_dir / "corpus_embeddings.npy")
    queries_ids = json.loads((model_dir / "queries_ids.json").read_text(encoding="utf-8"))
    corpus_ids = json.loads((model_dir / "corpus_ids.json").read_text(encoding="utf-8"))
    meta_path = model_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.setdefault("model", {})  # keep consistent shape
    # Basic sanity checks
    if len(queries_ids) != queries_emb.shape[0]:
        raise ValueError(f"{alias}: queries ids count != embeddings rows")
    if len(corpus_ids) != corpus_emb.shape[0]:
        raise ValueError(f"{alias}: corpus ids count != embeddings rows")
    if queries_emb.shape[1] != corpus_emb.shape[1]:
        raise ValueError(f"{alias}: embedding dim mismatch between queries and corpus")
    return CachedEmbeddings(
        alias=alias,
        queries_ids=queries_ids,
        corpus_ids=corpus_ids,
        queries_emb=queries_emb,
        corpus_emb=corpus_emb,
        meta=meta,
    )


def search_embeddings(
    cached: CachedEmbeddings,
    top_k: int,
    qrels: Dict[str, Dict[str, int]],
    force_normalize: bool,
) -> Dict[str, Dict[str, float]]:
    """Search corpus for each query embedding using Faiss (inner product)."""
    corpus = maybe_normalize(cached.corpus_emb.astype(np.float32), force_normalize)
    queries = maybe_normalize(cached.queries_emb.astype(np.float32), force_normalize)
    dim = corpus.shape[1]

    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.train(corpus)
    index.add(corpus)

    results: Dict[str, Dict[str, float]] = {}
    qids = cached.queries_ids
    doc_ids = cached.corpus_ids

    for start in tqdm(range(0, len(queries), SEARCH_BATCH_SIZE), desc=f"Searching {cached.alias}"):
        end = min(start + SEARCH_BATCH_SIZE, len(queries))
        scores, indices = index.search(queries[start:end], k=top_k)
        for offset, (score_row, idx_row) in enumerate(zip(scores, indices)):
            qid = qids[start + offset]
            if qid not in qrels:
                continue  # skip queries without relevance labels
            results[qid] = {}
            for s, idx in zip(score_row, idx_row):
                if idx == -1:
                    continue
                results[qid][doc_ids[int(idx)]] = float(s)
    return results



def detect_models(cache_dir: Path) -> List[Path]:
    """Find subdirectories that look like cached model folders."""
    candidates = []
    for child in cache_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "queries_embeddings.npy").exists() and (child / "corpus_embeddings.npy").exists():
            candidates.append(child)
    return sorted(candidates)


def print_metrics(alias: str, metrics: Dict):
    print(f"\n{'='*60}")
    print(f"Model: {alias}")
    print("=" * 60)
    for name in ["ndcg", "map", "recall", "precision", "mrr"]:
        if name in metrics:
            print(metrics[name])


def print_comparison(rows: List[Dict[str, float]]):
    if not rows:
        return
    print(f"\n{'='*60}")
    print("Comparison (key metrics)")
    print("=" * 60)
    print(f"\n{'Model':<15} {'NDCG@10':>10} {'Recall@10':>12} {'MRR@10':>10} {'MAP@10':>10}")
    print("-" * 60)
    for row in rows:
        print(
            f"{row['alias']:<15} {row['NDCG@10']:>10.4f} {row['Recall@10']:>12.4f} {row['MRR@10']:>10.4f} {row['MAP@10']:>10.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cached embeddings from cache_embeddings.py across multiple models"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing cached embeddings per model alias",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing test_qrels.jsonl (and test_queries/corpus for sanity)",
    )
    parser.add_argument(
        "--model",
        nargs="*",
        help="Aliases to evaluate; default is all cached models in cache_dir",
    )
    parser.add_argument(
        "--k_values",
        nargs="*",
        type=int,
        default=DEFAULT_K_VALUES,
        help="List of cutoff values for metrics",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Retrieve top K documents per query for evaluation",
    )
    parser.add_argument(
        "--force_normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="L2 normalize cached embeddings before search (use if caches were not normalized)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Optional path to save evaluation results as JSON",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache_dir not found: {cache_dir}")

    if args.model:
        model_dirs = [cache_dir / alias for alias in args.model]
    else:
        model_dirs = detect_models(cache_dir)

    if not model_dirs:
        raise RuntimeError(f"No cached models found under {cache_dir}")

    qrels = load_qrels(args.data_dir)
    all_results = []
    summary_rows: List[Dict[str, float]] = []

    for model_dir in model_dirs:
        cached = load_cached_embeddings(model_dir)
        missing_qids = sum(1 for q in cached.queries_ids if q not in qrels)
        if missing_qids:
            print(f"  Warning: {cached.alias} has {missing_qids} queries without qrels; they will be skipped.")

        results = search_embeddings(
            cached,
            top_k=args.top_k,
            qrels=qrels,
            force_normalize=args.force_normalize,
        )
        metrics = evaluate_metrics(qrels, results, args.k_values)
        metrics["mrr"] = evaluate_mrr(qrels, results, args.k_values)

        print_metrics(cached.alias, metrics)
        summary_rows.append(format_results(cached.alias, metrics))

        all_results.append(
            {
                "alias": cached.alias,
                "meta": cached.meta,
                "metrics": metrics,
            }
        )

    print_comparison(summary_rows)

    if args.output_path:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "settings": {
                "cache_dir": str(cache_dir),
                "data_dir": args.data_dir,
                "k_values": args.k_values,
                "top_k": args.top_k,
                "force_normalize": args.force_normalize,
                "models": [str(d.name) for d in model_dirs],
            },
            "results": all_results,
        }
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved evaluation results to {output_path.resolve()}")

    print(f"\n{'='*60}\nEvaluation completed\n{'='*60}")


if __name__ == "__main__":
    main()
