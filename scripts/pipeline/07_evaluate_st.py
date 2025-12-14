#!/usr/bin/env python3
"""
Evaluate Fine-tuned LCA Embedding Model (Qwen3-Embedding) using Sentence Transformers

This script evaluates both the original and fine-tuned models on the test dataset
using Sentence Transformers' InformationRetrievalEvaluator and compares performance.

Usage:
    python scripts/evaluate_st.py

    # Specify paths
    python scripts/evaluate_st.py --finetuned_path data/output/lca-qwen3-st-finetuned

    # Skip raw model evaluation
    python scripts/evaluate_st.py --no_eval_raw
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator


# ============================================================
# Configuration
# ============================================================

DEFAULT_RAW_MODEL = "data/model/Qwen--Qwen3-Embedding-0.6B"
DEFAULT_FINETUNED_PATH = "data/output/lca-qwen3-st-finetuned"
DEFAULT_DATA_DIR = "data/ft_data"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LCA embedding models")
    parser.add_argument(
        "--raw_model",
        type=str,
        default=DEFAULT_RAW_MODEL,
        help="Path or name of the raw/base model",
    )
    parser.add_argument(
        "--finetuned_path",
        type=str,
        default=DEFAULT_FINETUNED_PATH,
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing test data",
    )
    parser.add_argument(
        "--eval_raw",
        action="store_true",
        default=True,
        help="Evaluate raw model (default: True)",
    )
    parser.add_argument(
        "--no_eval_raw",
        action="store_false",
        dest="eval_raw",
        help="Skip evaluation of raw model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for encoding"
    )
    parser.add_argument(
        "--corpus_chunk_size",
        type=int,
        default=50000,
        help="Number of corpus documents per chunk during evaluation (lower to reduce memory)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save evaluation results as JSON",
    )
    return parser.parse_args()


def load_evaluation_data(data_dir: str):
    """
    Load test queries, corpus, and qrels from data directory.

    Returns:
        queries: dict[str, str] - Query ID to query text
        corpus: dict[str, str] - Document ID to document text
        relevant_docs: dict[str, set[str]] - Query ID to set of relevant document IDs
    """
    print(f"\nLoading data from {data_dir}/...")

    # Load datasets
    queries_ds = load_dataset("json", data_files=f"{data_dir}/test_queries.jsonl")[
        "train"
    ]
    corpus_ds = load_dataset("json", data_files=f"{data_dir}/corpus.jsonl")["train"]
    qrels_ds = load_dataset("json", data_files=f"{data_dir}/test_qrels.jsonl")["train"]

    # Convert queries to dict
    queries = dict(zip(queries_ds["id"], queries_ds["text"]))

    # Convert corpus to dict
    # Note: corpus["text"] might be a list of lists, need to flatten
    corpus = {}
    for item in corpus_ds:
        doc_id = item["id"]
        text = item["text"]
        # Handle both list and string formats
        if isinstance(text, list):
            # If multiple texts, join them or use first one
            corpus[doc_id] = text[0] if len(text) == 1 else " ".join(text)
        else:
            corpus[doc_id] = text

    # Convert qrels to dict
    relevant_docs = {}
    for item in qrels_ds:
        qid = item["qid"]
        docid = item["docid"]
        if qid not in relevant_docs:
            relevant_docs[qid] = set()
        relevant_docs[qid].add(docid)

    print(f"  - Queries: {len(queries)}")
    print(f"  - Corpus:  {len(corpus)}")
    print(f"  - Qrels:   {len(relevant_docs)} queries with relevance judgments")

    return queries, corpus, relevant_docs


def evaluate_model(
    model_path: str,
    queries: dict,
    corpus: dict,
    relevant_docs: dict,
    batch_size: int,
    corpus_chunk_size: int,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a single model using Sentence Transformers' InformationRetrievalEvaluator.

    Returns:
        dict: Evaluation results with metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Path: {model_path}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = SentenceTransformer(
        model_path,
        trust_remote_code=True,
    )
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"  Max sequence length: {model.max_seq_length}")

    # Create evaluator
    print("\nRunning evaluation...")
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=f"lca-test-{model_name.lower().replace(' ', '-')}",
        batch_size=batch_size,
        corpus_chunk_size=corpus_chunk_size,
        show_progress_bar=True,
        # Metrics to compute
        ndcg_at_k=[1, 5, 10, 50, 100],
        map_at_k=[1, 5, 10, 50, 100],
        precision_recall_at_k=[1, 5, 10, 50, 100],
        mrr_at_k=[1, 5, 10, 50, 100],
    )

    # Run evaluation
    results = evaluator(model)

    # Print results
    print(f"\n{'-'*60}")
    print("Results:")
    print("-" * 60)

    # Extract and print key metrics
    key_metrics = [
        "ndcg@10",
        "ndcg@50",
        "ndcg@100",
        "map@10",
        "map@50",
        "map@100",
        "recall@10",
        "recall@50",
        "recall@100",
        "mrr@10",
        "mrr@50",
        "mrr@100",
    ]

    for metric in key_metrics:
        # Results keys include the evaluator name prefix
        for key, value in results.items():
            if metric in key.lower():
                print(f"  {metric:<15}: {value:.5f}")
                break

    return results


def save_results(output_path: str, data: dict):
    """Persist evaluation results to JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert any non-serializable values (e.g., numpy floats) to Python floats
    def to_serializable(obj):
        try:
            return float(obj)
        except Exception:
            return obj

    serializable = {
        key: {k: to_serializable(v) for k, v in value.items()}
        if isinstance(value, dict)
        else to_serializable(value)
        for key, value in data.items()
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to {output_file}")


def print_comparison(raw_results: dict, ft_results: dict):
    """Print comparison table between raw and fine-tuned models."""
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'Raw Model':>15} {'Fine-tuned':>15} {'Improvement':>15}")
    print("-" * 65)

    # Key metrics to compare
    compare_metrics = [
        ("NDCG@10", "ndcg@10"),
        ("NDCG@100", "ndcg@100"),
        ("Recall@10", "recall@10"),
        ("Recall@100", "recall@100"),
        ("MRR@10", "mrr@10"),
        ("MAP@10", "map@10"),
    ]

    for display_name, metric_key in compare_metrics:
        raw_value = None
        ft_value = None

        # Find metric in results (key includes evaluator name prefix)
        for key, value in raw_results.items():
            if metric_key in key.lower():
                raw_value = value
                break

        for key, value in ft_results.items():
            if metric_key in key.lower():
                ft_value = value
                break

        if raw_value is not None and ft_value is not None:
            improvement = (
                ((ft_value - raw_value) / raw_value * 100) if raw_value > 0 else 0
            )
            sign = "+" if improvement > 0 else ""
            print(
                f"{display_name:<20} {raw_value:>15.4f} {ft_value:>15.4f} {sign}{improvement:>13.1f}%"
            )
        else:
            print(f"{display_name:<20} {'N/A':>15} {'N/A':>15} {'N/A':>15}")


def main():
    args = parse_args()

    print("=" * 60)
    print("LCA Embedding Model Evaluation (Sentence Transformers)")
    print("=" * 60)

    # Load evaluation data
    queries, corpus, relevant_docs = load_evaluation_data(args.data_dir)

    raw_results = None
    ft_results = None

    # Evaluate raw model
    if args.eval_raw:
        raw_results = evaluate_model(
            args.raw_model,
            queries,
            corpus,
            relevant_docs,
            args.batch_size,
            args.corpus_chunk_size,
            model_name="Raw Model",
        )

    # Evaluate fine-tuned model
    if Path(args.finetuned_path).exists():
        ft_results = evaluate_model(
            args.finetuned_path,
            queries,
            corpus,
            relevant_docs,
            args.batch_size,
            args.corpus_chunk_size,
            model_name="Fine-tuned Model",
        )
    else:
        print(f"\nWarning: Fine-tuned model not found at {args.finetuned_path}")
        print("Skipping fine-tuned model evaluation.")

    # Print comparison if both models evaluated
    if raw_results and ft_results:
        print_comparison(raw_results, ft_results)

    if args.output_path:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "raw_model": args.raw_model,
            "finetuned_path": args.finetuned_path,
            "data_dir": args.data_dir,
            "batch_size": args.batch_size,
            "corpus_chunk_size": args.corpus_chunk_size,
            "eval_raw": args.eval_raw,
            "raw_results": raw_results,
            "finetuned_results": ft_results,
        }
        save_results(args.output_path, payload)

    print(f"\n{'='*60}")
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
