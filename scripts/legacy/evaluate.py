"""
Evaluate Fine-tuned LCA Embedding Model (BGE-M3-Embedding)

This script evaluates both the original and fine-tuned models
on the test dataset and compares their performance.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --finetuned_path data/output/lca-bge-m3-finetuned
    python scripts/evaluate.py --raw_model data/model/BAAI--bge-m3 --finetuned_path data/output/lca-bge-m3-finetuned
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import faiss # type: ignore
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DEFAULT_RAW_MODEL = "data/model/BAAI--bge-m3"
DEFAULT_FINETUNED_PATH = "data/output/lca-bge-m3-finetuned"
DEFAULT_DATA_DIR = "data/ft_data"

# Evaluation parameters
K_VALUES = [1, 5, 10, 50, 100]
SEARCH_BATCH_SIZE = 32
TOP_K = 100
DEFAULT_ENCODE_BATCH_SIZE = 128
DEFAULT_TARGET_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PREFERRED_DTYPE = torch.float16
FALLBACK_DTYPE = torch.float32
DEFAULT_OUTPUT_PATH = "data/output/eval_results.json"


def load_data(data_dir: str):
    """Load test queries, corpus, and qrels from data directory."""
    print(f"\nLoading data from {data_dir}/...")

    queries = load_dataset("json", data_files=f"{data_dir}/test_queries.jsonl")["train"]
    corpus = load_dataset("json", data_files=f"{data_dir}/corpus.jsonl")["train"]
    qrels = load_dataset("json", data_files=f"{data_dir}/test_qrels.jsonl")["train"]

    # Extract text
    queries_text = queries["text"]

    # corpus["text"] is a list of lists, flatten it
    corpus_text = []
    for text in corpus["text"]:
        if isinstance(text, list):
            corpus_text.extend(text)
        else:
            corpus_text.append(text)

    # Build qrels dict
    qrels_dict = {}
    for line in qrels:
        if line["qid"] not in qrels_dict:
            qrels_dict[line["qid"]] = {}
        qrels_dict[line["qid"]][line["docid"]] = line["relevance"]

    print(f"  - Queries: {len(queries_text)}")
    print(f"  - Corpus:  {len(corpus_text)}")
    print(f"  - Qrels:   {len(qrels_dict)}")

    return queries, queries_text, corpus, corpus_text, qrels_dict


def save_results(payload: dict, output_path: str):
    """Persist evaluation payload to disk as JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_file.resolve()}")


def search(model, queries_text, corpus_text, queries, corpus, encode_batch_size: int):
    """Encode queries and corpus, then search using Faiss."""
    print("\nEncoding and searching...")
    cuda_flag = torch.cuda.is_available()
    device_name = str(model.device)
    print(
        f"  Device check -> torch.cuda.is_available(): {cuda_flag}, model.device: {device_name}"
    )
    print(f"  Using encode batch size: {encode_batch_size}")

    # Encode queries
    print("  Encoding queries...")
    queries_embeddings = model.encode(
        queries_text,
        batch_size=encode_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Encode corpus
    print("  Encoding corpus...")
    corpus_embeddings = model.encode(
        corpus_text,
        batch_size=encode_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Create Faiss index
    dim = corpus_embeddings.shape[-1]
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    index.train(corpus_embeddings)
    index.add(corpus_embeddings)

    # Search
    query_size = len(queries_embeddings)
    all_scores = []
    all_indices = []

    for i in tqdm(range(0, query_size, SEARCH_BATCH_SIZE), desc="Searching"):
        j = min(i + SEARCH_BATCH_SIZE, query_size)
        query_embedding = queries_embeddings[i:j]
        score, indice = index.search(query_embedding.astype(np.float32), k=TOP_K)
        all_scores.append(score)
        all_indices.append(indice)

    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)

    # Format results
    results = {}
    for idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
        results[queries["id"][idx]] = {}
        for score, index in zip(scores, indices):
            if index != -1:
                results[queries["id"][idx]][corpus["id"][index]] = float(score)

    return results


def evaluate_metrics(qrels: dict, results: dict, k_values: list) -> list:
    """Evaluate NDCG, MAP, Recall, Precision metrics."""
    import pytrec_eval 

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )

    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

    return [ndcg, _map, recall, precision]


def evaluate_mrr(qrels: dict, results: dict, k_values: list) -> dict:
    """Evaluate MRR (Mean Reciprocal Rank)."""
    mrr = {}
    for k in k_values:
        mrr[f"MRR@{k}"] = 0.0

    for query_id, doc_scores in results.items():
        if query_id not in qrels:
            continue
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        for k in k_values:
            for rank, (doc_id, _) in enumerate(sorted_docs[:k], 1):
                if doc_id in qrels[query_id] and qrels[query_id][doc_id] > 0:
                    mrr[f"MRR@{k}"] += 1.0 / rank
                    break

    for k in k_values:
        mrr[f"MRR@{k}"] = round(mrr[f"MRR@{k}"] / len(qrels), 5)

    return mrr


def evaluate_model(
    model_path: str,
    queries_text,
    corpus_text,
    queries,
    corpus,
    qrels_dict,
    encode_batch_size: int,
    device: str,
    model_name: str = "Model",
):
    """Evaluate a single model and print results."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Path: {model_path}")
    print("=" * 60)

    # Load model using SentenceTransformer
    print("\nLoading model...")
    primary_device = device
    fp16_supported = primary_device.startswith("cuda") and torch.cuda.is_available()
    torch_dtype = PREFERRED_DTYPE if fp16_supported else FALLBACK_DTYPE
    model = SentenceTransformer(
        model_path,
        trust_remote_code=True,
        device=primary_device,
        model_kwargs={"torch_dtype": torch_dtype},
    )
    if primary_device.startswith("cuda") and not torch.cuda.is_available():
        print("  WARNING: CUDA specified but not detected, falling back to CPU.")
    if torch_dtype == PREFERRED_DTYPE:
        print("  Loaded model weights in float16 precision.")
    else:
        print("  Float16 not available for selected devices; using float32.")

    print(f"  Using single device: {primary_device}")

    # Search
    results = search(
        model,
        queries_text,
        corpus_text,
        queries,
        corpus,
        encode_batch_size,
    )

    # Evaluate
    eval_res = evaluate_metrics(qrels_dict, results, K_VALUES)
    mrr = evaluate_mrr(qrels_dict, results, K_VALUES)

    # Print results
    print(f"\n{'-'*60}")
    print("Results:")
    print("-" * 60)
    for res in eval_res:
        print(res)
    print(mrr)

    # Return metrics for comparison
    return {"eval_res": eval_res, "mrr": mrr}


def print_comparison(raw_metrics, ft_metrics):
    """Print comparison table between raw and fine-tuned models and return structured rows."""
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'Raw Model':>15} {'Fine-tuned':>15} {'Improvement':>15}")
    print("-" * 65)

    # Extract key metrics for comparison
    raw_ndcg10 = raw_metrics["eval_res"][0].get("NDCG@10", 0)
    ft_ndcg10 = ft_metrics["eval_res"][0].get("NDCG@10", 0)

    raw_recall10 = raw_metrics["eval_res"][2].get("Recall@10", 0)
    ft_recall10 = ft_metrics["eval_res"][2].get("Recall@10", 0)

    raw_mrr10 = raw_metrics["mrr"].get("MRR@10", 0)
    ft_mrr10 = ft_metrics["mrr"].get("MRR@10", 0)

    metrics = [
        ("NDCG@10", raw_ndcg10, ft_ndcg10),
        ("Recall@10", raw_recall10, ft_recall10),
        ("MRR@10", raw_mrr10, ft_mrr10),
    ]

    comparison_rows = []
    for name, raw, ft in metrics:
        improvement = ((ft - raw) / raw * 100) if raw > 0 else 0
        print(f"{name:<20} {raw:>15.4f} {ft:>15.4f} {improvement:>14.1f}%")
        comparison_rows.append(
            {
                "metric": name,
                "raw": raw,
                "finetuned": ft,
                "improvement_pct": round(improvement, 2) if raw > 0 else 0.0,
            }
        )

    return comparison_rows


def main():
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
        "--encode_batch_size",
        type=int,
        default=DEFAULT_ENCODE_BATCH_SIZE,
        help="Batch size for encoding queries/corpus (higher is faster but uses more VRAM)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_TARGET_DEVICE,
        help="Device to run encoding on (e.g., 'cuda:0' or 'cpu')",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save evaluation results as JSON (leave empty to skip saving)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LCA Embedding Model Evaluation (Qwen3-Embedding)")
    print("=" * 60)

    # Load data
    queries, queries_text, corpus, corpus_text, qrels_dict = load_data(args.data_dir)
    dataset_stats = {
        "queries": len(queries_text),
        "corpus": len(corpus_text),
        "qrels": len(qrels_dict),
    }

    raw_metrics = None
    ft_metrics = None

    # Evaluate raw model
    if args.eval_raw:
        raw_metrics = evaluate_model(
            args.raw_model,
            queries_text,
            corpus_text,
            queries,
            corpus,
            qrels_dict,
            encode_batch_size=args.encode_batch_size,
            device=args.device,
            model_name="Raw Model",
        )

    # Evaluate fine-tuned model
    ft_metrics = evaluate_model(
        args.finetuned_path,
        queries_text,
        corpus_text,
        queries,
        corpus,
        qrels_dict,
        encode_batch_size=args.encode_batch_size,
        device=args.device,
        model_name="Fine-tuned Model",
    )

    # Print comparison if both models evaluated
    comparison_rows = []
    if raw_metrics and ft_metrics:
        comparison_rows = print_comparison(raw_metrics, ft_metrics)

    if args.output_path:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "settings": {
                "raw_model": args.raw_model,
                "finetuned_model": args.finetuned_path,
                "data_dir": args.data_dir,
                "encode_batch_size": args.encode_batch_size,
                "device": args.device,
                "eval_raw": args.eval_raw,
                "k_values": K_VALUES,
            },
            "dataset_stats": dataset_stats,
        }
        if raw_metrics:
            payload["raw_metrics"] = raw_metrics
        if ft_metrics:
            payload["finetuned_metrics"] = ft_metrics
        if comparison_rows:
            payload["comparison"] = comparison_rows
        save_results(payload, args.output_path)
    else:
        print("\nSkipping saving results (no --output_path provided).")

    print(f"\n{'='*60}")
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
