"""
Evaluation utilities for retrieval tasks.
"""
from typing import Dict, List

import numpy as np


def maybe_normalize(arr: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def evaluate_metrics(qrels: Dict[str, Dict[str, int]], results: Dict[str, Dict[str, float]], k_values: List[int]) -> Dict:
    """Compute NDCG/MAP/Recall/Precision via pytrec_eval."""
    import pytrec_eval  # lazy import

    ndcg: Dict[str, float] = {}
    _map: Dict[str, float] = {}
    recall: Dict[str, float] = {}
    precision: Dict[str, float] = {}

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

    return {"ndcg": ndcg, "map": _map, "recall": recall, "precision": precision}


def evaluate_mrr(qrels: Dict[str, Dict[str, int]], results: Dict[str, Dict[str, float]], k_values: List[int]) -> Dict[str, float]:
    """Compute Mean Reciprocal Rank."""
    mrr: Dict[str, float] = {f"MRR@{k}": 0.0 for k in k_values}
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


def format_results(alias: str, metrics: Dict) -> Dict[str, float]:
    """Extract a compact subset for comparison."""
    ndcg10 = metrics["ndcg"].get("NDCG@10", 0.0)
    recall10 = metrics["recall"].get("Recall@10", 0.0)
    mrr10 = metrics["mrr"].get("MRR@10", 0.0)
    map10 = metrics["map"].get("MAP@10", 0.0)
    return {
        "alias": alias,
        "NDCG@10": ndcg10,
        "Recall@10": recall10,
        "MRR@10": mrr10,
        "MAP@10": map10,
    }
