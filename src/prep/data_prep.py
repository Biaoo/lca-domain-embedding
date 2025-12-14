"""
Data preparation utilities for LCA embedding fine-tuning.
"""
import json
import os
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from datasets import Dataset


def load_hard_negatives(path: str | None) -> Dict[str, List[str]]:
    if not path or not os.path.exists(path):
        return {}
    mapping: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            qid = entry.get("id")
            candidates = entry.get("hard_negatives", [])
            texts = [item["text"] for item in candidates if "text" in item]
            if qid and texts:
                mapping[qid] = texts
    return mapping


def balance_sources(
    df: pd.DataFrame,
    source_column: str,
    random_seed: int,
    flow_label: str = "flow",
    process_label: str = "process",
) -> pd.DataFrame:
    """Balance flow and process rows to 1:1 if both are present."""
    if source_column not in df.columns:
        print(f"  - Source column '{source_column}' not found, skip balancing")
        return df

    flow_mask = df[source_column] == flow_label
    process_mask = df[source_column] == process_label
    flow_count = int(flow_mask.sum())
    process_count = int(process_mask.sum())

    if flow_count == 0 or process_count == 0:
        print(
            f"  - Skip balancing: counts -> {process_label}: {process_count}, {flow_label}: {flow_count}"
        )
        return df

    target = min(flow_count, process_count)
    balanced_flow = df[flow_mask].sample(n=target, random_state=random_seed)
    balanced_process = df[process_mask].sample(n=target, random_state=random_seed)
    others = df[~(flow_mask | process_mask)]

    balanced_df = pd.concat([balanced_flow, balanced_process, others], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print(
        f"  - Balanced sources to 1:1 ({process_label}: {process_count}->{target}, "
        f"{flow_label}: {flow_count}->{target}); others kept: {len(others)}"
    )
    return balanced_df


def build_doc_id(row) -> str:
    uuid_raw = row.get("dataset_uuid", "")
    version_raw = row.get("dataset_version", "")
    uuid = "" if pd.isna(uuid_raw) else str(uuid_raw).strip()
    version = "" if pd.isna(version_raw) else str(version_raw).strip()
    return f"{uuid}|{version}" if version else uuid


def sample_negatives(df: pd.DataFrame, neg_num: int, random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample negatives excluding same doc_id; returns (df_with_neg,pos,neg, doc_df)."""
    doc_df = df[["doc_id", "doc_text"]].drop_duplicates(subset=["doc_id"], keep="first")
    doc_ids = doc_df["doc_id"].tolist()
    doc_texts = doc_df["doc_text"].tolist()
    print(f"  - Corpus unique docs: {len(doc_df)}")

    print(f"\n[3/6] Sampling {neg_num} negative texts per query (exclude same doc_id)...")
    np.random.seed(random_seed)

    neg_samples = []
    for doc_id in df["doc_id"]:
        available_idx = [i for i, d in enumerate(doc_ids) if d != doc_id]
        replace_flag = len(available_idx) < neg_num
        sampled_idx = np.random.choice(available_idx, size=neg_num, replace=replace_flag)
        neg_samples.append([doc_texts[int(i)] for i in sampled_idx])

    df = df.copy()
    df["pos"] = [[t] for t in df["doc_text"]]
    df["neg"] = neg_samples
    return df, doc_df


def inject_hard_negatives(df: pd.DataFrame, hard_neg_map: Dict[str, List[str]]) -> pd.DataFrame:
    if not hard_neg_map:
        return df
    merged_neg = []
    for idx, row in df.iterrows():
        negs = list(row["neg"])
        extras = hard_neg_map.get(row["id"])
        if extras:
            for text in extras:
                if text not in negs:
                    negs.append(text)
        merged_neg.append(negs)
    df = df.copy()
    df["neg"] = merged_neg
    return df


def split_and_save(df: pd.DataFrame, doc_df: pd.DataFrame, output_dir: str, test_size: float, random_seed: int):
    from datasets import Dataset

    ds = Dataset.from_pandas(df[["id", "query", "pos", "neg", "prompt", "doc_id"]])
    print(f"  - Dataset created with {len(ds)} examples")

    split = ds.train_test_split(test_size=test_size, shuffle=True, seed=random_seed)
    train = split["train"]
    test = split["test"]
    print(f"  - Training set: {len(train)} examples")
    print(f"  - Test set: {len(test)} examples")

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "training.json")
    train.to_json(train_path)
    print(f"  - Saved: {train_path}")

    queries = test.select_columns(column_names=["id", "query"])
    queries = queries.rename_column("query", "text")
    queries_path = os.path.join(output_dir, "test_queries.jsonl")
    queries.to_json(queries_path)
    print(f"  - Saved: {queries_path}")

    corpus_ds = Dataset.from_pandas(doc_df)
    corpus_ds = corpus_ds.rename_column("doc_id", "id")
    corpus_ds = corpus_ds.rename_column("doc_text", "text")
    corpus_path = os.path.join(output_dir, "corpus.jsonl")
    corpus_ds.to_json(corpus_path)
    print(f"  - Saved: {corpus_path}")

    qrels = test.select_columns(["id", "doc_id"])
    qrels = qrels.rename_column("id", "qid")
    qrels = qrels.rename_column("doc_id", "docid")
    qrels = qrels.add_column("relevance", [1] * len(qrels))
    qrels_path = os.path.join(output_dir, "test_qrels.jsonl")
    qrels.to_json(qrels_path)
    print(f"  - Saved: {qrels_path}")

    return train, test
