"""
Mine hard negative samples using a two-step approach:
1. 使用基础 embedding 模型（默认 BAAI--bge-m3）对 query 在语料中检索 Top-K 候选。
2. 通过大模型（LLM）判断候选是否属于“语义相近但不同工艺/地区”的 hard negative。

输出 JSONL，每行包含：
{
  "id": "<query id>",
  "query": "<query text>",
  "hard_negatives": [
      {"doc_id": "<corpus id>", "text": "<doc text>"}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from datasets import Dataset
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.model import DASHSCOPE_API_KEY, DASHSCOPE_API_URL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard negatives with retrieval + LLM filtering.")
    parser.add_argument("--data_dir", type=str, default="data/ft_data", help="Directory containing training.json and corpus.jsonl")
    parser.add_argument("--model_path", type=str, default="data/model/Qwen--Qwen3-Embedding-0.6B", help="Embedding model path/name for retrieval")
    parser.add_argument("--top_k", type=int, default=20, help="Top-K retrieval candidates per query")
    parser.add_argument("--max_hard_neg", type=int, default=3, help="Max hard negatives to keep per query")
    parser.add_argument("--max_queries", type=int, default=None, help="Optional limit on number of queries to process after dedupe")
    parser.add_argument("--llm_model", type=str, default="qwen-plus", help="LLM model name for classification")
    parser.add_argument("--output_path", type=str, default="data/ft_data/hard_negatives.jsonl", help="Where to store mined hard negatives")
    parser.add_argument("--truncate_chars", type=int, default=800, help="Max chars of document text to send to LLM")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file (skip processed query ids)")
    parser.add_argument("--no_dedupe_by_title", action="store_true", help="Disable dedupe by dataset title")
    parser.add_argument("--encode_device", type=str, default=None, help="Force encoding device (e.g., cpu, cuda:0)")
    parser.add_argument("--encode_batch_size", type=int, default=32, help="Batch size for corpus encoding")
    parser.add_argument("--embeddings_path", type=str, default="data/ft_data/corpus_embeddings.npy", help="Path to cache corpus embeddings")
    parser.add_argument("--encode_queries_path", type=str, default="data/ft_data/query_embeddings.npy", help="Cache for query embeddings")
    parser.add_argument("--encode_only", action="store_true", help="Only encode corpus/query embeddings and exit")
    return parser.parse_args()


def load_dataset(data_dir: str) -> tuple[Dataset, Dataset]:
    train_path = os.path.join(data_dir, "training.json")
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"training.json not found at {train_path}. Run prepare_ft_data.py first.")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"corpus.jsonl not found at {corpus_path}.")
    train_ds = Dataset.from_json(train_path)
    corpus_ds = Dataset.from_json(corpus_path)
    return train_ds, corpus_ds


def encode_corpus(
    corpus_texts: List[str],
    model: SentenceTransformer,
    batch_size: int,
) -> np.ndarray:
    return model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def encode_queries(
    records: List[dict],
    model: SentenceTransformer,
    batch_size: int,
) -> np.ndarray:
    texts = [rec["query"] for rec in records]
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def build_corpus_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def create_llm_client() -> OpenAI:
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_API_URL)


def classify_candidate(
    client: OpenAI,
    llm_model: str,
    query_text: str,
    positive_text: str,
    candidate_text: str,
) -> str:
    system_prompt = """You are an LCA domain expert. Your task is to evaluate the relationship between the query/positive document and a candidate document.
Classification rules:
- If the candidate describes the same process/region/goal as the positive sample, output "POSITIVE".
- If the candidate is highly related to the query but differs in process, region, or application, treat it as a “hard negative” and output "HARD_NEGATIVE".
- If the candidate is mostly unrelated to the query topic, output "RANDOM_NEGATIVE".
Return JSON only: {"label": "...", "reason": "..."}"""

    user_prompt = f"""Query:
{query_text}

Positive document:
{positive_text}

Candidate document:
{candidate_text}
"""

    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "hard_negative_label",
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["POSITIVE", "HARD_NEGATIVE", "RANDOM_NEGATIVE"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["label"],
                    "additionalProperties": False,
                },
            },
        },
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    return data.get("label", "RANDOM_NEGATIVE")


def extract_dataset_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return "unknown"


def load_processed_ids(path: Path) -> set[str]:
    processed = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "id" in entry:
                    processed.add(entry["id"])
            except json.JSONDecodeError:
                continue
    return processed


def build_query_records(
    train_ds: Dataset,
    dedupe_by_title: bool,
    max_queries: int | None,
) -> List[dict]:
    records = []
    seen_titles = set()
    for sample in train_ds:
        query_id = sample["id"]
        query_text = sample["query"]
        positive_text = sample["pos"][0]
        title = extract_dataset_title(positive_text)

        if dedupe_by_title:
            if title in seen_titles:
                continue
            seen_titles.add(title)

        records.append(
            {
                "id": query_id,
                "query": query_text,
                "positive": positive_text,
                "title": title,
            }
        )
        if max_queries and len(records) >= max_queries:
            break
    return records


def process_single_query(
    record: dict,
    query_embedding: np.ndarray,
    index: faiss.IndexFlatIP,
    corpus_ids: List[str],
    id_to_text: Dict[str, str],
    client: OpenAI,
    args: argparse.Namespace,
) -> dict | None:
    scores, indices = index.search(query_embedding.astype(np.float32), args.top_k)

    hard_negs = []
    seen_texts = set()

    for idx in indices[0]:
        candidate_id = corpus_ids[idx]
        candidate_text = id_to_text[candidate_id]

        if candidate_text.strip() == record["positive"].strip():
            continue
        if candidate_text in seen_texts:
            continue

        truncated_candidate = candidate_text[: args.truncate_chars]
        truncated_positive = record["positive"][: args.truncate_chars]

        label = classify_candidate(
            client,
            args.llm_model,
            record["query"],
            truncated_positive,
            truncated_candidate,
        )

        if label == "HARD_NEGATIVE":
            hard_negs.append(
                {
                    "doc_id": candidate_id,
                    "text": candidate_text,
                }
            )
            seen_texts.add(candidate_text)
        if len(hard_negs) >= args.max_hard_neg:
            break

    if hard_negs:
        return {
            "id": record["id"],
            "query": record["query"],
            "hard_negatives": hard_negs,
        }
    return None


def main():
    args = parse_args()
    train_ds, corpus_ds = load_dataset(args.data_dir)

    corpus_ids = corpus_ds["id"]
    raw_corpus_texts = corpus_ds["text"]

    def normalize_text(entry):
        if isinstance(entry, str):
            return entry
        if isinstance(entry, list) and entry:
            return entry[0]
        return str(entry)

    corpus_texts = [normalize_text(text) for text in raw_corpus_texts]
    id_to_text = dict(zip(corpus_ids, corpus_texts))

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = SentenceTransformer(
        args.model_path,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch_dtype},
    )
    print(f"Model dtype: {torch_dtype}")
    if args.encode_device:
        model.to(args.encode_device)

    embeddings_path = Path(args.embeddings_path) if args.embeddings_path else None
    if embeddings_path and embeddings_path.exists():
        print(f"Loading precomputed corpus embeddings from {embeddings_path}")
        corpus_embeddings = np.load(embeddings_path)
    else:
        print("Encoding corpus embeddings...")
        corpus_embeddings = encode_corpus(corpus_texts, model, args.encode_batch_size)
        if embeddings_path:
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, corpus_embeddings)
            print(f"Saved corpus embeddings to {embeddings_path}")

    dedupe_flag = not args.no_dedupe_by_title
    records = build_query_records(
        train_ds=train_ds,
        dedupe_by_title=dedupe_flag,
        max_queries=args.max_queries,
    )

    query_embeddings_path = Path(args.encode_queries_path) if args.encode_queries_path else None
    if query_embeddings_path and query_embeddings_path.exists():
        print(f"Loading precomputed query embeddings from {query_embeddings_path}")
        query_embeddings = np.load(query_embeddings_path)
    else:
        print("Encoding query embeddings...")
        query_embeddings = encode_queries(records, model, args.encode_batch_size)
        if query_embeddings_path:
            query_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(query_embeddings_path, query_embeddings)
            print(f"Saved query embeddings to {query_embeddings_path}")

    if args.encode_only:
        print("Encode-only flag set; exiting after embeddings generation.")
        return

    index = build_corpus_index(corpus_embeddings)
    client = create_llm_client()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = load_processed_ids(output_path) if args.resume else set()
    filtered_records = []
    filtered_embeddings = []
    for record, emb in zip(records, query_embeddings):
        if record["id"] in processed_ids:
            continue
        filtered_records.append(record)
        filtered_embeddings.append(emb)
    records = filtered_records
    query_embeddings_filtered = filtered_embeddings

    if not records:
        print("No queries to process (all done or filtered).")
        return

    mode = "a" if args.resume and output_path.exists() else "w"
    writer_lock = threading.Lock()

    with output_path.open(mode, encoding="utf-8") as wf:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    process_single_query,
                    record,
                    query_embeddings_filtered[idx].reshape(1, -1),
                    index,
                    corpus_ids,
                    id_to_text,
                    client,
                    args,
                )
                for idx, record in enumerate(records)
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Mining hard negatives"):
                result = future.result()
                if result:
                    with writer_lock:
                        wf.write(json.dumps(result, ensure_ascii=False) + "\n")
                        wf.flush()

    print(f"Hard negatives saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
