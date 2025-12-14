"""
Compare one or more queries against multiple documents across several embedding models.
Supports SiliconFlow API models and local SentenceTransformer checkpoints, and saves results to CSV.

Usage example:
    uv run python scripts/compare_pair_similarity.py \\
      --query "steel production" \\
      --query "waste steel treatment" \\
      --doc_file data/markdown/doc1.md \\
      --doc_file data/markdown/doc2.md \\
      --models "data/model/BAAI--bge-m3,data/output/lca-bge-m3-finetuned" \\
      --output_csv data/output/pair_similarity.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.cal_similarity import cosine_similarity
from src.embedding import get_embedding

DEFAULT_MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-m3",
]


def load_entries(values: List[str] | None, files: List[str] | None, prefix: str) -> list[dict[str, str]]:
    """Collect text entries from inline strings and/or file paths."""
    entries: list[dict[str, str]] = []
    counter = 1

    for text in values or []:
        entry_id = f"{prefix}_{counter}"
        entries.append({"id": entry_id, "text": text.strip(), "source": entry_id})
        counter += 1

    for file_path in files or []:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{prefix.capitalize()} file not found: {path}")
        entry_id = path.stem or f"{prefix}_file_{counter}"
        entries.append({"id": entry_id, "text": path.read_text(encoding="utf-8").strip(), "source": str(path)})
        counter += 1

    if not entries:
        raise ValueError(f"At least one {prefix} must be provided via --{prefix} or --{prefix}_file.")

    return entries


def main():
    parser = argparse.ArgumentParser(description="Compare multi-query multi-document similarity across embedding models.")
    parser.add_argument("--query", action="append", help="Query text (can be repeated)")
    parser.add_argument("--query_file", action="append", help="Path to a file containing a query text (can be repeated)")
    parser.add_argument("--doc", action="append", help="Document text (can be repeated)")
    parser.add_argument("--doc_file", action="append", help="Path to a document file (can be repeated)")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of embedding models to evaluate",
    )
    parser.add_argument("--output_csv", type=str, default="data/output/pair_similarity.csv", help="Path to save CSV results")
    args = parser.parse_args()

    queries = load_entries(args.query, args.query_file, prefix="query")
    documents = load_entries(args.doc, args.doc_file, prefix="doc")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("No embedding models specified.")

    print("=" * 60)
    print("Multi-Query vs. Multi-Document Similarity Comparison")
    print("=" * 60)
    print(f"Queries: {len(queries)} items, documents: {len(documents)}, models: {len(models)}")

    rows: list[dict[str, str | float]] = []
    for model in models:
        print(f"\n>>> Model: {model}")
        query_embeddings: dict[str, list[float]] = {}
        doc_embeddings: dict[str, list[float]] = {}

        for query in queries:
            emb = get_embedding(query["text"], model=model)
            if emb is None:
                raise RuntimeError(f"Failed to encode query '{query['id']}' with model {model}")
            query_embeddings[query["id"]] = emb

        for doc in documents:
            emb = get_embedding(doc["text"], model=model)
            if emb is None:
                raise RuntimeError(f"Failed to encode doc '{doc['id']}' with model {model}")
            doc_embeddings[doc["id"]] = emb

        for doc in documents:
            doc_emb = doc_embeddings[doc["id"]]
            preview = doc["text"][:80].replace("\n", " ")
            for query in queries:
                score = cosine_similarity(query_embeddings[query["id"]], doc_emb)
                rows.append(
                    {
                        "model": model,
                        "query_id": query["id"],
                        "query_source": query["source"],
                        "doc_id": doc["id"],
                        "doc_source": doc["source"],
                        "similarity": round(float(score), 6),
                    }
                )
                print(
                    f"  Query[{query['id']}] vs Doc[{doc['id']}] -> {score:.4f} | {preview}"
                    f"{'...' if len(doc['text']) > 80 else ''}"
                )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "query_id", "query_source", "doc_id", "doc_source", "similarity"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to CSV: {output_path.resolve()}" )


if __name__ == "__main__":
    """
uv run python scripts/compare_pair_similarity.py \
  --query "LCA profile of BOF (basic oxygen furnace) steel production" \
  --query "Lifecycle impacts of EAF (electric arc furnace) reinforcing steel" \
  --query "China hot-rolled steel" \
  --query "European hot-rolled steel" \
  --query "Waste steel disposal via inert landfill (environmental burden)" \
  --doc_file data/markdown/9ed5f062-bef6-309e-b265-80773cf14d82.md \
  --doc_file data/markdown/b24376a9-177c-3984-b007-66ee79f183d1.md \
  --doc_file data/markdown/2bc4178d-827f-3334-9f09-0d29d32812cf.md \
  --doc_file data/markdown/8a7b1681-54db-3dfa-a14f-e8d3d9a6ebc9.md \
  --doc_file data/markdown/7a7ee91a-8e1d-33c9-b780-0ad3a8fadc64.md \
  --doc_file data/markdown/e67df727-8046-31af-8828-b3684c6f568a.md \
  --doc_file data/markdown/dd0a1e98-1426-3b38-84ba-6f1431bba2f6.md \
  --doc_file data/markdown/6d4c8637-d590-38e3-a919-1f4ad330c3f6.md \
  --doc_file data/markdown/c4acfc12-b703-3641-9177-12f27b365683.md \
  --models "data/model/BAAI--bge-m3,data/output/lca-bge-m3-finetuned" \
  --output_csv data/eval/steel_similarity.csv

  
uv run python scripts/compare_pair_similarity.py \
  --query "China hot-rolled steel" \
  --query "European hot-rolled steel" \
  --doc_file data/markdown/9ed5f062-bef6-309e-b265-80773cf14d82.md \
  --doc_file data/markdown/b24376a9-177c-3984-b007-66ee79f183d1.md \
  --doc_file data/markdown/2bc4178d-827f-3334-9f09-0d29d32812cf.md \
  --doc_file data/markdown/8a7b1681-54db-3dfa-a14f-e8d3d9a6ebc9.md \
  --doc_file data/markdown/7a7ee91a-8e1d-33c9-b780-0ad3a8fadc64.md \
  --doc_file data/markdown/e67df727-8046-31af-8828-b3684c6f568a.md \
  --doc_file data/markdown/dd0a1e98-1426-3b38-84ba-6f1431bba2f6.md \
  --doc_file data/markdown/6d4c8637-d590-38e3-a919-1f4ad330c3f6.md \
  --doc_file data/markdown/c4acfc12-b703-3641-9177-12f27b365683.md \
  --doc_file data/markdown/ef440aff-8fbf-3987-bb33-bb93468bcd0d.md \
  --doc_file data/markdown/0e64b34e-8da9-3303-9c35-68c25114558a.md \
  --models "data/model/BAAI--bge-m3,data/output/lca-bge-m3-finetuned" \
  --output_csv data/eval/steel_similarity.csv
    """
    main()
