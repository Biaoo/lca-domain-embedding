#!/usr/bin/env python3
"""
Export evaluation queries/corpus + (optional) cached embeddings to a dataset that can be
loaded by Apple's Embedding Atlas.

Typical workflow:
  1) Cache embeddings (raw/ft/others):
       python scripts/pipeline/05_cache_embeddings.py --model raw=local:... --model ft=local:...

  2) Export a dataset with vector columns:
       python scripts/tools/export_embedding_atlas_dataset.py \
         --data_dir data/ft_data \
         --cache_dir data/eval_cache \
         --model raw --model ft \
         --out data/output/embedding_atlas/lca_eval.parquet

  3) Visualize in Embedding Atlas:
       embedding-atlas data/output/embedding_atlas/lca_eval.parquet --vector vector_ft --text text
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_DATA_DIR = "data/ft_data"
DEFAULT_CACHE_DIR = "data/eval_cache"
DEFAULT_OUT = "data/output/embedding_atlas/lca_eval.parquet"


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def coerce_text(value: Any) -> str:
    if isinstance(value, list):
        return value[0] if len(value) == 1 else " ".join(str(x) for x in value)
    return str(value)


_FIELD_RE = re.compile(r"^\\*\\*(.+?):\\*\\*\\s*(.+?)\\s*$")


def parse_markdown_meta(text: str) -> Dict[str, str]:
    """
    Heuristic metadata extraction from the markdown-like corpus text.
    Keeps it best-effort: missing fields are simply absent.
    """
    meta: Dict[str, str] = {}
    title: Optional[str] = None

    for raw in text.splitlines()[:120]:
        line = raw.strip()
        if not line:
            continue

        if title is None and line.startswith("#"):
            title = line.lstrip("#").strip()
            continue

        m = _FIELD_RE.match(line)
        if not m:
            continue
        key = m.group(1).strip().lower().replace(" ", "_")
        val = m.group(2).strip().strip("`")
        meta[key] = val

    if title is not None:
        meta["title"] = title

    # Normalize some known keys to stable names.
    normalized: Dict[str, str] = {}
    for k, v in meta.items():
        if k in {"entity"}:
            normalized["entity"] = v
        elif k in {"uuid"}:
            normalized["uuid"] = v
        elif k in {"version"}:
            normalized["version"] = v
        elif k in {"data_set_type", "dataset_type"}:
            normalized["data_set_type"] = v
        elif k in {"classification"}:
            normalized["classification"] = v
        elif k in {"cas"}:
            normalized["cas"] = v
        elif k in {"ec_number"}:
            normalized["ec_number"] = v
        elif k == "title":
            normalized["title"] = v
    return normalized


def load_qrels(path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Return:
      - qid -> list[docid]
      - docid -> list[qid]
    """
    if not path.exists():
        return {}, {}
    qid_to_docids: Dict[str, List[str]] = defaultdict(list)
    docid_to_qids: Dict[str, List[str]] = defaultdict(list)
    for row in read_jsonl(path):
        qid = str(row["qid"])
        docid = str(row["docid"])
        qid_to_docids[qid].append(docid)
        docid_to_qids[docid].append(qid)
    return dict(qid_to_docids), dict(docid_to_qids)


def load_cached_vectors(model_dir: Path) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Load cached embeddings produced by `scripts/pipeline/05_cache_embeddings.py`.
    Returns:
      - query_id -> vector(list[float])
      - doc_id   -> vector(list[float])
    """
    q_emb = np.load(model_dir / "queries_embeddings.npy")
    c_emb = np.load(model_dir / "corpus_embeddings.npy")
    q_ids = json.loads((model_dir / "queries_ids.json").read_text(encoding="utf-8"))
    c_ids = json.loads((model_dir / "corpus_ids.json").read_text(encoding="utf-8"))

    if len(q_ids) != q_emb.shape[0]:
        raise ValueError(f"{model_dir.name}: queries_ids length mismatch with embeddings")
    if len(c_ids) != c_emb.shape[0]:
        raise ValueError(f"{model_dir.name}: corpus_ids length mismatch with embeddings")

    q_map = {str(i): q_emb[idx].astype(np.float32).tolist() for idx, i in enumerate(q_ids)}
    c_map = {str(i): c_emb[idx].astype(np.float32).tolist() for idx, i in enumerate(c_ids)}
    return q_map, c_map


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def write_parquet(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pyarrow is required for --format parquet. "
            "Install it or use --format jsonl instead."
        ) from e

    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export LCA eval data (+ cached embeddings) for Embedding Atlas"
    )
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="Cached model alias under cache_dir (repeatable). Adds vector_{alias} column.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help="Output dataset path (.parquet or .jsonl recommended).",
    )
    p.add_argument(
        "--format",
        choices=["parquet", "jsonl"],
        default="parquet",
        help="Output format. parquet is preferred if available.",
    )
    p.add_argument(
        "--include",
        choices=["queries", "corpus", "both"],
        default="both",
        help="Which split to include in the output dataset.",
    )
    p.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Optional cap on exported rows (0 means no cap).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    out_path = Path(args.out)

    queries_path = data_dir / "test_queries.jsonl"
    corpus_path = data_dir / "corpus.jsonl"
    qrels_path = data_dir / "test_qrels.jsonl"

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing {queries_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing {corpus_path}")

    queries_rows = read_jsonl(queries_path)
    corpus_rows = read_jsonl(corpus_path)
    qid_to_docids, docid_to_qids = load_qrels(qrels_path)

    # Load cached vectors for each model alias.
    model_vectors: Dict[str, Tuple[Dict[str, List[float]], Dict[str, List[float]]]] = {}
    for alias in args.model:
        model_dir = cache_dir / alias
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Cached model not found: {model_dir} (run scripts/pipeline/05_cache_embeddings.py first)"
            )
        model_vectors[alias] = load_cached_vectors(model_dir)

    records: List[dict] = []
    if args.include in {"queries", "both"}:
        for row in queries_rows:
            qid = str(row["id"])
            text = coerce_text(row.get("text", ""))
            rec: Dict[str, Any] = {
                "kind": "query",
                "id": qid,
                "text": text,
                "num_positive_docs": len(qid_to_docids.get(qid, [])),
                "positive_docids": qid_to_docids.get(qid, []),
            }
            for alias, (q_map, _c_map) in model_vectors.items():
                if qid in q_map:
                    rec[f"vector_{alias}"] = q_map[qid]
            records.append(rec)

    if args.include in {"corpus", "both"}:
        for row in corpus_rows:
            docid = str(row["id"])
            text = coerce_text(row.get("text", ""))
            rec = {
                "kind": "corpus",
                "id": docid,
                "text": text,
                "num_matched_queries": len(docid_to_qids.get(docid, [])),
                "matched_qids": docid_to_qids.get(docid, []),
            }
            rec.update(parse_markdown_meta(text))
            for alias, (_q_map, c_map) in model_vectors.items():
                if docid in c_map:
                    rec[f"vector_{alias}"] = c_map[docid]
            records.append(rec)

    if args.max_rows and args.max_rows > 0:
        records = records[: args.max_rows]

    if args.format == "jsonl":
        write_jsonl(out_path, records)
    else:
        write_parquet(out_path, records)

    vector_cols = [f"vector_{a}" for a in args.model]
    print("Exported dataset for Embedding Atlas")
    print(f"  rows       : {len(records)}")
    print(f"  out        : {out_path.resolve()}")
    if vector_cols:
        print(f"  vector cols: {', '.join(vector_cols)}")
    else:
        print("  vector cols: (none)  (you can still use --text + --model in Embedding Atlas)")


if __name__ == "__main__":
    main()

