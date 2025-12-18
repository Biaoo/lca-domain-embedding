#!/usr/bin/env python3
"""
Export a directory of markdown documents into a single dataset (Parquet/JSONL)
that can be loaded by Apple's Embedding Atlas.

This is intended for visualizing the Supabase markdown exports under:
  data/supabase_exports_markdown/{flows,processs}/*.md

Example:
  .venv/bin/python scripts/tools/export_embedding_atlas_markdown_dir.py \
    --input_dir data/supabase_exports_markdown \
    --subset both \
    --out data/output/embedding_atlas/supabase_markdown.parquet

Then:
  embedding-atlas data/output/embedding_atlas/supabase_markdown.parquet \
    --text text \
    --model BIaoo/lca-qwen3-embedding \
    --trust-remote-code \
    --umap-metric cosine \
    --umap-random-state 42
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


UUID_VERSION_RE = re.compile(
    r"(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})-(?P<version>\d{2}\.\d{2}\.\d{3})$"
)
FIELD_RE = re.compile(r"^\*\*(.+?):\*\*\s*(.+?)\s*$")


DEFAULT_INPUT_DIR = "data/supabase_exports_markdown"
DEFAULT_OUT = "data/output/embedding_atlas/supabase_markdown.parquet"


def normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


@dataclass(frozen=True)
class Record:
    id: str
    kind: str  # flow | process | unknown
    uuid: Optional[str]
    version: Optional[str]
    title: str
    entity: Optional[str]
    location: Optional[str]
    classification: Optional[str]
    data_set_type: Optional[str]
    reference_flow: Optional[str]
    functional_unit: Optional[str]
    cas: Optional[str]
    ec_number: Optional[str]
    file_path: str
    text: str
    meta_json: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "uuid": self.uuid,
            "version": self.version,
            "title": self.title,
            "entity": self.entity,
            "location": self.location,
            "classification": self.classification,
            "data_set_type": self.data_set_type,
            "reference_flow": self.reference_flow,
            "functional_unit": self.functional_unit,
            "cas": self.cas,
            "ec_number": self.ec_number,
            "file_path": self.file_path,
            "text": self.text,
            "meta_json": self.meta_json,
        }


def infer_kind(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "flows" in parts:
        return "flow"
    if "processs" in parts:
        return "process"
    return "unknown"


def parse_uuid_version_from_stem(stem: str) -> tuple[Optional[str], Optional[str]]:
    m = UUID_VERSION_RE.match(stem)
    if not m:
        return None, None
    return m.group("uuid"), m.group("version")


def read_text_capped(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if max_chars <= 0:
            return f.read()
        return f.read(max_chars)


def strip_markdown(text: str) -> str:
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

    # Remove inline code ticks
    text = text.replace("`", "")

    # Strip common markdown prefixes while keeping content
    cleaned: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^#{1,6}\s+", "", s)  # headings
        s = re.sub(r"^[-*+]\s+", "", s)  # bullet
        cleaned.append(s)
    text = "\n".join(cleaned)

    # Collapse whitespace a bit
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_markdown_meta(text: str) -> tuple[str, Dict[str, str]]:
    """
    Parse best-effort metadata and title from the first part of the markdown.
    Returns (title, meta_dict).
    """
    meta: Dict[str, str] = {}
    title: Optional[str] = None

    for raw in text.splitlines()[:180]:
        line = raw.strip()
        if not line:
            continue

        if title is None and line.startswith("#"):
            title = line.lstrip("#").strip()
            continue

        m = FIELD_RE.match(line)
        if not m:
            continue
        key = normalize_key(m.group(1))
        val = m.group(2).strip().strip("`")
        meta[key] = val

    if title is None:
        title = meta.get("title") or "unknown"
    return title, meta


def build_record(path: Path, input_dir: Path, text_mode: str, max_chars: int) -> Record:
    raw_text = read_text_capped(path, max_chars=max_chars)
    title, meta = parse_markdown_meta(raw_text)
    uuid, version = parse_uuid_version_from_stem(path.stem)

    # Prefer explicit meta if present
    uuid = meta.get("uuid", uuid)
    version = meta.get("version", version)

    kind = infer_kind(path)
    entity = meta.get("entity")
    location = meta.get("location")
    classification = meta.get("classification")
    data_set_type = meta.get("data_set_type") or meta.get("data_set_type:")
    reference_flow = meta.get("reference_flow")
    functional_unit = meta.get("functional_unit")
    cas = meta.get("cas")
    ec_number = meta.get("ec_number")

    text = raw_text if text_mode == "markdown" else strip_markdown(raw_text)

    rel_path = str(path.relative_to(input_dir))
    meta_json = json.dumps(meta, ensure_ascii=False)

    return Record(
        id=path.stem,
        kind=kind,
        uuid=uuid,
        version=version,
        title=title,
        entity=entity,
        location=location,
        classification=classification,
        data_set_type=data_set_type,
        reference_flow=reference_flow,
        functional_unit=functional_unit,
        cas=cas,
        ec_number=ec_number,
        file_path=rel_path,
        text=text,
        meta_json=meta_json,
    )


def iter_markdown_files(input_dir: Path, subset: str) -> List[Path]:
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


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False))
            f.write("\n")


def write_parquet(path: Path, records: Iterator[Record], batch_size: int) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pyarrow is required for --format parquet. Install it or use --format jsonl."
        ) from e

    path.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    buffer: List[dict] = []

    def flush() -> None:
        nonlocal writer, buffer
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)
        buffer = []

    try:
        for rec in records:
            buffer.append(rec.to_dict())
            if len(buffer) >= batch_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()


def load_vector_cache(cache_dir: Path, alias: str) -> Tuple[Dict[str, int], Any]:
    """
    Load cached embeddings created by scripts/tools/cache_markdown_embeddings.py.
    Returns: (id_to_index, embeddings_mmap)
    """
    import numpy as np

    model_dir = cache_dir / alias
    ids_path = model_dir / "ids.json"
    emb_path = model_dir / "embeddings.npy"
    if not ids_path.exists() or not emb_path.exists():
        raise FileNotFoundError(f"Vector cache missing for alias '{alias}' under {model_dir}")

    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if not isinstance(ids, list):
        raise ValueError(f"Invalid ids.json for alias '{alias}'")
    id_to_index = {str(v): i for i, v in enumerate(ids)}
    emb = np.load(emb_path, mmap_mode="r")
    return id_to_index, emb


def write_parquet_with_vectors(
    path: Path,
    records: Iterator[Record],
    batch_size: int,
    vector_caches: Dict[str, Tuple[Dict[str, int], Any]],
) -> None:
    """
    Vector-efficient parquet writer:
      - scalar columns written normally
      - vector_{alias} columns written as FixedSizeList<float32> when possible
    """
    try:
        import numpy as np
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pyarrow is required for --format parquet. Install it or use --format jsonl."
        ) from e

    path.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    scalar_cols: Dict[str, List[Any]] = {}
    vector_cols: Dict[str, List[Optional[np.ndarray]]] = {
        f"vector_{alias}": [] for alias in vector_caches.keys()
    }

    def flush() -> None:
        nonlocal writer, scalar_cols, vector_cols
        if not scalar_cols:
            return

        arrays: List[pa.Array] = []
        names: List[str] = []

        # Keep scalar column order stable.
        for name in scalar_cols.keys():
            names.append(name)
            arrays.append(pa.array(scalar_cols[name]))

        for col_name, vecs in vector_cols.items():
            names.append(col_name)
            if not vecs:
                arrays.append(pa.array([], type=pa.list_(pa.float32())))
                continue
            if all(v is not None for v in vecs):
                mat = np.stack([v for v in vecs if v is not None]).astype(np.float32)
                dim = mat.shape[1]
                flat = pa.array(mat.reshape(-1), type=pa.float32())
                arrays.append(pa.FixedSizeListArray.from_arrays(flat, dim))
            else:
                arrays.append(
                    pa.array(
                        [v.astype(np.float32).tolist() if v is not None else None for v in vecs],
                        type=pa.list_(pa.float32()),
                    )
                )

        table = pa.Table.from_arrays(arrays, names=names)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)

        scalar_cols = {}
        vector_cols = {k: [] for k in vector_cols.keys()}

    def push_scalar_row(row: dict) -> None:
        nonlocal scalar_cols
        for k, v in row.items():
            if k.startswith("vector_"):
                continue
            if k not in scalar_cols:
                scalar_cols[k] = []
            scalar_cols[k].append(v)

    def push_vector_row(row_id: str) -> None:
        for alias, (id_to_index, emb) in vector_caches.items():
            idx = id_to_index.get(row_id)
            if idx is None:
                vector_cols[f"vector_{alias}"].append(None)
            else:
                vector_cols[f"vector_{alias}"].append(emb[int(idx)])

    try:
        for rec in records:
            row = rec.to_dict()
            push_scalar_row(row)
            push_vector_row(rec.id)
            if len(next(iter(scalar_cols.values()))) >= batch_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a markdown directory as an Embedding Atlas dataset"
    )
    p.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--format", choices=["parquet", "jsonl"], default="parquet")
    p.add_argument(
        "--subset",
        choices=["flows", "processs", "both"],
        default="both",
        help="Which subfolders to include under input_dir.",
    )
    p.add_argument(
        "--text_mode",
        choices=["markdown", "plain"],
        default="markdown",
        help="Whether to keep markdown or strip it to plain text for embedding.",
    )
    p.add_argument(
        "--max_chars",
        type=int,
        default=20000,
        help="Max chars per document to read (0 for no cap).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only export first N files (0 means no limit).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Parquet write batch size.",
    )
    p.add_argument(
        "--vector_cache_dir",
        type=str,
        default="",
        help="Directory containing cached vectors from cache_markdown_embeddings.py.",
    )
    p.add_argument(
        "--vector_model",
        action="append",
        default=[],
        help="Alias name(s) under vector_cache_dir. Adds vector_{alias} columns.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    paths = iter_markdown_files(input_dir, subset=args.subset)
    if not paths:
        raise RuntimeError(f"No markdown files found under {input_dir} (subset={args.subset})")

    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    def records_iter() -> Iterator[Record]:
        for p in paths:
            yield build_record(
                p,
                input_dir=input_dir,
                text_mode=args.text_mode,
                max_chars=args.max_chars,
            )

    out_path = Path(args.out)
    vector_caches: Dict[str, Tuple[Dict[str, int], Any]] = {}
    if args.vector_cache_dir and args.vector_model:
        cache_dir = Path(args.vector_cache_dir)
        for alias in args.vector_model:
            vector_caches[alias] = load_vector_cache(cache_dir, alias)

    if args.format == "jsonl":
        if vector_caches:
            raise ValueError("JSONL export does not support vectors; use --format parquet.")
        write_jsonl(out_path, records_iter())
    else:
        if vector_caches:
            write_parquet_with_vectors(
                out_path,
                records_iter(),
                batch_size=args.batch_size,
                vector_caches=vector_caches,
            )
        else:
            write_parquet(out_path, records_iter(), batch_size=args.batch_size)

    print("Exported dataset for Embedding Atlas")
    print(f"  files : {len(paths)}")
    print(f"  out   : {out_path.resolve()}")
    print(f"  text  : text (mode={args.text_mode}, max_chars={args.max_chars})")
    if vector_caches:
        print(f"  vectors: {', '.join([f'vector_{a}' for a in vector_caches.keys()])}")


if __name__ == "__main__":
    main()
