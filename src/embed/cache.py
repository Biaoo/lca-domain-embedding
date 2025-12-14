"""
Embedding/cache utilities shared by pipeline scripts.
"""
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class ModelSpec:
    alias: str
    provider: str  # "local" | "openrouter"
    value: str


def parse_model_spec(spec: str) -> ModelSpec:
    """Parse alias=provider:value into a ModelSpec."""
    if "=" not in spec or ":" not in spec:
        raise ValueError(
            f"Invalid model spec '{spec}'. Expected format alias=provider:value"
        )
    alias_part, rest = spec.split("=", 1)
    provider, value = rest.split(":", 1)
    alias = alias_part.strip()
    provider = provider.strip().lower()
    value = value.strip()
    if provider not in {"local", "openrouter"}:
        raise ValueError(
            f"Unsupported provider '{provider}' in spec '{spec}'. Use local/openrouter"
        )
    if not alias or not value:
        raise ValueError(f"Alias and value must be non-empty in spec '{spec}'")
    return ModelSpec(alias=alias, provider=provider, value=value)


def load_eval_texts(data_dir: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load queries and corpus from JSONL files keeping (id, text) order."""
    queries_ds = load_dataset("json", data_files=f"{data_dir}/test_queries.jsonl")[
        "train"
    ]
    corpus_ds = load_dataset("json", data_files=f"{data_dir}/corpus.jsonl")["train"]

    queries = list(zip(queries_ds["id"], queries_ds["text"]))

    corpus: List[Tuple[str, str]] = []
    for item in corpus_ds:
        doc_id = item["id"]
        text = item["text"]
        if isinstance(text, list):
            corpus.append((doc_id, text[0] if len(text) == 1 else " ".join(text)))
        else:
            corpus.append((doc_id, text))

    print(f"Loaded data -> queries: {len(queries)}, corpus: {len(corpus)}")
    return queries, corpus


def normalize_embeddings(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def embed_local(
    model_path: str, texts: List[str], batch_size: int, device: str, normalize: bool
) -> np.ndarray:
    model = SentenceTransformer(model_path, trust_remote_code=True, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_openrouter(
    model_id: str,
    texts: List[str],
    batch_size: int,
    api_key: str,
    url: str,
    referrer: str,
    site_title: str,
    timeout: int,
    normalize: bool,
    retries: int,
    retry_backoff: float,
) -> np.ndarray:
    if not api_key:
        raise ValueError("OpenRouter API key is required for openrouter provider")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referrer:
        headers["HTTP-Referer"] = referrer
    if site_title:
        headers["X-Title"] = site_title

    embeddings: List[List[float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"OpenRouter {model_id}"):
        batch = texts[i : i + batch_size]
        payload = {
            "model": model_id,
            "input": batch,
            "encoding_format": "float",
        }
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                if "data" not in data:
                    raise RuntimeError(
                        f"Unexpected response for batch starting at {i}: {data}"
                    )
                rows = sorted(data["data"], key=lambda x: x.get("index", 0))
                embeddings.extend([row["embedding"] for row in rows])
                break
            except Exception as e:
                last_err = e
                resp_text = ""
                if "resp" in locals():
                    resp_text = resp.text[:200].replace("\n", " ")
                print(
                    f"[Retry {attempt}/{retries}] OpenRouter batch starting at {i} failed: {e}. "
                    f"Resp: {resp_text}"
                )
                if attempt == retries:
                    raise
                time.sleep(retry_backoff * attempt)

    arr = np.asarray(embeddings, dtype=np.float32)
    if normalize:
        arr = normalize_embeddings(arr)
    return arr


def save_embeddings(
    spec: ModelSpec,
    output_dir: Path,
    queries: List[Tuple[str, str]],
    corpus: List[Tuple[str, str]],
    q_emb: np.ndarray,
    c_emb: np.ndarray,
    normalize: bool,
    data_dir: str,
    batch_size: int,
    device: str,
):
    model_dir = output_dir / spec.alias
    model_dir.mkdir(parents=True, exist_ok=True)

    np.save(model_dir / "queries_embeddings.npy", q_emb)
    np.save(model_dir / "corpus_embeddings.npy", c_emb)

    (model_dir / "queries_ids.json").write_text(
        json.dumps([qid for qid, _ in queries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (model_dir / "corpus_ids.json").write_text(
        json.dumps([doc_id for doc_id, _ in corpus], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta = {
        "model": {
            "alias": spec.alias,
            "provider": spec.provider,
            "value": spec.value,
        },
        "generated_at": datetime.utcnow().isoformat(),
        "data_dir": data_dir,
        "counts": {
            "queries": len(queries),
            "corpus": len(corpus),
        },
        "embedding_dim": int(q_emb.shape[1]) if q_emb.size else None,
        "normalize": normalize,
        "batch_size": batch_size,
        "device": device,
    }
    (model_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Saved embeddings for {spec.alias} -> {model_dir} (queries: {len(queries)}, corpus: {len(corpus)})"
    )
