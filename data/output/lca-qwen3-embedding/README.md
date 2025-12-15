---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- lifecycle-assessment
- climate-change
- carbon-emission
- sustainability
pipeline_tag: sentence-similarity
library_name: sentence-transformers
base_model: Qwen/Qwen3-Embedding-0.6B
license: mit
---

# lca-qwen3-embedding

Domain embedding model for lifecycle assessment (LCA) retrieval. It encodes sentences and short passages into 1024‑d L2-normalized embeddings for semantic search, similarity scoring, and clustering.

## Background

Generic embedding models work well in open domains, but professional LCA retrieval often involves long, structured records (e.g., geography/technology/time fields) and domain-specific terminology. This model is trained to better align embeddings with LCA retrieval queries and documents.

## Results (our evaluation setup)

On an internal evaluation derived from TianGong LCA records (converted from the Tidas structured format into retrieval-friendly text), this model improved over the base `Qwen3-Embedding-0.6B` on both ranking quality and tail coverage:

- vs base `Qwen3-Embedding-0.6B`: **NDCG@10 +31.2%**, **Recall@10 +25.7%**, **MRR@10 +33.5%**, **Recall@100 +11.5%**

Evaluation scale (this experiment):

- Train: 17,037 query-doc pairs
- Eval: 1,893 queries / 3,786 corpus docs / 1,893 qrels

### Model comparisons

Key metrics (@10):

| Model | NDCG@10 | Recall@10 | MRR@10 | MAP@10 |
| --- | ---: | ---: | ---: | ---: |
| `Qwen3-Embedding-0.6B` (base) | 0.5808 | 0.7200 | 0.5367 | 0.5367 |
| `lca-qwen3-embedding` (this model) | **0.7623** | **0.9049** | **0.7163** | **0.7163** |
| `codestral-embed-2505` | 0.6628 | 0.8045 | 0.6180 | 0.6180 |
| `qwen3-embedding-8b` | 0.5905 | 0.7369 | 0.5442 | 0.5442 |
| `qwen3-embedding-4b` | 0.5836 | 0.7290 | 0.5377 | 0.5377 |
| `bge-m3` | 0.5839 | 0.7264 | 0.5388 | 0.5388 |

Tail coverage (@100):

| Model | NDCG@100 | Recall@100 |
| --- | ---: | ---: |
| `Qwen3-Embedding-0.6B` (base) | 0.6171 | 0.8922 |
| `lca-qwen3-embedding` (this model) | **0.7826** | **0.9947** |
| `codestral-embed-2505` | 0.6872 | 0.9171 |
| `qwen3-embedding-8b` | 0.6258 | 0.9033 |
| `qwen3-embedding-4b` | 0.6164 | 0.8822 |
| `bge-m3` | 0.6156 | 0.8743 |

Protocol note: embeddings are L2-normalized; retrieval uses inner product (equivalent to cosine similarity) with top-100 candidates.

## Model details (from the exported config)

- **Backbone**: Qwen3 (`model_type=qwen3`; config architecture `Qwen3ForCausalLM`), `hidden_size=1024`, `num_hidden_layers=28`
- **Max sequence length**: `1024`
- **Embedding dimension**: `1024`
- **Pooling**: last-token pooling (`pooling_mode_lasttoken=true`, `include_prompt=true`)
- **Normalization**: L2 normalize
- **Similarity**: cosine
- **Prompts**: a `query` prompt is defined; the `document` prompt is empty

Module stack:

```
Transformer -> Pooling(last_token, include_prompt=true) -> Normalize
```

## Usage (SentenceTransformers)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BIaoo/lca-qwen3-embedding")  # replace with your HF repo id if forked/renamed
```

Retrieval example (encode queries and documents separately; apply the built-in query prompt):

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BIaoo/lca-qwen3-embedding")  # replace with your HF repo id if forked/renamed

queries = ["wood residue gasification heat recovery"]
docs = ["Report describing small-scale biomass CHP units used for district heating."]

q = model.encode(queries, prompt_name="query", normalize_embeddings=True)
d = model.encode(docs, normalize_embeddings=True)
scores = q @ d.T  # cosine similarity (because normalized)
print(scores)
```

Notes:

- Use `prompt_name="query"` to apply the query instruction prefix from `config_sentence_transformers.json`.
- The document-side prompt is empty; encoding documents with `encode(docs, ...)` is typically sufficient.

## Intended use

- Semantic search and reranking for LCA process/flow descriptions and metadata-rich technical text
- Similarity scoring for deduplication / clustering of LCA-related passages

## Limitations

- Trained and evaluated primarily on English technical/LCA text; performance may degrade in other languages or domains.
- Evaluation numbers are from a specific internal setup; validate on your own data before production use.

## Files

- `config.json`: Qwen3 model config
- `config_sentence_transformers.json`, `modules.json`, `sentence_bert_config.json`: SentenceTransformers configs (prompts, modules, max length)
- `model.safetensors`: weights
- `tokenizer.*`, `vocab.json`, `merges.txt`: tokenizer assets
- `1_Pooling/`, `2_Normalize/`: pooling / normalization modules
