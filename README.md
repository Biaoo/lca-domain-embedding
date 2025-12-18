# LCA Retrieval Domain Embedding Fine-tuning

This project asks one question: for professional Lifecycle Assessment (LCA) retrieval, how much does domain embedding fine-tuning help?

## Background & Goal

Generic embedding models work well in open domains, but LCA retrieval often requires domain constraints (e.g., geography/technology/time) and long, multi-field documents. We use a unified evaluation setup to quantify the impact of domain fine-tuning versus a generic baseline and cloud embedding models.

## Data Scale (this experiment)

- Data source: TianGong LCA data. The original records are in the Tidas structured format (process/flow plus metadata fields), and are converted into a retrieval-friendly text form for building the evaluation set.
- Train: 17,037 query-doc pairs
- Eval: 1,893 queries / 3,786 corpus docs / 1,893 qrels

## Comparison Setup (high level)

Models compared:

- `raw`: `Qwen3-Embedding-0.6B` (generic baseline)
- `ft`: `Qwen3-Embedding-0.6B` fine-tuned on LCA data
- Cloud baselines: `qwen3-embedding-8b` / `qwen3-embedding-4b`, `bge-m3`, `codestral-embed-2505`

## Metrics & Results Highlights

- Metrics: NDCG / MAP / Recall / Precision / MRR @ `{1,5,10,50,100}` (averaged per query)
- Summary: `ft` shows significant improvements over `raw` on both head ranking and tail coverage

## Model Effect (summary)

- `ft` vs `raw`: NDCG@10 +31.2%, Recall@10 +25.7%, MRR@10 +33.5%; Recall@100 +11.5%.

## Released Models & Usage

### Hugging Face (SentenceTransformers)

- Model: <https://huggingface.co/BIaoo/lca-qwen3-embedding>

```bash
pip install -U sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BIaoo/lca-qwen3-embedding")
emb = model.encode(["wood residue gasification heat recovery"], prompt_name="query", normalize_embeddings=True)
print(emb.shape)
```

### Ollama

- Model: <https://ollama.com/BiaoLuo/lca-qwen3-embedding>

```bash
ollama pull BiaoLuo/lca-qwen3-embedding
```

Get an embedding via the local Ollama server API:

```bash
curl http://localhost:11434/api/embeddings \
  -d '{"model":"BiaoLuo/lca-qwen3-embedding","prompt":"wood residue gasification heat recovery"}'
```

## Visualization (Embedding Atlas)

Embedding Atlas (<https://apple.github.io/embedding-atlas>) provides interactive embedding visualization: clustering, labels, cross-filtering on metadata, and nearest-neighbor search. In this repo there are two practical ways to use it:

### Option A: Compute embeddings from text (simplest)

Run Embedding Atlas on the eval queries + corpus and point it at your fine-tuned model:

```bash
pip install -U embedding-atlas
embedding-atlas data/ft_data/test_queries.jsonl data/ft_data/corpus.jsonl \
  --text text \
  --model BIaoo/lca-qwen3-embedding \
  --trust-remote-code \
  --umap-metric cosine \
  --umap-random-state 42
```

Tip: when loading multiple inputs, Embedding Atlas automatically adds a `FILE_NAME` column so you can filter query vs corpus points.

### Option B: Reuse cached embeddings (easy raw vs ft comparison)

1) Cache embeddings via `scripts/pipeline/05_cache_embeddings.py`  
2) Export an Embedding Atlas dataset with `vector_raw` / `vector_ft` columns:

```bash
.venv/bin/python scripts/tools/export_embedding_atlas_dataset.py \
  --data_dir data/ft_data \
  --cache_dir data/eval_cache \
  --model raw --model ft \
  --out data/output/embedding_atlas/lca_eval.parquet
```

Then launch Embedding Atlas with the vector column you want:

```bash
embedding-atlas data/output/embedding_atlas/lca_eval.parquet --vector vector_ft --text text --umap-metric cosine --umap-random-state 42
embedding-atlas data/output/embedding_atlas/lca_eval.parquet --vector vector_raw --text text --umap-metric cosine --umap-random-state 42
```

## Conclusion

On this LCA retrieval evaluation, domain embedding fine-tuning yields clear gains in both ranking quality and coverage versus the generic baseline, suggesting domain alignment is a cost-effective approach for professional retrieval.

## License

- [MIT LICENSE](LICENSE)

## Links & Citation (placeholders)

- arXiv: TBA
- Citation: TBA
- Hugging Face: <https://huggingface.co/BIaoo/lca-qwen3-embedding>
- Ollama: <https://ollama.com/BiaoLuo/lca-qwen3-embedding>
