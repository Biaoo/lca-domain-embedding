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

## Conclusion

On this LCA retrieval evaluation, domain embedding fine-tuning yields clear gains in both ranking quality and coverage versus the generic baseline, suggesting domain alignment is a cost-effective approach for professional retrieval.

## License

- [MIT LICENSE](LICENSE)

## Links & Citation (placeholders)

- arXiv: TBA
- Citation: TBA
- Hugging Face: TBA
- Ollama: TBA
