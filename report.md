# Domain Embedding Fine-tuning for LCA Retrieval and Evaluation

## Abstract

We build an end-to-end pipeline for lifecycle assessment (LCA) retrieval: converting raw Tidas data to Markdown and queries, fine-tuning Qwen3-Embedding-0.6B with contrastive learning, and evaluating on cached embeddings. The fine-tuned model outperforms the base and several cloud embedding models on both head and tail metrics, indicating domain fine-tuning effectively corrects generic sorting biases and improves recall.

## 1. Introduction

Generic embedding models perform well in open domains but show domain bias in LCA: a query like “steel” should prefer production or geography-related documents, yet generic models may mix in alloy/near-synonyms. LCA documents are long and multi-field (geography, technology, time), so domain-aware sorting is required. We explore domain fine-tuning and a unified evaluation pipeline to quantify gains and ensure reproducibility.

**Why domain embeddings matter**  
LLMs provide general reasoning but lack long-term domain memory; they rely on external context. A domain embedding model serves as an “external memory” index to retrieve professional knowledge quickly and accurately, enabling the “general reasoning (LLM) + domain retrieval (embedding)” synergy akin to human “broad cognition + deep expertise.”

## 2. Data and Preprocessing

- **Source/format**: TianGong LCA in Tidas format, converted to Markdown with process/flow/metadata (UUID, version, geography, technology).  
- **Scale**: Training set 17,037 query-doc pairs; eval set: 1,893 queries, 3,786 corpus docs, 1,893 qrels.  
- **Query generation**: LLM-generated queries per document, keeping query + dataset_uuid|version.  
- **Alignment/dedup**: `doc_id = dataset_uuid[|version]`; dedup (query, doc_id); dedup docs to form corpus.  
- **Negative sampling**: 10 negatives per query from different doc_id; optional hard negatives.  
- **Split/output**: fixed seed, test_size=0.1; outputs: training.json, test_queries.jsonl, corpus.jsonl, test_qrels.jsonl.

## 3. Model and Training

- **Backbone**: Qwen3-Embedding-0.6B.  
- **Objective**: MultipleNegativesRankingLoss, max_seq_length=1024.  
- **Hyperparams**: epochs=2, lr=1e-5, batch_size=8, warmup=0.1, weight_decay=0.01, bf16 (fallback fp32).  
- **Strategy**: single GPU disables auto DP; multi-GPU via accelerate. Output saved locally for encoding/eval.

## 4. Encoding and Evaluation Protocol

- **Caching**: one-time encoding of queries/corpus, storing vectors, IDs, meta per model alias.  
- **Retrieval**: Faiss inner-product Flat, Top-100, consistent across models.  
- **Metrics**: NDCG/MAP/Recall/Precision/MRR @ {1,5,10,50,100}, averaged over queries.  
- **Baselines**: raw (base Qwen3-Embedding-0.6B), ft (fine-tuned), cloud models qwen3-embedding-8b/4b, bge-m3, codestral.

## 5. Model Notes

- **raw**: ~0.7B multilingual embedding, contrastive pretraining, good for short/medium text.  
- **ft**: Qwen3-Embedding-0.6B fine-tuned on TianGong LCA to align domain ranking.  
- **qwen3-embedding-8b / 4b (cloud)**: larger-capacity Qwen3 embeddings, multilingual, longer context; 8B > 4B in representation power.  
- **bge-m3 (cloud)**: BAAI multilingual embedding, cross-domain robustness.  
- **codestral-embed-2505 (cloud)**: Mistral embedding for code/technical text, also strong on general retrieval.

## 6. Metrics (meaning)

- **NDCG@K**: ranking quality with position discounts.  
- **MAP@K**: average precision over top K.  
- **MRR@K**: earliest hit position (reciprocal rank).  
- **Recall@K**: coverage of relevant docs.  
- **Precision@K**: purity of the top-K results.  
Head metrics (NDCG/MRR@10) emphasize early precision; Recall@50/100 captures tail coverage.

## 7. Results

**Key metrics (@10)**

| Model | NDCG@10 | Recall@10 | MRR@10 | MAP@10 |
| --- | --- | --- | --- | --- |
| raw | 0.5808 | 0.7200 | 0.5367 | 0.5367 |
| ft | **0.7623** | **0.9049** | **0.7163** | **0.7163** |
| codestral | 0.6628 | 0.8045 | 0.6180 | 0.6180 |
| qwen8b | 0.5905 | 0.7369 | 0.5442 | 0.5442 |
| qwen4b | 0.5836 | 0.7290 | 0.5377 | 0.5377 |
| bge | 0.5839 | 0.7264 | 0.5388 | 0.5388 |

**Tail (@100)**

| Model | NDCG@100 | Recall@100 |
| --- | --- | --- |
| raw | 0.6171 | 0.8922 |
| ft | **0.7826** | **0.9947** |
| codestral | 0.6872 | 0.9171 |
| qwen8b | 0.6258 | 0.9033 |
| qwen4b | 0.6164 | 0.8822 |
| bge | 0.6156 | 0.8743 |

## 8. Analysis

1) Fine-tuning boosts both head and tail (NDCG@10 +31.2%, Recall@10 +25.7%, MRR@10 +33.5%; Recall@100 +11.5% vs raw), reducing generic bias.  
2) Cloud baselines: codestral is the best among cloud models but still behind the fine-tuned model; others offer only marginal gains over raw.  
3) Coverage + ranking: improved Recall@100 shows better tail coverage; higher NDCG/MRR indicates earlier correct hits—suitable for professional retrieval.

## 9. Conclusion and Future Work

The fine-tuned LCA embedding is the best-performing retrieval backend, validating the “LLM reasoning + domain memory” pattern. Future work: stronger hard negatives / adversarial perturbations; larger backbones with quantization/normalization for deployment; statistical significance tests and scenario-wise evaluations; exploring dynamic/online updates and cross-domain transfer to keep domain retrieval precise over time.
