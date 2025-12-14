# LCA 领域向量模型微调与评测项目

本项目面向 LCA 检索场景，涵盖数据生成、模型微调、向量缓存、统一评测与报告输出。README.ZH 作为对外概述，不包含具体使用步骤。

## 项目概述
- 目标：在 LCA 专业检索中构建并验证领域向量模型，量化相对通用与云端模型的排序与覆盖收益。
- 产出：微调模型、缓存向量及评测结果、可视化报告（中/英文）。

## 数据与规模
- 来源：TianGong LCA，Tidas 结构转 Markdown；LLM 生成查询。
- 清洗：doc_id 统一（uuid[|version]），(query, doc_id) 去重；文档去重；负采样 10 条/查询，可选 hard negatives；固定种子划分。
- 规模：训练集 17,037 条 query-doc；评测集 1,893 查询 / 3,786 语料 / 1,893 qrels。

## 方法与流程
1) 查询生成与数据准备：Tidas -> Markdown -> 查询，去重、负采样、划分，导出训练/评测文件。
2) 模型微调：Qwen3-Embedding-0.6B，对比学习（MultipleNegativesRankingLoss），bf16/单多卡兼容。
3) 向量缓存：一次性编码 queries/corpus，保存向量、ID、meta，减少重复计算。
4) 评测：Faiss 内积 Flat 检索 Top-100，pytrec_eval 计算 NDCG/MAP/Recall/Precision，另算 MRR。
5) 报告：关键指标与可视化（条形图、Recall 曲线），输出 HTML/MD（中/英）。

## 模型说明
- raw：Qwen3-Embedding-0.6B 通用嵌入。
- ft：在 LCA 域数据上微调的 Qwen3-Embedding-0.6B。
- 云端对照：qwen3-embedding-8b / 4b，bge-m3，codestral-embed-2505。

## 指标与结果要点
- 指标：NDCG/MAP/Recall/Precision/MRR @ {1,5,10,50,100}，查询级平均。
- 结论：微调模型在头部与长尾均显著优于 raw 与云端模型（NDCG@10 / Recall@10 / Recall@100 等明显提升）。

## 模型效果（摘要）
- 相对 raw，ft 提升：NDCG@10 +31.2%，Recall@10 +25.7%，MRR@10 +33.5%；Recall@100 +11.5%。
- 云端对照中 codestral 次优，但仍落后 ft；其余仅边际增益。

## 目录结构
- scripts/（pipeline/reports/tools/legacy）
- src/（embed/eval/prep/report/...）
- data/（ft_data, eval_cache, output 等）
- docs/（模板、历史报告）；根目录 report.md（EN）/ report.ZH.md（ZH）

## 依赖与环境
- 关键依赖：sentence-transformers、faiss、pytrec_eval、datasets、requests、numpy、pandas、tqdm 等。
- 云端模型：OpenRouter 需 `OPENROUTER_API_KEY`；bf16 视硬件支持。
- 请从仓库根目录运行脚本以保证 src 可导入。

## 许可
- License: MIT

## 链接与引用（占位）
- arXiv：待补
- 引用格式：待补
- Hugging Face / Ollama：待补
