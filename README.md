# LCA Domain Embedding Project

面向生命周期评价（LCA）检索场景的向量模型微调与评测项目。涵盖数据生成、模型微调、向量缓存、统一评测与报告输出。README 聚焦项目说明与结构，不包含使用步骤。

## 项目概述

- 目标：在 LCA 专业检索中构建并验证领域向量模型，量化其相对通用与云端模型的排序与覆盖收益。
- 产出：微调模型、缓存向量及评测结果、可视化报告（中/英文）。

## 数据与规模

- 来源：TianGong LCA，原始 Tidas 结构转 Markdown；LLM 生成查询。
- 清洗：doc_id 统一（uuid[|version]），(query, doc_id) 去重，文档层面去重；负采样 10 条/查询，可选 hard negatives；固定种子划分。
- 规模：训练集 17,037 条 query-doc；评测集 1,893 查询 / 3,786 语料 / 1,893 qrels。

## 方法与流水线（概览）

1) 查询生成与数据准备：Tidas -> Markdown -> 查询，去重、负采样、划分，导出训练/评测文件。
2) 模型微调：Qwen3-Embedding-0.6B，对比学习（MultipleNegativesRankingLoss），bf16/单多卡兼容。
3) 向量缓存：一次性编码 queries/corpus，保存向量、ID 映射、meta，减少重复计算。
4) 评测：Faiss 内积 Flat 检索 Top-100，pytrec_eval 计算 NDCG/MAP/Recall/Precision，另算 MRR。
5) 报告：汇总指标、图表（关键指标条形图、Recall 曲线），输出 HTML/MD（EN & ZH）。

## 模型说明

- raw：Qwen3-Embedding-0.6B 通用嵌入。
- ft：在 LCA 域数据上微调的 Qwen3-Embedding-0.6B。
- 云端对照：qwen3-embedding-8b / 4b，bge-m3，codestral-embed-2505。

## 指标与结果要点

- 指标集合：NDCG/MAP/Recall/Precision/MRR @ {1,5,10,50,100}，查询级平均。
- 关键结论：微调模型在头部与长尾均显著优于 raw 与云端模型（如 NDCG@10、Recall@10、Recall@100 均明显提升）。

## 模型效果（摘要）
- 相对原始模型（raw），微调模型（ft）提升：NDCG@10 +31.2%，Recall@10 +25.7%，MRR@10 +33.5%；Recall@100 +11.5%。
- 云端对照中 codestral 次优，但仍落后微调模型；其他云端模型仅带来边际增益。

## 目录结构

- scripts/
  - pipeline/：数据→训练→缓存→评测的编号脚本（01–07）
  - reports/：报告生成脚本
  - tools/：hard negatives、相似度、模型下载/上传、转换等工具
  - legacy/：旧入口（保留兼容）
- src/：可复用模块
  - embed/（缓存/编码），eval/（指标），prep/（数据准备），report/（渲染），train/、data/
- data/：ft_data、eval_cache、output（评测结果等）
- docs/：模板与历史报告；根目录 `report.md`（EN）、`report.ZH.md`（ZH）

## 依赖与环境

- 关键依赖：sentence-transformers、faiss、pytrec_eval、datasets、requests、numpy、pandas、tqdm 等。
- 云端模型：OpenRouter 需配置 `OPENROUTER_API_KEY`；bf16 支持视硬件而定。
- 从仓库根目录运行脚本以保证 `src` 可导入。

## 许可
- License：待补。

## 链接与引用（占位，后续补充）
- arXiv 链接：待补
- 引用格式：待补
- Hugging Face / Ollama 用法：待补
