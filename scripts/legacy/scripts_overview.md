## Scripts Overview & Usage

### 0. Supabase 数据导出

`scripts/export_supabase_json_ordered.py`  
作用：连接 Supabase 的 `flows` / `processes` 表，按 `state_code` 条件导出 `json_ordered` 字段，并以 `id-version.json` 命名存入本地。  
示例命令：  

```bash
  uv run python scripts/export_supabase_json_ordered.py \
    --state-codes 3 4 \
    --output-dir data/supabase_exports
```

`scripts/step1_generate_markdown.py`  
作用：将 flow/process JSON（支持 Supabase 导出或本地 TIDAS 数据）转换为 Markdown，自动输出到 `flows/` 与 `processes/` 子目录。  
示例命令：  

```bash
uv run python scripts/step1_generate_markdown.py \
  --input-dir data/supabase_exports \
  --output-dir data/supabase_exports_markdown \
  --lang en
```

### 1. 数据准备流程

`scripts/step2_generate_queries.py`  
作用：批量调用 LLM 为每个 markdown 生成查询，支持并发、断点与批量写出，默认对 flow 数据做 1:1 随机采样并与 process 数据合并（输出包含 `dataset_type` 列）。  
示例命令：  

```bash
uv run python scripts/step2_generate_queries.py \
  --process-dir data/supabase_exports_markdown/processs \
  --flow-ratio 1.0 \
  --output-file data/lca_embedding_dataset.csv \
  --concurrency 8
```

`scripts/step2_generate_single.py`  
作用：针对指定 markdown 文件生成查询，用于补单/修复，输出同样包含 `dataset_type` 列（用 `--dataset-type flow` 标记 flow 数据）。  
示例命令：  

```bash
uv run python scripts/step2_generate_single.py \
  data/markdown/xxx.md \
  --output-file data/lca_embedding_dataset.csv
```

`scripts/prepare_ft_data.py`  
作用：将含 query/文档 的 CSV 转换为 fine-tune 所需的 `training.json` / `corpus.jsonl` / `test_queries.jsonl` / `test_qrels.jsonl`，支持传入多个 CSV，并按 `dataset_type` 对 flow/process 做 1:1 平衡。  
示例命令：  

```bash
uv run python scripts/prepare_ft_data.py \
  --data_path data/tiangong_lca_embedding_dataset.csv data/tiangong_lca_embedding_dataset.csv \
  --output_dir data/ft_data \
  --hard_negatives_path data/ft_data/hard_negatives.jsonl
```

### 2. Hard Negative 挖掘

`scripts/mine_hard_negatives.py`  
作用：使用指定 embedding 模型检索 Top‑K 文档，调用 LLM 判定 Hard Negative，支持按标题去重、并发、断点。  
示例命令：  

```bash
uv run python scripts/mine_hard_negatives.py \
  --data_dir data/ft_data \
  --model_path data/model/Qwen--Qwen3-Embedding-0.6B \
  --top_k 20 \
  --max_hard_neg 3 \
  --concurrency 8 \
  --output_path data/ft_data/hard_negatives.jsonl
```

推荐分两步执行：

1. **仅编码并缓存向量（CPU 或 空闲 GPU）**

   ```bash
   uv run python scripts/mine_hard_negatives.py \
     --data_dir data/ft_data \
     --model_path data/model/Qwen--Qwen3-Embedding-0.6B \
     --encode_device cuda:1 \
     --encode_batch_size 16 \
     --encode_only \
     --embeddings_path data/ft_data/corpus_qwen3_embeddings.npy \
     --encode_queries_path data/ft_data/query_qwen3_embeddings.npy
   ```

2. **利用缓存进行 Hard Negative 挖掘**

   ```bash
   uv run python scripts/mine_hard_negatives.py \
     --data_dir data/ft_data \
     --model_path data/model/Qwen--Qwen3-Embedding-0.6B \
     --top_k 20 \
     --max_hard_neg 3 \
     --concurrency 8 \
     --embeddings_path data/ft_data/corpus_embeddings.npy \
     --encode_queries_path data/ft_data/query_embeddings.npy \
     --output_path data/ft_data/hard_negatives.jsonl
   ```

`scripts/apply_hard_negatives.py`  
作用：把挖到的 Hard Negative 追加到已有 `training.json` 中。  
示例命令：  

```bash
uv run python scripts/apply_hard_negatives.py \
  --training_path data/ft_data/training.json \
  --hard_negatives_path data/ft_data/hard_negatives.jsonl \
  --output_path data/ft_data/training_with_hard_neg.json
```

### 3. 模型下载与相似度对比

`scripts/download_model.py`  
作用：从 Hugging Face 或 ModelScope 下载模型，支持断点续传与镜像。  
示例命令：  

```bash
uv run python scripts/download_model.py --model-id BAAI/bge-m3
```

`scripts/compare_pair_similarity.py`  
作用：对多个 query 与多个 doc 的相似度进行对比，输出 CSV。  
示例命令：  

```bash
uv run python scripts/compare_pair_similarity.py \
  --query "BOF steel LCA" \
  --doc_file data/markdown/9ed5f062-bef6-309e-b265-80773cf14d82.md \
  --models "data/model/BAAI--bge-m3,data/output/lca-bge-m3-finetuned" \
  --output_csv data/eval/steel_similarity.csv
```

### 4. Fine-tune & Evaluate

`scripts/finetune.sh`  
作用：基于 FlagEmbedding 训练 BGE-M3 模型，自动检测 `training_with_hard_neg.json`。  
示例命令：  

```bash
bash scripts/finetune.sh          # 单卡
bash scripts/finetune.sh 2        # 多卡
```

`scripts/finetune_st.sh` + `scripts/finetune_st.py`  
作用：Sentence Transformers 版本的微调，默认使用带 Hard Negative 的训练集。  
示例命令：  

```bash
bash scripts/finetune_st.sh
bash scripts/finetune_st.sh 2
```

`scripts/evaluate.py`  
作用：对 BGE 模型（原始 vs 微调）进行检索评测。  
示例命令：  

```bash
uv run python scripts/evaluate.py \
  --raw_model data/model/BAAI--bge-m3 \
  --finetuned_path data/output/lca-bge-m3-finetuned
```

`scripts/evaluate_st.py`  
作用：Sentence Transformers 模型评测。  
示例命令：  

```bash
uv run python scripts/evaluate_st.py \
  --raw_model data/model/Qwen--Qwen3-Embedding-0.6B \
  --finetuned_path data/output/lca-qwen3-embedding

uv run python scripts/evaluate_st.py --raw_model data/model/Qwen--Qwen3-Embedding-0.6B --finetuned_path data/output/lca-qwen3-embedding --batch_size 8 --corpus_chunk_size 5000 --output_path data/output/eval_results.json
```

`scripts/cache_embeddings.py`  
作用：预先用多个模型（本地 / OpenRouter）对查询与语料编码并缓存向量，方便后续复用。  
示例命令：  

```bash
uv run python scripts/cache_embeddings.py \
  --data_dir data/ft_data \
  --output_dir data/eval_cache \
  --model raw=local:data/model/Qwen--Qwen3-Embedding-0.6B \
  --model ft=local:data/output/lca-qwen3-embedding \
  --model qwen8b=openrouter:qwen/qwen3-embedding-8b \
  --model qwen4b=openrouter:qwen/qwen3-embedding-4b \
  --model bge=openrouter:baai/bge-m3 \
  --model codestral=openrouter:mistralai/codestral-embed-2505 \
  --batch_size 64

uv run python scripts/cache_embeddings.py \
  --data_dir data/ft_data \
  --output_dir data/eval_cache \
  --model raw=local:data/model/Qwen--Qwen3-Embedding-0.6B \
  --model ft=local:data/output/lca-qwen3-embedding \
  --batch_size 8
```

`scripts/evaluate_cached_embeddings.py`  
作用：加载缓存向量，对多个模型进行检索指标对比，无需重复编码。  
示例命令：  

```bash
uv run python scripts/evaluate_cached_embeddings.py \
  --cache_dir data/eval_cache \
  --data_dir data/ft_data \
  --model raw ft qwen8b qwen4b bge codestral \
  --output_path data/output/cached_eval.json
```

`scripts/upload_hf_model.py`  
作用：将本地微调好的模型目录上传到 Hugging Face Hub，自动创建仓库并支持指定私有/公开、过滤文件等。  
使用前需准备：  

- Hugging Face 访问令牌 `hf_xxx`（`huggingface-cli login` 后可省略 `--token`）；  
- 目标仓库 ID `用户名/仓库名`（脚本可自动创建，默认上传 `data/output/lca-bge-m3-finetuned`）；  
- 可选：自定义提交信息、是否私有、上传/忽略的文件模式等。  
示例命令：  

```bash
uv run python scripts/upload_hf_model.py \
  --repo-id your-name/lca-bge-m3-ft \
  --token hf_xxx \
  --commit-message "Upload bge-m3 fine-tuned checkpoint"
```

`scripts/convert_to_gguf.py`  
作用：包装 llama.cpp 的 `convert_hf_to_gguf.py` 将 Hugging Face checkpoint（本地或远程）导出为 GGUF，支持自动下载转换脚本、指定输出目录/精度，并附带 Sentence Transformers 密集层选项。  
示例命令：  

```bash
uv run python scripts/convert_to_gguf.py \
  --model-dir data/output/lca-bge-m3-finetuned \
  --out-dir data/output/gguf \
  --outtype f16 \
  --include-sentence-transformers
```

`scripts/convert_to_gguf.sh`  
作用：直接调用同级目录下的 `llama.cpp` 仓库，将微调模型转换为 GGUF（含可选量化与测试）。  
示例命令：  

```bash
LLAMA_DIR=../llama.cpp MODEL_DIR=data/output/lca-bge-m3-finetuned \
  uv run bash scripts/convert_to_gguf.sh
```

`scripts/upload_ollama_model.sh`  
作用：把已生成的 GGUF 打包成 Ollama 模型（自动生成 Modelfile、调用 `ollama create`，并可选 `ollama push`）。  
示例命令：  

```bash
GGUF_PATH=data/output/gguf/lca-qwen3-embedding-f16.gguf \
MODEL_NAME=lca-qwen3-embedding \
PUSH_TARGET=BiaoLuo/lca-qwen3-embedding \
bash scripts/tools/upload_ollama_model.sh
```

### 5. 其他辅助脚本

`scripts/step1_generate_markdown.py` / `step2_generate_queries.py` / `step2_generate_single.py`：数据生成链路。  
`scripts/fix_sensitive_content.py`：对数据集中的敏感信息进行清理。  
`scripts/mine_hard_negatives.py` / `scripts/apply_hard_negatives.py`：Hard Negative 工作流。  
`scripts/compare_pair_similarity.py`：多模型相似度对比工具。  
若需更多参数，请使用 `--help` 查看，例如 `uv run python scripts/prepare_ft_data.py --help`。
