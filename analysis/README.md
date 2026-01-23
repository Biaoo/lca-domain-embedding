# Embedding 向量相似度分析

对比分析 `BIaoo/lca-qwen3-embedding` 模型在 vLLM, Ollama, AWS, SageMaker 四个平台的部署效果。

## 📊 快速概览

| 平台 | 与 vLLM 相似度 | 范数 | 状态 | 推荐指数 |
|------|---------------|------|------|---------|
| vLLM | 100% | 1.0000 | ✅ 标准 | ⭐⭐⭐⭐⭐ |
| Ollama | 99.99% | 1.0000 | ✅ 标准 | ⭐⭐⭐⭐⭐ |
| AWS | 99.96% | 0.9996 | ✅ 标准 | ⭐⭐⭐⭐ |
| SageMaker | **24.93%** | 114.07 | ❌ 问题 | ❌ 不推荐 |

## 🔍 核心发现

### ✅ vLLM, Ollama, AWS 高度一致

三个平台生成的 embedding 向量相似度超过 **99.96%**,证明它们:
- 使用相同的模型实现
- 采用标准的 embedding 处理流程
- 输出正确归一化

### ❌ SageMaker 存在根本性问题

**相似度仅 24.93%** - 这不是配置问题,而是平台限制!

**根本原因**:
1. SageMaker 的 Hugging Face 容器**不支持任何 embedding 专用任务**
2. 只能使用 `feature-extraction`,返回原始模型输出
3. 缺少 mean pooling 和 L2 归一化处理

## 📚 文档导航

### 从这里开始 👇

1. **[00_START_HERE.md](00_START_HERE.md)** - 完整导航文档
2. **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - 5分钟快速总结
3. **[QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)** - ASCII 快速参考

### 详细分析

- **[SIMILARITY_ANALYSIS.md](SIMILARITY_ANALYSIS.md)** - 详细相似度数据和统计信息
- **[SAGEMAKER_ANALYSIS.md](SAGEMAKER_ANALYSIS.md)** - SageMaker 问题深度剖析(含解决方案)
- **[README_ANALYSIS.md](README_ANALYSIS.md)** - 综合分析报告

## 🛠️ 运行脚本

### 分析脚本

```bash
# 计算所有平台的相似度
python calculate_similarity.py

# 查看相似度矩阵
python visualize_similarity.py

# 测试 SageMaker 归一化效果
python test_sagemaker_normalization.py
```

### 脚本说明

| 脚本 | 功能 |
|------|------|
| [calculate_similarity.py](calculate_similarity.py) | 计算余弦相似度、欧氏距离、向量统计 |
| [visualize_similarity.py](visualize_similarity.py) | 生成 4x4 相似度矩阵 |
| [test_sagemaker_normalization.py](test_sagemaker_normalization.py) | 验证手动归一化是否有效 |

## 📊 相似度矩阵

```
              vllm      ollama         aws   sagemaker
vllm      1.000000    0.999858    0.999628    0.249288  ⚠️
ollama    0.999858    1.000000    0.999776    0.248630  ⚠️
aws       0.999628    0.999776    1.000000    0.248047  ⚠️
sagemaker 0.249288    0.248630    0.248047    1.000000
```

## 💡 推荐部署方案

### 🥇 vLLM (最佳性能)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.embeddings.create(
    model="BIaoo/lca-qwen3-embedding",
    input="Your text here"
)
embedding = response.data[0].embedding
```

**优点**: 性能最优、完全兼容 OpenAI API、开箱即用

### 🥈 Ollama (最简单)

```bash
ollama pull BIaoo/lca-qwen3-embedding
ollama embed BIaoo/lca-qwen3-embedding "Your text here"
```

**优点**: 一键部署、零配置、适合开发测试

### 🥉 AWS Bedrock (企业级)

**优点**: 托管服务、自动扩展、企业级支持

### ⚠️ SageMaker - 不推荐

**问题**:
- ❌ 不支持 embedding 任务类型
- ❌ 输出未归一化(范数 = 114)
- ❌ 缺少 mean pooling
- ❌ 相似度仅 25%

**如必须使用**: 需要编写完整的自定义 inference.py 脚本

详见: [SAGEMAKER_ANALYSIS.md](SAGEMAKER_ANALYSIS.md)

## 📁 数据文件

```
../data/
├── embedding_vllm_no_prompt.json    ✅ 标准输出 (范数=1.0)
├── embedding_ollama.json            ✅ 标准输出 (范数=1.0)
├── embedding_aws.json               ✅ 标准输出 (范数=1.0)
└── embedding_sagemaker.json         ❌ 有问题 (范数=114)
```

## 🎯 结论

1. **vLLM, Ollama, AWS 完全一致** - 可互换使用
2. **SageMaker 不支持 embedding 任务** - 是平台限制
3. **强烈推荐使用 vLLM 或 Ollama** - 性能更好且开箱即用
4. **避免使用 SageMaker 默认配置** - 除非使用自定义代码

## 📖 快速查看

想快速了解结论?运行:

```bash
cat QUICK_REFERENCE.txt
```

---

**模型**: BIaoo/lca-qwen3-embedding
**维度**: 1024
**分析日期**: 2026-01-12
**测试数据**: "This is a test sentence for embedding generation."
