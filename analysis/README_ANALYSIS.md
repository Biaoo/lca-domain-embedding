# Embedding 向量差异分析 - 完整报告

## 📊 快速概览

本项目分析了四个不同平台生成的 embedding 向量的相似度:

| 平台 | 相似度组 | 与 vLLM 相似度 | 范数 | 状态 |
|------|---------|---------------|------|------|
| vLLM | 高度一致 | 100% | 1.0000 | ✅ 标准 |
| Ollama | 高度一致 | 99.99% | 1.0000 | ✅ 标准 |
| AWS | 高度一致 | 99.96% | 0.9996 | ✅ 标准 |
| SageMaker | **异常** | **24.93%** | 114.07 | ⚠️ 问题 |

## 🔍 主要发现

### ✅ vLLM, Ollama, AWS 三者高度一致

这三个平台的 embedding 向量几乎完全相同 (相似度 >99.96%),说明:
- 使用相同的模型权重
- 采用标准的 embedding 处理流程
- 输出经过正确的归一化

### ⚠️ SageMaker 存在严重差异

**问题**: SageMaker 的向量与其他平台相似度仅 ~25%

**根本原因**:

1. SageMaker 的 Hugging Face 容器**不支持任何 embedding 专用任务**
2. 只能使用 `feature-extraction`,但它返回原始模型输出,缺少必要的后处理
3. **没有 embedding 相关的内置支持** (无 `sentence-similarity`, `sentence-embedding` 等)

```python
# ❌ 问题配置
hub = {
    'HF_MODEL_ID': 'BIaoo/lca-qwen3-embedding',
    'HF_TASK': 'feature-extraction'  # 这里是问题所在!
}
```

## 🧪 技术分析

### SageMaker 的问题

#### 1. 缺少关键后处理步骤

`feature-extraction` 管道返回的是**原始模型输出**,缺少:

- ❌ **Mean Pooling**: 没有对 token embeddings 进行平均池化
- ❌ **L2 Normalization**: 没有归一化向量
- ❌ **Attention Mask 处理**: 没有正确处理 padding tokens

#### 2. 数值差异明显

| 指标 | vLLM/Ollama/AWS | SageMaker (原始) |
|------|----------------|-----------------|
| 范数 | ~1.0 | 114.07 |
| 均值 | ~0.0002 | -0.1574 |
| 标准差 | ~0.0312 | 3.5611 |
| 数值范围 | [-0.11, 0.11] | [-19.16, 10.34] |

#### 3. 归一化无法解决问题

测试结果显示:
- 归一化后标准差比例接近 1.0 (0.999)
- 但相似度仍然只有 24.93%
- **说明问题不是归一化,而是使用了错误的 token 位置或缺少 pooling**

### 前10维数值对比

```
维度  vLLM         SageMaker(归一化)  差异
0    -0.001047     0.023468          0.024515
1    -0.050476    -0.124082          0.073606  ⚠️ 差异大
2    -0.006383     0.001442          0.007826
3    -0.057560    -0.055894          0.001666
7    -0.001365    -0.165387          0.164021  ⚠️ 差异极大
```

某些维度差异显著,说明 SageMaker 使用了不同的输出逻辑。

## 💡 解决方案

### 方案 1: 自定义 Inference 脚本 ⭐ (最佳)

创建 `inference.py`:

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    return model, tokenizer

def predict_fn(data, model_tokenizer):
    model, tokenizer = model_tokenizer

    # Tokenize
    inputs = tokenizer(
        data['inputs'],
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    embeddings = torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # L2 Normalization
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.tolist()
```

### 方案 2: 切换到其他平台 (推荐)

**vLLM** (最佳性能):
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1")
response = client.embeddings.create(
    model="BIaoo/lca-qwen3-embedding",
    input="Your text here"
)
```

**Ollama** (最简单):
```bash
ollama pull BIaoo/lca-qwen3-embedding
ollama embed BIaoo/lca-qwen3-embedding "Your text here"
```

### 方案 3: 客户端后处理 (临时方案)

```python
import numpy as np

response = predictor.predict(data)
embedding = np.array(response[0][0])

# L2 归一化
embedding = embedding / np.linalg.norm(embedding)
```

⚠️ **注意**: 这种方式仍然缺少 mean pooling,效果可能不理想。

## 📈 相似度矩阵

```
              vllm      ollama         aws   sagemaker
vllm      1.000000    0.999858    0.999628    0.249288
ollama    0.999858    1.000000    0.999776    0.248630
aws       0.999628    0.999776    1.000000    0.248047
sagemaker 0.249288    0.248630    0.248047    1.000000
```

## 🛠️ 使用脚本

### 1. 计算向量相似度
```bash
python calculate_similarity.py
```

### 2. 可视化相似度矩阵
```bash
python visualize_similarity.py
```

### 3. 测试 SageMaker 归一化效果
```bash
python test_sagemaker_normalization.py
```

## 📚 详细文档

- **[SIMILARITY_ANALYSIS.md](SIMILARITY_ANALYSIS.md)** - 完整的相似度分析报告
- **[SAGEMAKER_ANALYSIS.md](SAGEMAKER_ANALYSIS.md)** - SageMaker 问题深度分析和解决方案

## 🎯 结论和建议

### ✅ 对于新项目

**推荐使用**:
1. **vLLM** - 最佳性能,完全兼容 OpenAI API
2. **Ollama** - 最易部署,开箱即用
3. **AWS Bedrock/ECS** - 企业级托管服务

### ⚠️ 如果必须使用 SageMaker

1. **不要使用** `feature-extraction` 任务类型
2. **必须使用**自定义 inference 脚本
3. **或者**在客户端进行完整的后处理

### 📊 性能对比

| 平台 | 部署难度 | 推理性能 | 输出质量 | 推荐指数 |
|------|---------|---------|---------|---------|
| vLLM | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Ollama | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| AWS | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| SageMaker (默认) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ |
| SageMaker (自定义) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🔗 相关资源

- [Hugging Face Feature Extraction](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.FeatureExtractionPipeline)
- [Sentence Transformers](https://www.sbert.net/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Ollama Documentation](https://ollama.com/docs)
- [SageMaker Custom Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html)

---

**分析日期**: 2026-01-12
**模型**: BIaoo/lca-qwen3-embedding
**维度**: 1024
