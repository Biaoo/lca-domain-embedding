# 🎯 Embedding 向量差异分析 - 最终总结

## 核心发现 💡

### ✅ vLLM, Ollama, AWS 完美一致

三个平台生成的 embedding 向量相似度超过 **99.96%**,证明它们:

- 使用相同的模型实现
- 采用标准的 embedding 处理流程
- 输出正确归一化

### ❌ SageMaker 存在根本性问题

**相似度仅 24.93%** - 这不是配置问题,而是平台限制!

## 🚨 关键发现:SageMaker 不支持 Embedding 任务

通过实际测试发现,SageMaker 的 Hugging Face 容器**根本不支持 embedding 相关的任务类型**:

```
Error: Unknown task sentence-similarity

Available tasks: ['audio-classification', 'automatic-speech-recognition',
'depth-estimation', 'document-question-answering', 'feature-extraction',
'fill-mask', 'image-classification', 'image-feature-extraction',
'image-segmentation', 'ner', 'object-detection', 'question-answering',
'sentiment-analysis', 'summarization', 'text-classification',
'text-generation', 'token-classification', 'translation', ...]
```

**注意**: 列表中**没有**:

- ❌ `sentence-similarity`
- ❌ `sentence-embedding`
- ❌ `embedding`
- ❌ 任何 embedding 相关的任务

## 🔍 为什么 `feature-extraction` 不行?

`feature-extraction` 是 SageMaker 中**唯一可用的选项**,但它:

1. **返回原始模型输出** (token embeddings 或 [CLS] token)
2. **不进行 mean pooling** (无法聚合 token embeddings)
3. **不进行 L2 归一化** (范数 = 114 而不是 1.0)
4. **不处理 attention mask** (padding tokens 影响结果)

### 数据证据

| 平台 | 数值范围 | 范数 | 均值 | 标准差 |
|------|---------|------|------|--------|
| vLLM/Ollama/AWS | [-0.11, 0.11] | 1.0 | ~0.0002 | ~0.03 |
| SageMaker | [-19.16, 10.34] | 114.07 | -0.157 | 3.56 |

### 归一化测试结果

即使手动归一化后:

- ✓ 标准差比例正常 (0.999)
- ✗ **相似度仍然只有 24.93%**
- ✗ 某些维度差异极大 (最大差异 0.164)

**结论**: 问题不是归一化,而是**使用了错误的输出层或缺少 pooling**。

## 📊 相似度矩阵

```
              vllm      ollama         aws   sagemaker
vllm      1.000000    0.999858    0.999628    0.249288  ⚠️
ollama    0.999858    1.000000    0.999776    0.248630  ⚠️
aws       0.999628    0.999776    1.000000    0.248047  ⚠️
sagemaker 0.249288    0.248630    0.248047    1.000000
```

## 💡 解决方案

### ⭐ 推荐方案:切换平台

**不要使用 SageMaker 的标准 Hugging Face 容器部署 embedding 模型!**

改用:

1. **vLLM** (最佳性能)
   - 完全兼容 OpenAI API
   - 内置 embedding 优化
   - 开箱即用

2. **Ollama** (最简单)
   - 一键部署
   - 自动处理 embedding
   - 无需配置

3. **AWS Bedrock/ECS** (企业级)
   - 托管服务
   - 标准 embedding API
   - 自动缩放

### 🔧 如果必须用 SageMaker

**唯一可行方案**:自定义 inference 脚本

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

    inputs = tokenizer(
        data['inputs'],
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

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

**部署复杂度**: ⭐⭐⭐⭐⭐ (需要打包模型、上传 S3、配置容器等)

## 🎯 平台对比与推荐

| 平台 | 输出正确性 | 部署难度 | 性能 | 成本 | 推荐指数 |
|------|-----------|---------|------|------|---------|
| **vLLM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Ollama** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **AWS Bedrock** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **SageMaker (默认)** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ❌ **不推荐** |
| **SageMaker (自定义)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

## 📁 完整文档

1. **[calculate_similarity.py](calculate_similarity.py)** - 主分析脚本
2. **[visualize_similarity.py](visualize_similarity.py)** - 相似度矩阵可视化
3. **[test_sagemaker_normalization.py](test_sagemaker_normalization.py)** - SageMaker 归一化测试
4. **[SIMILARITY_ANALYSIS.md](SIMILARITY_ANALYSIS.md)** - 详细相似度分析
5. **[SAGEMAKER_ANALYSIS.md](SAGEMAKER_ANALYSIS.md)** - SageMaker 问题深度剖析
6. **[README_ANALYSIS.md](README_ANALYSIS.md)** - 综合分析报告

## 🚀 快速开始

### 运行分析脚本

```bash
# 计算所有平台的相似度
python calculate_similarity.py

# 查看相似度矩阵
python visualize_similarity.py

# 测试 SageMaker 归一化效果
python test_sagemaker_normalization.py
```

### 部署正确的 Embedding 服务

**推荐:使用 vLLM**

```bash
# 安装
pip install vllm

# 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model BIaoo/lca-qwen3-embedding \
    --port 8000

# 使用
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.embeddings.create(
    model="BIaoo/lca-qwen3-embedding",
    input="Your text here"
)
print(response.data[0].embedding)
```

## 📝 结论

1. **✅ vLLM, Ollama, AWS 三者完美一致**,可互换使用
2. **❌ SageMaker 默认方式完全不可用**,相似度仅 25%
3. **⚠️ SageMaker 不支持 embedding 任务类型**,这是平台限制
4. **💡 强烈建议使用 vLLM 或 Ollama**,开箱即用且性能更好
5. **🔧 如必须用 SageMaker**,需要编写完整的自定义推理代码

## 🔗 相关资源

- [vLLM Documentation](https://docs.vllm.ai/)
- [Ollama](https://ollama.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [SageMaker Custom Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html)

---

**分析完成日期**: 2026-01-12
**模型**: BIaoo/lca-qwen3-embedding
**向量维度**: 1024
**分析文件数**: 4 (vLLM, Ollama, AWS, SageMaker)
