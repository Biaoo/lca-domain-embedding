# SageMaker Embedding 差异分析

## 问题概述

SageMaker 生成的 embedding 向量与 vLLM、Ollama、AWS 三者存在**显著差异**:
- 余弦相似度仅约 25% (其他三者之间 >99.96%)
- 向量未归一化 (范数 = 114.07 vs 1.0)

## 部署配置分析

### SageMaker 部署脚本

```python
hub = {
    'HF_MODEL_ID': 'BIaoo/lca-qwen3-embedding',
    'HF_TASK': 'feature-extraction'
}

huggingface_model = HuggingFaceModel(
    env=hub,
    role=role,
    transformers_version='4.51.3',
    pytorch_version='2.6.0',
    py_version='py312',
)

predictor = huggingface_model.deploy(
    serverless_inference_config=serverless_config
)
```

## ⚠️ 关键发现

**SageMaker 的 Hugging Face 容器不支持任何 embedding 专用的任务类型!**

尝试使用 `sentence-similarity` 会得到以下错误:

```
Unknown task sentence-similarity, available tasks are ['audio-classification',
'automatic-speech-recognition', 'depth-estimation', 'document-question-answering',
'feature-extraction', 'fill-mask', 'image-classification', 'image-feature-extraction',
'image-segmentation', 'image-text-to-text', 'image-to-image', 'image-to-text',
'mask-generation', 'ner', 'object-detection', 'question-answering',
'sentiment-analysis', 'summarization', 'table-question-answering',
'text-classification', 'text-generation', 'text-to-audio', 'text-to-speech',
'text2text-generation', 'token-classification', 'translation', 'video-classification',
'visual-question-answering', 'vqa', 'zero-shot-audio-classification',
'zero-shot-classification', 'zero-shot-image-classification',
'zero-shot-object-detection', 'translation_XX_to_YY']
```

**这意味着**: SageMaker 的标准 Hugging Face 容器**不适合部署 embedding 模型**,必须使用自定义推理代码!

## 根本原因分析

### 1. **使用了 `feature-extraction` 任务类型** ⚠️

这是造成差异的**核心原因**(也是唯一可用的选项):

**问题所在**:
- `HF_TASK: 'feature-extraction'` 使用的是 Hugging Face 的通用特征提取管道
- 这个管道会直接返回模型最后一层的隐藏状态,**不会**经过任何后处理
- **不会进行归一化、池化或其他专门针对 embedding 模型的处理**

**对比其他平台**:
```python
# vLLM/Ollama/AWS 使用的是专门的 embedding endpoint
# 这些平台会:
# 1. 对输出进行 mean pooling
# 2. 进行 L2 归一化
# 3. 返回标准化的 embedding 向量
```

### 2. **数据格式差异**

**SageMaker 输出** (未处理的原始 logits):
```json
[[[
    2.676863193511963,
    -14.15343952178955,
    ...
]]]
```
- 数值范围: -19.16 到 10.34
- 范数: 114.07
- **这是原始的模型输出,没有经过归一化**

**其他平台输出** (标准化 embedding):
```json
{
  "embedding": [
    -0.00077003258,
    -0.05064893141,
    ...
  ]
}
```
- 数值范围: -0.11 到 0.11
- 范数: ~1.0
- **经过 L2 归一化的标准 embedding**

### 3. **后处理缺失**

SageMaker 的 `feature-extraction` 管道缺少以下关键步骤:

#### a) Mean Pooling
```python
# 其他平台会对 token embeddings 进行 mean pooling
# 得到句子级别的 representation
```

#### b) L2 Normalization
```python
# 标准的 embedding 模型需要归一化
embedding = embedding / np.linalg.norm(embedding)
```

#### c) 特殊 Token 处理
```python
# 需要正确处理 attention_mask
# 排除 [PAD] 等特殊 token 的影响
```

## 验证测试

### 向量数值对比

| 维度 | SageMaker | vLLM | Ollama | AWS |
|------|-----------|------|--------|-----|
| 第1维 | 2.6769 | -0.0010 | -0.0008 | -0.0010 |
| 第2维 | -14.1534 | -0.0505 | -0.0506 | -0.0522 |
| 第3维 | 0.1645 | -0.0064 | -0.0064 | -0.0061 |
| 范数 | 114.07 | 1.0000 | 1.0000 | 0.9996 |

### 归一化后对比

即使对 SageMaker 向量进行归一化,与其他平台的相似度仍然只有 25%,说明:
1. 使用的是不同的输出层
2. 缺少 pooling 等关键处理步骤
3. 可能取的是错误的 token 位置

## 解决方案

### 方案 1: 使用自定义推理代码 ⭐ (推荐)

创建自定义的 `inference.py`:

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
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # L2 Normalization
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.tolist()
```

然后部署:
```python
huggingface_model = HuggingFaceModel(
    model_data='s3://your-bucket/model.tar.gz',  # 包含 inference.py
    role=role,
    transformers_version='4.51.3',
    pytorch_version='2.6.0',
    py_version='py312',
)
```

### 方案 2: 客户端后处理

如果无法修改部署,可以在客户端进行后处理:

```python
import numpy as np

response = predictor.predict(data)
embedding = np.array(response[0][0])

# L2 归一化
embedding = embedding / np.linalg.norm(embedding)
```

**注意**: 这种方式可能仍然缺少 mean pooling,效果可能不如方案 1。

### ~~方案 3: 使用 Sentence Transformers~~ ❌

**已验证不可行!** SageMaker 的 Hugging Face 容器**不支持** `sentence-similarity` 或 `sentence-embedding` 任务。

错误信息:
```
Unknown task sentence-similarity, available tasks are ['audio-classification',
'automatic-speech-recognition', 'depth-estimation', 'document-question-answering',
'feature-extraction', 'fill-mask', 'image-classification', ...
'text-generation', 'text-to-audio', 'text-to-speech', 'text2text-generation',
'token-classification', 'translation', ...]
```

**注意**: 可用任务列表中**没有任何 embedding 相关的任务类型**,这就是为什么只能使用 `feature-extraction`,导致输出不正确。

## 为什么其他平台一致?

### vLLM
- 专门为 embedding 模型优化
- 内置 mean pooling 和归一化
- 遵循 OpenAI embedding API 标准

### Ollama
- 使用相同的 embedding 处理管道
- 自动进行归一化
- 标准化的输出格式

### AWS (Bedrock/ECS)
- 使用标准的 embedding endpoint
- 遵循 AWS embedding 规范
- 内置后处理逻辑

## 结论

SageMaker 的差异**不是 bug**,而是因为:

1. ❌ `feature-extraction` 是通用任务,返回原始模型输出
2. ❌ 缺少 embedding 模型特定的后处理
3. ❌ 没有进行归一化和 pooling

**建议**:
- ✅ 使用自定义 inference 代码 (方案 1)
- ✅ 或切换到其他平台 (vLLM/Ollama 更适合 embedding 任务)
- ✅ 如果必须使用 SageMaker,至少在客户端进行归一化处理

## 参考资料

- [Hugging Face Feature Extraction Pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.FeatureExtractionPipeline)
- [Sentence Transformers Pooling](https://www.sbert.net/docs/usage/computing_sentence_embeddings.html)
- [SageMaker Custom Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html)
