# 📚 Embedding 向量相似度分析 - 导航

欢迎! 这是一个关于 BIaoo/lca-qwen3-embedding 模型在不同平台上部署的完整分析。

## 🎯 快速导航

### 1️⃣ 我只想看结论

👉 **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - 5分钟了解全部关键发现

**核心结论**:

- ✅ vLLM, Ollama, AWS 三者完美一致 (99.96% 相似度)
- ❌ SageMaker 默认配置完全不可用 (24.93% 相似度)
- ⚠️ SageMaker 不支持 embedding 任务类型
- 💡 **推荐使用 vLLM 或 Ollama**

### 2️⃣ 我想看详细数据

👉 **[SIMILARITY_ANALYSIS.md](SIMILARITY_ANALYSIS.md)** - 完整相似度分析报告

包含:

- 相似度矩阵
- 欧氏距离对比
- 向量统计信息
- 归一化状态检查

### 3️⃣ 我想知道 SageMaker 为什么不行

👉 **[SAGEMAKER_ANALYSIS.md](SAGEMAKER_ANALYSIS.md)** - SageMaker 问题深度剖析

包含:

- 根本原因分析
- 技术细节解释
- 三种解决方案(含完整代码)
- 与其他平台的对比

### 4️⃣ 我想要一份完整的报告

👉 **[README_ANALYSIS.md](README_ANALYSIS.md)** - 综合分析报告

包含:

- 所有平台对比
- 性能评估
- 推荐指数
- 最佳实践

## 🛠️ 分析脚本

### 运行现有分析

```bash
# 计算四个平台的向量相似度
python calculate_similarity.py

# 可视化相似度矩阵
python visualize_similarity.py

# 测试 SageMaker 归一化效果
python test_sagemaker_normalization.py
```

### 脚本说明

| 脚本 | 功能 | 输出 |
|------|------|------|
| [calculate_similarity.py](calculate_similarity.py) | 主分析脚本 | 余弦相似度、欧氏距离、统计信息 |
| [visualize_similarity.py](visualize_similarity.py) | 相似度矩阵 | 4x4 相似度矩阵表格 |
| [test_sagemaker_normalization.py](test_sagemaker_normalization.py) | SageMaker 测试 | 归一化前后对比 |

## 📊 关键数据

### 相似度对比

```
              vllm      ollama         aws   sagemaker
vllm      1.000000    0.999858    0.999628    0.249288
ollama    0.999858    1.000000    0.999776    0.248630
aws       0.999628    0.999776    1.000000    0.248047
sagemaker 0.249288    0.248630    0.248047    1.000000
```

### 归一化状态

| 平台 | 范数 | 状态 |
|------|------|------|
| vLLM | 1.0000 | ✅ 已归一化 |
| Ollama | 1.0000 | ✅ 已归一化 |
| AWS | 0.9996 | ✅ 已归一化 |
| SageMaker | 114.07 | ❌ 未归一化 |

## 💡 推荐方案

### 🥇 最佳选择:vLLM

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.embeddings.create(
    model="BIaoo/lca-qwen3-embedding",
    input="Your text here"
)

embedding = response.data[0].embedding
```

**优点**:

- ✅ 完全兼容 OpenAI API
- ✅ 性能最优
- ✅ 输出正确(已归一化)
- ✅ 生产就绪

### 🥈 最简单:Ollama

```bash
# 安装和运行
ollama pull BIaoo/lca-qwen3-embedding
ollama embed BIaoo/lca-qwen3-embedding "Your text here"
```

**优点**:

- ✅ 一键部署
- ✅ 零配置
- ✅ 输出正确
- ✅ 适合开发测试

### 🥉 企业级:AWS Bedrock

**优点**:

- ✅ 托管服务
- ✅ 自动扩展
- ✅ 输出正确
- ⚠️ 成本较高

### ⚠️ 不推荐:SageMaker (默认)

**问题**:

- ❌ 不支持 embedding 任务类型
- ❌ 输出未归一化
- ❌ 缺少 mean pooling
- ❌ 相似度仅 25%

**如必须使用**:需要自定义 inference.py (见 [SAGEMAKER_ANALYSIS.md](SAGEMAKER_ANALYSIS.md))

## 🔍 项目结构

```
lca-embedding/
├── 00_START_HERE.md              ← 你在这里!
├── FINAL_SUMMARY.md              ← 快速总结
├── SIMILARITY_ANALYSIS.md        ← 详细数据分析
├── SAGEMAKER_ANALYSIS.md         ← SageMaker 问题剖析
├── README_ANALYSIS.md            ← 综合报告
│
├── calculate_similarity.py       ← 主分析脚本
├── visualize_similarity.py       ← 相似度矩阵
├── test_sagemaker_normalization.py ← SageMaker 测试
│
├── data/
│   ├── embedding_vllm_no_prompt.json
│   ├── embedding_ollama.json
│   ├── embedding_aws.json
│   └── embedding_sagemaker.json
│
└── deploy-serverless-model.ipynb ← SageMaker 部署脚本
```

## 🚀 下一步

1. **阅读** [FINAL_SUMMARY.md](FINAL_SUMMARY.md) 了解核心发现
2. **运行** `python calculate_similarity.py` 查看完整数据
3. **选择**合适的部署平台(推荐 vLLM)
4. **参考** [SAGEMAKER_ANALYSIS.md](SAGEMAKER_ANALYSIS.md) 如果需要使用 SageMaker

## ❓ 常见问题

### Q: 为什么 SageMaker 的相似度这么低?

A: SageMaker 的 Hugging Face 容器不支持 embedding 任务,只能使用 `feature-extraction`,返回的是原始模型输出,缺少 mean pooling 和归一化。

### Q: SageMaker 能用吗?

A: 需要自定义 inference.py 脚本才能正确使用,部署复杂度高。**不推荐**。

### Q: 哪个平台最好?

A:

- **性能优先**: vLLM
- **简单易用**: Ollama
- **企业级**: AWS Bedrock

### Q: 手动归一化 SageMaker 的输出可以吗?

A: **不行**。测试显示即使归一化后相似度仍然只有 25%,问题是缺少 mean pooling,而非归一化。

### Q: 为什么其他三个平台一致?

A: 它们都使用了标准的 embedding 处理流程(mean pooling + L2 归一化),并且都基于相同的模型权重。

## 📞 联系方式

有问题?欢迎提 Issue!

---

**分析日期**: 2026-01-12
**模型**: BIaoo/lca-qwen3-embedding
**平台数**: 4 (vLLM, Ollama, AWS, SageMaker)
