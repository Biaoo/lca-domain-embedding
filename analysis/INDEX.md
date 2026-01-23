# 📖 分析文档索引

欢迎查看 `BIaoo/lca-qwen3-embedding` 模型在不同平台的部署分析。

## 🚀 快速开始

### 1️⃣ 我想快速了解结论 (5分钟)

👉 **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)**

**一句话总结**:

- ✅ vLLM/Ollama/AWS 完美一致 (99.96%+ 相似度)
- ❌ SageMaker 有严重问题 (24.93% 相似度)

### 2️⃣ 我想看完整的导航 (10分钟)

👉 **[00_START_HERE.md](00_START_HERE.md)**

包含:

- 完整文档导航
- 项目结构说明
- 常见问题解答
- 推荐部署方案

### 3️⃣ 我想看 ASCII 快速参考

👉 **[QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)**

```bash
cat QUICK_REFERENCE.txt
```

## 📚 详细分析文档

### 核心文档

| 文档 | 内容 | 阅读时间 |
|------|------|---------|
| [README.md](README.md) | 总览和快速开始 | 3分钟 |
| [FINAL_SUMMARY.md](FINAL_SUMMARY.md) | 核心发现和结论 | 5分钟 |
| [00_START_HERE.md](00_START_HERE.md) | 完整导航文档 | 10分钟 |

### 技术分析

| 文档 | 内容 | 适合人群 |
|------|------|---------|
| [SIMILARITY_ANALYSIS.md](SIMILARITY_ANALYSIS.md) | 相似度数据、统计信息 | 需要详细数据 |
| [SAGEMAKER_ANALYSIS.md](SAGEMAKER_ANALYSIS.md) | SageMaker 问题剖析 | SageMaker 用户 |
| [README_ANALYSIS.md](README_ANALYSIS.md) | 综合技术报告 | 技术深入了解 |

### 其他

| 文件 | 内容 |
|------|------|
| [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt) | ASCII 快速参考表 |

## 🛠️ 分析脚本

| 脚本 | 功能 | 用途 |
|------|------|------|
| [calculate_similarity.py](calculate_similarity.py) | 计算相似度 | 主分析脚本 |
| [visualize_similarity.py](visualize_similarity.py) | 相似度矩阵 | 可视化工具 |
| [test_sagemaker_normalization.py](test_sagemaker_normalization.py) | SageMaker 测试 | 验证归一化 |

### 运行脚本

```bash
# 计算所有平台的相似度
python calculate_similarity.py

# 查看相似度矩阵
python visualize_similarity.py

# 测试 SageMaker 归一化效果
python test_sagemaker_normalization.py
```

## 📊 核心数据

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

## 🎯 推荐方案

### ✅ 推荐使用

1. **vLLM** - 性能最优
2. **Ollama** - 最简单
3. **AWS Bedrock** - 企业级

### ❌ 不推荐

- **SageMaker (默认)** - 平台不支持 embedding 任务

## 📁 文件组织

```
analysis/
├── INDEX.md                          ← 你在这里
├── README.md                         ← 总览文档
├── 00_START_HERE.md                  ← 完整导航
├── FINAL_SUMMARY.md                  ← 快速总结
├── QUICK_REFERENCE.txt               ← ASCII 参考
│
├── SIMILARITY_ANALYSIS.md            ← 相似度分析
├── SAGEMAKER_ANALYSIS.md             ← SageMaker 分析
├── README_ANALYSIS.md                ← 综合报告
│
├── calculate_similarity.py           ← 主脚本
├── visualize_similarity.py           ← 可视化
└── test_sagemaker_normalization.py   ← 测试脚本
```

## 🔗 外部链接

- [主项目 README](../README.md)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Ollama](https://ollama.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

**建议**: 从 [FINAL_SUMMARY.md](FINAL_SUMMARY.md) 开始,然后根据需要深入其他文档。
