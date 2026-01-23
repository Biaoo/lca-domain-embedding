#!/usr/bin/env python3
"""
测试 SageMaker 向量手动归一化后的效果
"""

import json
import numpy as np
from pathlib import Path


def load_embedding(file_path):
    """从JSON文件加载embedding向量"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    if 'data' in data:
        return np.array(data['data'][0]['embedding'])
    elif 'embeddings' in data:
        return np.array(data['embeddings'][0])
    elif 'embedding' in data:
        return np.array(data['embedding'])
    elif isinstance(data, list):
        vec = data
        while isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list):
            vec = vec[0]
        return np.array(vec)
    else:
        raise ValueError(f"无法识别的文件格式: {file_path}")


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def main():
    base_dir = Path(__file__).parent / 'data'

    # 加载向量
    print("加载向量...")
    vllm = load_embedding(base_dir / 'embedding_vllm_no_prompt.json')
    sagemaker = load_embedding(base_dir / 'embedding_sagemaker.json')

    print(f"vLLM 向量: 范数={np.linalg.norm(vllm):.6f}")
    print(f"SageMaker 原始: 范数={np.linalg.norm(sagemaker):.6f}")
    print()

    # 归一化 SageMaker 向量
    sagemaker_normalized = sagemaker / np.linalg.norm(sagemaker)
    print(f"SageMaker 归一化后: 范数={np.linalg.norm(sagemaker_normalized):.6f}")
    print()

    # 计算相似度
    print("="*70)
    print("相似度对比")
    print("="*70)

    sim_original = cosine_similarity(vllm, sagemaker)
    sim_normalized = cosine_similarity(vllm, sagemaker_normalized)

    print(f"\nvLLM vs SageMaker (原始):     {sim_original:.6f}")
    print(f"vLLM vs SageMaker (归一化后): {sim_normalized:.6f}")

    print("\n" + "="*70)
    print("结论")
    print("="*70)

    if abs(sim_original - sim_normalized) < 0.001:
        print("✓ 归一化对余弦相似度没有影响 (符合预期)")
        print("  原因: 余弦相似度本身就对向量长度不敏感")

    if sim_normalized < 0.5:
        print("\n✗ 即使归一化后相似度仍然很低 (<50%)")
        print("\n这说明问题不仅仅是归一化,而是:")
        print("  1. SageMaker 使用了不同的输出层或 token 位置")
        print("  2. 缺少 mean pooling 等关键后处理步骤")
        print("  3. 可能使用了不同的推理逻辑")
        print("\n建议:")
        print("  - 使用自定义 inference.py 脚本 (见 SAGEMAKER_ANALYSIS.md)")
        print("  - 或切换到 vLLM/Ollama 等专门优化的平台")

    # 数值分布对比
    print("\n" + "="*70)
    print("数值分布对比")
    print("="*70)

    print(f"\nvLLM:")
    print(f"  均值:   {np.mean(vllm):.6f}")
    print(f"  标准差: {np.std(vllm):.6f}")
    print(f"  范围:   [{np.min(vllm):.6f}, {np.max(vllm):.6f}]")

    print(f"\nSageMaker (原始):")
    print(f"  均值:   {np.mean(sagemaker):.6f}")
    print(f"  标准差: {np.std(sagemaker):.6f}")
    print(f"  范围:   [{np.min(sagemaker):.6f}, {np.max(sagemaker):.6f}]")

    print(f"\nSageMaker (归一化后):")
    print(f"  均值:   {np.mean(sagemaker_normalized):.6f}")
    print(f"  标准差: {np.std(sagemaker_normalized):.6f}")
    print(f"  范围:   [{np.min(sagemaker_normalized):.6f}, {np.max(sagemaker_normalized):.6f}]")

    # 检查数值分布的相似性
    print("\n" + "="*70)
    print("分布相似性检查")
    print("="*70)

    # 比较标准差比例
    std_ratio = np.std(sagemaker_normalized) / np.std(vllm)
    print(f"\n标准差比例 (SageMaker/vLLM): {std_ratio:.3f}")

    if abs(std_ratio - 1.0) > 0.1:
        print("✗ 标准差差异显著,说明数值分布不同")
    else:
        print("✓ 标准差接近,数值分布相似")

    # 前10个维度对比
    print("\n" + "="*70)
    print("前10个维度数值对比")
    print("="*70)
    print(f"\n{'维度':<6} {'vLLM':<12} {'SageMaker(归一化)':<20} {'差异':<12}")
    print("-"*70)
    for i in range(10):
        diff = abs(vllm[i] - sagemaker_normalized[i])
        print(f"{i:<6} {vllm[i]:<12.6f} {sagemaker_normalized[i]:<20.6f} {diff:<12.6f}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
