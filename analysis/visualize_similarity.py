#!/usr/bin/env python3
"""
可视化embedding向量的相似度矩阵
"""

import json
import numpy as np
from pathlib import Path


def load_embedding(file_path):
    """从JSON文件加载embedding向量"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 根据不同的文件结构提取embedding
    if 'data' in data:
        # vllm格式
        return np.array(data['data'][0]['embedding'])
    elif 'embeddings' in data:
        # aws格式
        return np.array(data['embeddings'][0])
    elif 'embedding' in data:
        # ollama格式
        return np.array(data['embedding'])
    elif isinstance(data, list):
        # sagemaker格式: [[[embedding]]]
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
    # 定义文件路径
    base_dir = Path(__file__).parent / 'data'
    files = {
        'vllm': base_dir / 'embedding_vllm_no_prompt.json',
        'ollama': base_dir / 'embedding_ollama.json',
        'aws': base_dir / 'embedding_aws.json',
        'sagemaker': base_dir / 'embedding_sagemaker.json'
    }

    # 加载所有embedding
    embeddings = {}
    names = list(files.keys())

    for name, file_path in files.items():
        embeddings[name] = load_embedding(file_path)

    # 创建相似度矩阵
    n = len(names)
    similarity_matrix = np.zeros((n, n))

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            similarity_matrix[i, j] = cosine_similarity(
                embeddings[name1],
                embeddings[name2]
            )

    # 打印相似度矩阵
    print("\n" + "="*70)
    print("余弦相似度矩阵 (Cosine Similarity Matrix)")
    print("="*70)
    print()

    # 打印表头
    header = "          "
    for name in names:
        header += f"{name:>12}"
    print(header)
    print("-" * 70)

    # 打印每一行
    for i, name1 in enumerate(names):
        row = f"{name1:10}"
        for j in range(n):
            sim = similarity_matrix[i, j]
            if i == j:
                row += f"{'1.000000':>12}"
            else:
                row += f"{sim:>12.6f}"
        print(row)

    print()
    print("="*70)
    print("\n注意事项:")
    print("- 值为 1.000000: 向量与自身比较")
    print("- 值接近 1.0: 高度相似")
    print("- 值接近 0.0: 不相关")
    print("- 值为负数: 方向相反")
    print()

    # 检查归一化
    print("="*70)
    print("向量归一化检查")
    print("="*70)
    for name in names:
        norm = np.linalg.norm(embeddings[name])
        is_normalized = abs(norm - 1.0) < 0.01
        status = "✓ 已归一化" if is_normalized else "✗ 未归一化"
        print(f"{name:10}: 范数={norm:>10.6f}  {status}")

    print("\n结论:")
    print("-" * 70)
    if similarity_matrix[0, 1] > 0.99 and similarity_matrix[0, 2] > 0.99:
        print("✓ vllm, ollama, aws 三者高度一致 (>99.9%)")

    if similarity_matrix[0, 3] < 0.5:
        print("✗ sagemaker 与其他三者差异巨大 (<50%)")
        print("  可能原因:")
        print("  1. sagemaker 的向量未归一化")
        print("  2. 使用了不同的模型或配置")
        print("  3. 数据格式解析错误")

    print("="*70)


if __name__ == '__main__':
    main()
