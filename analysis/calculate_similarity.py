#!/usr/bin/env python3
"""
计算三个embedding文件之间的向量相似度
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


def euclidean_distance(vec1, vec2):
    """计算两个向量的欧氏距离"""
    return np.linalg.norm(vec1 - vec2)


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
    for name, file_path in files.items():
        try:
            embeddings[name] = load_embedding(file_path)
            print(f"✓ 成功加载 {name}: 维度 {len(embeddings[name])}")
        except Exception as e:
            print(f"✗ 加载 {name} 失败: {e}")
            return

    print("\n" + "="*60)
    print("向量相似度分析结果")
    print("="*60)

    # 检查是否需要归一化
    print("\n【归一化检查】")
    print("-"*60)
    normalized_embeddings = {}
    for name, vec in embeddings.items():
        norm = np.linalg.norm(vec)
        is_normalized = abs(norm - 1.0) < 0.01
        status = "✓ 已归一化" if is_normalized else "✗ 未归一化 (将自动归一化)"
        print(f"{name:10}: 范数={norm:>10.6f}  {status}")

        # 归一化向量
        if norm > 0:
            normalized_embeddings[name] = vec / norm
        else:
            normalized_embeddings[name] = vec

    # 计算所有配对的相似度
    pairs = [
        ('vllm', 'ollama'),
        ('vllm', 'aws'),
        ('vllm', 'sagemaker'),
        ('ollama', 'aws'),
        ('ollama', 'sagemaker'),
        ('aws', 'sagemaker')
    ]

    print("\n【余弦相似度 - 原始向量】(范围: -1到1, 越接近1越相似)")
    print("-"*60)
    for name1, name2 in pairs:
        sim = cosine_similarity(embeddings[name1], embeddings[name2])
        print(f"{name1:10} vs {name2:10}: {sim:.6f}")

    print("\n【余弦相似度 - 归一化后】(范围: -1到1, 越接近1越相似)")
    print("-"*60)
    for name1, name2 in pairs:
        sim = cosine_similarity(normalized_embeddings[name1], normalized_embeddings[name2])
        print(f"{name1:10} vs {name2:10}: {sim:.6f}")

    print("\n【欧氏距离 - 原始向量】(距离越小越相似)")
    print("-"*60)
    for name1, name2 in pairs:
        dist = euclidean_distance(embeddings[name1], embeddings[name2])
        print(f"{name1:10} vs {name2:10}: {dist:.6f}")

    print("\n【欧氏距离 - 归一化后】(距离越小越相似)")
    print("-"*60)
    for name1, name2 in pairs:
        dist = euclidean_distance(normalized_embeddings[name1], normalized_embeddings[name2])
        print(f"{name1:10} vs {name2:10}: {dist:.6f}")

    # 计算每个向量的范数
    print("\n【向量范数】")
    print("-"*60)
    for name, vec in embeddings.items():
        norm = np.linalg.norm(vec)
        print(f"{name:10}: {norm:.6f}")

    # 统计信息
    print("\n【向量统计信息】")
    print("-"*60)
    for name, vec in embeddings.items():
        print(f"\n{name}:")
        print(f"  维度:   {len(vec)}")
        print(f"  均值:   {np.mean(vec):.6f}")
        print(f"  标准差: {np.std(vec):.6f}")
        print(f"  最小值: {np.min(vec):.6f}")
        print(f"  最大值: {np.max(vec):.6f}")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
