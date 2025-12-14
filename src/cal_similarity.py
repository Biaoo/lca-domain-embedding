import numpy as np
from numpy.typing import ArrayLike


def cosine_similarity(vec1: ArrayLike, vec2: ArrayLike) -> float:
    """计算两个向量的余弦相似度。

    Args:
        vec1: 第一个向量
        vec2: 第二个向量

    Returns:
        两个向量的余弦相似度，范围为 [-1, 1]
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))
