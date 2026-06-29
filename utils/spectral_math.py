"""
光谱数学工具

提供通用的光谱数据预处理和数学运算函数。
"""

import numpy as np


def preprocess_spectral(data: np.ndarray) -> np.ndarray:
    """
    光谱预处理: 平移至正数 + L2 归一化。

    参数:
        data: (N, D) 光谱数据

    返回:
        normalized: (N, D) 归一化后的光谱
    """
    data = data - data.min(axis=1, keepdims=True)
    norm = np.linalg.norm(data, axis=1, keepdims=True) + 1e-8
    return data / norm


def spectral_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    计算光谱角 (SAM)。

    参数:
        x: (N, D) 或 (D,) 待测光谱
        y: (D,) 或 (N, D) 参考光谱

    返回:
        angles: (N,) 弧度制光谱角
    """
    x = np.atleast_2d(x)
    norm_x = np.linalg.norm(x, axis=1)

    if y.ndim == 1:
        norm_y = np.linalg.norm(y)
        dot = np.dot(x, y)
    else:
        norm_y = np.linalg.norm(y, axis=1)
        dot = np.sum(x * y, axis=1)

    cos_angle = dot / (norm_x * norm_y + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def l2_normalize(data: np.ndarray) -> np.ndarray:
    """L2 归一化。"""
    norm = np.linalg.norm(data, axis=1, keepdims=True) + 1e-8
    return data / norm
