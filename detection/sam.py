"""
光谱角检测 (SAM — Spectral Angle Mapper)

通过计算每个像素光谱与目标原型光谱之间的角度来衡量相似性。
角度越小越相似，可用于目标检测和分类。

参考论文:
    - SAM 是高光谱遥感领域最经典的目标检测算法
    - 本项目应用: 伪装目标检测（草地迷彩、荒漠迷彩等）

算法思路:
    1. L2 归一化（消除光照强度影响，保留光谱形状）
    2. 从目标样本计算平均光谱作为原型
    3. 计算每个像素与原型之间的光谱角
"""

import numpy as np


class SpectralAngleMapper:
    """
    光谱角检测器。

    参数:
        normalize: 是否对输入数据进行 L2 归一化（建议开启）
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.prototype: np.ndarray = None

    def fit(self, target_data: np.ndarray):
        """
        从目标样本拟合原型光谱。

        参数:
            target_data: (N, B) N 条目标光谱样本
        """
        if self.normalize:
            target_data = self._l2_normalize(target_data)

        self.prototype = target_data.mean(axis=0)
        if self.normalize:
            norm = np.linalg.norm(self.prototype) + 1e-8
            self.prototype = self.prototype / norm

    @staticmethod
    def _l2_normalize(data: np.ndarray) -> np.ndarray:
        """L2 归一化：消除光照强度，保留光谱形状。"""
        data = data - data.min(axis=1, keepdims=True)
        norm = np.linalg.norm(data, axis=1, keepdims=True) + 1e-8
        return data / norm

    @staticmethod
    def spectral_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算光谱角。

        参数:
            x: (N, D) 或 (D,) 待测光谱
            y: (D,) 或 (N, D) 参考光谱

        返回:
            angles: (N,) 弧度制的光谱角
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

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        计算数据中每个样本与原型的光谱角。

        参数:
            data: (N, B) 待测光谱数据

        返回:
            angles: (N,) 每个像素的光谱角（弧度），越小越像目标
        """
        if self.prototype is None:
            raise ValueError("请先调用 fit() 设置原型光谱。")

        if self.normalize:
            data = self._l2_normalize(data)

        return self.spectral_angle(data, self.prototype)

    def predict_score(self, data: np.ndarray) -> np.ndarray:
        """
        计算目标相似度分数（0~1，越大越像目标）。

        参数:
            data: (N, B) 待测光谱数据

        返回:
            scores: (N,) 目标相似度分数
        """
        angles = self.predict(data)
        # 角度越小越像目标 → 映射为 0~1 分数
        max_angle = np.pi / 2
        return np.clip(1.0 - angles / max_angle, 0.0, 1.0)
