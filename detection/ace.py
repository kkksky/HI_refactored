"""
自适应余弦估计 (ACE — Adaptive Cosine Estimator)

一种广义似然比检测器，在去除背景均值后计算
测试像素与目标光谱的余弦相似度，并根据背景协方差进行白化。

参考论文:
    - Kraut & Scharf 1999: ACE 的理论基础
    - 本项目应用: 伪装目标检测的基线算法

算法:
    ACE(x) = (dᵀR⁻¹(x-μ))² / ((dᵀR⁻¹d) · (x-μ)ᵀR⁻¹(x-μ))

    其中 d 是目标光谱，R 是背景协方差矩阵，μ 是背景均值
"""

import numpy as np


class ACEDetector:
    """
    ACE 检测器。

    参数:
        reg: 正则化系数
    """

    def __init__(self, reg: float = 1e-6):
        self.reg = reg
        self.mean: np.ndarray = None
        self.R_inv: np.ndarray = None
        self.target: np.ndarray = None

    def fit(self, data: np.ndarray, target: np.ndarray):
        """
        训练 ACE 检测器。

        参数:
            data: (P, B) 所有像素的光谱数据（用于估计背景统计）
            target: (B,) 或 (1, B) 目标光谱
        """
        target = target.ravel()
        B = data.shape[1]

        # 背景均值与协方差
        self.mean = np.mean(data, axis=0)
        Xc = data - self.mean
        R = (Xc.T @ Xc) / data.shape[0]
        R = R + self.reg * np.eye(B)
        self.R_inv = np.linalg.inv(R)
        self.target = target

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        计算每个像素的 ACE 分数。

        参数:
            data: (P, B) 待测光谱数据

        返回:
            scores: (P,) ACE 分数（0~1，越高越像目标）
        """
        if self.R_inv is None:
            raise ValueError("请先调用 fit() 训练检测器。")

        Xc = data - self.mean
        d = self.target - self.mean

        d_R_inv = d @ self.R_inv  # (B,)
        numerator = (Xc @ self.R_inv @ d) ** 2  # (P,)
        denominator = (d_R_inv @ d) * np.sum(Xc @ self.R_inv * Xc, axis=1)

        denominator = np.maximum(denominator, 1e-12)
        return numerator / denominator

    def predict_binary(
        self, data: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """
        二值化检测结果。

        参数:
            data: (P, B) 待测光谱数据
            threshold: 二值化阈值

        返回:
            binary: (P,) 布尔数组
        """
        scores = self.predict(data)
        return scores > threshold
