"""
约束能量最小化 (CEM — Constrained Energy Minimization)

高光谱目标检测的经典线性滤波器方法。
设计一个 FIR 滤波器 w，使得输出能量最小化，
同时满足对目标信号 d 的约束 wᵀd = 1。

参考论文:
    - Harsanyi 1994: CEM 原始论文
    - 本项目应用: 伪装目标检测的基线算法

算法:
    w = R⁻¹d / (dᵀR⁻¹d)
    其中 R 是数据协方差矩阵，d 是目标光谱
"""

import numpy as np


class CEMDetector:
    """
    CEM 检测器。

    参数:
        reg: 正则化系数（防止协方差矩阵奇异）
    """

    def __init__(self, reg: float = 1e-6):
        self.reg = reg
        self.w: np.ndarray = None
        self.target: np.ndarray = None

    def fit(self, data: np.ndarray, target: np.ndarray):
        """
        训练 CEM 检测器。

        参数:
            data: (P, B) 所有像素的光谱数据
            target: (B,) 或 (1, B) 目标光谱
        """
        target = target.ravel()
        B = data.shape[1]

        # 协方差矩阵
        R = (data.T @ data) / data.shape[0]
        R = R + self.reg * np.eye(B)

        # CEM 滤波器: w = R⁻¹d / (dᵀR⁻¹d)
        R_inv_d = np.linalg.solve(R, target)
        denom = target @ R_inv_d
        self.w = R_inv_d / denom
        self.target = target

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        计算每个像素的 CEM 检测分数。

        参数:
            data: (P, B) 待测光谱数据

        返回:
            scores: (P,) 检测分数（越高越像目标）
        """
        if self.w is None:
            raise ValueError("请先调用 fit() 训练检测器。")
        return data @ self.w

    def predict_binary(
        self, data: np.ndarray, threshold: float = 1.0
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
