"""
多目标约束能量最小化 (MT-ICEM)

Multi-Target Constrained Energy Minimization 的 Python 实现。
使用二次规划或线性约束，实现对多个目标的同时检测。

参考:
    - MATLAB 实现: codes/MTICEM_refine.m, codes/MTICEM_1222.m
    - CEM 扩展: 从单目标到多目标的推广

算法:
    MT-ICEM 构建多个约束条件，每个目标一个，
    通过求解约束二次规划得到检测器权重。
"""

import numpy as np
from scipy.optimize import minimize


class MTICEMDetector:
    """
    MT-ICEM 多目标检测器。

    参数:
        reg: 正则化系数
        method: 求解方法 ('qp' 或 'pseudo_inverse')
    """

    def __init__(self, reg: float = 1e-6, method: str = "pseudo_inverse"):
        self.reg = reg
        self.method = method
        self.W: np.ndarray = None  # 权重矩阵 (c, B)
        self.targets: np.ndarray = None  # 目标光谱 (c, B)

    def fit(self, data: np.ndarray, target_matrix: np.ndarray):
        """
        训练 MT-ICEM 检测器。

        参数:
            data: (P, B) 所有像素的光谱数据
            target_matrix: (c, B) c 个目标的光谱矩阵
        """
        B = data.shape[1]
        c = target_matrix.shape[0]
        self.targets = target_matrix

        # 协方差矩阵
        R = (data.T @ data) / data.shape[0]
        R = R + self.reg * np.eye(B)
        R_inv = np.linalg.inv(R)

        # MT-ICEM 权重: W = R⁻¹D (DᵀR⁻¹D)⁻¹
        # 其中 D 是目标矩阵 (c, B)
        D = target_matrix.T  # (B, c)
        R_inv_D = R_inv @ D  # (B, c)
        denom = D.T @ R_inv_D  # (c, c)

        # 伪逆法
        if self.method == "pseudo_inverse":
            denom_inv = np.linalg.pinv(denom)
        else:
            denom_inv = np.linalg.inv(denom + self.reg * np.eye(c))

        self.W = R_inv_D @ denom_inv  # (B, c)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        计算每个像素对每个目标的检测分数。

        参数:
            data: (P, B) 待测光谱数据

        返回:
            scores: (P, c) 每个像素对每个目标的分数
        """
        if self.W is None:
            raise ValueError("请先调用 fit() 训练检测器。")
        return data @ self.W  # (P, c)

    def predict_max(self, data: np.ndarray) -> np.ndarray:
        """
        取所有目标中的最大响应。

        参数:
            data: (P, B) 待测光谱数据

        返回:
            max_scores: (P,) 最大值
        """
        scores = self.predict(data)
        return np.max(scores, axis=1)

    def predict_binary(
        self, data: np.ndarray, threshold: float = 1.0
    ) -> np.ndarray:
        """
        二值化检测结果。

        参数:
            data: (P, B) 待测光谱数据
            threshold: 二值化阈值

        返回:
            binary: (P,) 布尔数组（任一目标超过阈值即为正）
        """
        max_scores = self.predict_max(data)
        return max_scores > threshold
