"""
光谱角约束能量最小化 (SACE — Spectral Angle Constrained Energy minimization)

SACE 是结合光谱角约束与能量最小化的混合检测器，
对光谱变化具有更好的鲁棒性。

参考:
    - MATLAB 实现: codes/SACE_refine.m (前47行核心算法，后面600行为无关内容)
    - 本项目核心算法之一

算法思路:
    1. 使用非负最小二乘 (NNLS) 或正则化方法分离混合光谱
    2. 计算测试像素在目标子空间上的投影
    3. 计算光谱角约束的检测统计量
"""

import numpy as np
from scipy.optimize import nnls


class SACEDetector:
    """
    SACE 检测器。

    参数:
        reg: 正则化系数
        use_nnls: 是否使用非负最小二乘进行光谱解混
    """

    def __init__(self, reg: float = 1e-6, use_nnls: bool = False):
        self.reg = reg
        self.use_nnls = use_nnls
        self.mean: np.ndarray = None
        self.R_inv: np.ndarray = None
        self.target: np.ndarray = None

    def fit(self, data: np.ndarray, target: np.ndarray):
        """
        训练 SACE 检测器。

        参数:
            data: (P, B) 所有像素的光谱数据
            target: (B,) 或 (c, B) 目标光谱（可多目标，默认使用第一个）
        """
        if target.ndim > 1:
            target = target[0]
        target = target.ravel()
        B = data.shape[1]

        # 背景统计（与 ACE 类似的预处理）
        self.mean = np.mean(data, axis=0)
        Xc = data - self.mean
        R = (Xc.T @ Xc) / data.shape[0]
        R = R + self.reg * np.eye(B)
        self.R_inv = np.linalg.inv(R)
        self.target = target

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        计算 SACE 检测分数。

        参数:
            data: (P, B) 待测光谱数据

        返回:
            scores: (P,) SACE 分数
        """
        if self.R_inv is None:
            raise ValueError("请先调用 fit() 训练检测器。")

        if self.use_nnls:
            return self._predict_nnls(data)
        else:
            return self._predict_ace_like(data)

    def _predict_ace_like(self, data: np.ndarray) -> np.ndarray:
        """
        ACE 风格的 SACE 实现（默认）。
        SACE 在 ACE 基础上增加了光谱角约束。
        """
        Xc = data - self.mean
        d = self.target - self.mean

        # 第一步: ACE 分数
        d_R_inv = d @ self.R_inv
        ace_num = (Xc @ self.R_inv @ d) ** 2
        ace_den = (d_R_inv @ d) * np.sum(Xc @ self.R_inv * Xc, axis=1)
        ace_den = np.maximum(ace_den, 1e-12)
        ace_score = ace_num / ace_den

        # 第二步: 光谱角约束
        # 计算每个像素与目标光谱的余弦相似度
        norm_data = np.linalg.norm(data, axis=1) + 1e-8
        norm_target = np.linalg.norm(self.target) + 1e-8
        cos_angle = (data @ self.target) / (norm_data * norm_target)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # 结合 ACE 分数与光谱角（保留与目标方向一致的像素）
        # 光谱角的权重: cos_angle 越大（角度越小）越保留
        return ace_score * np.maximum(cos_angle, 0.0)

    def _predict_nnls(self, data: np.ndarray) -> np.ndarray:
        """
        基于非负最小二乘的 SACE 实现。

        将每个像素表示为端元（endmember）的非负线性组合，
        计算目标端元的丰度作为检测分数。
        """
        P = data.shape[0]
        scores = np.zeros(P)

        # 使用目标光谱和几个背景端元
        # 简化实现：直接使用目标作为唯一端元
        for i in range(P):
            # 求解 min ||data[i] - a * target||²  s.t. a >= 0
            a, residual = nnls(self.target[:, np.newaxis], data[i])
            scores[i] = a[0]

        return scores

    def predict_binary(
        self, data: np.ndarray, threshold: float = 0.7
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
