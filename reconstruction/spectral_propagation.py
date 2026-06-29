"""
光谱传播与三边滤波

实现从稀疏光谱采样点到全分辨率光谱数据的传播重建，
使用三边滤波 (Trilateral Filtering) 融合空间、光谱和时间信息。

参考论文:
    - Cao 2011 CVPR "Hybrid Camera": 首次提出光谱传播+三边滤波框架
    - Ma 2014 IJCV "High Spatial-Spectral Resolution": 改进的光谱传播算法

算法思路:
    光谱传播的核心是通过加权平均将稀疏采样点的光谱信息传播到全分辨率：

        S(p, λ) = Σᵢ w(p, qᵢ) · S(qᵢ, λ) / Σᵢ w(p, qᵢ)

    权重 w 由三部分组成：
        w(p, q) = w_spatial(p, q) · w_spectral(p, q) · w_temporal(p, q)

    其中:
        - w_spatial: 空间距离高斯权重 — 近邻像素贡献更大
        - w_spectral: 光谱/颜色相似性权重 — 颜色相近的像素共享更多信息
        - w_temporal: (视频) 时间颜色一致性 — 利用帧间连续性
"""

from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class SpectralPropagator:
    """
    光谱传播器。

    将稀疏光谱采样点传播到全分辨率光谱数据立方体。

    参数:
        spatial_sigma: 空间距离高斯核标准差
        spectral_sigma: 光谱/颜色相似度高斯核标准差
        knn: 每个像素考虑的最近邻采样点数
    """

    def __init__(
        self,
        spatial_sigma: float = 10.0,
        spectral_sigma: float = 0.1,
        knn: int = 5,
    ):
        self.spatial_sigma = spatial_sigma
        self.spectral_sigma = spectral_sigma
        self.knn = knn

    def propagate(
        self,
        rgb_hr: np.ndarray,
        sparse_spectra: np.ndarray,
        sparse_coords: np.ndarray,
    ) -> np.ndarray:
        """
        从稀疏采样重建全分辨率光谱。

        参数:
            rgb_hr: (H, W, 3) 高分辨率 RGB 图像（作为光谱传播的引导）
            sparse_spectra: (N, C) N 个稀疏采样点的光谱
            sparse_coords: (N, 2) 稀疏采样点的 (y, x) 坐标

        返回:
            full_spectrum: (H, W, C) 重建的全分辨率光谱数据立方体
        """
        H, W = rgb_hr.shape[:2]
        C = sparse_spectra.shape[1]
        full_spectrum = np.zeros((H, W, C), dtype=np.float32)
        weight_sum = np.zeros((H, W), dtype=np.float32)

        # 将 RGB 转换为 LAB 空间（更符合视觉相似性）
        rgb_lab = cv2.cvtColor(rgb_hr, cv2.COLOR_RGB2LAB).astype(np.float32)

        # 为每个稀疏点创建高斯影响图
        for i in range(len(sparse_coords)):
            sy, sx = sparse_coords[i]
            spectrum = sparse_spectra[i]

            # 空间距离权重
            yy, xx = np.mgrid[0:H, 0:W]
            spatial_w = np.exp(
                -((yy - sy) ** 2 + (xx - sx) ** 2) / (2 * self.spatial_sigma ** 2)
            )

            # 颜色相似性权重
            color_diff = np.linalg.norm(rgb_lab - rgb_lab[sy, sx], axis=2)
            spectral_w = np.exp(
                -(color_diff ** 2) / (2 * self.spectral_sigma ** 2)
            )

            # 联合权重
            w = spatial_w * spectral_w

            # 累加加权光谱
            for c in range(C):
                full_spectrum[:, :, c] += w * spectrum[c]
            weight_sum += w

        # 归一化
        weight_sum = np.maximum(weight_sum, 1e-10)
        for c in range(C):
            full_spectrum[:, :, c] /= weight_sum

        return full_spectrum

    def propagate_with_trilateral_filter(
        self,
        rgb_hr: np.ndarray,
        sparse_spectra: np.ndarray,
        sparse_coords: np.ndarray,
        prev_spectrum: Optional[np.ndarray] = None,
        temporal_sigma: float = 0.05,
    ) -> np.ndarray:
        """
        三边滤波光谱传播（含时间维度）。

        当提供前一帧的光谱时，额外引入时间一致性权重。

        参数:
            rgb_hr: (H, W, 3) 高分辨率 RGB
            sparse_spectra: (N, C) 稀疏光谱
            sparse_coords: (N, 2) 稀疏坐标
            prev_spectrum: (H, W, C) 前一帧全分辨率光谱（可选）
            temporal_sigma: 时间一致性高斯核标准差

        返回:
            full_spectrum: (H, W, C) 重建的全分辨率光谱
        """
        H, W = rgb_hr.shape[:2]
        C = sparse_spectra.shape[1]
        full_spectrum = np.zeros((H, W, C), dtype=np.float32)
        weight_sum = np.zeros((H, W), dtype=np.float32)

        rgb_lab = cv2.cvtColor(rgb_hr, cv2.COLOR_RGB2LAB).astype(np.float32)

        for i in range(len(sparse_coords)):
            sy, sx = sparse_coords[i]
            spectrum = sparse_spectra[i]

            yy, xx = np.mgrid[0:H, 0:W]
            w_spatial = np.exp(
                -((yy - sy) ** 2 + (xx - sx) ** 2) / (2 * self.spatial_sigma ** 2)
            )

            color_diff = np.linalg.norm(rgb_lab - rgb_lab[sy, sx], axis=2)
            w_spectral = np.exp(-(color_diff ** 2) / (2 * self.spectral_sigma ** 2))

            w = w_spatial * w_spectral

            # 时间权重：如果提供了前一帧，当前帧与前一帧的颜色差异
            if prev_spectrum is not None:
                temporal_diff = np.linalg.norm(rgb_lab - prev_spectrum[:, :, :3], axis=2)
                w_temporal = np.exp(-(temporal_diff ** 2) / (2 * temporal_sigma ** 2))
                w = w * w_temporal

            for c in range(C):
                full_spectrum[:, :, c] += w * spectrum[c]
            weight_sum += w

        weight_sum = np.maximum(weight_sum, 1e-10)
        for c in range(C):
            full_spectrum[:, :, c] /= weight_sum

        return full_spectrum
