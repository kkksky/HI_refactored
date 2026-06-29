"""
棱镜色散光谱重建

基于柯西色散公式 (Cauchy dispersion formula) 的棱镜色散模型，
实现从多通道图像到光谱数据的物理重建。

参考论文:
    - Du 2009 ICCV "Prism-Based Multi-Spectral Video": 棱镜色散系统建模
    - Cao 2011 PAMI "Prism-Mask System": 棱镜-掩膜系统的色散标定方法
    - Feng 2014 OE "Amici Prism": 阿米西棱镜的色散特性与光谱重建

算法思路:
    棱镜对不同波长的光具有不同的折射率，导致同一空间点在图像中
    随波长不同产生像素偏移。本模块利用柯西色散公式：

        n(λ) = A + B/λ² + C/λ⁴

    建模波长与像素偏移的关系，建立查找表 (LUT)，
    通过矩阵高级索引实现毫秒级的光谱重构。
"""

import cv2
import numpy as np
from tqdm import tqdm

from config import TARGET_WAVELENGTHS, NUM_BANDS


def calibrate_prism_dispersion(
    mat: np.ndarray,
    mono_wavelength: int = 535,
    mono_channel_idx: int = 18,
    dispersion_strength: float = 15.0,
) -> tuple:
    """
    自动标定棱镜色散系统的几何与色散参数。

    通过单色光照射图像，自动检测亮斑位置作为空间基准，
    利用柯西色散公式计算每个波长的像素偏移量。

    参数:
        mat: 加载的 3D 高光谱矩阵 (H, W, C)
        mono_wavelength: 单色光的真实波长 (nm)
        mono_channel_idx: 单色光在矩阵中的通道索引 (0 ~ C-1)
        dispersion_strength: 色散强度因子，控制偏移量幅度

    返回:
        base_points: 空间基准点坐标列表 [(x, y), ...]
        lut_offsets: (C, 2) 每个波长的像素偏移量 [dx, dy]

    色散模型:
        dx(λ) = D * ( λ_ref² / λ² - 1 )
        dy(λ) = 0  (假设色散仅发生在水平方向)
    """
    h, w, num_channels = mat.shape

    # 1. 提取单色光图像
    mono_img = mat[:, :, mono_channel_idx].astype(np.uint8)

    # 2. 自动提取亮斑质心
    mono_img_8u = cv2.normalize(mono_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ret, thresh = cv2.threshold(mono_img_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"自动二值化阈值: {ret}")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    mono_points = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 5:  # 过滤噪声
            mono_points.append(centroids[i])

    # 按物理网格排序
    mono_points = sorted(mono_points, key=lambda p: (round(p[1] / 10) * 10, p[0]))
    mono_points = np.array(mono_points)
    print(f"🎯 自动识别出 {len(mono_points)} 个空间采样点")

    # 3. 柯西色散公式 → 偏移量 LUT
    ref_factor = 1.0 / (mono_wavelength ** 2)

    lut_offsets = np.zeros((num_channels, 2), dtype=np.float32)
    for c in range(num_channels):
        current_wl = TARGET_WAVELENGTHS[c]
        dx = dispersion_strength * ((1.0 / (current_wl ** 2)) / ref_factor - 1.0)
        lut_offsets[c] = [dx, 0.0]  # 假设色散仅发生在 X 方向

    return list(mono_points), lut_offsets


def reconstruct_spectrum_fast(
    mat: np.ndarray,
    base_points: list,
    lut_offsets: np.ndarray,
) -> np.ndarray:
    """
    利用 NumPy 高级索引进行快速光谱重建。

    通过 LUT 偏移量，使用批量矩阵寻址一次性提取所有采样点的光谱。

    参数:
        mat: (H, W, C) 原始高光谱数据立方体
        base_points: 空间基准点坐标列表 [(x, y), ...]
        lut_offsets: (C, 2) 每个波长的像素偏移量

    返回:
        spectral_features: (N, 1, C) 纯净光谱矩阵，可直接用于 1D-CNN
    """
    h, w, num_channels = mat.shape
    num_points = len(base_points)

    base_pts = np.array(base_points)
    spectral_features = np.zeros((num_points, 1, num_channels), dtype=np.float32)

    for c in range(num_channels):
        dx, dy = lut_offsets[c]

        # 计算当前波段的实际像素坐标（偏移后）
        u = np.clip(np.round(base_pts[:, 0] + dx).astype(np.int32), 0, w - 1)
        v = np.clip(np.round(base_pts[:, 1] + dy).astype(np.int32), 0, h - 1)

        # 矩阵级寻址 — 一次性提取所有采样点
        spectral_features[:, 0, c] = mat[v, u, c]

    return spectral_features
