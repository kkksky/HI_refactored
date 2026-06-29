"""
图像配准工具

提供 ECC (Enhanced Correlation Coefficient) 自动配准和手动对齐功能。

参考:
    - Evangelidis & Psarakis "Parametric Image Alignment using ECC" (2008)
    - 本项目应用: 高光谱不同波段间的空间对齐（消除棱镜色散导致的偏移）
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def register_ecc(
    img_ref: np.ndarray,
    img_moving: np.ndarray,
    warp_mode: int = cv2.MOTION_TRANSLATION,
    max_iterations: int = 100,
    termination_eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ECC 图像配准。

    使用增强相关系数 (ECC) 方法进行亚像素精度配准。

    参数:
        img_ref: 参考图像
        img_moving: 待配准图像
        warp_mode: 变换类型 (cv2.MOTION_TRANSLATION / AFFINE / HOMOGRAPHY)
        max_iterations: 最大迭代次数
        termination_eps: 收敛阈值

    返回:
        img_aligned: 配准后的图像
        warp_matrix: 变换矩阵
    """
    # 初始化变换矩阵
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # 转换为 float32
    img_ref_f = img_ref.astype(np.float32)
    img_moving_f = img_moving.astype(np.float32)

    # ECC 配准
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, termination_eps)
    try:
        _, warp_matrix = cv2.findTransformECC(
            img_ref_f, img_moving_f, warp_matrix, warp_mode, criteria
        )
    except cv2.error as e:
        print(f"ECC 配准失败: {e}")
        return img_moving, warp_matrix

    # 应用变换
    h, w = img_ref.shape
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        img_aligned = cv2.warpPerspective(img_moving, warp_matrix, (w, h))
    else:
        img_aligned = cv2.warpAffine(img_moving, warp_matrix, (w, h))

    return img_aligned, warp_matrix


def manual_alignment(
    img1: np.ndarray,
    img2: np.ndarray,
    initial_translation: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    手动平移对齐（交互式工具的前端）。

    通过计算中值偏移进行初始对齐。

    参数:
        img1: 参考图像
        img2: 待配准图像
        initial_translation: 初始平移量 (dx, dy)

    返回:
        aligned: 平移对齐后的图像
    """
    dy, dx = initial_translation
    h, w = img1.shape
    aligned = np.zeros_like(img2)
    y_start, x_start = max(0, dy), max(0, dx)
    y_end, x_end = min(h, h + dy), min(w, w + dx)

    aligned[y_start:y_end, x_start:x_end] = img2[
        max(0, -dy): h - max(0, dy),
        max(0, -dx): w - max(0, dx)
    ]
    return aligned
