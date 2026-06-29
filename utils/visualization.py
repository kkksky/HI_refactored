"""
可视化工具

提供光谱曲线绘制、数据分析和可视化辅助功能。
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from config import TARGET_WAVELENGTHS


def plot_spectral_curves(
    spectra: np.ndarray,
    wavelengths: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
    title: str = "光谱曲线",
    xlabel: str = "波段 (nm)",
    ylabel: str = "强度",
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    绘制光谱曲线。

    参数:
        spectra: (N, C) 光谱数据，N 条曲线，C 个波段
        wavelengths: 波段波长列表，默认使用 445-905nm
        labels: 每条曲线的标签
        title: 图表标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        show: 是否显示
        save_path: 保存路径（可选）
    """
    if wavelengths is None:
        wavelengths = TARGET_WAVELENGTHS

    plt.figure(figsize=(10, 6))

    for i in range(spectra.shape[0]):
        label = labels[i] if labels and i < len(labels) else None
        plt.plot(wavelengths, spectra[i], "-o", label=label, markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if labels:
        plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    spectra_list: List[np.ndarray],
    labels_list: List[str],
    wavelengths: Optional[List[int]] = None,
    title: str = "光谱对比",
):
    """
    绘制多组光谱曲线的对比。

    参数:
        spectra_list: 光谱数据列表，每组 (N_i, C)
        labels_list: 每组标签
        wavelengths: 波长列表
        title: 图表标题
    """
    if wavelengths is None:
        wavelengths = TARGET_WAVELENGTHS

    plt.figure(figsize=(12, 6))

    colors = ["blue", "red", "green", "orange", "purple"]
    for i, (spectra, label) in enumerate(zip(spectra_list, labels_list)):
        mean_spec = spectra.mean(axis=0)
        std_spec = spectra.std(axis=0)
        color = colors[i % len(colors)]

        plt.plot(wavelengths, mean_spec, color=color, label=label, linewidth=2)
        plt.fill_between(
            wavelengths,
            mean_spec - std_spec,
            mean_spec + std_spec,
            color=color,
            alpha=0.2,
        )

    plt.xlabel("波长 (nm)")
    plt.ylabel("反射率")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_detection_overlay(
    image: np.ndarray,
    overlay_mask: np.ndarray,
    title: str = "检测结果",
    alpha: float = 0.5,
    save_path: Optional[str] = None,
):
    """
    在图像上叠加检测结果。

    参数:
        image: (H, W) 灰度图像或 (H, W, 3) RGB
        overlay_mask: (H, W) 布尔掩膜
        title: 标题
        alpha: 透明度
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 10))

    if image.ndim == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    # 红色 RGBA 叠加
    if overlay_mask.ndim == 2:
        H, W = overlay_mask.shape
        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[..., 0] = 1.0  # R
        rgba[..., 3] = overlay_mask.astype(float) * alpha
        plt.imshow(rgba)

    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
