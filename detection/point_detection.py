"""
高光谱点源检测

从 3D 数据立方体 (H, W, C) 中检测点源目标。
提供 CPU (SciPy) 和 GPU (PyTorch) 两种实现。

算法步骤:
    1. 高斯加权窗口平滑（去噪）
    2. 动态阈值过滤（mean + k * std）
    3. 局部极大值抑制 (CMS, Center-surround Maximum Suppression)

参考论文:
    - Du 2009 ICCV: 点源检测与跟踪的框架
    - 通用高光谱处理: 局部极大值检测的经典方法
"""

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from tqdm import tqdm

from config import (
    GAUSS_K_SIZE,
    GAUSS_SIGMA,
    MAX_FILTER_SIZE,
    BATCH_SIZE,
    DETECTION_THRESHOLD,
    DEVICE,
)


def weighted_window_sum(
    mat: np.ndarray,
    k_size: int = GAUSS_K_SIZE,
    sigma: float = GAUSS_SIGMA,
) -> np.ndarray:
    """
    高斯加权窗口求和。

    使用 OpenCV 的 GaussianBlur 实现高效的高斯加权平均。

    参数:
        mat: 输入矩阵
        k_size: 高斯核大小（需为奇数）
        sigma: 高斯标准差

    返回:
        加权求和结果
    """
    weighted_avg = cv2.GaussianBlur(
        mat, (k_size, k_size), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101
    )
    # 从加权平均还原为加权和
    kernel_x = cv2.getGaussianKernel(k_size, sigma)
    kernel_2d_sum = np.sum(kernel_x * kernel_x.T)
    return weighted_avg * kernel_2d_sum


def get_gaussian_kernel_2d(
    k_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """在 GPU 上生成二维高斯核。"""
    coords = torch.arange(k_size, device=device, dtype=dtype) - (k_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.view(-1, 1) @ g.view(1, -1)
    return kernel_2d.view(1, 1, k_size, k_size)


def process_hyperspectral_cpu(
    hyperspectral_data: np.ndarray,
    window_size: Tuple[int, int] = (12, 12),
    k_size: int = GAUSS_K_SIZE,
    sigma: float = GAUSS_SIGMA,
    threshold: float = DETECTION_THRESHOLD,
) -> np.ndarray:
    """
    CPU 版高光谱点源检测（使用 SciPy）。

    对整个数据立方体进行逐通道的:
        1. 高斯加权平滑
        2. 动态阈值过滤
        3. 局部极大值抑制

    参数:
        hyperspectral_data: (H, W, C) 高光谱数据
        window_size: 局部极大值检测窗口
        k_size: 高斯核大小
        sigma: 高斯标准差
        threshold: 动态阈值系数

    返回:
        (H, W, C) 检测结果（仅在局部最大值处保留原值）
    """
    # 预处理：加权平滑 + 归一化
    data_norm = weighted_window_sum(hyperspectral_data, k_size, sigma).astype(np.float32) / 65535.0

    # 动态阈值
    channel_means = np.mean(data_norm, axis=(0, 1))
    channel_stds = np.std(data_norm, axis=(0, 1))
    dynamic_thresholds = channel_means + threshold * channel_stds
    data_norm[data_norm < dynamic_thresholds] = 0

    # 局部极大值抑制（逐通道）
    result = np.zeros_like(data_norm)
    print("正在进行局部极大值抑制 (CPU)...")
    for c in tqdm(range(data_norm.shape[2]), desc="处理进度"):
        channel_data = data_norm[:, :, c]
        if np.any(channel_data):
            l_max = maximum_filter(channel_data, size=window_size)
            mask = (channel_data == l_max) & (channel_data > 0)
            result[:, :, c] = np.where(mask, channel_data, 0)

    return result


def process_hyperspectral_gpu(
    hyperspectral_data: np.ndarray,
    k_size: int = GAUSS_K_SIZE,
    sigma: float = GAUSS_SIGMA,
    window_size: Tuple[int, int] = MAX_FILTER_SIZE,
    threshold: float = DETECTION_THRESHOLD,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
) -> np.ndarray:
    """
    GPU 加速版高光谱点源检测（分批处理，显存优化）。

    使用 PyTorch 的 F.conv2d + F.max_pool2d 加速，
    按 batch 分次处理通道以减少显存占用。

    参数:
        hyperspectral_data: (H, W, C) 高光谱数据
        k_size: 高斯核大小
        sigma: 高斯标准差
        window_size: 局部极大值检测窗口
        threshold: 动态阈值系数
        batch_size: 每批处理的通道数
        device: 计算设备

    返回:
        (H, W, C) 检测结果
    """
    H, W, C = hyperspectral_data.shape
    output_np = np.zeros((H, W, C), dtype=np.float32)

    kernel = get_gaussian_kernel_2d(k_size, sigma, torch.device(device), torch.float32)
    pad_size = k_size // 2
    pool_pad = (window_size[0] // 2, window_size[1] // 2)

    for i in tqdm(range(0, C, batch_size), desc="GPU 处理通道", unit="batch"):
        start = i
        end = min(i + batch_size, C)

        batch_data = hyperspectral_data[:, :, start:end]
        batch_tensor = (
            torch.from_numpy(batch_data).to(device).to(torch.float32)
        )
        batch_tensor = batch_tensor.permute(2, 0, 1).unsqueeze(1)  # [B, 1, H, W]

        # 高斯平滑
        padded = F.pad(batch_tensor, [pad_size] * 4, mode="reflect")
        weighted_avg = F.conv2d(padded, kernel, padding=0)
        data_normalized = weighted_avg / 65535.0

        # 动态阈值
        means = data_normalized.mean(dim=(2, 3), keepdim=True)
        stds = data_normalized.std(dim=(2, 3), keepdim=True)
        thresholds = means + threshold * stds
        data_normalized = torch.where(
            data_normalized > thresholds,
            data_normalized,
            torch.zeros_like(data_normalized),
        )

        # 局部极大值抑制
        l_max = F.max_pool2d(
            data_normalized,
            kernel_size=window_size,
            stride=1,
            padding=pool_pad,
        )
        if l_max.shape[2:] != data_normalized.shape[2:]:
            l_max = F.interpolate(l_max, size=(H, W), mode="nearest")

        mask = (data_normalized == l_max) & (data_normalized > 0)
        final_batch = torch.where(mask, data_normalized, torch.zeros_like(data_normalized))

        # 写回 CPU
        processed = final_batch.squeeze(1).permute(1, 2, 0).cpu().numpy()
        output_np[:, :, start:end] = processed

        del batch_tensor, data_normalized, l_max, final_batch

    return output_np
