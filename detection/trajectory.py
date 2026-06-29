"""
贯穿目标轨迹追踪

从点源检测结果中提取跨波段的"贯穿目标"轨迹，
构建 Survival Cube 和坐标映射字典。

算法思路:
    1. 逆向回溯：从最后一个波段开始，反向搜索能贯穿全通道的起点
    2. 正向提取：从起点出发，逐波段追踪，构建连续轨迹
    3. ID 传播：为每个贯穿点分配唯一 ID，维护跨波段的对应关系

参考论文:
    - Du 2009 ICCV: 跨波段目标追踪与轨迹构建
    - 本项目核心创新: Survival Cube 算法
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from tqdm import tqdm

from config import TRACKING_WINDOW, BACKWARD_WINDOW


def get_survival_cube(
    mat_filtered: np.ndarray,
    tracking_window: int = TRACKING_WINDOW,
    backward_window: int = BACKWARD_WINDOW,
) -> Tuple[np.ndarray, dict, dict]:
    """
    经典版贯穿目标轨迹追踪。

    返回 perfect_cube_mask, coords_dict, id_to_key。

    参数:
        mat_filtered: (H, W, C) 点源检测后的二值/浮点数据
        tracking_window: 正向追踪的邻域窗口大小
        backward_window: 逆向筛选的邻域窗口大小

    返回:
        perfect_cube_mask: (H, W, C) 贯穿目标的布尔掩膜
        coords_dict: {id: [(channel, y, x), ...]} 轨迹字典
        id_to_key: {(y, x): id} ID 映射表
    """
    h, w, c = mat_filtered.shape
    layers = [mat_filtered[:, :, i] > 0 for i in range(c)]

    # --- 逆向回溯 ---
    backward_params = {"size": (backward_window, backward_window), "origin": (0, 0)}
    backward_masks = [None] * c
    survival_mask = layers[-1]
    backward_masks[-1] = survival_mask

    for i in tqdm(range(c - 2, -1, -1), desc="逆向筛选"):
        has_future = maximum_filter(survival_mask, **backward_params)
        survival_mask = layers[i] & has_future
        backward_masks[i] = survival_mask
        if not np.any(survival_mask):
            print("\n⚠️ 无点能贯穿全通道。")
            return np.zeros((h, w, c), dtype=bool), {}, {}

    # --- 正向提取 ---
    forward_params = {"size": (tracking_window, tracking_window), "origin": (0, 0)}
    perfect_cube_mask = np.zeros((h, w, c), dtype=bool)
    perfect_cube_mask[:, :, 0] = backward_masks[0]

    # 初始化 ID 矩阵
    current_id_map = np.zeros((h, w), dtype=np.int32)
    y0, x0 = np.where(perfect_cube_mask[:, :, 0])
    num_seeds = len(y0)
    current_id_map[y0, x0] = np.arange(1, num_seeds + 1)

    # 预分配坐标存储
    id_results = [[] for _ in range(num_seeds + 1)]
    for idx in range(num_seeds):
        id_results[idx + 1].append((np.int16(0), y0[idx], x0[idx]))

    for i in tqdm(range(1, c), desc="正向提取"):
        can_be_reached = maximum_filter(perfect_cube_mask[:, :, i - 1], **forward_params)
        current_layer_mask = backward_masks[i] & can_be_reached
        perfect_cube_mask[:, :, i] = current_layer_mask

        can_be_reached_ids = maximum_filter(current_id_map, **forward_params)
        current_id_map = np.where(current_layer_mask, can_be_reached_ids, 0)

        curr_y, curr_x = np.where(current_id_map > 0)
        curr_ids = current_id_map[curr_y, curr_x]
        for y, x, tid in zip(curr_y, curr_x, curr_ids):
            id_results[tid].append((np.int16(i), np.int16(y), np.int16(x)))

    id_to_key = {i: (int(y0[i - 1]), int(x0[i - 1])) for i in range(1, num_seeds + 1)}
    coords_dict = {id_to_key[i]: id_results[i] for i in range(1, num_seeds + 1)}

    return perfect_cube_mask, coords_dict, id_to_key


def get_survival_cube_optimized(
    mat_filtered: np.ndarray,
    tracking_window: int = TRACKING_WINDOW,
    backward_window: int = BACKWARD_WINDOW,
) -> Tuple[np.ndarray, dict, dict]:
    """
    内存优化版贯穿目标轨迹追踪。

    使用预分配的 3D numpy 数组替代逐层列表，减少 Python 对象开销。

    参数与 get_survival_cube 相同。
    """
    h, w, c = mat_filtered.shape
    layers = mat_filtered > 0
    filter_params = {"size": (tracking_window, tracking_window), "origin": (0, 0)}

    # --- 逆向回溯 （向量化）---
    backward_masks = np.zeros((h, w, c), dtype=bool)
    survival_mask = layers[:, :, -1]
    backward_masks[:, :, -1] = survival_mask

    for i in range(c - 2, -1, -1):
        has_future = maximum_filter(survival_mask, **filter_params)
        survival_mask = layers[:, :, i] & has_future
        backward_masks[:, :, i] = survival_mask
        if not np.any(survival_mask):
            return np.zeros((h, w, c), dtype=bool), {}, {}

    # --- 正向提取 ---
    perfect_cube_mask = np.zeros((h, w, c), dtype=bool)
    perfect_cube_mask[:, :, 0] = backward_masks[:, :, 0]

    current_id_map = np.zeros((h, w), dtype=np.int32)
    y0, x0 = np.where(perfect_cube_mask[:, :, 0])
    num_seeds = len(y0)
    current_id_map[y0, x0] = np.arange(1, num_seeds + 1)

    id_results = [[] for _ in range(num_seeds + 1)]
    for idx in range(num_seeds):
        id_results[idx + 1].append((np.int16(0), y0[idx], x0[idx]))

    for i in range(1, c):
        can_be_reached = maximum_filter(perfect_cube_mask[:, :, i - 1], **filter_params)
        current_layer_mask = backward_masks[:, :, i] & can_be_reached
        perfect_cube_mask[:, :, i] = current_layer_mask

        can_be_reached_ids = maximum_filter(current_id_map, **filter_params)
        current_id_map = np.where(current_layer_mask, can_be_reached_ids, 0)

        curr_y, curr_x = np.where(current_id_map > 0)
        curr_ids = current_id_map[curr_y, curr_x]

        for y, x, tid in zip(curr_y, curr_x, curr_ids):
            id_results[tid].append((np.int16(i), np.int16(y), np.int16(x)))

    id_to_key = {i: (int(y0[i - 1]), int(x0[i - 1])) for i in range(1, num_seeds + 1)}
    coords_dict = {id_to_key[i]: id_results[i] for i in range(1, num_seeds + 1)}

    return perfect_cube_mask, coords_dict, id_to_key


def get_survival_cube_gpu(
    mat_filtered: np.ndarray,
    tracking_window: int = TRACKING_WINDOW,
    device: str = "cuda",
) -> Tuple[np.ndarray, dict, dict]:
    """
    GPU 加速版贯穿目标轨迹追踪。

    使用 PyTorch 的 MaxPool2d 替代 SciPy 的 maximum_filter 进行加速，
    所有坐标操作在 GPU 上的张量完成，最后统一转回 CPU。

    参数:
        mat_filtered: (H, W, C) 点源检测结果
        tracking_window: 邻域窗口大小
        device: 计算设备

    返回:
        同 get_survival_cube
    """
    data = torch.from_numpy(mat_filtered).to(device)
    C, H, W = data.shape[2], data.shape[0], data.shape[1]
    layers = (data.permute(2, 0, 1) > 0).byte()
    pad = (tracking_window - 1) // 2

    # --- 逆向回溯 (GPU MaxPool2d) ---
    backward_masks = torch.zeros((C, H, W), dtype=torch.uint8, device=device)
    survival_mask = layers[-1]
    backward_masks[-1] = survival_mask

    for i in tqdm(range(C - 2, -1, -1), desc="GPU 逆向筛选"):
        has_future = F.max_pool2d(
            survival_mask.float().view(1, 1, H, W),
            kernel_size=tracking_window,
            stride=1,
            padding=pad,
        )
        survival_mask = (layers[i] > 0) & (has_future.view(H, W) > 0)
        backward_masks[i] = survival_mask
        if not survival_mask.any():
            print(f"\n⚠️ 在第 {i} 层失去所有贯穿路径。")
            return None, {}, {}

    # --- 正向提取 ---
    y0, x0 = torch.where(backward_masks[0])
    num_points = len(y0)
    current_id_map = torch.zeros((H, W), dtype=torch.int32, device=device)
    current_id_map[y0, x0] = torch.arange(1, num_points + 1, dtype=torch.int32, device=device)

    all_coords = torch.full((C, num_points, 2), -1, dtype=torch.int16, device=device)
    all_coords[0, :, 0] = y0.short()
    all_coords[0, :, 1] = x0.short()

    for i in tqdm(range(1, C), desc="GPU 正向提取"):
        can_be_reached_ids = (
            F.max_pool2d(
                current_id_map.float().view(1, 1, H, W),
                kernel_size=tracking_window,
                stride=1,
                padding=pad,
            )
            .int()
            .view(H, W)
        )

        current_mask = (backward_masks[i] > 0) & (can_be_reached_ids > 0)
        next_id_map = torch.where(current_mask, can_be_reached_ids, torch.zeros_like(current_id_map))

        curr_y, curr_x = torch.where(next_id_map > 0)
        curr_ids = next_id_map[curr_y, curr_x]
        if curr_ids.numel() > 0:
            all_coords[i, (curr_ids - 1).long(), 0] = curr_y.short()
            all_coords[i, (curr_ids - 1).long(), 1] = curr_x.short()

        current_id_map = next_id_map

    # --- 后处理 → CPU 字典 ---
    final_mask = torch.zeros((C, H, W), dtype=torch.bool, device=device)
    for i in range(C):
        valid = all_coords[i, :, 0] >= 0
        final_mask[i, all_coords[i, valid, 0].long(), all_coords[i, valid, 1].long()] = True

    res_coords = all_coords.cpu().numpy()
    y0_cpu, x0_cpu = y0.cpu().numpy(), x0.cpu().numpy()

    id_to_key = {i + 1: (int(y0_cpu[i]), int(x0_cpu[i])) for i in range(num_points)}
    coords_dict = {}
    for i in range(num_points):
        key = id_to_key[i + 1]
        coords_dict[key] = [
            (j, int(res_coords[j, i, 0]), int(res_coords[j, i, 1]))
            for j in range(C)
            if res_coords[j, i, 0] >= 0
        ]

    return final_mask.permute(1, 2, 0).cpu().numpy(), coords_dict, id_to_key
