"""
光谱数据集定义

提供三元组数据集 (TripletDataset) 和通用光谱数据集 (SpectralDataset)。
支持从 .npy 文件或通过函数获取光谱数据。

注意:
    - TripletDataset 默认使用所有目标类别（修复旧代码只使用 target class 1 的 bug）
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import NUM_BANDS


class SpectralDataset(Dataset):
    """
    通用光谱数据集。

    对每个样本进行 Z-score 标准化。
    支持选择部分波段。

    参数:
        data: (N, input_dim) 光谱数据
        input_dim: 使用的波段数（默认 60）
    """

    def __init__(self, data: np.ndarray, input_dim: int = 60):
        self.data = data[:, :input_dim].copy()
        np.random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.data[idx]
        x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.tensor(x, dtype=torch.float32)


class PairSpectralDataset(Dataset):
    """
    返回 (x, x) 对的光谱数据集（用于自编码器）。

    参数:
        data: (N, input_dim) 光谱数据
    """

    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        return x, x


def get_background_data(
    background_path: str = "background.npy",
    target_paths: Optional[List[str]] = None,
) -> np.ndarray:
    """
    加载背景光谱数据。

    参数:
        background_path: 背景数据文件路径 (.npy)
        target_paths: 目标数据文件路径列表（可选，合并作为背景）

    返回:
        data: (N, 93) 光谱数据
    """
    data = np.load(background_path)
    if target_paths:
        for path in target_paths:
            try:
                t = np.load(path)
                data = np.vstack([data, t])
            except (FileNotFoundError, ValueError):
                print(f"⚠️ 无法加载 {path}")
    return data


def get_target_data(
    paths: Optional[Dict[int, str]] = None,
) -> Dict[int, np.ndarray]:
    """
    加载目标光谱数据。

    修复: 旧代码只使用 target class 1，这里加载所有类别。

    参数:
        paths: 类别编号到文件路径的映射，如 {1: "target1.npy", 2: "target2.npy", ...}

    返回:
        targets: {class_id: (N, 93) 光谱数据}
    """
    if paths is None:
        paths = {1: "target1.npy", 2: "target2.npy", 3: "target3.npy"}

    targets = {}
    for cls_id, path in paths.items():
        try:
            data = np.load(path)
            targets[cls_id] = data
            print(f"✅ 加载目标类别 {cls_id}: {data.shape}")
        except FileNotFoundError:
            print(f"⚠️ 目标类别 {cls_id} 文件 {path} 不存在，跳过")
    return targets


class TripletDataset(Dataset):
    """
    三元组数据集 (Anchor, Positive, Negative)。

    用于训练 SpectralEmbeddingNet 的对比嵌入。
    每个三元组包含一个锚点、一个正样本（同类别）和一个负样本（不同类别）。

    参数:
        background_path: 背景数据路径
        target_paths: 目标数据路径映射
        input_dim: 使用波段数
        neg_ratio: 负样本采样比例
    """

    def __init__(
        self,
        background_path: str = "background.npy",
        target_paths: Optional[Dict[int, str]] = None,
        input_dim: int = NUM_BANDS,
        neg_ratio: float = 1.0,
    ):
        self.input_dim = input_dim

        # 加载背景（作为负样本）
        self.background = np.load(background_path)[:, :input_dim]

        # 加载所有目标类别（修复: 旧代码只取 target class 1）
        self.targets = get_target_data(target_paths)

        # 从每个目标类别中采样
        all_target_data = []
        for cls_id, data in self.targets.items():
            sampled = random.sample(
                list(data[:, :input_dim]),
                min(int(len(data) * 0.3), 500),
            )
            all_target_data.extend(sampled)

        self.target = np.array(all_target_data)
        print(f"✅ TripletDataset: 背景 {len(self.background)}, "
              f"目标 (所有类别) {len(self.target)}")

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = self.target[idx]

        # 正样本: 同类别的另一个样本
        positive_idx = random.randint(0, len(self.target) - 1)
        positive = self.target[positive_idx]

        # 负样本: hard negative 采样（与锚点最相似的背景样本）
        negative = self._hard_negative_sampling(anchor)

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(negative, dtype=torch.float32),
        )

    def _hard_negative_sampling(self, anchor: np.ndarray) -> np.ndarray:
        """从背景中采样与锚点最相似的 hard negative。"""
        # 余弦相似度
        anchor_norm = anchor / (np.linalg.norm(anchor) + 1e-8)
        bg_norm = self.background / (
            np.linalg.norm(self.background, axis=1, keepdims=True) + 1e-8
        )
        similarities = bg_norm @ anchor_norm

        # 选择 top-10% 最相似的背景样本
        k = max(1, len(self.background) // 10)
        top_k_idx = np.argsort(similarities)[-k:]
        chosen = np.random.choice(top_k_idx)
        return self.background[chosen]
