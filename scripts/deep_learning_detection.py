#!/usr/bin/env python3
"""
深度学习高光谱目标检测 — 与 ACE/SACE 对比测试。

实现三种方法并对比:
  1. ACE         — 自适应余弦估计 (传统基线)
  2. SACE        — 光谱角约束能量最小化 (传统最佳)
  3. SpectralAE  — 自编码器异常检测 (无监督深度学习)
  4. Spectral1DCNN — 1维CNN分类器 (有监督深度学习)

所有方法使用相同的输入数据: 反射率 + 均值归一化 + 同一套饱和波段过滤。

输出:
  output/deep_learning/ — 对比结果
    ├── comparison_table.txt     — 定量指标对比
    ├── roc_curves.png           — ROC 曲线 (有真值标注点)
    ├── score_maps.png           — 所有方法的 Score Map 对比
    ├── score_distributions.png  — 分数分布对比
    ├── detection_overlays.png   — 检测结果叠加图
    └── spectra_analysis.png     — 目标/背景光谱 + AE 重建误差分析

用法:
  python scripts/deep_learning_detection.py --dx 195 --dy -30
  python scripts/deep_learning_detection.py --filter full --target-dir output/filtered_targets
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# 添加项目根
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import (
    subtract_dark_current,
    compute_reflectance,
    detect_saturated_bands,
    normalize_reflectance,
)
from detection.ace import ACEDetector
from detection.sace import SACEDetector
from noise_filter import NotchFilter, analyze_fft

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ───── 常量 ─────
WAVELENGTHS = np.arange(445, 906, 5, dtype=int)
RECT_H, RECT_W = 6, 53
MIN_AREA = 1117
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 设备: {DEVICE}")


# ═══════════════════════════════════════════════════════════
# 1. 数据加载 (与 run_real_pipeline.py 完全相同)
# ═══════════════════════════════════════════════════════════

def load_images(data_dir: str) -> dict:
    """加载 4 张核心 TIF 图像。"""
    paths = {
        "spec_base": os.path.join(data_dir, "5ms.tif"),
        "dark": os.path.join(data_dir, "P11070000.tif"),
        "illuminance": os.path.join(data_dir, "5ms_sky.tif"),
        "gray": os.path.join(data_dir, "view2.tif"),
    }
    images = {}
    for name, p in paths.items():
        if not os.path.exists(p):
            print(f"❌ 文件不存在: {p}")
            sys.exit(1)
        img = tifffile.imread(p)
        print(f"  {name:>15}: {img.shape}, dtype={img.dtype}")
        images[name] = img
    return images


def compute_reflectance_cube(images: dict, notch_filter: NotchFilter = None,
                             filter_level: str = 'none') -> np.ndarray:
    """计算全图反射率: (spec - dark) / (sky - dark)。可选 notch 滤波。"""
    img_spec = subtract_dark_current(images["spec_base"], images["dark"])
    img_sky = subtract_dark_current(images["illuminance"], images["dark"])

    # Level 1: Sky 预滤波
    if notch_filter and filter_level in ('sky', 'full'):
        print("  🌀 Level 1: Sky 2D notch filtering...")
        analyze_fft(img_sky, "Sky 滤波前")
        img_sky = notch_filter.filter_image_2d(img_sky)
        analyze_fft(img_sky, "Sky 滤波后")

    reflect = compute_reflectance(img_spec, img_sky)

    # Level 2: Reflectance 逐波段滤波
    if notch_filter and filter_level in ('reflectance', 'full'):
        print("  🌀 Level 2: Reflectance 逐波段1D notch filtering...")
        analyze_fft(reflect, "Reflectance 滤波前")
        ref_3d = reflect[:, :, np.newaxis]
        ref_clean = notch_filter.filter_reflectance_cube(ref_3d)
        reflect = ref_clean[:, :, 0]
        analyze_fft(reflect, "Reflectance 滤波后")

    print(f"  反射率: 形状={reflect.shape}, 范围=[{reflect.min():.4f},{reflect.max():.1f}]")
    return reflect


def extract_spectral_vectors(reflect: np.ndarray, hi_dir: str) -> tuple:
    """通过 coords_dict.json 提取所有标定点的光谱向量。"""
    coords_path = os.path.join(hi_dir, "coords_dict.json")
    with open(coords_path, "r") as f:
        coords_dict = json.load(f)

    n_bands = 93
    valid_items = [(idx_str, spec) for idx_str, spec in coords_dict.items()
                   if len(spec) == n_bands]
    n_points = len(valid_items)
    data_vector = np.zeros((n_points, n_bands), dtype=np.float64)
    first_coords = np.zeros((n_points, 2), dtype=int)

    for i, (idx_str, spec) in enumerate(valid_items):
        row = np.array([reflect[s[1], s[2]] for s in spec], dtype=np.float64)
        data_vector[i, :] = row
        first_coords[i] = [spec[0][1], spec[0][2]]

    print(f"  光谱向量: ({n_points}, {n_bands})")
    return data_vector, first_coords


def load_target_templates(hi_dir: str, target_dir: str = None) -> dict:
    """加载 target{1-3}.npy。"""
    targets = {}
    for i in [1, 2, 3]:
        path = None
        if target_dir:
            p = os.path.join(target_dir, f"target{i}.npy")
            if os.path.exists(p):
                path = p
        if path is None:
            path = os.path.join(hi_dir, f"target{i}.npy")
        if os.path.exists(path):
            t = np.load(path)
            targets[i] = t
    return targets


def get_labels_from_mask(first_coords: np.ndarray, hi_dir: str) -> np.ndarray:
    """
    通过 mask.npy 获取每个标定点的真值标签。

    返回:
        labels: (N,)  0=背景, 1=目标1(草地伪装网), 2=目标2(军绿迷彩), 3=目标3(沙漠迷彩)
    """
    mask_path = os.path.join(hi_dir, "dataset", "mask.npy")
    if not os.path.exists(mask_path):
        print(f"⚠️ mask.npy 不存在: {mask_path}，跳过标签获取")
        return None

    mask = np.load(mask_path)
    # mask 值: 4→目标1, 5→目标2, 6→目标3
    mapping = {4: 1, 5: 2, 6: 3}
    label_map = np.vectorize(lambda x: mapping.get(x, 0))(mask)

    labels = np.zeros(len(first_coords), dtype=np.int32)
    for i, (y, x) in enumerate(first_coords):
        if 0 <= y < label_map.shape[0] and 0 <= x < label_map.shape[1]:
            labels[i] = label_map[y, x]

    n_target = (labels > 0).sum()
    print(f"  mask.npy 标注: 背景={len(labels)-n_target}, 目标={n_target}")
    for c in [1, 2, 3]:
        print(f"    目标{c}: {(labels==c).sum()} 点")
    return labels


def filter_and_normalize(data: np.ndarray, targets: dict, labels: np.ndarray = None):
    """
    饱和波段过滤 + 均值归一化。
    targets 和 labels 的数据维度会与 data 同步过滤。
    """
    good, bad = detect_saturated_bands(data, threshold_ratio=10.0)
    if len(bad) > 0:
        wl_str = f"{WAVELENGTHS[good[0]]}-{WAVELENGTHS[good[-1]]}nm"
        print(f"  → 保留波段: [{good[0]}-{good[-1]}] = {wl_str}")

    data_f = data[:, good]
    targets_f = {}
    for i, t in targets.items():
        targets_f[i] = t[:, good] if t.shape[1] == 93 else t

    data_n = normalize_reflectance(data_f, method="mean")
    targets_n = {}
    for i, t in targets_f.items():
        targets_n[i] = normalize_reflectance(t, method="mean")

    labels_n = labels.copy() if labels is not None else None
    return data_n, targets_n, labels_n, good


def stratified_split(labels: np.ndarray, test_ratio: float = 0.15,
                     val_ratio: float = 0.15, seed: int = 42) -> tuple:
    """
    分层划分训练/验证/测试集，保持各类比例一致。

    返回:
        train_idx, val_idx, test_idx: 三个索引数组
    """
    from sklearn.model_selection import train_test_split
    np.random.seed(seed)

    # 用是否为目标作分层 (0=背景, 1=目标)
    y_strat = (labels > 0).astype(np.int32)

    # 先分出测试集
    train_val_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=test_ratio,
        random_state=seed,
        stratify=y_strat,
    )

    # 从剩余中分出验证集
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    y_train_val = y_strat[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adjusted,
        random_state=seed + 1,
        stratify=y_train_val,
    )

    # 统计
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        n_total = len(idx)
        n_tgt = (labels[idx] > 0).sum()
        print(f"  {name:>5}: {n_total:5d} samples ({n_tgt} targets, {n_tgt/n_total*100:.1f}%)")

    return train_idx, val_idx, test_idx




def extract_spatial_patches(reflect: np.ndarray, first_coords: np.ndarray,
                             good_bands: np.ndarray, hi_dir: str,
                             patch_size: int = 5) -> np.ndarray:
    '''
    从全图反射率中提取每个标定点为中心的 NxN 空间邻域 patch。

    参数:
        reflect: (H, W) 全图反射率 (单波段, 因为波段过已滤后操作)
        first_coords: (N, 2) 每个标定点的 (y, x)
        good_bands: (K,) 保留的波段索引
        hi_dir: HI 目录路径 (用于加载 coords_dict.json 获取每波段坐标)
        patch_size: patch 大小 (奇数)

    返回:
        patches: (N, K, patch_size, patch_size) 光谱-空间 patch
    '''
    # 需要加载 coords_dict.json 获取每个点在各波段的精确坐标
    import json
    coords_path = os.path.join(hi_dir, 'coords_dict.json')
    with open(coords_path, 'r') as f:
        coords_dict = json.load(f)

    n_points = len(first_coords)
    n_bands = len(good_bands)
    pad = patch_size // 2
    H, W = reflect.shape

    patches = np.zeros((n_points, n_bands, patch_size, patch_size), dtype=np.float64)

    # 构建 idx_str -> band 坐标的映射
    # coords_dict: {idx_str: [[band, y, x], ...]} for 93 bands
    idx_map = {}
    for idx_str, specs in coords_dict.items():
        if len(specs) == 93:
            spec_dict = {}
            for b, y, x in specs:
                spec_dict[b] = (y, x)
            idx_map[idx_str] = spec_dict

    # Map first_coords to idx_str
    # id_to_key.json maps (y,x) -> idx
    id_to_key_path = os.path.join(hi_dir, 'id_to_key.json')
    with open(id_to_key_path, 'r') as f:
        id_to_key = json.load(f)

    # Build reverse map: (y,x) -> idx_str
    yx_to_idx = {}
    for key, val in id_to_key.items():
        y, x = eval(key)
        yx_to_idx[(y, x)] = str(val)

    for i, (y, x) in enumerate(first_coords):
        idx_str = yx_to_idx.get((int(y), int(x)))
        if idx_str is None or idx_str not in idx_map:
            continue

        band_coords = idx_map[idx_str]
        for bi, orig_band_idx in enumerate(good_bands):
            # band index in coords_dict is 1-based
            band_key = orig_band_idx + 1
            if band_key in band_coords:
                cy, cx = band_coords[band_key]
                # Extract patch with reflection padding at boundaries
                y1 = max(0, cy - pad)
                y2 = min(H, cy + pad + 1)
                x1 = max(0, cx - pad)
                x2 = min(W, cx + pad + 1)

                patch = reflect[y1:y2, x1:x2]

                # Pad if at boundary
                py1 = max(0, pad - cy)
                py2 = patch_size - max(0, cy + pad + 1 - H)
                px1 = max(0, pad - cx)
                px2 = patch_size - max(0, cx + pad + 1 - W)

                if patch.size > 0:
                    patches[i, bi, py1:py2, px1:px2] = patch

    print(f'  Patches: ({n_points}, {n_bands}, {patch_size}, {patch_size})')
    return patches


def run_spatial_cnn_detection(model: SpectralSpatialCNN, patches: np.ndarray) -> np.ndarray:
    '''用训练好的光谱-空间 CNN 预测目标概率。'''
    model.eval()
    data_t = torch.from_numpy(patches).float().to(DEVICE)
    with torch.no_grad():
        scores = model(data_t).cpu().numpy()
    return scores


def train_spatial_cnn(
    patches: np.ndarray,
    train_labels: np.ndarray,
    n_bands: int,
    train_idx: np.ndarray = None,
    val_idx: np.ndarray = None,
    patch_size: int = 5,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 64,
    val_split: float = 0.2,
    patience: int = 30,
    use_focal: bool = True,
) -> SpectralSpatialCNN:
    '''
    训练光谱-空间 2D-CNN 分类器。

    参数:
        patches: (N, C, H, W) 光谱-空间 patch 数据
        train_labels: (N,) 标签
        use_focal: 使用 FocalLoss (True) 或 加权 BCE (False)
    '''
    y_binary = (train_labels > 0).astype(np.float32)

    if train_idx is None or val_idx is None:
        n = len(patches)
        indices = np.random.permutation(n)
        n_val = int(n * val_split)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        print(f'    (internal split: train={len(train_idx)}, val={len(val_idx)})')

    n_target = int(y_binary[train_idx].sum())
    n_bg = len(train_idx) - n_target
    n_train = len(train_idx)
    print(f'  Training SpectralSpatialCNN (训练={n_train}, 目标={n_target}, 背景={n_bg}, '
          f'focal={use_focal}, patch={patch_size}x{patch_size})...')

    # Z-score per band
    train_patches = patches[train_idx].copy()
    val_patches = patches[val_idx].copy()
    for bi in range(n_bands):
        band_data = train_patches[:, bi, :, :]
        m = band_data.mean()
        s = band_data.std() + 1e-8
        train_patches[:, bi] = (train_patches[:, bi] - m) / s
        val_patches[:, bi] = (val_patches[:, bi] - m) / s

    X_train = torch.from_numpy(train_patches).float()
    y_train = torch.from_numpy(y_binary[train_idx]).float()
    X_val = torch.from_numpy(val_patches).float()
    y_val = torch.from_numpy(y_binary[val_idx]).float()

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    model = SpectralSpatialCNN(n_bands, patch_size).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    if use_focal:
        # alpha for FocalLoss: balanced weight for POSITIVE class (rare class = higher weight)
        alpha_pos = n_bg / (n_bg + max(n_target, 1))
        print(f'  Using FocalLoss: alpha_pos={alpha_pos:.4f}, gamma=2.0')
        criterion = FocalLoss(alpha=float(alpha_pos), gamma=2.0)
    else:
        pos_weight_val = n_bg / max(n_target, 1)
        base_criterion = nn.BCELoss(reduction='none')
        def criterion(pred, yb):
            loss = base_criterion(pred, yb)
            weights = torch.where(yb > 0.5, float(pos_weight_val), 1.0)
            return (loss * weights).mean()

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    t0 = __import__('time').time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f'    Epoch {epoch:4d}/{epochs}, '
                  f'Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, '
                  f'LR={optimizer.param_groups[0]["lr"]:.2e}')

        if no_improve >= patience:
            print(f'    Early stop at Epoch {epoch}')
            break

    model.load_state_dict(best_state)
    t_elapsed = __import__('time').time() - t0
    print(f'  SpectralSpatialCNN done (best_val_loss={best_val_loss:.4f}, {t_elapsed:.1f}s)')
    return model


# ═══════════════════════════════════════════════════════════
# 2. 深度学习模型定义
# ═══════════════════════════════════════════════════════════

class SpectralAE(nn.Module):
    """
    光谱自编码器 — 异常检测。
    训练: 只学习重建背景光谱
    检测: 重建误差 = anomaly score (越高越像目标)
    """
    def __init__(self, input_dim: int = 93):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def anomaly_score(self, x):
        """返回每个样本的重建 MSE (越高越异常/越像目标)。"""
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            # 逐像素 MSE
            mse = torch.mean((x - recon) ** 2, dim=1)
        return mse.cpu().numpy()

class Spectral1DCNN(nn.Module):
    """
    1D-CNN 光谱分类器 — 有监督检测。
    输入: (B, 1, D) — 一维光谱
    输出: (B, 1) — 目标概率
    """
    def __init__(self, input_dim: int = 79):  # 默认 79 波段(过滤后)
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # 计算 flatten 后的维度
        self._to_linear = None
        self._get_conv_output(input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _get_conv_output(self, D):
        """前向传播 conv 部分，确定 flatten 后维度。"""
        x = torch.zeros(1, 1, D)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        # x shape: (B, 1, D)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(1)


# ═══════════════════════════════════════════════════════════
# 2b. 增强版 SpectralAE (瓶颈=8, L1正则, Dropout)
# ═══════════════════════════════════════════════════════════

class SpectralAEv2(nn.Module):
    """
    增强版光谱自编码器 — 更紧瓶颈 + L1稀疏约束。
    架构: 93/79 → 64 → 32 → 8(瓶颈) → 32 → 64 → 93/79
    """
    def __init__(self, input_dim: int = 79, bottleneck_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(32, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def get_encoding(self, x):
        """返回瓶颈层编码 (用于 L1 正则)。"""
        return self.encoder(x)

    def anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            mse = torch.mean((x - recon) ** 2, dim=1)
        return mse.cpu().numpy()


# ═══════════════════════════════════════════════════════════
# 2c. InfoNCE 对比学习 + Mahalanobis 检测器
# ═══════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss -- auto-focus on hard samples.
    FL(p_t) = -alpha_t (1-p_t)^gamma log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(1e-8, 1 - 1e-8)
        pt = torch.where(target > 0.5, pred, 1 - pred)
        alpha_t = torch.where(target > 0.5, self.alpha, 1.0 - self.alpha)
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt)
        return loss.mean()


class SpectralSpatialCNN(nn.Module):
    """
    Spectral-Spatial 2D-CNN detector.
    Input: (B, C, H, W) where C=bands, H=W=patch_size
    """
    def __init__(self, n_bands: int = 79, patch_size: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_bands, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        f = self.features(x).view(x.size(0), -1)
        return self.classifier(f).squeeze(1)



class SpectralInfoNCE(nn.Module):
    """
    InfoNCE 对比学习编码器。
    训练后: 用 Mahalanobis 距离检测异常 (目标=异常=远离背景分布)。

    架构参考: HI/对比学习.py
    输入 → 128 → ReLU → 64 → ReLU → 16 → L2-Norm
    """
    def __init__(self, input_dim: int = 79, emb_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return nn.functional.normalize(z, dim=1)  # L2归一化

    def encode_np(self, x: np.ndarray) -> np.ndarray:
        """numpy 输入 → numpy 嵌入。"""
        self.eval()
        with torch.no_grad():
            z = self.forward(torch.from_numpy(x).float().to(DEVICE))
        return z.cpu().numpy()


class InfoNCELoss(nn.Module):
    """SimCLR-style InfoNCE 损失。"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.matmul(z, z.T) / self.temp  # (2B, 2B)
        # 标签: z1 的正样本是 z2 的对应行, z2 的正样本是 z1 的对应行
        labels = torch.arange(B, 2 * B, device=z.device)
        labels = torch.cat([labels, torch.arange(B, device=z.device)], dim=0)
        return nn.functional.cross_entropy(sim, labels)


def augment_spectra(x: torch.Tensor) -> torch.Tensor:
    """光谱数据增强: 噪声 + 缩放 + 循环位移。"""
    noise = torch.randn_like(x) * 0.01
    scale = torch.rand(x.size(0), 1, device=x.device) * 0.1 + 0.95  # [0.95, 1.05]
    shift = torch.roll(x, 1, dims=1) * 0.01
    return x * scale + noise + shift


def train_info_nce(
    bg_data: np.ndarray,
    input_dim: int,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    patience: int = 20,
) -> SpectralInfoNCE:
    """
    训练 InfoNCE 对比学习编码器 (仅用背景数据)。
    """
    print(f"\n  🔧 训练 InfoNCE (背景={len(bg_data)}样本, {epochs}epochs)...")

    dataset = TensorDataset(torch.from_numpy(bg_data).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SpectralInfoNCE(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = InfoNCELoss(temperature=0.07)

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for (x,) in loader:
            x = x.to(DEVICE)
            # 两次独立增强
            x1 = augment_spectra(x)
            x2 = augment_spectra(x)
            z1 = model(x1)
            z2 = model(x2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch:4d}/{epochs}, Loss={avg_loss:.4f}")

        if no_improve >= patience:
            print(f"    ✅ 提前停止于 Epoch {epoch}")
            break

    model.load_state_dict(best_state)
    t_elapsed = time.time() - t0
    print(f"  ✅ InfoNCE 训练完成 (best_loss={best_loss:.4f}, {t_elapsed:.1f}s)")
    return model


def run_info_nce_detection(model: SpectralInfoNCE, bg_data: np.ndarray,
                            data: np.ndarray) -> np.ndarray:
    """
    用训练好的 InfoNCE 模型做 Mahalanobis 距离异常检测。

    1. 嵌入背景数据 → 拟合高斯 (均值, 协方差)
    2. 嵌入测试数据 → 计算马氏距离
    3. 距离越远 → 越像目标 (异常)
    """
    from scipy.spatial.distance import mahalanobis

    model.eval()
    # 嵌入所有背景数据
    bg_emb = model.encode_np(bg_data)
    bg_mean = bg_emb.mean(axis=0)
    bg_cov = np.cov(bg_emb.T) + 1e-5 * np.eye(bg_emb.shape[1])
    try:
        bg_cov_inv = np.linalg.inv(bg_cov)
    except np.linalg.LinAlgError:
        bg_cov_inv = np.linalg.pinv(bg_cov)

    # 嵌入测试数据
    test_emb = model.encode_np(data)

    # 计算马氏距离
    diff = test_emb - bg_mean
    scores = np.sqrt(np.sum(diff @ bg_cov_inv * diff, axis=1))

    return scores


# ═══════════════════════════════════════════════════════════
# 3. 训练函数
# ═══════════════════════════════════════════════════════════

def train_autoencoder(
    bg_data: np.ndarray,
    input_dim: int,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    patience: int = 20,
) -> SpectralAE:
    """
    训练光谱自编码器 (仅用背景数据)。
    """
    print(f"\n  🔧 训练 SpectralAE (背景={len(bg_data)}样本, {epochs}epochs)...")

    dataset = TensorDataset(torch.from_numpy(bg_data).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SpectralAE(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for (x,) in loader:
            x = x.to(DEVICE)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch:4d}/{epochs}, Loss={avg_loss:.6f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

        if no_improve >= patience:
            print(f"    ✅ 提前停止于 Epoch {epoch}")
            break

    model.load_state_dict(best_state)
    t_elapsed = time.time() - t0
    print(f"  ✅ AE 训练完成 (best_loss={best_loss:.6f}, {t_elapsed:.1f}s)")
    return model


def train_autoencoder_v2(
    bg_data: np.ndarray,
    input_dim: int,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    patience: int = 20,
    l1_lambda: float = 1e-5,
    bottleneck_dim: int = 8,
) -> SpectralAEv2:
    """
    训练增强版光谱自编码器 (更紧瓶颈 + L1稀疏正则)。
    """
    print(f"\n  🔧 训练 SpectralAEv2 (瓶颈={bottleneck_dim}, L1λ={l1_lambda:.1e}, "
          f"背景={len(bg_data)}样本)...")

    dataset = TensorDataset(torch.from_numpy(bg_data).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SpectralAEv2(input_dim, bottleneck_dim=bottleneck_dim, dropout=0.2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for (x,) in loader:
            x = x.to(DEVICE)
            recon = model(x)
            mse_loss = criterion(recon, x)
            # L1 稀疏正则: 对编码器输出加 L1 惩罚
            z = model.get_encoding(x)
            l1_loss = torch.mean(torch.abs(z))
            loss = mse_loss + l1_lambda * l1_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch:4d}/{epochs}, MSE={avg_loss:.6f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

        if no_improve >= patience:
            print(f"    ✅ 提前停止于 Epoch {epoch}")
            break

    model.load_state_dict(best_state)
    t_elapsed = time.time() - t0
    print(f"  ✅ SpectralAEv2 训练完成 (best_loss={best_loss:.6f}, {t_elapsed:.1f}s)")
    return model


def train_1dcnn(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    input_dim: int,
    train_idx: np.ndarray = None,
    val_idx: np.ndarray = None,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 128,
    val_split: float = 0.2,
    patience: int = 30,
) -> Spectral1DCNN:
    """
    训练 1D-CNN 分类器 (有监督)。
    输入数据会转换为 (N, 1, D) 形状。
    """
    # 二值化标签: 0=背景, 1=目标
    y_binary = (train_labels > 0).astype(np.float32)

    # 如果外部未提供索引，内部随机分割
    if train_idx is None or val_idx is None:
        n = len(train_data)
        indices = np.random.permutation(n)
        n_val = int(n * val_split)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        print(f"    (internal split: train={len(train_idx)}, val={len(val_idx)})")

    # 统计类别权重 (处理类别不平衡: 目标极少, 给予更高权重)
    n_train = len(train_idx)
    n_target = int(y_binary[train_idx].sum())
    n_bg = n_train - n_target
    pos_weight_val = n_bg / max(n_target, 1)
    print(f"  🔧 训练 1D-CNN (训练={n_train}, 目标={n_target}, 背景={n_bg}, "
          f"正样本权重={pos_weight_val:.1f})...")

    # 数据集
    X_train = torch.from_numpy(train_data[train_idx]).float().unsqueeze(1)  # (N, 1, D)
    y_train = torch.from_numpy(y_binary[train_idx]).float()
    X_val = torch.from_numpy(train_data[val_idx]).float().unsqueeze(1)
    y_val = torch.from_numpy(y_binary[val_idx]).float()

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = Spectral1DCNN(input_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    # 带权重的 BCE Loss (模型输出为 sigmoid 概率, 使用 BCELoss reduction='none' 以便加权)
    criterion = nn.BCELoss(reduction='none')

    # 类别权重: 正样本(目标)损失权重更高

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            # 加权 BCE: 正样本权重 = pos_weight_val
            loss = criterion(pred, yb)
            # 对正样本加权 (yb==1 时 loss 乘以 pos_weight_val)
            if pos_weight_val > 1:
                weights = torch.where(yb > 0.5, pos_weight_val, 1.0)
                loss = (loss * weights).mean()
            else:
                loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                if pos_weight_val > 1:
                    weights = torch.where(yb > 0.5, pos_weight_val, 1.0)
                    loss = (loss * weights).mean()
                else:
                    loss = loss.mean()
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch:4d}/{epochs}, "
                  f"Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

        if no_improve >= patience:
            print(f"    ✅ 提前停止于 Epoch {epoch}")
            break

    model.load_state_dict(best_state)
    t_elapsed = time.time() - t0
    print(f"  ✅ 1D-CNN 训练完成 (best_val_loss={best_val_loss:.4f}, {t_elapsed:.1f}s)")
    return model


# ═══════════════════════════════════════════════════════════
# 4. 检测函数
# ═══════════════════════════════════════════════════════════

def run_ace(data: np.ndarray, targets: dict) -> np.ndarray:
    """运行 ACE 检测器，返回每个像素的 max-over-targets 分数。"""
    scores_multi = np.zeros((data.shape[0], 3))
    for ti in range(3):
        tgt = targets[ti + 1].mean(axis=0)
        det = ACEDetector(reg=1e-6)
        det.fit(data, tgt)
        scores_multi[:, ti] = det.predict(data)
    return scores_multi.max(axis=1)


def run_sace(data: np.ndarray, targets: dict) -> np.ndarray:
    """运行 SACE 检测器。"""
    scores_multi = np.zeros((data.shape[0], 3))
    for ti in range(3):
        tgt = targets[ti + 1].mean(axis=0)
        det = SACEDetector(reg=1e-6)
        det.fit(data, tgt)
        scores_multi[:, ti] = det.predict(data)
    return scores_multi.max(axis=1)


def run_ae_detection(model: SpectralAE, data: np.ndarray) -> np.ndarray:
    """用训练好的 AE 计算异常分数。"""
    data_t = torch.from_numpy(data).float().to(DEVICE)
    scores = model.anomaly_score(data_t)
    return scores


def run_cnn_detection(model: Spectral1DCNN, data: np.ndarray) -> np.ndarray:
    """用训练好的 1D-CNN 预测目标概率。"""
    model.eval()
    data_t = torch.from_numpy(data).float().unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        scores = model(data_t).cpu().numpy()
    return scores


# ═══════════════════════════════════════════════════════════
# 4b. 无监督增强算法
# ═══════════════════════════════════════════════════════════

def compute_spectral_derivative(data: np.ndarray, deriv: int = 1,
                                 window: int = 7, polyorder: int = 2) -> np.ndarray:
    """光谱导数 via SavGol. 一阶导数消除光照, 放大吸收特征差异."""
    from scipy.signal import savgol_filter
    return savgol_filter(data, window_length=window, polyorder=polyorder,
                          deriv=deriv, axis=1)


def run_rx(data: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """RX anomaly detector: Mahalanobis distance from background. No target needed."""
    mean = np.mean(data, axis=0)
    Xc = data - mean
    R = (Xc.T @ Xc) / data.shape[0] + reg * np.eye(data.shape[1])
    R_inv = np.linalg.inv(R)
    return np.sum(Xc @ R_inv * Xc, axis=1)


def run_ssrx(data: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Subspace RX: PCA whiten then L2 distance. Suppresses background noise."""
    from sklearn.decomposition import PCA
    data_pca = PCA(n_components=n_components, whiten=True).fit_transform(data)
    return np.sum(data_pca ** 2, axis=1)


def run_derivative_ace(data: np.ndarray, targets: dict,
                        deriv: int = 1, window: int = 7) -> np.ndarray:
    """ACE in spectral derivative space. Amplifies material differences."""
    from detection.ace import ACEDetector
    data_d = compute_spectral_derivative(data, deriv=deriv, window=window)
    scores = np.zeros((data.shape[0], 3))
    for ti in range(3):
        t = targets[ti + 1]
        t_d = compute_spectral_derivative(t, deriv=deriv, window=window)
        det = ACEDetector(reg=1e-6)
        det.fit(data_d, t_d.mean(axis=0))
        scores[:, ti] = det.predict(data_d)
    return scores.max(axis=1)


def run_derivative_sace(data: np.ndarray, targets: dict,
                          deriv: int = 1, window: int = 7) -> np.ndarray:
    """SACE in spectral derivative space."""
    from detection.sace import SACEDetector
    data_d = compute_spectral_derivative(data, deriv=deriv, window=window)
    scores = np.zeros((data.shape[0], 3))
    for ti in range(3):
        t = targets[ti + 1]
        t_d = compute_spectral_derivative(t, deriv=deriv, window=window)
        det = SACEDetector(reg=1e-6)
        det.fit(data_d, t_d.mean(axis=0))
        scores[:, ti] = det.predict(data_d)
    return scores.max(axis=1)


def run_ensemble_unsupervised(data: np.ndarray, targets: dict) -> np.ndarray:
    """Ensemble of ACE, DerivACE, RX, SSRX, SAM. Weighted min-max fusion."""
    from detection.sam import SpectralAngleMapper as SAMDetector
    all_scores = {}
    all_scores["ACE"] = run_ace(data, targets)
    all_scores["DerivACE"] = run_derivative_ace(data, targets, deriv=1)
    all_scores["Deriv2ACE"] = run_derivative_ace(data, targets, deriv=2)
    all_scores["RX"] = run_rx(data)
    all_scores["SSRX"] = run_ssrx(data)
    sm = np.zeros((data.shape[0], 3))
    for ti in range(3):
        det = SAMDetector(normalize=True)
        det.fit(targets[ti + 1].mean(axis=0)[np.newaxis, :])
        sm[:, ti] = 1.0 - det.predict(data) / np.pi
    all_scores["SAM"] = sm.max(axis=1)
    weights = {"ACE": 1.0, "DerivACE": 1.2, "Deriv2ACE": 0.6,
               "RX": 0.8, "SSRX": 0.6, "SAM": 0.5}
    ensemble = np.zeros(data.shape[0])
    tw = 0
    for name, s in all_scores.items():
        smin, smax = s.min(), s.max()
        normed = (s - smin) / (smax - smin + 1e-10)
        w = weights.get(name, 1.0)
        ensemble += w * normed
        tw += w
    return ensemble / tw


# ═══════════════════════════════════════════════════════════
# 5. Score Map 生成 (与 run_real_pipeline.py 一致)
# ═══════════════════════════════════════════════════════════

def generate_score_map(
    scores: np.ndarray,
    first_coords: np.ndarray,
    gray_shape: tuple,
    reg_offset: tuple = (195, -30),
) -> np.ndarray:
    """将检测分数映射到灰度图像空间。"""
    H, W = gray_shape
    dy, dx = reg_offset
    score_map = np.zeros((H, W), dtype=np.float64)

    order = np.argsort(first_coords[:, 1])
    coords_sorted = first_coords[order]
    scores_sorted = scores[order]

    for (y, x), s in zip(coords_sorted, scores_sorted):
        y_gray = y + dy
        x_gray = x + dx
        if y_gray < 0 or y_gray >= H or x_gray < 0 or x_gray >= W:
            continue
        y1 = max(0, min(y_gray, H - 1))
        y2 = min(y_gray + RECT_H, H)
        x1 = max(0, min(x_gray, W - 1))
        x2 = min(x_gray + RECT_W, W)
        score_map[y1:y2, x1:x2] = max(score_map[y1:y2, x1:x2].max(), s)

    return score_map


def filter_connected_components(score_map: np.ndarray, threshold: float) -> np.ndarray:
    """连通区域过滤。"""
    from scipy import ndimage as ndi
    binary = score_map > threshold
    labeled, num_features = ndi.label(binary, structure=np.ones((3, 3)))
    component_sizes = np.bincount(labeled.ravel())

    keep = np.zeros_like(binary, dtype=bool)
    kept_count = 0
    for label_id in range(1, num_features + 1):
        if label_id < len(component_sizes) and component_sizes[label_id] >= MIN_AREA:
            keep[labeled == label_id] = True
            kept_count += 1
    return keep


# ═══════════════════════════════════════════════════════════
# 6. 评估指标
# ═══════════════════════════════════════════════════════════

def compute_metrics(scores: np.ndarray, labels: np.ndarray = None) -> dict:
    """
    计算定量指标。
    如果有 labels (真值)，计算 ROC/AUC。
    """
    metrics = {
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }

    # 目标/背景分别统计
    if labels is not None:
        bg_mask = labels == 0
        tgt_mask = labels > 0
        metrics["bg_mean"] = float(scores[bg_mask].mean()) if bg_mask.any() else 0
        metrics["bg_std"] = float(scores[bg_mask].std()) if bg_mask.any() else 0
        metrics["tgt_mean"] = float(scores[tgt_mask].mean()) if tgt_mask.any() else 0
        metrics["tgt_std"] = float(scores[tgt_mask].std()) if tgt_mask.any() else 0
        metrics["separability"] = float(
            abs(metrics["tgt_mean"] - metrics["bg_mean"]) / max(metrics["bg_std"] + metrics["tgt_std"], 1e-10)
        )

        # ROC (延迟导入避免启动报错)
        try:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, thresholds_roc = roc_curve(labels > 0, scores)
            metrics["roc_fpr"] = fpr
            metrics["roc_tpr"] = tpr
            metrics["roc_thresholds"] = thresholds_roc
            metrics["auc"] = auc(fpr, tpr)
        except ImportError:
            print("  ⚠️ sklearn not available, skipping AUC computation")
            metrics["auc"] = 0.0

    return metrics


def evaluate_score_map(score_map: np.ndarray, thres: float) -> dict:
    """评估 score map 质量。"""
    binary = score_map > thres
    det_pixels = binary.sum()

    bg_mask = ~binary
    bg_vals = score_map[bg_mask]
    bg_mean = bg_vals.mean() if len(bg_vals) > 0 else 0
    bg_std = bg_vals.std() if len(bg_vals) > 0 else 0
    max_score = score_map.max()

    return {
        "det_pixels": int(det_pixels),
        "bg_mean": float(bg_mean),
        "bg_std": float(bg_std),
        "max_score": float(max_score),
        "score": det_pixels / 1000 + bg_std * 5000,
    }


# ═══════════════════════════════════════════════════════════
# 6b. 地真评估 (基于 mask.npy 的全图像素级对比)
# ═══════════════════════════════════════════════════════════

def load_ground_truth_mask(hi_dir: str, first_coords: np.ndarray,
                           gray_shape: tuple,
                           reg_offset: tuple = (195, -30)) -> np.ndarray:
    """
    加载 mask.npy 并在灰度图像空间中生成二值地真检测图。

    将每个标定点的 mask 标注 (4/5/6=目标) 通过同样的配准偏移+矩形膨胀
    映射到灰度图像空间，得到与 score_map 空间对齐的地真图。

    返回:
        gt_map: (H, W) bool, True=目标区域
    """
    mask_path = os.path.join(hi_dir, "dataset", "mask.npy")
    if not os.path.exists(mask_path):
        print(f"  ⚠️ mask.npy not found: {mask_path}")
        return None

    mask = np.load(mask_path)  # (2048, 2048), spectral camera space
    # 4/5/6 → target
    is_target = np.isin(mask, [4, 5, 6])

    H, W = gray_shape
    dy, dx = reg_offset
    gt_map = np.zeros((H, W), dtype=bool)

    n_target = 0
    for (y, x) in first_coords:
        if 0 <= y < is_target.shape[0] and 0 <= x < is_target.shape[1] and is_target[y, x]:
            n_target += 1
            y_gray = y + dy
            x_gray = x + dx
            if y_gray < 0 or y_gray >= H or x_gray < 0 or x_gray >= W:
                continue
            y1 = max(0, min(y_gray, H - 1))
            y2 = min(y_gray + RECT_H, H)
            x1 = max(0, min(x_gray, W - 1))
            x2 = min(x_gray + RECT_W, W)
            gt_map[y1:y2, x1:x2] = True

    print(f"  Ground truth: {n_target} target points → {gt_map.sum()} px in grayscale space")
    return gt_map


def compute_pixel_metrics(binary_det: np.ndarray, gt_map: np.ndarray) -> dict:
    """
    在完整图像空间计算像素级检测指标。

    参数:
        binary_det: (H, W) bool, 检测二值图
        gt_map: (H, W) bool, 地真二值图

    返回:
        metrics: TP, FP, FN, precision, recall, F1, IoU
    """
    TP = np.sum(binary_det & gt_map)
    FP = np.sum(binary_det & ~gt_map)
    FN = np.sum(~binary_det & gt_map)
    TN = np.sum(~binary_det & ~gt_map)

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    F1 = 2 * precision * recall / max(precision + recall, 1e-10)
    IoU = TP / max(TP + FP + FN, 1)

    return {
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(F1),
        "IoU": float(IoU),
    }


# ═══════════════════════════════════════════════════════════
# 7. 可视化
# ═══════════════════════════════════════════════════════════

def visualize_comparison(
    results: dict,
    gray_img: np.ndarray,
    gt_map: np.ndarray,
    output_dir: str,
    reg_offset: tuple,
):
    """Generate comparison plots (all English labels to avoid CJK font issues)."""
    method_names = [k for k in results.keys()
                    if k not in ("spectra", "labels", "gt_map") and "score_map" in results[k]]
    dy, dx = reg_offset
    H, W = gray_img.shape
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    n_methods = len(method_names)

    # ── Fig 1: Score Maps + Ground Truth ──
    n_cols = max(3, (n_methods + 1) // 2 + 1)
    fig, axes = plt.subplots(2, n_cols,
                             figsize=(6 * n_cols, 12))
    axes = axes.flatten()

    # First subplot: Ground Truth
    ax = axes[0]
    if gt_map is not None:
        gt_overlay = np.zeros((H, W, 4))
        gt_overlay[..., 1] = 1.0  # green
        gt_overlay[..., 3] = gt_map.astype(float) * 0.5
        ax.imshow(gray_img, cmap="gray")
        ax.imshow(gt_overlay)
        ax.set_title(f"Ground Truth\n{gt_map.sum()}px target", fontsize=10)
    else:
        ax.imshow(gray_img, cmap="gray")
        ax.set_title("Ground Truth\n(not available)", fontsize=10)
    ax.axis("off")

    for i, name in enumerate(method_names):
        ax = axes[i + 1]
        sm = results[name]["score_map"]
        im = ax.imshow(sm, cmap="jet", vmin=0)
        ax.set_title(f"{name}\nAUC={results[name].get('auc', 0):.4f}"
                     f"\nF1={results[name].get('F1', 0):.3f}"
                     f"\ndet={results[name]['det_pixels']}px")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    for i in range(n_methods + 1, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Score Map Comparison — All Methods", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_maps.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ score_maps.png")

    # ── Fig 2: Detection Overlays (with GT outline) ──
    n_cols2 = max(2, (n_methods + 1) // 2)
    fig, axes = plt.subplots(2, n_cols2,
                             figsize=(6 * n_cols2, 12))
    axes = axes.flatten()

    # First: GT overlay reference
    ax = axes[0]
    ax.imshow(gray_img, cmap="gray")
    if gt_map is not None:
        gt_overlay = np.zeros((H, W, 4))
        gt_overlay[..., 1] = 1.0
        gt_overlay[..., 3] = gt_map.astype(float) * 0.5
        ax.imshow(gt_overlay)
    ax.set_title("Ground Truth (target area)", fontsize=10)
    ax.axis("off")

    for i, name in enumerate(method_names):
        ax = axes[i + 1] if i + 1 < len(axes) else axes[-1]
        ax.imshow(gray_img, cmap="gray")
        # Detection in red
        det_overlay = np.zeros((H, W, 4))
        det_overlay[..., 0] = 1.0  # red
        det_overlay[..., 3] = results[name]["binary"].astype(float) * 0.6
        ax.imshow(det_overlay)
        # GT outline in green
        if gt_map is not None:
            gt_contour = np.zeros((H, W, 4))
            gt_contour[..., 1] = 1.0
            gt_contour[..., 3] = gt_map.astype(float) * 0.3
            ax.imshow(gt_contour)

        pm = results[name].get("pixel_metrics", {})
        ax.set_title(f"{name}\nF1={pm.get('F1', 0):.3f} IoU={pm.get('IoU', 0):.3f}\n"
                     f"P={pm.get('precision', 0):.3f} R={pm.get('recall', 0):.3f}",
                     fontsize=10)
        ax.axis("off")

    for i in range(n_methods + 1 if n_methods + 1 < len(axes) else 1, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Detection vs Ground Truth (red=detection, green=GT)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detection_overlays.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ detection_overlays.png")

    # ── Fig 3: ROC Curves ──
    has_roc = any("roc_fpr" in results[m] for m in method_names)
    if has_roc:
        fig, ax = plt.subplots(figsize=(8, 7))

        for i, name in enumerate(method_names):
            if "roc_fpr" in results[name]:
                ax.plot(results[name]["roc_fpr"], results[name]["roc_tpr"],
                        lw=2, label=f"{name} (AUC={results[name]['auc']:.4f})",
                        color=colors[i % len(colors)])

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — All Methods", fontsize=14)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ roc_curves.png")

    # ── Fig 4: Score Distributions ──
    fig, axes = plt.subplots(n_methods, 1, figsize=(10, 3 * n_methods))
    if n_methods == 1:
        axes = [axes]

    for i, name in enumerate(method_names):
        ax = axes[i]
        scores = results[name]["scores_raw"]
        ax.hist(scores, bins=80, alpha=0.7, color=colors[i % len(colors)])
        ax.axvline(results[name]["threshold"], color="r", ls="--",
                   label=f"threshold={results[name]['threshold']:.3f}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} — Score Distribution")
        ax.legend(fontsize=9)
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ score_distributions.png")

    # ── Fig 5: Spectral Analysis ──
    if "ae_scores" in results or any("ae" in m.lower() for m in method_names):
        ae_model = None
        for name in method_names:
            if "ae" in name.lower() and "model" in results[name]:
                ae_model = results[name]["model"]
                break
        if ae_model is not None and "spectra" in results:
            _plot_spectra_analysis(results, output_dir)


def _plot_spectra_analysis(results: dict, output_dir: str):
    """Plot spectral analysis: target/background mean spectra + AE reconstruction error."""
    spectra = results["spectra"]
    labels = results.get("labels")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Mean spectra
    ax = axes[0, 0]
    bg = spectra[labels == 0] if labels is not None else spectra
    ax.plot(bg.mean(axis=0), "b-", alpha=0.7, label="Background mean", lw=2)
    ax.fill_between(range(len(bg.mean(axis=0))),
                    bg.mean(axis=0) - bg.std(axis=0),
                    bg.mean(axis=0) + bg.std(axis=0),
                    alpha=0.2, color="blue")
    if labels is not None:
        for c in [1, 2, 3]:
            mask = labels == c
            if mask.any():
                ax.plot(spectra[mask].mean(axis=0), "--", lw=2,
                        label=f"Target{c} (n={mask.sum()})")
    ax.set_xlabel("Band Index")
    ax.set_ylabel("Reflectance (normalized)")
    ax.set_title("Target vs Background — Mean Spectra")
    ax.legend()

    # Top-right: AE reconstruction error distribution
    ax = axes[0, 1]
    for name in results:
        if "ae" in name.lower() and "model" in results[name]:
            model = results[name]["model"]
            data_t = torch.from_numpy(spectra).float().to(DEVICE)
            errors = model.anomaly_score(data_t)
            if labels is not None:
                bg_err = errors[labels == 0]
                tgt_err = errors[labels > 0]
                ax.hist(bg_err, bins=50, alpha=0.5, color="blue", label=f"BG std={bg_err.std():.4f}")
                ax.hist(tgt_err, bins=50, alpha=0.5, color="red", label=f"Target std={tgt_err.std():.4f}")
            ax.set_xlabel("Recon MSE")
            ax.set_ylabel("Count")
            ax.set_title("AE Reconstruction Error")
            ax.legend()
            ax.set_yscale("log")
            break

    # Bottom-left: AE reconstruction examples
    ax = axes[1, 0]
    for name in results:
        if "ae" in name.lower() and "model" in results[name]:
            model = results[name]["model"]
            model.eval()
            if labels is not None:
                bg_idx = np.where(labels == 0)[0][:3]
                tgt_idx = np.where(labels > 0)[0][:3]
                for idx in bg_idx:
                    x = torch.from_numpy(spectra[idx]).float().unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        recon = model(x).cpu().numpy().squeeze()
                    ax.plot(spectra[idx], "b-", alpha=0.5)
                    ax.plot(recon, "b--", alpha=0.5)
                for idx in tgt_idx:
                    x = torch.from_numpy(spectra[idx]).float().unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        recon = model(x).cpu().numpy().squeeze()
                    ax.plot(spectra[idx], "r-", alpha=0.5)
                    ax.plot(recon, "r--", alpha=0.5)
            ax.set_title("AE Reconstruction (solid=original, dashed=recon)")
            ax.set_xlabel("Band Index")
            break

    # Bottom-right: Legend
    ax = axes[1, 1]
    ax.axis("off")
    info = (
        "Analysis Notes\n\n"
        "BG: non-target annotated pixels\n"
        "Target1: grass camo net (81 spectra)\n"
        "Target2: military green (124 spectra)\n"
        "Target3: desert camo (20 spectra)\n\n"
        "AE: train on BG only, detect anomaly\n"
        "  high recon error -> target\n\n"
        "1D-CNN: supervised classification\n"
        "  P(target) > threshold -> detected"
    )
    ax.text(0.1, 0.9, info, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spectra_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ spectra_analysis.png")


# ═══════════════════════════════════════════════════════════
# 8. 主函数
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="深度学习高光谱目标检测 — 与 ACE/SACE 对比")
    parser.add_argument("--data-dir", default=None, help="数据目录")
    parser.add_argument("--hi-dir", default=None, help="HI 目录")
    parser.add_argument("--target-dir", default=None, help="滤波后目标模板目录")
    parser.add_argument("--dx", type=int, default=195, help="配准偏移 x")
    parser.add_argument("--dy", type=int, default=-30, help="配准偏移 y")
    parser.add_argument("--output", default="output/deep_learning", help="输出目录")
    parser.add_argument("--no-train", action="store_true",
                        help="跳过训练，使用已有模型（需要模型文件存在）")
    parser.add_argument("--epochs-ae", type=int, default=200,
                        help="AE 训练轮数 (默认: 200)")
    parser.add_argument("--epochs-cnn", type=int, default=300,
                        help="1D-CNN 训练轮数 (默认: 300)")
    parser.add_argument("--skip-ace", action="store_true", help="跳过 ACE 基线")
    parser.add_argument("--skip-sace", action="store_true", help="跳过 SACE 基线")
    parser.add_argument("--filter", default="none",
                        choices=["none", "sky", "reflectance", "scores", "full"],
                        help="Notch 滤波级别 (默认: none)")
    parser.add_argument("--post", default="none",
                        choices=["none", "median5", "median7", "median9",
                                 "gaussian", "open", "close", "med_open", "full"],
                        help="Score map 后处理 (默认: none)")
    parser.add_argument("--post-kernel", type=int, default=5,
                        help="后处理滤波核大小 (默认: 5)")
    parser.add_argument("--enhanced-ae", action="store_true",
                        help="使用增强版 AE (收紧瓶颈+L1正则)")
    parser.add_argument("--info-nce", action="store_true",
                        help="使用 InfoNCE + Mahalanobis 检测器")
    parser.add_argument("--cnn-threshold", type=float, default=None,
                        help="1D-CNN 固定阈值 (默认自动扫描最佳)")
    parser.add_argument("--sace-medopen", action="store_true",
                        help="添加 SACE+med_open 对比方法")
    parser.add_argument("--spatial-cnn", action="store_true",
                        help="使用光谱-空间 Patch CNN (2D卷积)")
    parser.add_argument("--no-focal", action="store_true",
                        help="SpatialCNN 不使用 Focal Loss, 用加权 BCE")
    parser.add_argument("--patch-size", type=int, default=5,
                        help="SpatialCNN 的 patch 大小 (默认: 5)")
    parser.add_argument("--unsupervised", action="store_true",
                        help="使用无监督增强算法 (DerivACE, RX, Ensemble)")
    parser.add_argument("--skip-cnn", action="store_true",
                        help="跳过 1D-CNN 训练 (省时间)")
    parser.add_argument("--skip-ae", action="store_true",
                        help="跳过 SpectralAE 训练")
    processor = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = processor.data_dir or os.path.join(base, "..", "data", "1")
    hi_dir = processor.hi_dir or os.path.join(base, "..", "HI")
    output_dir = processor.output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("🧠 深度学习高光谱目标检测 — 与 ACE/SACE 对比")
    print("=" * 65)
    t_start = time.time()

    # ── 初始化滤波器 ──
    notch_filter = None
    if processor.filter != "none":
        notch_filter = NotchFilter()
        print(f"\n🌀 Notch 滤波: {processor.filter}")
        print(f"  {notch_filter}")

    # ── 1. 加载数据 ──
    print("\n📂 加载图像...")
    images = load_images(data_dir)
    gray_img = images["gray"]

    print("\n📐 计算反射率 (含可选 Notch 滤波)...")
    reflect = compute_reflectance_cube(images, notch_filter, processor.filter)

    print("\n🔬 提取光谱向量...")
    data_vector, first_coords = extract_spectral_vectors(reflect, hi_dir)

    print("\n🎯 加载目标模板...")
    targets = load_target_templates(hi_dir, processor.target_dir)

    print("\n🏷️  获取真值标签 (从 mask.npy)...")
    labels = get_labels_from_mask(first_coords, hi_dir)

    print("\n🧹 波段过滤 + 归一化...")
    data_n, targets_n, labels_n, good_bands = filter_and_normalize(
        data_vector, targets, labels
    )
    n_bands = data_n.shape[1]
    print(f"  最终数据: {data_n.shape}, 目标: { {i: t.shape for i, t in targets_n.items()} }")

    # ── 2a. 分层分割 (训练/验证/测试) ──
    print("\n📊 数据集分割 (分层保持目标比例)...")
    train_idx, val_idx, test_idx = stratified_split(labels_n)
    print(f"    训练集: {len(train_idx)} / 验证集: {len(val_idx)} / 测试集: {len(test_idx)}")

    # 为 ACE/SACE 评估也只用测试集，但训练可以用全量（无监督）
    # 注意: ACE/SACE 的 fit() 使用全量数据估计背景统计（无监督），
    # 但打分和评估限制在测试集
    # 而对于监督方法，训练只在 train_idx 上

    # ── 2. 定义方法列表 ──
    methods_to_run = []
    if not processor.skip_ace:
        methods_to_run.append("ACE")
    if not processor.skip_sace:
        methods_to_run.append("SACE")
    if processor.sace_medopen:
        methods_to_run.append("SACE+med_open")
    if not processor.skip_ae:
        methods_to_run.append("SpectralAE")
    if processor.enhanced_ae:
        methods_to_run.append("SpectralAEv2")
    if processor.info_nce:
        methods_to_run.append("InfoNCE")
    if processor.spatial_cnn:
        methods_to_run.append("SpectralSpatialCNN")
    if processor.unsupervised:
        methods_to_run.extend(["DerivACE", "DerivSACE", "RX", "SSRX", "Ensemble"])
    if not processor.skip_cnn:
        methods_to_run.append("Spectral1DCNN")

    results = {}

    # ── 3. 运行传统方法 ──
    if "ACE" in methods_to_run:
        print(f"\n{'─' * 50}")
        print("🎯 ACE (传统基线)")
        ace_scores = run_ace(data_n, targets_n)
        results["ACE"] = {"scores_raw": ace_scores}

    if "SACE" in methods_to_run:
        print(f"\n{'─' * 50}")
        print("🎯 SACE (传统最佳)")
        sace_scores = run_sace(data_n, targets_n)
        results["SACE"] = {"scores_raw": sace_scores}

        # SACE+med_open: 同样分数但 score map 做 med_open 后处理
        if "SACE+med_open" in methods_to_run:
            results["SACE+med_open"] = {"scores_raw": sace_scores.copy()}

    # ── 4. 无监督增强算法 (无需训练) ──
    if any(m in methods_to_run for m in ["DerivACE", "DerivSACE", "RX", "SSRX", "Ensemble"]):
        print(f"\n{'─' * 50}")
        print("🔬 无监督增强算法")

    if "DerivACE" in methods_to_run:
        print("  📈 DerivACE (一阶导数+ACE)...")
        results["DerivACE"] = {"scores_raw": run_derivative_ace(data_n, targets_n, deriv=1)}
        print("  📈 Deriv2ACE (二阶导数+ACE)...")
        results["Deriv2ACE"] = {"scores_raw": run_derivative_ace(data_n, targets_n, deriv=2)}

    if "DerivSACE" in methods_to_run:
        print("  📈 DerivSACE (一阶导数+SACE)...")
        results["DerivSACE"] = {"scores_raw": run_derivative_sace(data_n, targets_n, deriv=1)}

    if "RX" in methods_to_run:
        print("  📈 RX (马氏距离异常检测)...")
        results["RX"] = {"scores_raw": run_rx(data_n)}

    if "SSRX" in methods_to_run:
        print("  📈 SSRX (PCA+RX)...")
        results["SSRX"] = {"scores_raw": run_ssrx(data_n)}

    if "Ensemble" in methods_to_run:
        print("  📈 Ensemble (多检测器融合)...")
        results["Ensemble"] = {"scores_raw": run_ensemble_unsupervised(data_n, targets_n)}

    # ── 4b. 训练/运行深度学习方法 ──
    bg_mask = labels_n == 0 if labels_n is not None else slice(None)
    tgt_mask = labels_n > 0 if labels_n is not None else slice(None)

    bg_data = data_n[bg_mask] if isinstance(bg_mask, np.ndarray) else data_n
    print(f"\n  DL 训练数据: 背景={len(bg_data)}, "
          f"目标={data_n[tgt_mask].shape[0] if isinstance(tgt_mask, np.ndarray) else 'N/A'}")

    # SpectralAE
    if "SpectralAE" in methods_to_run:
        print(f"\n{'─' * 50}")
        ae_path = os.path.join(output_dir, "ae_model.pth")
        if processor.no_train and os.path.exists(ae_path):
            print("🌀 加载已有 AE 模型...")
            ae_model = SpectralAE(n_bands).to(DEVICE)
            ae_model.load_state_dict(torch.load(ae_path, map_location=DEVICE))
        else:
            ae_model = train_autoencoder(bg_data, n_bands, epochs=processor.epochs_ae)
            torch.save(ae_model.state_dict(), ae_path)
            print(f"  💾 模型保存: {ae_path}")
        ae_scores = run_ae_detection(ae_model, data_n)
        results["SpectralAE"] = {"scores_raw": ae_scores, "model": ae_model}

    # SpectralAEv2 (增强版)
    if "SpectralAEv2" in methods_to_run:
        print(f"\n{'─' * 50}")
        aev2_path = os.path.join(output_dir, "aev2_model.pth")
        if processor.no_train and os.path.exists(aev2_path):
            print("🌀 加载已有 SpectralAEv2 模型...")
            aev2_model = SpectralAEv2(n_bands).to(DEVICE)
            aev2_model.load_state_dict(torch.load(aev2_path, map_location=DEVICE))
        else:
            aev2_model = train_autoencoder_v2(bg_data, n_bands, epochs=processor.epochs_ae)
            torch.save(aev2_model.state_dict(), aev2_path)
            print(f"  💾 模型保存: {aev2_path}")
        aev2_scores = run_ae_detection(aev2_model, data_n)
        results["SpectralAEv2"] = {"scores_raw": aev2_scores, "model": aev2_model}

    # InfoNCE + Mahalanobis
    if "InfoNCE" in methods_to_run:
        print(f"\n{'─' * 50}")
        nce_path = os.path.join(output_dir, "info_nce_model.pth")
        if processor.no_train and os.path.exists(nce_path):
            print("🌀 加载已有 InfoNCE 模型...")
            nce_model = SpectralInfoNCE(n_bands).to(DEVICE)
            nce_model.load_state_dict(torch.load(nce_path, map_location=DEVICE))
        else:
            nce_model = train_info_nce(bg_data, n_bands, epochs=processor.epochs_ae)
            torch.save(nce_model.state_dict(), nce_path)
            print(f"  💾 模型保存: {nce_path}")
        nce_scores = run_info_nce_detection(nce_model, bg_data, data_n)
        results["InfoNCE"] = {"scores_raw": nce_scores, "model": nce_model}

    # Spectral1DCNN
    if "Spectral1DCNN" in methods_to_run:
        print(f"\n{'─' * 50}")
        cnn_path = os.path.join(output_dir, "cnn_model.pth")
        cnn_trained = False
        cnn_model = None
        if processor.no_train and os.path.exists(cnn_path):
            print("🌀 加载已有 1D-CNN 模型...")
            cnn_model = Spectral1DCNN(n_bands).to(DEVICE)
            cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
            cnn_trained = True
        else:
            if labels_n is not None and (labels_n > 0).any():
                cnn_model = train_1dcnn(data_n, labels_n, n_bands,
                                        train_idx=train_idx, val_idx=val_idx,
                                        epochs=processor.epochs_cnn)
                torch.save(cnn_model.state_dict(), cnn_path)
                print(f"  💾 模型保存: {cnn_path}")
                cnn_trained = True
            else:
                print("⚠️  无目标标签，跳过 1D-CNN 训练")
                methods_to_run.remove("Spectral1DCNN")

        if cnn_trained and cnn_model is not None:
            cnn_scores = run_cnn_detection(cnn_model, data_n)
            results["Spectral1DCNN"] = {"scores_raw": cnn_scores, "model": cnn_model}

    # SpectralSpatialCNN
    if "SpectralSpatialCNN" in methods_to_run:
        print(f"\n{'─' * 50}")
        print("🧠 Spectral-Spatial Patch CNN")
        scnn_path = os.path.join(output_dir, "spatial_cnn_model.pth")
        scnn_trained = False
        scnn_model = None

        # Extract spatial patches from full reflectance image
        print("📦 Extracting spatial patches from full reflectance...")
        patches = extract_spatial_patches(
            reflect, first_coords, good_bands, hi_dir,
            patch_size=processor.patch_size
        )

        if processor.no_train and os.path.exists(scnn_path):
            print("🌀 Loading existing model...")
            scnn_model = SpectralSpatialCNN(n_bands, processor.patch_size).to(DEVICE)
            scnn_model.load_state_dict(torch.load(scnn_path, map_location=DEVICE))
            scnn_trained = True
        else:
            if labels_n is not None and (labels_n > 0).any():
                scnn_model = train_spatial_cnn(
                    patches, labels_n, n_bands,
                    train_idx=train_idx, val_idx=val_idx,
                    patch_size=processor.patch_size,
                    epochs=processor.epochs_cnn,
                    use_focal=not processor.no_focal,
                )
                torch.save(scnn_model.state_dict(), scnn_path)
                print(f"  💾 Model saved: {scnn_path}")
                scnn_trained = True
            else:
                print("⚠️  No target labels, skipping SpectralSpatialCNN")
                methods_to_run.remove("SpectralSpatialCNN")

        if scnn_trained and scnn_model is not None:
            scnn_scores = run_spatial_cnn_detection(scnn_model, patches)
            results["SpectralSpatialCNN"] = {"scores_raw": scnn_scores, "model": scnn_model}

    # ── 5. 评估 ──
    print(f"\n{'=' * 50}")
    print("📊 评估所有方法")
    print('=' * 50)

    print("\n📌 加载全图地真 (mask.npy → grayscale space)...")
    gt_map = load_ground_truth_mask(hi_dir, first_coords, gray_img.shape,
                                    reg_offset=(processor.dy, processor.dx))
    results["gt_map"] = gt_map

    # 基础阈值
    THRESHOLDS = {"ACE": 0.18, "SACE": 0.18, "SACE+med_open": 0.18,
                  "SpectralAE": None, "SpectralAEv2": None,
                  "InfoNCE": None, "Spectral1DCNN": 0.5,
                  "SpectralSpatialCNN": 0.5,
                  "DerivACE": 0.18, "Deriv2ACE": 0.18,
                  "DerivSACE": 0.18,
                  "RX": None, "SSRX": None, "Ensemble": None}

    # CNN 阈值扫描范围
    CNN_THRESHOLDS_TO_TRY = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]

    all_metrics = {}
    for name in methods_to_run:
        if name not in results:
            continue
        scores = results[name]["scores_raw"]

        # 自动确定 AE / InfoNCE 阈值
        if name in ("SpectralAE", "SpectralAEv2"):
            if labels_n is not None:
                bg_scores = scores[labels_n == 0]
                thr = np.percentile(bg_scores, 98)
                print(f"  {name}: auto-threshold P98(BG) = {thr:.6f}")
            else:
                thr = np.percentile(scores, 95)
            THRESHOLDS[name] = thr

        if name == "InfoNCE":
            if labels_n is not None:
                bg_scores = scores[labels_n == 0]
                thr = np.percentile(bg_scores, 95)
                print(f"  {name}: auto-threshold P95(BG) = {thr:.6f}")
            else:
                thr = np.percentile(scores, 90)
            THRESHOLDS[name] = thr

        if name in ("RX", "SSRX"):
            if labels_n is not None:
                bg_scores = scores[labels_n == 0]
                thr = np.percentile(bg_scores, 95)
            else:
                thr = np.percentile(scores, 95)
            THRESHOLDS[name] = thr
            print(f"  {name}: auto-threshold P95(BG) = {thr:.4f}")

        if name == "Ensemble":
            if labels_n is not None:
                bg_scores = scores[labels_n == 0]
                thr = np.percentile(bg_scores, 90)
            else:
                thr = np.percentile(scores, 90)
            THRESHOLDS[name] = thr
            print(f"  {name}: auto-threshold P90(BG) = {thr:.4f}")

        # ── 分数指标 (仅在测试集上计算 AUC!) ──
        metrics = compute_metrics(scores[test_idx], labels_n[test_idx])
        test_auc = metrics.get("auc", 0)

        # ── 1D-CNN / SpatialCNN 阈值扫描 (在验证集上) ──
        if name in ("Spectral1DCNN", "SpectralSpatialCNN"):
            val_scores = scores[val_idx]
            val_labels = labels_n[val_idx] if labels_n is not None else None
            best_f1 = -1
            best_thr = 0.5

            if getattr(processor, 'cnn_threshold', None) is not None:
                best_thr = processor.cnn_threshold
            else:
                for thr in CNN_THRESHOLDS_TO_TRY:
                    from sklearn.metrics import f1_score
                    pred_binary = (val_scores > thr).astype(np.int32)
                    f1_try = f1_score(val_labels > 0, pred_binary) if val_labels is not None else 0
                    if f1_try > best_f1:
                        best_f1 = f1_try
                        best_thr = thr
                print(f"  {name}: val-set threshold scan → best F1={best_f1:.4f} @ thr={best_thr}")

            threshold = best_thr
            THRESHOLDS[name] = threshold

        # ── 传统方法阈值 ──
        else:
            threshold = THRESHOLDS.get(name, 0.5)

        # ── Score Map (用全量分数生成，用于可视化) ──
        score_map = generate_score_map(scores, first_coords, gray_img.shape,
                                       reg_offset=(processor.dy, processor.dx))

        # 后处理
        apply_post = (
            (name == "SACE+med_open") or
            (name in ("ACE", "SACE") and processor.post != "none")
        )
        if apply_post:
            temp_filt = NotchFilter()
            print(f"  🌀 {name}: 应用后处理 {processor.post if name in ('ACE','SACE') else 'med_open'}...")
            post_method = "med_open" if name == "SACE+med_open" else processor.post
            score_map = temp_filt.filter_score_map(score_map, method=post_method,
                                                    kernel_size=processor.post_kernel)

        # ── Score Map → 二值检测 (全量，可视化用) ──
        binary = filter_connected_components(score_map, threshold)
        sm_metrics = evaluate_score_map(score_map, threshold)
        pixel_metrics = compute_pixel_metrics(binary, gt_map) if gt_map is not None else {}

        results[name].update({
            "threshold": threshold,
            "score_map": score_map,
            "binary": binary,
            "det_pixels": sm_metrics["det_pixels"],
            "bg_std": sm_metrics["bg_std"],
            "sm_score": sm_metrics["score"],
            "auc": test_auc,
            "auc_note": "test_set_only",
            "pixel_metrics": pixel_metrics,
            "F1": pixel_metrics.get("F1", 0),
            "IoU": pixel_metrics.get("IoU", 0),
        })
        # ROC 曲线用测试集
        if "roc_fpr" in metrics:
            results[name]["roc_fpr"] = metrics["roc_fpr"]
            results[name]["roc_tpr"] = metrics["roc_tpr"]

        results["spectra"] = data_n
        results["labels"] = labels_n

        all_metrics[name] = {
            "AUC": f"{test_auc:.4f}",
            "AUC_note": "test_set",
            "det_px": sm_metrics["det_pixels"],
            "bg_std": f"{sm_metrics['bg_std']:.4f}",
            "max_score": f"{sm_metrics['max_score']:.4f}",
            "composite": f"{sm_metrics['score']:.2f}",
            "F1": f"{pixel_metrics.get('F1', 0):.4f}",
            "IoU": f"{pixel_metrics.get('IoU', 0):.4f}",
            "precision": f"{pixel_metrics.get('precision', 0):.4f}",
            "recall": f"{pixel_metrics.get('recall', 0):.4f}",
        }

        if "separability" in metrics:
            all_metrics[name]["sep"] = f"{metrics['separability']:.2f}"

    # ── 5. Print comparison table ──
    print(f"\n{'─' * 80}")
    print(f"{'Method':<16} {'AUC*':<8} {'F1**':<8} {'IoU':<8} {'P':<8} {'R':<8} {'det_px':<10} {'composite':<10}")
    print(f"{'─' * 80}")
    for name in methods_to_run:
        if name not in all_metrics:
            continue
        m = all_metrics[name]
        print(f"{name:<16} {m['AUC']:<8} {m['F1']:<8} {m['IoU']:<8} "
              f"{m['precision']:<8} {m['recall']:<8} {m['det_px']:<10} {m['composite']:<10}")

    # Save comparison table
    table_path = os.path.join(output_dir, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write("DL vs Traditional — Hyperspectral Target Detection (with GT evaluation)\n")
        f.write(f"{'=' * 90}\n\n")
        f.write(f"Data: {data_n.shape[0]} points ({len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test), {n_bands} bands\n")
        f.write(f"GT: mask.npy targets (4/5/6) mapped to grayscale space\n")
        f.write(f"Target templates: ")
        for i, t in targets_n.items():
            f.write(f"target{i}({t.shape[0]} samples) ")
        f.write(f"\n\n")
        f.write(f"{'Method':<16} {'AUC':<8} {'F1':<8} {'IoU':<8} {'P':<8} {'R':<8} "
                f"{'det_px':<10} {'bg_std':<10} {'composite':<10}\n")
        f.write(f"{'─' * 90}\n")
        for name in methods_to_run:
            if name not in all_metrics:
                continue
            m = all_metrics[name]
            f.write(f"{name:<16} {m['AUC']:<8} {m['F1']:<8} {m['IoU']:<8} "
                    f"{m['precision']:<8} {m['recall']:<8} {m['det_px']:<10} "
                    f"{m['bg_std']:<10} {m['composite']:<10}\n")
        f.write(f"\n{'─' * 90}\n")
        f.write("composite = det_px/1000 + bg_std*5000 (lower is better)\n")
        f.write("AUC* = area under ROC curve at spectral point level (test set only)\n")
        f.write("F1**/IoU/P/R = pixel-level metrics against GT (full image, for reference)\n")
        f.write(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test\n")
    print(f"  ✅ comparison_table: {table_path}")

    # ── 6. 可视化 ──
    print(f"\n🎨 生成可视化...")
    visualize_comparison(results, gray_img, gt_map, output_dir, (processor.dy, processor.dx))

    # ── 完成 ──
    t_elapsed = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"✅ Done! Elapsed: {t_elapsed:.1f}s")
    print(f"📂 Output: {os.path.abspath(output_dir)}")
    print(f"{'=' * 65}")

    # Best method
    valid = {k: v for k, v in all_metrics.items() if "AUC" in v}
    if valid:
        best_auc = max(valid, key=lambda k: float(valid[k]["AUC"]))
        print(f"\n🏆 Best test-set AUC: {best_auc} ({valid[best_auc]['AUC']})")

        best_score = min(valid, key=lambda k: float(valid[k]["composite"]))
        print(f"🏆 Best composite: {best_score} ({valid[best_score]['composite']})")

        best_f1 = max(valid, key=lambda k: float(valid[k]["F1"]))
        print(f"🏆 Best pixel-level F1: {best_f1} ({valid[best_f1]['F1']})\n"
              f"    * AUC = test set only | F1/P/R = full image pixel-level")


if __name__ == "__main__":
    main()
