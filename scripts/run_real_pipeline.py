#!/usr/bin/env python3
"""
真实高光谱数据端到端 Pipeline。

从原始 TIF 文件出发:
  1. 暗电流校正 + 反射率计算 (orig-dark)/(sky-dark)
  2. 通过 coords_dict.json 提取光谱向量
  3. 自动检测并移除饱和波段 (840-905nm)
  4. 均值归一化
  5. 运行 5 种检测算法 (CEM/ACE/SAM/SACE/MTICEM)
  6. 灰度-光谱相机配准偏移 (dx=79)
  7. 检测点 → 6×53 矩形膨胀 → 连通区域过滤 (area≥1117)
  8. 伪彩色 + 灰度图生成

用法:
  python scripts/run_real_pipeline.py
  python scripts/run_real_pipeline.py --dx 195 --dy -30
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile

matplotlib.use("Agg")

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import (
    subtract_dark_current,
    compute_reflectance,
    detect_saturated_bands,
    normalize_reflectance,
)
from detection.cem import CEMDetector
from detection.ace import ACEDetector
from detection.sam import SpectralAngleMapper as SAMDetector
from detection.mticem import MTICEMDetector
from detection.sace import SACEDetector
from noise_filter import NotchFilter, analyze_fft


# ───── 常量 ─────
WAVELENGTHS = np.arange(445, 906, 5, dtype=int)  # 93 个波段
RECT_H, RECT_W = 6, 53  # 检测点膨胀矩形
MIN_AREA = 1117  # 连通区域面积阈值


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
        print(f"  {name:>15}: {img.shape}, dtype={img.dtype}, "
              f"均值={img.mean():.1f}, 范围=[{img.min()},{img.max()}]")
        images[name] = img
    return images


def compute_reflectance_cube(images: dict, notch_filter: NotchFilter = None,
                             filter_level: str = 'none') -> np.ndarray:
    """
    计算全图反射率。
    公式: (spec - dark) / (sky - dark)

    参数:
        images: 图像字典
        notch_filter: NotchFilter 实例 (为 None 时不滤波)
        filter_level: 'none', 'sky', 'reflectance', 'full'
    """
    img_spec = subtract_dark_current(images["spec_base"], images["dark"])
    img_sky = subtract_dark_current(images["illuminance"], images["dark"])

    # Level 1: Sky 预滤波 (源头去除干涉条纹)
    if notch_filter and filter_level in ('sky', 'full'):
        print("  🌀 ① Sky 预滤波 (2D 陷波)...")
        analyze_fft(img_sky, "Sky 滤波前")
        img_sky = notch_filter.filter_image_2d(img_sky)
        analyze_fft(img_sky, "Sky 滤波后")

    reflect = compute_reflectance(img_spec, img_sky)
    print(f"  反射率: 形状={reflect.shape}, 范围=[{reflect.min():.4f},{reflect.max():.1f}]")

    # Level 2: Reflectance 逐波段滤波 (消除残留)
    if notch_filter and filter_level in ('reflectance', 'full'):
        print(f"  🌀 ② Reflectance 逐波段滤波 (79 波段 x 列 FFT)...")
        # 分析滤波前 FFT
        analyze_fft(reflect, "Reflectance 滤波前")
        # 反射率是 2D (H,W)，需要 reshape 为 (H,W,1) 以便使用 filter_reflectance_cube
        ref_3d = reflect[:, :, np.newaxis]
        ref_clean = notch_filter.filter_reflectance_cube(ref_3d)
        reflect = ref_clean[:, :, 0]
        analyze_fft(reflect, "Reflectance 滤波后")

    return reflect


def extract_spectral_vectors(reflect: np.ndarray, hi_dir: str) -> tuple:
    """
    通过 coords_dict.json 坐标映射提取所有标定点的光谱向量。

    返回:
        data_vector: (N, 93) 光谱向量
        first_coords: (N, 2) 第一波段 (y, x) 坐标
    """
    coords_path = os.path.join(hi_dir, "coords_dict.json")
    print(f"  加载坐标映射: {coords_path}")

    with open(coords_path, "r") as f:
        coords_dict = json.load(f)

    n_bands = 93
    # 只保留有完整 93 波段坐标的条目
    valid_items = [(idx_str, spec) for idx_str, spec in coords_dict.items()
                   if len(spec) == n_bands]
    n_points = len(valid_items)
    data_vector = np.zeros((n_points, n_bands), dtype=np.float64)
    first_coords = np.zeros((n_points, 2), dtype=int)  # [y, x]

    for i, (idx_str, spec) in enumerate(valid_items):
        row = np.array([reflect[s[1], s[2]] for s in spec], dtype=np.float64)
        data_vector[i, :] = row
        first_coords[i] = [spec[0][1], spec[0][2]]  # [y, x]

    print(f"  光谱向量: ({n_points}, {n_bands}), "
          f"范围=[{data_vector.min():.6f}, {data_vector.max():.1f}]")
    return data_vector, first_coords


def load_target_templates(hi_dir: str, target_dir: str = None) -> dict:
    """
    加载预先提取的目标光谱 (target1-3.npy)。

    参数:
        hi_dir: HI 目录 (默认位置)
        target_dir: 自定义目标文件目录 (优先级高于 hi_dir)
    返回 {1: array, 2: array, 3: array}。
    """
    targets = {}
    for i in [1, 2, 3]:
        # 优先从 target_dir 加载
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
            print(f"  target{i}: {path}, shape={t.shape}, "
                  f"均值={t.mean():.4f}, 范围=[{t.min():.4f},{t.max():.4f}]")
    return targets


def filter_bands(data: np.ndarray, targets: dict) -> tuple:
    """
    自动检测并移除饱和波段。
    返回: (data_filtered, target_spectra, good_bands)
    """
    good, bad = detect_saturated_bands(data, threshold_ratio=10.0)
    if len(bad) > 0:
        print(f"  → 保留波段: [{good[0]}-{good[-1]}], "
              f"波长 {WAVELENGTHS[good[0]]}-{WAVELENGTHS[good[-1]]}nm")

    data_f = data[:, good]
    target_spec = {}
    for i, t in targets.items():
        t_f = t[:, good] if t.shape[1] == 93 else t
        target_spec[i] = t_f

    return data_f, target_spec, good


def normalize_data(data: np.ndarray, targets: dict) -> tuple:
    """均值归一化。"""
    data_n = normalize_reflectance(data, method="mean")
    targets_n = {}
    for i, t in targets.items():
        targets_n[i] = normalize_reflectance(t, method="mean")
    return data_n, targets_n


def get_threshold(method: str, scene: int = 1) -> float:
    """获取检测阈值（基于真实数据 P99 分析）。"""
    thresholds = {
        "CEM": 1.5,    # P99 ≈ 1.50
        "ACE": 0.18,   # P99 ≈ 0.18
        "SAM": 0.975,  # P99 ≈ 0.974 (SAM分数越高越像目标)
        "MTICEM": 1.5,
        "SACE": 0.18,
    }
    return thresholds.get(method, 0.7)


def run_detection(data: np.ndarray, targets: dict, method: str, scene: int) -> tuple:
    """
    运行指定检测器。

    对单目标检测器 (CEM/ACE/SAM/SACE)，依次对 3 类目标 (target1-3)
    分别计算检测分数，取各目标最高分（类似 MTICEM 的 predict_max）。

    返回:
        scores: (N,) 每个像素的分数（多目标取 max）
        threshold: 使用的检测阈值
    """
    print(f"\n  🎯 运行检测: {method}")

    # 3 类目标均值光谱
    target_list = [targets[i].mean(axis=0) for i in [1, 2, 3]]
    target_labels = ["草地伪装网", "军绿迷彩", "沙漠迷彩"]

    if method == "CEM":
        scores_multi = np.zeros((data.shape[0], 3))
        for ti, (tgt, label) in enumerate(zip(target_list, target_labels)):
            det = CEMDetector(reg=1e-6)
            det.fit(data, tgt)
            scores_multi[:, ti] = det.predict(data)
            print(f"    target{ti+1} ({label}): 范围=[{scores_multi[:, ti].min():.4f}, {scores_multi[:, ti].max():.4f}]")
        scores = scores_multi.max(axis=1)

    elif method == "ACE":
        scores_multi = np.zeros((data.shape[0], 3))
        for ti, (tgt, label) in enumerate(zip(target_list, target_labels)):
            det = ACEDetector(reg=1e-6)
            det.fit(data, tgt)
            scores_multi[:, ti] = det.predict(data)
            print(f"    target{ti+1} ({label}): 范围=[{scores_multi[:, ti].min():.4f}, {scores_multi[:, ti].max():.4f}]")
        scores = scores_multi.max(axis=1)

    elif method == "SAM":
        scores_multi = np.zeros((data.shape[0], 3))
        for ti, (tgt, label) in enumerate(zip(target_list, target_labels)):
            det = SAMDetector(normalize=True)
            det.fit(tgt[np.newaxis, :])
            angles = det.predict(data)
            scores_multi[:, ti] = 1.0 - angles / np.pi
            print(f"    target{ti+1} ({label}): 范围=[{scores_multi[:, ti].min():.4f}, {scores_multi[:, ti].max():.4f}]")
        scores = scores_multi.max(axis=1)

    elif method == "SACE":
        scores_multi = np.zeros((data.shape[0], 3))
        for ti, (tgt, label) in enumerate(zip(target_list, target_labels)):
            det = SACEDetector(reg=1e-6)
            det.fit(data, tgt)
            scores_multi[:, ti] = det.predict(data)
            print(f"    target{ti+1} ({label}): 范围=[{scores_multi[:, ti].min():.4f}, {scores_multi[:, ti].max():.4f}]")
        scores = scores_multi.max(axis=1)

    elif method == "MTICEM":
        D = np.array(target_list)
        det = MTICEMDetector(reg=1e-6)
        det.fit(data, D)
        scores_multi = det.predict(data)  # (P, c)
        scores = scores_multi.max(axis=1)  # 取各目标最高分

    else:
        raise ValueError(f"未知检测方法: {method}")

    thres = get_threshold(method, scene)
    binary = scores > thres
    print(f"    ─────────────────────────────────")
    print(f"    最终分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"    阈值: {thres}")
    print(f"    检测像素: {binary.sum()} / {len(scores)} ({100*binary.sum()/len(scores):.1f}%)")

    return scores, thres


def generate_score_map(
    scores: np.ndarray,
    first_coords: np.ndarray,
    gray_shape: tuple,
    reg_offset: tuple = (195, -30),
) -> np.ndarray:
    """
    将检测点分数映射到灰度图像空间。

    光谱坐标 → 灰度坐标: (y, x) → (y+dy, x+dx)，与 pseudo_color.py 一致。
    每个点膨胀为 RECT_H×RECT_W 矩形，按列排序后涂抹分数。
    """
    H, W = gray_shape
    dy, dx = reg_offset

    if len(first_coords) != len(scores):
        raise ValueError(
            f"坐标数量 ({len(first_coords)}) 与分数 ({len(scores)}) 不匹配"
        )

    # 全尺寸分数图 (H, W) — 直接对应灰度图像空间
    score_map = np.zeros((H, W), dtype=np.float64)

    # 按列排序（确保列大的后画，覆盖小的）
    order = np.argsort(first_coords[:, 1])  # 按 x 排序
    coords_sorted = first_coords[order]
    scores_sorted = scores[order]

    for (y, x), s in zip(coords_sorted, scores_sorted):
        # 光谱 → 灰度配准偏移
        y_gray = y + dy
        x_gray = x + dx
        if y_gray < 0 or y_gray >= H or x_gray < 0 or x_gray >= W:
            continue
        y1 = max(0, min(y_gray, H - 1))
        y2 = min(y_gray + RECT_H, H)
        x1 = max(0, min(x_gray, W - 1))
        x2 = min(x_gray + RECT_W, W)
        score_map[y1:y2, x1:x2] = s

    return score_map


def filter_connected_components(score_map: np.ndarray, threshold: float) -> np.ndarray:
    """
    连通区域过滤：保留面积 ≥ MIN_AREA 的区域。
    """
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

    print(f"    连通区域: {num_features} 个, 保留 ≥{MIN_AREA}px: {kept_count} 个")
    return keep


def visualize_results(
    gray_img: np.ndarray,
    score_map: np.ndarray,
    binary_mask: np.ndarray,
    scores: np.ndarray,
    method: str,
    output_dir: str,
    threshold: float = 1.0,
    reg_offset: tuple = (195, -30),
):
    """生成综合可视化结果（score_map 已与灰度图像配准对齐）。"""
    dy, dx = reg_offset
    # 不再裁剪灰度图 — score_map 已是全尺寸 (H, W) 且已包含配准偏移

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. 灰度图
    ax = axes[0, 0]
    ax.imshow(gray_img, cmap="gray")
    ax.set_title("Visible Grayscale (view2)")
    ax.axis("off")

    # 2. 分数热力图
    ax = axes[0, 1]
    im = ax.imshow(score_map, cmap="jet", vmin=0)
    ax.set_title(f"{method} Score Map")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 3. 二值检测 (透明度叠加)
    ax = axes[0, 2]
    ax.imshow(gray_img, cmap="gray")
    overlay = np.zeros((*gray_img.shape, 4))
    overlay[..., 0] = 1.0  # 红色
    overlay[..., 3] = binary_mask.astype(float) * 0.6
    ax.imshow(overlay)
    ax.set_title(f"{method} Detection")
    ax.axis("off")

    # 4. 分数直方图
    ax = axes[1, 0]
    ax.hist(scores, bins=100, alpha=0.7, label="all pixels")
    ax.axvline(threshold, color="r", linestyle="--", label=f"threshold={threshold}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Score Distribution")

    # 5. 分数排序
    ax = axes[1, 1]
    order = np.argsort(scores)[::-1]
    ax.plot(np.arange(len(scores)), scores[order], lw=1)
    ax.axhline(threshold, color="r", linestyle="--", label=f"threshold={threshold}")
    ax.set_xlabel("Pixel (sorted)")
    ax.set_ylabel("Score")
    ax.set_title("Score Ranking")
    ax.legend()

    # 6. 检测统计
    ax = axes[1, 2]
    ax.axis("off")
    info = (
        f"Method: {method}\n"
        f"Total pixels: {len(scores)}\n"
        f"Detected: {binary_mask.sum()}\n"
        f"Score range: [{scores.min():.4f}, {scores.max():.4f}]\n"
        f"Score mean: {scores.mean():.4f}\n"
        f"Reg offset: dx={dx}\n"
        f"Expansion: {RECT_H}×{RECT_W}\n"
        f"Min area: {MIN_AREA}px"
    )
    ax.text(0.1, 0.9, info, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"summary_{method}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 结果图: {save_path}")

    # 单独保存检测叠加
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.imshow(gray_img, cmap="gray")
    overlay2 = np.zeros((*gray_img.shape, 4))
    overlay2[..., 0] = 1.0
    overlay2[..., 3] = binary_mask.astype(float) * 0.6
    ax2.imshow(overlay2)
    ax2.set_title(f"{method} Detection Overlay (dx={dx}, dy={dy})")
    ax2.axis("off")
    overlay_path = os.path.join(output_dir, f"overlay_{method}.png")
    plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 叠加图: {overlay_path}")


def main():
    parser = argparse.ArgumentParser(description="真实高光谱数据端到端 Pipeline")
    parser.add_argument("--data-dir", default=None,
                        help="数据目录 (默认: ../../data/1)")
    parser.add_argument("--hi-dir", default=None,
                        help="HI 旧代码目录 (默认: ../../HI)")
    parser.add_argument("--target-dir", default=None,
                        help="目标模板目录 (默认: 从 hi-dir 加载 target1-3.npy)")
    parser.add_argument("--method", default="CEM",
                        choices=["CEM", "ACE", "SAM", "SACE", "MTICEM", "all"],
                        help="检测方法 (默认: CEM)")
    parser.add_argument("--scene", type=int, default=1, choices=[1, 2, 3],
                        help="场景编号 (默认: 1，影响检测阈值)")
    parser.add_argument("--output", default="output/real_pipeline",
                        help="输出目录")
    parser.add_argument("--min-wl", type=float, default=445,
                        help="最小保留波长 (默认: 445nm)")
    parser.add_argument("--max-wl", type=float, default=835,
                        help="最大保留波长 (默认: 835nm, 去掉饱和NIR)")
    parser.add_argument("--dx", type=int, default=195,
                        help="配准偏移 x (光谱→灰度, 默认: 195)")
    parser.add_argument("--dy", type=int, default=-30,
                        help="配准偏移 y (光谱→灰度, 默认: -30)")
    parser.add_argument("--filter", default="none",
                        choices=["none", "sky", "reflectance", "scores", "full"],
                        help="空间频域陷波滤波级别 (默认: none, full=三级全开)")
    parser.add_argument("--post", default="median5",
                        choices=["none", "median5", "median7", "median9",
                                 "gaussian", "open", "close", "med_open", "full"],
                        help="Score map 后处理方法 (默认: median5)")
    parser.add_argument("--post-kernel", type=int, default=5,
                        help="后处理滤波核大小 (默认: 5)")
    args = parser.parse_args()

    # 路径
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(base, "..", "data", "1")
    hi_dir = args.hi_dir or os.path.join(base, "..", "HI")
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("🌈 真实高光谱数据 Pipeline")
    print("=" * 65)
    t_start = time.time()

    # ── 初始化滤波器 ──
    notch_filter = None
    if args.filter != "none":
        notch_filter = NotchFilter()
        print(f"\n🌀 空间频域陷波滤波: {args.filter}")
        print(f"  {notch_filter}")
        filter_suffix = f"_filter_{args.filter}"
    else:
        filter_suffix = ""

    # ── 1. 加载图像 ──
    print("\n📂 加载 TIF 图像...")
    images = load_images(data_dir)

    # ── 2. 计算反射率 (可选滤波) ──
    print("\n📐 计算反射率...")
    reflect = compute_reflectance_cube(images, notch_filter, args.filter)

    # ── 3. 提取光谱向量 ──
    print("\n🔬 提取光谱向量...")
    data_vector, first_coords = extract_spectral_vectors(reflect, hi_dir)

    # ── 4. 加载目标光谱 ──
    print("\n🎯 加载目标模板...")
    targets = load_target_templates(hi_dir, args.target_dir)

    # ── 5. 波段过滤 + 归一化 ──
    print("\n🧹 波段过滤...")
    data_f, targets_f, good_bands = filter_bands(data_vector, targets)

    print("\n📊 归一化...")
    data_n, targets_n = normalize_data(data_f, targets_f)

    print(f"\n  最终数据: {data_n.shape}, "
          f"目标: { {i: t.shape for i, t in targets_n.items()} }")

    # ── 6. 检测 ──
    methods = ["CEM", "ACE", "SAM", "SACE", "MTICEM"] if args.method == "all" else [args.method]

    for method in methods:
        print(f"\n{'─' * 50}")
        scores, thres = run_detection(data_n, targets_n, method, args.scene)

        # ── 7. 生成分数图 ──
        print("\n🗺️  生成空间分数图 (配准偏移 dx={}, dy={})...".format(args.dx, args.dy))
        gray_img = images["gray"]
        score_map = generate_score_map(scores, first_coords, gray_img.shape,
                                       reg_offset=(args.dy, args.dx))

        # Level 3: Score Map 后处理
        post_method = args.post
        if notch_filter and args.filter in ('scores', 'full') and post_method != 'none':
            print(f"  🌀 ③ Score Map 后处理: {post_method}...")
            score_map = notch_filter.filter_score_map(score_map, method=post_method,
                                                       kernel_size=args.post_kernel)
        elif post_method != 'none':
            print(f"  🌀 Score Map 后处理 (独立模式): {post_method}...")
            from noise_filter import NotchFilter as _NF
            temp_filt = _NF()
            score_map = temp_filt.filter_score_map(score_map, method=post_method,
                                                    kernel_size=args.post_kernel)

        # ── 8. 连通区域过滤 ──
        print(f"\n🔍 连通区域过滤 (阈值={thres})...")
        binary = filter_connected_components(score_map, thres)

        # ── 9. 可视化 ──
        print("\n🎨 生成可视化...")
        method_label = f"{method}+Filter" if notch_filter else method
        visualize_results(gray_img, score_map, binary, scores, method_label, output_dir,
                          threshold=thres, reg_offset=(args.dy, args.dx))

    t_elapsed = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"✅ 完成! 耗时: {t_elapsed:.1f}s")
    print(f"📂 输出: {os.path.abspath(output_dir)}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
