#!/usr/bin/env python3
"""
ACE 目标样本数量影响测试。

分别用 1、3、5 个目标样本（从 target1-3 各取前 N 个）运行 ACE 检测，
比较目标光谱稳定性对检测结果的影响。

用法:
  python scripts/test_ace_samples.py
  python scripts/test_ace_samples.py --dx 195 --dy -30
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import (
    subtract_dark_current,
    compute_reflectance,
    detect_saturated_bands,
    normalize_reflectance,
)
from detection.ace import ACEDetector

WAVELENGTHS = np.arange(445, 906, 5, dtype=int)
RECT_H, RECT_W = 6, 53
MIN_AREA = 1117


def load_images(data_dir: str) -> dict:
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
        images[name] = tifffile.imread(p)
    return images


def compute_reflectance_cube(images: dict) -> np.ndarray:
    img_spec = subtract_dark_current(images["spec_base"], images["dark"])
    img_sky = subtract_dark_current(images["illuminance"], images["dark"])
    reflect = compute_reflectance(img_spec, img_sky)
    print(f"  反射率: {reflect.shape}, 范围=[{reflect.min():.4f},{reflect.max():.1f}]")
    return reflect


def extract_spectral_vectors(reflect: np.ndarray, hi_dir: str) -> tuple:
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
    return data_vector, first_coords


def load_target_templates(hi_dir: str) -> dict:
    targets = {}
    for i in [1, 2, 3]:
        path = os.path.join(hi_dir, f"target{i}.npy")
        if os.path.exists(path):
            targets[i] = np.load(path)
    return targets


def filter_bands(data: np.ndarray, targets: dict) -> tuple:
    good, bad = detect_saturated_bands(data, threshold_ratio=10.0)
    data_f = data[:, good]
    target_spec = {}
    for i, t in targets.items():
        t_f = t[:, good] if t.shape[1] == 93 else t
        target_spec[i] = t_f
    return data_f, target_spec, good


def normalize_data(data: np.ndarray, targets: dict) -> tuple:
    data_n = normalize_reflectance(data, method="mean")
    targets_n = {}
    for i, t in targets.items():
        targets_n[i] = normalize_reflectance(t, method="mean")
    return data_n, targets_n


def generate_score_map(scores, first_coords, gray_shape, dy, dx):
    H, W = gray_shape
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
        score_map[y1:y2, x1:x2] = s
    return score_map


def filter_connected_components(score_map, threshold):
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
    return keep, num_features, kept_count


def visualize_results(gray_img, score_map, binary_mask, scores, threshold,
                      output_dir, method, dx, dy):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes[0, 0]
    ax.imshow(gray_img, cmap="gray")
    ax.set_title("Visible Grayscale (view2)")
    ax.axis("off")
    ax = axes[0, 1]
    im = ax.imshow(score_map, cmap="jet", vmin=0)
    ax.set_title(f"{method} Score Map")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[0, 2]
    ax.imshow(gray_img, cmap="gray")
    overlay = np.zeros((*gray_img.shape, 4))
    overlay[..., 0] = 1.0
    overlay[..., 3] = binary_mask.astype(float) * 0.6
    ax.imshow(overlay)
    ax.set_title(f"{method} Detection")
    ax.axis("off")
    ax = axes[1, 0]
    ax.hist(scores, bins=100, alpha=0.7)
    ax.axvline(threshold, color="r", linestyle="--", label=f"th={threshold}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Score Distribution")
    ax = axes[1, 1]
    order = np.argsort(scores)[::-1]
    ax.plot(np.arange(len(scores)), scores[order], lw=1)
    ax.axhline(threshold, color="r", linestyle="--", label=f"th={threshold}")
    ax.set_xlabel("Pixel (sorted)")
    ax.set_ylabel("Score")
    ax.set_title("Score Ranking")
    ax.legend()
    ax = axes[1, 2]
    ax.axis("off")
    info = (
        f"Method: {method}\n"
        f"Total pixels: {len(scores)}\n"
        f"Detected: {binary_mask.sum()}\n"
        f"Score range: [{scores.min():.4f}, {scores.max():.4f}]\n"
        f"Score mean: {scores.mean():.4f}\n"
        f"Reg offset: dx={dx}, dy={dy}\n"
        f"Expansion: {RECT_H}×{RECT_W}\n"
        f"Min area: {MIN_AREA}px"
    )
    ax.text(0.1, 0.9, info, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"summary_{method}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.imshow(gray_img, cmap="gray")
    overlay2 = np.zeros((*gray_img.shape, 4))
    overlay2[..., 0] = 1.0
    overlay2[..., 3] = binary_mask.astype(float) * 0.6
    ax2.imshow(overlay2)
    ax2.set_title(f"{method} Detection Overlay")
    ax2.axis("off")
    plt.savefig(os.path.join(output_dir, f"overlay_{method}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ACE 样本数影响测试")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--hi-dir", default=None)
    parser.add_argument("--dx", type=int, default=195)
    parser.add_argument("--dy", type=int, default=-30)
    parser.add_argument("--output", default="output/ace_samples")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(base, "..", "data", "1")
    hi_dir = args.hi_dir or os.path.join(base, "..", "HI")
    output_base = args.output

    print("=" * 65)
    print("🌈 ACE 目标样本数量影响测试")
    print("=" * 65)
    t_start = time.time()

    # ── 加载数据（一次性） ──
    print("\n📂 加载 TIF 图像...")
    images = load_images(data_dir)
    print("\n📐 计算反射率...")
    reflect = compute_reflectance_cube(images)
    print("\n🔬 提取光谱向量...")
    data_vector, first_coords = extract_spectral_vectors(reflect, hi_dir)
    print("\n🎯 加载目标模板...")
    targets_full = load_target_templates(hi_dir)
    print("\n🧹 波段过滤...")
    data_f, targets_f, good_bands = filter_bands(data_vector, targets_full)
    print("\n📊 归一化...")
    data_n, targets_n = normalize_data(data_f, targets_f)

    target_labels = ["草地伪装网", "军绿迷彩", "沙漠迷彩"]

    # ── 对 1、3、5 个样本分别测试 ──
    for n_samples in [1, 3, 5, None]:
        if n_samples is None:
            tag = "all"
            label = "全部样本"
        else:
            tag = str(n_samples)
            label = f"前{n_samples}个样本"

        output_dir = os.path.join(output_base, f"ace_samples_{tag}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"🔬 ACE 测试: 每类目标取 {label}")
        print(f"{'=' * 50}")

        # 截取目标样本
        targets_sub = {}
        for i in [1, 2, 3]:
            t = targets_n[i]
            if n_samples is not None and n_samples < t.shape[0]:
                targets_sub[i] = t[:n_samples]
            else:
                targets_sub[i] = t

        # 对 3 类目标各跑 ACE，取 max
        scores_multi = np.zeros((data_n.shape[0], 3))
        for ti in [0, 1, 2]:
            tgt = targets_sub[ti + 1]
            d = tgt.mean(axis=0)
            det = ACEDetector(reg=1e-6)
            det.fit(data_n, d)
            s = det.predict(data_n)
            scores_multi[:, ti] = s
            print(f"  target{ti+1} ({target_labels[ti]}, N={tgt.shape[0]}): "
                  f"范围=[{s.min():.4f}, {s.max():.4f}]")

        scores = scores_multi.max(axis=1)

        # 生成分数图 + 连通区域过滤
        thres = 0.18
        gray_img = images["gray"]
        score_map = generate_score_map(scores, first_coords, gray_img.shape, args.dy, args.dx)
        binary, n_components, n_kept = filter_connected_components(score_map, thres)

        print(f"  ─────────────────────────────")
        print(f"  最终分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  检测像素: {(scores > thres).sum()} / {len(scores)} "
              f"({100*(scores > thres).sum()/len(scores):.1f}%)")
        print(f"  连通区域: {n_components} 个, 保留 ≥{MIN_AREA}px: {n_kept} 个")

        # 可视化
        method = f"ACE_N{tag}"
        visualize_results(gray_img, score_map, binary, scores, thres,
                          output_dir, method, args.dx, args.dy)
        print(f"  ✅ 结果已保存到 {output_dir}/")

    t_elapsed = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"✅ 全部完成! 耗时: {t_elapsed:.1f}s")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
