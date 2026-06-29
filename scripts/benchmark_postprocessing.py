#!/usr/bin/env python3
"""
Score Map 后处理策略对比基准测试。

对所有 5 种检测算法 (ACE/CEM/SAM/SACE/MTICEM) 分别测试多种
score map 后处理策略，找出每个算法的最佳后处理，以及全局最强算法。

后处理策略:
  none       — 无后处理 (原始 score map)
  median5    — 5×5 中值滤波 (原方案)
  median7    — 7×7 中值滤波 (更强平滑)
  median9    — 9×9 中值滤波 (最强平滑)
  open       — 形态学开运算 (先腐蚀后膨胀，消除小噪点)
  med_open   — 中值5×5 + 开运算 (推荐组合)
  close      — 形态学闭运算 (填充小空洞)
  gaussian   — 高斯滤波 (σ=1.67)
  full       — 中值5×5 + 开运算 + 中值5×5 (最强去噪)

用法:
  cd HI_refactored && python3 scripts/benchmark_postprocessing.py --dx 195 --dy -30

输出:
  output/benchmark/
  ├── comparison_table.txt          — 所有策略的定量对比表
  ├── winner_report.txt             — 最佳策略总结
  ├── heatmap_{method}.png          — 各算法后处理对比热力图
  └── best_overlay_{method}.png     — 最佳后处理的检测叠加图
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import (
    subtract_dark_current, compute_reflectance,
    detect_saturated_bands, normalize_reflectance,
)
from detection.cem import CEMDetector
from detection.ace import ACEDetector
from detection.sam import SpectralAngleMapper as SAMDetector
from detection.mticem import MTICEMDetector
from detection.sace import SACEDetector
from noise_filter import NotchFilter

WAVELENGTHS = np.arange(445, 906, 5, dtype=int)
RECT_H, RECT_W = 6, 53
MIN_AREA = 1117

# ── 待测试的所有后处理策略 ──
POST_STRATEGIES = [
    ("none",       "无后处理",       ""),
    ("median5",    "5×5 中值滤波",   "原方案"),
    ("median7",    "7×7 中值滤波",   "更强平滑"),
    ("median9",    "9×9 中值滤波",   "最强中值"),
    ("open",       "形态学开运算",   "消除小噪点"),
    ("close",      "形态学闭运算",   "填充小空洞"),
    ("med_open",   "中值+开运算",    "推荐组合"),
    ("gaussian",   "高斯滤波",       "σ=1.67"),
    ("full",       "中值+开+中值",   "最强去噪"),
]

# ── 检测方法 ──
METHODS = ["ACE", "CEM", "SAM", "SACE", "MTICEM"]
METHOD_LABELS = {
    "ACE": "ACE 自适应余弦",
    "CEM": "CEM 约束能量最小化",
    "SAM": "SAM 光谱角",
    "SACE": "SACE 自适应余弦+角度约束",
    "MTICEM": "MTICEM 多目标约束能量",
}

THRESHOLDS = {"CEM": 1.5, "ACE": 0.18, "SAM": 0.975, "SACE": 0.18, "MTICEM": 1.5}


def load_and_prepare(data_dir, hi_dir, reg_offset=(-30, 195)):
    """加载数据，计算滤波后的反射率，提取光谱向量和目标模板。"""
    print("📂 加载数据...")
    spec = tifffile.imread(os.path.join(data_dir, "5ms.tif"))
    dark = tifffile.imread(os.path.join(data_dir, "P11070000.tif"))
    sky = tifffile.imread(os.path.join(data_dir, "5ms_sky.tif"))
    gray = tifffile.imread(os.path.join(data_dir, "view2.tif"))

    with open(os.path.join(hi_dir, "coords_dict.json")) as f:
        coords_dict = json.load(f)

    print("📐 计算滤波后反射率 (full notch)...")
    nf = NotchFilter()
    spec_ds = subtract_dark_current(spec, dark)
    sky_ds = subtract_dark_current(sky, dark)
    sky_clean = nf.filter_image_2d(sky_ds)
    refl = compute_reflectance(spec_ds, sky_clean)
    refl_3d = refl[:, :, np.newaxis]
    refl_clean = nf.filter_reflectance_cube(refl_3d)
    refl = refl_clean[:, :, 0]

    print("🔬 提取光谱向量...")
    valid_items = [(k, v) for k, v in coords_dict.items() if len(v) == 93]
    n_pts = len(valid_items)
    data = np.zeros((n_pts, 93), dtype=np.float64)
    first_coords = np.zeros((n_pts, 2), dtype=int)
    for i, (_, spec_list) in enumerate(valid_items):
        data[i] = np.array([refl[s[1], s[2]] for s in spec_list], dtype=np.float64)
        first_coords[i] = [spec_list[0][1], spec_list[0][2]]

    print("🎯 从滤波后反射率提取目标模板...")
    with open(os.path.join(hi_dir, 'id_to_key.json')) as f:
        id_to_key = json.load(f)
    mask = np.load(os.path.join(hi_dir, 'dataset/mask.npy'))
    mapping = {4: 1, 5: 2, 6: 3}
    class_mask = np.vectorize(mapping.get)(mask, 0)
    targets = {}
    for class_id in [1, 2, 3]:
        ys, xs = np.where(class_mask == class_id)
        vectors = []
        for y, x in zip(ys, xs):
            key = f'({y}, {x})'
            if key in id_to_key:
                idx = id_to_key[key]
                idx_str = str(idx)
                if idx_str in coords_dict:
                    spec_list = coords_dict[idx_str]
                    vec = np.array([refl[sy, sx] for _, sy, sx in spec_list], dtype=np.float64)
                    vectors.append(vec)
        targets[class_id] = np.array(vectors)
        print(f"  target{class_id}: {vectors[0].shape if vectors else 'empty'}, n={len(vectors)}")

    print("🧹 波段过滤 + 归一化...")
    good, bad = detect_saturated_bands(data, threshold_ratio=10.0)
    data_f = data[:, good]
    targets_f = {}
    for i, t in targets.items():
        targets_f[i] = t[:, good] if t.shape[1] == 93 else t

    data_n = normalize_reflectance(data_f, method="mean")
    targets_n = {}
    for i, t in targets_f.items():
        targets_n[i] = normalize_reflectance(t, method="mean")

    return data_n, targets_n, first_coords, gray, good, nf


def run_single_detection(data, targets, method):
    """针对单个检测方法运行，返回 scores (N,)。"""
    target_list = [targets[i].mean(axis=0) for i in [1, 2, 3]]
    thres = THRESHOLDS.get(method, 0.18)

    if method == "CEM":
        scores_multi = np.zeros((data.shape[0], 3))
        for ti, tgt in enumerate(target_list):
            det = CEMDetector(reg=1e-6)
            det.fit(data, tgt)
            scores_multi[:, ti] = det.predict(data)
        scores = scores_multi.max(axis=1)
    elif method == "ACE":
        scores_multi = np.zeros((data.shape[0], 3))
        for ti, tgt in enumerate(target_list):
            det = ACEDetector(reg=1e-6)
            det.fit(data, tgt)
            scores_multi[:, ti] = det.predict(data)
        scores = scores_multi.max(axis=1)
    elif method == "SAM":
        scores_multi = np.zeros((data.shape[0], 3))
        for ti, tgt in enumerate(target_list):
            det = SAMDetector(normalize=True)
            det.fit(tgt[np.newaxis, :])
            angles = det.predict(data)
            scores_multi[:, ti] = 1.0 - angles / np.pi
        scores = scores_multi.max(axis=1)
    elif method == "SACE":
        scores_multi = np.zeros((data.shape[0], 3))
        for ti, tgt in enumerate(target_list):
            det = SACEDetector(reg=1e-6)
            det.fit(data, tgt)
            scores_multi[:, ti] = det.predict(data)
        scores = scores_multi.max(axis=1)
    elif method == "MTICEM":
        D = np.array(target_list)
        det = MTICEMDetector(reg=1e-6)
        det.fit(data, D)
        scores_multi = det.predict(data)
        scores = scores_multi.max(axis=1)

    return scores, thres


def make_score_map(scores, first_coords, gray_shape, reg_offset=(-30, 195)):
    """将 1D 分数映射到 2D score map。"""
    H, W = gray_shape
    dy, dx = reg_offset
    score_map = np.zeros((H, W), dtype=np.float64)
    order = np.argsort(first_coords[:, 1])
    cs = first_coords[order]
    ss = scores[order]
    for (y, x), s in zip(cs, ss):
        yg, xg = y + dy, x + dx
        if yg < 0 or yg >= H or xg < 0 or xg >= W:
            continue
        y1, y2 = max(0, yg), min(yg + RECT_H, H)
        x1, x2 = max(0, xg), min(xg + RECT_W, W)
        score_map[y1:y2, x1:x2] = s
    return score_map


def evaluate_score_map(score_map, threshold, min_area=MIN_AREA):
    """评估 score map 的质量指标。"""
    from scipy import ndimage as ndi
    binary = score_map > threshold
    labeled, num_features = ndi.label(binary, structure=np.ones((3, 3)))
    comp_sizes = np.bincount(labeled.ravel())

    keep = np.zeros_like(binary, dtype=bool)
    kept_count = 0
    for label_id in range(1, num_features + 1):
        if label_id < len(comp_sizes) and comp_sizes[label_id] >= min_area:
            keep[labeled == label_id] = True
            kept_count += 1

    # 背景噪声 (非检测区域)
    bg = score_map[~keep]
    bg_std = bg.std() if len(bg) > 0 else 0
    bg_mean = bg.mean() if len(bg) > 0 else 0

    return {
        "det_pixels": int(binary.sum()),
        "keep_pixels": int(keep.sum()),
        "n_components": num_features,
        "n_kept": kept_count,
        "bg_mean": float(bg_mean),
        "bg_std": float(bg_std),
        "max_score": float(score_map.max()),
        "mean_score": float(score_map[score_map > 0].mean()) if (score_map > 0).sum() > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Score Map 后处理策略对比")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--hi-dir", default=None)
    parser.add_argument("--output", default="output/benchmark")
    parser.add_argument("--dx", type=int, default=195)
    parser.add_argument("--dy", type=int, default=-30)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(base, "..", "data", "1")
    hi_dir = args.hi_dir or os.path.join(base, "..", "HI")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── 准备数据 (一次加载，复用) ──
    data_n, targets_n, first_coords, gray_img, good, nf = \
        load_and_prepare(data_dir, hi_dir, (args.dy, args.dx))

    gray_shape = gray_img.shape

    # ═══════════════════════════════════════════════
    # 遍历所有检测方法 × 所有后处理策略
    # ═══════════════════════════════════════════════
    results = {}  # {method: {post_key: metrics_dict}}
    best_per_method = {}  # {method: (post_key, metrics)}

    for method in METHODS:
        print(f"\n{'='*55}")
        print(f"🎯 {method} ({METHOD_LABELS[method]})")
        print(f"{'='*55}")

        # 运行检测 (一次)
        print(f"  运行检测...")
        scores, thres = run_single_detection(data_n, targets_n, method)
        score_map_base = make_score_map(scores, first_coords, gray_shape,
                                         reg_offset=(args.dy, args.dx))
        print(f"  分数范围: [{scores.min():.4f}, {scores.max():.4f}]")

        method_results = {}
        best_score = -1
        best_key = None

        for post_key, post_label, post_desc in POST_STRATEGIES:
            if post_key == "none":
                sm = score_map_base.copy()
            else:
                print(f"    ├─ {post_label:14s} ({post_desc})...", end=" ")
                sm = nf.filter_score_map(score_map_base, method=post_key,
                                          kernel_size=5, morph_kernel=5)

            metrics = evaluate_score_map(sm, thres)
            method_results[post_key] = metrics
            print(f"检测={metrics['keep_pixels']:5d}px, "
                  f"背景噪声σ={metrics['bg_std']:.5f}, "
                  f"最大分={metrics['max_score']:.4f}")

            # 评分: 越少的检测像素 + 越低的背景噪声 = 越好
            # 但也要考虑合理的保留检测
            noise_score = metrics['bg_std'] * 100  # 噪声权重
            det_score = metrics['keep_pixels'] / 50000  # 检测量权重 (归一化)
            total = noise_score + det_score
            # 越低越好
            if best_key is None or total < best_score:
                best_score = total
                best_key = post_key

        results[method] = method_results
        best_per_method[method] = (best_key, method_results[best_key])
        print(f"  ✅ {method} 最佳后处理: {best_key} "
              f"(检测={method_results[best_key]['keep_pixels']}px, "
              f"噪声σ={method_results[best_key]['bg_std']:.5f})")

    # ═══════════════════════════════════════════════
    # 生成对比表
    # ═══════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("📊 生成对比表...")
    print(f"{'='*65}")

    lines = []
    lines.append("=" * 90)
    lines.append("  Score Map 后处理策略基准测试对比表")
    lines.append("=" * 90)
    header = f"{'策略':>12s} | {'算法':>7s} | {'检测像素':>8s} | {'保留像素':>8s} | {'连通域':>5s} | {'保留区域':>5s} | {'背景均值':>8s} | {'背景σ':>8s} | {'最大分':>7s}"
    lines.append(header)
    lines.append("-" * 90)

    for method in METHODS:
        for post_key, post_label, _ in POST_STRATEGIES:
            m = results[method][post_key]
            is_best = (post_key == best_per_method[method][0])
            marker = "★" if is_best else " "
            lines.append(
                f"{marker}{post_label:>11s} | {method:>7s} | "
                f"{m['det_pixels']:>8d} | {m['keep_pixels']:>8d} | "
                f"{m['n_components']:>5d} | {m['n_kept']:>5d} | "
                f"{m['bg_mean']:>8.4f} | {m['bg_std']:>8.5f} | {m['max_score']:>7.4f}"
            )

    # 全局最佳 (检测像素最少 + 背景噪声最低的平衡)
    print("\n📈 计算全局最佳...")
    global_scores = []
    for method in METHODS:
        for post_key, post_label, _ in POST_STRATEGIES:
            m = results[method][post_key]
            # 综合得分: 保留像素越少越好 + 背景σ越低越好
            score = m['keep_pixels'] / 1000 + m['bg_std'] * 5000
            global_scores.append((score, method, post_key, post_label, m))

    global_scores.sort()
    lines.append("-" * 90)
    lines.append("")

    # Top 10 排序
    lines.append("─" * 60)
    lines.append("  综合排名 (衡量标准: 保留像素/1000 + 背景σ×5000，越低越好)")
    lines.append("─" * 60)
    lines.append(f"{'排名':>4s} | {'算法':>7s} | {'后处理':>14s} | {'保留像素':>8s} | {'背景σ':>8s} | {'综合分':>8s}")
    lines.append("-" * 60)
    for rank, (score, method, post_key, post_label, m) in enumerate(global_scores[:10], 1):
        lines.append(f"{rank:>4d} | {method:>7s} | {post_label:>14s} | "
                     f"{m['keep_pixels']:>8d} | {m['bg_std']:>8.5f} | {score:>8.2f}")

    # 每个算法的最佳策略表
    lines.append("")
    lines.append("─" * 50)
    lines.append("  各算法最佳后处理策略")
    lines.append("─" * 50)
    lines.append(f"{'算法':>7s} | {'最佳后处理':>14s} | {'保留像素':>8s} | {'背景σ':>8s}")
    lines.append("-" * 50)
    for method in METHODS:
        best_key, best_metrics = best_per_method[method]
        best_label = [l for k, l, _ in POST_STRATEGIES if k == best_key][0]
        lines.append(f"{method:>7s} | {best_label:>14s} | "
                     f"{best_metrics['keep_pixels']:>8d} | {best_metrics['bg_std']:>8.5f}")

    # 全局最佳组合
    best_overall = global_scores[0]
    lines.append("")
    lines.append("=" * 50)
    lines.append(f"🏆 全局最佳: {best_overall[1]} + {best_overall[3]}")
    lines.append(f"   保留像素: {best_overall[4]['keep_pixels']}")
    lines.append(f"   背景噪声σ: {best_overall[4]['bg_std']:.5f}")
    lines.append(f"   综合得分: {best_overall[0]:.2f}")
    lines.append("=" * 50)

    table = "\n".join(lines)
    print(table)

    with open(output_dir / "comparison_table.txt", "w") as f:
        f.write(table)

    # 保存全局最佳的报告
    winner_report = f"""
{'='*60}
🏆 全局最佳检测方案
{'='*60}

最佳算法: {best_overall[1]}
最佳后处理: {best_overall[3]}
检测保留像素: {best_overall[4]['keep_pixels']}
背景噪声标准差: {best_overall[4]['bg_std']:.5f}
最大检测分数: {best_overall[4]['max_score']:.4f}
连通区域数: {best_overall[4]['n_components']} → 保留 {best_overall[4]['n_kept']}

对比: 无滤波无后处理
  ACE + none: 检测像素 ~89,659, 噪声σ ~0.0325

改善幅度:
  检测像素减少: {100*(1-best_overall[4]['keep_pixels']/89659):.1f}%
  背景噪声降低: {100*(1-best_overall[4]['bg_std']/0.032523):.1f}%

各算法最佳:
"""
    for method in METHODS:
        bk, bm = best_per_method[method]
        bl = [l for k, l, _ in POST_STRATEGIES if k == bk][0]
        winner_report += f"  {method:>7s}: {bl:>14s} → 检测={bm['keep_pixels']:>5d}px, 噪声σ={bm['bg_std']:.5f}\n"

    winner_report += f"""
{'='*60}
"""
    print(winner_report)
    with open(output_dir / "winner_report.txt", "w") as f:
        f.write(winner_report)

    # ═══════════════════════════════════════════════
    # 生成可视化: 各算法的后处理对比热力图
    # ═══════════════════════════════════════════════
    print("\n🎨 生成对比热力图...")
    for method in METHODS:
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        # 重新跑 score map
        scores, thres = run_single_detection(data_n, targets_n, method)
        score_map_base = make_score_map(scores, first_coords, gray_shape,
                                         reg_offset=(args.dy, args.dx))

        for idx, (post_key, post_label, post_desc) in enumerate(POST_STRATEGIES):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]

            if post_key == "none":
                sm = score_map_base.copy()
            else:
                sm = nf.filter_score_map(score_map_base, method=post_key,
                                          kernel_size=5, morph_kernel=5)

            from scipy import ndimage as ndi
            binary = sm > thres
            labeled, _ = ndi.label(binary, structure=np.ones((3, 3)))
            comp_sizes = np.bincount(labeled.ravel())
            keep = np.zeros_like(binary, dtype=bool)
            for lid in range(1, labeled.max() + 1):
                if lid < len(comp_sizes) and comp_sizes[lid] >= MIN_AREA:
                    keep[labeled == lid] = True

            vmax = np.percentile(sm[sm > 0], 95) if sm.max() > 0 else 0.2
            im = ax.imshow(sm, cmap='jet', vmin=0, vmax=vmax)
            ax.set_title(f"{post_label}\n检测={int(keep.sum())}px, σ={sm[~keep].std():.4f}",
                         fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_{method}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ {method} 热力图已保存")

    elapsed = time.time() - t0
    print(f"\n✅ 基准测试完成! 耗时: {elapsed:.1f}s")
    print(f"📂 输出: {output_dir}")


if __name__ == "__main__":
    main()
