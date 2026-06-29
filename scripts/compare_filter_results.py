#!/usr/bin/env python3
"""
水波纹噪声滤波效果对比验证。

同时运行 ACE 检测 with/without 滤波，对比:
  1. Column-mean FFT (验证 17.5px 峰被抑制)
  2. Score map 视觉对比 (验证同心圆消失)
  3. 检测结果对比 (验证目标信息保留)
  4. 定量指标: 背景噪声标准差下降

用法:
  cd HI_refactored && python3 scripts/compare_filter_results.py

输出: output/filter_comparison/ 目录下
"""

import argparse
import json
import os
import sys
import time
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.signal import find_peaks
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import (
    subtract_dark_current, compute_reflectance,
    detect_saturated_bands, normalize_reflectance,
)
from detection.ace import ACEDetector
from noise_filter import NotchFilter, analyze_fft

WAVELENGTHS = np.arange(445, 906, 5, dtype=int)
RECT_H, RECT_W = 6, 53
MIN_AREA = 1117
BAND_IDX = np.arange(93)


def load_data(data_dir, hi_dir):
    """Load all TIFs and compute reflectance."""
    spec = tifffile.imread(os.path.join(data_dir, "5ms.tif"))
    dark = tifffile.imread(os.path.join(data_dir, "P11070000.tif"))
    sky = tifffile.imread(os.path.join(data_dir, "5ms_sky.tif"))
    gray = tifffile.imread(os.path.join(data_dir, "view2.tif"))

    with open(os.path.join(hi_dir, "coords_dict.json")) as f:
        coords_dict = json.load(f)

    return spec, dark, sky, gray, coords_dict


def compute_reflectance_with_filter(spec, dark, sky, notch_filter=None, filter_mode='none'):
    """Compute reflectance with optional notch filtering."""
    spec_ds = subtract_dark_current(spec, dark)
    sky_ds = subtract_dark_current(sky, dark)

    if notch_filter and filter_mode in ('sky', 'full'):
        sky_ds = notch_filter.filter_image_2d(sky_ds)

    refl = compute_reflectance(spec_ds, sky_ds)

    if notch_filter and filter_mode in ('reflectance', 'full'):
        refl_3d = refl[:, :, np.newaxis]
        refl_clean = notch_filter.filter_reflectance_cube(refl_3d)
        refl = refl_clean[:, :, 0]

    return refl


def extract_vectors(reflect, coords_dict):
    """Extract spectral vectors from reflectance via coords_dict."""
    valid_items = [(k, v) for k, v in coords_dict.items() if len(v) == 93]
    n_pts = len(valid_items)
    data = np.zeros((n_pts, 93), dtype=np.float64)
    first_pos = np.zeros((n_pts, 2), dtype=int)
    for i, (_, spec) in enumerate(valid_items):
        data[i] = np.array([reflect[s[1], s[2]] for s in spec], dtype=np.float64)
        first_pos[i] = [spec[0][1], spec[0][2]]
    return data, first_pos


def load_targets(hi_dir):
    """Load pre-extracted target templates (from unfiltered reflectance)."""
    targets = {}
    for i in [1, 2, 3]:
        path = os.path.join(hi_dir, f"target{i}.npy")
        if os.path.exists(path):
            targets[i] = np.load(path)
    return targets


def extract_targets_from_reflectance(reflect, hi_dir):
    """从反射率立方体中提取目标光谱 (使用 id_to_key + mask + coords_dict).

    与 HI/dataset.py 中的 get_target_data() 逻辑一致，
    但允许从任意反射率 (如滤波后的) 中提取。

    返回: {1: (N1,93), 2: (N2,93), 3: (N3,93)}
    """
    with open(os.path.join(hi_dir, 'id_to_key.json')) as f:
        id_to_key = json.load(f)
    mask = np.load(os.path.join(hi_dir, 'dataset/mask.npy'))
    with open(os.path.join(hi_dir, 'coords_dict.json')) as f:
        coords_dict = json.load(f)

    # mask 值 4/5/6 → target 类别 1/2/3
    mapping = {4: 1, 5: 2, 6: 3}
    from numpy import vectorize
    class_mask = vectorize(mapping.get)(mask, 0)

    targets = {}
    for class_id in [1, 2, 3]:
        ys, xs = np.where(class_mask == class_id)
        vectors = []
        missed = 0
        for y, x in zip(ys, xs):
            key = f'({y}, {x})'
            if key in id_to_key:
                idx = id_to_key[key]
                idx_str = str(idx)
                if idx_str in coords_dict:
                    spec_list = coords_dict[idx_str]
                    data = np.zeros(93, dtype=np.float64)
                    for b, sy, sx in spec_list:
                        data[b] = reflect[sy, sx]
                    vectors.append(data)
                else:
                    missed += 1
            else:
                missed += 1
        targets[class_id] = np.array(vectors)
        print(f"  target{class_id}: 提取 {len(vectors)} 条光谱 (missed={missed})")
    return targets


def run_ace_detection(data, targets, good_bands):
    """Run ACE detection with all 3 targets (multi-target max)."""
    target_list = [targets[i].mean(axis=0) for i in [1, 2, 3]]
    scores_multi = np.zeros((data.shape[0], 3))
    for ti, tgt in enumerate(target_list):
        det = ACEDetector(reg=1e-6)
        det.fit(data, tgt)
        scores_multi[:, ti] = det.predict(data)
    return scores_multi.max(axis=1)


def make_score_map(scores, first_coords, gray_shape, dy=-30, dx=195):
    """Map 1D scores to 2D score map."""
    H, W = gray_shape
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


def compute_column_fft(image):
    """Compute column-mean FFT of an image."""
    col = image.mean(axis=0).astype(np.float64)
    col -= col.mean()
    fft = np.abs(np.fft.rfft(col))
    freqs = np.fft.rfftfreq(len(col))
    return freqs, fft


def is_value(*args):
    return True


def main():
    parser = argparse.ArgumentParser(description="滤波效果对比验证")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--hi-dir", default=None)
    parser.add_argument("--output", default="output/filter_comparison")
    parser.add_argument("--dx", type=int, default=195)
    parser.add_argument("--dy", type=int, default=-30)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(base, "..", "data", "1")
    hi_dir = args.hi_dir or os.path.join(base, "..", "HI")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("📊 水波纹滤波效果对比验证")
    print("=" * 65)
    t0 = time.time()

    # ── 加载 ──
    print("\n📂 加载数据...")
    spec, dark, sky, gray, coords_dict = load_data(data_dir, hi_dir)
    targets_raw = load_targets(hi_dir)  # 来自 HI/target{1-3}.npy (未滤波的)

    # ── 创建滤波器 ──
    nf = NotchFilter()
    print(f"  滤波器: {nf}")

    # ── 计算反射率 (两种模式) ──
    print("\n📐 计算反射率 (无滤波)...")
    ref_raw = compute_reflectance_with_filter(spec, dark, sky)
    print("📐 计算反射率 (有滤波)...")
    ref_filt = compute_reflectance_with_filter(spec, dark, sky, nf, 'full')

    # ── 提取目标光谱 (分别从原始/滤波后的反射率中提取) ──
    print("\n🎯 提取目标光谱...")
    print("  ── 从原始反射率 ──")
    targets_raw_extracted = extract_targets_from_reflectance(ref_raw, hi_dir)
    print("  ── 从滤波后反射率 ──")
    targets_filt = extract_targets_from_reflectance(ref_filt, hi_dir)

    # ── 提取光谱向量 ──
    print("\n🔬 提取光谱向量...")
    data_raw, first_coords = extract_vectors(ref_raw, coords_dict)
    data_filt, _ = extract_vectors(ref_filt, coords_dict)

    # ── 波段过滤 + 归一化 ──
    good, bad = detect_saturated_bands(data_raw, threshold_ratio=10.0)
    data_raw_f = data_raw[:, good]
    data_filt_f = data_filt[:, good]
    # 未滤波目标 (来自 HI/target{1-3}.npy) 用于原始检测
    targets_raw_f = {}
    for i, t in targets_raw.items():
        targets_raw_f[i] = (t[:, good] if t.shape[1] == 93 else t)
    # 滤波后目标 (从滤波反射率中提取) 用于滤波检测
    targets_filt_f = {}
    for i, t in targets_filt.items():
        targets_filt_f[i] = (t[:, good] if t.shape[1] == 93 else t)

    data_raw_n = normalize_reflectance(data_raw_f, method="mean")
    data_filt_n = normalize_reflectance(data_filt_f, method="mean")
    targets_raw_n = {}
    for i, t in targets_raw_f.items():
        targets_raw_n[i] = normalize_reflectance(t, method="mean")
    targets_filt_n = {}
    for i, t in targets_filt_f.items():
        targets_filt_n[i] = normalize_reflectance(t, method="mean")

    # ── ACE 检测 ──
    print("\n🎯 ACE 检测...")
    print("  ── 无滤波 (目标/场景均来自原始反射率) ──")
    scores_raw = run_ace_detection(data_raw_n, targets_raw_n, good)
    print(f"    分数范围: [{scores_raw.min():.4f}, {scores_raw.max():.4f}]")
    print("  ── 有滤波 (目标/场景均来自滤波后反射率) ──")
    scores_filt = run_ace_detection(data_filt_n, targets_filt_n, good)
    print(f"    分数范围: [{scores_filt.min():.4f}, {scores_filt.max():.4f}]")

    # ── 生成 score map ──
    print("\n🗺️  生成分数图...")
    gray_shape = gray.shape
    sm_raw = make_score_map(scores_raw, first_coords, gray_shape, args.dy, args.dx)
    sm_filt = make_score_map(scores_filt, first_coords, gray_shape, args.dy, args.dx)

    # Score map 后处理 (filtered 版)
    sm_filt_smooth = nf.filter_score_map(sm_filt, method='median', kernel_size=5)

    # ── 连通区域 ──
    thres = 0.18
    binary_raw = sm_raw > thres
    binary_filt = sm_filt_smooth > thres

    # ═══════════════════════════════════════════
    # 对比图 1: Column FFT 对比
    # ═══════════════════════════════════════════
    print("\n📈 生成 FFT 对比图...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Sky FFT
    sky_ds = subtract_dark_current(sky, dark)
    _, sky_fft_raw = compute_column_fft(sky_ds)
    sky_filt_ds = nf.filter_image_2d(sky_ds)
    _, sky_fft_filt = compute_column_fft(sky_filt_ds)
    sky_freq = np.fft.rfftfreq(sky.shape[1])

    ax = axes[0, 0]
    ax.plot(sky_freq[:200], sky_fft_raw[:200], 'b-', lw=1.2, alpha=0.8, label='Raw Sky')
    ax.plot(sky_freq[:200], sky_fft_filt[:200], 'r-', lw=1.2, alpha=0.8, label='Filtered Sky')
    ax.axvline(1/17.5, color='gray', ls='--', lw=0.5)
    ax.annotate("17.5px", (1/17.5, ax.get_ylim()[1]*0.9), fontsize=8, ha='center', color='gray')
    ax.axvline(1/8.8, color='gray', ls='--', lw=0.5)
    ax.annotate("8.8px", (1/8.8, ax.get_ylim()[1]*0.9), fontsize=8, ha='center', color='gray')
    ax.set_xlim(0, 0.2)
    ax.set_xlabel("Frequency (px⁻¹)")
    ax.set_ylabel("|FFT|")
    ax.set_title("Sky Column FFT: Raw vs Filtered")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 17.5px peak zoom
    ax = axes[0, 1]
    mask = (sky_freq > 0.03) & (sky_freq < 0.08)
    ax.plot(sky_freq[mask], sky_fft_raw[mask], 'b-', lw=2, label=f'Raw (peak={sky_fft_raw[mask].max():.0f})')
    ax.plot(sky_freq[mask], sky_fft_filt[mask], 'r-', lw=2, label=f'Filtered (peak={sky_fft_filt[mask].max():.0f})')
    ax.axvline(1/17.5, color='gray', ls='--')
    reduction = 100 * (1 - sky_fft_filt[mask].max() / sky_fft_raw[mask].max())
    ax.text(0.04, 0.6, f"17.5px reduction: {reduction:.1f}%",
            transform=ax.transAxes, fontsize=11, color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax.set_xlabel("Frequency (px⁻¹)")
    ax.set_ylabel("|FFT|")
    ax.set_title("17.5px Peak Detail")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reflectance FFT
    _, ref_fft_raw = compute_column_fft(ref_raw)
    _, ref_fft_filt = compute_column_fft(ref_filt)
    ref_freq = np.fft.rfftfreq(ref_raw.shape[1])

    ax = axes[1, 0]
    ax.plot(ref_freq[:200], ref_fft_raw[:200], 'b-', lw=1.2, alpha=0.8, label='Raw')
    ax.plot(ref_freq[:200], ref_fft_filt[:200], 'r-', lw=1.2, alpha=0.8, label='Filtered')
    ax.axvline(1/17.5, color='gray', ls='--', lw=0.5)
    ax.axvline(1/8.8, color='gray', ls='--', lw=0.5)
    ax.set_xlim(0, 0.2)
    ax.set_xlabel("Frequency (px⁻¹)")
    ax.set_ylabel("|FFT|")
    ax.set_title("Reflectance Column FFT: Raw vs Filtered")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reflectance 17.5px detail
    ax = axes[1, 1]
    mask2 = (ref_freq > 0.03) & (ref_freq < 0.08)
    ax.plot(ref_freq[mask2], ref_fft_raw[mask2], 'b-', lw=2, label=f'Raw (peak={ref_fft_raw[mask2].max():.0f})')
    ax.plot(ref_freq[mask2], ref_fft_filt[mask2], 'r-', lw=2, label=f'Filtered (peak={ref_fft_filt[mask2].max():.0f})')
    ax.axvline(1/17.5, color='gray', ls='--')
    reduction_r = 100 * (1 - ref_fft_filt[mask2].max() / ref_fft_raw[mask2].max())
    ax.text(0.04, 0.6, f"17.5px reduction: {reduction_r:.1f}%",
            transform=ax.transAxes, fontsize=11, color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax.set_xlabel("Frequency (px⁻¹)")
    ax.set_ylabel("|FFT|")
    ax.set_title("Reflectance 17.5px Detail")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fft_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ FFT 对比图已保存")

    # ═══════════════════════════════════════════
    # 对比图 2: Score Map 对比
    # ═══════════════════════════════════════════
    print("\n🗺️  生成 Score Map 对比图...")

    # Crop center region for easier comparison
    H, W = sm_raw.shape
    yc, xc = H//2, W//2
    half = 400
    crop_raw = sm_raw[yc-half:yc+half, xc-half:xc+half]
    crop_filt = sm_filt_smooth[yc-half:yc+half, xc-half:xc+half]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    ax = axes[0, 0]
    vmax = np.percentile(sm_raw[sm_raw > 0], 95)
    im = ax.imshow(sm_raw, cmap='jet', vmin=0, vmax=vmax)
    ax.set_title("ACE Score Map (No Filter)")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    vmax2 = np.percentile(sm_filt_smooth[sm_filt_smooth > 0], 95)
    im = ax.imshow(sm_filt_smooth, cmap='jet', vmin=0, vmax=vmax2)
    ax.set_title("ACE Score Map (Filtered)")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    diff = sm_raw - sm_filt_smooth
    vmax_d = np.percentile(np.abs(diff), 95)
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
    ax.set_title("Difference (Raw - Filtered)")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Zoomed crop
    ax = axes[1, 0]
    im = ax.imshow(crop_raw, cmap='jet', vmin=0)
    ax.set_title(f"Crop Center ({half*2}x{half*2}) No Filter")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    im = ax.imshow(crop_filt, cmap='jet', vmin=0)
    ax.set_title("Crop Center Filtered")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Score histograms
    ax = axes[1, 2]
    ax.hist(scores_raw, bins=80, alpha=0.5, label=f'Raw (μ={scores_raw.mean():.4f}, σ={scores_raw.std():.4f})')
    ax.hist(scores_filt, bins=80, alpha=0.5, label=f'Filtered (μ={scores_filt.mean():.4f}, σ={scores_filt.std():.4f})')
    ax.axvline(thres, color='r', ls='--', label=f'threshold={thres}')
    ax.set_xlabel("ACE Score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.set_title("Score Distribution Comparison")

    plt.tight_layout()
    plt.savefig(output_dir / "score_map_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Score Map 对比图已保存")

    # ═══════════════════════════════════════════
    # 对比图 3: Detection comparison
    # ═══════════════════════════════════════════
    print("\n🎯 生成检测结果对比图...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    gray_norm = gray.astype(np.float32)
    lo, hi = np.percentile(gray_norm, 2), np.percentile(gray_norm, 98)
    gray_display = np.clip((gray_norm - lo) / (hi - lo + 1e-6), 0, 1)

    for idx, (sm, scores, label, is_filtered) in enumerate([
        (sm_raw, scores_raw, 'No Filter', False),
        (sm_filt_smooth, scores_filt, 'Full Filter', True),
    ]):
        row = idx
        binary = sm > thres

        # Overlay
        ax = axes[row, 0]
        ax.imshow(gray_display, cmap='gray')
        ov = np.zeros((*gray_display.shape, 4))
        ov[..., 0] = 1.0
        ov[..., 3] = binary.astype(float) * 0.6
        ax.imshow(ov)
        ax.set_title(f"Detection: {label}")
        ax.axis('off')

        # Score map
        ax = axes[row, 1]
        vmax_sm = np.percentile(sm[sm > 0], 95) if sm.max() > 0 else 0.2
        im = ax.imshow(sm, cmap='jet', vmin=0, vmax=vmax_sm)
        ax.set_title(f"Score Map: {label}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Info text
        n_noise = sm.shape[0] * sm.shape[1]
        n_bg = n_noise - int(binary.sum())
        bg_mean = sm[~binary].mean() if (~binary).sum() > 0 else 0
        bg_std = sm[~binary].std() if (~binary).sum() > 0 else 0
        print(f"  {label:15s}: 检测={int(binary.sum()):5d} px, "
              f"噪声均值={bg_mean:.6f}, 噪声标准差={bg_std:.6f}")

    plt.tight_layout()
    plt.savefig(output_dir / "detection_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 检测结果对比图已保存")

    # ═══════════════════════════════════════════
    # 生成综合报告
    # ═══════════════════════════════════════════
    print("\n📋 生成定量报告...")

    # Compute background noise metrics (non-detection region)
    binary_r = sm_raw > thres
    binary_f = sm_filt_smooth > thres

    bg_raw = sm_raw[~binary_r]
    bg_filt = sm_filt_smooth[~binary_f]

    # Compute noise in a region known to be background (corners)
    H, W = sm_raw.shape
    corner = sm_raw[0:H//4, 0:W//4]
    corner_filt = sm_filt_smooth[0:H//4, 0:W//4]
    # Also corners with the actual detection area
    all_detection = binary_r | binary_f
    true_bg_raw = sm_raw[~all_detection]
    true_bg_filt = sm_filt_smooth[~all_detection]
    score_noise_std_raw = true_bg_raw.std()
    score_noise_std_filt = true_bg_filt.std()

    report = f"""
{'='*65}
  水波纹滤波效果定量报告
{'='*65}

1. 频域滤除效果 (Sky 图像)
   - Sky 17.5px 幅度: {sky_fft_raw[mask].max():.0f} → {sky_fft_filt[mask].max():.0f}
   - 抑制率: {reduction:.1f}%

2. 频域滤除效果 (Reflectance 图像)
   - Reflectance 17.5px 幅度: {ref_fft_raw[mask2].max():.0f} → {ref_fft_filt[mask2].max():.0f}
   - 抑制率: {reduction_r:.1f}%

3. ACE 检测分数统计
   - 原始: 范围=[{scores_raw.min():.4f}, {scores_raw.max():.4f}],
            mean={scores_raw.mean():.6f}, std={scores_raw.std():.6f}
   - 滤波后: 范围=[{scores_filt.min():.4f}, {scores_filt.max():.4f}],
            mean={scores_filt.mean():.6f}, std={scores_filt.std():.6f}

4. Score Map 背景噪声
   - 原始背景标准差: {score_noise_std_raw:.6f}
   - 滤波后背景标准差: {score_noise_std_filt:.6f}
   - 噪声降低: {(1 - score_noise_std_filt / score_noise_std_raw) * 100:.1f}%

5. 检测结果
   - 原始检测像素: {int(binary_r.sum())} / {H*W}
   - 滤波后检测像素: {int(binary_f.sum())} / {H*W}

{'='*65}
  结论: {"✅ 滤波有效: 同心圆噪声显著抑制" if reduction > 80 else "⚠️ 滤波效果有限"}
{'='*65}
"""
    print(report)

    with open(output_dir / "report.txt", 'w') as f:
        f.write(report)
    print(f"  ✅ 报告已保存: {output_dir / 'report.txt'}")

    elapsed = time.time() - t0
    print(f"\n✅ 对比验证完成! 耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
