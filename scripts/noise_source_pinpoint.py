#!/usr/bin/env python3
"""
精确定位水波纹噪声根源。

上一步发现：
  - Sky residual 中有 17.5px 周期 (天空光不均匀)
  - 波段中心偏移 dX/dλ = -0.2864 px/nm (棱镜色散)
  - ACE score map 中波纹约 17px 周期

需要确认：
  1. 17.5px 噪声到底在 sky 中还是 spec 中还是 dark 中？
  2. 这个噪声在哪个波段最明显？
  3. 如何影响 ACE 检测结果？
  4. 不同检测算法是否都受影响？

用法:
  cd HI_refactored && python scripts/noise_source_pinpoint.py
"""

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
from scipy import ndimage as ndi
from scipy.signal import find_peaks, correlate2d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import subtract_dark_current, compute_reflectance

WAVELENGTHS = np.arange(445, 906, 5, dtype=int)
N_BANDS = 93

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hi_dir = os.path.join(base, "..", "HI")
data_dir = os.path.join(base, "..", "data", "1")
output_dir = Path(os.path.join(base, "output", "ace_samples"))

# Load data
print("📂 加载数据...")
spec = tifffile.imread(os.path.join(data_dir, "5ms.tif"))
dark = tifffile.imread(os.path.join(data_dir, "P11070000.tif"))
sky = tifffile.imread(os.path.join(data_dir, "5ms_sky.tif"))
gray = tifffile.imread(os.path.join(data_dir, "view2.tif"))
print(f"  Spec: {spec.shape}, Dark: {dark.shape}, Sky: {sky.shape}")

with open(os.path.join(hi_dir, "coords_dict.json")) as f:
    coords_dict = json.load(f)

# Collect x positions for each band
print("\n📐 分析各波段 X 位置分布...")
band_x_positions = {}
for idx_str, spec_list in coords_dict.items():
    if len(spec_list) == N_BANDS:
        for b in range(N_BANDS):
            y, x = spec_list[b][1], spec_list[b][2]
            if b not in band_x_positions:
                band_x_positions[b] = []
            band_x_positions[b].append(x)

band_x_mean = {}
for b in range(N_BANDS):
    band_x_mean[b] = np.mean(band_x_positions[b])

# Show X vs wavelength
print("\n📈 X 位置 vs 波长:")
wls = np.arange(445, 906, 5)
x_means = np.array([band_x_mean[b] for b in range(N_BANDS)])

# Linear fit
coeff_x = np.polyfit(wls, x_means, 1)
print(f"  X_mean(λ) = {coeff_x[0]:.4f} × λ + {coeff_x[1]:.1f}")
print(f"  dX/dλ = {coeff_x[0]:.4f} px/nm")
print(f"  全谱漂移: {coeff_x[0] * 460:.1f} px")

# ========= 核心分析: 17.5px 源自何处？ =========
print("\n" + "="*65)
print("🔬 17.5px 周期溯源")
print("="*65)

# For EACH raw image (spec, dark, sky, spec_dark, ref), extract
# the pixel values at the positions used for each band, then look
# for the 17.5px period in the column direction.

# First check: what's the average X range for a few bands?
print("\n  各波段 X 范围 (用于定位数据区域):")
for b in [0, 15, 30, 45, 60, 78, 85]:
    xs = band_x_positions[b]
    print(f"    Band {b:2d} ({WAVELENGTHS[b]}nm): "
          f"X=[{min(xs)},{max(xs)}] Δ={max(xs)-min(xs)}")

# The period of the noise in the ACE score map is ~17px
# Let's look at what happens at each step:
# 1. Raw sky image crop at the region used by each band
# 2. Raw spec image crop
# 3. Dark-subtracted
# 4. Reflectance

# Pick the region where most bands' data falls
all_x = np.concatenate(list(band_x_positions.values()))
print(f"\n  全局 X 范围: [{all_x.min()}, {all_x.max()}]")

# For a more targeted test: look at a specific Y row across the full X range
# and check for periodicity in spec, dark, sky, and reflectance
H, W = spec.shape
test_y = H // 2  # Middle row

# But different bands have data at different X positions...
# Better approach: for each band, extract the values at ALL its calibration
# points, then compute the autocorrelation along X.

# Let's test a more focused region - for each of a few bands, crop the
# image region where its data falls, and check FFT

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
test_bands = [0, 15, 30, 45, 60, 78]  # 445nm to 835nm

for idx, b in enumerate(test_bands[:4]):  # First 4 bands for columns
    xs = np.array(band_x_positions[b])
    ys = np.array([coords_dict[k][b][1] for k in list(coords_dict.keys())[:len(band_x_positions[b])]])

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # Crop region from each image
    for row, (img, label) in enumerate([
        (spec, "Raw Spec"),
        (dark, "Dark"),
        (sky, "Sky"),
        (spec.astype(float) - dark.astype(float), "Spec - Dark"),
    ]):
        crop = img[y0:y1, x0:x1]

        ax = axes[row, idx]
        # Show FFT of column means to find periodicity
        col_mean = crop.mean(axis=0)
        if len(col_mean) > 5:
            fft = np.abs(np.fft.rfft(col_mean - col_mean.mean()))
            freqs = np.fft.rfftfreq(len(col_mean))

            # Look in the 0.03-0.1 px⁻¹ range (10-33px periods)
            mask = (freqs > 0.01) & (freqs < 0.2)
            if mask.any():
                ax.plot(freqs[mask], fft[mask])
                peaks, props = find_peaks(fft[mask], height=fft[mask].mean()*2)
                for p in peaks:
                    actual_p = p + np.where(mask)[0][0]
                    period = 1.0 / freqs[actual_p] if freqs[actual_p] > 0 else 0
                    ax.axvline(freqs[actual_p], color='r', ls='--', alpha=0.5)
                    print(f"    {label}, Band {b}: period ≈ {period:.1f}px @ freq={freqs[actual_p]:.4f}")

        ax.set_xlabel("Freq (px⁻¹)")
        ax.set_ylabel("|FFT|")
        ax.set_title(f"{label} Band {b} ({WAVELENGTHS[b]}nm)")

plt.tight_layout()
plt.savefig(output_dir / "noise_source_by_band.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 多图像多波段周期分析图已保存")


# ========= 更精确的分析: 用所有标定点提取的 X 列 FFT =========
print("\n" + "="*65)
print("📊 精确 17.5px 周期检测 — 用所有标定点数据")
print("="*65)

# Extract for each raw image: the pixel values at each calibration point's
# position for each band, and look for periodicity in the X direction

def extract_band_slice(img, coords_dict, band_idx):
    """Extract pixel values at the positions used by a given band."""
    vals = []
    xs = []
    for idx_str, spec_list in coords_dict.items():
        if len(spec_list) > band_idx:
            y, x = spec_list[band_idx][1], spec_list[band_idx][2]
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                vals.append(img[y, x])
                xs.append(x)
    return np.array(vals), np.array(xs)


def find_period_x_img(img, label=""):
    """For a given 2D image, extract per-band slices and check each for 17.5px period."""
    periods_found = []
    for b in range(0, 78, 3):  # Every 3rd band
        vals, xs = extract_band_slice(img, coords_dict, b)
        if len(vals) < 20:
            continue

        # Group by Y and check each row
        y_vals = []
        for idx_str, spec_list in coords_dict.items():
            if len(spec_list) > b:
                y_vals.append(spec_list[b][1])

        y_vals = np.array(y_vals[:len(vals)])

        # For each unique Y, collect the X values and look for periodicity
        unique_ys = np.sort(np.unique(y_vals))
        periods_row = []
        for y_test in unique_ys[::3]:  # Every 3rd row
            mask = y_vals == y_test
            if mask.sum() < 5:
                continue
            x_row = xs[mask]
            v_row = vals[mask]
            order = np.argsort(x_row)
            x_sorted = x_row[order]
            v_sorted = v_row[order]

            if len(v_sorted) > 10:
                dx = np.median(np.diff(x_sorted))
                if dx == 0:
                    continue
                fft = np.abs(np.fft.rfft(v_sorted - v_sorted.mean()))
                freqs = np.fft.rfftfreq(len(v_sorted), d=dx)
                # Focus on 5-50px range
                mask_f = (freqs > 0.02) & (freqs < 0.2)
                if mask_f.sum() > 0:
                    peaks, props = find_peaks(fft[mask_f], height=fft[mask_f].mean()*2)
                    for p in peaks:
                        actual_p = p + np.where(mask_f)[0][0]
                        if freqs[actual_p] > 0:
                            period = 1.0 / freqs[actual_p]
                            if 5 < period < 200:
                                periods_row.append(period)

        if periods_row:
            median_p = np.median(periods_row)
            mean_p = np.mean(periods_row)
            print(f"    {label} Band {b:2d} ({WAVELENGTHS[b]}nm): "
                  f"period ≈ {median_p:.1f}px (mean={mean_p:.1f}, n={len(periods_row)} rows)")
            periods_found.extend(periods_row)

    return periods_found


print("\n  检查 Raw Spec 中的周期...")
p_spec = find_period_x_img(spec, "Raw Spec")

print("\n  检查 Dark 中的周期...")
p_dark = find_period_x_img(dark, "Dark")

print("\n  检查 Sky 中的周期...")
p_sky = find_period_x_img(sky, "Sky")

spec_ds = (spec.astype(float) - dark.astype(float))
print("\n  检查 Spec-Dark 中的周期...")
p_spec_ds = find_period_x_img(spec_ds, "Spec-Dark")

print("\n  检查 Reflectance 中的周期...")
ref = compute_reflectance(spec_ds, subtract_dark_current(sky, dark))
p_ref = find_period_x_img(ref, "Reflectance")

print("\n" + "="*65)
print("📋 周期统计汇总")
print("="*65)

for name, periods in [("Raw Spec", p_spec), ("Dark", p_dark), ("Sky", p_sky),
                       ("Spec-Dark", p_spec_ds), ("Reflectance", p_ref)]:
    if len(periods) > 0:
        arr = np.array(periods)
        print(f"  {name:15s}: N={len(periods):5d}, "
              f"median={np.median(arr):.1f}±{np.std(arr):.1f}px, "
              f"range=[{arr.min():.1f},{arr.max():.1f}]")
    else:
        print(f"  {name:15s}: 未检测到明显周期")


# ========= 检查不同检测算法受噪声影响程度 =========
print("\n" + "="*65)
print("🎯 各算法噪声敏感性对比")
print("="*65)

# Load ACE score maps from different n_samples to see if ripple changes
# Also look at CEM, SAM, SACE to see if they have the same ripple
pipeline_dir = os.path.join(base, "output", "real_pipeline")

# Check if any score .npy files exist
score_files = list(Path(pipeline_dir).glob("*.npy"))
for sf in score_files:
    data_scores = np.load(sf)
    if data_scores.ndim == 2:
        H, W = data_scores.shape
        print(f"  {sf.name}: {H}×{W} score map")
        # FFT analysis
        center = data_scores[data_scores.shape[0]//4:3*data_scores.shape[0]//4,
                             data_scores.shape[1]//4:3*data_scores.shape[1]//4]
        fft = np.abs(np.fft.fftshift(np.fft.fft2(center)))
        # Radial profile
        cy, cx = fft.shape[0]//2, fft.shape[1]//2
        yy, xx = np.ogrid[:fft.shape[0], :fft.shape[1]]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2).astype(int)
        r_flat = r.ravel()
        fft_flat = fft.ravel()
        r_max = min(200, cy, cx)
        radial = np.array([fft_flat[r_flat == ri].mean() for ri in range(r_max) if (r_flat == ri).any()])
        peaks, _ = find_peaks(radial, height=radial[:10].mean()*2, distance=5)
        for p in peaks:
            if p > 2:
                print(f"    径向峰值 @ r={p} → 周期 ≈ {1/p*50:.1f}px" if p > 0 else "")
    else:
        print(f"  {sf.name}: 1D scores ({data_scores.shape})")

print(f"\n{'='*65}")
print(f"✅ 分析完成!")
print(f"{'='*65}")
