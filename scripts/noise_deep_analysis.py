#!/usr/bin/env python3
"""
深度学习分析：ACE 分数图中的水波纹/同心圆噪声。

目标：
  1. 确认噪声是否存在于原始光谱图像 (dark / spec / sky) 还是仅出现在反射率/ACE 结果中
  2. 精确定位不同波段的波纹中心，跟踪中心随波长的漂移
  3. 验证 etalon 效应：空间频率 ∝ 1/λ ？
  4. 验证色散棱镜 + sensor pattern 假说：中心随 λ 线性漂移？
  5. 验证 Moiré / 稀疏采样假说
  6. 给出物理来源判断

用法:
  cd HI_refactored && python scripts/noise_deep_analysis.py
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
from scipy import ndimage as ndi
from scipy.signal import find_peaks, correlate2d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import subtract_dark_current, compute_reflectance

WAVELENGTHS = np.arange(445, 906, 5, dtype=int)
N_BANDS = 93


def load_data(data_dir, hi_dir):
    """Load all TIF images."""
    paths = {
        "spec_base": os.path.join(data_dir, "5ms.tif"),
        "dark": os.path.join(data_dir, "P11070000.tif"),
        "illuminance": os.path.join(data_dir, "5ms_sky.tif"),
        "gray": os.path.join(data_dir, "view2.tif"),
    }
    images = {}
    for name, p in paths.items():
        if not os.path.exists(p):
            print(f"ERROR: {p} not found")
            sys.exit(1)
        images[name] = tifffile.imread(p)
        print(f"  {name}: {images[name].shape}, dtype={images[name].dtype}")
    return images


def compute_dark_stats(dark):
    """Analyze dark current image for fixed pattern noise."""
    print(f"\n{'='*60}")
    print("🔍 [1] 暗场图像 (Dark) 固定模式噪声分析")
    print(f"{'='*60}")
    print(f"  Shape: {dark.shape}, dtype={dark.dtype}")
    print(f"  Range: [{dark.min()}, {dark.max()}]")
    print(f"  Mean±Std: {dark.mean():.2f} ± {dark.std():.2f}")

    # Row/column profiles
    col_mean = dark.mean(axis=0)
    row_mean = dark.mean(axis=1)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(dark, cmap='gray', vmax=np.percentile(dark, 99))
    plt.title(f"Dark Image\nμ={dark.mean():.1f} σ={dark.std():.1f}")
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.plot(col_mean)
    plt.title("Column Mean Profile")
    plt.xlabel("Column"); plt.ylabel("Mean DN")

    plt.subplot(2, 3, 3)
    plt.plot(row_mean)
    plt.title("Row Mean Profile")
    plt.xlabel("Row"); plt.ylabel("Mean DN")

    # FFT of column means to look for periodicity
    if len(col_mean) > 0:
        fft_col = np.abs(np.fft.rfft(col_mean - col_mean.mean()))
        freq_col = np.fft.rfftfreq(len(col_mean))
        plt.subplot(2, 3, 4)
        plt.plot(freq_col[:500], fft_col[:500])
        plt.xlabel("Spatial Frequency (1/px)"); plt.ylabel("|FFT|")
        plt.title("FFT of Column Profile")

        # Find peaks
        peaks, props = find_peaks(fft_col[:500], height=fft_col[:500].mean()*3)
        for p in peaks:
            period = 1.0 / freq_col[p] if freq_col[p] > 0 else np.inf
            print(f"    峰 @ freq={freq_col[p]:.4f} → 周期 ≈ {period:.1f} px, 幅度={fft_col[p]:.0f}")
        if len(peaks) == 0:
            print("  无显著周期成分")

    # 2D FFT of dark center crop
    H, W = dark.shape
    crop = dark[H//4:3*H//4, W//4:3*W//4]
    fft2d = np.fft.fftshift(np.log10(np.abs(np.fft.fft2(crop)) + 1))
    plt.subplot(2, 3, 5)
    plt.imshow(fft2d, cmap='inferno', extent=[-0.5, 0.5, -0.5, 0.5])
    plt.title("2D FFT (log) of Dark Center")
    plt.xlabel("kx"); plt.ylabel("ky")

    plt.subplot(2, 3, 6)
    plt.axis('off')
    info = (f"Dark noise analysis\n"
            f"Sensor fixed pattern\n"
            f"Period search: column FFT\n"
            f"2D FFT for directional noise")
    plt.text(0.1, 0.9, info, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace")
    plt.tight_layout()
    plt.savefig(output_dir / "noise_dark_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ dark分析图已保存")


def analyze_band_crops(images, output_dir, band_indices=None):
    """
    [2] 提取多个波段的光谱原始数据区域，直接观察波纹。
    对于每个波段，从 spec、sky、dark 各自提取 crop，计算 reflectance。
    """
    print(f"\n{'='*60}")
    print("🔍 [2] 多波段原始图像波纹直接观察")
    print(f"{'='*60}")

    if band_indices is None:
        # Pick ~10 bands evenly spaced plus saturated region
        band_indices = list(range(0, 60, 6)) + list(range(60, N_BANDS, 3))
        band_indices = sorted(set(band_indices))

    spec = images["spec_base"]
    dark = images["dark"]
    sky = images["illuminance"]
    gray = images["gray"]

    # Dark-subtracted
    spec_ds = subtract_dark_current(spec, dark)
    sky_ds = subtract_dark_current(sky, dark)
    # Reflectance
    ref = compute_reflectance(spec_ds, sky_ds)

    print(f"  Spectral (dark-sub): {spec_ds.shape}")
    print(f"  Reflectance: {ref.shape}")

    # The spectral images are 2D: prism disperses different wavelengths
    # to different spatial columns. coords_dict maps (band, y, x).
    # But the spec image is still 2048×2048 — where is each band?
    # The key insight: the coords_dict tells us that band 0 → spec[s[0][1], s[0][2]],
    # band 41 → spec[s[41][1], s[41][2]].
    # But these are at DIFFERENT pixel positions within the 2048×2048 image.
    # Different bands are at different spatial positions on the sensor!

    # So let's look at the actual positions used for each band across all calibration points.
    with open(os.path.join(hi_dir, "coords_dict.json")) as f:
        coords_dict = json.load(f)

    # Collect Y, X positions used for each band across all points
    band_positions = {b: [] for b in band_indices}
    for idx_str, spec_list in coords_dict.items():
        if len(spec_list) >= max(band_indices) + 1:
            for b in band_indices:
                y, x = spec_list[b][1], spec_list[b][2]
                band_positions[b].append((y, x))

    # Print position statistics per band
    print(f"\n  各波段在 sensor 上的采样位置范围:")
    for b in band_indices[:6]:  # First few
        positions = np.array(band_positions[b])
        if len(positions) > 0:
            y0, y1 = positions[:, 0].min(), positions[:, 0].max()
            x0, x1 = positions[:, 1].min(), positions[:, 1].max()
            print(f"    band {b:2d} ({WAVELENGTHS[b]}nm): Y=[{y0},{y1}] X=[{x0},{x1}], N={len(positions)}")
    print(f"    ... 共 {len(band_indices)} 个波段")

    # For a subset of bands, show the raw spec region and its FFT
    # Pick 4 bands at different wavelengths
    show_bands = [band_indices[len(band_indices)//5*i] for i in range(5)]
    # Make sure they're distinct
    show_bands = sorted(set(show_bands))[:6]

    fig, axes = plt.subplots(5, 4, figsize=(20, 22))

    for row, b in enumerate(show_bands[:5]):
        positions = np.array(band_positions[b])
        if len(positions) == 0:
            continue

        y_c = int(positions[:, 0].mean())
        x_c = int(positions[:, 1].mean())
        half = 200  # crop half-size

        y1 = max(0, y_c - half)
        y2 = min(spec.shape[0], y_c + half)
        x1 = max(0, x_c - half)
        x2 = min(spec.shape[1], x_c + half)

        # Raw spec crop
        spec_crop = spec[y1:y2, x1:x2]
        # Dark crop (same region)
        dark_crop = dark[y1:y2, x1:x2]
        # Dark-subtracted spec
        spec_ds_crop = spec_ds[y1:y2, x1:x2]
        # Reflectance crop
        sky_crop_ds = sky_ds[y1:y2, x1:x2]
        ref_region = np.divide(spec_ds_crop, sky_crop_ds,
                               out=np.zeros_like(spec_ds_crop, dtype=float),
                               where=sky_crop_ds > 0)

        # Show raw spec, dark-sub, ref, and FFT of ref
        ax = axes[row, 0]
        im = ax.imshow(spec_crop, cmap='gray')
        ax.set_title(f"Band {b} ({WAVELENGTHS[b]}nm)\nRaw Spec")
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[row, 1]
        im = ax.imshow(dark_crop, cmap='gray')
        ax.set_title(f"Dark (same region)")
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[row, 2]
        im = ax.imshow(spec_ds_crop, cmap='gray')
        ax.set_title(f"Spec - Dark")
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[row, 3]
        # Show FFT of ref crop to look for periodic patterns
        ref_crop = ref_region
        fft = np.fft.fftshift(np.log10(np.abs(np.fft.fft2(ref_crop)) + 1))
        im = ax.imshow(fft, cmap='inferno',
                       extent=[-1, 1, -1, 1])
        ax.set_title(f"Reflectance FFT (log)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    axes[0, 0].text(-0.3, 0.5, "Raw\nSpec", transform=axes[0, 0].transAxes,
                    fontsize=12, va='center', ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / "noise_band_crops.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 多波段原始切片图已保存")

    return spec_ds, sky_ds, ref, coords_dict, band_positions


def analyze_ripple_center_shift(ref, coords_dict, output_dir):
    """
    [3] 精确测量不同波段的波纹中心位置，跟踪中心随波长的漂移。

    方法：
    - 对于每个波段，提取该波段在所有标定点位置的 spec_ds 值
    - 构建该波段的稀疏"图像"
    - 用互相关或自相关找波纹周期
    - 用径向轮廓找到中心
    """
    print(f"\n{'='*60}")
    print("🔍 [3] 波纹中心随波长漂移分析")
    print(f"{'='*60}")

    valid_items = [(idx_str, spec) for idx_str, spec in coords_dict.items()
                   if len(spec) == N_BANDS]
    n_points = len(valid_items)
    print(f"  标定点: {n_points}")

    # Extract first band positions (spatial reference)
    first_positions = np.zeros((n_points, 2), dtype=int)
    for i, (_, spec) in enumerate(valid_items):
        first_positions[i] = [spec[0][1], spec[0][2]]  # y, x from band 0

    # Create a sparse image - find which points are near each other
    # to detect the ripple wave front

    # Approach: pick a central band and build a dense-enough region
    # to see the ripple, then correlate with neighboring bands

    def build_sparse_map(reflectance, band_idx, positions, shape=(2048, 2048)):
        """Build a map using nearest-neighbor interpolation on sparse data."""
        # Extract band values
        vals = np.zeros(n_points, dtype=np.float64)
        valid_mask = np.ones(n_points, dtype=bool)
        for i, (_, spec) in enumerate(valid_items):
            s = spec[band_idx]
            y, x = s[1], s[2]
            if 0 <= y < reflectance.shape[0] and 0 <= x < reflectance.shape[1]:
                vals[i] = reflectance[y, x]
            else:
                valid_mask[i] = False
        return vals, positions[valid_mask]

    # Analyze across all bands - look at the "center" of each band's data
    # by finding the centroid of the brightest region
    print(f"\n  计算各波段数据质心...")

    centers_wl = []
    for b in range(N_BANDS):
        vals, pos = build_sparse_map(ref, b, first_positions)
        if len(vals) == 0:
            continue

        # Normalize vals to 0-1 for centroid calculation
        vn = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)

        # Weighted centroid
        if vn.sum() > 0:
            cy = (pos[:, 0] * vn).sum() / vn.sum()
            cx = (pos[:, 1] * vn).sum() / vn.sum()
        else:
            cy, cx = pos.mean(axis=0)

        centers_wl.append({
            'band': b,
            'wavelength': WAVELENGTHS[b],
            'cx': cx,
            'cy': cy,
        })

    centers_wl = np.array([(c['band'], c['wavelength'], c['cx'], c['cy'])
                           for c in centers_wl])

    print(f"  完成 {len(centers_wl)} 个波段质心计算")
    print(f"  X 中心范围: [{centers_wl[:, 2].min():.1f}, {centers_wl[:, 2].max():.1f}]")
    print(f"  Y 中心范围: [{centers_wl[:, 3].min():.1f}, {centers_wl[:, 3].max():.1f}]")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.plot(centers_wl[:, 1], centers_wl[:, 2], 'o-', ms=3, lw=1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Centroid X")
    ax.set_title("X centroid vs Wavelength")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(centers_wl[:, 1], centers_wl[:, 3], 'o-', ms=3, lw=1, color='orange')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Centroid Y")
    ax.set_title("Y centroid vs Wavelength")
    ax.grid(True, alpha=0.3)

    # X centroid vs wavelength linear fit
    if len(centers_wl) > 5:
        from numpy.polynomial.polynomial import polyfit
        wl = centers_wl[:, 1]
        cx = centers_wl[:, 2]
        coeff_x = np.polyfit(wl, cx, 1)
        cy = centers_wl[:, 3]
        coeff_y = np.polyfit(wl, cy, 1)

        ax = axes[2]
        ax.plot(wl, cx, 'o', ms=3, label='X')
        ax.plot(wl, np.polyval(coeff_x, wl), '-', label=f'X fit: {coeff_x[0]:.4f}·λ + {coeff_x[1]:.1f}')
        ax.plot(wl, cy, 'o', ms=3, label='Y', color='orange')
        ax.plot(wl, np.polyval(coeff_y, wl), '-', color='orange',
                label=f'Y fit: {coeff_y[0]:.4f}·λ + {coeff_y[1]:.1f}')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Centroid position")
        ax.legend()
        ax.set_title(f"Linear fit\n"
                     f"dX/dλ = {coeff_x[0]:.4f} px/nm, dY/dλ = {coeff_y[0]:.4f} px/nm")
        ax.grid(True, alpha=0.3)

        # Check if centroid drift matches prism dispersion
        # A typical prism might disperse ~0.01-0.1 px/nm
        print(f"\n  质心漂移率:")
        print(f"    dX/dλ = {coeff_x[0]:.4f} px/nm")
        print(f"    dY/dλ = {coeff_y[0]:.4f} px/nm")
        print(f"    跨 93 波段总漂移: X={coeff_x[0]*460:.1f} px, Y={coeff_y[0]*460:.1f} px")

    plt.tight_layout()
    plt.savefig(output_dir / "noise_center_vs_wavelength.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 波纹中心漂移图已保存")

    return centers_wl


def analyze_self_correlation(ref, coords_dict, output_dir):
    """
    [4] 对每个波段的稀疏数据进行自相关分析，精确测量波纹周期。

    方法：
    - 对选定波段构建网格化插值图
    - 计算 2D 自相关
    - 测量径向周期
    - 检查周期 vs 波长关系 (etalon 检验)
    """
    print(f"\n{'='*60}")
    print("🔍 [4] 波纹周期 vs 波长 (Etalon 检验)")
    print(f"{'='*60}")

    valid_items = [(idx_str, spec) for idx_str, spec in coords_dict.items()
                   if len(spec) == N_BANDS]
    n_points = len(valid_items)

    first_positions = np.zeros((n_points, 2), dtype=int)
    for i, (_, spec) in enumerate(valid_items):
        first_positions[i] = [spec[0][1], spec[0][2]]

    # Select bands for dense analysis
    wavelengths = WAVELENGTHS

    # For each band, build a "dense" image using the region with most points
    # and compute radial profile to find period

    results = []
    for band_idx in list(range(0, N_BANDS, 3)):  # Every 3rd band
        # Get values
        vals = np.zeros(n_points, dtype=np.float64)
        valid = np.ones(n_points, dtype=bool)
        for i, (_, spec) in enumerate(valid_items):
            s = spec[band_idx]
            y, x = s[1], s[2]
            if 0 <= y < ref.shape[0] and 0 <= x < ref.shape[1]:
                vals[i] = ref[y, x]
            else:
                valid[i] = False

        vals = vals[valid]
        pos = first_positions[valid]

        if len(vals) < 50:
            continue

        # Grid the values in a cropped region
        y0, y1 = pos[:, 0].min(), pos[:, 0].max()
        x0, x1 = pos[:, 1].min(), pos[:, 1].max()

        # If the data is too sparse (most pixels missing), skip period measurement
        # Instead, use the raw vals to look for spatial periodicity along rows/cols

        # Sort by X position and look at column profiles
        order_x = np.argsort(pos[:, 1])
        pos_sorted = pos[order_x]
        vals_sorted = vals[order_x]

        # Sort by Y position
        order_y = np.argsort(pos[:, 0])
        pos_sorted_y = pos[order_y]
        vals_sorted_y = vals[order_y]

        # For etalon test, compute FFT of values along X and find dominant period
        # but only where we have a continuous run of points in X
        unique_x = np.sort(np.unique(pos[:, 1]))
        if len(unique_x) < 10:
            continue

        # Try to find period by looking at pairs of points with similar Y
        # and measuring the correlation as a function of dx
        periods_x = []

        # More robust: take all points, compute pairwise distances vs value correlation
        # Simplified: look at pairs with dy≈0
        y_values = np.sort(np.unique(pos[:, 0]))
        for y_test in y_values[::5]:  # Every 5th Y row
            mask = pos[:, 0] == y_test
            if mask.sum() < 5:
                continue
            x_pts = pos[mask, 1]
            v_pts = vals[mask]
            # Sort by X
            order = np.argsort(x_pts)
            x_pts = x_pts[order]
            v_pts = v_pts[order]

            # FFT along this row
            if len(v_pts) > 10:
                fft = np.abs(np.fft.rfft(v_pts - v_pts.mean()))
                freqs = np.fft.rfftfreq(len(v_pts), d=np.median(np.diff(x_pts)))
                peaks, props = find_peaks(fft, height=fft.mean()*2)
                for p in peaks:
                    if freqs[p] > 0.001:  # > 0.001 px⁻¹ → period < 1000 px
                        period = 1.0 / freqs[p]
                        periods_x.append(period)

        if len(periods_x) > 0:
            median_period = np.median(periods_x)
            results.append({
                'band': band_idx,
                'wavelength': wavelengths[band_idx],
                'period': median_period,
                'period_std': np.std(periods_x),
                'n_rows': len(periods_x),
            })
            print(f"  Band {band_idx:2d} ({wavelengths[band_idx]}nm): "
                  f"period ≈ {median_period:.1f}±{np.std(periods_x):.1f}px "
                  f"(from {len(periods_x)} rows)")

    if len(results) > 1:
        results.sort(key=lambda r: r['wavelength'])
        periods = np.array([r['period'] for r in results])
        wls = np.array([r['wavelength'] for r in results])
        errs = np.array([r['period_std'] for r in results])

        print(f"\n  周期范围: [{periods.min():.1f}, {periods.max():.1f}] px")
        print(f"  平均周期: {periods.mean():.1f} ± {periods.std():.1f} px")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.errorbar(wls, periods, yerr=errs, fmt='o-', capsize=3)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Spatial Period (px)")
        ax.set_title("Ripple Period vs Wavelength")
        ax.grid(True, alpha=0.3)

        # Etalon test: if thin-film interference, spatial frequency ∝ 1/λ
        # frequency = 1/period
        freqs = 1.0 / periods
        ax = axes[0, 1]
        ax.plot(wls, freqs, 'o-')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Spatial Frequency (px⁻¹)")
        ax.set_title("Spatial Frequency vs Wavelength")
        ax.grid(True, alpha=0.3)

        # Etalon: frequency ∝ 1/λ → plot frequency vs 1/λ
        inv_wl = 1.0 / wls
        ax = axes[1, 0]
        ax.plot(inv_wl, freqs, 'o-')
        coeff = np.polyfit(inv_wl, freqs, 1)
        ax.plot(inv_wl, np.polyval(coeff, inv_wl), '--',
                label=f'Linear fit: freq = {coeff[0]:.2f}·1/λ + {coeff[1]:.6f}')
        ax.set_xlabel("1/λ (1/nm)")
        ax.set_ylabel("Spatial Frequency (px⁻¹)")
        ax.set_title("Etalon Test: linear freq ∝ 1/λ?")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # If etalon: period * wavelength = constant
        pw = periods * wls
        ax = axes[1, 1]
        ax.plot(wls, pw, 'o-')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Period × λ (px·nm)")
        ax.set_title(f"Etalon Test: constant = {pw.mean():.0f}±{pw.std():.0f}?\n"
                     f"CV = {pw.std()/pw.mean()*100:.1f}%")
        ax.grid(True, alpha=0.3)
        print(f"  Period × λ: {pw.mean():.0f} ± {pw.std():.0f} px·nm "
              f"(CV = {pw.std()/pw.mean()*100:.1f}%)")
        print(f"  Etalon判定: {'✅ 可能是 etalon (CV < 10%)' if pw.std()/pw.mean()*100 < 10 else '❌ 不是 etalon 或周期测量误差大'}")
        print(f"  Frequency vs 1/λ: slope={coeff[0]:.2f}, R²估算可用")

        plt.tight_layout()
        plt.savefig(output_dir / "noise_etalon_test.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Etalon 检验图已保存")

    else:
        print("  ⚠️ 数据太少，无法进行 etalon 分析")

    return results


def analyze_ace_score_map_ripple(output_dir):
    """
    [5] 直接在 ACE score map 上分析波纹特征。

    查看不同 n_samples 的 ACE score map，确认波纹是否稳定存在。
    """
    print(f"\n{'='*60}")
    print("🔍 [5] ACE Score Map 波纹分析")
    print(f"{'='*60}")

    ace_base = str(output_base)
    ace_dirs = [
        ("ace_samples_1", "ACE N=1"),
        ("ace_samples_3", "ACE N=3"),
        ("ace_samples_5", "ACE N=5"),
        ("ace_samples_all", "ACE N=all"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (dname, label) in enumerate(ace_dirs):
        score_path = os.path.join(ace_base, dname, f"score_ACE_N{dname.split('_')[-1]}.npy")
        # Check for .npz or other formats
        score_data = None
        for ext in ['.npy', '.npz']:
            p = os.path.join(ace_base, dname, f"score_ACE_N{dname.split('_')[-1]}{ext}")
            if os.path.exists(p):
                score_data = p
                break

        ax = axes[idx // 2, idx % 2]

        npz_path = os.path.join(ace_base, dname, "summary_ACE_Nall.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            if 'score_map' in data:
                sm = data['score_map']
                im = ax.imshow(sm, cmap='jet', vmax=np.percentile(sm[sm > 0], 95))
                ax.set_title(f"{label}\nScore Map (from .npz)")
                plt.colorbar(im, ax=ax, fraction=0.046)
                continue

        # Try to load the score map from the summary png? No — use .npy if exists
        npy_path = os.path.join(ace_base, dname, "ace_scores.npy")
        npy_path2 = os.path.join(ace_base, dname, "scores_ACE.npy")
        found = False
        for np_path in [npy_path, npy_path2, score_data] if isinstance(score_data, str) else [npy_path, npy_path2]:
            if np_path and os.path.exists(np_path):
                sm = np.load(np_path)
                if sm.ndim == 1:
                    ax.text(0.5, 0.5, f"1D scores ({len(sm)} pts)\nneed score_map",
                            transform=ax.transAxes, ha='center', va='center')
                else:
                    im = ax.imshow(sm, cmap='jet')
                    ax.set_title(f"{label}\nScore Map")
                    plt.colorbar(im, ax=ax, fraction=0.046)
                found = True
                break

        if not found:
            ax.text(0.5, 0.5, "Score map not found\nin output directory",
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f"{label}")

    plt.tight_layout()
    plt.savefig(output_dir / "noise_ace_score_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ ACE score map 对比图已保存")


def analyze_sky_illumination(images, output_dir):
    """
    [6] 分析 illumination (sky) 图像中是否存在波纹。
    如果 sky 中有波纹，说明是光照不均匀，在反射率除法后会被放大。
    """
    print(f"\n{'='*60}")
    print("🔍 [6] 天空光 (Sky) 均匀性分析")
    print(f"{'='*60}")

    sky = images["illuminance"]
    dark = images["dark"]
    gray = images["gray"]

    print(f"  Sky shape: {sky.shape}, dtype={sky.dtype}")
    print(f"  Sky range: [{sky.min()}, {sky.max()}]")
    print(f"  Sky mean±std: {sky.mean():.1f} ± {sky.std():.1f}")
    print(f"  Sky SNR: {sky.mean()/sky.std():.1f}")

    # The sky is a single 2D image (like dark), not multi-band
    # The prism disperses spectral info across the sensor, but sky/dark are
    # single-frame captures (not dispersed)

    # Analyze sky vs dark
    sky_ds = sky.astype(float) - dark.astype(float)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    ax = axes[0, 0]
    im = ax.imshow(sky, cmap='gray', vmax=np.percentile(sky, 99))
    ax.set_title(f"Sky Image\nμ={sky.mean():.1f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(dark, cmap='gray')
    ax.set_title(f"Dark Image\nμ={dark.mean():.1f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    im = ax.imshow(sky_ds, cmap='gray')
    ax.set_title(f"Sky - Dark\nμ={sky_ds.mean():.1f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Sky flatness: fit a 2D polynomial and look at residual
    H, W = sky.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # 2D quadratic fit to sky
    A = np.column_stack([np.ones(H*W), xx.ravel(), yy.ravel(),
                         xx.ravel()**2, yy.ravel()**2, xx.ravel()*yy.ravel()])
    b = sky.ravel().astype(float)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        sky_fit = (A @ coeffs).reshape(H, W)
        sky_residual = sky - sky_fit

        ax = axes[0, 3]
        im = ax.imshow(sky_fit, cmap='viridis')
        ax.set_title(f"Sky 2D Quadratic Fit")
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, 0]
        im = ax.imshow(sky_residual, cmap='RdBu_r',
                       vmax=np.percentile(np.abs(sky_residual), 95))
        ax.set_title(f"Sky Residual\nσ={sky_residual.std():.1f} ({sky_residual.std()/sky.mean()*100:.2f}%)")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # FFT of sky residual
        crop = sky_residual[H//4:3*H//4, W//4:3*W//4]
        fft = np.fft.fftshift(np.log10(np.abs(np.fft.fft2(crop)) + 1))
        ax = axes[1, 1]
        im = ax.imshow(fft, cmap='inferno', extent=[-0.5, 0.5, -0.5, 0.5])
        ax.set_title("FFT of Sky Residual (log)")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Column profile of sky residual
        ax = axes[1, 2]
        ax.plot(sky_residual.mean(axis=0), lw=0.5)
        ax.set_xlabel("Column"); ax.set_ylabel("Mean Residual")
        ax.set_title("Column Profile of Sky Residual")
        ax.grid(True, alpha=0.3)

        # Row profile
        ax = axes[1, 3]
        ax.plot(sky_residual.mean(axis=1), lw=0.5, color='orange')
        ax.set_xlabel("Row"); ax.set_ylabel("Mean Residual")
        ax.set_title("Row Profile of Sky Residual")
        ax.grid(True, alpha=0.3)

        print(f"  天空光均匀性:")
        print(f"    Raw sky: σ/μ = {sky.std()/sky.mean()*100:.2f}%")
        print(f"    Residual: σ = {sky_residual.std():.1f} ({sky_residual.std()/sky.mean()*100:.2f}%)")

        # Check residual for periodic pattern
        residual_col = sky_residual.mean(axis=0)
        fft_col = np.abs(np.fft.rfft(residual_col - residual_col.mean()))
        freq_col = np.fft.rfftfreq(len(residual_col))
        peaks, props = find_peaks(fft_col[:500], height=fft_col[:500].mean()*3)
        for p in peaks:
            period = 1.0 / freq_col[p] if freq_col[p] > 0 else np.inf
            print(f"    ⚠️  Sky residual 中有周期成分: {period:.1f} px")

    except Exception as e:
        print(f"  Sky fitting error: {e}")

    plt.tight_layout()
    plt.savefig(output_dir / "noise_sky_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 天空光分析图已保存")


def analyze_gray_image(images, output_dir):
    """
    [7] 分析 view2 (grayscale) 图像中是否有波纹。
    如果 view2 中也有波纹，问题可能在前置光学而非色散系统。
    """
    print(f"\n{'='*60}")
    print("🔍 [7] 灰度图像 (view2) 波纹检查")
    print(f"{'='*60}")

    gray = images["gray"]
    H, W = gray.shape

    print(f"  Gray shape: {gray.shape}, dtype={gray.dtype}")
    print(f"  Gray range: [{gray.min()}, {gray.max()}]")
    print(f"  Gray mean±std: {gray.mean():.1f} ± {gray.std():.1f}")

    # FFT of gray
    crop = gray[H//4:3*H//4, W//4:3*W//4]
    fft = np.fft.fftshift(np.log10(np.abs(np.fft.fft2(crop)) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.imshow(gray, cmap='gray')
    ax.set_title("Grayscale (view2)")

    ax = axes[1]
    im = ax.imshow(fft, cmap='inferno', extent=[-0.5, 0.5, -0.5, 0.5])
    ax.set_title("FFT (log) of Center Crop")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Radial profile of FFT
    cy, cx = fft.shape[0]//2, fft.shape[1]//2
    yy, xx = np.ogrid[:fft.shape[0], :fft.shape[1]]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2).astype(int)
    r_flat = r.ravel()
    fft_flat = fft.ravel()

    r_max = min(cy, cx, 300)
    radial = np.array([fft_flat[r_flat == ri].mean() for ri in range(r_max) if (r_flat == ri).any()])

    ax = axes[2]
    ax.plot(radial)
    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("Mean |FFT| (log)")
    ax.set_title("Radial Profile of FFT")

    # Find peaks in radial profile
    peaks, props = find_peaks(radial, height=radial.mean()*1.5, distance=5)
    for p in peaks:
        if p > 5:  # Skip DC
            ax.axvline(p, color='r', linestyle='--', alpha=0.5)
            print(f"  ⚠️ Gray FFT 径向 Profile 峰值 @ r={p} px → 周期 ≈ {1/p*50 if p>0 else 'inf':.1f} px" if p > 0 else "")

    plt.tight_layout()
    plt.savefig(output_dir / "noise_gray_fft.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 灰度图 FFT 分析已保存")


def summarize_findings(centers_wl, etalon_results, has_sky_ripple, has_dark_periodicity):
    """综合所有发现，给出物理来源判断。"""
    print(f"\n{'='*60}")
    print("📊 综合诊断报告")
    print(f"{'='*60}")

    print(f"""
─────────────────────────────────────────────────────────────
  水波纹噪声物理来源诊断
─────────────────────────────────────────────────────────────

  可能来源:
  A. Etalon 效应 (thin-film interference)
     - 特征: 周期 × λ = 常数，频率 ∝ 1/λ
     - 来源: 传感器微透镜/保护玻璃/滤光片的多重反射
     - 判断: 需周期随波长单调变化

  B. 棱镜色散 + Sensor 固定模式噪声
     - 特征: 不同波段中心位置线性漂移 (dX/dλ 恒定)
     - 原因: 棱镜将不同波长分散到不同 sensor 区域
     - 判断: 中心随 λ 线性漂移

  C. 莫尔条纹 (Moiré) — 稀疏采样 + 插值
     - 特征: 仅出现在插值/网格化后，原始数据中不存在
     - 原因: 13040 非均匀点 + nearest-neighbor 插值
     - 判断: 仅在 score map 中可见，原始数据中无

  D. 光照不均匀 / 天地光校正不完善
     - 特征: sky 图像中有类似波纹 / 反射率除法后放大
     - 判断: 查看 sky residual 中是否有周期成分

  E. 图像传感器读出噪声 / 列相关噪声
     - 特征: dark 图像中有列条纹或固定模式
     - 判断: dark 图像 FFT 中有没有周期成分
─────────────────────────────────────────────────────────────
""")

    # Collect evidence
    evidence = []

    if centers_wl is not None and len(centers_wl) > 5:
        wl = centers_wl[:, 1]
        cx = centers_wl[:, 2]
        coeff_x = np.polyfit(wl, cx, 1)
        drift = coeff_x[0] * 460  # Total drift across full spectral range

        if abs(coeff_x[0]) > 0.001:
            evidence.append(f"✅ 中心漂移: dX/dλ = {coeff_x[0]:.4f} px/nm → 全谱漂移 {drift:.1f} px")
            evidence.append(f"   这支持 B: 棱镜色散 — 不同波长投射到不同 sensor 区域")
        else:
            evidence.append(f"❌ 中心无明显波长漂移 (dX/dλ = {coeff_x[0]:.4f} px/nm)")
    else:
        evidence.append("❌ 中心漂移分析数据不足")

    if etalon_results and len(etalon_results) > 1:
        periods = np.array([r['period'] for r in etalon_results])
        wls = np.array([r['wavelength'] for r in etalon_results])
        pw = periods * wls
        cv = pw.std() / pw.mean() * 100

        if cv < 10:
            evidence.append(f"✅ Etalon 可能性大: 周期×λ 变异系数 {cv:.1f}% < 10%")
            evidence.append(f"   周期范围: [{periods.min():.1f}, {periods.max():.1f}] px")
        else:
            evidence.append(f"❌ 非 Etalon: 周期×λ 变异系数 {cv:.1f}% > 10%")

    evidence.append(f"\n  根本原因判断:")

    # Make judgment
    sensor_pattern = has_dark_periodicity
    centering_drift = centers_wl is not None and len(centers_wl) > 5 and abs(np.polyfit(centers_wl[:, 1], centers_wl[:, 2], 1)[0]) > 0.001

    if centering_drift:
        evidence.append(f"  🎯 最可能: 棱镜色散 + 传感器固定模式噪声 (B)")
        evidence.append(f"     不同波长在 sensor 上的位置不同, 如果 sensor 存在固定模式噪声")
        evidence.append(f"     或微透镜阵列周期, 会在不同位置产生不同中心的波纹。")
        evidence.append(f"     在 ACE score map 中, 由于检测公式涉及矩阵运算, 这些噪声被放大。")

    if sensor_pattern:
        evidence.append(f"  ⚠️ 确认传感器有周期性的列/行固定模式噪声")

    evidence.append(f"  🛠 缓解方案:")
    evidence.append(f"     1. 对 score map 做空间带阻滤波器 (陷波滤波器) — 最简单")
    evidence.append(f"     2. 改善反射率计算: 增加空间平滑的 illumination 校正")
    evidence.append(f"     3. 使用更稳健的检测器 (如对噪声不敏感的 SACE 是否同样有波纹?)")
    evidence.append(f"     4. 对 score map 做低通/中值滤波后再二值化")

    for line in evidence:
        print(f"  {line}")

    # Save report
    report_path = output_dir / "noise_diagnosis_report.txt"
    with open(report_path, 'w') as f:
        f.write("水波纹噪声诊断报告\n")
        f.write("=" * 60 + "\n")
        for line in evidence:
            f.write(line + "\n")
    print(f"\n  ✅ 诊断报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="水波纹噪声深度分析")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--hi-dir", default=None)
    parser.add_argument("--output", default="output/ace_samples")
    args = parser.parse_args()

    global base, hi_dir, output_base, output_dir
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(base, "..", "data", "1")
    hi_dir = args.hi_dir or os.path.join(base, "..", "HI")
    output_base = args.output
    output_dir = Path(output_base)

    print("=" * 60)
    print("🌊 水波纹噪声深度分析")
    print("=" * 60)
    t0 = time.time()

    images = load_data(data_dir, hi_dir)

    # 1. Dark analysis
    compute_dark_stats(images["dark"])

    # 6. Sky illumination
    analyze_sky_illumination(images, output_dir)

    # 7. Gray image
    analyze_gray_image(images, output_dir)

    # 2. Band crops
    spec_ds, sky_ds, ref, coords_dict, band_positions = analyze_band_crops(images, output_dir)

    # 3. Center shift
    centers_wl = analyze_ripple_center_shift(ref, coords_dict, output_dir)

    # 4. Etalon test
    etalon_results = analyze_self_correlation(ref, coords_dict, output_dir)

    # 5. ACE score maps
    analyze_ace_score_map_ripple(output_dir)

    # Summary
    has_dark_periodicity = False
    has_sky_ripple = False
    summarize_findings(centers_wl, etalon_results, has_sky_ripple, has_dark_periodicity)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"✅ 全部完成! 耗时: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
