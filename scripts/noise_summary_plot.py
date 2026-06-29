#!/usr/bin/env python3
"""
生成噪声诊断的可视化总结图。
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import subtract_dark_current, compute_reflectance

WAVELENGTHS = np.arange(445, 906, 5, dtype=int)
N_BANDS = 93
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hi_dir = os.path.join(base, "..", "HI")
data_dir = os.path.join(base, "..", "data", "1")
output_dir = os.path.join(base, "output", "ace_samples")

spec = tifffile.imread(os.path.join(data_dir, "5ms.tif"))
dark = tifffile.imread(os.path.join(data_dir, "P11070000.tif"))
sky = tifffile.imread(os.path.join(data_dir, "5ms_sky.tif"))

with open(os.path.join(hi_dir, "coords_dict.json")) as f:
    coords_dict = json.load(f)

# Load ACE score map - try to generate it
print("📊 生成综合诊断图...")

fig = plt.figure(figsize=(20, 24))

# ─── Row 1: column FFT对比 ───
ax = plt.subplot(5, 2, 1)
for name, img, color in [("Dark", dark, 'gray'), ("Sky", sky, 'blue'), ("Spec", spec, 'red')]:
    col = img.mean(axis=0).astype(float)
    col -= col.mean()
    fft = np.abs(np.fft.rfft(col))
    freqs = np.fft.rfftfreq(len(col))
    ax.semilogy(freqs[:200], fft[:200], color=color, label=name, lw=1, alpha=0.8)
ax.set_xlabel("Spatial Frequency (px⁻¹)")
ax.set_ylabel("|FFT|")
ax.set_title("Column-Mean FFT: Dark vs Sky vs Spec")
ax.legend()
ax.set_xlim(0, 0.2)
ax.grid(True, alpha=0.3)

# ─── Row 1 right: 17.5px region zoomed ───
ax = plt.subplot(5, 2, 2)
for name, img, color in [("Sky", sky, 'blue'), ("Spec", spec, 'red')]:
    col = img.mean(axis=0).astype(float)
    col -= col.mean()
    fft = np.abs(np.fft.rfft(col))
    freqs = np.fft.rfftfreq(len(col))
    mask = (freqs > 0.03) & (freqs < 0.08)
    ax.plot(freqs[mask], fft[mask], color=color, label=name, lw=2)
    peaks, props = find_peaks(fft[mask], height=fft[mask].mean()*2)
    for p in peaks:
        actual_p = p + np.where(mask)[0][0]
        ax.axvline(freqs[actual_p], color=color, ls='--', alpha=0.5)
        period = 1.0/freqs[actual_p]
        ax.annotate(f"{period:.1f}px\n{name}",
                   (freqs[actual_p], fft[actual_p]*1.1),
                   fontsize=8, ha='center', color=color)
ax.set_xlabel("Spatial Frequency (px⁻¹)")
ax.set_ylabel("|FFT|")
ax.set_title("17.5px 周期峰 (0.03-0.08 px⁻¹)")
ax.legend()
ax.grid(True, alpha=0.3)

# ─── Row 2: Optics path ───
ax = plt.subplot(5, 2, 3)
# Show the 17.5px pattern in Sky as a column profile section
col_sky = sky.mean(axis=0).astype(float)
x_axis = np.arange(len(col_sky))
ax.plot(x_axis[500:1000], col_sky[500:1000], 'b-', lw=0.8)
# Overlay the detrended version
from scipy.ndimage import uniform_filter1d
sky_smooth = uniform_filter1d(col_sky, size=35)
sky_detrend = col_sky - sky_smooth
ax2 = ax.twinx()
ax2.plot(x_axis[500:1000], sky_detrend[500:1000], 'r-', lw=1, alpha=0.8)
ax2.axhline(0, color='gray', ls='--', lw=0.5)
ax.set_xlabel("Column")
ax.set_ylabel("Mean DN (Sky)", color='b')
ax2.set_ylabel("Detrended", color='r')
ax.set_title("Sky Column Profile (cols 500-1000)\nBlue=raw, Red=detrended (17.5px ripple)")

# ─── Row 2 right: Band X position drift ───
ax = plt.subplot(5, 2, 4)
band_x_means = []
for b in range(N_BANDS):
    xs = []
    for idx_str, spec_list in coords_dict.items():
        if len(spec_list) > b:
            xs.append(spec_list[b][2])
    band_x_means.append(np.mean(xs))
band_x_means = np.array(band_x_means)
wls = np.arange(445, 906, 5)
ax.plot(wls, band_x_means, 'o-', ms=3, lw=1)
coeff = np.polyfit(wls, band_x_means, 1)
ax.plot(wls, np.polyval(coeff, wls), '--', label=f"dX/dλ={coeff[0]:.4f} px/nm")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Mean X Position (px)")
ax.set_title(f"Prism Dispersion: Band X Position vs λ\n"
             f"Total ΔX={coeff[0]*460:.1f}px across 93 bands")
ax.legend()
ax.grid(True, alpha=0.3)

# ─── Row 3: The core mechanism diagram ───
ax = plt.subplot(5, 2, (5,6))
# Draw a conceptual diagram showing the noise mechanism
ax.axis('off')
mechanism_text = """
水波纹噪声产生机理:
    1. 根源: 光学路径中的干涉条纹 (etalon effect)
       ─ 在 Sky 图像中检测到强 17.5px 周期 (暗场中无此周期)
       ─ 谐波: 8.8px (2次), 5.8px (3次)

    2. 棱镜色散: dX/dλ = 0.09 px/nm
       ─ 93个波段横跨 ~42px 的 sensor 区域
       ─ 同一场景点的不同波段数据来自 sensor 不同列

    3. 数据流:
       反射率 R(x,y,λ) = (Spec − Dark) / (Sky − Dark)
       ─ Sky 的周期性非均匀性除到反射率中
       ─ 各波段的周期性相位 ∝ (X₀ + 0.09·λ) / 17.5px

    4. ACE 检测:
       ACE(x) = (dᵀR⁻¹(x-μ))² / ((dᵀR⁻¹d)·(x-μ)ᵀR⁻¹(x-μ))
       ─ 协方差逆矩阵 R⁻¹ 放大光谱维的周期性
       ─ 93个波段的相位变化 → 空间同心圆
       ─ 不同波段"中心"不同 → 用户观察到的"波纹中心漂移"

    5. 为何暗场无此噪声:
       暗场无光照 → 无光学干涉条纹 → 无 17.5px 周期

┌─────────────────────────────────────────────────────────────┐
│  缓解方案:                                                   │
│  A) 频域陷波滤波: 对 2D 反射率图像做带阻滤波 (17.5/8.8px)    │
│  B) Sky 低通滤波: 平滑 Sky 后再除, 减少周期性非均匀性        │
│  C) Score Map 后处理: 对 ACE 结果做 2D 中值/高斯滤波         │
│  D) 空间维度Binning: 2×2 bin 缩小周期影响 (但损失分辨率)     │
└─────────────────────────────────────────────────────────────┘
"""
ax.text(0, 0.95, mechanism_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace', linespacing=1.5)

# ─── Row 4: Spec left vs right comparison (showing pattern persists) ───
ax = plt.subplot(5, 2, 7)
for region, x_slice, color in [("Left 1-512", slice(0,512), 'r'),
                                ("Mid 512-1024", slice(512,1024), 'g'),
                                ("Mid 1024-1536", slice(1024,1536), 'b'),
                                ("Right 1536-2048", slice(1536,2048), 'orange')]:
    col = spec[:, x_slice].mean(axis=0).astype(float)
    col -= col.mean()
    fft = np.abs(np.fft.rfft(col))
    freqs = np.fft.rfftfreq(len(col))
    mask = (freqs > 0.03) & (freqs < 0.08)
    ax.plot(freqs[mask], fft[mask], color=color, label=region, lw=1.5)
    peaks, _ = find_peaks(fft[mask], height=fft[mask].mean()*2)
    for p in peaks:
        actual_p = p + np.where(mask)[0][0]
        ax.axvline(freqs[actual_p], color=color, ls='--', alpha=0.3)
ax.set_xlabel("Frequency (px⁻¹)")
ax.set_ylabel("|FFT|")
ax.set_title("Spec: 17.5px 周期在不同图像区域都存在")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ─── Row 4 right: Dark pattern periods (harmonics) ───
ax = plt.subplot(5, 2, 8)
col = dark.mean(axis=0).astype(float)
col -= col.mean()
fft = np.abs(np.fft.rfft(col))
freqs = np.fft.rfftfreq(len(col))
ax.semilogy(freqs[:300], fft[:300], 'gray', lw=1)
peaks, props = find_peaks(fft[:300], height=fft[:300].mean()*3)
for p in peaks:
    period = 1.0/freqs[p] if freqs[p] > 0 else np.inf
    ax.axvline(freqs[p], color='r', ls='--', alpha=0.5)
    ax.annotate(f"{period:.0f}px", (freqs[p], fft[p]*1.5),
                fontsize=7, ha='center', rotation=90)
ax.set_xlabel("Frequency (px⁻¹)")
ax.set_ylabel("|FFT|")
ax.set_title("Dark 噪声: 2048/n 谐波 (读出时钟相关)\n无 17.5px 峰 → 非传感器电子噪声")
ax.grid(True, alpha=0.3)

# ─── Row 5: Band position ΔX histogram ───
ax = plt.subplot(5, 2, 9)
# Show ΔX between consecutive bands
band_x_stds = []
for b in range(N_BANDS):
    xs = []
    for idx_str, spec_list in coords_dict.items():
        if len(spec_list) > b:
            xs.append(spec_list[b][2])
    band_x_stds.append(np.std(xs))
ax.plot(wls, band_x_stds, 'o-', ms=3, lw=1, color='purple')
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("X position spread (px)")
ax.set_title("Each Band's X Coverage Width\n"
             f"mean={np.mean(band_x_stds):.1f}px, constant across bands")
ax.grid(True, alpha=0.3)

ax = plt.subplot(5, 2, 10)
# Calculate ΔX from band N to N+1 per point
deltas = []
sample_keys = list(coords_dict.keys())[:500]
for k in sample_keys:
    s = coords_dict[k]
    if len(s) == N_BANDS:
        for b in range(N_BANDS-1):
            deltas.append(s[b+1][2] - s[b][2])
ax.hist(deltas, bins=30, alpha=0.7, color='teal')
ax.axvline(np.mean(deltas), color='r', ls='--', label=f"mean={np.mean(deltas):.3f}px/band")
ax.set_xlabel("ΔX per band (px)")
ax.set_ylabel("Count")
ax.set_title(f"Inter-Band X Step\n= {np.mean(deltas):.3f} px/band")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "noise_mechanism_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ 综合诊断图已保存")
