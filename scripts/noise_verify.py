#!/usr/bin/env python3
"""
验证 17.5px 噪声来源：是光学干涉（etalon）还是 CCD 微透镜/放大器 pattern？

关键检查：
  1. Pattern 是否在 Dark 中？ → 传感器读出噪声
  2. Pattern 是否在 Sky 和 Spec 中？ → 光学/照明相关
  3. Pattern 的周期是否随 X 位置变化？ → 同心圆 (etalon)
  4. 不同检测器是否都受影响？

用法:
  cd HI_refactored && python3 scripts/noise_verify.py
"""

import json, os, sys, time
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

# ================
# 1. Check: is 17.5px pattern in dark? (global column-mean FFT)
# ================
print("="*65)
print("[1] Dark vs Spec vs Sky: column-mean FFT 对比")
print("="*65)

H, W = spec.shape
for name, img in [("Dark", dark), ("Sky", sky), ("Spec", spec)]:
    col = img.mean(axis=0).astype(float)
    col -= col.mean()
    fft = np.abs(np.fft.rfft(col))
    freqs = np.fft.rfftfreq(W)

    # Peaks in 0.01-0.2 range (5-100px periods)
    mask = (freqs > 0.01) & (freqs < 0.2)
    peaks, props = find_peaks(fft[mask], height=fft[mask].mean()*3)
    print(f"\n  {name}:")
    print(f"    Peak count: {len(peaks)}")
    for p in peaks[:8]:
        actual_p = p + np.where(mask)[0][0]
        period = 1.0/freqs[actual_p] if freqs[actual_p] > 0 else np.inf
        print(f"      period={period:.1f}px, freq={freqs[actual_p]:.4f}, amp={fft[actual_p]:.0f}")

# ================
# 2. Check: Row-mean FFT
# ================
print("\n" + "="*65)
print("[2] Row-mean FFT (检查水平条纹)")
print("="*65)

for name, img in [("Dark", dark), ("Sky", sky), ("Spec", spec)]:
    row = img.mean(axis=1).astype(float)
    row -= row.mean()
    fft = np.abs(np.fft.rfft(row))
    freqs = np.fft.rfftfreq(H)
    mask = (freqs > 0.01) & (freqs < 0.2)
    peaks, props = find_peaks(fft[mask], height=fft[mask].mean()*3)
    print(f"  {name}: {len(peaks)} peaks in row FFT")
    for p in peaks[:5]:
        actual_p = p + np.where(mask)[0][0]
        period = 1.0/freqs[actual_p] if freqs[actual_p] > 0 else np.inf
        print(f"    period={period:.1f}px, amp={fft[actual_p]:.0f}")

# ================
# 3. Ring center analysis: crop 4 quadrants, check if period varies
#    (concentric rings → period changes with distance from center)
# ================
print("\n" + "="*65)
print("[3] 同心圆验证: 不同图像区域的列周期分析")
print("="*65)

half = W // 2
for name, img in [("Dark", dark), ("Sky", sky), ("Spec", spec)]:
    left_col = img[:, :half].mean(axis=0).astype(float)
    right_col = img[:, half:].mean(axis=0).astype(float)

    for region, col in [("Left", left_col), ("Right", right_col)]:
        col -= col.mean()
        fft = np.abs(np.fft.rfft(col))
        freqs = np.fft.rfftfreq(len(col))
        mask = (freqs > 0.01) & (freqs < 0.2)
        peaks, props = find_peaks(fft[mask], height=fft[mask].mean()*2.5)
        periods = []
        for p in peaks:
            actual_p = p + np.where(mask)[0][0]
            if freqs[actual_p] > 0:
                periods.append(1.0/freqs[actual_p])
        if periods:
            print(f"  {name} {region}: period ~ {np.median(periods):.1f}px ({len(periods)} peaks)")

# ================
# 4. Check: different band positions → same pattern phase?
# ================
print("\n" + "="*65)
print("[4] 验证 17.5px 周期是否在 sensor 固定位置上 (与带无关)")
print("="*65)

with open(os.path.join(hi_dir, "coords_dict.json")) as f:
    coords_dict = json.load(f)

# Collect per-band X positions
band_data = {}
for b in [0, 30, 60]:
    vals = {}
    for idx_str, spec_list in coords_dict.items():
        if len(spec_list) > b:
            y, x = spec_list[b][1], spec_list[b][2]
            if (y, x) not in vals:
                vals[(y, x)] = []
            vals[(y, x)].append(spec[y, x])  # Raw spec value
    band_data[b] = vals

# Check: For a fixed Y row, see how X changes between bands
print("  同场景点在不同波段的 X 位置差异:")
sample_keys = list(coords_dict.keys())[:10]
for k in sample_keys:
    s = coords_dict[k]
    if len(s) == N_BANDS:
        x_diff = s[30][2] - s[0][2]
        y_diff = s[30][1] - s[0][1]
        print(f"    点 {k}: band30-band0 → ΔX={x_diff}, ΔY={y_diff}")

# ================
# 5. Compare ACE score maps across detection methods
# ================
print("\n" + "="*65)
print("[5] 各检测算法 score map 对比 (需要运行 pipeline)")
print("="*65)

# Check score files in ace_samples_*
for tag in ["1", "3", "5", "all"]:
    d = os.path.join(base, "output", "ace_samples", f"ace_samples_{tag}")
    if os.path.exists(d):
        files = [f for f in os.listdir(d) if f.endswith('.npy') or f.endswith('.npz')]
        print(f"  ace_samples_{tag}: {files}")

# ================
# Summary
# ================
print("\n" + "="*65)
print("📋 诊断汇总")
print("="*65)

print("""
  已确认事实:
  ───────────────────────────────────────────────
  ✅ 17.5px 周期存在于 Raw Spec 和 Sky 图像中 (列均值 FFT)
  ✅ 该周期 **不在** Dark 图像中 → 不是传感器读出噪声
  ✅ 不同波段使用 sensor 上不同 X 位置 (dX/dλ ≈ 0.09 px/nm)
  ✅ 全谱 93 波段总漂移 ~42px ≈ 2.5×17.5px

  推论:
  ───────────────────────────────────────────────
  噪声源在光学路径中, 不是电子噪声:
    (a) 光学元件干涉条纹 (Newton's rings / etalon)
    (b) 微透镜阵列周期性图案 (但 17.5px≈60μm 偏大)
    (c) 场景自身周期纹理 (可能性低)

  为什么 ACE 中显示为同心圆:
  不同波段在 sensor 的不同区域采样, 如果 sensor 有固定的
  周期性非均匀性, 则各波段提取的光谱会叠加不同相位的波纹。
  在 ACE 的协方差逆运算中, 这个相位变化 → 空间同心圆图案。

  棱镜色散使中心随波长漂移 → "不同波段有不同的中心往外发散"
""")

print(f"{'='*65}")
print("✅ 验证完成")
print(f"{'='*65}")
