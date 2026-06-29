"""
光谱数据预处理工具

提供暗电流校正、归一化、Savitzky-Golay 平滑、PCHIP 插值等预处理功能。

参考论文:
    - Ma 2014 IJCV: 光谱数据的平滑与插值方法
    - Cao 2011 CVPR: 预处理流程（去除暗电流、平场校正）
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


def subtract_dark_current(img: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """
    暗电流/暗场校正。

    计算: ``result = clip(img - dark, 0)``
    防止 uint16 相减溢出出现 65535。

    参数:
        img: 原始图像
        dark: 暗场图像

    返回:
        校正后的图像，类型与 img 一致
    """
    res = img.astype(np.int32) - dark.astype(np.int32)
    return np.clip(res, 0, None).astype(img.dtype)


def normalize_to_float32(data: np.ndarray, max_val: float = 65535.0) -> np.ndarray:
    """
    将 uint16 数据归一化为 float32。

    参数:
        data: 输入数据 (uint16)
        max_val: 最大值（默认 65535 即 16-bit 满量程）

    返回:
        float32 归一化数据
    """
    return data.astype(np.float32) / max_val


def compute_reflectance(
    raw: np.ndarray,
    reference: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    计算相对反射率。

    公式: ``reflectance = raw / reference``
    先对两者做暗电流校正后再调用此函数。

    参数:
        raw: 目标图像
        reference: 参考图像（如天空光、白板）
        eps: 防零除小量

    返回:
        float32 反射率数据
    """
    raw = raw.astype(np.float64)
    reference = reference.astype(np.float64)
    reference[reference == 0] = eps
    raw[raw == 0] = eps
    refl = raw / reference
    refl[~np.isfinite(refl)] = 0
    return refl.astype(np.float32)


def savgolay_smooth(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    axis: int = 0,
) -> np.ndarray:
    """
    Savitzky-Golay 光谱平滑滤波。

    参数:
        data: 输入光谱数据 (..., bands, ...)
        window_length: 窗口长度（必须为奇数）
        polyorder: 多项式阶数
        axis: 光谱维度的轴

    返回:
        平滑后的数据
    """
    return savgol_filter(
        data.astype(np.float64),
        window_length=window_length,
        polyorder=polyorder,
        axis=axis,
    )


def pchip_interpolate(
    data: np.ndarray,
    nan_threshold: int = 5,
    axis: int = 0,
) -> np.ndarray:
    """
    分段三次 Hermite 插值 (PCHIP)，用于填充光谱中的缺失值。

    算法思路:
        1. 找到每一条光谱中连续非零/非 NaN 的区间
        2. 对每个区间中间的采样点保留，其余置 NaN
        3. 用 PCHIP 插值填充 NaN
        4. 最后端点外推至 0

    参数:
        data: 输入光谱数据
        nan_threshold: 有效点数低于此值则跳过插值
        axis: 光谱维度

    返回:
        插值填充后的数据
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        was_1d = True
    else:
        was_1d = False

    X = data.T if axis == 0 else data
    num_bands, n_pix = X.shape

    # 检测连续非零段
    dX = np.diff(X, axis=0)
    is_change = dX != 0

    start_mask = np.vstack([np.ones((1, n_pix), dtype=bool), is_change])
    end_mask = np.vstack([is_change, np.ones((1, n_pix), dtype=bool)])

    X_tmp = X.copy()
    x_coords = np.arange(1, num_bands + 1)

    for k in range(n_pix):
        s_idx = np.where(start_mask[:, k])[0]
        e_idx = np.where(end_mask[:, k])[0]

        for m in range(len(s_idx)):
            s, e = s_idx[m], e_idx[m]
            if e >= s:
                mid = (s + e) // 2
                X_tmp[s : e + 1, k] = np.nan
                X_tmp[mid, k] = X[mid, k]

    # PCHIP 插值
    X_interp = np.zeros_like(X_tmp, dtype=np.float32)
    for n in range(n_pix):
        v = X_tmp[:, n]
        good = ~np.isnan(v)
        if np.count_nonzero(good) < nan_threshold:
            continue

        x2 = np.append(x_coords[good], num_bands + 1)
        v2 = np.append(v[good], 0)
        pchip = PchipInterpolator(x2, v2)
        X_interp[:, n] = pchip(x_coords)

    X_interp[X_interp < 0] = 0

    result = X_interp.T if axis == 0 else X_interp
    if was_1d:
        result = result.ravel()

    return result


def detect_saturated_bands(
    data: np.ndarray,
    threshold_ratio: float = 10.0,
    clean_ratio: float = 0.5,
) -> tuple:
    """
    检测光谱数据中饱和/异常波段。

    真实高光谱相机在 NIR 端响应弱，sky 信号接近零，
    导致反射率 (orig-dark)/(sky-dark) 在尾部波段爆炸。
    此函数使用前一半清洁波段的中位数作为全局基线。

    算法:
        1. 计算每个波段的均值
        2. 取前 clean_ratio 比例波段的中位数作为全局基线
        3. 标记均值超过 baseline * threshold_ratio 的波段

    参数:
        data: (N, B) 光谱数据
        threshold_ratio: 超过基线多少倍视为饱和（默认 10 倍）
        clean_ratio: 用前多少比例波段估计基线（默认 0.5，即前一半）

    返回:
        good_bands: (K,) 正常波段的整数索引
        bad_bands: (M,) 饱和/异常波段的整数索引
    """
    if data.ndim != 2:
        raise ValueError(f"输入应为 (N, B) 2D 数组，当前 shape={data.shape}")

    band_means = np.mean(data, axis=0)
    n_bands = len(band_means)

    # 用前一半波段的中位数作为全局基线
    n_clean = max(1, int(n_bands * clean_ratio))
    baseline = np.median(band_means[:n_clean])
    baseline = max(baseline, 1e-10)

    ratio = band_means / baseline
    bad_flags = ratio > threshold_ratio

    bad_bands = np.where(bad_flags)[0]
    good_bands = np.where(~bad_flags)[0]

    if len(bad_bands) > 0:
        print(f"⚠️ 检测到 {len(bad_bands)} 个饱和/异常波段 (阈值>{threshold_ratio}x):")
        for b in bad_bands:
            print(f"   波段 {b}: 均值={band_means[b]:.4f}, 基线={baseline:.4f}, 比率={band_means[b]/baseline:.1f}x")

    return good_bands, bad_bands

    if len(bad_bands) > 0:
        wls = [445 + b * 5 for b in bad_bands]
        print(f"⚠️ 检测到 {len(bad_bands)} 个饱和波段:")
        for b, wl in zip(bad_bands, wls):
            ratio_b = ratio[b]
            print(f"   波段 {b} ({wl}nm): 均值={band_means[b]:.1f}, 基线={baseline[b]:.1f}, 比率={ratio_b:.1f}x")

    return good_bands, bad_bands


def clip_spectral_range(
    data: np.ndarray,
    wavelengths: np.ndarray,
    min_wl: float = 445.0,
    max_wl: float = 835.0,
) -> tuple:
    """
    截取指定波长范围的光谱数据。

    参数:
        data: (N, B) 光谱数据
        wavelengths: (B,) 每个波段对应的波长（nm）
        min_wl: 最小保留波长
        max_wl: 最大保留波长

    返回:
        clipped: (N, K) 截取后数据
        clipped_wl: (K,) 截取后波长
    """
    mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
    return data[:, mask], wavelengths[mask]


def normalize_reflectance(
    data: np.ndarray,
    method: str = "mean",
) -> np.ndarray:
    """
    反射率归一化，消除光照强度差异。

    参数:
        data: (N, B) 反射率数据
        method: "mean" — 除以各像素均值; "l2" — L2 归一化; "none" — 不做处理

    返回:
        normalized: (N, B) 归一化数据
    """
    if method == "none":
        return data
    elif method == "mean":
        norm = data.mean(axis=1, keepdims=True) + 1e-10
        return data / norm
    elif method == "l2":
        norm = np.linalg.norm(data, axis=1, keepdims=True) + 1e-10
        return data / norm
    else:
        raise ValueError(f"未知归一化方法: {method}")
