"""
空间频域陷波滤波器 — 去除高光谱成像中的光学干涉条纹噪声。

检测到的噪声频率:
  f1 = 1/17.5 ≈ 0.0571 px⁻¹ (基频，干涉条纹)
  f2 = 1/8.8  ≈ 0.1136 px⁻¹ (2次谐波)
  f3 = 1/5.9  ≈ 0.1695 px⁻¹ (3次谐波)

使用高斯窄带陷波: 在频域以高斯窗乘性衰减，避免振铃伪影。

使用示例:
  from noise_filter import NotchFilter

  nf = NotchFilter()
  sky_clean = nf.filter_image_2d(sky_image)
  ref_clean = nf.filter_reflectance_cube(reflectance_cube)
  score_clean = nf.filter_score_map(score_map)
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks


# ── 默认陷波频率 (基于实测 FFT 峰值) ──
# 对应周期: 17.5px, 8.8px, 5.9px
DEFAULT_NOTCH_FREQS = np.array([1/17.5, 1/8.8, 1/5.9])


class NotchFilter:
    """
    高斯窄带陷波滤波器。

    参数:
        notch_freqs: 要滤除的空间频率列表 (px⁻¹)
        sigma: 高斯陷波宽度 (px⁻¹)，越小陷波越窄
        attenuation: 衰减系数 (0~1)，1=完全移除，0.95=保留5%能量避免振铃
    """

    def __init__(self, notch_freqs=None, sigma=0.005, attenuation=0.95):
        self.notch_freqs = np.asarray(
            notch_freqs if notch_freqs is not None else DEFAULT_NOTCH_FREQS
        )
        self.sigma = sigma
        self.attenuation = attenuation
        # 显示每个陷波对应的周期
        self.notch_periods = 1.0 / self.notch_freqs

    def _build_notch_1d(self, n: int, dx: float = 1.0) -> np.ndarray:
        """
        构建 1D 高斯陷波滤波器。

        参数:
            n: FFT 长度 (信号点数)
            dx: 采样间隔 (px)

        返回:
            mask: (n//2 + 1,) 实FFT频域滤波器，乘性
        """
        freqs = np.fft.rfftfreq(n, d=dx)
        mask = np.ones_like(freqs)
        for f0 in self.notch_freqs:
            # 双边陷波 (正负频率各一个，rfft 正频率包含负频率的幅度)
            # 高斯陷波，避免振铃
            notch = np.exp(-0.5 * ((freqs - f0) / self.sigma) ** 2)
            # 负频率部分: rfft 只包含正频率，不需要额外处理
            # 但对于靠近 DC 的频率要小心不要移除
            mask *= (1.0 - self.attenuation * notch)
        return mask

    def filter_image_1d(self, image: np.ndarray) -> np.ndarray:
        """
        逐行 1D FFT 列陷波滤波。

        对图像的每一行做列方向 FFT (即对 X 轴做 FFT)，
        去除指定频率成分后 IFFT 恢复。

        参数:
            image: (H, W) 2D 图像

        返回:
            filtered: (H, W) 滤波后的图像
        """
        H, W = image.shape
        mask = self._build_notch_1d(W)
        filtered = np.zeros_like(image, dtype=np.float64)

        for y in range(H):
            row = image[y, :].astype(np.float64)
            fft = np.fft.rfft(row)
            fft_filtered = fft * mask
            row_clean = np.fft.irfft(fft_filtered, n=W)
            filtered[y, :] = row_clean

        return filtered

    def filter_image_2d(self, image: np.ndarray) -> np.ndarray:
        """
        2D FFT 陷波滤波 — 用于 Sky 图像一次处理。

        在 2D 频域中同时滤除 X 和 Y 方向的指定频率。
        由于噪声主要是列方向 (X) 周期，对 Y 方向不做陷波。

        参数:
            image: (H, W) 2D 图像

        返回:
            filtered: (H, W) 滤波后的图像
        """
        H, W = image.shape
        # 对 X 方向 (列) 构建陷波
        mask_x = self._build_notch_1d(W)
        # 扩展为 2D: 保留所有 Y 频率，只在 X 方向陷波
        mask_2d = np.tile(mask_x, (H, 1))  # (H, W//2 + 1)

        fft = np.fft.rfft2(image.astype(np.float64))
        fft_filtered = fft * mask_2d
        filtered = np.fft.irfft2(fft_filtered, s=image.shape)
        return filtered

    def filter_reflectance_cube(self, cube: np.ndarray) -> np.ndarray:
        """
        对反射率立方体逐波段 1D 列陷波滤波。

        参数:
            cube: (H, W, B) 反射率数据 (B=波段数)

        返回:
            filtered: (H, W, B) 滤波后的反射率
        """
        H, W, B = cube.shape
        filtered = np.zeros_like(cube, dtype=np.float64)
        mask = self._build_notch_1d(W)

        for b in range(B):
            band = cube[:, :, b].astype(np.float64)
            for y in range(H):
                row = band[y, :]
                fft = np.fft.rfft(row)
                fft *= mask
                band[y, :] = np.fft.irfft(fft, n=W)
            filtered[:, :, b] = band

        return filtered.astype(cube.dtype)

    def filter_score_map(self, score_map: np.ndarray,
                         method: str = 'median',
                         kernel_size: int = 5,
                         morph_kernel: int = 5) -> np.ndarray:
        """
        Score map 后处理平滑 / 形态学去噪。

        参数:
            score_map: (H, W) 检测分数图
            method: 'median' — 中值滤波 (默认)
                    'median5' → 5×5, 'median7' → 7×7, 'median9' → 9×9
                    'gaussian' — 高斯滤波
                    'bilateral' — 双边滤波 (保边)
                    'open' — 形态学开运算 (先腐蚀后膨胀，消除小噪点)
                    'close' — 形态学闭运算 (先膨胀后腐蚀，填充空洞)
                    'med_open' — 中值 + 开运算 (推荐)
                    'full' — 中值 + 开运算 + 中值 (最强去噪)
            kernel_size: 中值/高斯滤波核大小 (奇数)
            morph_kernel: 形态学操作核大小 (奇数)

        返回:
            smoothed: (H, W) 处理后的分数图
        """
        if method == 'median':
            return ndi.median_filter(score_map, size=kernel_size)
        elif method in ('median5', 'median7', 'median9'):
            k = {'median5': 5, 'median7': 7, 'median9': 9}.get(method, 5)
            return ndi.median_filter(score_map, size=k)
        elif method == 'gaussian':
            sigma = kernel_size / 3.0
            return ndi.gaussian_filter(score_map, sigma=sigma)
        elif method == 'bilateral':
            sigma_s = kernel_size / 3.0
            sigma_r = 0.1 * (score_map.max() - score_map.min() + 1e-10)
            return ndi.gaussian_filter(score_map, sigma=(sigma_s, sigma_s))
        elif method in ('open', 'close', 'med_open', 'full'):
            from scipy import ndimage as ndi_morph
            struct = ndi_morph.generate_binary_structure(2, 2)  # 十字形结构元
            se = ndi_morph.iterate_structure(struct, morph_kernel // 2)

            result = score_map.copy()

            if method in ('med_open', 'full'):
                # 先中值滤波
                result = ndi.median_filter(result, size=kernel_size)
                # 开运算 (腐蚀后膨胀) — 消除小亮点
                result = ndi_morph.grey_opening(result, footprint=se)
                if method == 'full':
                    # 再中值滤波
                    result = ndi.median_filter(result, size=kernel_size)
            elif method == 'open':
                result = ndi_morph.grey_opening(result, footprint=se)
            elif method == 'close':
                result = ndi_morph.grey_closing(result, footprint=se)

            return result
        else:
            raise ValueError(f"未知滤波方法: {method}")

    def plot_filter_response(self, n: int = 2048, dx: float = 1.0,
                             save_path: str = None):
        """
        绘制滤波器频率响应曲线。

        参数:
            n: FFT 点数 (用于计算频率轴)
            dx: 采样间隔
            save_path: 保存路径 (可选)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        freqs = np.fft.rfftfreq(n, d=dx)
        mask = self._build_notch_1d(n, dx)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(freqs[:200], mask[:200], 'b-', lw=2)
        for f0 in self.notch_freqs:
            ax1.axvline(f0, color='r', ls='--', alpha=0.5)
            period = 1/f0
            ax1.annotate(f"{period:.1f}px", (f0, 0.5),
                        fontsize=9, ha='center', rotation=90, color='r')
        ax1.set_xlabel("Spatial Frequency (px⁻¹)")
        ax1.set_ylabel("Gain")
        ax1.set_title(f"Notch Filter Frequency Response\n"
                      f"σ={self.sigma:.4f}, A={self.attenuation:.2f}")
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

        # 对数坐标
        ax2.semilogy(freqs[:200], mask[:200], 'b-', lw=2)
        for f0 in self.notch_freqs:
            ax2.axvline(f0, color='r', ls='--', alpha=0.5)
        ax2.set_xlabel("Spatial Frequency (px⁻¹)")
        ax2.set_ylabel("Gain (log)")
        ax2.set_title("Frequency Response (log scale)")
        ax2.set_ylim(0.001, 1.1)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            return fig
            plt.close()

    def __repr__(self):
        periods = ", ".join(f"{p:.1f}px" for p in self.notch_periods)
        return (f"NotchFilter(periods=[{periods}], "
                f"σ={self.sigma:.4f}, A={self.attenuation:.2f})")


# ── 便捷函数 ──

def analyze_fft(image: np.ndarray, title: str = "") -> dict:
    """
    分析图像的列方向 FFT，检测显著周期。

    参数:
        image: (H, W) 图像
        title: 标识名称 (仅用于 print)

    返回:
        {period: amplitude} 字典
    """
    H, W = image.shape
    col_mean = image.mean(axis=0).astype(np.float64)
    col_mean -= col_mean.mean()
    fft = np.abs(np.fft.rfft(col_mean))
    freqs = np.fft.rfftfreq(W)

    # 检测 0.01-0.2 px⁻¹ 范围的峰
    mask = (freqs > 0.01) & (freqs < 0.2)
    peaks, props = find_peaks(fft[mask], height=fft[mask].mean() * 3)

    results = {}
    for p in peaks:
        actual_p = p + np.where(mask)[0][0]
        period = 1.0 / freqs[actual_p] if freqs[actual_p] > 0 else np.inf
        results[period] = fft[actual_p]

    if title:
        if results:
            info = ", ".join(f"{p:.1f}px: {a:.0f}" for p, a in
                           sorted(results.items()))
        else:
            info = "无显著周期"
        print(f"  {title}: {info}")

    return results


if __name__ == "__main__":
    # 简单自测
    nf = NotchFilter()
    print(nf)
    print(f"  陷波频率: {nf.notch_freqs}")
    print(f"  对应周期: {nf.notch_periods}")

    # 测试滤波响应
    test = np.random.randn(512, 2048)
    t0 = __import__('time').time()
    result = nf.filter_image_1d(test)
    elapsed = __import__('time').time() - t0
    print(f"  1D 滤波 512x2048: {elapsed:.3f}s")

    # 测试 2D
    t0 = __import__('time').time()
    result2 = nf.filter_image_2d(test)
    elapsed2 = __import__('time').time() - t0
    print(f"  2D 滤波 512x2048: {elapsed2:.3f}s")

    print("\n✅ NotchFilter 模块加载成功")
