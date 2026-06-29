"""
测试工具：高光谱检测测试数据生成。

提供两种生成模式：
  1. `make_cem_data()` — 原有的基于多项式的合成生成器（向后兼容）
  2. `make_realistic_data()` — 基于真实 background.npy 统计量的逼真数据
"""

import os
import pickle

import numpy as np


def _load_real_vectors(hi_dir: str = None) -> tuple:
    """
    加载真实光谱数据用于统计建模。

    返回:
        background: (N_bg, B) 背景光谱
        targets: dict {1: (N1, B), 2: (N2, B), 3: (N3, B)} 目标光谱
        wavelengths: (B,) 波长数组
    """
    if hi_dir is None:
        hi_dir = os.path.join(os.path.dirname(__file__), "..", "..", "HI")
    hi_dir = os.path.abspath(hi_dir)

    bg_path = os.path.join(hi_dir, "background.npy")
    if not os.path.exists(bg_path):
        raise FileNotFoundError(f"{bg_path} 不存在，无法使用真实数据统计")

    background = np.load(bg_path)
    # 移除饱和波段 (79-92)
    background = background[:, :79]

    targets = {}
    for i in [1, 2, 3]:
        t_path = os.path.join(hi_dir, f"target{i}.npy")
        if os.path.exists(t_path):
            t = np.load(t_path)
            targets[i] = t[:, :79]  # 同样移除饱和波段

    wavelengths = np.arange(445, 840, 5, dtype=int)

    return background, targets, wavelengths


def make_realistic_data(
    num_pixels: int = 2000,
    target_ratio: float = 0.1,
    seed: int = 42,
    use_reflectance: bool = True,
    hi_dir: str = None,
) -> tuple:
    """
    基于真实数据统计生成逼真的检测测试数据。

    流程:
      1. 加载 background.npy 提取背景统计量 (均值谱、协方差特征)
      2. 从真实背景分布采样
      3. 使用真实 target 光谱作为检测目标
      4. 将目标注入背景中

    参数:
        num_pixels: 总像素数
        target_ratio: 目标像素比例
        seed: 随机种子
        use_reflectance: 是否模拟反射率分布
        hi_dir: HI 旧代码目录（包含 background.npy 等）

    返回:
        data: (num_pixels, B) 光谱数据
        target_spectrum: (B,) 目标光谱（用于检测器训练）
        gt_labels: (num_pixels,) bool 数组，True=目标
    """
    background, targets, _ = _load_real_vectors(hi_dir)

    rng = np.random.RandomState(seed)
    n_target = int(num_pixels * target_ratio)
    n_bg = num_pixels - n_target

    B = background.shape[1]

    # 1. 背景建模
    bg_mean = background.mean(axis=0)
    # PCA 分解协方差
    Xc = background - bg_mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    n_components = min(30, B)
    principal_components = Vt[:n_components]  # (n_comp, B)
    eigenvalues = S[:n_components]

    # 2. 生成背景像素
    data = np.zeros((num_pixels, B))

    # 从真实背景随机抽取 + PCA 扰动
    n_bootstrap = min(n_bg, background.shape[0])
    base_indices = rng.choice(background.shape[0], n_bootstrap, replace=False)
    data[:n_bg] = background[base_indices].copy()

    # 对剩余背景像素用 PCA 生成
    if n_bg > n_bootstrap:
        remaining = n_bg - n_bootstrap
        coeffs = rng.normal(0, 1, (remaining, n_components)) * np.sqrt(eigenvalues)
        synthetic_bg = bg_mean + coeffs @ principal_components
        # 添加无相关噪声
        noise = rng.normal(0, 0.01, (remaining, B))
        synthetic_bg = np.maximum(synthetic_bg + noise, 0)
        data[n_bootstrap:n_bg] = synthetic_bg

    # 3. 使用真实目标光谱
    target_spectrum = targets[1].mean(axis=0)

    # 4. 注入目标
    target_indices = rng.choice(num_pixels, n_target, replace=False)
    for idx in target_indices:
        # 用目标替换背景，加随机缩放和噪声
        scale = rng.uniform(0.8, 1.2)
        data[idx] = target_spectrum * scale
        data[idx] += rng.normal(0, 0.01, B)
        data[idx] = np.maximum(data[idx], 0)

    gt_labels = np.zeros(num_pixels, dtype=bool)
    gt_labels[target_indices] = True

    return data, target_spectrum, gt_labels


def make_cem_data(
    num_pixels: int = 1000,
    num_bands: int = 93,
    target_ratio: float = 0.1,
    noise_std: float = 0.01,
    spectral_separation: float = 0.3,
    seed: int = 42,
):
    """
    生成高光谱检测测试数据（向后兼容的合成模式）。

    参数:
        num_pixels: 总像素数
        num_bands: 波段数
        target_ratio: 目标占比
        noise_std: 噪声标准差
        spectral_separation: 分离度
        seed: 随机种子

    返回:
        data: (num_pixels, B) 光谱数据
        target_spectrum: (B,) 目标光谱
        gt_labels: (num_pixels,) bool 数组
    """

    # ——— 以下是向后兼容的合成数据模式 ———
    rng = np.random.RandomState(seed)
    bands = np.arange(num_bands)

    bg_variants = []
    for _ in range(5):
        poly = np.polyval(
            [
                rng.uniform(-2e-5, 2e-5),
                rng.uniform(-0.003, 0.003),
                rng.uniform(-0.1, 0.1),
                rng.uniform(0.5, 1.5),
            ],
            bands,
        )
        sine = 0.05 * rng.uniform(0.5, 1.5) * np.sin(
            2 * np.pi * bands / rng.uniform(15, 30)
        )
        bg_variants.append(poly + sine)

    n_target = int(num_pixels * target_ratio)
    n_bg = num_pixels - n_target
    data = np.zeros((num_pixels, num_bands))

    for i in range(n_bg):
        v1, v2 = rng.randint(0, 5, 2)
        alpha = rng.uniform(0, 1)
        base = bg_variants[v1] * alpha + bg_variants[v2] * (1 - alpha)
        noise = rng.normal(0, noise_std, num_bands)
        data[i] = base + noise

    target_base = bg_variants[rng.randint(0, len(bg_variants))]
    target_spectrum = target_base.copy()
    target_spectrum -= spectral_separation * 0.5 * np.exp(
        -((bands - 25) ** 2) / 40
    )
    target_spectrum += spectral_separation * 0.5 * np.exp(
        -((bands - 55) ** 2) / 30
    )

    gt_labels = np.zeros(num_pixels, dtype=bool)
    target_indices = rng.choice(num_pixels, n_target, replace=False)
    for idx in target_indices:
        noise = rng.normal(0, noise_std * 0.8, num_bands)
        data[idx] = target_spectrum + noise + rng.uniform(-0.02, 0.02, num_bands)
    gt_labels[target_indices] = True

    return data, target_spectrum, gt_labels


if __name__ == "__main__":
    # 测试
    print("=== 合成模式 ===")
    data, target, gt = make_cem_data(num_pixels=500)
    print(f"data: {data.shape}, target: {target.shape}, GT: {gt.sum()}/{len(gt)}")

    print("\n=== 真实模式 ===")
    try:
        data_r, target_r, gt_r = make_realistic_data(num_pixels=500)
        print(f"data: {data_r.shape}, target: {target_r.shape}, GT: {gt_r.sum()}/{len(gt_r)}")
        bg = data_r[~gt_r]
        tg = data_r[gt_r]
        print(f"背景均值: {bg.mean():.4f}, 目标均值: {tg.mean():.4f}")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
