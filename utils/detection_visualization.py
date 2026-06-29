"""
检测结果可视化 — 在可见光图像上标注目标并生成伪彩色图像

核心功能:
  1. 高光谱立方体 → 伪彩色 RGB (R=650nm, G=550nm, B=450nm)
  2. 可见光灰度图生成 (visible band加权平均)
  3. 检测结果叠加标注 (矩形/圆形/热力图)
  4. 色散偏差修正: 光谱检测坐标 → 可见光图像坐标的映射
  5. 综合诊断图: 原图 + 标注 + 伪彩 + 分数分布

色散修正原理:
  棱镜色散使得不同波长的光落在传感器不同位置。
  同一物理点在 band_i 和 band_j 之间存在像素偏移。
  偏移量由柯西色散公式描述:
      dx(λ) = D · (λ_ref² / λ² - 1)
  当我们将高光谱检测结果映射到可见光图像时，
  需要用检测波段的偏移量减去可见光参考波段的偏移量。

参考:
  - Du 2009: 棱镜色散系统的空间偏移建模
  - Feng 2014: Amici 棱镜色散特性
  - calibrate_prism_dispersion() 中的柯西公式实现
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import TARGET_WAVELENGTHS, NUM_BANDS


# ============================================================
# 色散修正
# ============================================================

def compute_dispersion_offset(
    from_band: int,
    to_band: int,
    dispersion_strength: float = 15.0,
    ref_wavelength: int = 535,
) -> float:
    """
    计算两个波段之间的色散偏移量 (像素)。

    使用柯西色散模型:
        dx(λ) = D · (λ_ref² / λ² - 1)

    参数:
        from_band: 原波段索引 (0~92)
        to_band: 目标波段索引
        dispersion_strength: 色散强度 D
        ref_wavelength: 参考波长 (nm)，与 calibrate_prism_dispersion 一致

    返回:
        dx: 从 from_band 到 to_band 的 x 方向偏移量 (像素)
            正值表示向右偏移
    """
    wl_from = TARGET_WAVELENGTHS[from_band]
    wl_to = TARGET_WAVELENGTHS[to_band]
    ref_factor = 1.0 / (ref_wavelength ** 2)

    dx_from = dispersion_strength * ((1.0 / (wl_from ** 2)) / ref_factor - 1.0)
    dx_to = dispersion_strength * ((1.0 / (wl_to ** 2)) / ref_factor - 1.0)

    return dx_to - dx_from


def dispersion_correct_points(
    points: np.ndarray,
    from_band: int,
    to_band: int,
    dispersion_strength: float = 15.0,
    ref_wavelength: int = 535,
) -> np.ndarray:
    """
    对一组检测点进行色散修正。

    参数:
        points: (N, 2) 坐标数组，每行 [y, x]
        from_band: 检测所用波段索引
        to_band: 目标可视化波段索引 (如可见光参考波段)
        dispersion_strength: 色散强度
        ref_wavelength: 参考波长

    返回:
        corrected: (N, 2) 修正后的坐标
    """
    dx = compute_dispersion_offset(from_band, to_band, dispersion_strength, ref_wavelength)
    corrected = points.copy().astype(np.float32)
    corrected[:, 1] += dx  # x 方向偏移
    return corrected


def get_calibration_dispersion_offsets(
    spec_yx: dict,
    first_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从标定数据中提取每个波段相对于第一波段的偏移量。

    利用 spec_yx 中各波段 y 坐标与第一波段 y 坐标的中位数差异
    作为该波段的偏移量。

    参数:
        spec_yx: {band_idx_str: (M,) y_coords}
        first_coords: (M, 2) 第一波段的 (y, x) 坐标数组

    返回:
        offset_y: (C,) 每个波段的 y 偏移量 (像素)
        offset_x: (C,) 每个波段的 x 偏移量 (像素) — 从标定数据估算
                  注：spec_yx 仅存储 y 坐标，x 偏移用色散模型补充
    """
    if first_coords.ndim == 1:
        # spec_yx 只存了 y 坐标（1D），无法获取 x 偏移
        C = len(spec_yx)
        offset_y = np.zeros(C)
        for b_str, y_coords in spec_yx.items():
            b = int(b_str) - 1
            if b < C:
                diff = y_coords - first_coords[:len(y_coords)]
                offset_y[b] = np.median(diff) if len(diff) > 0 else 0.0

        # x 偏移使用色散模型估算
        offset_x = np.array([
            compute_dispersion_offset(0, b) for b in range(C)
        ])
        return offset_y, offset_x

    # first_coords 是 2D (M, 2)
    C = len(spec_yx)
    offset_y = np.zeros(C)
    offset_x = np.zeros(C)
    fcy = first_coords[:, 0]
    fcx = first_coords[:, 1]

    for b_str, y_coords in spec_yx.items():
        b = int(b_str) - 1
        if b >= C:
            continue
        n = min(len(y_coords), len(fcy))
        if n == 0:
            continue
        offset_y[b] = np.median(y_coords[:n] - fcy[:n])
        offset_x[b] = np.median(y_coords[:n] - fcx[:n])

    return offset_y, offset_x


# ============================================================
# 伪彩色和灰度图像生成
# ============================================================

def compute_pseudo_rgb(
    hyperspectral_cube: np.ndarray,
    r_wavelength: int = 650,
    g_wavelength: int = 550,
    b_wavelength: int = 450,
    gamma: float = 1.0,
    clip_percentile: float = 2.0,
) -> np.ndarray:
    """
    从高光谱立方体生成伪彩色 RGB 图像。

    将三个指定波长的灰度切片映射为 RGB 通道，
    并做百分位拉伸和 Gamma 校正增强视觉效果。

    参数:
        hyperspectral_cube: (H, W, C) 高光谱数据，C=93
        r_wavelength: 红通道波长 (nm)，默认 650nm
        g_wavelength: 绿通道波长 (nm)，默认 550nm (可见光最敏感)
        b_wavelength: 蓝通道波长 (nm)，默认 450nm
        gamma: Gamma 校正系数 (<1 提亮暗部, >1 压暗)
        clip_percentile: 每通道拉伸的百分位 (默认 2%)

    返回:
        rgb: (H, W, 3) float32 RGB 图像，范围 [0, 1]
    """
    def _find_band(wl: int) -> int:
        """找到最接近指定波长的波段索引"""
        idx = np.argmin(np.abs(np.array(TARGET_WAVELENGTHS) - wl))
        return idx

    def _stretch_channel(ch: np.ndarray, p: float) -> np.ndarray:
        """百分位拉伸"""
        lo, hi = np.percentile(ch, [p, 100 - p])
        if hi - lo < 1e-6:
            return np.zeros_like(ch)
        stretched = (ch.astype(np.float32) - lo) / (hi - lo)
        return np.clip(stretched, 0, 1)

    H, W, C = hyperspectral_cube.shape
    r_idx = _find_band(r_wavelength)
    g_idx = _find_band(g_wavelength)
    b_idx = _find_band(b_wavelength)

    r_ch = _stretch_channel(hyperspectral_cube[:, :, r_idx], clip_percentile)
    g_ch = _stretch_channel(hyperspectral_cube[:, :, g_idx], clip_percentile)
    b_ch = _stretch_channel(hyperspectral_cube[:, :, b_idx], clip_percentile)

    rgb = np.stack([r_ch, g_ch, b_ch], axis=2)

    # Gamma 校正
    if gamma != 1.0:
        rgb = rgb ** (1.0 / gamma)

    return rgb.astype(np.float32)


def compute_visible_grayscale(
    hyperspectral_cube: np.ndarray,
    wl_start: int = 450,
    wl_end: int = 650,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    从高光谱立方体计算可见光灰度图像。

    默认对 450~650nm 范围内的波段加权平均，
    模拟人眼对可见光的感知。

    参数:
        hyperspectral_cube: (H, W, C) 高光谱数据
        wl_start: 起始波长 (nm)
        wl_end: 结束波长 (nm)
        weights: 可选的自定义权重 (n_bands_selected,)

    返回:
        gray: (H, W) float32 灰度图像，0~1 归一化
    """
    # 找到波长范围内的波段索引
    band_indices = [
        i for i, wl in enumerate(TARGET_WAVELENGTHS)
        if wl_start <= wl <= wl_end
    ]

    if not band_indices:
        raise ValueError(f"波长范围 [{wl_start}, {wl_end}] 无对应波段")

    selected = hyperspectral_cube[:, :, band_indices].astype(np.float32)

    if weights is not None:
        assert len(weights) == len(band_indices), \
            f"权重长度 {len(weights)} 与波段数 {len(band_indices)} 不匹配"
        weights = weights / weights.sum()
        gray = np.dot(selected, weights)
    else:
        # 人眼视觉灵敏度加权: 绿色权重最高
        wl_weights = np.array([
            1.0 - abs(wl - 555) / 200  # 555nm 是人眼最敏感波长
            for wl in [TARGET_WAVELENGTHS[i] for i in band_indices]
        ])
        wl_weights = np.clip(wl_weights, 0.1, 1.0)
        wl_weights = wl_weights / wl_weights.sum()
        gray = np.dot(selected, wl_weights)

    # 百分位拉伸
    lo, hi = np.percentile(gray, [2, 98])
    if hi - lo > 1e-6:
        gray = np.clip((gray - lo) / (hi - lo), 0, 1)
    else:
        gray = np.zeros_like(gray)

    return gray.astype(np.float32)


# ============================================================
# 检测标注
# ============================================================

def overlay_detections(
    image: np.ndarray,
    detection_points: Optional[np.ndarray] = None,
    detection_mask: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    score_threshold: Optional[float] = None,
    marker: str = "circle",
    marker_size: int = 5,
    color: Union[str, Tuple] = "red",
    alpha: float = 0.6,
    show_legend: bool = True,
) -> np.ndarray:
    """
    在图像上叠加检测结果。

    支持两种输入:
        1. detection_points: (N, 2) 坐标数组 [y, x]
        2. detection_mask: (H, W) 布尔掩膜

    参数:
        image: (H, W) 灰度或 (H, W, 3) RGB 图像，float32 [0,1]
        detection_points: 检测点坐标
        detection_mask: 布尔检测掩膜
        scores: 每个点的检测分数 (与 detection_points 对应)
        score_threshold: 分数阈值 (仅显示分数高于此的点)
        marker: 'circle' 画圆, 'rect' 画矩形框, 'cross' 画十字
        marker_size: 标记大小 (像素)
        color: matplotlib 颜色名或 RGB tuple
        alpha: 叠加透明度
        show_legend: 是否显示图例

    返回:
        overlay: (H, W, 4) RGBA 叠加图像
    """
    H, W = image.shape[:2]
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    n_det = 0

    # 从 mask 提取点
    if detection_mask is not None:
        points = np.argwhere(detection_mask)  # (N, 2) [y, x]
        if scores is not None and score_threshold is not None:
            valid = scores >= score_threshold
            points = points[valid]
    elif detection_points is not None:
        points = detection_points
    else:
        return overlay

    if len(points) == 0:
        return overlay

    # 过滤分数
    if scores is not None and score_threshold is not None:
        assert len(scores) == len(points), \
            f"scores 长度 {len(scores)} != points 长度 {len(points)}"
        valid = scores >= score_threshold
        points = points[valid]
        scores = scores[valid]

    n_det = len(points)

    # 解析颜色
    if isinstance(color, str):
        from matplotlib.colors import to_rgb
        rgb = to_rgb(color)
    else:
        rgb = tuple(color[:3])

    # 在 points 位置画标记
    for i, (y, x) in enumerate(points):
        y, x = int(round(y)), int(round(x))
        if y < 0 or y >= H or x < 0 or x >= W:
            continue

        if marker == "circle":
            cv2.circle(overlay, (x, y), marker_size, (*rgb, alpha), -1)
            cv2.circle(overlay, (x, y), marker_size, (*rgb, 1.0), 1)
        elif marker == "rect":
            half = marker_size
            x1, y1 = max(0, x - half), max(0, y - half)
            x2, y2 = min(W - 1, x + half), min(H - 1, y + half)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (*rgb, alpha), -1)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (*rgb, 1.0), 1)
        elif marker == "cross":
            cv2.drawMarker(overlay, (x, y), (*rgb, alpha),
                           cv2.MARKER_CROSS, marker_size * 2, 1)

    return overlay


def create_score_heatmap(
    scores: np.ndarray,
    coords: np.ndarray,
    image_shape: Tuple[int, int],
    radius: int = 3,
    colormap: str = "hot",
    normalize: bool = True,
) -> np.ndarray:
    """
    从稀疏检测分数生成密集热力图。

    参数:
        scores: (N,) 每个检测点的分数
        coords: (N, 2) 坐标 [y, x]
        image_shape: (H, W) 输出图像尺寸
        radius: 高斯扩散半径
        colormap: matplotlib colormap 名称
        normalize: 是否归一化到 [0, 1]

    返回:
        heatmap: (H, W, 3) 伪彩色热力图
    """
    H, W = image_shape
    score_map = np.zeros((H, W), dtype=np.float32)

    for i, (y, x) in enumerate(coords):
        y, x = int(round(y)), int(round(x))
        if y < 0 or y >= H or x < 0 or x >= W:
            continue
        score = scores[i] if i < len(scores) else 1.0
        # 高斯扩散
        y_min, y_max = max(0, y - radius), min(H, y + radius + 1)
        x_min, x_max = max(0, x - radius), min(W, x + radius + 1)
        for dy in range(y_min, y_max):
            for dx in range(x_min, x_max):
                dist = np.sqrt((dy - y) ** 2 + (dx - x) ** 2)
                weight = max(0, 1 - dist / radius)
                score_map[dy, dx] += score * weight

    if normalize and score_map.max() > 0:
        score_map = score_map / score_map.max()

    # 应用 colormap
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(score_map)[:, :, :3]  # (H, W, 3)
    return heatmap.astype(np.float32)


def detections_to_label_image(
    binary_mask: np.ndarray,
    class_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    将检测掩膜转换为标签图像 (每个连通区域一个 ID)。

    参数:
        binary_mask: (H, W) 布尔检测掩膜
        class_ids: 每个连通区域的类别 ID (可选)

    返回:
        label_img: (H, W) int32 标签图像，0=背景，1+=目标
    """
    from skimage import measure

    labeled = measure.label(binary_mask, connectivity=2)
    return labeled.astype(np.int32)


# ============================================================
# 综合可视化
# ============================================================

def plot_detection_summary(
    image_gray: np.ndarray,
    detection_mask: Optional[np.ndarray] = None,
    detection_points: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    pseudo_rgb: Optional[np.ndarray] = None,
    score_heatmap: Optional[np.ndarray] = None,
    title: str = "检测结果",
    save_path: Optional[str] = None,
    dpi: int = 150,
    show: bool = True,
):
    """
    绘制综合检测结果图。

    布局:
        [可见光原图]  [检测标注叠加]
        [伪彩色RGB]   [分数热力图/检测区域]

    参数:
        image_gray: (H, W) 可见光灰度图像
        detection_mask: (H, W) 布尔检测掩膜
        detection_points: (N, 2) 检测点 [y, x]
        scores: (N,) 检测分数
        pseudo_rgb: (H, W, 3) 伪彩色 RGB
        score_heatmap: (H, W, 3) 分数热力图
        title: 图表标题
        save_path: 保存路径 (如 .png)
        dpi: 分辨率
        show: 是否显示
    """
    n_subplots = 1
    if detection_mask is not None or detection_points is not None:
        n_subplots += 1
    if pseudo_rgb is not None:
        n_subplots += 1
    if score_heatmap is not None:
        n_subplots += 1

    fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 5))
    if n_subplots == 1:
        axes = [axes]

    plot_idx = 0

    # 1. 可见光原图
    ax = axes[plot_idx]
    ax.imshow(image_gray, cmap="gray")
    ax.set_title("可见光灰度图像", fontsize=10)
    ax.axis("off")
    plot_idx += 1

    # 2. 检测标注叠加
    if detection_mask is not None or detection_points is not None:
        ax = axes[plot_idx]
        ax.imshow(image_gray, cmap="gray")
        overlay = overlay_detections(
            image_gray,
            detection_mask=detection_mask,
            detection_points=detection_points,
            scores=scores,
        )
        # 叠加 RGBA
        ax.imshow(overlay)
        n_det = np.count_nonzero(detection_mask) if detection_mask is not None else len(detection_points)
        ax.set_title(f"检测标注 ({n_det} 像素)", fontsize=10)
        ax.axis("off")
        plot_idx += 1

    # 3. 伪彩色 RGB
    if pseudo_rgb is not None:
        ax = axes[plot_idx]
        ax.imshow(pseudo_rgb)
        ax.set_title("伪彩色 RGB\n(R=650 G=550 B=450)", fontsize=10)
        ax.axis("off")
        plot_idx += 1

    # 4. 分数热力图
    if score_heatmap is not None:
        ax = axes[plot_idx]
        ax.imshow(score_heatmap)
        ax.set_title("检测分数热力图", fontsize=10)
        ax.axis("off")
        plot_idx += 1

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✅ 结果图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_dual_with_dispersion_correction(
    image_gray: np.ndarray,
    detection_mask: np.ndarray,
    coords_dict: Optional[dict] = None,
    calibration_offsets: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    visible_band_idx: int = 21,
    detection_band_idx: int = 0,
    title: str = "色散修正对比",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    绘制带色散修正的检测对比图。

    左图: 无修正（直接标注）
    右图: 有修正（考虑色散偏移后标注）

    参数:
        image_gray: (H, W) 可见光灰度图像
        detection_mask: (H, W) 检测掩膜 (在检测波段坐标系中)
        coords_dict: 轨迹字典 {id: [(band, y, x), ...]}
        calibration_offsets: (offset_y, offset_x) 每个波段的偏移量
        visible_band_idx: 可见光参考波段索引
        detection_band_idx: 检测波段索引
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 左图: 无修正 ---
    ax1.imshow(image_gray, cmap="gray")
    overlay1 = overlay_detections(image_gray, detection_mask=detection_mask)
    ax1.imshow(overlay1)
    ax1.set_title("无色散修正\n(检测坐标直接叠加)", fontsize=10)
    ax1.axis("off")

    # --- 右图: 有色散修正 ---
    ax2.imshow(image_gray, cmap="gray")

    if calibration_offsets is not None:
        offset_y, offset_x = calibration_offsets
        dy = offset_y[detection_band_idx] - offset_y[visible_band_idx] if visible_band_idx < len(offset_y) else 0
        dx = offset_x[detection_band_idx] - offset_x[visible_band_idx] if visible_band_idx < len(offset_x) else 0

        # 平移检测掩膜
        if dy != 0 or dx != 0:
            H, W = detection_mask.shape
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            corrected_mask = cv2.warpAffine(
                detection_mask.astype(np.uint8), M, (W, H)
            ) > 0
        else:
            corrected_mask = detection_mask
    else:
        # 使用色散模型
        corrected_mask = detection_mask

    overlay2 = overlay_detections(image_gray, detection_mask=corrected_mask)
    ax2.imshow(overlay2)
    correction_info = f"偏移: dx={dx:.1f}px, dy={dy:.1f}px" if 'dx' in locals() else "模型修正"
    ax2.set_title(f"有色散修正\n({correction_info})", fontsize=10)
    ax2.axis("off")

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ 色散修正对比图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# GIF / 帧序列
# ============================================================

def create_detection_animation_frame(
    frame_idx: int,
    hyperspectral_cube: np.ndarray,
    detection_mask: np.ndarray,
    score: Optional[float] = None,
    band_idx: Optional[int] = None,
) -> np.ndarray:
    """
    生成检测动画的单个帧 (用于 GIF 或视频)。

    参数:
        frame_idx: 帧序号
        hyperspectral_cube: (H, W, C) 全数据
        detection_mask: (H, W) 当前帧检测掩膜
        score: 当前帧检测分数
        band_idx: 当前帧对应波段 (用于显示)

    返回:
        frame: (H, W, 3) uint8 图像
    """
    if band_idx is None:
        band_idx = frame_idx
    if band_idx >= hyperspectral_cube.shape[2]:
        band_idx = hyperspectral_cube.shape[2] - 1

    # 当前波段灰度
    ch = hyperspectral_cube[:, :, band_idx].astype(np.float32)
    lo, hi = np.percentile(ch, [5, 95])
    if hi - lo > 1e-6:
        ch = np.clip((ch - lo) / (hi - lo), 0, 1)

    # 转 RGB
    frame = np.stack([ch, ch, ch], axis=2)

    # 叠加检测
    overlay = np.zeros((*ch.shape, 4), dtype=np.float32)
    points = np.argwhere(detection_mask)
    for y, x in points:
        if 0 <= y < ch.shape[0] and 0 <= x < ch.shape[1]:
            cv2.circle(overlay, (x, y), 2, (1.0, 0.0, 0.0, 0.7), -1)

    frame_out = frame * (1 - overlay[:, :, 3:]) + overlay[:, :, :3] * overlay[:, :, 3:]

    # 加文字
    wl = TARGET_WAVELENGTHS[band_idx] if band_idx < len(TARGET_WAVELENGTHS) else band_idx
    text = f"Frame {frame_idx}  {wl}nm"
    if score is not None:
        text += f"  Score={score:.3f}"

    frame_8u = (np.clip(frame_out, 0, 1) * 255).astype(np.uint8)
    cv2.putText(frame_8u, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return frame_8u


# ============================================================
# 保存结果
# ============================================================

def save_pseudo_color(
    rgb: np.ndarray,
    save_path: str,
    bit_depth: int = 8,
) -> str:
    """
    保存伪彩色 RGB 图像。

    参数:
        rgb: (H, W, 3) float32 [0,1]
        save_path: 保存路径 (支持 .png/.jpg/.tif)
        bit_depth: 位深度 (8 或 16)

    返回:
        save_path
    """
    save_path = str(save_path)
    if bit_depth == 16:
        img_16u = (np.clip(rgb, 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(save_path, img_16u)
    else:
        img_8u = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(save_path, img_8u)

    print(f"✅ 伪彩色图像已保存: {save_path}")
    return save_path


def save_overlay_image(
    base_image: np.ndarray,
    detection_mask: np.ndarray,
    save_path: str,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> str:
    """
    保存带检测标注的叠加图像。

    参数:
        base_image: (H, W) 或 (H, W, 3) uint8/uint16/float32
        detection_mask: (H, W) 布尔掩膜
        save_path: 保存路径
        color: BGR 颜色 (默认蓝色)
        alpha: 透明度

    返回:
        save_path
    """
    save_path = str(save_path)

    # 确保 uint8
    if base_image.dtype != np.uint8:
        if base_image.dtype == np.uint16:
            img = (base_image / 257).astype(np.uint8)
        elif base_image.max() <= 1.0:
            img = (base_image * 255).astype(np.uint8)
        else:
            img = base_image.astype(np.uint8)
    else:
        img = base_image

    # 单通道 → 3 通道
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_rgb = img.copy()

    # 创建彩色覆盖层
    overlay = img_rgb.copy()
    overlay[detection_mask] = color

    # 融合
    result = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)

    cv2.imwrite(save_path, result)
    print(f"✅ 标注图像已保存: {save_path}")
    return save_path


# ============================================================
# 命令行工具
# ============================================================

def visualize_from_npy(
    data_path: str,
    detection_results: Optional[Dict] = None,
    save_dir: str = ".",
    prefix: str = "detection",
    show: bool = False,
):
    """
    从 .npy 数据文件和检测结果生成可视化。

    参数:
        data_path: .npy 数据文件路径 (N, B)
        detection_results: 检测结果字典，需包含:
            - 'scores': (N,) 检测分数
            - 'binary': (N,) 布尔检测结果
            - 'coords': (N, 2) 坐标 [y, x] (可选)
        save_dir: 保存目录
        prefix: 文件名前缀
        show: 是否显示
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_path)
    print(f"📊 加载数据: {data.shape}")

    if detection_results is None:
        print("⚠️ 未提供检测结果，仅保存光谱曲线")
        from .visualization import plot_spectral_curves
        plot_spectral_curves(data[:10], save_path=str(save_dir / f"{prefix}_spectra.png"))
        return

    scores = detection_results.get("scores", None)
    binary = detection_results.get("binary", None)
    coords = detection_results.get("coords", None)

    # 绘制分数直方图
    if scores is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(scores, bins=50, alpha=0.7, color="steelblue")
        if binary is not None:
            ax.axvline(x=scores[binary].min() if binary.any() else 0,
                       color="red", linestyle="--", label="检测阈值")
        ax.set_xlabel("检测分数")
        ax.set_ylabel("频数")
        ax.set_title(f"检测分数分布 ({prefix})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(save_dir / f"{prefix}_score_hist.png"), dpi=150)
        print(f"✅ 分数直方图已保存")
        if show:
            plt.show()
        else:
            plt.close()

    # 绘制 top-N 光谱
    if scores is not None and data.ndim == 2:
        n_top = min(20, len(data))
        top_idx = np.argsort(scores)[-n_top:]
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in top_idx:
            ax.plot(data[i], color="red", alpha=0.1)
        ax.plot(data[np.argsort(scores)[-1]], color="red", linewidth=2, label="Top-1")
        ax.set_xlabel("波段")
        ax.set_ylabel("强度")
        ax.set_title(f"Top-{n_top} 检测像素光谱")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(save_dir / f"{prefix}_top_spectra.png"), dpi=150)
        print(f"✅ Top光谱已保存")
        if show:
            plt.show()
        else:
            plt.close()

    if coords is not None and data.ndim == 2:
        print(f"  📍 坐标数据: {coords.shape}")


# ============================================================
# 运行示例
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("🔍 检测可视化工具 — 用法示例")
    print("=" * 50)
    print()
    print("从 Python 代码使用:")
    print()
    print("  from utils.detection_visualization import (")
    print("      compute_pseudo_rgb,")
    print("      compute_visible_grayscale,")
    print("      overlay_detections,")
    print("      plot_detection_summary,")
    print("      save_overlay_image,")
    print("  )")
    print()
    print("  # 1. 从高光谱立方体生成伪彩色 RGB")
    print('  rgb = compute_pseudo_rgb(hyperspectral_cube,')
    print("      r_wavelength=650, g_wavelength=550, b_wavelength=450)")
    print()
    print("  # 2. 生成可见光灰度图")
    print("  gray = compute_visible_grayscale(hyperspectral_cube)")
    print()
    print("  # 3. 叠加检测结果")
    print("  overlay = overlay_detections(gray, detection_mask=mask)")
    print()
    print("  # 4. 保存")
    print('  save_overlay_image(gray, mask, "result.png")')
    print()
    print("命令行脚本:")
    print("  python scripts/color_detection.py --help")
