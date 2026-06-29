#!/usr/bin/env python3
"""
高光谱目标检测 → 伪彩色可视化流水线。

从高光谱 TIF 序列或 .npy 数据出发:
  1. 生成可见光灰度图 + 伪彩色 RGB
  2. 运行 5 种检测算法 (SAM/CEM/ACE/MTICEM/SACE)
  3. 在可见光图像上标注检测结果 (含色散修正)
  4. 生成综合报告图

用法:
  # 方式 1: 从 TIF 序列 + 标定数据
  python scripts/color_detection.py --data ./tif_folder \\
      --calibration calibration_dict.json --scene 2 --method SACE

  # 方式 2: 从预提取的 .npy 数据
  python scripts/color_detection.py --npy data_vectors.npy --target target.npy

  # 方式 3: 使用合成测试数据
  python scripts/color_detection.py --demo --method CEM

色散修正原理:
  棱镜将不同波长的光分散到传感器不同区域。一个目标在 450nm 波段
  的位置可能在 650nm 波段偏移了数个像素。当把检测结果叠加到
  可见光图像上时，需要根据折射率差异做坐标平移补偿。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt

from config import (
    TARGET_WAVELENGTHS,
    NUM_BANDS,
    SCENE_CONFIGS,
    BIN_THRESHOLD,
    DEVICE,
)
from data import load_hyperspectral_cube
from data.calibration import CalibrationLoader
from detection import (
    CEMDetector, ACEDetector, MTICEMDetector,
    SACEDetector, SpectralAngleMapper,
)
from utils.detection_visualization import (
    compute_pseudo_rgb,
    compute_visible_grayscale,
    overlay_detections,
    create_score_heatmap,
    plot_detection_summary,
    save_pseudo_color,
    save_overlay_image,
    dispersion_correct_points,
    compute_dispersion_offset,
    get_calibration_dispersion_offsets,
    plot_dual_with_dispersion_correction,
)


# ============================================================
# 检测引擎
# ============================================================

def run_detector(method: str, data: np.ndarray, target: np.ndarray):
    """
    运行指定检测器。

    参数:
        method: 'SAM'/'CEM'/'ACE'/'MTICEM'/'SACE'
        data: (P, B) 光谱数据
        target: (B,) 目标光谱

    返回:
        scores: (P,) 检测分数 (越大越像目标)
        binary: (P,) 布尔检测结果
        detector: 检测器实例
    """
    P, B = data.shape
    data_f32 = data.astype(np.float64)
    target_f32 = target.astype(np.float64)

    if method == "SAM":
        det = SpectralAngleMapper(normalize=True)
        det.fit(target_f32.reshape(1, -1))
        scores = -det.predict(data_f32)  # 负角度 → 越大越好
        th = 0.3  # SAM 角度阈值

    elif method == "CEM":
        det = CEMDetector(reg=1e-6)
        det.fit(data_f32, target_f32)
        scores = det.predict(data_f32)
        th = BIN_THRESHOLD.get("SACE", 0.5) if scores.max() < 3 else BIN_THRESHOLD.get("MTICEM", 1.0)

    elif method == "ACE":
        det = ACEDetector(reg=1e-6)
        det.fit(data_f32, target_f32)
        scores = det.predict(data_f32)
        th = 0.5

    elif method == "MTICEM":
        det = MTICEMDetector(reg=1e-6)
        det.fit(data_f32, target_f32.reshape(1, -1))
        scores = det.predict_max(data_f32)
        th = BIN_THRESHOLD.get("MTICEM", 1.0)

    elif method == "SACE":
        det = SACEDetector(reg=1e-6, use_nnls=False)
        det.fit(data_f32, target_f32)
        scores = det.predict(data_f32)
        th = BIN_THRESHOLD.get("SACE", 0.7)

    else:
        raise ValueError(f"未知检测方法: {method}")

    # 自适应阈值: 用 mean + 2*std 作为默认
    score_mean = scores.mean()
    score_std = scores.std()
    adaptive_th = max(th, score_mean + 2.0 * score_std)

    binary = scores > adaptive_th
    if not binary.any():
        # 退化: 使用 top-5% 作为检测
        top_k = max(1, P // 20)
        th_idx = np.argsort(scores)[-top_k:].min()
        th_adaptive = scores[th_idx]
        binary = scores > th_adaptive
        adaptive_th = th_adaptive

    n_det = int(binary.sum())
    print(f"  📊 检测器: {method}")
    print(f"    分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"    阈值: {adaptive_th:.4f}")
    print(f"    检测像素: {n_det}/{P} ({100*n_det/P:.1f}%)")

    return scores, binary, det


# ============================================================
# 目标光谱提取
# ============================================================

def extract_target_spectrum(data: np.ndarray, method: str = "auto"):
    """
    从数据中提取目标光谱。

    策略:
      - 'auto': 取 scores 最高的像素作为目标
      - 'mean_top': 取 top-5% 像素的平均
      - 'first': 取第 1 个像素 (适用于背景已知场景)
    """
    if method == "first":
        return data[0]

    # 先用 CEM 预筛选
    from detection.cem import CEMDetector
    det = CEMDetector(reg=1e-6)
    det.fit(data.astype(np.float64), data[0].astype(np.float64))
    scores = det.predict(data.astype(np.float64))

    if method == "mean_top":
        top_k = max(1, len(data) // 20)
        top_idx = np.argsort(scores)[-top_k:]
        return data[top_idx].mean(axis=0)
    else:
        # 'auto': 最高分像素
        best_idx = np.argmax(scores)
        return data[best_idx]


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="高光谱目标检测 → 伪彩色可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/color_detection.py --demo --method SACE
  python scripts/color_detection.py --npy data.npy --target target.npy
  python scripts/color_detection.py --data ./tif/ --calibration calib.json --scene 2
        """,
    )

    # 数据输入
    parser.add_argument("--data", type=str, default=None,
                        help="TIF 图像文件夹路径")
    parser.add_argument("--calibration", type=str, default=None,
                        help="标定字典 JSON 路径")
    parser.add_argument("--npy", type=str, default=None,
                        help="预提取的光谱数据 .npy 文件")
    parser.add_argument("--target", type=str, default=None,
                        help="目标光谱 .npy 文件")
    parser.add_argument("--demo", action="store_true",
                        help="使用合成数据演示")

    # 参数
    parser.add_argument("--scene", type=int, default=2, choices=[1, 2],
                        help="场景类型 (1=新样机, 2=旧样机)")
    parser.add_argument("--method", type=str, default="SACE",
                        choices=["SAM", "CEM", "ACE", "MTICEM", "SACE"],
                        help="检测方法")
    parser.add_argument("--gene-sel", type=int, default=2, choices=[1, 2],
                        help="1=响应图, 2=反射率")
    parser.add_argument("--output", "-o", type=str, default="output/color_detection",
                        help="输出目录")

    # 可视化选项
    parser.add_argument("--show", action="store_true",
                        help="显示图像 (需要 GUI)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="输出图像分辨率")
    parser.add_argument("--no-color", action="store_true",
                        help="跳过伪彩色生成")
    parser.add_argument("--dispersion-correction", action="store_true",
                        help="启用色散修正")

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🌈 高光谱目标检测 → 伪彩色可视化")
    print("=" * 60)

    # ============================================================
    # 1. 加载数据
    # ============================================================
    hyperspectral_cube = None
    data_vectors = None
    calibration_loader = None
    coords_for_overlay = None
    target_spectrum = None

    if args.demo:
        print("\n📦 使用合成数据演示")
        rng = np.random.RandomState(42)
        H, W, C = 64, 64, 93

        # 合成高光谱立方体
        cube = np.zeros((H, W, C), dtype=np.float32)
        bands = np.arange(C)
        bg_pattern = np.polyval([-1e-5, 0.003, -0.1, 1], bands) + 0.5

        # 添加背景变化
        for y in range(H):
            for x in range(W):
                variation = rng.uniform(0.9, 1.1)
                noise = rng.normal(0, 0.01, C)
                cube[y, x] = bg_pattern * variation + noise

        # 插入目标 (2 个方形区域)
        target_spec = bg_pattern.copy()
        target_spec -= 0.3 * np.exp(-((bands - 25) ** 2) / 40)
        target_spec += 0.25 * np.exp(-((bands - 55) ** 2) / 30)
        target_spec += rng.normal(0, 0.005, C)

        # 目标区域 1
        cube[20:28, 15:23] = target_spec + rng.normal(0, 0.005, (8, 8, C))
        # 目标区域 2 (更小，测试亚像素检测)
        cube[40:45, 40:45] = target_spec * 0.7 + bg_pattern * 0.3 + rng.normal(0, 0.005, (5, 5, C))

        hyperspectral_cube = cube

        # 从立方体提取光谱向量
        data_vectors = cube.reshape(-1, C)
        target_spectrum = target_spec.astype(np.float32)

        print(f"  合成立方体: {cube.shape}")
        print(f"  数据向量: {data_vectors.shape}")
        print(f"  目标区域: (20:28,15:23) + (40:45,40:45)")

    elif args.npy:
        print(f"\n📂 加载光谱数据: {args.npy}")
        data_vectors = np.load(args.npy).astype(np.float32)
        print(f"  数据形状: {data_vectors.shape}")

        if args.target:
            target_spectrum = np.load(args.target).astype(np.float32)
            print(f"  目标光谱: {target_spectrum.shape}")
        else:
            target_spectrum = extract_target_spectrum(data_vectors)
            print(f"  自动提取目标光谱")

    elif args.data:
        print(f"\n📂 加载 TIF 序列: {args.data}")
        hyperspectral_cube = load_hyperspectral_cube(args.data)

        if hyperspectral_cube is None:
            print("❌ 数据加载失败")
            return 1

        if args.calibration:
            print(f"📏 加载标定数据: {args.calibration}")
            config = SCENE_CONFIGS.get(args.scene, SCENE_CONFIGS[2])
            calibration_loader = CalibrationLoader(args.scene, config)
            calibration_loader.load_calibration_dict(args.calibration)
            calibration_loader.generate_coords()

            # 提取光谱向量
            try:
                img_spec, _, img_sky, _ = calibration_loader.load_images()
                data_vectors = calibration_loader.extract_spectral_vectors(
                    img_spec, img_sky, gene_sel=args.gene_sel
                )
                data_vectors = calibration_loader.remove_outliers(data_vectors)
                print(f"  光谱向量: {data_vectors.shape}")
            except Exception as e:
                print(f"  ⚠️ 光谱提取跳过: {e}")

    else:
        parser.print_help()
        print("\n❌ 请指定 --demo, --npy 或 --data")
        return 1

    # ============================================================
    # 2. 生成伪彩色 + 灰度
    # ============================================================
    if hyperspectral_cube is not None:
        print("\n🎨 生成伪彩色图像...")
        rgb = compute_pseudo_rgb(hyperspectral_cube, gamma=0.8)
        gray = compute_visible_grayscale(hyperspectral_cube)

        save_pseudo_color(rgb, str(output_dir / "pseudo_rgb.png"))
        cv2.imwrite(str(output_dir / "visible_gray.png"),
                    (np.clip(gray, 0, 1) * 255).astype(np.uint8))
        print(f"  ✅ 伪彩色 + 灰度已保存")

    # ============================================================
    # 3. 检测
    # ============================================================
    if data_vectors is not None and target_spectrum is not None:
        print(f"\n🎯 运行检测 (方法: {args.method})...")
        scores, binary, detector = run_detector(
            args.method, data_vectors, target_spectrum
        )

        # 保存检测分数
        np.save(str(output_dir / "detection_scores.npy"), scores)
        np.save(str(output_dir / "detection_binary.npy"), binary)

        # --- 在图像上标注 ---
        if hyperspectral_cube is not None and binary is not None:
            H, W = hyperspectral_cube.shape[:2]

            if calibration_loader is not None and args.dispersion_correction:
                # 使用标定数据做色散修正
                spec_yx, first_coords = calibration_loader.generate_coords()
                offsets = get_calibration_dispersion_offsets(spec_yx, first_coords)
                visible_band = 21  # ~550nm 可见光
                det_band = 0       # 标定坐标默认在 band 0

                # 创建检测掩膜（映射到图像空间）
                if data_vectors.shape[0] == len(first_coords):
                    # 标定点位与数据向量一一对应
                    det_mask = np.zeros((H, W), dtype=bool)
                    fy = first_coords[:, 0].astype(int)
                    fx = first_coords[:, 1].astype(int)
                    for i in range(len(fy)):
                        if 0 <= fy[i] < H and 0 <= fx[i] < W and binary[i]:
                            det_mask[fy[i], fx[i]] = True

                    # 色散修正
                    plot_dual_with_dispersion_correction(
                        gray, det_mask,
                        calibration_offsets=offsets,
                        visible_band_idx=visible_band,
                        detection_band_idx=det_band,
                        save_path=str(output_dir / "dispersion_correction.png"),
                        show=args.show,
                    )

                    # 修正坐标
                    dy = offsets[0][det_band] - offsets[0][visible_band]
                    dx = offsets[1][det_band] - offsets[1][visible_band]
                    if abs(dx) > 0.5 or abs(dy) > 0.5:
                        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
                        corrected_mask = cv2.warpAffine(
                            det_mask.astype(np.uint8), M, (W, H)
                        ) > 0
                        print(f"  色散修正: dx={dx:.1f}, dy={dy:.1f}")
                    else:
                        corrected_mask = det_mask
                else:
                    corrected_mask = None
                    print(f"  ⚠️ 数据点与标定点数不匹配，跳过色散修正")
            else:
                corrected_mask = None

            # 标注叠加
            if corrected_mask is not None:
                overlay_mask = corrected_mask
            else:
                overlay_mask = binary  # 在图像上标注所有检测点

            # 如果 binary 是一维数组且我们有高光谱立方体，reshape 成图像
            if binary.ndim == 1 and hyperspectral_cube is not None:
                H_img, W_img = hyperspectral_cube.shape[:2]
                if len(binary) == H_img * W_img:
                    overlay_mask2 = binary.reshape(H_img, W_img)
                else:
                    # 非严格对应，取前 H*W 个元素
                    n_total = H_img * W_img
                    n_use = min(len(binary), n_total)
                    arr = np.zeros(n_total, dtype=bool)
                    arr[:n_use] = binary[:n_use]
                    overlay_mask2 = arr.reshape(H_img, W_img)
            else:
                overlay_mask2 = binary

            # 保存标注图
            if hyperspectral_cube is not None:
                save_overlay_image(
                    gray, overlay_mask2,
                    str(output_dir / f"detection_{args.method}.png"),
                    color=(0, 0, 255), alpha=0.4,
                )
                print(f"  ✅ 检测标注已保存")

            # 分数热力图
            if scores is not None and hyperspectral_cube is not None:
                heatmap = create_score_heatmap(
                    scores[:min(len(scores), H*W)],
                    np.column_stack([
                        np.arange(min(len(scores), H*W)) // W,
                        np.arange(min(len(scores), H*W)) % W,
                    ]),
                    (H, W), radius=4,
                )
                cv2.imwrite(str(output_dir / f"score_heatmap_{args.method}.png"),
                            (heatmap * 255).astype(np.uint8)[:, :, ::-1])
                print(f"  ✅ 热力图已保存")

            # 综合报告图
            if not args.no_color and hyperspectral_cube is not None:
                try:
                    plot_detection_summary(
                        gray,
                        detection_mask=overlay_mask2,
                        scores=scores,
                        pseudo_rgb=rgb if not args.no_color else None,
                        title=f"检测结果: {args.method}",
                        save_path=str(output_dir / "summary.png"),
                        dpi=args.dpi,
                        show=args.show,
                    )
                except Exception as e:
                    print(f"  ⚠️ 综合图生成跳过: {e}")

    # ============================================================
    # 4. 输出报告
    # ============================================================
    report_path = output_dir / "检测结果报告.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 高光谱目标检测结果报告\n\n")
        f.write(f"## 基本信息\n")
        f.write(f"- **检测方法**: {args.method}\n")
        f.write(f"- **波长范围**: {TARGET_WAVELENGTHS[0]}-{TARGET_WAVELENGTHS[-1]}nm\n")
        f.write(f"- **波段数**: {NUM_BANDS}\n\n")
        f.write(f"## 输出文件\n\n")
        f.write(f"| 文件 | 说明 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| `pseudo_rgb.png` | 伪彩色 RGB (R=650, G=550, B=450nm) |\n")
        f.write(f"| `visible_gray.png` | 可见光灰度图像 |\n")
        f.write(f"| `detection_{args.method}.png` | 检测结果标注叠加图 |\n")
        f.write(f"| `score_heatmap_{args.method}.png` | 检测分数热力图 |\n")
        f.write(f"| `summary.png` | 综合检测报告图 |\n")
        f.write(f"| `detection_scores.npy` | 检测分数数据 |\n")
        f.write(f"| `detection_binary.npy` | 二值检测结果 |\n")

        if data_vectors is not None:
            f.write(f"\n## 检测统计\n\n")
            f.write(f"- **总像素数**: {data_vectors.shape[0]}\n")
            f.write(f"- **检测到**: {int(binary.sum()) if binary is not None else 'N/A'}\n")
            f.write(f"- **波段数**: {data_vectors.shape[1]}\n")
            if scores is not None:
                f.write(f"- **分数范围**: [{scores.min():.4f}, {scores.max():.4f}]\n")
                f.write(f"- **分数均值**: {scores.mean():.4f}\n")

    print(f"\n📄 报告已生成: {report_path}")
    print(f"📂 所有输出: {output_dir.resolve()}")
    print("=" * 60)
    print("✅ 完成！")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
