#!/usr/bin/env python3
"""
运行目标检测（使用已提取的光谱数据）。

用法:
    python run_detection.py --data data_vector.npy [--method SACE] [--threshold 0.7]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from detection import (
    CEMDetector, ACEDetector, MTICEMDetector, SACEDetector, SpectralAngleMapper
)


def main():
    parser = argparse.ArgumentParser(description="高光谱目标检测")
    parser.add_argument("--data", type=str, required=True,
                        help="光谱数据 .npy 文件")
    parser.add_argument("--target", type=str, default=None,
                        help="目标光谱 .npy 文件（可选）")
    parser.add_argument("--method", type=str, default="SACE",
                        choices=["CEM", "ACE", "MTICEM", "SACE", "SAM"],
                        help="检测方法")
    parser.add_argument("--threshold", type=float, default=None,
                        help="二值化阈值")
    parser.add_argument("--plot", action="store_true",
                        help="绘制检测结果")

    args = parser.parse_args()

    # 加载数据
    data = np.load(args.data).astype(np.float32)
    P, B = data.shape
    print(f"📊 加载数据: {data.shape}")

    # 构建目标光谱
    if args.target:
        target = np.load(args.target).astype(np.float32)
        target_mean = target.mean(axis=0)
    else:
        # 取前几个像素作为示例目标
        target_mean = data[:3].mean(axis=0)

    # 选择检测器
    method_map = {
        "CEM": CEMDetector,
        "ACE": ACEDetector,
        "MTICEM": MTICEMDetector,
        "SACE": SACEDetector,
    }

    if args.method in method_map:
        detector = method_map[args.method]()
        if args.method == "MTICEM":
            detector.fit(data, data[:3])
            scores = detector.predict_max(data)
        else:
            detector.fit(data, target_mean)
            scores = detector.predict(data)
    elif args.method == "SAM":
        detector = SpectralAngleMapper()
        detector.fit(data[:3])
        scores = -detector.predict(data)

    # 统计
    avg_score = float(np.mean(scores))
    max_score = float(np.max(scores))
    print(f"\n📈 检测分数统计:")
    print(f"   均值: {avg_score:.4f}")
    print(f"   最大值: {max_score:.4f}")

    # 可选绘制
    if args.plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.hist(scores, bins=50, alpha=0.7)
        plt.xlabel("检测分数")
        plt.ylabel("频数")
        plt.title(f"{args.method} 检测分数分布")
        plt.grid(True, alpha=0.3)

        plt.subplot(122)
        # 按分数排序显示前 100 个像素的光谱
        top_idx = np.argsort(scores)[-100:]
        plt.plot(data[top_idx].T, color="red", alpha=0.1)
        plt.plot(target_mean, color="blue", linewidth=2, label="目标平均")
        plt.xlabel("波段")
        plt.ylabel("强度")
        plt.title("Top-100 检测像素光谱")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("✅ 检测完成")


if __name__ == "__main__":
    main()
