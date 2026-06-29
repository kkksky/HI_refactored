#!/usr/bin/env python3
"""
运行完整的高光谱数据处理流水线。

用法:
    python run_pipeline.py --data <tif_folder> --calibration <calib.json> [--scene 2] [--method SACE]
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import SpectralPipeline


def main():
    parser = argparse.ArgumentParser(description="高光谱数据处理流水线")
    parser.add_argument("--data", type=str, required=True,
                        help="TIF 图像文件夹路径")
    parser.add_argument("--calibration", type=str, required=True,
                        help="标定字典 JSON 路径")
    parser.add_argument("--scene", type=int, default=2, choices=[1, 2],
                        help="场景类型 (1=新样机, 2=旧样机)")
    parser.add_argument("--method", type=str, default="SACE",
                        choices=["CEM", "ACE", "MTICEM", "SACE", "SAM"],
                        help="目标检测方法")
    parser.add_argument("--no-gpu", action="store_true",
                        help="禁用 GPU")
    parser.add_argument("--reflectance", action="store_true",
                        help="使用反射率模式（默认响应图）")

    args = parser.parse_args()

    pipeline = SpectralPipeline(scene=args.scene)

    results = pipeline.run_all(
        data_folder=args.data,
        calibration_json=args.calibration,
        method=args.method,
        use_gpu=not args.no_gpu,
        gene_sel=2 if args.reflectance else 1,
    )

    print("\n" + "=" * 50)
    print("流水线完成！")
    print(f"数据形状: {results['data_shape']}")
    print(f"轨迹数量: {results['num_trajectories']}")
    print(f"光谱矩阵: {results['spectra_shape']}")
    det = results["detection"]
    print(f"检测方法: {det['method']}")
    print(f"检测像素: {det['num_detections']}/{det['total_pixels']}")
    print(f"有效区域: {det.get('valid_regions', 'N/A')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
