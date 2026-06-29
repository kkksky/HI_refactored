#!/usr/bin/env python3
"""
训练光谱自编码器。

用法:
    python train_autoencoder.py [--data background.npy] [--epochs 50000]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.autoencoder import train_autoencoder


def main():
    parser = argparse.ArgumentParser(description="训练光谱自编码器")
    parser.add_argument("--data", type=str, default="background.npy",
                        help="训练数据路径")
    parser.add_argument("--epochs", type=int, default=50000,
                        help="训练轮数")
    parser.add_argument("--input-dim", type=int, default=93,
                        help="输入维度")
    parser.add_argument("--emb-dim", type=int, default=16,
                        help="嵌入维度")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="批次大小")
    args = parser.parse_args()

    # 加载数据
    import numpy as np
    data = np.load(args.data)

    print(f"🚀 开始训练自编码器")
    print(f"   数据: {data.shape}")
    print(f"   轮数: {args.epochs}")
    print(f"   嵌入维度: {args.emb_dim}")

    model = train_autoencoder(
        data=data[:, :args.input_dim],
        input_dim=args.input_dim,
        emb_dim=args.emb_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("✅ 训练完成！模型已保存.")


if __name__ == "__main__":
    main()
