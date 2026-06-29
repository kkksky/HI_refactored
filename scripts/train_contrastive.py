#!/usr/bin/env python3
"""
训练对比学习模型。

用法:
    python train_contrastive.py [--method simclr] [--epochs 300]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import DataLoader

from learning.dataset import SpectralDataset
from learning.contrastive_simclr import SimCLR
from learning.contrastive_infonce import InfoNCEContrastive
from config import DEVICE


def main():
    parser = argparse.ArgumentParser(description="训练对比学习模型")
    parser.add_argument("--method", type=str, default="simclr",
                        choices=["simclr", "infonce"],
                        help="对比学习方法")
    parser.add_argument("--data", type=str, default="background.npy",
                        help="训练数据路径")
    parser.add_argument("--epochs", type=int, default=300,
                        help="训练轮数")
    parser.add_argument("--input-dim", type=int, default=60,
                        help="输入维度")
    parser.add_argument("--emb-dim", type=int, default=32,
                        help="嵌入维度")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="批次大小")
    args = parser.parse_args()

    # 加载数据
    data = np.load(args.data)
    dataset = SpectralDataset(data, input_dim=args.input_dim)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, drop_last=True)

    if args.method == "simclr":
        model = SimCLR(
            input_dim=args.input_dim,
            emb_dim=args.emb_dim,
        )
        model.train_model(loader, epochs=args.epochs, device=DEVICE)
        torch.save(model.state_dict(), "simclr_model.pth")
        print(f"✅ SimCLR 模型已保存为 simclr_model.pth")
    else:
        model = InfoNCEContrastive(
            input_dim=args.input_dim,
            emb_dim=args.emb_dim,
        )
        from torch import optim
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for x in loader:
                x = x.to(DEVICE)
                x1 = model.augment(x)
                x2 = model.augment(x)
                z1 = model.encoder(x1)
                z2 = model.encoder(x2)
                loss = model.infonce_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

        torch.save(model.state_dict(), "infonce_model.pth")
        print(f"✅ InfoNCE 模型已保存为 infonce_model.pth")


if __name__ == "__main__":
    import torch
    main()
