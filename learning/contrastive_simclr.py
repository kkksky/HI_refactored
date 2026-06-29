"""
SimCLR 对比学习

使用 SimCLR 风格的对比学习训练光谱嵌入。
对每条光谱施加数据增强（噪声、缩放、平移），
最大化同一条光谱不同增强版本之间的相似度。

修复:
    - 旧代码 `对比学习.py` 中的 _contrastive_loss_ 实现错误：
      使用 F.cross_entropy(logits, labels) 始终以 0 为标签
      正确实现应从 positives 生成正标签
    - 新代码: 标准的 NT-Xent loss 实现

参考:
    - Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" (ICML 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import SpectralEmbeddingNet
from config import DEVICE, CONTRASTIVE_TEMPERATURE, BATCH_SIZE_TRAIN, EMBEDDING_DIM


class SimCLR(nn.Module):
    """
    SimCLR 对比学习。

    参数:
        input_dim: 输入光谱维度
        emb_dim: 嵌入维度
        temperature: NT-Xent loss 的温度参数
    """

    def __init__(
        self,
        input_dim: int = 93,
        emb_dim: int = EMBEDDING_DIM,
        temperature: float = CONTRASTIVE_TEMPERATURE,
    ):
        super().__init__()
        self.encoder = SpectralEmbeddingNet(input_dim, emb_dim)
        self.temperature = temperature

    @staticmethod
    def augment(x: torch.Tensor) -> torch.Tensor:
        """
        光谱数据增强。

        包括:
            1. 加性高斯噪声
            2. 随机缩放
            3. 随机平移（roll + 小幅度）

        参数:
            x: (B, D) 光谱批数据

        返回:
            augmented: (B, D) 增强后的光谱
        """
        noise = torch.randn_like(x) * 0.01
        scale = torch.rand(x.size(0), 1, device=x.device) * 0.1 + 0.95
        shift = torch.roll(x, 1, dims=1) * 0.01
        return x * scale + noise + shift

    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent 损失 (Normalized Temperature-scaled Cross Entropy)。

        === 修复: 旧代码的 bug ===
        旧代码 `对比学习.py` 中错误使用 labels=torch.zeros(...) 作为所有样本的标签。
        正确实现:
            - logits[i, j] 表示样本 i 与样本 j 的相似度
            - 正样本对是 (i, i+B) 和 (i+B, i)
            - 标签应为 torch.arange(B, 2*B) 和 torch.arange(0, B)

        参数:
            z1: (B, D) 第一批增强的嵌入
            z2: (B, D) 第二批增强的嵌入

        返回:
            loss: 标量损失
        """
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        # 相似度矩阵
        sim = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # 移除自身对比
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim[~mask].view(2 * batch_size, -1)

        # === 正确标签: 正样本在另一个 batch 中的对应位置 ===
        # 对于 z1[i], 它的正样本是 z2[i] (位置在 batch_size + i)
        # 对于 z2[i], 它的正样本是 z1[i] (位置在 i)
        #
        # 注意：移除自身对比（对角线）后，正样本索引需左移：
        #   - 若正样本在原矩阵中位于对角线右侧（i < B），
        #     移除 column i 后索引减 1
        #   - 若正样本在原矩阵中位于对角线左侧（i >= B），
        #     移除 column i 不影响，索引不变
        labels = torch.cat([
            torch.arange(batch_size - 1, 2 * batch_size - 1, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ])

        return F.cross_entropy(sim, labels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return self.nt_xent_loss(z1, z2)

    def train_model(
        self,
        train_loader: DataLoader,
        epochs: int = 300,
        lr: float = 1e-3,
        device: str = DEVICE,
    ):
        """
        训练 SimCLR 模型。

        参数:
            train_loader: 训练数据 DataLoader
            epochs: 训练轮数
            lr: 学习率
            device: 计算设备
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for x in train_loader:
                x = x.to(device)

                x1 = self.augment(x)
                x2 = self.augment(x)

                z1 = self.encoder(x1)
                z2 = self.encoder(x2)

                loss = self.nt_xent_loss(z1, z2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"SimCLR Epoch {epoch + 1}: Loss = {total_loss:.4f}")

        return self
