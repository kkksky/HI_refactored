"""
InfoNCE 对比学习 + 马氏距离异常检测

结合 InfoNCE 对比学习和概率建模：
    1. 使用 InfoNCE loss 学习光谱嵌入
    2. 在嵌入空间拟合背景的高斯分布 (Gaussian fit)
    3. 使用马氏距离 (Mahalanobis distance) 进行异常检测

参考:
    - Oord et al. "Representation Learning with Contrastive Predictive Coding" (2018)
    - 马氏距离: 经典高维异常检测方法

修复:
    - 旧代码 `Contrastive.py` 中 torch.cov() 需要 PyTorch ≥ 2.0
    - 新代码: 增加兼容性处理（torch ≥ 1.x 备选实现）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import SpectralEmbeddingNet
from .dataset import SpectralDataset
from config import DEVICE, CONTRASTIVE_TEMPERATURE, EMBEDDING_DIM


def mahalanobis_distance(
    x: torch.Tensor,
    mean: torch.Tensor,
    inv_cov: torch.Tensor,
) -> torch.Tensor:
    """
    马氏距离计算。

    D²(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)

    参数:
        x: (N, D) 待测数据
        mean: (D,) 分布均值
        inv_cov: (D, D) 协方差矩阵的逆

    返回:
        dist: (N,) 每个点的马氏距离
    """
    diff = x - mean
    return torch.sqrt(torch.sum(diff @ inv_cov * diff, dim=1))


def compute_cov(x: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """
    计算协方差矩阵（兼容 PyTorch < 2.0）。

    === 修复: torch.cov() 兼容性 ===
    - PyTorch ≥ 2.0: 直接使用 torch.cov()
    - PyTorch < 2.0: 手动实现

    参数:
        x: (N, D) 数据矩阵
        reg: 正则化系数（防止奇异）

    返回:
        cov: (D, D) 协方差矩阵
    """
    if hasattr(torch, "cov"):
        # PyTorch ≥ 2.0
        return torch.cov(x.T) + reg * torch.eye(x.size(1), device=x.device)
    else:
        # 手动实现
        mean = x.mean(dim=0, keepdim=True)
        x_centered = x - mean
        cov = (x_centered.T @ x_centered) / (x.size(0) - 1)
        return cov + reg * torch.eye(x.size(1), device=x.device)


class InfoNCEContrastive(nn.Module):
    """
    InfoNCE 对比学习。

    参数:
        input_dim: 输入光谱维度
        emb_dim: 嵌入维度
        temperature: InfoNCE 温度参数
    """

    def __init__(
        self,
        input_dim: int = 93,
        emb_dim: int = EMBEDDING_DIM,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.encoder = SpectralEmbeddingNet(input_dim=input_dim, emb_dim=emb_dim)
        self.temperature = temperature

    @staticmethod
    def augment(x: torch.Tensor) -> torch.Tensor:
        """
        光谱数据增强（numpy 兼容版）。

        参数:
            x: (B, D) 光谱批数据

        返回:
            x1, x2: 两个增强版本
        """
        noise = torch.randn_like(x) * 0.01
        scale = torch.rand(x.size(0), 1, device=x.device) * 0.1 + 0.95
        return x * scale + noise

    def infonce_loss(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE 损失。

        参数:
            z1: (B, D) 增强版本1的嵌入
            z2: (B, D) 增强版本2的嵌入

        返回:
            loss: 标量损失
        """
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim[~mask].view(2 * batch_size, -1)

        # 注意：移除对角线后正样本索引左移（详见 contrastive_simclr.py 注释）
        labels = torch.cat([
            torch.arange(batch_size - 1, 2 * batch_size - 1, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ])

        return F.cross_entropy(sim, labels)

    def fit_gaussian(
        self, loader: DataLoader, device: str = DEVICE
    ):
        """
        在嵌入空间拟合高斯分布。

        参数:
            loader: 背景数据的 DataLoader
            device: 计算设备

        返回:
            mean: (D,) 均值向量
            inv_cov: (D, D) 协方差逆矩阵
        """
        self.eval()
        feats = []

        with torch.no_grad():
            for x in loader:
                x = x.to(device)
                z = self.encoder(x)
                feats.append(z.cpu())

        feats = torch.cat(feats, dim=0)

        mean = feats.mean(dim=0)
        cov = compute_cov(feats)
        inv_cov = torch.inverse(cov)

        return mean, inv_cov

    def evaluate(
        self,
        normal_loader: DataLoader,
        anomaly_loader: DataLoader,
        mean: torch.Tensor,
        inv_cov: torch.Tensor,
        device: str = DEVICE,
    ):
        """
        评估检测性能。

        参数:
            normal_loader: 正常样本 DataLoader
            anomaly_loader: 异常样本 DataLoader
            mean: 高斯均值
            inv_cov: 协方差逆矩阵
            device: 计算设备

        返回:
            normal_scores: 正常样本的马氏距离
            anomaly_scores: 异常样本的马氏距离
        """
        self.eval()

        def get_scores(loader):
            scores = []
            with torch.no_grad():
                for x in loader:
                    x = x.to(device)
                    z = self.encoder(x)
                    dist = mahalanobis_distance(z.cpu(), mean, inv_cov)
                    scores.append(dist)
            return torch.cat(scores)

        normal_scores = get_scores(normal_loader)
        anomaly_scores = get_scores(anomaly_loader)

        return normal_scores, anomaly_scores
