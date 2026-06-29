"""
神经网络模型定义

包含三种光谱特征提取网络:
1. SpectralEmbeddingNet — MLP 光谱嵌入网络
2. SpectralAE — 光谱自编码器
3. OneDCNN — 一维卷积神经网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEmbeddingNet(nn.Module):
    """
    光谱嵌入网络。

    结构: Linear(93→64→32→16) + ReLU + L2 Normalization
    用于对比学习和三元组训练。

    参数:
        input_dim: 输入光谱维度 (默认 93)
        emb_dim: 嵌入维度 (默认 32)
    """

    def __init__(self, input_dim: int = 93, emb_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class SpectralAE(nn.Module):
    """
    光谱自编码器。

    结构:
        Encoder: Linear(93→64→32→16) + ReLU
        Decoder: Linear(16→32→64→93) + ReLU

    用于异常检测：正常光谱重建误差小，异常光谱重建误差大。

    参数:
        input_dim: 输入光谱维度 (默认 93)
        emb_dim: 嵌入维度 (默认 16)
    """

    def __init__(self, input_dim: int = 93, emb_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, emb_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class OneDCNN(nn.Module):
    """
    1D CNN 光谱嵌入网络。

    使用 1D 卷积提取光谱的局部结构特征。

    参数:
        input_dim: 输入光谱维度
        emb_dim: 嵌入维度
    """

    def __init__(self, input_dim: int = 93, emb_dim: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # 计算卷积后的特征长度
        self._conv_out_dim = self._compute_conv_out(input_dim)
        self.fc = nn.Linear(self._conv_out_dim, emb_dim)

    def _compute_conv_out(self, input_dim: int) -> int:
        """计算卷积后的特征维度。"""
        dummy = torch.zeros(1, 1, input_dim)
        out = self.conv(dummy)
        return out.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) → (B, 1, input_dim)
        x = x.unsqueeze(1)
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        emb = self.fc(features)
        return F.normalize(emb, p=2, dim=1)
