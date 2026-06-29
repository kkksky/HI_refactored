"""
深度学习模块

提供多种深度学习方法用于光谱数据分析:
- 光谱嵌入网络 (SpectralEmbeddingNet, 1D-CNN)
- 光谱自编码器 (Autoencoder) 用于异常检测
- SimCLR 对比学习和 InfoNCE 对比学习

参考论文:
    - 对比学习: Chen et al. SimCLR (ICML 2020)
    - 自编码器: Hinton & Salakhutdinov (Science 2006)
"""

from .models import SpectralEmbeddingNet, SpectralAE, OneDCNN
from .dataset import TripletDataset, SpectralDataset
from .autoencoder import train_autoencoder, compute_anomaly_score
from .contrastive_simclr import SimCLR
from .contrastive_infonce import InfoNCEContrastive

__all__ = [
    "SpectralEmbeddingNet",
    "SpectralAE",
    "OneDCNN",
    "TripletDataset",
    "SpectralDataset",
    "train_autoencoder",
    "compute_anomaly_score",
    "SimCLR",
    "InfoNCEContrastive",
]
