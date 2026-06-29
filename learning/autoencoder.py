"""
光谱自编码器训练与异常检测

使用自编码器（仅用正常光谱训练）进行异常检测。
正常光谱的重建误差小，异常光谱的重建误差大。

修复:
    - 旧代码 AutoEncoder.py 中模型保存逻辑错误：当 loss 未改善时不更新 loss_last
    - old: else 分支无条件覆盖 loss_last → 不能正确"最佳"模型
    - new: 仅在 loss 改善时保存，并且始终跟踪最佳 loss
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import SpectralAE
from .dataset import PairSpectralDataset, get_background_data
from config import AE_EPOCHS, AE_EMBEDDING_DIM, AE_INPUT_DIM, BATCH_SIZE_TRAIN, DEVICE


def train_autoencoder(
    data: Optional[np.ndarray] = None,
    input_dim: int = AE_INPUT_DIM,
    emb_dim: int = AE_EMBEDDING_DIM,
    epochs: int = AE_EPOCHS,
    batch_size: int = BATCH_SIZE_TRAIN,
    lr: float = 1e-3,
    save_best: str = "ae_best_model.pth",
    save_last: str = "ae_last_model.pth",
    device: str = DEVICE,
) -> SpectralAE:
    """
    训练光谱自编码器。

    仅使用"正常"光谱训练。模型学会重建正常光谱；
    异常光谱的重建误差会很大，从而实现异常检测。

    参数:
        data: (N, input_dim) 训练光谱数据，为 None 时从 background.npy 加载
        input_dim: 输入维度
        emb_dim: 嵌入维度
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        save_best: 最佳模型保存路径
        save_last: 最新模型保存路径
        device: 计算设备

    返回:
        训练好的 SpectralAE 模型
    """
    if data is None:
        data = get_background_data()[:, :input_dim]

    dataset = PairSpectralDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SpectralAE(input_dim=input_dim, emb_dim=emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    print("🚀 开始训练自编码器...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            recon = model(x)
            loss = criterion(recon, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}")

            # === 修复: 正确的模型保存逻辑 ===
            # 旧代码 bug: else 分支无条件赋值 loss_last (BUG!)
            # 修复: 仅在 loss 改善时才保存并更新 best_loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_best)
                print(f"  ✅ 保存最佳模型 (loss={best_loss:.6f})")

            # 定期保存最新模型
            torch.save(model.state_dict(), save_last)

    print(f"🏁 训练完成！最佳 Loss: {best_loss:.6f}")
    return model


def compute_anomaly_score(
    model: SpectralAE,
    data: torch.Tensor,
) -> torch.Tensor:
    """
    计算异常分数（重建误差）。

    参数:
        model: 训练好的自编码器
        data: (N, D) 待检测光谱

    返回:
        scores: (N,) 每个样本的 MSE 重建误差
    """
    model.eval()
    with torch.no_grad():
        recon = model(data)
        scores = ((data - recon) ** 2).mean(dim=1)
    return scores


def get_anomaly_threshold(
    model: SpectralAE,
    loader: DataLoader,
    percentile: float = 99.0,
    device: str = DEVICE,
) -> float:
    """
    计算异常检测阈值。

    使用正常数据的重建误差分布，取指定百分位数作为阈值。

    参数:
        model: 训练好的自编码器
        loader: 正常数据的 DataLoader
        percentile: 百分位数（默认 99）
        device: 计算设备

    返回:
        threshold: 异常检测阈值
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            errors = compute_anomaly_score(model, x)
            all_errors.extend(errors.cpu().numpy())

    threshold = float(np.percentile(all_errors, percentile))
    print(f"📊 异常检测阈值 (P{percentile}): {threshold:.6f}")
    return threshold
