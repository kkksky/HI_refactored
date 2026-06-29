"""
光谱重建模块

实现从原始多波段图像到光谱数据的重建，包括:
- 棱镜色散模型 (Prism Dispersion Model) — 基于柯西色散公式的像素偏移标定
- 光谱传播 (Spectral Propagation) — 从稀疏采样到全分辨率光谱的加权重建
"""

from .prism_dispersion import calibrate_prism_dispersion, reconstruct_spectrum_fast
from .spectral_propagation import SpectralPropagator

__all__ = [
    "calibrate_prism_dispersion",
    "reconstruct_spectrum_fast",
    "SpectralPropagator",
]
