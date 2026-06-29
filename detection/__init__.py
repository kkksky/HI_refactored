"""
目标检测模块

提供完整的高光谱目标检测算法链:
- 点源检测: 从 3D 数据立方体中检测点源目标
- 轨迹追踪: 跨波段的贯穿目标轨迹提取
- 经典检测器: SAM, CEM, ACE, MT-ICEM, SACE

参考论文:
    - Du 2009: 点源检测与追踪框架
    - Cao 2011 PAMI: 棱镜-掩膜系统的目标检测
"""

from .point_detection import (
    weighted_window_sum,
    process_hyperspectral_cpu,
    process_hyperspectral_gpu,
    get_gaussian_kernel_2d,
)
from .trajectory import (
    get_survival_cube,
    get_survival_cube_optimized,
    get_survival_cube_gpu,
)
from .sam import SpectralAngleMapper
from .cem import CEMDetector
from .ace import ACEDetector
from .mticem import MTICEMDetector
from .sace import SACEDetector

__all__ = [
    "weighted_window_sum",
    "process_hyperspectral_cpu",
    "process_hyperspectral_gpu",
    "get_gaussian_kernel_2d",
    "get_survival_cube",
    "get_survival_cube_optimized",
    "get_survival_cube_gpu",
    "SpectralAngleMapper",
    "CEMDetector",
    "ACEDetector",
    "MTICEMDetector",
    "SACEDetector",
]
