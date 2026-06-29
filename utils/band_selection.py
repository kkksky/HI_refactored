"""
波段选择算法

提供多种高光谱波段选择方法，用于降维和关键波段识别。

支持的算法:
    - ECA: 欧氏距离约束算法
    - EFDPCF: 增强快速密度峰值聚类
    - FVGBS: 快速体积梯度波段选择
    - MNBS: 最小噪声波段选择
    - OPBS: 正交投影波段选择
    - 手动选择: 根据经验选定

参考:
    - MATLAB 实现: codes/band_sel_test.m, codes/Band_Selection/
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class BandSelector:
    """
    波段选择器。

    参数:
        method: 选择方法 ('eca', 'efdpcf', 'fvgbs', 'mnbs', 'opbs', 'manual', 'none')
    """

    # 预定义波段选择（波长 nm → 转换为波段索引）
    PRESET_BANDS: Dict[str, List[int]] = {
        "eca": [695, 785, 790, 795, 800, 805, 810, 815, 820, 825],
        "efdpcf": [490, 540, 570, 630, 660, 740, 765, 845, 850, 865],
        "fvgbs": [515, 535, 640, 695, 715, 760, 795, 815, 830, 850],
        "mnbs": [485, 505, 550, 590, 625, 645, 680, 715, 755, 875],
        "opbs": [510, 550, 635, 695, 720, 760, 790, 815, 835, 855],
        "manual": [490, 505, 550, 625, 630, 665, 675, 755, 760, 850],
    }

    def __init__(self, method: str = "manual"):
        if method not in self.PRESET_BANDS and method != "none":
            raise ValueError(
                f"未知方法: {method}. 可选: {list(self.PRESET_BANDS.keys()) + ['none']}"
            )
        self.method = method

    @staticmethod
    def wavelength_to_index(wavelengths: List[int], start_nm: int = 440) -> np.ndarray:
        """
        波长 (nm) → 波段索引 (0-based)。

        参数:
            wavelengths: 波长列表
            start_nm: 起始波长（默认 440nm）

        返回:
            indices: 0-based 波段索引
        """
        return ((np.array(wavelengths) - start_nm) // 5 - 1).astype(int)

    def get_selected_bands(self, num_bands: int = 93) -> Optional[np.ndarray]:
        """
        获取选中的波段索引。

        参数:
            num_bands: 总波段数

        返回:
            indices: (K,) 选中的波段索引，method='none' 返回 None
        """
        if self.method == "none":
            return None
        bands = self.PRESET_BANDS[self.method]
        indices = self.wavelength_to_index(bands)
        return indices[(indices >= 0) & (indices < num_bands)]

    def select(self, data: np.ndarray) -> np.ndarray:
        """
        对光谱数据进行波段选择。

        参数:
            data: (P, B) 光谱数据

        返回:
            selected: (P, K) 波段选择后的数据（K <= B）
        """
        indices = self.get_selected_bands(data.shape[1])
        if indices is None:
            return data
        return data[:, indices]
