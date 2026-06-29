"""
光谱标定数据加载与反射率计算

处理暗电流校正后的光谱数据，根据标定字典提取光谱向量，
支持响应图（gene_sel=1）和反射率（gene_sel=2）两种模式。

参考论文:
    - Ma 2014 HighSpatialSpectral: 反射率计算与光谱标定方法
    - Feng 2014 AmiciPrism: 棱镜系统的光谱标定流程
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from .loader import imread_unicode


class CalibrationLoader:
    """
    标定数据加载器。

    管理标定字典 (calibration_dict) 的加载、光谱坐标生成、
    以及从原始 TIF 图像中提取光谱向量。

    参数:
        scene: 场景编号 (1=新样机, 2=旧样机)
        config: 配置字典，需包含 gray/dark/illuminance/spec_base/calibration 路径
    """

    def __init__(
        self,
        scene: int = 2,
        config: Optional[dict] = None,
    ):
        self.scene = scene
        self.config = config or {}
        self.calibration_dict: Optional[dict] = None
        self.spec_yx: Optional[dict] = None
        self.first_coords: Optional[np.ndarray] = None

    def load_calibration_dict(self, json_path: str) -> dict:
        """加载标定字典 JSON。"""
        with open(json_path, "r", encoding="utf-8") as f:
            self.calibration_dict = json.load(f)
        print(f"✅ 标定字典加载完成，包含 {len(self.calibration_dict)} 个标定点")
        return self.calibration_dict

    def load_images(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载标定所需的 TIF 图像。

        返回:
            (img_spec_base, img_dark_base, img_sky_base, img_gray_base)
        """
        img_gray_base = tifffile.imread(self.config["gray"])
        img_dark_base = tifffile.imread(self.config["dark"])
        img_sky_base = tifffile.imread(self.config["illuminance"])
        img_spec_base = tifffile.imread(self.config["spec_base"])

        # 暗电流校正
        img_spec_base = self._subtract_clip(img_spec_base, img_dark_base)
        img_sky_base = self._subtract_clip(img_sky_base, img_dark_base)

        return img_spec_base, img_dark_base, img_sky_base, img_gray_base

    @staticmethod
    def _subtract_clip(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """暗电流校正：相减并截断负值。"""
        res = img1.astype(np.int32) - img2.astype(np.int32)
        return np.clip(res, 0, None).astype(np.uint16)

    def generate_coords(self, cache: bool = True) -> Tuple[dict, np.ndarray]:
        """
        从标定字典生成光谱坐标映射。

        参数:
            cache: 是否缓存为 .pkl 文件加速后续加载

        返回:
            spec_yx: dict，键为波段索引字符串，值为 (N, 2) 坐标数组
            first_coords: (N, 2) 第一波段的坐标数组
        """
        if self.calibration_dict is None:
            raise ValueError("请先调用 load_calibration_dict() 加载标定数据")

        cache_dir = Path(".")
        spec_yx_path = cache_dir / "spec_yx.pkl"
        first_coords_path = cache_dir / "first_coords.pkl"

        if cache and spec_yx_path.exists() and first_coords_path.exists():
            with open(spec_yx_path, "rb") as f:
                loaded = pickle.load(f)
            with open(first_coords_path, "rb") as f:
                loaded_first = pickle.load(f)
            # 验证缓存数据结构：如果是新格式 (N,2) 则使用，否则重新生成
            if isinstance(loaded, dict):
                first_key = next(iter(loaded))
                arr = loaded[first_key]
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 2:
                    self.spec_yx = loaded
                    self.first_coords = loaded_first
                    return self.spec_yx, self.first_coords
            # 缓存格式不匹配，重新生成
            print("⚠️ 缓存数据结构旧，重新生成...")

        spec_yx: Dict[str, list] = {}
        for _, spec in self.calibration_dict.items():
            for band in spec:
                b_index = str(band[0] + 1)
                if b_index not in spec_yx:
                    spec_yx[b_index] = []
                spec_yx[b_index].append(band[1:])  # [y, x] 坐标对

        # 列表 → numpy
        for key in spec_yx:
            spec_yx[key] = np.array(spec_yx[key])

        self.spec_yx = spec_yx
        self.first_coords = spec_yx.get("1", np.array([]))

        if cache:
            with open(spec_yx_path, "wb") as f:
                pickle.dump(spec_yx, f)
            with open(first_coords_path, "wb") as f:
                pickle.dump(self.first_coords, f)

        return self.spec_yx, self.first_coords

    def extract_spectral_vectors(
        self,
        img_spec_base: np.ndarray,
        img_sky_base: np.ndarray,
        num_bands: int = 93,
        gene_sel: int = 2,
        rmflat_flag: bool = False,
    ) -> np.ndarray:
        """
        从标定图像中提取所有标定点的光谱向量。

        算法思路:
            1. 对每个波段，根据 spec_yx 获取该波段所有标定点的坐标
            2. 根据 gene_sel 选择响应图或反射率模式
            3. 反射率模式：raw / sky，暗电流已校正
            4. 可选平场校正（scene=1 时）

        参数:
            img_spec_base: 暗电流校正后的光谱基底图像
            img_sky_base: 暗电流校正后的天空/参考图像
            num_bands: 波段数 (默认 93)
            gene_sel: 1=响应图, 2=反射率
            rmflat_flag: 是否进行平场校正

        返回:
            data_vector: (N, num_bands) 光谱向量矩阵
        """
        if self.spec_yx is None or self.first_coords is None:
            self.generate_coords()

        num_points = self.first_coords.shape[0]
        eps_i = 1e-6

        if gene_sel == 1:
            data_vector = np.zeros((num_points, num_bands), dtype=np.uint16)
        else:
            data_vector = np.zeros((num_points, num_bands), dtype=np.float32)

        # 平场系数（仅 scene=1）
        gama_all = None
        if self.scene == 1 and gene_sel == 2 and rmflat_flag:
            gama_all = self._compute_flatfield_coefficients(num_bands)

        for b in range(1, num_bands + 1):
            key = str(b)
            if key not in self.spec_yx:
                data_vector[:, b - 1] = 0
                continue

            coords_b = self.spec_yx[key]  # (M, 2) — [y, x]
            if coords_b.shape[0] != num_points:
                data_vector[:, b - 1] = 0
                continue

            y_idx = coords_b[:, 0].astype(int)
            x_idx = coords_b[:, 1].astype(int)

            if gene_sel == 1:
                data_vector[:, b - 1] = img_spec_base[y_idx, x_idx]
            else:
                raw_vals = img_spec_base[y_idx, x_idx].astype(np.float64)
                sky_vals = img_sky_base[y_idx, x_idx].astype(np.float64)

                if gama_all is not None:
                    sky_vals *= gama_all[b - 1]

                # 防零
                sky_vals[sky_vals == 0] = eps_i
                raw_vals[raw_vals == 0] = eps_i

                refl = raw_vals / sky_vals
                refl[~np.isfinite(refl)] = 0
                data_vector[:, b - 1] = refl.astype(np.float32)

        return data_vector

    @staticmethod
    def _compute_flatfield_coefficients(num_bands: int) -> np.ndarray:
        """计算平场校正系数（PCHIP 插值）。"""
        gama_idx = np.array([8, 9, 10, 11, 12, 13, 17, 26, 34, 51, 53, 55, 58, 61])
        gama_val = np.array([1.8, 1.2, 1, 0.8, 1.6, 0.3, 0.2, 0.3, 0.4, 0.3, 0.1, 0.08, 0.07, 0.06])
        bands = np.arange(1, num_bands)
        gama_all = np.full(bands.shape, np.nan)
        interp_range = np.arange(8, 75)

        pchip = PchipInterpolator(gama_idx, gama_val)
        gama_all[interp_range - 1] = pchip(interp_range)
        gama_all[0:7] = gama_val[0]
        gama_all[74:91] = 0
        gama_all[np.isnan(gama_all)] = 0

        return gama_all

    def remove_outliers(self, data_vector: np.ndarray) -> np.ndarray:
        """异常大值处理（对数域离群值过滤）。"""
        X = np.abs(data_vector)
        valid = X > 0
        if not np.any(valid):
            return data_vector

        ref = np.median(np.log10(X[valid]))
        with np.errstate(divide="ignore"):
            log_X = np.log10(X)
        log_diff = np.abs(log_X - ref)
        mask_err = (log_diff >= 1) & valid
        data_vector[mask_err] = 0
        return data_vector
