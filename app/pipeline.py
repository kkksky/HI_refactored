"""
高光谱数据处理流水线

整合数据加载、标定、点源检测、轨迹追踪、目标检测等步骤，
提供端到端的处理流程。

参考论文:
    - 综合本项目所有 7 篇论文的处理流程
    - 步骤 1-3: Du 2009, Feng 2014 (硬件采集与标定)
    - 步骤 4-5: Cao 2011, Ma 2014 (点源检测与追踪)
    - 步骤 6: 本项目核心目标检测算法
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage import measure

from config import (
    DETECTION_METHOD,
    BIN_THRESHOLD,
    SCENE_CONFIGS,
    RECT_H,
    RECT_W,
    AREA_THRESHOLD,
    PCA_SEL,
    PCA_COMPONENTS,
    DEVICE,
)
from data import load_hyperspectral_cube
from data.calibration import CalibrationLoader
from detection import (
    process_hyperspectral_cpu,
    process_hyperspectral_gpu,
    get_survival_cube,
    SACEDetector,
    CEMDetector,
    ACEDetector,
    MTICEMDetector,
    SpectralAngleMapper,
)
from utils.band_selection import BandSelector


class SpectralPipeline:
    """
    高光谱数据处理流水线。

    提供端到端的处理流程:
    load → calibrate → point_detection → trajectory → target_detection → visualize

    参数:
        scene: 场景编号 (1=新样机, 2=旧样机)
        device: 计算设备
    """

    def __init__(self, scene: int = 2, device: str = DEVICE):
        self.scene = scene
        self.device = device
        self.config = SCENE_CONFIGS.get(scene, SCENE_CONFIGS[2])

        # 流水线中间结果
        self.hyperspectral_data: Optional[np.ndarray] = None
        self.filtered_data: Optional[np.ndarray] = None
        self.survival_mask: Optional[np.ndarray] = None
        self.coords_dict: Optional[dict] = None
        self.id_to_key: Optional[dict] = None
        self.calibration_loader: Optional[CalibrationLoader] = None
        self.data_vector: Optional[np.ndarray] = None

    def load_data(self, folder_path: str) -> np.ndarray:
        """
        步骤 1: 加载高光谱数据立方体。

        参数:
            folder_path: 包含 TIF 图像序列的文件夹

        返回:
            (H, W, C) 数据立方体
        """
        print("=" * 50)
        print("步骤 1/6: 加载高光谱数据")
        print("=" * 50)
        self.hyperspectral_data = load_hyperspectral_cube(folder_path)
        return self.hyperspectral_data

    def load_calibration(self, json_path: str) -> CalibrationLoader:
        """
        步骤 2: 加载标定数据。

        参数:
            json_path: 标定字典 JSON 路径

        返回:
            CalibrationLoader 实例
        """
        print("=" * 50)
        print("步骤 2/6: 加载标定数据")
        print("=" * 50)
        self.calibration_loader = CalibrationLoader(self.scene, self.config)
        self.calibration_loader.load_calibration_dict(json_path)
        self.calibration_loader.generate_coords()
        return self.calibration_loader

    def detect_points(self, use_gpu: bool = True) -> np.ndarray:
        """
        步骤 3: 点源检测。

        参数:
            use_gpu: 是否使用 GPU 加速

        返回:
            (H, W, C) 检测结果
        """
        print("=" * 50)
        print("步骤 3/6: 点源检测")
        print("=" * 50)
        if self.hyperspectral_data is None:
            raise ValueError("请先调用 load_data()")

        # 去除最后一行（可能是噪声）
        self.hyperspectral_data[2047, :, :] = 0

        if use_gpu and self.device == "cuda":
            self.filtered_data = process_hyperspectral_gpu(self.hyperspectral_data, device=self.device)
        else:
            self.filtered_data = process_hyperspectral_cpu(self.hyperspectral_data)

        return self.filtered_data

    def track_trajectories(self) -> Tuple[np.ndarray, dict, dict]:
        """
        步骤 4: 贯穿目标轨迹追踪。

        返回:
            (survival_mask, coords_dict, id_to_key)
        """
        print("=" * 50)
        print("步骤 4/6: 贯穿目标轨迹追踪")
        print("=" * 50)
        if self.filtered_data is None:
            raise ValueError("请先调用 detect_points()")

        binary = np.where(self.filtered_data > 0, 1, 0).astype(np.uint8)
        self.survival_mask, self.coords_dict, self.id_to_key = get_survival_cube(binary)
        return self.survival_mask, self.coords_dict, self.id_to_key

    def load_and_extract_spectra(
        self, gene_sel: int = 2, sg_flag: bool = False
    ) -> np.ndarray:
        """
        步骤 5: 提取光谱向量。

        参数:
            gene_sel: 1=响应图, 2=反射率
            sg_flag: 是否进行 SG 平滑

        返回:
            (N, B) 光谱向量矩阵
        """
        print("=" * 50)
        print("步骤 5/6: 提取光谱向量")
        print("=" * 50)
        if self.calibration_loader is None:
            raise ValueError("请先调用 load_calibration()")

        img_spec, _, img_sky, _ = self.calibration_loader.load_images()
        self.data_vector = self.calibration_loader.extract_spectral_vectors(
            img_spec, img_sky, gene_sel=gene_sel
        )

        # 异常大值处理
        self.data_vector = self.calibration_loader.remove_outliers(self.data_vector)

        if sg_flag:
            from data.preprocessing import savgolay_smooth
            self.data_vector = savgolay_smooth(self.data_vector, axis=1)

        print(f"✅ 光谱向量提取完成: {self.data_vector.shape}")
        return self.data_vector

    def run_detection(
        self,
        method: str = DETECTION_METHOD,
        band_select: str = "manual",
        use_pca: bool = PCA_SEL,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        步骤 6: 目标检测。

        参数:
            method: 检测方法 ('CEM', 'ACE', 'MTICEM', 'SACE', 'SAM')
            band_select: 波段选择方法
            use_pca: 是否进行 PCA 预处理
            threshold: 二值化阈值

        返回:
            包含检测结果的字典
        """
        print("=" * 50)
        print(f"步骤 6/6: 目标检测 (方法: {method})")
        print("=" * 50)

        if self.data_vector is None:
            raise ValueError("请先调用 load_and_extract_spectra()")

        # 加载标签数据
        label_map_path = self.config["label_map"]
        label_map = tifffile.imread(label_map_path) if label_map_path.endswith('.mat') else None
        _ = label_map  # 供后续使用

        # 加载 first_coords 和 label_map
        first_coords = self.calibration_loader.first_coords

        # 构建目标光谱
        # 加载 label_map_full 来提取目标
        vs_cut = tifffile.imread(self.config["gray"])

        # 简单的目标提取（需根据实际标注数据调整）
        from scipy.io import loadmat
        try:
            mat = loadmat(label_map_path)
            key = [k for k in mat.keys() if 'label_map' in k][0]
            label_map_mat = mat[key]
        except (FileNotFoundError, IndexError, TypeError):
            label_map_mat = None

        M = self.data_vector.astype(np.float32)
        P, B = M.shape

        # 波段选择
        selector = BandSelector(band_select)
        M_sel = selector.select(M)
        B_sel = M_sel.shape[1] if M_sel is not None else B

        # PCA
        if use_pca and B_sel > PCA_COMPONENTS:
            from sklearn.decomposition import PCA as SkPCA
            pca = SkPCA(n_components=PCA_COMPONENTS)
            M_pca = pca.fit_transform(M_sel) if M_sel is not None else pca.fit_transform(M)
        else:
            M_pca = M_sel if M_sel is not None else M

        # 构建目标光谱（简化：使用数据矩阵的一小部分作为目标）
        # 实际使用时应加载真实的目标标注
        target_spectra = M_pca[:3]  # 取前 3 个像素作为示例目标

        # 初始化检测器
        if method == "CEM":
            detector = CEMDetector()
            detector.fit(M_pca, target_spectra[0])
            scores = detector.predict(M_pca)
        elif method == "ACE":
            detector = ACEDetector()
            detector.fit(M_pca, target_spectra[0])
            scores = detector.predict(M_pca)
        elif method == "MTICEM":
            detector = MTICEMDetector()
            detector.fit(M_pca, target_spectra)
            scores = detector.predict_max(M_pca)
        elif method == "SACE":
            detector = SACEDetector()
            detector.fit(M_pca, target_spectra[0])
            scores = detector.predict(M_pca)
        elif method == "SAM":
            detector = SpectralAngleMapper()
            detector.fit(target_spectra[:3])
            # SAM 返回角度，越小越像目标 → 取负作为分数
            scores = -detector.predict(M_pca)
        else:
            raise ValueError(f"未知方法: {method}")

        # 二值化
        th = threshold or BIN_THRESHOLD.get(method, 0.5)
        binary = scores > th

        # 生成 score_map 和 overlay
        H, W = vs_cut.shape[:2] if vs_cut.ndim == 2 else vs_cut.shape[:2]
        Ny_all = first_coords[:, 0].astype(int) if first_coords is not None else None
        Nx_all = first_coords[:, 1].astype(int) if first_coords is not None else None

        result = {
            "scores": scores,
            "binary": binary,
            "num_detections": int(np.count_nonzero(binary)),
            "total_pixels": P,
            "method": method,
            "threshold": th,
        }

        # 生成覆盖图
        if Ny_all is not None and Nx_all is not None:
            valid_mask = (Ny_all < H) & (Nx_all < W)
            Ny_all, Nx_all = Ny_all[valid_mask], Nx_all[valid_mask]
            binary_subset = binary[valid_mask] if len(binary) == len(valid_mask) else binary

            overlay = np.zeros((H, W), dtype=bool)
            detect_coords_y = Ny_all[binary_subset]
            detect_coords_x = Nx_all[binary_subset]
            for y, x in zip(detect_coords_y, detect_coords_x):
                y2 = min(H, y + RECT_H)
                x2 = min(W, x + RECT_W)
                overlay[y:y2, x:x2] = True

            # 连通区域筛选
            labeled = measure.label(overlay, connectivity=2)
            props = measure.regionprops(labeled)
            overlay_filtered = np.zeros((H, W), dtype=bool)
            for prop in props:
                if prop.area >= AREA_THRESHOLD:
                    coords = tuple(prop.coords.T)
                    overlay_filtered[coords] = True

            result["overlay"] = overlay_filtered
            result["valid_regions"] = len(
                [p for p in props if p.area >= AREA_THRESHOLD]
            )

        print(f"✅ 检测完成: {result['num_detections']}/{P} 像素")
        return result

    def save_trajectory(self, output_dir: str = "."):
        """保存轨迹数据为 JSON。"""
        if self.coords_dict is not None and self.id_to_key is not None:
            from utils.io_helpers import save_coords

            save_coords(self.coords_dict, Path(output_dir) / "coords_dict.json")
            with open(Path(output_dir) / "id_to_key.json", "w", encoding="utf-8") as f:
                json.dump(
                    {str(k): v for k, v in self.id_to_key.items()},
                    f,
                    ensure_ascii=False,
                )
            print("✅ 轨迹数据已保存")

    def run_all(
        self,
        data_folder: str,
        calibration_json: str,
        method: str = DETECTION_METHOD,
        use_gpu: bool = True,
        gene_sel: int = 2,
    ) -> Dict:
        """
        运行完整流水线。

        参数:
            data_folder: TIF 数据文件夹
            calibration_json: 标定字典 JSON 路径
            method: 检测方法
            use_gpu: 是否使用 GPU
            gene_sel: 1=响应图 2=反射率

        返回:
            包含所有结果的字典
        """
        self.load_data(data_folder)
        self.detect_points(use_gpu)
        self.track_trajectories()
        self.load_calibration(calibration_json)
        self.load_and_extract_spectra(gene_sel=gene_sel)
        detection_result = self.run_detection(method=method)
        self.save_trajectory()

        return {
            "data_shape": self.hyperspectral_data.shape if self.hyperspectral_data is not None else None,
            "num_trajectories": len(self.coords_dict) if self.coords_dict else 0,
            "spectra_shape": self.data_vector.shape if self.data_vector is not None else None,
            "detection": detection_result,
        }
