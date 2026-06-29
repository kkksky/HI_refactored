"""
高光谱数据加载器

从文件夹中按波长排序加载 TIF 图像序列，拼接为 3D 数据立方体 (H, W, C)。

参考论文:
    - Feng 2014 AmiciPrism: 棱镜色散的高光谱视频采集（数据格式定义）
    - Ma 2014 HighSpatialSpectral: 高空间-光谱分辨率数据获取
"""

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm


def imread_unicode(file_path: str) -> np.ndarray:
    """
    支持中文路径的图像读取。

    使用 OpenCV 的 imdecode 绕过 Windows/Linux 的编码限制。

    参数:
        file_path: 图像文件路径（支持中文和特殊字符）

    返回:
        解码后的 numpy 数组，失败返回 None
    """
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def load_hyperspectral_cube(
    folder_path: str,
    suffix: str = ".tif",
    max_workers: int = 16,
) -> np.ndarray:
    """
    将指定文件夹内的多张 TIF 图片按波长排序并拼接成 3D 矩阵 (H, W, C)。

    文件名应包含波长信息，如 ``445nm.tif``, ``450nm.tif``, ...

    算法思路:
        1. 筛选并排序文件（按 ``xxxnm`` 中的数字）
        2. 预分配连续内存 (H, W, C)
        3. 多线程并行加载（ThreadPoolExecutor）
        4. 检查尺寸一致性，跳过异常文件

    参数:
        folder_path: 图像数据所在文件夹路径
        suffix: 图像文件后缀，默认为 '.tif'
        max_workers: 最大线程数，建议 8-16

    返回:
        mat: 3D 数据立方体 (H, W, C)，失败返回 None
    """
    # 1. 筛选并按波长排序
    try:
        file_names = sorted(
            [f.name for f in os.scandir(folder_path) if f.name.endswith(suffix)],
            key=lambda x: int(x.split("nm")[0]) if "nm" in x else 0,
        )
    except Exception as e:
        print(f"❌ 筛选文件出错: {e}")
        return None

    if not file_names:
        print("⚠️ 未找到匹配的图片文件。")
        return None

    # 2. 预分配内存
    first_path = os.path.join(folder_path, file_names[0])
    first_img = imread_unicode(first_path)
    if first_img is None:
        print(f"❌ 无法读取首张图片: {first_path}")
        return None

    h, w = first_img.shape[:2]
    num_channels = len(file_names)
    mat = np.zeros((h, w, num_channels), dtype=first_img.dtype)

    # 3. 并行加载
    def load_and_place(idx: int):
        try:
            name = file_names[idx]
            full_path = os.path.join(folder_path, name)
            img = imread_unicode(full_path)
            if img is not None:
                if img.shape[:2] == (h, w):
                    mat[:, :, idx] = img
                else:
                    print(f"⚠️ 尺寸不匹配跳过: {name}")
        except Exception as e:
            print(f"❌ 加载 {file_names[idx]} 失败: {e}")

    print(f"🚀 开始加载数据 (共 {num_channels} 个波段)...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(load_and_place, range(num_channels)),
                total=num_channels,
                desc="并行加载高光谱数据",
                unit="张",
            )
        )

    print(f"✅ 成功完成！最终数据形状: {mat.shape}")
    return mat
