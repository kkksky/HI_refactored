"""
IO 工具函数

提供 JSON 读写、Labelme → Mask 转换等通用 IO 功能。
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


def json_to_mask(
    json_path: str,
    img_shape: Tuple[int, int] = (2048, 2048),
    categories: Optional[List[str]] = None,
) -> np.ndarray:
    """
    将 labelme JSON 标注转换为多类别掩膜。

    参数:
        json_path: labelme 生成的 JSON 路径
        img_shape: (H, W) 输出图像尺寸
        categories: 类别名称列表

    返回:
        mask_matrix: (H, W, N) 多类别掩膜
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if categories is None:
        categories = sorted(set(shape["label"] for shape in data["shapes"]))

    num_classes = len(categories)
    mask_matrix = np.zeros((img_shape[0], img_shape[1], num_classes), dtype=np.uint8)
    cat_to_id = {cat: i for i, cat in enumerate(categories)}

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in cat_to_id:
            continue

        class_id = cat_to_id[label]
        points = shape["points"]

        mask = Image.new("L", (img_shape[1], img_shape[0]), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(points, outline=1, fill=1)
        mask_matrix[:, :, class_id] = np.logical_or(
            mask_matrix[:, :, class_id], np.array(mask)
        )

    return mask_matrix


def save_coords(coords_dict: dict, json_path: str):
    """
    保存坐标字典为 JSON（确保所有类型可序列化）。

    参数:
        coords_dict: {(y, x): [(channel, y, x), ...]}
        json_path: 保存路径
    """
    serializable = {}
    for key, value in coords_dict.items():
        str_key = f"({int(key[0])}, {int(key[1])})"
        serializable[str_key] = [
            [int(c), int(y), int(x)] for c, y, x in value
        ]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_coords(json_path: str) -> dict:
    """
    从 JSON 加载坐标字典。

    参数:
        json_path: JSON 文件路径

    返回:
        coords_dict: {(y, x): [(channel, y, x), ...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    coords_dict = {}
    for str_key, value in raw.items():
        key = tuple(map(int, str_key.strip("()").split(", ")))
        coords_dict[key] = [tuple(map(int, v)) for v in value]

    return coords_dict
