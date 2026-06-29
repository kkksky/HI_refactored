"""
Streamlit 光谱查看器

基于旧代码 hi_gui_optimized_streamlit.py 和 hi_streamlit.py 整合重构。
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

from config import TARGET_WAVELENGTHS


def load_coords(json_path: str) -> dict:
    """加载坐标字典。"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    coords = {}
    for key, value in raw.items():
        try:
            k = tuple(map(int, key.strip("()").split(", ")))
            coords[k] = value
        except (ValueError, TypeError):
            continue
    return coords


def find_nearest(y: int, x: int, coords: dict) -> Optional[tuple]:
    """查找最近的坐标点。"""
    if not coords:
        return None
    min_dist = float("inf")
    nearest = None
    for cy, cx in coords.keys():
        dist = (cy - y) ** 2 + (cx - x) ** 2
        if dist < min_dist:
            min_dist = dist
            nearest = (cy, cx)
    return nearest


def streamlit_app():
    """Streamlit 光谱查看器。"""
    st.set_page_config(page_title="高光谱数据查看器", layout="wide")
    st.title("🔬 高光谱数据查看器")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("图像")
        # 在实际使用时替换为真实图像路径
        img_path = st.text_input("图像路径", value="view2.tif")
        try:
            from PIL import Image
            img = Image.open(img_path)
            st.image(img, caption="背景图像", use_column_width=True)
        except (FileNotFoundError, OSError):
            st.warning(f"无法加载图像: {img_path}")

    with col2:
        st.subheader("光谱分析")

        coords_path = st.text_input("坐标字典路径", value="coords_dict.json")
        coords = {}
        try:
            coords = load_coords(coords_path)
            st.success(f"已加载 {len(coords)} 个采样点")
        except (FileNotFoundError, json.JSONDecodeError):
            st.warning(f"无法加载坐标: {coords_path}")

        # 坐标输入
        y = st.number_input("Y 坐标", value=1024, min_value=0, max_value=2047)
        x = st.number_input("X 坐标", value=1024, min_value=0, max_value=2047)

        if st.button("查询"):
            nearest = find_nearest(y, x, coords)
            if nearest:
                st.success(f"最近采样点: {nearest}")
                spectrum = coords[nearest]

                # 绘制光谱
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 4))
                channels = [s[0] for s in spectrum]
                values = [s[1] for s in spectrum]
                ax.plot(channels, values, "-o", markersize=3)
                ax.set_xlabel("波段")
                ax.set_ylabel("灰度值")
                ax.set_title(f"光谱曲线 - 坐标 {nearest}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.error("未找到附近的采样点")


if __name__ == "__main__":
    streamlit_app()
