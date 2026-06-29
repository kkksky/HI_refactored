"""
Tikinter 光谱查看器（整合版）

提供交互式高光谱数据可视化界面：
- 点击查看采样点光谱
- 多选点对比
- 光谱曲线绘制

基于旧代码 hi_gui.py 和 hi_gui_optimized.py 整合重构。
"""

import json
import tkinter as tk
from tkinter import ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk

from config import TARGET_WAVELENGTHS


class SpectralViewer:
    """
    光谱查看器主窗口。

    参数:
        master: tkinter 父窗口
        image_path: 背景图像路径
        coords_path: 坐标字典 JSON 路径
    """

    def __init__(
        self,
        master: tk.Tk,
        image_path: str,
        coords_path: Optional[str] = None,
    ):
        self.master = master
        master.title("高光谱数据查看器")
        master.geometry("1200x800")

        # 加载图像
        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)

        # 加载坐标
        self.coords = {}
        if coords_path:
            self._load_coords(coords_path)

        self.selected_points = []
        self._build_ui()

    def _load_coords(self, path: str):
        """加载坐标字典。"""
        import json
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for key, value in raw.items():
            try:
                k = tuple(map(int, key.strip("()").split(", ")))
                self.coords[k] = value
            except (ValueError, TypeError):
                continue

    def _build_ui(self):
        """构建 UI。"""
        # 主布局
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧图像区
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_frame, width=600, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.bind("<Button-1>", self.on_click)

        # 右侧控制区
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        ttk.Label(right_frame, text="光谱查看器", font=("Arial", 16)).pack(pady=10)
        ttk.Label(right_frame, text="点击左侧图像选择点").pack()

        # 清空按钮
        ttk.Button(right_frame, text="清空选择", command=self.clear_points).pack(pady=5)

        # 光谱曲线图
        self.fig = Figure(figsize=(4, 3), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("波段")
        self.ax.set_ylabel("灰度值")
        self.ax.set_title("光谱曲线")

        self.chart = FigureCanvasTkAgg(self.fig, right_frame)
        self.chart.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 点列表
        self.listbox = tk.Listbox(right_frame, height=8)
        self.listbox.pack(fill=tk.X, padx=5, pady=5)

    def on_click(self, event):
        """图像点击事件。"""
        x, y = event.x, event.y
        self.selected_points.append((y, x))
        self.listbox.insert(tk.END, f"点 {len(self.selected_points)}: ({y}, {x})")

        # 在画布上标记
        r = 3
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            outline="red", width=2,
        )

        # 查找最近坐标
        if self.coords:
            nearest = self._find_nearest(y, x)
            if nearest:
                self.listbox.insert(tk.END, f"  → 最近采样点: {nearest}")

        # 绘制光谱
        if len(self.selected_points) <= 5:
            self._plot_selected_spectra()

    def _find_nearest(self, y: int, x: int) -> Optional[tuple]:
        """查找最近的坐标点。"""
        if not self.coords:
            return None
        min_dist = float("inf")
        nearest = None
        for cy, cx in self.coords.keys():
            dist = (cy - y) ** 2 + (cx - x) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = (cy, cx)
        return nearest

    def _plot_selected_spectra(self):
        """绘制选中点的光谱曲线。"""
        self.ax.clear()
        colors = ["blue", "red", "green", "orange", "purple"]

        for i, (y, x) in enumerate(self.selected_points[:5]):
            color = colors[i % len(colors)]
            # 从坐标字典获取光谱
            if self.coords and (y, x) in self.coords:
                spectrum = self.coords[(y, x)]
                channels = [s[0] for s in spectrum]
                values = [s[1] for s in spectrum]
                self.ax.plot(channels, values, "-o", color=color, label=f"点 {i+1}", markersize=2)

        self.ax.set_xlabel("波段")
        self.ax.set_ylabel("灰度值")
        self.ax.legend(fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.chart.draw()

    def clear_points(self):
        """清空所有选中点。"""
        self.selected_points.clear()
        self.listbox.delete(0, tk.END)
        self.ax.clear()
        self.ax.set_xlabel("波段")
        self.ax.set_ylabel("灰度值")
        self.chart.draw()
        # 重建图像（清除标记）
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)


def launch_gui(image_path: str, coords_path: Optional[str] = None):
    """
    启动 Tkinter GUI。

    参数:
        image_path: 背景图像路径
        coords_path: 坐标 JSON 路径（可选）
    """
    root = tk.Tk()
    viewer = SpectralViewer(root, image_path, coords_path)
    root.mainloop()


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "view2.tif"
    coords = sys.argv[2] if len(sys.argv) > 2 else None
    launch_gui(img, coords)
