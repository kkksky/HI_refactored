"""
光谱视频摄像机 — 配置文件

集中管理所有路径、参数和超参数，避免硬编码。
支持通过环境变量覆盖默认值。
"""

import os
from pathlib import Path

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
HI_ROOT = PROJECT_ROOT.parent / "HI"  # 旧代码目录（数据/模型所在地）

# ============================================================
# 设备选择
# ============================================================
DEVICE = os.environ.get("HI_DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu")

# ============================================================
# 光谱参数
# ============================================================
WAVELENGTH_START = 445        # nm
WAVELENGTH_END = 905          # nm
WAVELENGTH_STEP = 5           # nm
NUM_BANDS = 93                # (905 - 445) / 5 + 1 = 93

TARGET_WAVELENGTHS = list(range(WAVELENGTH_START, WAVELENGTH_END + 1, WAVELENGTH_STEP))

# ============================================================
# 点源检测参数
# ============================================================
GAUSS_K_SIZE = 7              # 高斯核尺寸
GAUSS_SIGMA = 1.0             # 高斯核标准差
MAX_FILTER_SIZE = (13, 13)    # 局部极大值检测窗口
BATCH_SIZE = 32               # GPU 批处理大小
DETECTION_THRESHOLD = 0.2     # 动态阈值系数: mean + threshold * std

# ============================================================
# 轨迹追踪参数
# ============================================================
TRACKING_WINDOW = 7           # 贯穿追踪邻域窗口大小 (7x7)
BACKWARD_WINDOW = 5           # 逆向筛选窗口大小 (5x5) — 与 get_perfect_survival_cube 兼容

# ============================================================
# 目标检测参数
# ============================================================
# 默认检测方法
DETECTION_METHOD = "SACE"     # "MTICEM" | "SACE" | "SAM" | "CEM" | "ACE"
PCA_SEL = False
PCA_COMPONENTS = 10

# 二值化阈值
BIN_THRESHOLD = {
    "MTICEM": 1.0,
    "SACE": 0.7,
}

# 后处理
RECT_H = 6                    # 叠加矩形高度
RECT_W = 53                   # 叠加矩形宽度
AREA_THRESHOLD = 1117         # 连通区域面积阈值

# ============================================================
# 深度学习参数
# ============================================================
EMBEDDING_DIM = 32            # 对比学习嵌入维度
AE_EMBEDDING_DIM = 16         # 自编码器嵌入维度
AE_INPUT_DIM = 93

CONTRASTIVE_TEMPERATURE = 0.1
CONTRASTIVE_EPOCHS = 3000
AE_EPOCHS = 50000
TRIPLET_EPOCHS = 500000

LEARNING_RATE = 1e-3
BATCH_SIZE_TRAIN = 256

# ============================================================
# 数据预处理参数
# ============================================================
SG_WINDOW_LENGTH = 11         # Savitzky-Golay 滤波窗口
SG_POLYORDER = 3              # Savitzky-Golay 滤波多项式阶数

# ============================================================
# 文件路径（默认指向旧 HI/ 目录中的数据）
# ============================================================
DATA_DIR = HI_ROOT

# 标定数据路径（场景选择）
SCENE_CONFIGS = {
    1: {  # 新样机数据
        "gray": str(HI_ROOT / "datanew" / "2ms_sky.tif"),
        "dark": str(HI_ROOT / "datanew" / "P11070000.tif"),
        "illuminance": str(HI_ROOT / "datanew" / "5ms_sky.tif"),
        "spec_base": str(HI_ROOT / "datanew" / "5ms.tif"),
        "label_map": str(HI_ROOT / "codes" / "datanew" / "label_map_full4.mat"),
        "calibration": str(HI_ROOT / "codes" / "datanew" / "calibration_dict.json"),
    },
    2: {  # 旧样机数据
        "gray": str(HI_ROOT / "dataold" / "P34340000.tif"),
        "dark": str(HI_ROOT / "dataold" / "noise_bk_15ms.tif"),
        "illuminance": str(HI_ROOT / "dataold" / "sky_5ms.tif"),
        "spec_base": str(HI_ROOT / "dataold" / "1_low_gain_15ms_1.tif"),
        "label_map": str(HI_ROOT / "codes" / "dataold" / "label_map_full2.mat"),
        "calibration": str(HI_ROOT / "codes" / "dataold" / "calibration_dict.json"),
    },
}
