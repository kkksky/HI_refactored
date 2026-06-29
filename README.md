# 高光谱数据处理系统

> 基于论文方法重构的光谱视频摄像机算法库
>
> 本项目从 `../HI/`（旧代码）重构而来，保留所有原始功能的同时修复了已知 bug，
> 并按照 7 篇光谱视频论文的算法体系进行模块化组织。

---

## 目录

- [项目概述](#项目概述)
- [论文参考](#论文参考)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [模块说明](#模块说明)
- [与旧代码的对比](#与旧代码的对比)
- [Bug 修复清单](#bug-修复清单)
- [算法说明](#算法说明)

---

## 项目概述

本项目是一个**高光谱图像（Hyperspectral Imaging, HI）数据处理与分析系统**，
主要用于光谱视频相机的数据处理和目标检测。核心功能包括：

1. **高光谱数据加载** — 从 TIF 图像序列加载多波段数据立方体 (H, W, C)
2. **光谱标定与重建** — 基于棱镜色散模型的光谱重建
3. **点源检测与追踪** — GPU 加速的贯穿目标轨迹提取
4. **目标检测** — SAM/CEM/ACE/MT-ICEM/SACE 多种算法
5. **深度学习分析** — 自编码器异常检测、对比学习光谱嵌入
6. **交互式可视化** — Tkinter 桌面版和 Streamlit Web 版

### 数据特性

- 93 个光谱波段：445nm — 905nm，步长 5nm
- 16-bit TIF 格式高光谱图像
- 典型场景：伪装目标检测（草地迷彩、荒漠迷彩、伪装网）
- 双相机系统：新样机（高帧率）与旧样机（全波段）

---

## 论文参考

本项目涉及的 7 篇核心论文覆盖了光谱视频采集的四种主要技术路线：

| # | 论文 | 年份 | 会议/期刊 | 技术路线 | 对应模块 |
|---|------|------|-----------|---------|---------|
| 1 | Du et al. "Prism-Based Multi-Spectral Video" | 2009 | ICCV | **棱镜-多光谱视频**: 使用棱镜将不同波长分散到传感器不同区域 | `reconstruction/prism_dispersion.py` |
| 2 | Cao et al. "Prism-Mask System" | 2011 | IEEE PAMI | **棱镜-掩膜成像**: 在棱镜系统中加入掩膜实现稀疏高效采样 | `reconstruction/prism_dispersion.py` |
| 3 | Cao et al. "Hybrid Camera System" | 2011 | CVPR | **混合相机**: RGB 相机 + 灰度光谱相机，通过**三边滤波**实现光谱传播 | `reconstruction/spectral_propagation.py` |
| 4 | Ma et al. "Content-Adaptive Acquisition" | 2014 | Optics Letters | **内容自适应**: 使用 SLM 动态调整采样模式，最大化信息获取效率 | `reconstruction/adaptive_sampling.py` (规划中) |
| 5 | Feng et al. "Amici Prism" | 2014 | Optical Engineering | **阿米西棱镜**: 紧凑型棱镜设计，适合集成到视频相机中 | `reconstruction/prism_dispersion.py` |
| 6 | Ma et al. "High Spatial-Spectral Resolution" | 2014 | IJCV | **高分光谱**: 改进的光谱传播算法，实现高空间-光谱分辨率 | `reconstruction/spectral_propagation.py` |
| 7 | Zhao et al. "Heterogeneous Camera Array" | 2017 | Optical Engineering | **异构阵列**: 多相机 + 宽带滤色器阵列，通过解复用恢复光谱 | `reconstruction/demultiplexing.py` (规划中) |

论文全文可在 `../publications/` 目录找到。

---

## 系统架构

```
HI_refactored/
├── README.md                          ← 本文档
├── requirements.txt                   ← 依赖清单
├── config.py                          ← 集中配置管理
│
├── data/                              ← 数据加载与预处理
│   ├── loader.py                      ← load_hyperspectral_cube (多线程TIF加载)
│   ├── calibration.py                 ← 标定字典加载、光谱向量提取
│   └── preprocessing.py               ← 暗电流校正、归一化、SG滤波、PCHIP插值
│
├── reconstruction/                    ← 光谱重建
│   ├── prism_dispersion.py            ← 柯西色散模型 → LUT + 快速光谱重构
│   └── spectral_propagation.py        ← 三边滤波光谱传播
│
├── detection/                         ← 目标检测
│   ├── point_detection.py             ← 点源检测 (CPU/GPU 两种实现)
│   ├── trajectory.py                  ← 贯穿目标轨迹追踪 (CPU/GPU)
│   ├── sam.py                         ← 光谱角检测 (SAM)
│   ├── cem.py                         ← 约束能量最小化 (CEM)
│   ├── ace.py                         ← 自适应余弦估计 (ACE)
│   ├── mticem.py                      ← 多目标约束能量最小化 (MT-ICEM)
│   └── sace.py                        ← 光谱角约束能量最小化 (SACE)
│
├── learning/                          ← 深度学习
│   ├── models.py                      ← SpectralEmbeddingNet / SpectralAE / OneDCNN
│   ├── dataset.py                     ← TripletDataset / SpectralDataset
│   ├── autoencoder.py                 ← 自编码器训练 + 异常检测
│   ├── contrastive_simclr.py          ← SimCLR 对比学习
│   └── contrastive_infonce.py         ← InfoNCE 对比学习 + 马氏距离
│
├── utils/                             ← 工具函数
│   ├── band_selection.py              ← 波段选择 (ECA/EFDPCF/FVGBS/MNBS/OPBS)
│   ├── spectral_math.py               ← L2归一化、光谱角计算
│   ├── image_registration.py          ← ECC 图像配准
│   ├── visualization.py               ← 光谱曲线绘制、检测结果可视化
│   └── io_helpers.py                  ← JSON读写、Labelme→Mask转换
│
├── app/                               ← 应用入口
│   ├── pipeline.py                    ← 完整数据处理流水线
│   ├── gui_tkinter.py                 ← Tkinter 交互式光谱查看器
│   └── gui_streamlit.py              ← Streamlit Web 光谱查看器
    ├── noise_filter.py                ← 陷波滤波核心模块
    ├── scripts/                           ← 可执行脚本
    │   ├── run_pipeline.py                ← 运行完整流水线
    │   ├── run_real_pipeline.py           ← 真实数据流水线 (噪声滤波)
    │   ├── run_detection.py               ← 运行目标检测
    │   ├── train_autoencoder.py           ← 训练自编码器
    │   ├── train_contrastive.py           ← 训练对比学习模型
    │   ├── compare_filter_results.py      ← 滤波效果对比
    │   └── benchmark_postprocessing.py    ← 后处理策略基准

```

### 数据流

```
TIF 图像序列 (文件夹)
    │
    ▼
data/loader.py ──→ 3D 数据立方体 (H, W, 93)
    │
    ▼
detection/point_detection.py ──→ 点源掩膜
    │
    ▼
detection/trajectory.py ──→ 轨迹字典 + ID 映射
    │
    ▼
data/calibration.py ──→ 光谱向量矩阵 (N, 93)
    │
    ├──▶ detection/*.py ──→ 目标检测结果
    ├──▶ learning/autoencoder.py ──→ 异常检测
    ├──▶ learning/contrastive_*.py ──→ 光谱嵌入
    └──▶ app/gui_*.py ──→ 交互式可视化
```

---

## 快速开始

### 安装

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 确保数据目录存在（默认从 ../HI/ 加载数据）
#    或修改 config.py 中的路径
```

### 运行流水线

```bash
# 完整流水线（从 TIF 加载到检测结果）
python scripts/run_pipeline.py \
    --data "D:/dataset/HI/1/450-900标定" \
    --calibration "calibration_dict.json" \
    --scene 2 \
    --method SACE
```

### 训练模型

```bash
# 训练自编码器（异常检测）
python scripts/train_autoencoder.py --data background.npy --epochs 50000

# 训练对比学习
python scripts/train_contrastive.py --method simclr --epochs 300
```

### 启动 GUI

```bash
# Tkinter 版本
python app/gui_tkinter.py view2.tif coords_dict.json

# Streamlit 版本
streamlit run app/gui_streamlit.py
```

---

## 模块说明

### data/ — 数据加载与预处理

| 函数/类 | 说明 |
|---------|------|
| `load_hyperspectral_cube()` | 多线程加载 TIF 序列为 3D 立方体 |
| `imread_unicode()` | 支持中文路径的图像读取 |
| `CalibrationLoader` | 标定字典管理、光谱向量提取、反射率计算 |
| `subtract_dark_current()` | 暗电流校正 |
| `compute_reflectance()` | 相对反射率计算 (raw / reference) |
| `savgolay_smooth()` | Savitzky-Golay 光谱平滑 |

### reconstruction/ — 光谱重建

| 函数/类 | 说明 |
|---------|------|
| `calibrate_prism_dispersion()` | 柯西色散模型系统标定 |
| `reconstruct_spectrum_fast()` | 基于高级索引的毫秒级光谱重构 |
| `SpectralPropagator` | 三边滤波光谱传播（空间+光谱+时间） |

### detection/ — 目标检测

| 类 | 原理 | 适用场景 |
|----|------|---------|
| `SpectralAngleMapper` (SAM) | 计算光谱夹角 | 快速筛查，光照变化鲁棒 |
| `CEMDetector` | 最小化背景能量 | 已知单一目标 |
| `ACEDetector` | 广义似然比 | 背景统计已知，亚像素目标 |
| `MTICEMDetector` | 多目标约束能量最小化 | 多个已知目标 |
| `SACEDetector` | 光谱角+能量约束 | 光谱变化大时的鲁棒检测 |

### learning/ — 深度学习

| 类 | 方法 | 用途 |
|----|------|------|
| `SpectralEmbeddingNet` | MLP + L2 归一化 | 光谱嵌入特征提取 |
| `SpectralAE` | 自编码器 | 异常检测（重建误差） |
| `SimCLR` | NT-Xent 对比学习 | 无监督光谱表征学习 |
| `InfoNCEContrastive` | InfoNCE + 马氏距离 | 可解释的异常检测 |

---

## 与旧代码的对比

新旧代码的文件对应关系详见 `../HI/代码分类文档.md`。

### 改进要点

| 方面 | 旧代码 | 新代码 |
|------|--------|--------|
| **组织** | 40+ 文件混杂在一个目录 | 6 个功能模块 + 4 个脚本 |
| **配置** | 硬编码路径 (D:\...) | `config.py` 集中管理 |
| **依赖** | 无 requirements.txt | 完整的依赖清单 |
| **文档** | AGENTS.md (部分过时) | README.md + 函数 docstring |
| **Bug** | 6 个已知 bug | 已全部修复 |
| **无关文件** | 115.py, agent.py 等混入 | 已标记待删除 |
| **重复** | a_matlab.py vs matlab光谱法.py | 合并为统一检测器接口 |
| **论文** | 无明确对应 | 明确标注每篇论文对应模块 |

---

## Bug 修复清单

以下是重构过程中修复的旧代码 bug：

### Bug 1: 对比学习 loss 实现错误
- **旧文件**: `对比学习.py` 第 48-66 行
- **问题**: `F.cross_entropy(logits, labels)` 中 `labels` 始终为 `torch.zeros(...)`，
  意味着所有样本都标记为类别 0，违反对比学习原则
- **修复**: 在新代码 `learning/contrastive_simclr.py` 的 `nt_xent_loss()` 中，
  正确生成正样本标签：`labels = cat([arange(B, 2B), arange(0, B)])`

### Bug 2: `torch.cov()` 兼容性
- **旧文件**: `Contrastive.py` 第 135 行
- **问题**: `torch.cov(feats.T)` 需要 PyTorch ≥ 2.0
- **修复**: 在新代码 `learning/contrastive_infonce.py` 的 `compute_cov()` 中，
  检测 `hasattr(torch, "cov")` 并提供回退手动实现

### Bug 3: AutoEncoder 模型保存逻辑错误
- **旧文件**: `AutoEncoder.py` 第 89-94 行
- **问题**: `else` 分支无条件覆盖 `loss_last`，即使 loss 未改善也会更新，
  导致保存的"最佳"模型名不副实
- **修复**: 在新代码 `learning/autoencoder.py` 中，仅在 `avg_loss < best_loss` 时
  保存并更新 `best_loss`

### Bug 4: dataset.py 只使用 target class 1
- **旧文件**: `dataset.py` 第 129 行
- **问题**: `self.target = random.sample(self.targets[1], ...)` 只取第一类目标，
  忽略 target2.npy 和 target3.npy
- **修复**: 在新代码 `learning/dataset.py` 的 `TripletDataset` 中，
  遍历 `self.targets` 字典取出所有类别的数据

### Bug 5: filter_params 不一致
- **旧文件**: `my_fuc.py`
- **问题**: `get_dict()` 用 7×7 窗口，`get_perfect_survival_cube()` 用 5×5 窗口
- **修复**: 在新代码 `detection/trajectory.py` 中，window size 由 `config.py` 统一管理，
  不同阶段使用不同的命名参数明确区分

### Bug 6: 占位算法返回随机数
- **旧文件**: `a_matlab.py` 第 283-293 行
- **问题**: `MTICEM_refine()` 和 `SACE_refine()` 返回 `np.random.rand()`
- **修复**: 在新代码 `detection/mticem.py` 和 `detection/sace.py` 中实现了真实的
  MT-ICEM 和 SACE 算法（基于 MATLAB 参考代码翻译）

### 其他修复

| 问题 | 旧代码 | 修复 |
|------|--------|------|
| 硬编码 API 密钥 | `agent.py` 中包含 `sk-e67f...` | 该文件标记待删除 |
| 无关小说内容 | `a_光谱重构.py` docstring 混入小说 | 新代码清除 |
| 无 requirements.txt | — | 已添加 |

---

## 算法说明

### 棱镜色散模型 (Prism Dispersion Model)

参考: Du 2009, Feng 2014

棱镜对不同波长的光有不同的折射率。柯西色散公式：

```
n(λ) = A + B/λ² + C/λ⁴
```

折射率随波长增加而减小 → 短波长的偏折更大。

在图像平面上，这表现为每个波长的像素偏移：

```
dx(λ) = D * (λ_ref² / λ² - 1)
dy(λ) = 0  (假设色散仅发生在水平方向)
```

本系统实现中，通过单色光照射自动标定空间基准点，然后用矩阵高级索引
一次性提取所有采样点的光谱，实现毫秒级重建。

### 三边滤波光谱传播 (Trilateral Filtering)

参考: Cao 2011 CVPR, Ma 2014 IJCV

从稀疏光谱采样点重建全分辨率光谱，使用加权融合：

```
S(p, λ) = Σᵢ w(p, qᵢ) · S(qᵢ, λ) / Σᵢ w(p, qᵢ)
```

权重由三部分组成：

- **空间权重**: `w_spatial = exp(-||p - q||² / 2σ_s²)` — 近邻像素贡献更大
- **光谱权重**: `w_spectral = exp(-||I(p) - I(q)||² / 2σ_c²)` — 颜色相近的光谱更相似
- **时间权重**: `w_temporal = exp(-||I_t - I_{t-1}||² / 2σ_t²)` — 帧间连续性

### 目标检测算法

| 算法 | 核心公式 |
|------|---------|
| CEM | min_w wᵀRw s.t. wᵀd = 1 → w = R⁻¹d / (dᵀR⁻¹d) |
| ACE | ACE(x) = (dᵀR⁻¹(x-μ))² / ((dᵀR⁻¹d) · (x-μ)ᵀR⁻¹(x-μ)) |
| MT-ICEM | W = R⁻¹D(DᵀR⁻¹D)⁻¹, score_i = max_c xᵀW[:,c] |
| SACE | ACE score × cos_angle(x, d) |
| SAM | θ = arccos(xᵀd / (||x|| · ||d||)) |

### 对比学习 (SimCLR)

使用 NT-Xent 损失：

```
L(i, j) = -log(exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ))
```

其中 (i, j) 是同一光谱的两个增强版本，τ 是温度参数。

数据增强策略：
1. 加性高斯噪声 (σ=0.01)
2. 随机缩放 (0.95 ~ 1.05)
3. 小幅度平移

### 自编码器异常检测

训练: 仅用正常光谱训练，最小化 MSE 重建误差。

检测: 对测试光谱，若 `MSE(x, AE(x)) > threshold` 则判定为异常。

阈值从训练集的重建误差分布的 P99 百分位确定。

---

## 参考文献

所有论文的 PDF 和中文解读见 `../publications/` 目录。

1. Du et al. "Prism-Based Multi-Spectral Video", ICCV 2009
2. Cao et al. "Prism-Mask System", IEEE PAMI 2011
3. Cao et al. "Hybrid Camera System", CVPR 2011
4. Ma et al. "Content-Adaptive Acquisition", Optics Letters 2014
5. Feng et al. "Amici Prism System", Optical Engineering 2014
6. Ma et al. "High Spatial-Spectral Resolution", IJCV 2014
7. Zhao et al. "Heterogeneous Camera Array", Optical Engineering 2017

---

## License

本项目为南京大学计算成像实验室 (CITE) 研究代码。
