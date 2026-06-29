# 🔧 高光谱目标检测 — 全部代码改动与实验结果说明

## 项目: 光谱视频摄像机 · HI_refactored
## 用户: kkksky (GitHub)
## 仓库: https://github.com/kkksky/HI_refactored

---

## 📋 改动总览

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `noise_filter.py` | **新建** | 空间频域陷波滤波核心模块 |
| 2 | `scripts/run_real_pipeline.py` | **修改** | 增加滤波/后处理/目标目录参数 |
| 3 | `scripts/compare_filter_results.py` | **修改** | 修复目标模板不匹配问题 |
| 4 | `scripts/benchmark_postprocessing.py` | **新建** | 后处理策略基准测试脚本 |
| 5 | `output/filtered_targets/` | **新建** | 滤波后匹配目标模板 |
| 6 | `output/水波纹噪声分析报告.md` | **新建** | 水波纹噪声分析报告 |
| 7 | `output/benchmark/` | **新建** | 45种方案对比结果 |
| 8 | `scripts/deep_learning_detection.py` | **新建** | DL vs 传统综合对比框架 |
| 9 | `README.md` | **修改** | 去除南京大学引用 |

---

## Part 1: 水波纹噪声消除系统

### 1.1 `noise_filter.py` —陷波滤波器模块

空间频域高斯窄带陷波滤波器，去除高光谱图像中的光学干涉条纹噪声。

**核心类: `NotchFilter`**

| 方法 | 作用 | 计算量 |
|------|------|--------|
| `filter_image_1d(img)` | 逐行1D列FFT陷波 | ~27ms (512×2048) |
| `filter_image_2d(img)` | 2D FFT陷波 (X方向) | ~41ms (512×2048) |
| `filter_reflectance_cube(cube)` | 逐波段1D列FFT陷波 | B×H次FFT |
| `filter_score_map(sm, method)` | Score Map后处理 | O(HW) |

**陷波参数**: 频率 1/17.5, 1/8.8, 1/5.9 px⁻¹ (基频+谐波)
**Score Map后处理**: median5/7/9, gaussian, open/close, med_open, full

### 1.2 三级级联滤波架构

```
Sky 图像 → ① 2D陷波 → 反射率 → ② 逐波段1D陷波 → Score Map → ③ 后处理
```

CLI: `--filter {none,sky,reflectance,scores,full}` + `--post {none,med_open,...}`

### 1.3 Bug修复: 目标模板不匹配

用户发现滤波后检测效果变差，原因是目标模板来自未滤波反射率。
修复: `extract_targets_from_reflectance()` 从滤波后反射率重新提取目标。

---

## Part 2: 深度学习 vs 传统方法综合对比

### 2.1 `scripts/deep_learning_detection.py` — 统一对比框架

一个脚本实现全部方法的对比，基于全图 `mask.npy` 地真评估。

**支持的方法:**

| 方法 | 类型 | 需要训练 | 需目标光谱 |
|------|------|---------|-----------|
| ACE | 传统GLRT | ❌ | ✅ |
| SACE | 传统角度+能量 | ❌ | ✅ |
| SpectralAE | 无监督DL(AE) | ✅ | ❌ |
| Spectral1DCNN | 有监督DL(1D-CNN) | ✅ | ❌ |
| SpectralSpatialCNN | 有监督DL(2D-CNN) | ✅ | ❌ |
| DerivACE | 光谱导数+ACE | ❌ | ✅ |
| RX / SSRX | 异常检测(马氏距离) | ❌ | ❌ |
| Ensemble | 多方法融合 | ❌ | ✅ |

**评估指标:** AUC(测试集) + F1/IoU/Precision/Recall(像素级)

### 2.2 实验 1: 1D-CNN 阈值扫描

- 对 Spectral1DCNN 在验证集上扫描阈值 [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
- 选最佳 F1 对应的阈值用于测试集评估

### 2.3 实验 2: SACE + med_open 后处理

- ACE/SACE 的 score map 可做 med_open 后处理
- Notch Filter 对传统方法帮助显著: ACE的F1从0.503→0.588 (+17%)

### 2.4 实验 3: Notch Filter + SACE 全流程

对比无滤波 vs 全滤波(full+med_open)下所有方法的 F1 变化

### 2.5 实验 4: AE 架构增强 (失败)

- 收紧瓶颈 16→8 + L1稀疏正则 + Dropout
- 结果: 瓶颈太紧导致信息坍缩, F1=0.000

### 2.6 实验 5: InfoNCE + Mahalanobis 检测器 (失败)

- 复用 HI/对比学习.py 的 InfoNCE 对比学习方案
- 结果: 嵌入空间区分度不够, F1=0.015

### 2.7 实验 6: Spectral-Spatial Patch CNN

- 5×5 空间邻域 × 93 波段的光谱-空间联合 2D-CNN
- 问题: 色散导致波段间空间错位, CNN学不到一致的空间特征
- F1=0.512 (远不如1D-CNN的0.695)

### 2.8 实验 7: 训练/验证/测试集分割修复

**重要修正**: 之前 AUC 和 F1 是用全量数据算的 (含训练集), 严重虚高。

| 阶段 | AUC | F1 |
|------|-----|-----|
| 修复前 (含训练数据泄露) | 0.995 | 0.770 |
| 修复后 (测试集独立) | **0.9925** | **0.695** |

### 2.9 实验 8: 无监督增强算法

| 方法 | F1 | AUC | 分析 |
|------|-----|-----|------|
| ACE (基线) | 0.588 | 0.964 | 无监督中最佳 |
| DerivACE | 0.556 | 0.959 | 导数放大噪声 |
| RX/SSRX | 0.000 | 0.42 | 纯无监督无效 |
| Ensemble | 0.230 | 0.963 | 召回率极高(0.964)但精度低 |

---

## Part 3: 最终结论与推荐

### 3.1 最终排名 (测试集独立评估)

| 方法 | F1 | AUC | Precision | Recall | 特点 |
|------|-----|-----|-----------|--------|------|
| **1D-CNN + Notch** | **0.695** | **0.9925** | 0.664 | 0.730 | 🏆 综合最强 |
| ACE + Notch | 0.588 | 0.964 | 0.703 | 0.505 | 无需训练, 稳定 |
| SACE + Notch | 0.580 | 0.964 | 0.706 | 0.492 | 角度约束 |
| DerivACE | 0.556 | 0.959 | 0.552 | 0.560 | 导数效果有限 |
| SpectralSpatialCNN | 0.512 | 0.964 | 0.835 | 0.370 | 色散导致错位 |
| Ensemble | 0.230 | 0.963 | 0.131 | 0.964 | 召回率最高 |

### 3.2 关键发现

1. **1D-CNN 是当前最佳** — 非线性光谱滤波器 + 良好正则化
2. **Notch Filter 对传统方法帮助大** — ACE F1 +17%, SACE +15%
3. **无监督方法打不过有监督** — 目标和背景光谱差异太微妙
4. **光谱-空间联合因色散错位无效** — 棱镜色散导致波段间像素偏移

### 3.3 推荐用法

```bash
# 实战最佳 (含训练/验证/测试分割)
python scripts/deep_learning_detection.py --dx 195 --dy -30 \
  --filter full --post med_open \
  --target-dir output/filtered_targets \
  --output output/result

# 仅评估传统方法 (快速, 无监督)
python scripts/deep_learning_detection.py --dx 195 --dy -30 \
  --filter full --post med_open \
  --target-dir output/filtered_targets \
  --skip-cnn --skip-ae
```

---

## Part 4: Git 配置

```bash
# SSH 方式 (已配置)
git remote set-url origin git@github.com:kkksky/HI_refactored.git
# SSH 密钥: ~/.ssh/github_kkksky (ed25519)
# 推送:
GIT_SSH_COMMAND="ssh -i ~/.ssh/github_kkksky" git push

# 日常提交:
git add <文件>
git commit -m "说明"
git push
```

---

*生成时间: 2026-06-29*
