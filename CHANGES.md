# 🔧 高光谱检测噪声抑制 — 代码改动说明

## 项目: 光谱视频摄像机 · HI_refactored

---

## 📋 改动总览

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `noise_filter.py` | **新建** | 空间频域陷波滤波核心模块 |
| 2 | `scripts/run_real_pipeline.py` | **修改** | 增加滤波/后处理/目标目录参数 |
| 3 | `scripts/compare_filter_results.py` | **修改** | 修复目标模板不匹配问题 |
| 4 | `scripts/benchmark_postprocessing.py` | **新建** | 后处理策略基准测试脚本 |
| 5 | `output/filtered_targets/` | **新建** | 滤波后匹配目标模板 |
| 6 | `output/水波纹噪声分析报告.md` | **新建** | 图文并茂的完整分析报告 |
| 7 | `output/benchmark/` | **新建** | 45种方案对比结果 |

---

## 1. `noise_filter.py` — 陷波滤波器模块 (新建)

**位置**: `HI_refactored/noise_filter.py`

### 功能
空间频域高斯窄带陷波滤波器，去除高光谱图像中的光学干涉条纹噪声。

### 核心类: `NotchFilter`

| 方法 | 作用 | 计算量 |
|------|------|--------|
| `filter_image_1d(img)` | 逐行1D列FFT陷波 | ~27ms (512×2048) |
| `filter_image_2d(img)` | 2D FFT陷波 (X方向) | ~41ms (512×2048) |
| `filter_reflectance_cube(cube)` | 逐波段1D列FFT陷波 | B×H次FFT |
| `filter_score_map(sm, method)` | Score Map后处理 | O(HW) |

### 陷波参数
- **频率**: 1/17.5, 1/8.8, 1/5.9 px⁻¹ (基频+谐波)
- **带宽**: σ=0.005 px⁻¹ (高斯窄带)
- **衰减**: A=0.95 (保留5%避免振铃)

### Score Map后处理方法
| 方法 | 说明 |
|------|------|
| `median5/7/9` | N×N中值滤波 |
| `gaussian` | 高斯平滑 (σ=kernel/3) |
| `open` | 形态学开运算 (腐蚀→膨胀) |
| `close` | 形态学闭运算 (膨胀→腐蚀) |
| `med_open` | 中值5×5 + 开运算 (推荐) |
| `full` | 中值5×5 + 开运算 + 中值5×5 |

---

## 2. `scripts/run_real_pipeline.py` — Pipeline (修改)

### 新增CLI参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--filter` | str | `none` | 滤波级别: `none/sky/reflectance/scores/full` |
| `--post` | str | `median5` | Score Map后处理: `none/median5/median7/median9/gaussian/open/close/med_open/full` |
| `--post-kernel` | int | 5 | 后处理滤波核大小 |
| `--target-dir` | str | None | 目标模板目录 (默认从hi-dir加载) |

### 新增数据流

```
  反射率计算:
    raw → subtract_dark → compute_reflectance
        → [Sky 2D陷波] → [Reflectance逐波段陷波] → 输出

  检测后处理:
    score_map → [中值滤波/开运算/组合] → 连通区域过滤 → 可视化
```

### 典型用法

```bash
# 原始流程 (无滤波)
python scripts/run_real_pipeline.py --dx 195 --dy -30

# 全套降噪 (推荐)
python scripts/run_real_pipeline.py --dx 195 --dy -30 \
    --filter full --post med_open --method all \
    --target-dir output/filtered_targets
```

---

## 3. `scripts/compare_filter_results.py` — 滤波效果对比 (修改)

### 修复: 目标模板匹配问题

**问题**: 原始脚本使用 `HI/target{1-3}.npy` (从**未滤波**反射率提取) 去检测滤波后场景 → 目标/场景不匹配 → 虚假的过度抑制

**修复**: 新增 `extract_targets_from_reflectance(reflect, hi_dir)` 函数

**提取流程**:
```
id_to_key.json (13042个标注点)
  → 筛选 mask=4/5/6 → target 1/2/3
  → 通过 coords_dict.json 获取93波段坐标
  → 从滤波后反射率中提取光谱值
  → 输出: {1: (81,93), 2: (124,93), 3: (20,93)}
```

**验证**: 提取数量与原始目标完全一致 (81/124/20条光谱)

---

## 4. `scripts/benchmark_postprocessing.py` — 基准测试 (新建)

### 功能
对5种检测算法 × 9种后处理策略 = 45种组合进行全面对比

### 评分标准
```
综合分 = 保留像素/1000 + 背景σ×5000  (越低越好)
```

### 输出
- `output/benchmark/comparison_table.txt` — 全量对比表
- `output/benchmark/winner_report.txt` — 最佳方案
- `output/benchmark/heatmap_{METHOD}.png` — 每种算法的后处理对比热力图

### 用法
```bash
python scripts/benchmark_postprocessing.py --dx 195 --dy -30
```

---

## 5. `output/filtered_targets/` — 滤波后目标模板 (新建)

`run_real_pipeline.py` 调用 `load_target_templates(hi_dir, target_dir)` 时：
- 如果 `--target-dir` 指定了 `output/filtered_targets/`，优先加载该目录下的目标
- 此时目标和场景数据都在滤波后反射率中提取，完全匹配

### 验证指标

| 对比项 | 原始目标(不匹配) | 滤波后目标(匹配) |
|--------|-----------------|-----------------|
| ACE检测像素 (score map) | 7,402 (过度抑制) | 34,416 (合理) |
| ACE最大分数 | 0.3602 | 0.4131 |
| 结论 | ❌ 错误对比 | ✅ 正确对比 |

---

## 6. 诊断脚本 (未改动，保留)

以下脚本为之前会话创建的噪声分析工具，未修改：

| 脚本 | 用途 |
|------|------|
| `scripts/noise_summary_plot.py` | 噪声机理综合诊断图 |
| `scripts/noise_verify.py` | 噪声来源交叉验证 |
| `scripts/noise_deep_analysis.py` | 深度噪声分析 |
| `scripts/noise_source_pinpoint.py` | 噪声源精确定位 |

---

## 7. 推荐使用方式

### 新检测基准

```bash
# 1. 生成滤波后匹配目标
python -c "
from noise_filter import NotchFilter
from data.preprocessing import subtract_dark_current, compute_reflectance
# ... (详见 benchmark_postprocessing.py)
"

# 2. 跑全流程
python scripts/run_real_pipeline.py \
    --dx 195 --dy -30 \
    --filter full --post med_open --method all \
    --target-dir output/filtered_targets
```

### 推荐方案: SACE + Notch Filter(Full) + med_open

| 维度 | 选择 | 理由 |
|------|------|------|
| 检测算法 | **SACE** | 角度约束+协方差，综合抗噪最强 |
| 空间滤波 | **Notch Full** | 去除光学干涉条纹源头 |
| Score Map | **med_open** | 保留最大分的同时降低背景噪声 |
| 目标模板 | **滤波后提取** | 目标/场景数据同源匹配 |

---

## 附: git 提交建议

如需提交到git，建议按功能分3个commit：

```
commit 1: 添加陷波滤波器模块
  - noise_filter.py (新建)
  - run_real_pipeline.py: 添加 --filter 参数

commit 2: 修复目标模板匹配 + 增强后处理
  - compare_filter_results.py: extract_targets_from_reflectance()
  - noise_filter.py: filter_score_map 增加形态学方法
  - run_real_pipeline.py: 添加 --post/--target-dir 参数
  - output/filtered_targets/ (生成物)

commit 3: 基准测试 + 完整报告
  - scripts/benchmark_postprocessing.py (新建)
  - output/水波纹噪声分析报告.md (新建)
  - output/benchmark/ (生成物)
```

---

## 附: Git 提交与上传

```bash
# 1. 克隆到本地（首次）
git clone https://github.com/KKKSKY/HI_refactored.git
cd HI_refactored

# 2. 日常开发流程
git add <修改的文件>
git commit -m "说明改了什么"
git push

# 3. 如何生成 Personal Access Token（用于HTTPS提交）
#    浏览器 → GitHub Settings → Developer settings → Personal access tokens
#    → Fine-grained tokens → Generate new token
#    权限: Contents → Read and write
#    → 复制 token

# 4. 首次推送到新仓库
git remote add origin https://github.com/KKKSKY/HI_refactored.git
git push -u origin master

# 注: Token 只在推送时临时使用，不要提交到代码中
```

---

*生成时间: 2026-06-29*
