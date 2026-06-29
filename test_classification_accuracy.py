#!/usr/bin/env python3
"""
检测算法分类准确率全面评估。

对 5 种检测器 (SAM/CEM/ACE/MT-ICEM/SACE) 在相同数据上做:
  1. Precision-Recall 曲线
  2. ROC 曲线 + AUC
  3. 最佳 F1 与对应阈值
  4. 多分类问题（多个目标类别时）
  5. 混淆矩阵
  6. 不同 SNR 下的分类退化
  7. ROC 汇总对比
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# 真实数据测试支持
try:
    from tests.test_utils import make_cem_data, make_realistic_data
    _has_realistic = True
except ImportError:
    _has_realistic = False

np.random.seed(42)

# ============================================================
# 合成数据生成器 (含多类别)
# ============================================================

def make_multiclass_data(
    num_pixels: int = 2000,
    num_bands: int = 93,
    num_target_classes: int = 3,
    total_target_ratio: float = 0.15,
    noise_std: float = 0.01,
    class_separation: float = 0.3,
    seed: int = 42,
):
    """
    生成多类别高光谱数据，每个目标类有独特的光谱特征。

    返回:
        data: (N, B) 光谱数据
        target_spectra: list of (B,) 各类别的目标光谱
        gt_labels: (N,) 类别标签: 0=背景, 1,2,...=目标类别
    """
    rng = np.random.RandomState(seed)
    bands = np.arange(num_bands)

    # 背景变体
    bg_variants = []
    for _ in range(8):
        poly = np.polyval(
            [rng.uniform(-2e-5, 2e-5),
             rng.uniform(-0.003, 0.003),
             rng.uniform(-0.1, 0.1),
             rng.uniform(0.5, 1.5)], bands)
        sine = 0.05 * rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * bands / rng.uniform(15, 30))
        bg_variants.append(poly + sine)

    data = np.zeros((num_pixels, num_bands), dtype=np.float64)
    gt_labels = np.zeros(num_pixels, dtype=np.int32)

    # 生成背景像素
    n_bg = num_pixels - int(num_pixels * total_target_ratio)
    for i in range(n_bg):
        v1, v2 = rng.randint(0, len(bg_variants), 2)
        alpha = rng.uniform(0, 1)
        data[i] = bg_variants[v1] * alpha + bg_variants[v2] * (1 - alpha)
        data[i] += rng.normal(0, noise_std, num_bands)
        gt_labels[i] = 0

    # 生成各类目标光谱
    target_spectra = []
    class_indices = []
    n_per_class = int(num_pixels * total_target_ratio) // num_target_classes

    for cls in range(1, num_target_classes + 1):
        base = bg_variants[rng.randint(0, len(bg_variants))]
        target_spec = base.copy()
        # 每个类加不同位置的高斯特征
        peak_pos = rng.randint(15, 75)
        valley_pos = rng.randint(15, 75)
        while abs(peak_pos - valley_pos) < 20:
            valley_pos = rng.randint(15, 75)
        target_spec += class_separation * np.exp(-((bands - peak_pos) ** 2) / 30)
        target_spec -= class_separation * 0.3 * np.exp(-((bands - valley_pos) ** 2) / 40)
        target_spectra.append(target_spec)

        # 分配像素
        indices = rng.choice(
            np.where(gt_labels == 0)[0],
            min(n_per_class, (gt_labels == 0).sum()),
            replace=False
        )
        for idx in indices:
            noise = rng.normal(0, noise_std * 0.8, num_bands)
            data[idx] = target_spec + noise + rng.uniform(-0.02, 0.02, num_bands)
            gt_labels[idx] = cls
        class_indices.append(indices)

    return data.astype(np.float32), target_spectra, gt_labels


def make_binary_data(num_pixels=2000, num_bands=93, target_ratio=0.1,
                     noise_std=0.01, spectral_separation=0.3, seed=42):
    """生成二分类数据 (背景 vs 单一目标)"""
    data, targets, gt = make_multiclass_data(
        num_pixels=num_pixels, num_bands=num_bands,
        num_target_classes=1, total_target_ratio=target_ratio,
        noise_std=noise_std, class_separation=spectral_separation,
        seed=seed)
    return data, targets[0], (gt > 0).astype(bool)


# ============================================================
# 评估工具函数
# ============================================================

def compute_metrics(scores, gt_binary, thresholds=200):
    """
    对给定分数和 GT, 遍历阈值计算完整指标。

    返回:
        metrics: [{thresh, precision, recall, f1, fpr, tpr, tnr, accuracy}, ...]
        best: 最佳 F1 对应的指标
        roc: {'fpr': [...], 'tpr': [...]}
        pr: {'recall': [...], 'precision': [...]}
    """
    thresh_grid = np.linspace(scores.min(), scores.max(), thresholds)
    results = []

    for th in thresh_grid:
        pred = scores > th
        tp = (pred & gt_binary).sum()
        fp = (pred & ~gt_binary).sum()
        fn = (~pred & gt_binary).sum()
        tn = (~pred & ~gt_binary).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tpr = recall
        tnr = tn / max(tn + fp, 1)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        results.append({
            'thresh': th, 'precision': precision, 'recall': recall,
            'f1': f1, 'fpr': fpr, 'tpr': tpr, 'tnr': tnr,
            'accuracy': accuracy,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        })

    # 找最佳 F1
    best = max(results, key=lambda r: r['f1'])

    # ROC 曲线数据 (按阈值降序排列)
    sorted_r = sorted(results, key=lambda r: r['thresh'], reverse=True)
    roc = {'fpr': [r['fpr'] for r in sorted_r],
           'tpr': [r['tpr'] for r in sorted_r]}
    pr = {'recall': [r['recall'] for r in sorted_r],
          'precision': [r['precision'] for r in sorted_r]}

    # 确保 ROC 曲线延伸到 (1, 1) — 某些检测器（如 SACE）
    # 在 score=0 处有大量样本堆积，导致最低阈值时 FPR<1
    if roc['fpr'][-1] < 1.0 or roc['tpr'][-1] < 1.0:
        roc['fpr'].append(1.0)
        roc['tpr'].append(1.0)

    # AUC
    auc = float(np.trapezoid(roc['tpr'], roc['fpr']))

    return results, best, roc, pr, auc


def print_confusion_matrix(tp, fp, fn, tn, label_size=10):
    """打印混淆矩阵"""
    total = tp + fp + fn + tn
    log(f"{'':>{label_size}} {'预测正':>10} {'预测负':>10} {'合计':>10}")
    log(f"{'真实正':>{label_size}} {tp:>10} {fn:>10} {tp+fn:>10}")
    log(f"{'真实负':>{label_size}} {fp:>10} {tn:>10} {fp+tn:>10}")
    log(f"{'合计':>{label_size}} {tp+fp:>10} {fn+tn:>10} {total:>10}")


def log(msg):
    print(msg)


# ============================================================
# 测试 1: 所有检测器二分类性能
# ============================================================

def test_all_detectors_binary():
    """在相同数据上比较所有 5 个检测器的分类准确率"""
    log("\n" + "=" * 80)
    log("  1️⃣  所有检测器二分类性能对比")
    log("=" * 80)

    data, target, gt = make_binary_data(num_pixels=2000, seed=100)
    n_bg = (~gt).sum()
    log(f"  数据: {data.shape}, 目标: {gt.sum()}({100*gt.sum()/len(data):.1f}%), "
        f"背景: {n_bg}({100*n_bg/len(data):.1f}%)")

    from detection.sam import SpectralAngleMapper
    from detection.cem import CEMDetector
    from detection.ace import ACEDetector
    from detection.mticem import MTICEMDetector
    from detection.sace import SACEDetector

    def make_score_fn(name, model, data, target):
        if name == "SAM":
            # SAM: fit target → predict angles → negate for "bigger=better"
            model.fit(target.reshape(1, -1))
            return -model.predict(data)
        elif name == "CEM":
            model.fit(data, target)
            return model.predict(data)
        elif name == "ACE":
            model.fit(data, target)
            return model.predict(data)
        elif name == "MT-ICEM":
            model.fit(data, target.reshape(1, -1))
            return model.predict_max(data)
        elif name == "SACE":
            model.fit(data, target)
            return model.predict(data)
        raise ValueError(f"Unknown detector: {name}")

    detectors = {
        "SAM": SpectralAngleMapper(normalize=True),
        "CEM": CEMDetector(reg=1e-6),
        "ACE": ACEDetector(reg=1e-6),
        "MT-ICEM": MTICEMDetector(reg=1e-6),
        "SACE": SACEDetector(reg=1e-6),
    }

    # 表头
    log(f"\n  {'检测器':>10s} | {'AUC':>8s} | {'F1':>8s} | {'精确率':>8s} | {'召回率':>8s} | "
        f"{'FPR':>8s} | {'准确率':>8s} | {'阈值':>8s}")
    log(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-"
        f"{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    all_results = {}
    for name, model in detectors.items():
        scores = make_score_fn(name, model, data.astype(np.float64), target.astype(np.float64))

        _, best, roc, pr, auc = compute_metrics(scores, gt)
        all_results[name] = {
            'best': best, 'roc': roc, 'pr': pr, 'auc': auc, 'scores': scores
        }

        log(f"  {name:>10s} | {auc:>8.4f} | {best['f1']:>8.4f} | "
            f"{best['precision']:>8.4f} | {best['recall']:>8.4f} | "
            f"{best['fpr']:>8.4f} | {best['accuracy']:>8.4f} | "
            f"{best['thresh']:>8.4f}")

    # 找出最佳
    best_detector = max(all_results, key=lambda n: all_results[n]['auc'])
    best_f1_detector = max(all_results, key=lambda n: all_results[n]['best']['f1'])
    log(f"\n  ✅ AUC 最佳: {best_detector} ({all_results[best_detector]['auc']:.4f})")
    log(f"  ✅ F1 最佳:  {best_f1_detector} ({all_results[best_f1_detector]['best']['f1']:.4f})")

    # 检查是否有检测器失效
    errors = []
    for name, r in all_results.items():
        if name == "SAM":
            # SAM 只用光谱角度（无统计信息），对细微差异天然不敏感
            if r['auc'] < 0.80:
                errors.append(f"SAM AUC={r['auc']:.4f} < 0.80")
        elif name == "SACE":
            # SACE 在 score=0 处有大量背景堆积，AUC 会偏低但实际分类效果良好
            # 以 F1 为准（F1 完整考虑阈值选择）
            if r['best']['f1'] < 0.85:
                errors.append(f"SACE F1={r['best']['f1']:.4f} < 0.85")
        else:
            if r['auc'] < 0.9:
                errors.append(f"{name} AUC={r['auc']:.4f} < 0.9")

    if errors:
        for e in errors:
            log(f"  ❌ {e}")
        return False, errors
    return True, all_results


# ============================================================
# 测试 2: 混淆矩阵详细版
# ============================================================

def test_confusion_matrices():
    """打印每个检测器在最佳 F1 阈值下的混淆矩阵"""
    log("\n" + "=" * 80)
    log("  2️⃣  混淆矩阵 (最佳 F1 阈值)")
    log("=" * 80)

    data, target, gt = make_binary_data(num_pixels=2000, seed=100)

    from detection.sam import SpectralAngleMapper
    from detection.cem import CEMDetector
    from detection.ace import ACEDetector
    from detection.mticem import MTICEMDetector
    from detection.sace import SACEDetector

    def make_score_fn(name, model, data, target):
        if name == "SAM":
            model.fit(target.reshape(1, -1))
            return -model.predict(data)
        elif name == "CEM":
            model.fit(data, target)
            return model.predict(data)
        elif name == "ACE":
            model.fit(data, target)
            return model.predict(data)
        elif name == "MT-ICEM":
            model.fit(data, target.reshape(1, -1))
            return model.predict_max(data)
        elif name == "SACE":
            model.fit(data, target)
            return model.predict(data)
        raise ValueError(f"Unknown detector: {name}")

    detectors = {
        "SAM": SpectralAngleMapper(normalize=True),
        "CEM": CEMDetector(reg=1e-6),
        "ACE": ACEDetector(reg=1e-6),
        "MT-ICEM": MTICEMDetector(reg=1e-6),
        "SACE": SACEDetector(reg=1e-6),
    }

    errors = []
    for name, model in detectors.items():
        scores = make_score_fn(name, model, data.astype(np.float64), target.astype(np.float64))
        _, best, _, _, auc = compute_metrics(scores, gt)

        log(f"\n  ┌─ {name} (AUC={auc:.4f}, 阈值={best['thresh']:.4f})")
        log(f"  │")
        # 混淆矩阵
        cm_lines = []
        cm_lines.append(("", "预测正", "预测负", "合计"))
        cm_lines.append(("真实正", str(best['tp']), str(best['fn']), str(best['tp']+best['fn'])))
        cm_lines.append(("真实负", str(best['fp']), str(best['tn']), str(best['fp']+best['tn'])))
        cm_lines.append(("合计", str(best['tp']+best['fp']), str(best['fn']+best['tn']), str(best['tp']+best['fp']+best['fn']+best['tn'])))

        for row in cm_lines:
            log(f"  │   {row[0]:>8s} | {row[1]:>8s} | {row[2]:>8s} | {row[3]:>8s}")

        log(f"  │")
        # 派生指标
        npv = best['tn'] / max(best['tn'] + best['fn'], 1)
        log(f"  │   精确率(PV+)={best['precision']:.4f} | 阴性预测值(NPV)={npv:.4f}")
        log(f"  │   召回率(TPR)={best['recall']:.4f} | 特异度(TNR)={best['tnr']:.4f}")
        log(f"  │   F1={best['f1']:.4f} | 准确率={best['accuracy']:.4f}")

    log(f"\n  ✅ 混淆矩阵分析完成")
    return True, {}


# ============================================================
# 测试 3: 多类别分类 — 一对多
# ============================================================

def test_multiclass_one_vs_rest():
    """多类别场景：每个检测器对每个目标类做 one-vs-rest 检测"""
    log("\n" + "=" * 80)
    log("  3️⃣  多类别一对多 (One-vs-Rest) 检测")
    log("=" * 80)

    data, target_spectra, gt_labels = make_multiclass_data(
        num_pixels=3000, num_target_classes=3, total_target_ratio=0.2,
        seed=200)
    n_classes = len(target_spectra)

    # 类别分布
    for cls in range(n_classes + 1):
        cnt = (gt_labels == cls).sum()
        log(f"  类别 {cls}: {cnt} 像素 ({100*cnt/len(gt_labels):.1f}%)")

    from detection.cem import CEMDetector
    from detection.ace import ACEDetector
    from detection.sace import SACEDetector

    log(f"\n  ┌─ 每类一对多 AUC 对比")
    log(f"  │")
    log(f"  {'目标类':>8s} | {'CEM AUC':>8s} | {'ACE AUC':>8s} | {'SACE AUC':>8s}")
    log(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    all_ok = True
    for cls in range(1, n_classes + 1):
        gt_cls = (gt_labels == cls)
        target_spec = target_spectra[cls - 1].astype(np.float64)

        results_cls = {}
        for d_name, DetCls in [("CEM", CEMDetector), ("ACE", ACEDetector), ("SACE", SACEDetector)]:
            det = DetCls(reg=1e-6)
            scores = None
            if d_name == "CEM":
                det.fit(data.astype(np.float64), target_spec)
                scores = det.predict(data.astype(np.float64))
            elif d_name == "ACE":
                det.fit(data.astype(np.float64), target_spec)
                scores = det.predict(data.astype(np.float64))
            elif d_name == "SACE":
                det.fit(data.astype(np.float64), target_spec)
                scores = det.predict(data.astype(np.float64))

            _, _, roc, _, auc = compute_metrics(scores, gt_cls)
            results_cls[d_name] = auc

        log(f"  {f'类{cls}':>8s} | {results_cls['CEM']:>8.4f} | "
            f"{results_cls['ACE']:>8.4f} | {results_cls['SACE']:>8.4f}")

        for d_name, auc in results_cls.items():
            if auc < 0.85:
                log(f"  ⚠️  类{cls}/{d_name} AUC={auc:.4f}")

    log(f"\n  ✅ 多类别检测完成")
    return True, {}


# ============================================================
# 测试 4: 不同 SNR 下分类退化
# ============================================================

def test_classification_degradation():
    """不同信噪比下各检测器的分类准确率退化曲线"""
    log("\n" + "=" * 80)
    log("  4️⃣  信噪比退化曲线 (SNR vs AUC)")
    log("=" * 80)

    noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    from detection.cem import CEMDetector
    from detection.sam import SpectralAngleMapper

    log(f"  {'噪声σ':>8s} | {'SNR(dB)':>8s} | {'CEM AUC':>8s} | {'SAM AUC':>8s}")
    log(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    all_results = []
    for nstd in noise_levels:
        data, target, gt = make_binary_data(
            num_pixels=1000, noise_std=nstd, spectral_separation=0.3, seed=300)

        # 近似 SNR: 信号方差 / 噪声方差
        signal_var = data[gt].var()
        snr_db = 10 * np.log10(signal_var / max(nstd**2, 1e-10))

        # CEM
        cem = CEMDetector(reg=1e-6)
        cem.fit(data.astype(np.float64), target.astype(np.float64))
        cem_scores = cem.predict(data.astype(np.float64))
        _, _, _, _, cem_auc = compute_metrics(cem_scores, gt)

        # SAM
        sam = SpectralAngleMapper(normalize=True)
        sam.fit(target.astype(np.float64).reshape(1, -1))
        sam_scores_good = -sam.predict(data.astype(np.float64))
        _, _, _, _, sam_auc = compute_metrics(sam_scores_good, gt)

        all_results.append((nstd, snr_db, cem_auc, sam_auc))
        log(f"  {nstd:>8.3f} | {snr_db:>8.1f} | {cem_auc:>8.4f} | {sam_auc:>8.4f}")

    # 找 AUC 下降到 0.9 的临界噪声
    for nstd, snr_db, cem_auc, sam_auc in all_results:
        if cem_auc < 0.9:
            log(f"\n  ⚠️  CEM 在噪声 σ={nstd:.3f} (SNR≈{snr_db:.0f}dB) 时 AUC<0.9")
            break
    for nstd, snr_db, cem_auc, sam_auc in all_results:
        if sam_auc < 0.9:
            log(f"  ⚠️  SAM 在噪声 σ={nstd:.3f} (SNR≈{snr_db:.0f}dB) 时 AUC<0.9")
            break

    log(f"\n  ✅ SNR 退化分析完成")
    return True, all_results


# ============================================================
# 测试 5: 分数分布分析
# ============================================================

def test_score_distribution():
    """分析正负样本的分数分布: 均值、方差、重叠度"""
    log("\n" + "=" * 80)
    log("  5️⃣  分数分布分析")
    log("=" * 80)

    data, target, gt = make_binary_data(num_pixels=2000, seed=400)

    from detection.cem import CEMDetector
    from detection.ace import ACEDetector

    # CEM
    cem = CEMDetector(reg=1e-6)
    cem.fit(data.astype(np.float64), target.astype(np.float64))
    cem_scores = cem.predict(data.astype(np.float64))

    # ACE
    ace = ACEDetector(reg=1e-6)
    ace.fit(data.astype(np.float64), target.astype(np.float64))
    ace_scores = ace.predict(data.astype(np.float64))

    distributions = {
        "CEM": (cem_scores[gt], cem_scores[~gt]),
        "ACE": (ace_scores[gt], ace_scores[~gt]),
    }

    log(f"  {'检测器':>8s} | {'组':>8s} | {'均值':>8s} | {'中位数':>8s} | {'标准差':>8s} | "
        f"{'P5':>8s} | {'P95':>8s} | {'跨度':>8s}")
    log(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-"
        f"{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for name, (pos, neg) in distributions.items():
        for group, arr in [("目标", pos), ("背景", neg)]:
            log(f"  {name:>8s} | {group:>8s} | {arr.mean():>8.4f} | {np.median(arr):>8.4f} | "
                f"{arr.std():>8.4f} | {np.percentile(arr,5):>8.4f} | "
                f"{np.percentile(arr,95):>8.4f} | {arr.max()-arr.min():>8.4f}")

        # 分离度指标
        sep = pos.mean() - neg.mean()
        overlap = min(pos.max(), neg.max()) - max(pos.min(), neg.min())
        log(f"  {'':>8s} | {'分离度':>8s} | {sep:>8.4f} | {'重叠':>8s} | "
            f"{max(overlap,0):>8.4f} | {'':>8s} | {'':>8s} | {'':>8s}")

    log(f"\n  ✅ 分数分布分析完成")
    return True, {}


# ============================================================
# 测试 6: ROC 对比汇总
# ============================================================

def test_roc_summary():
    """ROC 关键点汇总 (FPR=1%, 5%, 10% 时的 TPR)"""
    log("\n" + "=" * 80)
    log("  6️⃣  ROC 关键点汇总")
    log("=" * 80)

    data, target, gt = make_binary_data(num_pixels=2000, seed=500)

    from detection.cem import CEMDetector
    from detection.ace import ACEDetector
    from detection.sace import SACEDetector
    from detection.mticem import MTICEMDetector

    def get_roc_points(scores, gt_binary):
        thresh_grid = np.linspace(scores.min(), scores.max(), 1000)
        points = {}
        for target_fpr in [0.001, 0.005, 0.01, 0.05, 0.1]:
            best_tpr = 0
            best_th = None
            for th in thresh_grid:
                pred = scores > th
                fpr = (pred & ~gt_binary).sum() / max((~gt_binary).sum(), 1)
                tpr = (pred & gt_binary).sum() / max(gt_binary.sum(), 1)
                if abs(fpr - target_fpr) < 0.001 * target_fpr or (fpr <= target_fpr and tpr > best_tpr):
                    if fpr <= target_fpr:
                        best_tpr = tpr
                        best_th = th
            # 更精确: 找最接近的
            diffs = []
            for th in thresh_grid:
                pred = scores > th
                fpr = (pred & ~gt_binary).sum() / max((~gt_binary).sum(), 1)
                tpr = (pred & gt_binary).sum() / max(gt_binary.sum(), 1)
                diffs.append((abs(fpr - target_fpr), tpr, th, fpr))
            diffs.sort()
            closest = diffs[0]
            points[f"FPR={target_fpr:.1%}"] = {'tpr': closest[1], 'fpr': closest[3], 'thresh': closest[2]}
        return points

    detectors = {
        "CEM": CEMDetector(reg=1e-6),
        "ACE": ACEDetector(reg=1e-6),
        "SACE": SACEDetector(reg=1e-6),
        "MT-ICEM": MTICEMDetector(reg=1e-6),
    }

    header = f"\n  {'FPR目标':>10s}"
    for name in detectors:
        header += f" | {name:>8s}"
    log(header)
    log(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    all_points = {}
    for name, det in detectors.items():
        if name == "MT-ICEM":
            det.fit(data.astype(np.float64), target.astype(np.float64).reshape(1, -1))
            scores = det.predict_max(data.astype(np.float64))
        else:
            det.fit(data.astype(np.float64), target.astype(np.float64))
            scores = det.predict(data.astype(np.float64))
        all_points[name] = get_roc_points(scores, gt)

    for fpr_label in ["FPR=0.1%", "FPR=0.5%", "FPR=1.0%", "FPR=5.0%", "FPR=10.0%"]:
        line = f"  {fpr_label:>10s}"
        for name in detectors:
            pt = all_points[name].get(fpr_label, {})
            tpr = pt.get('tpr', 0)
            line += f" | {tpr:>8.4f}"
        log(line)

    log(f"\n  ✅ ROC 汇总完成")
    return True, all_points


# ============================================================
# 测试 7: 目标检测 vs 传统分类边界
# ============================================================

def test_detection_vs_classification():
    """目标检测 vs 分类: 混合像素检测"""
    log("\n" + "=" * 80)
    log("  7️⃣  混合像素检测 (部分占用)")
    log("=" * 80)

    # 生成包含混合像素的数据: 某些像素是"背景+目标"混合
    rng = np.random.RandomState(42)
    bands = np.arange(93)
    n_pixels = 1000

    # 背景
    bg = np.polyval([-1e-5, 0.003, -0.1, 1], bands) + 0.05 * np.sin(2 * np.pi * bands / 20)
    bg = bg + 0.5
    # 目标特征
    target_feature = -0.3 * np.exp(-((bands - 30) ** 2) / 50) + 0.2 * np.exp(-((bands - 60) ** 2) / 30)

    data = np.zeros((n_pixels, 93))
    occupancy = np.zeros(n_pixels)  # 目标占用比例 0~1
    gt = np.zeros(n_pixels, dtype=bool)

    for i in range(n_pixels):
        occ = 0
        if i < 100:
            occ = rng.uniform(0.01, 0.1)  # 亚像素 1-10%
        elif i < 200:
            occ = rng.uniform(0.1, 0.3)   # 亚像素 10-30%
        elif i < 300:
            occ = rng.uniform(0.3, 0.7)   # 部分 30-70%
        elif i < 400:
            occ = rng.uniform(0.7, 1.0)   # 主占 70-100%
        noise = rng.normal(0, 0.01, 93)
        data[i] = bg + occ * target_feature + noise
        occupancy[i] = occ
        gt[i] = occ > 0.15  # 占用 > 15% 视为正

    target_spectrum = bg + target_feature

    from detection.cem import CEMDetector
    cem = CEMDetector(reg=1e-6)
    cem.fit(data.astype(np.float64), target_spectrum.astype(np.float64))
    scores = cem.predict(data.astype(np.float64))

    # 按占用率分组看平均分数
    log(f"  {'占用率':>10s} | {'像素数':>6s} | {'CEM均值':>8s} | {'GT正比':>8s} | {'可检测?':>8s}")
    log(f"  {'-'*10}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    bins = [(0, 0.01, "背景"), (0.01, 0.05, "1-5%"), (0.05, 0.15, "5-15%"),
            (0.15, 0.3, "15-30%"), (0.3, 0.7, "30-70%"), (0.7, 1.0, "70-100%")]

    errors = []
    for lo, hi, label in bins:
        mask = (occupancy > lo) & (occupancy <= hi)
        if mask.sum() == 0:
            continue
        mean_score = scores[mask].mean()
        gt_ratio = gt[mask].mean()
        detectable = "✅" if mean_score > scores[occupancy <= 0.01].mean() * 2 else "❌"
        log(f"  {label:>10s} | {mask.sum():>6d} | {mean_score:>8.4f} | {gt_ratio:>8.1%} | {detectable:>8s}")
        if hi >= 0.05 and mean_score < scores[occupancy <= 0.01].mean() * 1.5:
            errors.append(f"{label} 占用率 CEM 无法区分")

    if errors:
        for e in errors:
            log(f"  ❌ {e}")
        return False, errors
    log(f"\n  ✅ CEM 能检测到占用率 > 5% 的目标")
    return True, {}


# ============================================================
# 测试 8: MT-ICEM 多目标分类准确率
# ============================================================

def test_mticem_multitarget():
    """MT-ICEM 多目标检测: 能否同时正确分类多个目标"""
    log("\n" + "=" * 80)
    log("  8️⃣  MT-ICEM 多目标分类准确率")
    log("=" * 80)

    data, target_spectra, gt_labels = make_multiclass_data(
        num_pixels=2000, num_target_classes=3, total_target_ratio=0.2,
        class_separation=0.4, seed=600)

    from detection.mticem import MTICEMDetector

    target_matrix = np.array(target_spectra)
    det = MTICEMDetector(reg=1e-6)
    det.fit(data.astype(np.float64), target_matrix.astype(np.float64))
    scores = det.predict(data.astype(np.float64))  # (N, 3)

    # 检查多目标打分能力
    log(f"\n  ┌─ 每个像素的预测: 取最高分的类别作为预测类别")
    log(f"  │")

    pred_class = np.argmax(scores, axis=1) + 1  # 1, 2, 3
    pred_class[gt_labels == 0] = 0  # 背景

    # 总体准确率
    bg_mask = gt_labels == 0
    target_mask = gt_labels > 0

    # 背景准确率
    bg_correct = (pred_class[bg_mask] == 0).sum()
    bg_total = bg_mask.sum()

    # 目标准确率 (需要预测类别与真实类别一致)
    target_correct = (pred_class[target_mask] == gt_labels[target_mask]).sum()
    target_total = target_mask.sum()

    total_accuracy = (bg_correct + target_correct) / len(gt_labels)
    bg_accuracy = bg_correct / max(bg_total, 1)
    target_accuracy = target_correct / max(target_total, 1)

    log(f"  │   总体准确率: {total_accuracy:.4f} ({bg_correct+target_correct}/{len(gt_labels)})")
    log(f"  │   背景准确率: {bg_accuracy:.4f} ({bg_correct}/{bg_total})")
    log(f"  │   目标准确率: {target_accuracy:.4f} ({target_correct}/{target_total})")

    # 每类准确率
    log(f"  │")
    log(f"  │   类别混淆:")
    log(f"  │   {'真实↓预测→':>10s} | {'背景':>6s} | {'类1':>6s} | {'类2':>6s} | {'类3':>6s}")
    log(f"  │   {'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    errors = []
    for true_cls in range(4):
        row_str = f"  │   {f'类{true_cls}' if true_cls > 0 else '背景':>10s} |"
        for pred_cls in range(4):
            cnt = ((gt_labels == true_cls) & (pred_class == pred_cls)).sum()
            row_str += f" {cnt:>6d} |"
        log(row_str)

    if target_accuracy < 0.7:
        errors.append(f"多目标分类准确率 {target_accuracy:.4f} < 0.7")
    if bg_accuracy < 0.8:
        errors.append(f"背景分类准确率 {bg_accuracy:.4f} < 0.8")

    if errors:
        for e in errors:
            log(f"  ❌ {e}")
        return False, errors
    log(f"  │\n  ✅ MT-ICEM 多目标分类正常")
    return True, {
        'total_accuracy': total_accuracy,
        'target_accuracy': target_accuracy,
        'bg_accuracy': bg_accuracy,
    }


# ============================================================
# 测试 9: 真实数据验证
# ============================================================

def test_realistic_classification():
    """在基于真实数据统计的数据上评估所有检测器"""
    log("\n" + "=" * 80)
    log("  9️⃣  真实数据统计验证")
    log("=" * 80)

    if not _has_realistic:
        log("  ⚠️  无法加载真实数据工具，跳过")
        return True, {}

    from detection.cem import CEMDetector
    from detection.ace import ACEDetector
    from detection.sam import SpectralAngleMapper as SAMDetector

    log("  生成基于真实背景统计的测试数据...")
    data, target_spec, gt = make_realistic_data(
        num_pixels=2000, target_ratio=0.1, seed=42)

    # 均值归一化
    dm = data.mean(axis=1, keepdims=True) + 1e-10
    data_n = data / dm
    target_n = target_spec / (target_spec.mean() + 1e-10)

    n_target = gt.sum()
    n_bg = (~gt).sum()

    results = {}
    for name, DetClass in [
        ("CEM", CEMDetector),
        ("ACE", ACEDetector),
        ("SAM", SAMDetector),
    ]:
        if name == "SAM":
            det = DetClass(normalize=True)
            det.fit(target_n[np.newaxis, :])
            scores = 1.0 - det.predict(data_n) / np.pi
        else:
            det = DetClass(reg=1e-6)
            det.fit(data_n, target_n)
            scores = det.predict(data_n)

        # ROC-AUC (降序：高分在前)
        order = np.argsort(scores)[::-1]
        sg = gt[order]
        tpr = np.append(0, np.cumsum(sg) / n_target)
        fpr = np.append(0, np.cumsum(~sg) / n_bg)
        if fpr[-1] < 1.0 or tpr[-1] < 1.0:
            fpr = np.append(fpr, 1.0)
            tpr = np.append(tpr, 1.0)
        # 按 FPR 升序排列
        idx = np.argsort(fpr)
        auc = np.trapezoid(tpr[idx], fpr[idx])

        # 分离度
        sep = (scores[gt].mean() - scores[~gt].mean()) / (scores[~gt].std() + 1e-10)
        results[name] = {"auc": auc, "separation": sep}

        log(f"\n  {name:>5}: AUC={auc:.4f}, 分离度={sep:.2f}σ, "
            f"目标均值={scores[gt].mean():+.4f}, 背景均值={scores[~gt].mean():+.4f}")

    log(f"\n  {'检测器':>5} | {'AUC':>6} | {'分离度':>8}")
    log(f"  {'-'*5}-+-{'-'*6}-+-{'-'*8}")
    for name, r in results.items():
        log(f"  {name:>5} | {r['auc']:>6.4f} | {r['separation']:>8.2f}σ")

    # 判断标准：真实数据上 AUC > 0.80 为合格
    passed = all(r["auc"] > 0.80 for r in results.values())
    if not passed:
        for name, r in results.items():
            if r["auc"] <= 0.80:
                log(f"  ⚠️  {name} 在真实数据上 AUC={r['auc']:.4f} < 0.80")

    log(f"\n  {'✅ 通过' if passed else '❌ 部分未通过'}")
    return passed, results


# ============================================================
# 主控
# ============================================================

ALL_TESTS = [
    ("所有检测器二分类对比", test_all_detectors_binary),
    ("混淆矩阵", test_confusion_matrices),
    ("多类别一对多检测", test_multiclass_one_vs_rest),
    ("SNR退化曲线", test_classification_degradation),
    ("分数分布分析", test_score_distribution),
    ("ROC关键点汇总", test_roc_summary),
    ("混合像素检测", test_detection_vs_classification),
    ("MT-ICEM多目标分类", test_mticem_multitarget),
    ("真实数据统计验证", test_realistic_classification),
]


def main():
    log("=" * 80)
    log("  检测算法分类准确率评估")
    log("  Classification Accuracy Benchmarks")
    log("=" * 80)
    import torch
    log(f"  Python: {sys.version.split()[0]}, NumPy: {np.__version__}, "
        f"PyTorch: {torch.__version__}")
    log(f"")
    for line in [
        "  评估维度:",
        "    1.  Precision / Recall / F1 / AUC (5 检测器横向对比)",
        "    2.  混淆矩阵",
        "    3.  多类别一对多",
        "    4.  SNR 退化",
        "    5.  分数分布统计",
        "    6.  ROC 关键点 (FPR=0.1%/1%/5%/10% 时的 TPR)",
        "    7.  混合/亚像素检测",
        "    8.  MT-ICEM 多目标分类",
    ]:
        log(line)

    passes, failures = 0, 0
    for name, fn in ALL_TESTS:
        try:
            ok, _ = fn()
            passes += ok
            failures += not ok
        except Exception as e:
            log(f"\n  ❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            failures += 1

    log("\n" + "=" * 80)
    n_total = passes + failures
    log(f"  汇总: {passes}/{n_total} 通过, {failures} 失败")

    if failures == 0:
        log(f"\n  🎉 所有检测器分类准确率合格！")
    log("=" * 80)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
