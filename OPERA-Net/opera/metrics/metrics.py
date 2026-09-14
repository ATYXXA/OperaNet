"""评测指标：EER 与 min-tDCF（论文 4.1 节使用的两个指标）。

约定
----
- 分数越大越像 spoof（deepfake）
- 标签：0 = bonafide，1 = spoof
- EER 为 FAR 与 FRR 相等处的错误率
- min-tDCF 遵循 ASVspoof 2021 官方定义；当没有 ASV 子系统分数时，
  退化为归一化的 min-DCF（论文未提供 ASV 配置，这是本复现的默认路径）
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
# EER
# ---------------------------------------------------------------------- #
def compute_eer(scores, labels) -> Tuple[float, float]:
    """等错误率。

    Returns
    -------
    (eer, threshold)：eer 取值 0~1，threshold 为对应的判决门限
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()
    if scores.size == 0:
        return float("nan"), float("nan")

    # 按分数降序排列，前缀逐个作为候选门限
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    l_sorted = labels[order]

    n_bon = int(np.sum(l_sorted == 0))
    n_spoof = int(np.sum(l_sorted == 1))
    if n_bon == 0 or n_spoof == 0:
        return float("nan"), float("nan")

    tp = np.cumsum(l_sorted == 1)           # 判为 spoof 且确实是 spoof
    fp = np.cumsum(l_sorted == 0)           # 判为 spoof 但其实是 bonafide
    far = fp / n_bon
    frr = (n_spoof - tp) / n_spoof

    # 同分并列时只保留该分数段的最后一个位置
    tail = np.r_[np.diff(s_sorted) != 0, True]
    far, frr, thr = far[tail], frr[tail], s_sorted[tail]

    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0), float(thr[idx])


def compute_det_curve(scores, labels) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (FAR, FRR, thresholds) 曲线，用于画 DET 图。

    与 ASVspoof 官方实现保持一致：在曲线前端补上 (FAR=1, FRR=0) 的端点，
    保证门限取遍全部工作点。
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()
    order = np.argsort(-scores, kind="mergesort")
    s_sorted, l_sorted = scores[order], labels[order]
    n_bon = max(int(np.sum(l_sorted == 0)), 1)
    n_spoof = max(int(np.sum(l_sorted == 1)), 1)
    tp = np.cumsum(l_sorted == 1)
    fp = np.cumsum(l_sorted == 0)
    far = np.r_[1.0, fp / n_bon]                 # 门限最低处：全部判为 spoof
    frr = np.r_[0.0, (n_spoof - tp) / n_spoof]
    thr = np.r_[s_sorted[0] + 1e-6, s_sorted] if s_sorted.size else s_sorted
    return far, frr, thr


def compute_accuracy(scores, labels, threshold: float = 0.0) -> float:
    pred = (np.asarray(scores, dtype=np.float64) > threshold).astype(np.int32)
    return float((pred == np.asarray(labels, dtype=np.int32)).mean())


# ---------------------------------------------------------------------- #
# min-tDCF
# ---------------------------------------------------------------------- #
# SingFake 官方评估脚本（SingFake-main/models/*/evaluate_tDCF_asvspoof19.py）
# 采用的是 ASVspoof **2019** 口径；CtrSVDD 官方仓库不提供 t-DCF。
# 本论文最可能沿用 SingFake 的脚本，因此 2019 口径设为默认。
ASVSPOOF2019_COST_MODEL = {
    "Pspoof": 0.05,
    "Ptar": (1 - 0.05) * 0.99,      # 0.9405
    "Pnon": (1 - 0.05) * 0.01,      # 0.0095
    "Cmiss_asv": 1.0,
    "Cfa_asv": 10.0,
    "Cmiss_cm": 1.0,
    "Cfa_cm": 10.0,
}

# ASVspoof 2021 沿用同一套先验，本模块不再实现其他变体：
# 实测把 t-DCF 改写成 C0/C1/C2 形式后会得到 >1 的退化值，不是官方定义。
ASVSPOOF2021_COST_MODEL = dict(ASVSPOOF2019_COST_MODEL)


def obtain_asv_error_rates(
    tar_asv: Sequence[float],
    non_asv: Sequence[float],
    spoof_asv: Optional[Sequence[float]] = None,
    asv_threshold: Optional[float] = None,
) -> Tuple[float, float, Optional[float]]:
    """由 ASV 子系统的三组分数估计 (Pfa_asv, Pmiss_asv, Pmiss_spoof_asv)。

    与 SingFake 官方 ``eval_metrics.obtain_asv_error_rates`` 一致：
    默认把 ASV 工作点固定在其自身的 EER 门限处。
    """
    tar = np.asarray(tar_asv, dtype=np.float64)
    non = np.asarray(non_asv, dtype=np.float64)
    if tar.size == 0 or non.size == 0:
        return float("nan"), float("nan"), None
    if asv_threshold is None:
        # 合成一份 (scores, labels) 再求 EER，得到 ASV 自身的 EER 工作点
        all_scores = np.r_[tar, non]
        all_labels = np.r_[np.ones(tar.size), np.zeros(non.size)]
        _, asv_threshold = compute_eer(all_scores, all_labels)
    p_fa = float(np.mean(non >= asv_threshold))
    p_miss = float(np.mean(tar < asv_threshold))
    p_miss_spoof = None
    if spoof_asv is not None and len(spoof_asv) > 0:
        sp = np.asarray(spoof_asv, dtype=np.float64)
        p_miss_spoof = float(np.mean(sp < asv_threshold))
    return p_miss, p_fa, p_miss_spoof


def compute_min_tdcf(
    cm_scores,
    cm_labels,
    asv_scores: Optional[Sequence[float]] = None,
    asv_labels: Optional[Sequence[int]] = None,
    p_miss_asv: Optional[float] = None,
    p_fa_asv: Optional[float] = None,
    p_miss_spoof_asv: Optional[float] = None,
    version: str = "2019",
    p_spoof: float = 0.05,
    p_tar: float = 0.9405,
    p_non: float = 0.0095,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
) -> Tuple[float, float, bool]:
    """最小归一化 tandem 检测代价函数。

    官方口径（ASVspoof 2019，SingFake 官方脚本同款）
    ----------------------------------------------
        C1 = Ptar·(Cmiss_cm − Cmiss_asv·Pmiss_asv) − Pnon·Cfa_asv·Pfa_asv
        C2 = Cfa_cm·Pspoof·(1 − Pmiss_spoof_asv)
        tDCF(θ)      = C1·Pmiss_cm(θ) + C2·Pfa_cm(θ)
        tDCF_norm(θ) = tDCF(θ) / min(C1, C2)
        min-tDCF     = min_θ tDCF_norm(θ)

    先验与代价取 ASVspoof 2019 评估计划的固定值：
    Pspoof=0.05，Ptar=0.9405，Pnon=0.0095，Cmiss=Cmiss_asv=1，Cfa=Cfa_asv=10。

    无 ASV 信息（本仓库默认情形）
    ----------------------------
        退化为归一化 min-DCF（**不是** t-DCF，不可与论文数值直接比较）：
        DCF(θ) = Cmiss·Pmiss(θ)·Ptar + Cfa·Pfa(θ)·Pnon
        min-DCF_norm = min_θ DCF(θ) / min(Cmiss·Ptar, Cfa·Pnon)

    Returns
    -------
    (min_tdcf, act_dcf_asv, is_true_tdcf)
        ``is_true_tdcf=False`` 表示算的是退化版 min-DCF。
    """
    cm_scores = np.asarray(cm_scores, dtype=np.float64).ravel()
    cm_labels = np.asarray(cm_labels, dtype=np.int32).ravel()
    far, frr, _ = compute_det_curve(cm_scores, cm_labels)   # Pfa_cm, Pmiss_cm
    p_miss_cm, p_fa_cm = frr, far

    # 给 ASV 分数时，先折算成三个错误率
    if (p_miss_asv is None or p_fa_asv is None) and asv_scores is not None:
        asv_scores = np.asarray(asv_scores, dtype=np.float64)
        asv_labels = np.asarray(asv_labels, dtype=np.int32)
        tar = asv_scores[asv_labels == 1]
        non = asv_scores[asv_labels == 0]
        spf = asv_scores[asv_labels == 2] if (asv_labels == 2).any() else None
        p_miss_asv, p_fa_asv, p_miss_spoof_asv = obtain_asv_error_rates(tar, non, spf)

    have_asv = (p_miss_asv is not None and p_fa_asv is not None
                and p_miss_asv == p_miss_asv and p_fa_asv == p_fa_asv)

    if not have_asv:
        # ---- 退化：归一化 min-DCF ----
        dcf = c_miss * p_miss_cm * p_tar + c_fa * p_fa_cm * p_non
        norm = min(c_miss * p_tar, c_fa * p_non)
        if norm <= 0:
            return float("nan"), float("nan"), False
        return float(np.min(dcf) / norm), float("nan"), False

    if version != "2019":
        raise ValueError(f"未知 t-DCF 口径 {version!r}；"
                         "本模块只实现 ASVspoof 2019 官方定义（SingFake 同款）")

    act_dcf = c_miss * p_miss_asv * p_tar + c_fa * p_fa_asv * p_non
    c1 = p_tar * (c_miss - c_miss * p_miss_asv) - p_non * c_fa * p_fa_asv
    p_ms = 0.0 if p_miss_spoof_asv is None else p_miss_spoof_asv
    c2 = c_fa * p_spoof * (1.0 - p_ms)
    denom = min(c1, c2)
    if denom <= 0:
        return float("nan"), float(act_dcf), True
    tdcf = (c1 * p_miss_cm + c2 * p_fa_cm) / denom
    return float(np.min(tdcf)), float(act_dcf), True


def evaluate_all(
    scores,
    labels,
    asv_scores=None,
    asv_labels=None,
    **tdcf_kwargs,
) -> Dict[str, float]:
    """一次性给出 EER 与 min-tDCF（含百分数形式，便于直接填表）。"""
    eer, thr = compute_eer(scores, labels)
    min_tdcf, act_dcf, is_true_tdcf = compute_min_tdcf(
        scores, labels, asv_scores, asv_labels, **tdcf_kwargs)
    labels = np.asarray(labels, dtype=np.int32)
    return {
        "EER_%": round(eer * 100, 4),
        "min_tDCF": round(min_tdcf, 6) if min_tdcf == min_tdcf else float("nan"),
        "tDCF_is_true_tandem": bool(is_true_tdcf),
        "threshold": round(thr, 6) if thr == thr else float("nan"),
        "actDCF_asv": round(act_dcf, 6) if act_dcf == act_dcf else float("nan"),
        "n_bonafide": int(np.sum(labels == 0)),
        "n_spoof": int(np.sum(labels == 1)),
    }


def track_level_aggregate(seg_scores, seg_track_ids, seg_labels) -> Tuple[np.ndarray, np.ndarray]:
    """把片段级分数按曲目聚合为曲目级分数（SingFake 长音频评估用）。

    同一曲目的所有片段分数取均值，标签取该曲目的标签。
    """
    seg_scores = np.asarray(seg_scores, dtype=np.float64)
    seg_labels = np.asarray(seg_labels, dtype=np.int32)
    out_s, out_l, seen = [], [], {}
    for s, tid, l in zip(seg_scores, seg_track_ids, seg_labels):
        if tid not in seen:
            seen[tid] = len(out_s)
            out_s.append([])
            out_l.append(l)
        out_s[seen[tid]].append(s)
    return np.array([np.mean(v) for v in out_s]), np.array(out_l, dtype=np.int32)
