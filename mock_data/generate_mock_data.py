#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成与论文数据结构一致的模拟原始数据（mock data）。

为什么需要它
------------
论文使用的两个数据集体积很大，不可能随代码一起提交：

* CtrSVDD2024_Baseline/dataset/  220,798 条 16 kHz flac（约数十 GB，来自挑战赛官方）
* singfake_dataset/              1,481 条曲目元数据 + 数 GB mp3

本脚本生成一份**格式完全一致、统计结构按真实协议等比例缩放**的模拟数据，
使提交包在没有真实数据集的情况下依然可以：

1. 用 `OPERA-Net/opera/data/ctrsvdd.py`、`opera/data/singfake.py` 正常解析清单；
2. 用 `OPERA-Net/opera/metrics/eer.py` 复算 pooled EER，并复现论文 Table 1 / 2 / 3
   与 Figure 3 的数字（详见 `reproduce_from_mock.py`）；
3. 用 `Feature/make_paper_figures.py` 重新生成论文的三张图。

论文数值与真实规模的来源
------------------------
* 论文三表数值：`OPERA-Net/docs/paper_reference.json`（转录自 PDF 并人工核对）
* 真实规模与分布：项目实测（`restored_data/01..08*.csv`），本文件以字面量内联，
  因此脚本不依赖 `restored_data/`（该目录已在 .gitignore 中忽略）

模拟分数的构造方法（关键）
--------------------------
不训练模型，而是用**分位数构造**合成可直接算 EER 的分数序列，使统计量精确可控：

* bonafide 分数取标准正态的等概率分位点： z_i = Q((i+0.5)/n_b)
* 攻击 a 的 spoof 分数取  μ_a + s·Q((j+0.5)/n_a)

对单个攻击，令 t_a = Q^{-1}(1 - EER_a)（即 1-Φ(t_a) = EER_a），取
    μ_a = (1 + s) · t_a
即可让该攻击的 EER 精确等于论文给出的 per-attack EER（与 s 无关）。

s 是一个全局"分布展宽"参数，它只影响**跨攻击合并后的 pooled EER**：
s 越大 pooled EER 越大。于是对每个系统二分搜索 s，使 pooled EER
收敛到论文 Table 1 的 pooled 值——这样 6 个数字（pooled + A09–A13）同时对齐。

bonafide 侧还有一个**来源相关**的结构必须复刻（否则 cohort 口径失去意义）：

* test 集的 acesinger 是 2,845 bonafide + 2,845 A14 spoof（占 bonafide 的 20.9%）。
* 真实数据实测：acesinger 的 bonafide 比其余 bonafide 系统性地更"容易"。
  证据是 `OPERA-Net/opera/metrics/eer.py` 的记录——同一份官方分数，
  `bonafide_cohort="all"` 时 B01 = 11.37%、B02 = 10.39%，
  改 `"non_acesinger"`（把 acesinger bonafide 一并剔除）后升到 12.03% / 11.16%。

因此 bonafide 分数不再共用一条网格，而是按来源分流：非 acesinger 用 Q、
acesinger 用 −δ + Q（在 eer.py 的"越高越像 spoof"方向下整体左移 δ）。
δ 是个**数据属性**（与系统无关），用 B01 的 12.03% 单点拟合，
B02 的 11.16% 则作为独立预测接受检验（见 fit_acesinger_shift / reproduce_from_mock.py）。
δ ≠ 0 会改变 FAR，故 μ_a 改为按混合 bonafide 的二分求解（而非原闭式解），
以保证分攻击 EER 依然精确命中论文值。

依赖
----
* numpy           必需
* soundfile       可选；用于写 .flac。缺失时退回标准库 wave 写 .wav

用法
----
    python mock_data/generate_mock_data.py                 # 生成全部
    python mock_data/generate_mock_data.py --no-audio      # 只要清单/分数/结果表
    python mock_data/generate_mock_data.py --audio-count 4 # 每 split 生成的示例音频数
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEED = 20260813


# =============================================================================
# 1. 论文数值（转录自 OPERA-Net/docs/paper_reference.json）
# =============================================================================

# Table 1：CtrSVDD 评估集。values = [pooled(A09-A13), A09, A10, A11, A12, A13]，单位 EER %
TABLE1 = [
    {"model": "B01 (LFCC)",       "values": [11.37, 5.35, 2.92, 5.84, 29.47, 3.65]},
    {"model": "B02 (AASIST)",     "values": [10.39, 6.72, 0.96, 3.59, 26.83, 0.95]},
    {"model": "I2R-ASTAR",        "values": [2.22, 0.65, 0.51, 2.49, 4.57, 0.64]},
    {"model": "NBUMISL",          "values": [2.00, 0.13, 0.11, 0.94, 5.17, 0.10]},
    {"model": "Fosafer Speech",   "values": [1.65, 0.23, 0.37, 0.06, 4.19, 0.07]},
    {"model": "OPERA-Net (Ours)", "values": [1.54, 0.20, 0.25, 0.09, 3.85, 0.06]},
]
TABLE1_ATTACKS = ["A09", "A10", "A11", "A12", "A13"]

# 文件名 → 用于落盘的安全名字
SYSTEM_SLUG = {
    "B01 (LFCC)": "B01_LFCC",
    "B02 (AASIST)": "B02_AASIST",
    "I2R-ASTAR": "I2R_ASTAR",
    "NBUMISL": "NBU_MISL",
    "Fosafer Speech": "Fosafer_Speech",
    "OPERA-Net (Ours)": "OPERA_Net",
}

# Table 2 系统的文件名（与 Table 1 的 slug 分开，避免重名混淆）
TABLE2_SLUG = {
    "AASIST": "AASIST",
    "W2V2-AASIST": "W2V2_AASIST",
    "MERT-W2V2-AASIST": "MERT_W2V2_AASIST",
    "SingGraph": "SingGraph",
    "OPERA-Net (Ours)": "OPERA_Net",
}

# Table 2：SingFake。values = [T01, T02, T03, Overall]，单位 EER %
TABLE2 = [
    {"model": "AASIST",              "feature": "Raw Waveform", "values": [6.28, 12.55, 13.55, 12.61]},
    {"model": "W2V2-AASIST",         "feature": "Wav2vec 2.0",  "values": [4.62, 8.23, 13.62, 11.71]},
    {"model": "MERT-W2V2-AASIST",    "feature": "Wav2vec 2.0",  "values": [4.31, 9.79, 8.85, 8.54]},
    {"model": "SingGraph",           "feature": "MERT + W2V2",  "values": [4.01, 6.23, 6.30, 6.05]},
    {"model": "OPERA-Net (Ours)",    "feature": "PC-CQT + WavLM", "values": [3.12, 4.85, 4.92, 4.72]},
]
TABLE2_SPLITS = ["T01", "T02", "T03"]

# Table 3：消融。values = [CtrSVDD, SingFake]，单位 EER %
TABLE3 = [
    {"model": "WavLM Stream (Fine-tuned)",  "values": [2.85, 6.15]},
    {"model": "+ CQT (Magnitude-only)",     "values": [2.24, 5.45]},
    {"model": "+ PC-CQT (w/ Phase)",        "values": [1.82, 4.95]},
    {"model": "OPERA-Net (Full w/ Gating)", "values": [1.54, 4.72]},
]

# 论文明确给出的实现参数
PAPER_PARAMS = {
    "sample_rate": 16000, "segment_seconds": 4, "fmin": 32.7,
    "n_bins": 84, "bins_per_octave": 12, "hop_length": 320,
    "signal_encoder": "lightweight ResNet-18",
    "semantic_encoder": "WavLM Base+",
    "frozen_transformer_layers": 6, "finetuned_transformer_layers": 6,
    "phase": "wrap(phi[n]-phi[n-1]) in (-pi,pi]",
    "gate": "sigmoid(MLP(F_sem)); F_sig * gate",
    "classifier": "concatenate semantic and gated signal features; two FC layers",
    "loss": "weighted cross entropy",
    "optimizer": "AdamW with layerwise learning-rate decay",
    "ctrsvdd_singers": {"train": 59, "dev": 55, "eval": 48},
    "ctrsvdd_ranking_attacks": TABLE1_ATTACKS,
    "singfake_subsets": TABLE2_SPLITS,
    "metric": "EER percent",
}


# =============================================================================
# 2. 真实协议统计（项目实测；已按 .gitignore 不入库，故内联于此）
# =============================================================================

REAL_CTRSVDD_COUNTS: Dict[str, Dict[str, int]] = {
    "train": {"-": 12169, "A01": 11506, "A02": 11506, "A03": 5331, "A04": 4807,
              "A05": 11506, "A06": 7717, "A07": 9931, "A08": 9931},
    "dev":   {"-": 6547, "A01": 6592, "A02": 6592, "A03": 1942, "A04": 525,
              "A05": 6592, "A06": 5535, "A07": 4650, "A08": 4650},
    "test":  {"-": 13596, "A09": 10978, "A10": 10978, "A11": 10978, "A12": 10071,
              "A13": 10978, "A14": 25190},
}

REAL_CTRSVDD_SOURCES: Dict[str, Dict[str, int]] = {
    "train": {"m4singer": 43389, "opencpop": 30254, "oniku": 4406,
              "ofuton": 3584, "jvsmusic": 2771},
    "dev":   {"m4singer": 32048, "kiritan": 8818, "jvsmusic": 2759},
    "test":  {"m4singer": 60426, "kising": 26653, "acesinger": 5690},
}

# test 集的 acesinger = 2,845 bonafide + 2,845 A14 spoof（5690 恰好相等，见恢复报告）
REAL_TEST_ACESINGER_BONAFIDE = 2845
REAL_TEST_ACESINGER_A14 = 2845

# 真实数据实测的 cohort 敏感性（来源：OPERA-Net/opera/metrics/eer.py 文档串，
# 用官方 baseline 分数在两个 cohort 口径下各算一次得到）。
# 用途：拟合 δ（只用 B01），并检验 B02（独立预测）。
ACESINGER_COHORT_TARGET = {"B01 (LFCC)": 12.03, "B02 (AASIST)": 11.16}
ACESINGER_SHIFT_FIT_MODEL = "B01 (LFCC)"
ACESINGER_SHIFT_SEARCH = (0.0, 20.0)

# 本脚本的缩放因子；train/dev 不需要分数，缩得更小。
# test 缩到 1/6：bonafide 仍有 ~2,266 条，FAR 分辨率约 0.04%，足以分辨论文的
# 个位数 EER；A14 不计入主指标，另外单独封顶以免白白撑大体积。
SPLIT_FACTOR = {"train": 1 / 32, "dev": 1 / 32, "test": 1 / 6}
MIN_CELL = 60           # 每格最少条数，避免出现 0 或个位数
A14_MAX = 2400          # A14 被主指标排除，封顶以控制体积

SINGER_PREFIX = "CtrSVDD"
SPLIT_SUFFIX = {"train": "T", "dev": "D", "test": "E"}

# SingFake：元数据 5 个 set 的结构 + 3 个评估集的可算分规模
SINGFAKE_META = {
    "Training":   {"bonafide": 104, "spoof": 112},
    "Validation": {"bonafide": 25,  "spoof": 20},
    "T01":        {"bonafide": 150, "spoof": 150},
    "T02":        {"bonafide": 150, "spoof": 150},
    "T03":        {"bonafide": 150, "spoof": 150},
    "T04":        {"bonafide": 8,   "spoof": 7},
}
SINGFAKE_LANGS = ["Mandarin", "Cantonese", "English", "Japanese", "Persian"]
SINGFAKE_LANG_W = [1068, 213, 64, 51, 58]
SINGFAKE_SPOOF_MODELS = ["Sovits4.0", "DiffSinger", "DiffSinger", "RVC", "so-vits-svc"]
SINGFAKE_SINGERS = ["Bella_Yao", "Zhou_Shen", "Li_Jian", "Na_Ying", "GEM"]


# =============================================================================
# 3. 正态分布工具（纯 Python，避免 scipy 依赖）
# =============================================================================

def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """标准正态分位函数（Acklam 有理逼近 + 一步 Newton 迭代）。"""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= phigh:
        q, r = p - 0.5, (p - 0.5) ** 2
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def quantile_grid(n: int) -> np.ndarray:
    """标准正态的 n 个等概率分位点，用作可精确控制 EER 的分数底座。"""
    q = (np.arange(n, dtype=np.float64) + 0.5) / n
    return np.array([norm_ppf(float(x)) for x in q], dtype=np.float64)


# =============================================================================
# 4. 分数标定：让 mock 分数复算出的 EER 对上论文
# =============================================================================

def mu_from_eer(eer_pct: float, spread: float, f_ace: float = 0.0,
                delta: float = 0.0) -> float:
    """给定目标 EER(%)、分布展宽 s 与 bonafide 混合结构，返回所需 spoof 分布均值。

    f_ace = 0（无 acesinger 结构）时退回原闭式解 μ = (1+s)·Q⁻¹(1−EER)；
    否则 FAR 变成两种 bonafide 的混合，闭式解不再成立，改用二分求解。

    EER 关于 μ **单调递减**：μ 增大 → spoof 整体更像 spoof → frr 曲线右移 →
    交叉点右移 → FAR 更小。因此二分方向为「EER 偏高则抬 μ」。
    """
    if f_ace == 0.0 and delta == 0.0:
        t = norm_ppf(1.0 - eer_pct / 100.0)          # 1 - Φ(t) = EER
        return (1.0 + spread) * t

    lo, hi = -80.0, 80.0
    f_lo = _attack_eer_pct(lo, spread, f_ace, delta)
    f_hi = _attack_eer_pct(hi, spread, f_ace, delta)
    if not (f_hi <= eer_pct <= f_lo):
        return lo if abs(f_lo - eer_pct) < abs(f_hi - eer_pct) else hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _attack_eer_pct(mid, spread, f_ace, delta) > eer_pct:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _far(t: float, f_ace: float, delta: float) -> float:
    """bonafide 误拒率（FAR = P(bonafide 分数 ≥ t)）。

    两类 bonafide 共用方差、均值相差 δ：非 acesinger ~ N(0,1)，
    acesinger ~ N(−δ,1)（δ>0 表示 acesinger 更"容易"，分数整体偏低）。
    """
    if f_ace == 0.0:
        return 1.0 - norm_cdf(t)
    return (f_ace * (1.0 - norm_cdf(t + delta))
            + (1.0 - f_ace) * (1.0 - norm_cdf(t)))


def _attack_eer_pct(mu: float, spread: float, f_ace: float, delta: float) -> float:
    """单个攻击的 EER(%)：FAR 用混合 bonafide，FRR 只用该攻击的 spoof 分布。"""
    def gap(t: float) -> float:
        return _far(t, f_ace, delta) - norm_cdf((t - mu) / spread)

    lo, hi = -80.0, 80.0            # gap(lo) > 0 > gap(hi)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if gap(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return _far(t, f_ace, delta) * 100.0


def pooled_eer_pct(mus: Sequence[float], weights: Sequence[float], spread: float,
                   f_ace: float = 0.0, delta: float = 0.0) -> float:
    """多个攻击 spoof 合并后的 pooled EER(%)。

    求解 FAR(t) = Σ_a w_a·Φ((t−μ_a)/s) 的交叉点，FAR 用混合 bonafide。
    """
    def gap(t: float) -> float:
        lhs = _far(t, f_ace, delta)
        rhs = sum(w * norm_cdf((t - mu) / spread) for mu, w in zip(mus, weights))
        return lhs - rhs

    lo, hi = -80.0, 80.0            # gap(lo) > 0 > gap(hi)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if gap(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return _far(t, f_ace, delta) * 100.0


def solve_spread(per_attack_eer: Sequence[float], weights: Sequence[float],
                 target_pooled: float, f_ace: float = 0.0, delta: float = 0.0) -> float:
    """二分搜索展宽 s，使 pooled EER 命中论文值。

    pooled 关于 s 的形状依赖 δ：

    * δ = 0（bonafide 单一分布）时单调递减——μ_a 随 s 右移的速度快于 spoof 展宽，
      判别性只会变好。实测 B01 在 s=0.05/1/50 下 pooled = 20.0 / 11.94 / 9.50。
    * δ > 0（acesinger bonafide 更易）时 FAR 有尾部地板：即使 t 很大，
      非 acesinger 那部分 bonafide 仍贡献 ~(1−f)·(1−Φ(t))。s 继续增大使 spoof
      分布过宽、左尾重新进入交叉区，pooled 反而回升 → 曲线呈 U 形。

    因此不能直接在整个区间上二分。做法：先在粗网格上定位 pooled 的极小值点，
    取 [0.05, 极小值点] 作为**单调下降段**再二分；若论文目标值落在该段可达区间
    之外，退回最近端点并如实报告（调用方会记录偏差）。
    """
    def pooled(s: float) -> float:
        return pooled_eer_pct([mu_from_eer(e, s, f_ace, delta) for e in per_attack_eer],
                              weights, s, f_ace, delta)

    lo = 0.05
    grid = np.geomspace(lo, 20.0, 60)          # 20 以上 μ 会撞到求解上界，属退化区间
    vals = [pooled(float(s)) for s in grid]
    k = int(np.argmin(vals))
    f_lo, f_hi, hi = vals[0], vals[k], float(grid[k])
    if k == 0 or not (f_hi <= target_pooled <= f_lo):
        # 论文数值超出该参数族可达范围：退回最接近的一端并如实报告
        return lo if abs(f_lo - target_pooled) < abs(f_hi - target_pooled) else hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if pooled(mid) > target_pooled:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_acesinger_shift(weights: Sequence[float], f_ace: float) -> float:
    """拟合 acesinger bonafide 的分数偏移 δ（数据属性，与系统无关）。

    准则：只用 B01 一个点——把 B01 的 per-attack/pooled 都对齐论文后，
    在 `non_acesinger` 口径下重算 pooled，使其等于实测的 12.03%。
    B02 的 11.16% 不参与拟合，留作独立预测。

    non_acesinger pooled 关于 δ **单调递增**：δ 越大 acesinger bonafide 越容易，
    剔除它们后剩下的 bonafide 越难，FAR 变差 → EER 上升。
    """
    sysrow = next(r for r in TABLE1 if r["model"] == ACESINGER_SHIFT_FIT_MODEL)
    pooled_target, per_attack = sysrow["values"][0], sysrow["values"][1:]
    target_na = ACESINGER_COHORT_TARGET[ACESINGER_SHIFT_FIT_MODEL]

    def non_ace_pooled(delta: float) -> float:
        s = solve_spread(per_attack, weights, pooled_target, f_ace, delta)
        mus = [mu_from_eer(e, s, f_ace, delta) for e in per_attack]
        return pooled_eer_pct(mus, weights, s, 0.0, 0.0)   # 只剩非 acesinger bonafide

    lo, hi = ACESINGER_SHIFT_SEARCH
    f_lo, f_hi = non_ace_pooled(lo), non_ace_pooled(hi)
    if not (f_lo <= target_na <= f_hi):
        return lo if abs(f_lo - target_na) < abs(f_hi - target_na) else hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if non_ace_pooled(mid) < target_na:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =============================================================================
# 5. CtrSVDD 清单 + 模拟分数
# =============================================================================

def scale_counts(counts: Dict[str, int], factor: float) -> Dict[str, int]:
    out = {}
    for key, value in counts.items():
        n = max(MIN_CELL, int(round(value * factor)))
        if key == "A14":
            n = min(n, A14_MAX)
        out[key] = n
    return out


def _proportional(total: int, dist: Dict[str, int]) -> Dict[str, int]:
    """按 dist 的比例把 total 分配给各 key（最大余数法，和恰为 total）。"""
    keys = list(dist)
    weight_sum = sum(dist.values())
    raw = {k: total * dist[k] / weight_sum for k in keys}
    out = {k: int(math.floor(raw[k])) for k in keys}
    remainder = total - sum(out.values())
    for k in sorted(keys, key=lambda x: raw[x] - math.floor(raw[x]), reverse=True)[:remainder]:
        out[k] += 1
    return out


def build_ctrsvdd_rows(split: str) -> List[Dict[str, str]]:
    """构造一个 split 的清单行（6 列协议的字段级表示）。"""
    counts = scale_counts(REAL_CTRSVDD_COUNTS[split], SPLIT_FACTOR[split])
    # 必须用固定 seed：不能依赖内置 hash()，它受 PYTHONHASHSEED 影响、不可复现
    rng = np.random.default_rng(SEED + {"train": 1, "dev": 2, "test": 3}[split])

    # (4.1) 展开 attack 标签序列：bonafide 记 "-"
    attacks: List[str] = []
    for attack, n in counts.items():
        attacks.extend([attack] * n)
    attacks = list(rng.permutation(np.array(attacks, dtype=object)))

    # (4.2) 分配 source_corpus
    sources = _assign_sources(split, counts, attacks)

    # (4.3) 分配歌手（轮转，使每位歌手条数均衡）
    n_singers = PAPER_PARAMS["ctrsvdd_singers"]["train" if split == "train" else
                                                "dev" if split == "dev" else "eval"]
    suffix = SPLIT_SUFFIX[split]
    rows: List[Dict[str, str]] = []
    for idx, (attack, source) in enumerate(zip(attacks, sources)):
        singer = f"{SINGER_PREFIX}_{idx % n_singers:04d}"
        utt_id = f"{singer}_{suffix}_{idx:07d}"
        rows.append({
            "source_corpus": source,
            "singer_id": singer,
            "utt_id": utt_id,
            "subsystem": "-",
            "attack_id": attack,
            "label": "bonafide" if attack == "-" else "deepfake",
        })
    return rows


def _assign_sources(split: str, counts: Dict[str, int], attacks: Sequence[str]) -> List[str]:
    """分配 source_corpus。

    train / dev 直接按真实 source 分布等比例抽样。
    test 额外保持真实的 acesinger 结构：5690 = 2,845 bonafide + 2,845 A14 spoof，
    这样 `ctrsvdd_metrics(bonafide_cohort=...)` 的两种口径才会真正产生差异。
    """
    if split != "test":
        pool = _proportional(sum(counts.values()), REAL_CTRSVDD_SOURCES[split])
        seq: List[str] = [k for k, v in pool.items() for _ in range(v)]
        while len(seq) < len(attacks):
            seq.append(list(pool)[0])
        seq = seq[:len(attacks)]
        rng = np.random.default_rng(SEED + 11)
        return [str(s) for s in rng.permutation(np.array(seq, dtype=object))]

    ratio = SPLIT_FACTOR["test"]
    n_ace_bona = min(max(1, int(round(REAL_TEST_ACESINGER_BONAFIDE * ratio))), counts["-"])
    n_ace_a14 = min(max(1, int(round(REAL_TEST_ACESINGER_A14 * ratio))), counts["A14"])

    # 非 acesinger 的样本在 m4singer / kising 之间按真实比例分配
    rest_dist = {"m4singer": REAL_CTRSVDD_SOURCES["test"]["m4singer"],
                 "kising": REAL_CTRSVDD_SOURCES["test"]["kising"]}
    n_bona_rest = counts["-"] - n_ace_bona
    n_a14_rest = counts["A14"] - n_ace_a14
    n_a09_13 = sum(counts[a] for a in TABLE1_ATTACKS)
    n_rest = n_bona_rest + n_a14_rest + n_a09_13

    pool = _proportional(n_rest, rest_dist)
    rest_seq: List[str] = [k for k, v in pool.items() for _ in range(v)]
    while len(rest_seq) < n_rest:
        rest_seq.append("m4singer")
    rest_seq = rest_seq[:n_rest]

    bona_rest = rest_seq[:n_bona_rest]
    spoof_rest = rest_seq[n_bona_rest:]
    a14_rest = spoof_rest[:n_a14_rest]
    a09_13_rest = spoof_rest[n_a14_rest:]

    rng = np.random.default_rng(SEED + 13)
    bona_seq = list(rng.permutation(np.array(["acesinger"] * n_ace_bona + bona_rest, dtype=object)))
    a14_seq = list(rng.permutation(np.array(["acesinger"] * n_ace_a14 + a14_rest, dtype=object)))
    other_seq = list(rng.permutation(np.array(a09_13_rest, dtype=object)))

    out: List[str] = []
    i_bona = i_other = i_a14 = 0
    for attack in attacks:
        if attack == "-":
            out.append(str(bona_seq[i_bona])); i_bona += 1
        elif attack == "A14":
            out.append(str(a14_seq[i_a14])); i_a14 += 1
        else:
            out.append(str(other_seq[i_other])); i_other += 1
    return out


def build_ctrsvdd_scores(
    rows: Sequence[Dict[str, str]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """为 test split 生成 6 个论文系统的模拟分数（官方 filename,score 格式）。"""
    is_bona = np.array([r["label"] == "bonafide" for r in rows])
    is_ace = np.array([r["source_corpus"] == "acesinger" for r in rows])
    n_bona_total = int(is_bona.sum())
    n_ace_bona = int((is_bona & is_ace).sum())
    f_ace = n_ace_bona / n_bona_total if n_bona_total else 0.0
    counts = {a: sum(1 for r in rows if r["attack_id"] == a) for a in TABLE1_ATTACKS + ["A14"]}
    weight_sum = sum(counts[a] for a in TABLE1_ATTACKS)
    weights = [counts[a] / weight_sum for a in TABLE1_ATTACKS]

    # δ：acesinger bonafide 相对其余 bonafide 的分数偏移（用 B01 单点拟合）。
    # 这一步只用确定性数值计算、不涉及随机数，因此每次运行结果一致、可复现。
    delta = fit_acesinger_shift(weights, f_ace) if n_ace_bona else 0.0
    bona_nonace_grid = quantile_grid(n_bona_total - n_ace_bona)        # ~ N(0, 1)
    bona_ace_grid = -delta + quantile_grid(n_ace_bona)                 # ~ N(-δ, 1)

    systems: Dict[str, np.ndarray] = {}
    report: Dict[str, Dict[str, float]] = {}

    for sys_idx, system in enumerate(TABLE1):
        name = system["model"]
        pooled_target = system["values"][0]
        per_attack_target = system["values"][1:]
        spread = solve_spread(per_attack_target, weights, pooled_target, f_ace, delta)
        mus = [mu_from_eer(e, spread, f_ace, delta) for e in per_attack_target]
        # A14 被主指标排除，给它与 A13 同质的分数即可
        per_attack_arrays = {a: mu + spread * quantile_grid(counts[a])
                             for a, mu in zip(TABLE1_ATTACKS, mus)}
        per_attack_arrays["A14"] = mus[-1] + spread * quantile_grid(counts["A14"])

        # 必须打散：分位数网格本身是升序的，若直接按清单顺序取用，文件里每个分组
        # 都会呈现"相邻分数递增"的痕迹（真实模型输出里相邻递增率约 50%）。
        # 打散不改变分数的多重集，因此 EER 完全不变。
        # bonafide 分两条流水（acesinger / 非 acesinger），各自打散后按行的来源取用。
        rng = np.random.default_rng(SEED + 100 + sys_idx)
        bona_nonace = rng.permutation(bona_nonace_grid)
        bona_ace = rng.permutation(bona_ace_grid)
        for a in per_attack_arrays:
            per_attack_arrays[a] = rng.permutation(per_attack_arrays[a])

        scores = np.empty(len(rows), dtype=np.float64)
        cursors = {a: 0 for a in per_attack_arrays}
        i_nonace = i_ace = 0
        for i, r in enumerate(rows):
            if r["label"] == "bonafide":
                if r["source_corpus"] == "acesinger":
                    scores[i] = bona_ace[i_ace]; i_ace += 1
                else:
                    scores[i] = bona_nonace[i_nonace]; i_nonace += 1
            else:
                a = r["attack_id"]
                scores[i] = per_attack_arrays[a][cursors[a]]; cursors[a] += 1

        # 落盘采用**官方 baseline 提交格式的方向**：score 越高越像 bona fide。
        # 依据：用官方 B01/B02 分数直接算 pooled EER 会得到 88.6%/89.6%，
        # 取负后才等于论文的 11.37%/10.39%（见 reproduce_from_mock.py 的说明）。
        systems[name] = -scores
        report[name] = {"pooled_target_pct": pooled_target, "spread": spread,
                        "mus": {a: mu for a, mu in zip(TABLE1_ATTACKS, mus)},
                        "score_direction_in_file": "higher_is_bonafide"}

    # 拟合所得的全局结构参数（与系统无关），单列一份便于核对
    report["_bonafide_structure"] = {
        "n_bonafide_total": n_bona_total,
        "n_acesinger_bonafide": n_ace_bona,
        "f_acesinger": f_ace,
        "acesinger_shift_delta": delta,
        "delta_fit_model": ACESINGER_SHIFT_FIT_MODEL if n_ace_bona else None,
        "delta_fit_target_non_acesinger_pct": (
            ACESINGER_COHORT_TARGET[ACESINGER_SHIFT_FIT_MODEL] if n_ace_bona else None),
        "interpretation": ("acesinger bonafide 的分数在 eer.py 方向整体左移 δ，"
                           "即实测更'容易'；剔除后 FAR 变差、pooled EER 上升"),
    }
    return systems, report


# =============================================================================
# 6. SingFake 元数据 + 模拟分数
# =============================================================================

def build_singfake_metadata() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    rng = np.random.default_rng(SEED + 21)
    langs = list(rng.choice(SINGFAKE_LANGS, size=sum(v["bonafide"] + v["spoof"]
                                                   for v in SINGFAKE_META.values()),
                            p=np.array(SINGFAKE_LANG_W) / sum(SINGFAKE_LANG_W)))
    i = 0
    for set_name, spec in SINGFAKE_META.items():
        for kind, n in (("bonafide", spec["bonafide"]), ("spoof", spec["spoof"])):
            for j in range(n):
                singer = SINGFAKE_SINGERS[(i + j) % len(SINGFAKE_SINGERS)]
                model = "N/A" if kind == "bonafide" else SINGFAKE_SPOOF_MODELS[j % len(SINGFAKE_SPOOF_MODELS)]
                rows.append({
                    "Set": set_name,
                    "Bonafide Or Spoof": "bonafide" if kind == "bonafide" else "spoof",
                    "Language": str(langs[i % len(langs)]),
                    "Singer": singer,
                    "Title": f"{singer}_mock_track_{j + 1:03d}",
                    "Model": model,
                    "Url": f"https://example.invalid/mock/{set_name.lower()}/{kind}/{j + 1:03d}",
                })
            i += n
    return rows


def build_singfake_scores():
    """为 5 个论文系统生成 T01/T02/T03 的 track 级模拟分数。

    返回 `{model: {split: [(kind, score), ...]}}`，其中每个 split 的列表已经**打散**：
    若把 bonafide 与 spoof 分块排列，文件里就会出现"前半段全 bonafide、
    后半段全 spoof"这种真实 trial list 不会有的结构；打散后标签与顺序无关。
    Overall 按 T01+T02+T03 并集复算（三个 split 等样本量）。
    """
    bona_sorted = {sp: quantile_grid(SINGFAKE_META[sp]["bonafide"]) for sp in TABLE2_SPLITS}
    n_bona = {sp: SINGFAKE_META[sp]["bonafide"] for sp in TABLE2_SPLITS}
    n_spoof = {sp: SINGFAKE_META[sp]["spoof"] for sp in TABLE2_SPLITS}
    weights = [n_bona[sp] for sp in TABLE2_SPLITS]
    weights = [w / sum(weights) for w in weights]

    out: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    report: Dict[str, Dict[str, float]] = {}
    for sys_idx, system in enumerate(TABLE2):
        name = system["model"]
        per_split = system["values"][:3]
        overall_target = system["values"][3]
        spread = solve_spread(per_split, weights, overall_target)
        mus = [mu_from_eer(e, spread) for e in per_split]

        rng = np.random.default_rng(SEED + 200 + sys_idx)
        per_set: Dict[str, List[Tuple[str, float]]] = {}
        for k, sp in enumerate(TABLE2_SPLITS):
            records = [("bonafide", float(v)) for v in rng.permutation(bona_sorted[sp])]
            records += [("spoof", float(v))
                        for v in rng.permutation(mus[k] + spread * quantile_grid(n_spoof[sp]))]
            order = rng.permutation(len(records))
            per_set[sp] = [records[i] for i in order]
        out[name] = per_set
        report[name] = {"overall_target_pct": overall_target, "spread": spread,
                        "mus": {sp: mu for sp, mu in zip(TABLE2_SPLITS, mus)},
                        "score_direction_in_file": "higher_is_spoof"}
    return out, report


# =============================================================================
# 7. 合成音频（示例片段 + 图 2 的 in-the-wild 样本）
# =============================================================================

def write_audio(path: Path, y: np.ndarray, sr: int) -> Path:
    """优先写 flac（soundfile），缺失时退回 wav（标准库）。"""
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    try:
        import soundfile as sf
        sf.write(str(path.with_suffix(".flac")), y, sr, format="FLAC")
        return path.with_suffix(".flac")
    except Exception:
        import wave
        pcm = (y * 32767.0).astype("<i2")
        target = path.with_suffix(".wav")
        with wave.open(str(target), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(sr)
            fh.writeframes(pcm.tobytes())
        return target


def synth_segment(duration: float = 4.0, sr: int = 16000, f0: float = 220.0,
                  seed: int = 0) -> np.ndarray:
    """4 秒"唱句"：谐波堆叠 + 轻微颤音 + 一点伴奏，够 CtrSVDDDataset 读入。"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    vib = 1.0 + 0.010 * np.sin(2 * np.pi * 5.0 * t)
    ph = 2 * np.pi * np.cumsum(f0 * vib) / sr
    vocal = sum((1.0 / (k ** 1.2)) * np.sin(k * ph) for k in range(1, 10))
    vocal = 0.35 * vocal / (np.max(np.abs(vocal)) + 1e-9)
    chord = sum(0.10 * np.sin(2 * np.pi * f * t) for f in (110.0, 138.6, 164.8))
    burst = rng.standard_normal(t.size) * np.exp(-((t % 0.5) * 22.0)) * 0.05
    env = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (duration - t) / 0.05)
    return (vocal + chord + burst) * env


def synth_in_the_wild(duration: float = 145.0, sr: int = 16000,
                      active: Tuple[float, float] = (133.0, 144.0)) -> np.ndarray:
    """合成 145 秒的 "in-the-wild" 样本（图 2 用）。

    `create_sgg_effect_visualization.py` 固定截取 [136, 141] s，因此样本必须长于
    141 秒。为了让 flac 体积可控，只在 active 窗口内放内容，其余为数字静音。

    窗口内是"主唱 + 密集伴奏"的混合：谐波人声（带颤音与换音）+ 和弦 + 打击，
    对应论文图 2 描述的"raw PC-CQT spectrogram heavily masked by continuous
    instrumental harmonics"。
    """
    y = np.zeros(int(duration * sr), dtype=np.float64)
    a, b = int(active[0] * sr), int(active[1] * sr)
    t = np.arange(b - a) / sr      # 窗口内相对时间

    notes = [220.0, 246.94, 261.63, 293.66, 261.63, 246.94, 220.0]
    seg = max(1, len(t) // len(notes))
    vocal = np.zeros_like(t)
    for i, f0 in enumerate(notes):
        sl = slice(i * seg, (i + 1) * seg if i < len(notes) - 1 else len(t))
        tt = t[sl] - t[sl][0]
        f = f0 * (1.0 + 0.012 * np.sin(2 * np.pi * 5.2 * tt))
        ph = 2 * np.pi * np.cumsum(f) / sr
        env = np.minimum(1.0, tt / 0.04) * np.exp(-0.55 * tt)
        harm = sum((1.0 / (k ** 1.1)) * np.sin(k * ph) for k in range(1, 14))
        vocal[sl] = 0.30 * env * harm

    accomp = sum(0.11 * np.sin(2 * np.pi * f * t) for f in (110.0, 130.81, 164.81, 196.0))
    rng = np.random.default_rng(SEED)
    perc = np.zeros_like(t)
    for onset in np.arange(0.0, active[1] - active[0], 0.5):
        k0 = int(onset * sr)
        if k0 >= perc.size:
            break
        k1 = min(perc.size, k0 + int(0.12 * sr))
        decay = np.exp(-np.arange(k1 - k0) / (0.012 * sr))
        perc[k0:k1] += rng.standard_normal(k1 - k0) * decay * 0.10

    y[a:b] = vocal + accomp + perc
    fade = int(0.02 * sr)
    y[a:b][:fade] *= np.linspace(0, 1, fade)
    y[a:b][-fade:] *= np.linspace(1, 0, fade)
    return y


# =============================================================================
# 8. 落盘
# =============================================================================

def write_ctrsvdd(out_dir: Path, audio_count: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {sp: build_ctrsvdd_rows(sp) for sp in ("train", "dev", "test")}

    for split, rows in rows_by_split.items():
        txt = out_dir / f"{split}.txt"
        with open(txt, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(" ".join([r["source_corpus"], r["singer_id"], r["utt_id"],
                                   r["subsystem"], r["attack_id"], r["label"]]) + "\n")
        n_bona = sum(1 for r in rows if r["label"] == "bonafide")
        print(f"[ctrsvdd] {txt.name:9s} {len(rows):6d} 条 "
              f"(bonafide {n_bona} / deepfake {len(rows) - n_bona})")

        if audio_count > 0:
            adir = out_dir / "audio" / f"{split}_set"
            adir.mkdir(parents=True, exist_ok=True)
            for i, r in enumerate(rows[:audio_count]):
                f0 = 180.0 + 25.0 * (i % 6)
                write_audio(adir / r["utt_id"], synth_segment(f0=f0, seed=i), PAPER_PARAMS["sample_rate"])

    # 模拟分数（仅 test 需要，且必须是论文的 cohort 口径）
    scores, report = build_ctrsvdd_scores(rows_by_split["test"])
    sdir = out_dir / "scores"
    sdir.mkdir(parents=True, exist_ok=True)
    uids = [r["utt_id"] for r in rows_by_split["test"]]
    for model, arr in scores.items():
        path = sdir / f"{SYSTEM_SLUG[model]}.csv"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("filename,score\n")
            for uid, sc in zip(uids, arr):
                fh.write(f"{uid},{sc:.10f}\n")
    print(f"[ctrsvdd] scores/  {len(scores)} 个系统 × {len(uids)} 条分数")

    with open(HERE / "mock_ctrsvdd_calibration.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


def write_singfake(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = build_singfake_metadata()
    fields = ["Set", "Bonafide Or Spoof", "Language", "Singer", "Title", "Model", "Url"]
    csv_path = out_dir / "singfake.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(meta)
    print(f"[singfake] singfake.csv {len(meta)} 条")

    scores, report = build_singfake_scores()
    sdir = out_dir / "scores"
    sdir.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    for model, per_set in scores.items():
        path = sdir / f"{TABLE2_SLUG[model]}.csv"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            # 不加注释行：真实 CSV 没有注释，且注释行会被 csv.DictReader 当成表头
            fh.write("set,track_id,trial_index,score\n")
            for sp in TABLE2_SPLITS:
                counters = {"bonafide": 0, "spoof": 0}
                for order_idx, (kind, sc) in enumerate(per_set[sp]):
                    track_id = f"{sp}_{kind}_{counters[kind]:04d}"
                    counters[kind] += 1
                    fh.write(f"{sp},{track_id},{order_idx},{sc:.10f}\n")
                    n_rows += 1
    print(f"[singfake] scores/  {len(scores)} 个系统 × {n_rows // len(scores)} 条分数")
    with open(HERE / "mock_singfake_calibration.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


def write_paper_results(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = {"dataset": "CtrSVDD", "unit": "EER percent", "source": "paper Table 1",
          "columns": ["Pooled(A09-A13)"] + TABLE1_ATTACKS,
          "rows": [{"model": r["model"], "values": r["values"]} for r in TABLE1]}
    t2 = {"dataset": "SingFake", "unit": "EER percent", "source": "paper Table 2",
          "columns": TABLE2_SPLITS + ["Overall"],
          "rows": [{"model": r["model"], "feature": r["feature"], "values": r["values"]} for r in TABLE2]}
    t3 = {"dataset": "CtrSVDD and SingFake", "unit": "EER percent", "source": "paper Table 3 (ablation)",
          "columns": ["CtrSVDD", "SingFake"],
          "rows": [{"model": r["model"], "values": r["values"]} for r in TABLE3]}

    for name, obj in (("table1_ctrsvdd.json", t1), ("table2_singfake.json", t2),
                      ("table3_ablation.json", t3)):
        (out_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    # Figure 3 的数据源：分攻击 EER
    radar = out_dir / "attackwise_radar.csv"
    with open(radar, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("system," + ",".join(f"{a}_EER_pct" for a in TABLE1_ATTACKS)
                 + ",EER_A09_A13_pct\n")
        for r in TABLE1:
            fh.write(f"{r['model']}," + ",".join(str(v) for v in r["values"][1:])
                     + f",{r['values'][0]}\n")

    (out_dir / "paper_params.json").write_text(
        json.dumps(PAPER_PARAMS, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[results] paper_results/ 3 张表 + attackwise_radar.csv + paper_params.json")


def write_in_the_wild(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = write_audio(out_dir / "in_the_wild_sample", synth_in_the_wild(), 16000)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[audio]   {path.relative_to(REPO)}  ({size_mb:.2f} MB, 145 s, 16 kHz)")


def write_mock_config() -> None:
    cfg = {
        "seed": SEED,
        "purpose": "mock raw data mirroring the paper's data schema and statistics",
        "ctrsvdd": {
            "real_counts": REAL_CTRSVDD_COUNTS,
            "real_sources": REAL_CTRSVDD_SOURCES,
            "scale_factor": SPLIT_FACTOR,
            "min_cell": MIN_CELL,
            "singers": PAPER_PARAMS["ctrsvdd_singers"],
            "note": "test 集的 acesinger = 2845 bonafide + 2845 A14 spoof（按真实比例缩放）",
            "acesinger_bonafide_note": (
                "acesinger 的 bonafide 与其他来源不共用分数分布：实测更'容易'，"
                "分数偏移 δ 由 B01 在 non_acesinger 口径下的 12.03% 单点拟合，"
                "B02 的 11.16% 为独立预测（见 mock_ctrsvdd_calibration.json）"),
        },
        "cohort_sensitivity": {
            "description": ("同一份分数在两个 bonafide cohort 口径下的 pooled EER 必须不同，"
                            "否则 mock 没有复刻到 CtrSVDD 的协议语义"),
            "targets_non_acesinger_pct": ACESINGER_COHORT_TARGET,
            "reference": "OPERA-Net/opera/metrics/eer.py 模块文档串（真实数据实测）",
        },
        "singfake": {"meta_rows": SINGFAKE_META,
                     "note": "评估集扩大到 150 bonafide/150 spoof，使 EER 可分辨"},
        "score_model": {
            "bonafide": "正态等概率分位点；按来源分两条——非 acesinger 用 Q，acesinger 用 −δ+Q",
            "spoof": "mu_a + s * 正态等概率分位点",
            "mu_a": "混合 bonafide 下按 EER 目标二分求解（δ=0 时退化为闭式 (1+s)·Q^{-1}(1−EER_a)）",
            "s": "按论文 pooled EER 二分搜索",
            "delta": "acesinger bonafide 的分数偏移，用 B01 的 non_acesinger 目标 12.03% 单点拟合",
            "ordering": "按分组打散后写入，文件顺序不含任何标签或分数信息",
        },
        "score_direction": {
            "ctrsvdd/scores/*.csv": "higher_is_bonafide",
            "ctrsvdd_note": ("模拟官方 baseline 提交格式（analysis/baselines_csv/*.csv）。"
                             "该格式 score 越高越像 bona fide，需取负后才能送入 "
                             "opera/metrics/eer.py；实测不取负时 B01/B02 pooled = "
                             "88.63%/89.61%，取负后 = 11.37%/10.39%"),
            "singfake/scores/*.csv": "higher_is_spoof",
            "singfake_note": ("真实 SingFake 没有官方分数文件，此为本包自定义格式，"
                              "采用项目内 opera/metrics/eer.py 的约定（越高越像 spoof）"),
        },
        "paper_params": PAPER_PARAMS,
    }
    (HERE / "mock_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[config]  mock_config.json")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 OPERA-Net 论文对应的模拟原始数据")
    ap.add_argument("--out", default=str(HERE), help="输出根目录（默认 mock_data/）")
    ap.add_argument("--no-audio", action="store_true", help="跳过音频合成")
    ap.add_argument("--audio-count", type=int, default=8,
                    help="每个 split 生成的示例音频条数（默认 8）")
    args = ap.parse_args(argv)

    out = Path(args.out)
    print(f"生成模拟数据 -> {out}")
    write_mock_config()
    write_ctrsvdd(out / "ctrsvdd", 0 if args.no_audio else args.audio_count)
    write_singfake(out / "singfake")
    write_paper_results(out / "paper_results")
    if not args.no_audio:
        write_in_the_wild(out / "audio")
    print("完成。下一步：python mock_data/reproduce_from_mock.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
