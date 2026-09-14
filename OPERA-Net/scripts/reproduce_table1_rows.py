"""复现论文 Table 1 中 B01 (LFCC) / B02 (AASIST) 两行。

论文 Table 1 报告 SVDD 2024 Challenge 官方 baseline 在 CtrSVDD evaluation set 上
的 pooled EER（A09–A13）与分攻击 EER。项目内保有官方分数文件
（analysis/baselines_csv/B01.csv、B02.csv）与 ground truth（92,769 条），
因此这两行是论文结果中**唯一可用本地数据独立复算**的部分。

本脚本同时用四种口径重算，用于判定论文数值的来源：

  A. eer.py `ctrsvdd_metrics` —— 挑战赛 notebook 口径
     （排除 attack A14 的 spoof，且排除 source=acesinger 的全部样本）
  B. sklearn `roc_curve` 口径 —— 官方 CtrSVDD2024_Baseline/calculate_eer.py 同款，
     cohort 同 A
  C. 同 A 的 cohort，但用线性插值求 far=frr 交点
  D. 只排除 A14 的 spoof，保留 acesinger 的 2,845 条 bonafide

输出：restored_data/current_pdf/table1_baseline_reproduction.json

用法
----
python scripts/reproduce_table1_rows.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANALYSIS = ROOT.parent / "CtrSVDD2024_Baseline" / "analysis"

# 论文 Table 1（当前 PDF 第 3 页）：Pooled 与 A09–A13
PAPER = {
    "B01": {"label": "B01 (LFCC) [24]", "pooled": 11.37,
            "A09": 5.35, "A10": 2.92, "A11": 5.84, "A12": 29.47, "A13": 3.65},
    "B02": {"label": "B02 (AASIST) [24]", "pooled": 10.39,
            "A09": 6.72, "A10": 0.96, "A11": 3.59, "A12": 26.83, "A13": 0.95},
}
ATTACKS = ["A09", "A10", "A11", "A12", "A13"]


# ---------------------------------------------------------------------- #
def load_groundtruth() -> dict:
    rows = [l.split() for l in (ANALYSIS / "test_groundtruth.txt").read_text().splitlines() if l.strip()]
    # source singer utt_id - attack label
    return {r[2]: {"source": r[0], "singer": r[1], "attack": r[4],
                   "label": 0 if r[5].lower() == "bonafide" else 1} for r in rows}


def load_scores(name: str) -> dict:
    out = {}
    with (ANALYSIS / f"baselines_csv/{name}.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["filename"]] = float(r["score"])
    return out


# ---------------------------------------------------------------------- #
# 四种 EER 口径
# ---------------------------------------------------------------------- #
def eer_nearest(scores, labels):
    """eer.py 口径：unique threshold + 最近的 far/frr 交叉点，无插值。"""
    from opera.metrics import compute_eer
    return compute_eer(scores, labels)[0] * 100


def eer_sklearn(scores, labels):
    """官方 calculate_eer.py 口径：sklearn roc_curve + argmin|fpr-frr|。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    except ImportError:
        fpr, tpr = _roc_curve_manual(scores, labels)
    frr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - frr)))
    return float((fpr[i] + frr[i]) / 2) * 100


def _roc_curve_manual(scores, labels):
    """sklearn.metrics.roc_curve(drop_intermediate=False) 的等价实现。"""
    pos = labels == 1
    desc = np.argsort(-scores, kind="mergesort")
    s, y = scores[desc], pos[desc]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[idx]
    fps = np.cumsum(~y)[idx]
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    if fps[-1] == 0 or tps[-1] == 0:
        raise ValueError("ROC undefined")
    return fps / fps[-1], tps / tps[-1]


def eer_interp(scores, labels):
    """线性插值求 far = frr 的交点。"""
    from opera.metrics.eer import compute_det_curve
    far, frr, _ = compute_det_curve(scores, labels)
    far, frr = np.asarray(far), np.asarray(frr)
    d = far - frr                      # 单调递减（far 升、frr 降）
    sign = np.sign(d)
    cross = np.where(np.diff(sign) != 0)[0]
    if cross.size == 0:
        i = int(np.argmin(np.abs(d)))
        return float((far[i] + frr[i]) / 2) * 100
    i = cross[0]
    x0, x1 = far[i], far[i + 1]
    y0, y1 = frr[i], frr[i + 1]
    # 在 (far, frr) 平面上求 far = frr 的点：沿线段插值
    denom = (x1 - x0) - (y1 - y0)
    t = (y0 - x0) / denom if denom != 0 else 0.0
    t = float(np.clip(t, 0.0, 1.0))
    return float(x0 + t * (x1 - x0)) * 100


# ---------------------------------------------------------------------- #
def cohort(truth, uids, scheme: str):
    """按口径筛选 cohort，返回 (scores_mask, attacks, sources, labels)。"""
    attacks = np.array([truth[u]["attack"] for u in uids])
    sources = np.array([truth[u]["source"] for u in uids])
    labels = np.array([truth[u]["label"] for u in uids])
    if scheme == "A":       # challenge notebook：排除 A14 与所有 acesinger
        mask = (attacks != "A14") & (sources != "acesinger")
    elif scheme == "D":     # 只排除 A14 的 spoof，保留 acesinger bonafide
        mask = (attacks != "A14")
    else:
        raise ValueError(scheme)
    return mask, attacks, sources, labels


def evaluate(score_arr, truth, uids, scheme: str, eer_fn) -> dict:
    mask, attacks, sources, labels = cohort(truth, uids, scheme)
    s = -np.asarray(score_arr, dtype=np.float64)[mask]      # 官方文件 higher=bonafide
    y = labels[mask]
    atk = attacks[mask]
    out = {"pooled": eer_fn(s, y), "n_bonafide": int((y == 0).sum()),
           "n_spoof": int((y == 1).sum())}
    for a in ATTACKS:
        m = (atk == a) | (y == 0)      # 该攻击的 spoof + 全部 bonafide
        out[a] = eer_fn(s[m], y[m])
    return out


# ---------------------------------------------------------------------- #
def main() -> None:
    truth = load_groundtruth()
    results = []

    for name in ["B01", "B02"]:
        scores = load_scores(name)
        uids = list(scores)
        ids = set(uids)
        if len(ids) != len(uids):
            raise ValueError(f"{name}: 分数文件存在重复 utt_id")
        if ids != set(truth):
            raise ValueError(f"{name}: 分数未覆盖 ground truth 中的每条样本")

        row = {"system": name, "paper_label": PAPER[name]["label"],
               "n_scored": len(uids), "paper": PAPER[name], "variants": {}}
        score_arr = np.array([scores[u] for u in uids], dtype=np.float64)
        for tag, scheme, fn, desc in [
            ("A_legacy_results_csv", "A", eer_nearest,
             "官方 results.csv 口径：额外剔除 source=acesinger 的全部样本"),
            ("B_sklearn_same_cohort", "A", eer_sklearn,
             "sklearn roc_curve 口径（官方 calculate_eer.py 同款），cohort 同 A"),
            ("C_interpolated", "A", eer_interp,
             "线性插值求 far=frr 交点，cohort 同 A"),
            ("D_paper_table1", "D", eer_sklearn,
             "★论文 Table 1 口径：只排除 A14 的 spoof，保留全部 bonafide"),
        ]:
            m = evaluate(score_arr, truth, uids, scheme, fn)
            row["variants"][tag] = {"desc": desc, **m}
            print(f"[{name}] {tag:22} pooled={m['pooled']:.4f}  "
                  f"A09={m['A09']:.4f} A10={m['A10']:.4f} A11={m['A11']:.4f} "
                  f"A12={m['A12']:.4f} A13={m['A13']:.4f}  "
                  f"(n_bon={m['n_bonafide']:,} n_spoof={m['n_spoof']:,})")
        print(f"[{name}] {'论文 Table 1':22} pooled={PAPER[name]['pooled']:.2f}  "
              f"A09={PAPER[name]['A09']:.2f} A10={PAPER[name]['A10']:.2f} "
              f"A11={PAPER[name]['A11']:.2f} A12={PAPER[name]['A12']:.2f} "
              f"A13={PAPER[name]['A13']:.2f}")
        print()
        results.append(row)

    out = ROOT.parent / "restored_data/current_pdf/table1_baseline_reproduction.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
