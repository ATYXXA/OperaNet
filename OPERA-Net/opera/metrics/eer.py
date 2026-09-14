"""Current PDF EER: 0=bonafide, 1=spoof; higher score=spoof.

Tied scores move together; use the nearest threshold crossing, without
interpolation. Legacy ASV/min-tDCF code is not used by the current protocol.
"""
from __future__ import annotations
import numpy as np


def _validate(scores, labels):
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    if scores.size != labels.size or not scores.size:
        raise ValueError("scores and labels must be nonempty and equally sized")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("finite scores and binary labels are required")
    if np.unique(labels).size != 2:
        raise ValueError("EER requires both bonafide and spoof samples")
    return scores, labels.astype(np.int32)


def compute_det_curve(scores, labels):
    """(bonafide rejection, spoof acceptance, thresholds); spoof iff score>=threshold."""
    scores, labels = _validate(scores, labels)
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    tails = np.r_[s[1:] != s[:-1], True]
    bon_reject = np.cumsum(y == 0)[tails] / np.sum(y == 0)
    spoof_accept = 1 - np.cumsum(y == 1)[tails] / np.sum(y == 1)
    return np.r_[0., bon_reject], np.r_[1., spoof_accept], np.r_[np.inf, s[tails]]


def compute_eer(scores, labels):
    far, frr, thresholds = compute_det_curve(scores, labels)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thresholds[i])


def compute_accuracy(scores, labels, threshold=0.):
    s, y = _validate(scores, labels)
    return float(((s >= threshold) == y).mean())


def evaluate_all(scores, labels, *legacy_asv, **legacy_kwargs):
    if any(x is not None for x in legacy_asv) or legacy_kwargs:
        raise ValueError("Current PDF reports EER only; ASV/min-tDCF is not part of this evaluation")
    s, y = _validate(scores, labels)
    eer, threshold = compute_eer(s, y)
    return {"EER_%": eer * 100, "threshold": threshold if np.isfinite(threshold) else None,
            "n_bonafide": int(np.sum(y == 0)), "n_spoof": int(np.sum(y == 1)),
            "score_direction": "higher_is_spoof", "eer_method": "nearest_crossing_unique_thresholds"}


def track_level_aggregate(seg_scores, seg_track_ids, seg_labels):
    if not (len(seg_scores) == len(seg_track_ids) == len(seg_labels)):
        raise ValueError("segment scores, IDs and labels must align")
    groups = {}
    for score, uid, label in zip(seg_scores, seg_track_ids, seg_labels):
        if uid in groups and groups[uid][0] != label:
            raise ValueError(f"Conflicting labels for track {uid}")
        groups.setdefault(uid, (label, []))[1].append(float(score))
    return (np.array([np.mean(v[1]) for v in groups.values()]),
            np.array([v[0] for v in groups.values()], dtype=np.int32))


def ctrsvdd_metrics(scores, labels, attacks, sources, bonafide_cohort="all"):
    """CtrSVDD 评估集的 cohort 选择与 pooled EER。

    bonafide_cohort
    ---------------
    ``"all"``（默认，论文 Table 1 口径）
        只排除 attack A14 的 spoof，**保留全部 bonafide**（含 source=acesinger
        的 2,845 条）。实测用该口径重算官方 baseline 分数：
        B01 pooled = 11.37%、B02 pooled = 10.39%，与论文 Table 1 完全一致。

    ``"non_acesinger"``（CtrSVDD2024_Baseline/analysis/results.csv 口径）
        额外排除 source=acesinger 的全部样本。同一份分数在该口径下为
        B01 = 12.03%、B02 = 11.16%，比论文高约 0.7–0.8 个百分点。
        仅在需要与官方 results.csv 对账时使用。

    差异来源：ACESinger 既有 spoof（A14）也有 2,845 条 bonafide。
    把后者一并剔除会改变 FAR 的分母，从而系统性抬高 pooled EER。
    """
    scores, labels = _validate(scores, labels)
    attacks, sources = np.asarray(attacks), np.asarray(sources)
    if len(attacks) != len(scores) or len(sources) != len(scores):
        raise ValueError("attack and source metadata must align with scores")

    keep = (attacks != "A14")
    if bonafide_cohort == "non_acesinger":
        keep = keep & (sources != "acesinger")
    elif bonafide_cohort != "all":
        raise ValueError(f"unknown bonafide_cohort {bonafide_cohort!r}; "
                         "expected 'all' or 'non_acesinger'")

    bona = (labels == 0) & keep
    primary = keep
    if not np.isin(attacks[primary & (labels == 1)], [f"A{i:02}" for i in range(9, 14)]).all():
        raise ValueError("CtrSVDD ranking protocol requires the official test split")

    out = evaluate_all(scores[primary], labels[primary])
    out["protocol"] = f"ctrsvdd_a09_a13_bonafide_{bonafide_cohort}"
    out["all_attacks_diagnostic"] = evaluate_all(scores, labels)
    out["by_attack"] = {}
    for attack in [f"A{i:02}" for i in range(9, 15)]:
        if not np.any(attacks == attack):
            continue
        mask = (attacks == attack) | bona
        if np.unique(labels[mask]).size != 2:
            continue
        out["by_attack"][attack] = evaluate_all(scores[mask], labels[mask])
    return out
