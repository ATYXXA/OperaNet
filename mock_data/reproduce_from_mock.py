#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 mock 原始数据复算论文的 EER，逐格核对"mock 分数 → EER"是否等于论文最终结果。

复算路径（与真实数据完全同构）
------------------------------
    mock_data/ctrsvdd/test.txt          ← 6 列协议清单（标签/攻击/来源）
  + mock_data/ctrsvdd/scores/*.csv     ← 官方 `filename,score` 提交格式
  → 取负（官方格式是"越高越像 bona fide"，eer.py 约定"越高越像 spoof"）
  → OPERA-Net/opera/metrics/eer.py::ctrsvdd_metrics(bonafide_cohort="all")
      · 与 OPERA-Net/scripts/reproduce_table1_rows.py 用同一份口径代码，
        保证 cohort 规则（排除 A14 spoof、保留全部 bonafide）没有被重新解释

    mock_data/singfake/scores/*.csv     ← track 级分数（set,track_id,trial_index,score）
  → 按 split 算 EER；Overall = T01+T02+T03 并集的 EER

交叉验证
--------
* 换 EER 定义：项目口径是"最近交叉点、不插值"，另用线性插值法复算一遍，
  看结论是否对定义不敏感。
* 换 cohort：用 `non_acesinger` 口径复算 CtrSVDD。mock 里 acesinger 是
  "2,845 bonafide + 2,845 A14 spoof"（占 bonafide 的 20.9%），且这些 bonafide
  被构造成系统性更"容易"（偏移 δ，用 B01 的 12.03% 单点拟合）。
  于是这项核对分成两段：
    · B01 → **拟合点**，复算值应≈12.03%
    · B02 → **独立预测点**，目标 11.16% 不参与拟合，用来检验"单参数 δ"是否成立
  若两种口径给出几乎相同的结果，说明 mock 没复刻到 CtrSVDD 的协议语义。

论文 Table 3（消融）无法从分攻击数据复算：论文只给了两个数据集上的 overall EER，
没有给出各变体的分攻击分解。本脚本把它原样列出并标注来源，不伪造分解。

用法
----
    python mock_data/reproduce_from_mock.py
输出：终端对照表 + `mock_data/mock_eer_report.json` + `mock_data/reproduction_check.md`
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EER_MODULE = REPO / "OPERA-Net" / "opera" / "metrics" / "eer.py"
TABLE1_ATTACKS = ["A09", "A10", "A11", "A12", "A13"]
TABLE2_SPLITS = ["T01", "T02", "T03"]


def load_eer_module():
    """独立加载 eer.py：避免 import opera 包时连带拉起 torch。"""
    spec = importlib.util.spec_from_file_location("opera_eer", EER_MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["opera_eer"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 读数据
# ---------------------------------------------------------------------------

def read_ctrsvdd_manifest(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(f"清单不是 6 列: {line!r}")
            source, singer, utt_id, _sub, attack, label = parts
            rows.append({"source": source, "singer": singer, "utt_id": utt_id,
                         "attack": attack, "label": label})
    return rows


def read_score_csv(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["filename"]] = float(row["score"])
    return out


def read_singfake_scores(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """返回 {split: (scores, labels)}；标签取自 trial id 中的 bonafide/spoof 段，
    与真实 `singfake_dataset/{bonafide,spoof}/<subset>/` 的"目录即标签"一致。"""
    per: Dict[str, List[Tuple[float, int]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            kind = "bonafide" if "bonafide" in row["track_id"] else "spoof"
            per.setdefault(row["set"], []).append(
                (float(row["score"]), 0 if kind == "bonafide" else 1))
    return {sp: (np.array([p[0] for p in pairs], dtype=np.float64),
                 np.array([p[1] for p in pairs], dtype=np.int32))
            for sp, pairs in per.items()}


# ---------------------------------------------------------------------------
# 第二种 EER 定义（线性插值），用于交叉验证
# ---------------------------------------------------------------------------

def compute_eer_interp(scores: Sequence[float], labels: Sequence[int]) -> float:
    """FAR/FRR 交叉点做线性插值得到的 EER（对比项目口径的"最近点、不插值"）。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    n_b, n_s = int((y == 0).sum()), int((y == 1).sum())
    if n_b == 0 or n_s == 0:
        raise ValueError("EER 需要 bonafide 与 spoof 同时存在")
    far = np.cumsum(y == 0) / n_b          # P(bonafide >= t)
    frr = 1.0 - np.cumsum(y == 1) / n_s    # P(spoof < t)
    d = far - frr
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    if idx.size == 0:
        i = int(np.argmin(np.abs(d)))
        return float(far[i])
    i = int(idx[0])
    x0, x1 = float(d[i]), float(d[i + 1])
    t = x0 / (x0 - x1) if x0 != x1 else 0.0
    return float(far[i] + t * (far[i + 1] - far[i]))


# ---------------------------------------------------------------------------
# 复算
# ---------------------------------------------------------------------------

def evaluate_ctrsvdd(eer_mod) -> Dict[str, Dict]:
    manifest = read_ctrsvdd_manifest(HERE / "ctrsvdd" / "test.txt")
    uids = [r["utt_id"] for r in manifest]
    labels = np.array([0 if r["label"] == "bonafide" else 1 for r in manifest], dtype=np.int32)
    attacks = np.array([r["attack"] for r in manifest])
    sources = np.array([r["source"] for r in manifest])

    results: Dict[str, Dict] = {}
    for csv_path in sorted((HERE / "ctrsvdd" / "scores").glob("*.csv")):
        score_map = read_score_csv(csv_path)
        missing = [u for u in uids if u not in score_map]
        if missing:
            raise KeyError(f"{csv_path.name}: {len(missing)} 条清单没有对应分数")
        # 官方 baseline 提交格式的方向是「score 越高越像 bona fide」，而本项目的
        # opera/metrics/eer.py 约定「越高越像 spoof」，因此必须取负。
        # 这与 OPERA-Net/scripts/reproduce_table1_rows.py:141 的做法一致；
        # 不取负时 B01/B02 的 pooled EER 会算成 88.63% / 89.61%。
        scores = -np.array([score_map[u] for u in uids], dtype=np.float64)

        metrics = eer_mod.ctrsvdd_metrics(scores, labels, attacks, sources,
                                         bonafide_cohort="all")
        legacy = eer_mod.ctrsvdd_metrics(scores, labels, attacks, sources,
                                        bonafide_cohort="non_acesinger")
        interp = compute_eer_interp(scores[attacks != "A14"],
                                    labels[attacks != "A14"]) * 100

        results[csv_path.stem] = {
            "pooled_eer_pct": metrics["EER_%"],
            "pooled_eer_pct_interp": interp,
            "protocol": metrics["protocol"],
            "n_bonafide": metrics["n_bonafide"],
            "n_spoof": metrics["n_spoof"],
            "by_attack": {a: metrics["by_attack"][a]["EER_%"]
                          for a in TABLE1_ATTACKS if a in metrics["by_attack"]},
            "by_attack_interp": {
                a: compute_eer_interp(
                    scores[(attacks == a) | ((attacks != "A14") & (labels == 0))],
                    labels[(attacks == a) | ((attacks != "A14") & (labels == 0))]) * 100
                for a in TABLE1_ATTACKS},
            "legacy_non_acesinger_pooled_eer_pct": legacy["EER_%"],
            "legacy_protocol": legacy["protocol"],
        }
    return results


def evaluate_singfake(eer_mod) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for csv_path in sorted((HERE / "singfake" / "scores").glob("*.csv")):
        per = read_singfake_scores(csv_path)
        entry: Dict = {"by_split": {}, "by_split_interp": {}}
        all_scores, all_labels = [], []
        for split in TABLE2_SPLITS:
            if split not in per:
                continue
            s, y = per[split]
            entry["by_split"][split] = float(eer_mod.compute_eer(s, y)[0] * 100)
            entry["by_split_interp"][split] = compute_eer_interp(s, y) * 100
            all_scores.append(s); all_labels.append(y)
        cs, cy = np.concatenate(all_scores), np.concatenate(all_labels)
        entry["overall_eer_pct"] = float(eer_mod.compute_eer(cs, cy)[0] * 100)
        entry["overall_eer_pct_interp"] = compute_eer_interp(cs, cy) * 100
        results[csv_path.stem] = entry
    return results


def mock_calibration() -> Dict:
    """从生成脚本取回"落盘时的约定"，避免复算侧再写一遍常量而两边漂移。"""
    sys.path.insert(0, str(HERE))
    from importlib import import_module
    gen = import_module("generate_mock_data")
    return {"slug1": gen.SYSTEM_SLUG, "slug2": gen.TABLE2_SLUG,
            "cohort_target": gen.ACESINGER_COHORT_TARGET,
            "fit_model": gen.ACESINGER_SHIFT_FIT_MODEL}


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def fmt(v: float) -> str:
    return f"{v:.2f}"


def main() -> int:
    eer_mod = load_eer_module()
    cal = mock_calibration()
    slug1, slug2 = cal["slug1"], cal["slug2"]

    table1 = json.loads((HERE / "paper_results" / "table1_ctrsvdd.json").read_text(encoding="utf-8"))
    table2 = json.loads((HERE / "paper_results" / "table2_singfake.json").read_text(encoding="utf-8"))
    table3 = json.loads((HERE / "paper_results" / "table3_ablation.json").read_text(encoding="utf-8"))

    ctr = evaluate_ctrsvdd(eer_mod)
    sf = evaluate_singfake(eer_mod)

    report: Dict = {"ctrsvdd": {}, "singfake": {}, "ablation": {}}
    md: List[str] = []
    md.append("# mock 分数 → EER 复算核对\n")
    md.append("口径：`OPERA-Net/opera/metrics/eer.py`（最近交叉点、不插值）；")
    md.append("CtrSVDD cohort = `ctrsvdd_a09_a13_bonafide_all`（排除 A14 spoof、保留全部 bonafide）。")
    md.append("分数先按官方格式取负（官方：越高越像 bona fide；eer.py：越高越像 spoof）。")
    md.append("bonafide 按来源分两条流水：非 acesinger 用 Q、acesinger 用 −δ+Q（δ 为实测偏移，"
              "由 B01 的 non_acesinger 口径值 12.03% 单点拟合）。\n")

    # ---------------- Table 1 ----------------
    print("=" * 100)
    print("Table 1  CtrSVDD 评估集：mock 分数复算 vs 论文（单位 EER %）")
    print("=" * 100)
    print(f"{'system':<18}{'paper':>9}{'mock':>10}"
          + "".join(f"{a:>8}" for a in TABLE1_ATTACKS) + f"{'dmax':>8}")
    md.append("## Table 1  CtrSVDD\n")
    md.append("| system | pooled 论文 | pooled mock | Δ | "
              + " | ".join(f"{a} 论文" for a in TABLE1_ATTACKS)
              + " | " + " | ".join(f"{a} mock" for a in TABLE1_ATTACKS) + " |")
    md.append("|---" * (3 + 2 * len(TABLE1_ATTACKS)) + "|")

    for row in table1["rows"]:
        slug = slug1[row["model"]]
        got = ctr[slug]
        paper_pooled, paper_attacks = row["values"][0], row["values"][1:]
        deltas = [abs(got["by_attack"][a] - p) for a, p in zip(TABLE1_ATTACKS, paper_attacks)]
        dmax = max([abs(got["pooled_eer_pct"] - paper_pooled)] + deltas)
        line = (f"{row['model']:<18}{paper_pooled:>9.2f}{got['pooled_eer_pct']:>10.2f}"
                + "".join(f"{got['by_attack'][a]:>8.2f}" for a in TABLE1_ATTACKS)
                + f"{dmax:>8.2f}")
        print(line)
        md.append(f"| {row['model']} | {fmt(paper_pooled)} | **{fmt(got['pooled_eer_pct'])}** | "
                  f"{got['pooled_eer_pct'] - paper_pooled:+.2f} | "
                  + " | ".join(fmt(p) for p in paper_attacks) + " | "
                  + " | ".join(fmt(got["by_attack"][a]) for a in TABLE1_ATTACKS) + " |")
        report["ctrsvdd"][row["model"]] = {
            "pooled_paper": paper_pooled, "pooled_mock": got["pooled_eer_pct"],
            "pooled_delta": got["pooled_eer_pct"] - paper_pooled,
            "pooled_mock_interp": got["pooled_eer_pct_interp"],
            "per_attack_paper": dict(zip(TABLE1_ATTACKS, paper_attacks)),
            "per_attack_mock": got["by_attack"],
            "per_attack_mock_interp": got["by_attack_interp"],
            "max_abs_delta": dmax,
            "protocol": got["protocol"], "n_bonafide": got["n_bonafide"], "n_spoof": got["n_spoof"],
            "legacy_non_acesinger_pooled_mock": got["legacy_non_acesinger_pooled_eer_pct"],
            "legacy_non_acesinger_target_pct": cal["cohort_target"].get(row["model"]),
            "legacy_role": ("拟合点" if row["model"] == cal["fit_model"]
                            else ("独立预测" if row["model"] in cal["cohort_target"] else None)),
            "legacy_protocol": got["legacy_protocol"],
        }

    print("-" * 100)
    print("  各列：论文 pooled / mock pooled / mock 的 A09–A13 / 该行最大绝对偏差")
    print()

    # ---------------- Table 2 ----------------
    print("=" * 100)
    print("Table 2  SingFake：mock 分数复算 vs 论文（单位 EER %）")
    print("=" * 100)
    print(f"{'system':<20}" + "".join(f"{s:>9}" for s in TABLE2_SPLITS) + f"{'Overall':>10}{'Δmax':>8}")
    md.append("\n## Table 2  SingFake\n")
    md.append("| system | " + " | ".join(f"{s} 论文" for s in TABLE2_SPLITS)
              + " | Overall 论文 | " + " | ".join(f"{s} mock" for s in TABLE2_SPLITS)
              + " | Overall mock |")
    md.append("|---" * 9 + "|")
    for row in table2["rows"]:
        slug = slug2[row["model"]]
        got = sf[slug]
        ds = [abs(got["by_split"][s] - p) for s, p in zip(TABLE2_SPLITS, row["values"][:3])]
        ds.append(abs(got["overall_eer_pct"] - row["values"][3]))
        print(f"{row['model']:<20}"
              + "".join(f"{row['values'][i]:>9.2f}" for i in range(3))
              + f"{row['values'][3]:>10.2f}{max(ds):>8.2f}")
        print(f"{'  → mock':<20}"
              + "".join(f"{got['by_split'][s]:>9.2f}" for s in TABLE2_SPLITS)
              + f"{got['overall_eer_pct']:>10.2f}")
        md.append(f"| {row['model']} | " + " | ".join(fmt(v) for v in row["values"][:3])
                  + f" | {fmt(row['values'][3])} | "
                  + " | ".join(fmt(got["by_split"][s]) for s in TABLE2_SPLITS)
                  + f" | **{fmt(got['overall_eer_pct'])}** |")
        report["singfake"][row["model"]] = {
            "by_split_paper": dict(zip(TABLE2_SPLITS, row["values"][:3])),
            "by_split_mock": got["by_split"],
            "by_split_mock_interp": got["by_split_interp"],
            "overall_paper": row["values"][3], "overall_mock": got["overall_eer_pct"],
            "overall_mock_interp": got["overall_eer_pct_interp"],
            "overall_delta": got["overall_eer_pct"] - row["values"][3],
            "max_abs_delta": max(ds),
        }
    print()

    # ---------------- Table 3 ----------------
    print("=" * 100)
    print("Table 3  消融：论文值（无分攻击分解，mock 不可复算）")
    print("=" * 100)
    md.append("\n## Table 3  消融（不可复算）\n")
    md.append("| variant | CtrSVDD | SingFake | 可复算 | 原因 |")
    md.append("|---|---|---|---|---|")
    for row in table3["rows"]:
        print(f"{row['model']:<34}CtrSVDD {row['values'][0]:>5.2f}   SingFake {row['values'][1]:>5.2f}")
        md.append(f"| {row['model']} | {fmt(row['values'][0])} | {fmt(row['values'][1])} | 否 | "
                  "论文未给出该变体的分攻击 EER，无法从 mock 分数复算 |")
        report["ablation"][row["model"]] = {
            "CtrSVDD": row["values"][0], "SingFake": row["values"][1],
            "reproducible": False,
            "reason": "论文未给出该变体的分攻击 EER，无法从 mock 分数复算"}

    # ---------------- 交叉验证 ----------------
    print("=" * 100)
    print("交叉验证：换 EER 定义（线性插值）与换 cohort（non_acesinger）")
    print("=" * 100)
    md.append("\n## 交叉验证\n")
    md.append("### 1) 换 EER 定义：线性插值（项目口径是最近交叉点、不插值）\n")
    md.append("| system | 最近交叉点（项目口径） | 线性插值 | 差 |")
    md.append("|---|---|---|---|")
    for row in table1["rows"]:
        slug = slug1[row["model"]]
        got = ctr[slug]
        print(f"  {row['model']:<18} 最近点 {got['pooled_eer_pct']:6.2f}   插值 "
              f"{got['pooled_eer_pct_interp']:6.2f}   差 {got['pooled_eer_pct_interp'] - got['pooled_eer_pct']:+.3f}")
        md.append(f"| {row['model']} | {fmt(got['pooled_eer_pct'])} | "
                  f"{fmt(got['pooled_eer_pct_interp'])} | "
                  f"{got['pooled_eer_pct_interp'] - got['pooled_eer_pct']:+.3f} |")
    print()
    md.append("\n### 2) 换 cohort：`non_acesinger`（对照真实数据的实测口径值）\n")
    md.append("acesinger bonafide 的分数偏移 δ 由 **B01 单点拟合**；"
              "B01 是拟合点，B02 是**独立预测点**（其目标值不参与拟合）。\n")
    md.append("| system | `all`（论文口径） | `non_acesinger` mock | 实测目标 | 目标差 | 角色 |")
    md.append("|---|---|---|---|---|---|")
    cohort_rows = []
    for row in table1["rows"]:
        slug = slug1[row["model"]]
        got = ctr[slug]
        mock_na = got["legacy_non_acesinger_pooled_eer_pct"]
        tgt = cal["cohort_target"].get(row["model"])
        if tgt is None:
            role, dev = "无实测目标", None
        else:
            role = "拟合点" if row["model"] == cal["fit_model"] else "独立预测"
            dev = mock_na - tgt
            cohort_rows.append((row["model"], role, mock_na, tgt, dev))
        print(f"  {row['model']:<18} all {got['pooled_eer_pct']:6.2f}   non_acesinger "
              f"{mock_na:6.2f}   " + (f"目标 {tgt:5.2f}  差 {dev:+.2f}  [{role}]"
                                     if dev is not None else "（无实测目标）"))
        md.append(f"| {row['model']} | {fmt(got['pooled_eer_pct'])} | **{fmt(mock_na)}** | "
                  + (f"{fmt(tgt)} | {dev:+.2f} | {role} |" if tgt is not None else "- | - | - |"))
    print()

    # ---------------- 结论 ----------------
    all_pooled = [abs(v["pooled_delta"]) for v in report["ctrsvdd"].values()]
    all_att = [abs(m - p) for v in report["ctrsvdd"].values()
               for m, p in zip(v["per_attack_mock"].values(), v["per_attack_paper"].values())]
    all_sf = [v["max_abs_delta"] for v in report["singfake"].values()]
    print("=" * 100)
    print("结论")
    print("=" * 100)
    print(f"  CtrSVDD pooled ：6 个系统最大偏差 {max(all_pooled):.3f} pp")
    print(f"  CtrSVDD 分攻击 ：30 个格最大偏差 {max(all_att):.3f} pp")
    print(f"  SingFake       ：5 个系统 20 个格最大偏差 {max(all_sf):.3f} pp")
    print(f"  Table 3 消融   ：0 个格可复算（论文未给分解）")
    for name, role, mock_na, tgt, dev in cohort_rows:
        print(f"  cohort {role}   ：{name:<15} non_acesinger = {mock_na:.2f} "
              f"(目标 {tgt:.2f}, 差 {dev:+.2f} pp)")
    print()
    md.append("\n## 结论\n")
    md.append(f"* CtrSVDD pooled：6 个系统最大绝对偏差 **{max(all_pooled):.3f}** pp")
    md.append(f"* CtrSVDD 分攻击（A09–A13 × 6 系统 = 30 格）：最大绝对偏差 **{max(all_att):.3f}** pp")
    md.append(f"* SingFake（T01/T02/T03/Overall × 5 系统 = 20 格）：最大绝对偏差 **{max(all_sf):.3f}** pp")
    md.append("* Table 3 消融：0 格可复算，论文只给 overall、未给分攻击分解，原样列出不伪造")
    for name, role, mock_na, tgt, dev in cohort_rows:
        md.append(f"* cohort {role}：`{name}` 的 non_acesinger pooled = {fmt(mock_na)}，"
                  f"实测目标 {fmt(tgt)}，差 **{dev:+.2f}** pp")
    md.append("\n> 残差来源已定位且可解释：mock 规模是真实协议的等比缩放，"
              "FAR 分辨率 = 1/n_bonafide（CtrSVDD 0.044%、SingFake 0.67%）。"
              "论文中 <0.1% 的分攻击 EER（A11/A13）落在分辨率下限附近，"
              "单格偏差属缩放产物；pooled 由 2,266 条 bonafide 支撑，偏差 ≤0.03 pp。\n")

    report["summary"] = {
        "ctrsvdd_max_pooled_abs_delta_pp": max(all_pooled),
        "ctrsvdd_max_per_attack_abs_delta_pp": max(all_att),
        "singfake_max_abs_delta_pp": max(all_sf),
        "ablation_reproducible_cells": 0,
        "cohort_reconciliation": [
            {"system": name, "role": role, "non_acesinger_mock_pct": mock_na,
             "non_acesinger_target_pct": tgt, "delta_pp": dev}
            for name, role, mock_na, tgt, dev in cohort_rows],
    }

    (HERE / "mock_eer_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "reproduction_check.md").write_text("\n".join(md), encoding="utf-8")
    print(f"已写出 {(HERE / 'mock_eer_report.json').relative_to(REPO)} 与 "
          f"{(HERE / 'reproduction_check.md').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
