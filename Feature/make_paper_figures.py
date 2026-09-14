#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键重新生成论文中的三张图。

图 ↔ 脚本 ↔ 数据 的对应关系
----------------------------
| 论文图   | 生成脚本                                  | 数据来源                          |
|----------|-------------------------------------------|-----------------------------------|
| Figure 1 | draw_semantic_guided_gating_flowchart.py  | 无（架构示意图，常量）            |
| Figure 2 | create_sgg_effect_visualization.py        | mock_data/audio/in_the_wild_sample|
| Figure 3 | 雷达图.py                                  | 论文 Table 1 的 A09–A13 EER（常量）|

其中 Figure 3 的数值与 `mock_data/paper_results/attackwise_radar.csv` 完全一致，
Figure 2 需要一段长于 141 秒的 "in-the-wild" 样本（脚本固定截取 [136, 141] s），
提交包里用 `mock_data/generate_mock_data.py` 合成的 145 秒样本替代。

为什么用 override 而不是改脚本
------------------------------
这三个脚本是从项目里原样保留的论文出图代码，**不做任何修改**，
以保持"论文图 = 该脚本产物"的可追溯性。本运行器通过 importlib 加载它们，
再把模块级路径常量（OUT_DIR / OUTPUT_DIR）与音频选择函数指向 mock 数据，
因此既不改动原脚本，也不需要真实的歌曲 mp3。

用法
----
    python Feature/make_paper_figures.py            # 生成三张图
    python Feature/make_paper_figures.py --only 3   # 只生成 Figure 3
输出目录：`Feature/figures/`
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Sequence

REPO = Path(__file__).resolve().parents[1]
FEATURE = Path(__file__).resolve().parent
MOCK = REPO / "mock_data"
FIGDIR = FEATURE / "figures"

# 无头环境下只保存文件，不需要交互式绘窗后端；必须在 import pyplot 之前设置
import matplotlib  # noqa: E402
matplotlib.use("Agg")


def _load(alias: str, filename: str):
    """按文件名加载 Feature/ 下的出图脚本（不改动脚本本身）。"""
    path = FEATURE / filename
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _mock_sample() -> Path:
    hits = sorted((MOCK / "audio").glob("in_the_wild_sample.*"))
    if not hits:
        raise FileNotFoundError(
            f"缺少 mock 音频：请先运行  python mock_data/generate_mock_data.py\n"
            f"（期望生成 {(MOCK / 'audio').relative_to(REPO)}/in_the_wild_sample.flac）")
    return hits[0]


def figure1(out_dir: Path) -> None:
    """Figure 1: OPERA-Net 整体架构（PC-CQT 流 + 语义流 + 语义引导门控）。"""
    mod = _load("f1_flowchart", "draw_semantic_guided_gating_flowchart.py")
    mod.OUTPUT_DIR = out_dir
    mod.draw_flowchart()


def figure2(out_dir: Path) -> None:
    """Figure 2: in-the-wild 样本上的语义引导门控可视化（4 联图）。"""
    mod = _load("f2_sgg_effect", "create_sgg_effect_visualization.py")
    mod.OUT_DIR = out_dir
    sample = _mock_sample()
    mod.choose_audio = (lambda: sample)          # 绕过原脚本对根目录 *.mp3 的搜索
    mod.main()


def figure3(out_dir: Path) -> None:
    """Figure 3: CtrSVDD 评估集 A09–A13 的分攻击 EER 雷达图。"""
    mod = _load("f3_radar", "雷达图.py")
    mod.OUT_DIR = out_dir
    mod.main()


FIGURES: dict[str, tuple[str, Callable[[Path], None], tuple[str, ...]]] = {
    "1": ("Figure 1  整体架构", figure1, ("semantic_guided_gating_flowchart.png",)),
    "2": ("Figure 2  语义引导门控可视化", figure2, ("semantic_guided_gating_effect_5s.png",)),
    "3": ("Figure 3  分攻击 EER 雷达图", figure3, ("ctrsvdd_radar_eer.png",)),
}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="重新生成论文的三张图")
    ap.add_argument("--only", default="1,2,3", help="只跑指定图，如 --only 2,3")
    ap.add_argument("--out", default=str(FIGDIR), help="输出目录（默认 Feature/figures）")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = [k.strip() for k in args.only.split(",") if k.strip()]
    unknown = [k for k in wanted if k not in FIGURES]
    if unknown:
        raise SystemExit(f"未知的图编号 {unknown}；可选 {sorted(FIGURES)}")

    produced: list[Path] = []
    for key in wanted:
        label, fn, expected = FIGURES[key]
        print(f"\n=== {label} ===")
        fn(out_dir)
        for name in expected:
            path = out_dir / name
            status = "OK " if path.exists() else "缺失"
            size = f"{path.stat().st_size / 1024:.0f} KB" if path.exists() else "-"
            print(f"  [{status}] {path.relative_to(REPO)}  ({size})")
            produced.append(path)

    print(f"\n共生成 {len(produced)} 个文件 -> {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
