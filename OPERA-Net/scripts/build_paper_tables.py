"""Current PDF Tables 1-3; reported values never populate measured results."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/paper_reference.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def load_metrics(exp_root, config, folder):
    if config is None:
        return None
    p = exp_root / Path(config).stem / folder / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    provenance = m.get("provenance", {})
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    scores = p.parent / "scores.jsonl"
    if (provenance.get("status") != "measured" or not scores.exists()
            or provenance.get("scores_sha256") != digest(scores)
            or provenance.get("pdf_sha256") != digest(ROOT.parent / ref["source_file"])):
        return None
    expected = "ctrsvdd_a09_a13_bonafide_all" if folder == "ctrsvdd_eval" else None
    if expected and provenance.get("scope") != expected:
        return None
    if folder == "singfake_eval" and set(provenance.get("subsets", [])) != {"T01", "T02", "T03"}:
        return None
    return m


def measured_values(table, config, exp_root):
    ctr = load_metrics(exp_root, config, "ctrsvdd_eval")
    sing = load_metrics(exp_root, config, "singfake_eval")
    if table == 1:
        if not ctr:
            return [None]*6
        return [ctr.get("EER_%")] + [ctr.get("by_attack", {}).get(a, {}).get("EER_%") for a in ["A09","A10","A11","A12","A13"]]
    if table == 2:
        if not sing:
            return [None]*4
        return [sing.get("by_subset", {}).get(s, {}).get("EER_%") for s in ["T01","T02","T03"]] + [sing.get("EER_%")]
    return [ctr.get("EER_%") if ctr else None, sing.get("EER_%") if sing else None]


def build(table, exp_root, out):
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    tab = ref["tables"][str(table)]
    lines = [f"# Table {table} 对照（EER %）", "",
             f"来源：当前 PDF 第 {tab['page']} 页。单元格为论文报告值 / 本地实测值；缺少可验证分数则留空。",
             "SingFake 的评估粒度和 Overall 合并规则仍属复现假设；Table 3 的 CtrSVDD 暂沿用 Table 1 协议。", "",
             "| Model | " + " | ".join(tab["columns"]) + " |",
             "|" + "---|"*(len(tab["columns"])+1)]
    rows = []
    for row in tab["rows"]:
        measured = measured_values(table, row["config"], exp_root)
        cells = []
        for column, expected, actual in zip(tab["columns"], row["values"], measured):
            cells.append(f"{expected:.2f} / " + ("未测" if actual is None else f"{actual:.2f}"))
            rows.append(dict(model=row["model"], column=column, paper_reported=expected,
                             measured=actual, delta=None if actual is None else actual-expected,
                             unit="EER percent", paper_page=tab["page"],
                             status="no_verified_local_run" if actual is None else "measured_under_declared_reproduction_protocol"))
        lines.append("| " + row["model"] + " | " + " | ".join(cells) + " |")
    out.mkdir(parents=True, exist_ok=True)
    (out/f"table{table}_reconciliation.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/f"table{table}_reconciliation.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--table", type=int, choices=[1,2,3], default=1)
    ap.add_argument("--exp_root", type=Path, default=ROOT/"exp")
    ap.add_argument("--out_dir", type=Path, default=ROOT/"results/current_pdf")
    args = ap.parse_args()
    for t in ([1,2,3] if args.all else [args.table]):
        build(t,args.exp_root,args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
