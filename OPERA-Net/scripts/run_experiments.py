"""Run only implemented OPERA-Net variants for current PDF Tables 1-3."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import yaml
from build_paper_tables import build

ROOT = Path(__file__).resolve().parents[1]


def build_plan(table, train_only=False):
    names = ["opera_net"] if table in (1, 2) else ["ablation_wavlm_only", "ablation_cqt_mag", "ablation_pc_cqt", "opera_net"]
    plan = []
    for name in names:
        cfg = yaml.safe_load((ROOT/f"configs/{name}.yaml").read_text(encoding="utf-8"))
        common = ["--config", f"configs/{name}.yaml"]
        plan.append([sys.executable, "train.py", *common, "--out_dir", f"exp/{name}"])
        if train_only:
            continue
        ev = cfg["data"]["test"]
        plan.append([sys.executable, "evaluate.py", *common, "--ckpt", f"exp/{name}/best.pth",
                     "--eval_kind", "ctrsvdd", "--eval_root", ev["root"], "--eval_splits", *ev["splits"],
                     "--by_attack", "--out", f"exp/{name}/ctrsvdd_eval"])
        ce = cfg["data"]["cross_eval"]
        plan.append([sys.executable, "evaluate.py", *common, "--ckpt", f"exp/{name}/best.pth",
                     "--eval_kind", "singfake", "--eval_root", ce["root"], "--eval_subsets", *ce["subsets"],
                     *(["--track_level"] if ce.get("track_level") else []), "--out", f"exp/{name}/singfake_eval"])
    return plan


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--table",type=int,choices=[1,2,3],default=3)
    ap.add_argument("--plan",action="store_true")
    ap.add_argument("--train_only",action="store_true")
    ap.add_argument("--summarize",action="store_true")
    args=ap.parse_args()
    if args.summarize:
        build(args.table,ROOT/"exp",ROOT/"results/current_pdf")
        return
    plan=build_plan(args.table,args.train_only)
    if not args.plan and not args.train_only:
        # Fail before expensive training if any requested evaluation subset is absent.
        for cmd in plan:
            if "--eval_subsets" in cmd:
                root=Path(cmd[cmd.index("--eval_root")+1])
                for subset in ["T01","T02","T03"]:
                    for label in ["bonafide","spoof"]:
                        if not (root/label/subset).is_dir():
                            raise FileNotFoundError(f"Required evaluation input missing: {root/label/subset}")
    print("Only OPERA-Net variants are implemented here. External challenge/SingGraph rows remain references.")
    for cmd in plan:
        print(subprocess.list2cmdline(cmd))
        if not args.plan:
            subprocess.run(cmd,cwd=ROOT,check=True)


if __name__ == "__main__":
    main()
