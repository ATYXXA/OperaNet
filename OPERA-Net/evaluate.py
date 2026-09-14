"""Evaluate the current PDF protocol and save auditable sample-level scores."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from opera.data import build_dataset, get_collate
from opera.metrics import evaluate_all, ctrsvdd_metrics, track_level_aggregate
from opera.models import build_model
from opera.trainer import run_inference
from opera.utils import load_config, load_checkpoint


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_kind", choices=["ctrsvdd", "singfake"], default="ctrsvdd")
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--eval_splits", nargs="+", default=["test"])
    ap.add_argument("--eval_subsets", nargs="+", default=["T01", "T02", "T03"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device")
    ap.add_argument("--track_level", action="store_true")
    ap.add_argument("--by_attack", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.get("device", "cpu"))
    if args.eval_kind == "ctrsvdd":
        if args.eval_splits != ["test"]:
            raise ValueError("Paper ranking evaluation requires test; training handles dev EER")
        ds = build_dataset(kind="ctrsvdd", root=args.eval_root, splits=args.eval_splits,
                           check_exists=True, mode="center")
    else:
        ds = build_dataset(kind="singfake", root=args.eval_root, subsets=args.eval_subsets, mode="center")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=get_collate(args.eval_kind))
    model = build_model(cfg["model"]["name"], cfg["model"].get("params", {})).to(device)
    load_checkpoint(args.ckpt, model, device=device)
    res = run_inference(model, loader, device)
    scores, labels = res["scores"], res["labels"]
    ids = res["utt_id"]
    subsets = res["attack"]
    if args.track_level:
        if args.eval_kind != "singfake":
            raise ValueError("track_level is only supported for SingFake")
        scores, labels = track_level_aggregate(scores, res["track_id"], labels)
        first = {}
        for i, uid in enumerate(res["track_id"]):
            first.setdefault(uid, i)
        ids = list(first)
        subsets = [res["attack"][i] for i in first.values()]
    if args.eval_kind == "ctrsvdd":
        metrics = ctrsvdd_metrics(scores, labels, res["attack"], res["source"])
        scope = metrics["protocol"]
    else:
        metrics = evaluate_all(scores, labels)
        sub = np.asarray(subsets)
        metrics["by_subset"] = {s: evaluate_all(scores[sub == s], labels[sub == s]) for s in sorted(set(subsets))}
        scope = "singfake_" + ("track_mean" if args.track_level else "segment") + "_pooled"
        metrics["protocol"] = scope
        metrics["pooling_status"] = "reproduction_choice_not_specified_in_pdf"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scores.jsonl").open("w", encoding="utf-8") as f:
        for i, uid in enumerate(ids):
            row = {"utt_id": uid, "label": int(labels[i]), "score": float(scores[i]), "group": subsets[i]}
            if args.eval_kind == "ctrsvdd":
                row["source"] = res["source"][i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.track_level:
        with (out / "segment_scores.jsonl").open("w", encoding="utf-8") as f:
            for i, uid in enumerate(res["utt_id"]):
                f.write(json.dumps({"utt_id": uid, "track_id": res["track_id"][i],
                                   "score": float(res["scores"][i]), "label": int(res["labels"][i])},
                                  ensure_ascii=False) + "\n")
    ref = Path(__file__).parent / "docs/paper_reference.json"
    paper = Path(__file__).parent.parent / json.loads(ref.read_text(encoding="utf-8"))["source_file"]
    metrics["provenance"] = {"status": "measured", "checkpoint_sha256": sha256(args.ckpt),
                             "config_sha256": sha256(args.config), "pdf_sha256": sha256(paper),
                             "scores_sha256": sha256(out / "scores.jsonl"), "scope": scope,
                             "subsets": sorted(set(subsets)), "evaluation_unit": "track" if args.track_level else "segment"}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if "by_attack" in metrics:
        (out / "by_attack.json").write_text(json.dumps(metrics["by_attack"], indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
