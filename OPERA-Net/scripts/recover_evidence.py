"""Recover local records and source statistics, never infer predictions from EER.

Run from any directory. Uses only the Python standard library. Output is JSON/JSONL
and Markdown; CSV inputs are kept unchanged. FLAC STREAMINFO is inspected, not
decoded, and its stored audio MD5 is not claimed as a newly verified checksum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIO = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac"}
SUBSETS = {"Training": "train", "Validation": "valid", "T01": "T01", "T02": "T02", "T03": "T03", "T04": "T04"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def flac_info(path):
    with path.open("rb") as f:
        if f.read(4) != b"fLaC":
            raise ValueError("not a native FLAC stream")
        header = f.read(4)
        if len(header) != 4 or header[0] & 127 != 0 or int.from_bytes(header[1:], "big") != 34:
            raise ValueError("missing STREAMINFO")
        b = f.read(34)
        if len(b) != 34:
            raise ValueError("truncated STREAMINFO")
    packed = int.from_bytes(b[10:18], "big")
    sr = packed >> 44
    frames = packed & ((1 << 36) - 1)
    if not sr or not frames:
        raise ValueError("unknown sample rate or frame count")
    return {"sample_rate": sr, "channels": ((packed >> 41) & 7) + 1,
            "bits_per_sample": ((packed >> 36) & 31) + 1, "num_samples": frames,
            "duration_seconds": frames / sr, "stored_pcm_md5": b[18:34].hex()}


def recover_ctr(root, out):
    data = root / "CtrSVDD2024_Baseline/dataset"
    stats, attacks, sources = [], Counter(), Counter()
    ids, singers, hashes = {}, {}, defaultdict(set)
    missing, invalid = [], []
    for split in ("train", "dev", "test"):
        records, seen, singers_here = [], set(), set()
        labels, rates, channels = Counter(), Counter(), Counter()
        for line_no, line in enumerate((data / f"{split}.txt").read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            p = line.split()
            if len(p) != 6 or p[-1] not in ("bonafide", "deepfake", "spoof"):
                raise ValueError(f"Invalid protocol row {split}:{line_no}")
            source, singer, uid, system, attack, label = p
            if uid in seen:
                raise ValueError(f"Duplicate utterance ID {split}:{uid}")
            seen.add(uid)
            singers_here.add(singer)
            label_id = int(label != "bonafide")
            path = data / f"{split}_set" / f"{uid}.flac"
            exists = path.is_file()
            row = dict(dataset="CtrSVDD", split=split, utt_id=uid, source=source, singer_id=singer,
                       subsystem=system, attack=attack, label=label, label_id=label_id,
                       path=path.relative_to(root).as_posix(), exists=exists,
                       source_protocol=f"CtrSVDD2024_Baseline/dataset/{split}.txt", source_line=line_no,
                       # 论文 Table 1 口径：test split 中排除 A14 的 spoof，
                       # 保留全部 bonafide（含 acesinger 的 2,845 条）
                       provenance="local_protocol", paper_primary_eval=(split == "test" and attack != "A14"))
            if exists:
                row["size_bytes"] = path.stat().st_size
                try:
                    row.update(flac_info(path))
                    rates[row["sample_rate"]] += 1
                    channels[row["channels"]] += 1
                    key = row["stored_pcm_md5"]
                    if key != "0" * 32:
                        hashes[key].add(split)
                except (ValueError, OSError) as exc:
                    row["header_error"] = str(exc)
                    invalid.append(row["path"])
            else:
                missing.append(row["path"])
            labels[label_id] += 1
            attacks[(split, attack, label)] += 1
            sources[(split, source)] += 1
            records.append(row)
        write_jsonl(out / f"ctrsvdd_{split}.jsonl", records)
        ids[split], singers[split] = seen, singers_here
        s = dict(split=split, total=len(records), bonafide=labels[0], spoof=labels[1], singers=len(singers_here),
                 audio_present=sum(r["exists"] for r in records), sample_rates=dict(rates), channels=dict(channels),
                 header_duration_hours=sum(r.get("duration_seconds", 0) for r in records) / 3600)
        if split == "test":
            primary = [r for r in records if r["paper_primary_eval"]]
            write_jsonl(out / "ctrsvdd_eval_a09_a13.jsonl", primary)
            s["primary_eval"] = dict(total=len(primary), bonafide=sum(r["label_id"] == 0 for r in primary),
                                     spoof=sum(r["label_id"] == 1 for r in primary), singers=len({r["singer_id"] for r in primary}))
        stats.append(s)
        print(f"CtrSVDD {split}: {s['total']} records, {s['audio_present']} files", flush=True)
    overlaps = [{"splits": [a, b], "utterance_ids": len(ids[a] & ids[b]), "singer_ids": len(singers[a] & singers[b])}
                for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))]
    return {"splits": stats, "split_overlap": overlaps,
            "stored_pcm_md5_cross_split_groups": sum(len(v) > 1 for v in hashes.values()),
            "missing_audio": missing, "invalid_flac_headers": invalid,
            "attack_distribution": [dict(split=k[0], attack=k[1], label=k[2], count=v) for k, v in sorted(attacks.items())],
            "source_distribution": [dict(split=k[0], source=k[1], count=v) for k, v in sorted(sources.items())]}


def candidates(title):
    safe = "".join(c for c in title if c.isalnum() or c in " _-（）《》.").strip()[:100].replace(" ", "_")
    return {title.strip(), safe}


def recover_sing(root, out):
    base = root / "singfake_dataset"
    with (base / "singfake.csv").open(encoding="utf-8-sig", newline="") as f:
        metadata = list(csv.DictReader(f))
    lookup = defaultdict(list)
    for n, row in enumerate(metadata, 2):
        for name in candidates(row["Title"]):
            lookup[(SUBSETS.get(row["Set"], row["Set"]), row["Bonafide Or Spoof"].lower(), name)].append(n)
    files, counts = [], Counter()
    matched_rows = set()
    for label in ("bonafide", "spoof"):
        for path in sorted((base / label).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO:
                continue
            relative = path.relative_to(base)
            subset = relative.parts[1]
            hits = sorted(set(lookup.get((subset, label, path.stem), [])))
            if len(hits) == 1:
                matched_rows.add(hits[0])
            files.append(dict(dataset="SingFake", subset=subset, label=label, label_id=int(label == "spoof"),
                              file_id=relative.as_posix(), path=path.relative_to(root).as_posix(),
                              size_bytes=path.stat().st_size, provenance="local_audio_path",
                              metadata_candidate_lines=hits,
                              metadata_link_status="exact_filename_candidate_not_content_verified" if len(hits) == 1 else "ambiguous" if hits else "unmatched"))
            counts[(subset, label)] += 1
    write_jsonl(out / "singfake_audio_inventory.jsonl", files)
    write_jsonl(out / "singfake_source_metadata.jsonl", [dict(source_line=i, provenance="local_csv", **r) for i, r in enumerate(metadata, 2)])
    all_summaries = []
    for source_set, subset in SUBSETS.items():
        labels = {"bonafide", "spoof"} | {r["Bonafide Or Spoof"].lower() for r in metadata if r["Set"] == source_set}
        for label in sorted(labels):
            relevant = [r for r in metadata if r["Set"] == source_set and r["Bonafide Or Spoof"].lower() == label]
            all_summaries.append(dict(subset=subset, label=label, metadata_rows=len(relevant),
                                      audio_files=counts[(subset, label)], singers=len({r["Singer"] for r in relevant})))
    # Keep derivations as a recipe. Unknown codec settings MUST NOT become invented original data.
    t02 = [r for r in files if r["subset"] == "T02"]
    write_jsonl(out / "t03_parent_candidates.jsonl", [dict(parent_file_id=r["file_id"], source_path=r["path"],
                label=r["label"], codec=None, bitrate=None, output_path=None,
                status="recipe_incomplete_not_original_T03") for r in t02])
    return dict(metadata_rows=len(metadata), local_audio_files=len(files),
                exact_filename_candidate_rows=len(matched_rows), unmatched_metadata_rows=len(metadata)-len(matched_rows),
                groups=all_summaries, metadata_link_counts=dict(Counter(r["metadata_link_status"] for r in files)),
                language_distribution=dict(Counter(r["Language"] for r in metadata)),
                model_distribution=dict(Counter(r["Model"] for r in metadata)),
                t03_audio_files=sum(counts[("T03", l)] for l in ("bonafide", "spoof")))


def inventory(root):
    result = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name.startswith(".venv") or directory.name in ("tmp",):
            continue
        counts, size, checkpoints = Counter(), 0, []
        for folder, dirs, files in os.walk(directory):
            dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git", "current_pdf"))
            for name in files:
                p = Path(folder) / name
                counts[p.suffix.lower() or "<no_extension>"] += 1
                size += p.stat().st_size
                if p.suffix.lower() in (".pt", ".pth", ".ckpt", ".bin", ".safetensors"):
                    checkpoints.append(p.relative_to(root).as_posix())
        result.append(dict(directory=directory.name, files=sum(counts.values()), bytes=size,
                           extensions=dict(counts), checkpoint_paths=checkpoints))
    return result


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    out = args.out or root / "restored_data/current_pdf"
    out.mkdir(parents=True, exist_ok=True)
    paper = root / "OPERA-Net/docs/paper_reference.json"
    ref = json.loads(paper.read_text(encoding="utf-8"))
    ref["pdf_sha256"] = digest(root / ref["source_file"])
    write_json(out / "paper_reference.json", ref)
    manifest = []
    for p in [root / ref["source_file"], root / "tmp_template5.txt", paper,
              root / "wavlm-base-plus/config.json", root / "singfake_dataset/singfake.csv",
              *sorted((root / "CtrSVDD2024_Baseline/dataset").glob("*.txt")),
              root / "Feature/雷达图.py", root / "Feature/feature_gating_visualization.txt",
              root / "SingFake-main/dataset/classify_scripts/simulate_codec.py"]:
        manifest.append(dict(path=p.relative_to(root).as_posix(), size_bytes=p.stat().st_size, sha256=digest(p)))
    write_json(out / "source_hashes.json", manifest)
    write_json(out / "directory_inventory.json", inventory(root))
    ctr = recover_ctr(root, out)
    sing = recover_sing(root, out)
    result = dict(audit_time_utc=datetime.now(timezone.utc).isoformat(), ctrsvdd=ctr, singfake=sing,
                  verification_scope="all CtrSVDD protocol rows and FLAC headers; SingFake paths and CSV; no full audio decode or content-label verification",
                  cannot_recover=["original OPERA-Net checkpoints and prediction scores", "original random seeds or training history", "original T03 codec configuration", "unique raw observations from aggregated EER values"])
    write_json(out / "data_audit.json", result)
    print(json.dumps({"out": str(out), "CtrSVDD": [s["total"] for s in ctr["splits"]],
                      "SingFake": sing["local_audio_files"], "T03": sing["t03_audio_files"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
