"""CtrSVDD 数据集（论文的训练语料）。

清单格式（train.txt / dev.txt / test.txt，空格分隔 6 列）
------------------------------------------------------
    source_corpus  singer_id  utt_id  subsystem  attack_id  label
    kising  CtrSVDD_0125  CtrSVDD_0125_E_0000001  -  A10  deepfake
    ofuton  CtrSVDD_0000  CtrSVDD_0000_T_0000000  -  -     bonafide

对应音频：`{root}/{split}_set/{utt_id}.flac`（16 kHz 单声道 flac）

实测规模（本项目内文件）
----------------------
    train.txt  84,404（bonafide 12,169 / deepfake 72,235，攻击 A01–A08）
    dev.txt    43,625（bonafide  6,547 / deepfake 37,078，攻击 A01–A08）
    test.txt   92,769（bonafide 13,596 / deepfake 79,173，攻击 A09–A14）
    合计      220,798（bonafide 32,312 / deepfake 188,486）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ..audio import NUM_SAMPLES, TARGET_SR, fix_length, load_audio

LABEL_MAP = {"bonafide": 0, "spoof": 1, "deepfake": 1}
TRAIN_ATTACKS = [f"A0{i}" for i in range(1, 9)]      # A01–A08
EVAL_ATTACKS = ["A09", "A10", "A11", "A12", "A13", "A14"]


@dataclass
class Utterance:
    utt_id: str
    path: Path
    label: int                 # 0 = bonafide，1 = spoof
    attack: str                # "-" 表示 bonafide
    source: str
    singer: str
    split: str


def parse_list_file(
    list_file: str | Path,
    audio_dir: str | Path,
    split: str,
    check_exists: bool = False,
) -> List[Utterance]:
    """解析单个清单文件。"""
    list_file, audio_dir = Path(list_file), Path(audio_dir)
    records: List[Utterance] = []
    missing = 0
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(f"Invalid six-column protocol row: {line}")
            source, singer, utt_id, _subs, attack, label = parts[:6]
            if label.lower() not in LABEL_MAP:
                raise ValueError(f"Unknown label: {label}")
            wav_path = audio_dir / f"{utt_id}.flac"
            if check_exists and not wav_path.exists():
                raise FileNotFoundError(wav_path)
            records.append(Utterance(
                utt_id=utt_id,
                path=wav_path,
                label=LABEL_MAP.get(label.lower(), 1),
                attack=attack,
                source=source,
                singer=singer,
                split=split,
            ))
    if check_exists and missing:
        print(f"[ctrsvdd] {list_file.name}: {missing} 条音频缺失，已跳过")
    return records


def build_ctrsvdd_index(
    root: str | Path,
    splits: Sequence[str] = ("train", "dev"),
    check_exists: bool = False,
) -> List[Utterance]:
    """按 split 名组装索引。

    root 指向含 train.txt / dev.txt / test.txt 与 *_set/ 的目录，
    例如 C:/Code/SVDD/CtrSVDD2024_Baseline/dataset
    """
    root = Path(root)
    index: List[Utterance] = []
    for sp in splits:
        list_file = root / f"{sp}.txt"
        audio_dir = root / f"{sp}_set"
        if not list_file.exists():
            raise FileNotFoundError(list_file)
        index.extend(parse_list_file(list_file, audio_dir, sp, check_exists))
    return index


class CtrSVDDDataset(Dataset):
    """CtrSVDD  utterance 级数据集（每条已是数秒的短片段，直接定长裁剪/补零）。"""

    def __init__(
        self,
        index: Sequence[Utterance],
        num_samples: int = NUM_SAMPLES,
        target_sr: int = TARGET_SR,
        mode: Literal["random", "center", "first"] = "random",
        return_wav: bool = True,
    ) -> None:
        self.index = list(index)
        self.num_samples = num_samples
        self.target_sr = target_sr
        self.mode = mode
        self.return_wav = return_wav

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict:
        rec = self.index[i]
        wav = load_audio(rec.path, self.target_sr)
        wav = fix_length(wav, self.num_samples, self.mode)
        item = {
            "wav": wav,
            "label": rec.label,
            "utt_id": rec.utt_id,
            "attack": rec.attack,
            "source": rec.source,
        }
        return item

    def labels(self) -> List[int]:
        return [r.label for r in self.index]


def split_by_attack(index: Sequence[Utterance]) -> Dict[str, List[Utterance]]:
    """按攻击类型分组，便于输出 A09–A13 的分项 EER（论文雷达图口径）。"""
    groups: Dict[str, List[Utterance]] = {}
    for r in index:
        groups.setdefault(r.attack, []).append(r)
    return groups


def collate_fn(batch: List[Dict]) -> Dict:
    """把 list[dict] 整理为可直接送入模型的 batch。"""
    wavs = torch.stack([b["wav"] for b in batch], dim=0)          # (B, N)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {
        "wav": wavs,
        "label": labels,
        "utt_id": [b["utt_id"] for b in batch],
        "attack": [b["attack"] for b in batch],
        "source": [b["source"] for b in batch],
    }
