"""SingFake 数据集（论文的跨库测试集，"in-the-wild"）。

目录布局（本项目的 singfake_dataset/）
------------------------------------
    {bonafide,spoof}/{train,valid,T01,T02}/*.mp3

CSV 元数据：Set,Bonafide Or Spoof,Language,Singer,Title,Model,Url
    Set ∈ {Training, Validation, T01, T02, T04}

与 CtrSVDD 不同，SingFake 是完整歌曲（分钟级），必须先切成 4 秒片段；
评估时可选按曲目聚合分数（track-level）。

实测（本项目）
--------------
CSV 1,481 条；已下载切片 bonafide 1,010 / spoof 789
语言：Mandarin 1,068、Cantonese 213、English 64、Persian 58、Japanese 51 …
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from ..audio import TARGET_SR, fix_length, load_audio

LABEL_ALIASES = {"bonafide": 0, "spoof": 1, "deepfake": 1}
# CSV 中的 Set 名 → 本地目录名
SET_ALIASES = {
    "training": "train",
    "validation": "valid",
    "t01": "T01",
    "t02": "T02",
    "t03": "T03",
    "t04": "T04",
}
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")


@dataclass
class SingFakeTrack:
    path: Path
    label: int
    subset: str
    track_id: str          # 去重后的曲目标识（文件名主干）


@dataclass
class SingFakeSegment:
    path: Path
    start_sec: float
    end_sec: float
    label: int
    subset: str
    track_id: str


def _normalize_subset(name: str) -> str:
    return SET_ALIASES.get(name.strip().lower(), name.strip())


def scan_singfake_tracks(
    root: str | Path,
    subsets: Sequence[str] = ("T02",),
) -> List[SingFakeTrack]:
    """扫描目录下的音频文件，按所在子目录确定标签与子集。"""
    root = Path(root)
    wanted = {_normalize_subset(s) for s in subsets}
    tracks: List[SingFakeTrack] = []

    for label_name, label in (("bonafide", 0), ("spoof", 1)):
        label_dir = root / label_name
        if not label_dir.exists():
            raise FileNotFoundError(label_dir)
        for sub in sorted(wanted):
            sub_dir = label_dir / sub
            if not sub_dir.exists():
                raise FileNotFoundError(f"Required subset missing: {sub_dir}; cannot report a complete evaluation")
            found = []
            for p in sorted(sub_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                    found.append(SingFakeTrack(
                        path=p, label=label, subset=sub, track_id=p.relative_to(root).as_posix(),
                    ))
            if not found:
                raise ValueError(f"No audio in required subset: {sub_dir}")
            tracks.extend(found)
    return tracks


def read_singfake_csv(csv_path: str | Path) -> List[Dict]:
    """读取官方 CSV 元数据（用于统计与溯源，不直接用于建索引）。"""
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class SingFakeDataset(Dataset):
    """SingFake 片段级数据集。

    构造时只建立"曲目 → 片段"的索引（读一次音频时长），真正读波形在 __getitem__。

    Parameters
    ----------
    subsets: 使用哪些子集，论文跨库测试建议 T02（in-the-wild）
    segment_sec / hop_sec: 切片长度与步长，默认 4 s 无重叠
    mode: 固定为 center（评估集不需要随机裁剪）
    """

    def __init__(
        self,
        root: str | Path,
        subsets: Sequence[str] = ("T02",),
        segment_sec: float = 4.0,
        hop_sec: Optional[float] = None,
        target_sr: int = TARGET_SR,
        mode: Literal["random", "center", "first"] = "center",
        min_tail_sec: float = 1.0,
        max_tracks: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.segment_sec = segment_sec
        self.hop_sec = hop_sec or segment_sec
        self.target_sr = target_sr
        self.mode = mode
        self.num_samples = int(segment_sec * target_sr)

        tracks = scan_singfake_tracks(self.root, subsets)
        if max_tracks:
            tracks = tracks[:max_tracks]
        self.tracks = tracks

        # 建立片段索引：先读取每首曲目的时长（只需要 shape，不解码整段内容）
        self.segments: List[SingFakeSegment] = []
        for tr in tracks:
            dur = self._duration(tr.path)
            if dur is None or dur <= 0:
                raise ValueError(f"Cannot read audio duration: {tr.path}")
            seg_len = self.segment_sec
            hop = self.hop_sec
            if dur <= seg_len:
                self.segments.append(SingFakeSegment(
                    tr.path, 0.0, dur, tr.label, tr.subset, tr.track_id))
                continue
            n_seg = int((dur - seg_len) // hop) + 1
            for i in range(n_seg):
                st = i * hop
                self.segments.append(SingFakeSegment(
                    tr.path, st, st + seg_len, tr.label, tr.subset, tr.track_id))
            tail = dur - (n_seg - 1) * hop - seg_len
            if tail >= min_tail_sec:
                self.segments.append(SingFakeSegment(
                    tr.path, dur - seg_len, dur, tr.label, tr.subset, tr.track_id))

    @staticmethod
    def _duration(path: Path) -> Optional[float]:
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return float(info.duration)
        except Exception:
            pass
        try:
            import librosa
            return float(librosa.get_duration(path=str(path)))
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, i: int) -> Dict:
        seg = self.segments[i]
        y, _ = self._load_segment(seg)
        wav = fix_length(y, self.num_samples, self.mode)
        return {
            "wav": wav,
            "label": seg.label,
            "utt_id": f"{seg.track_id}#{seg.start_sec:.2f}",
            "track_id": seg.track_id,
            "attack": seg.subset,
        }

    def _load_segment(self, seg: SingFakeSegment):
        import librosa
        y, _ = librosa.load(
            str(seg.path), sr=self.target_sr, mono=True,
            offset=seg.start_sec, duration=min(self.segment_sec, seg.end_sec - seg.start_sec),
        )
        return torch.from_numpy(y.astype("float32")), self.target_sr

    def labels(self) -> List[int]:
        return [s.label for s in self.segments]

    def track_ids(self) -> List[str]:
        return [s.track_id for s in self.segments]

    def summary(self) -> Dict[str, int]:
        n_bon = sum(1 for s in self.segments if s.label == 0)
        return {
            "tracks": len(self.tracks),
            "segments": len(self.segments),
            "bonafide_segments": n_bon,
            "spoof_segments": len(self.segments) - n_bon,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    wavs = torch.stack([b["wav"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    out = {
        "wav": wavs,
        "label": labels,
        "utt_id": [b["utt_id"] for b in batch],
        "attack": [b["attack"] for b in batch],
    }
    if "track_id" in batch[0]:
        out["track_id"] = [b["track_id"] for b in batch]
    return out
