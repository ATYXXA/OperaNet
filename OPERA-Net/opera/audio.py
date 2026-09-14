"""波形读取与定长处理。

论文 4.2 节：所有音频重采样到 16 kHz，切成固定 4 秒窗口（不足补零，超出裁剪）。
4 s × 16 kHz = 64,000 samples。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import torch

TARGET_SR = 16000
WINDOW_SEC = 4.0
NUM_SAMPLES = int(TARGET_SR * WINDOW_SEC)  # 64000


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    """读取任意格式音频为单声道 16 kHz 张量。

    Returns
    -------
    torch.Tensor: shape (N,)，dtype float32
    """
    y, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))


def fix_length(
    wav: torch.Tensor,
    num_samples: int = NUM_SAMPLES,
    mode: Literal["random", "center", "first"] = "random",
) -> torch.Tensor:
    """将 1-D 波形裁剪/补零到固定长度。

    - 训练阶段用 random 裁剪做数据增广；
    - 评估阶段用 center 保证确定性。
    """
    n = wav.shape[-1]
    if n == num_samples:
        return wav
    if n > num_samples:
        if mode == "random":
            start = int(torch.randint(0, n - num_samples + 1, (1,)).item())
        elif mode == "center":
            start = (n - num_samples) // 2
        else:
            start = 0
        return wav[start:start + num_samples]
    out = torch.zeros(num_samples, dtype=wav.dtype)
    out[:n] = wav
    return out


def load_fixed_length(
    path: str | Path,
    num_samples: int = NUM_SAMPLES,
    target_sr: int = TARGET_SR,
    mode: Literal["random", "center", "first"] = "random",
) -> torch.Tensor:
    """读取 + 定长，dataset 里最常调用的入口。"""
    return fix_length(load_audio(path, target_sr), num_samples, mode)


def split_into_segments(
    path: str | Path,
    segment_sec: float = WINDOW_SEC,
    hop_sec: float | None = None,
    target_sr: int = TARGET_SR,
    min_tail_sec: float = 1.0,
) -> list[torch.Tensor]:
    """把长音频（如 SingFake 的完整歌曲）切成若干 4 秒片段。

    评估阶段通常 hop == segment（无重叠）；hop < segment 可做测试时增广。

    Parameters
    ----------
    min_tail_sec: 结尾不足该长度的残段被丢弃。
    """
    y = load_audio(path, target_sr)
    hop_sec = hop_sec or segment_sec
    seg_len = int(segment_sec * target_sr)
    hop_len = int(hop_sec * target_sr)
    n = y.shape[-1]
    if n < int(min_tail_sec * target_sr):
        return []
    if n <= seg_len:
        return [fix_length(y, seg_len, "center")]
    segs = []
    for start in range(0, n - seg_len + 1, hop_len):
        segs.append(y[start:start + seg_len])
    tail = n - (len(segs) - 1) * hop_len - seg_len if segs else n
    if tail >= int(min_tail_sec * target_sr):
        segs.append(y[-seg_len:])
    return segs


def peak_normalize(wav: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    peak = wav.abs().max()
    return wav / (peak + eps) if peak > eps else wav
