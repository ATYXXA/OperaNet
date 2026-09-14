"""PC-CQT：Phase-Consistent Constant-Q Transform 前端。

论文 3.1 节
-----------
给定波形 x，先求复 CQT 谱 X(k, n) = M(k, n) · e^{j φ(k, n)}，其中 k 为频率 bin、n 为帧索引。
常规做法只取对数幅度 S_mag = log(1 + M(k, n))。
OPERA-Net 额外显式建模相位的时间演化，计算瞬时频率（IF）导数：

    S_phase(k, n) = wrap( φ(k, n) − φ(k, n−1) )        wrap 到 (−π, π]

该导数特征会突出合成歌声中神经声码器留下的非自然相位跳变。
最终按通道拼接得到 X_pc ∈ R^{2×K×N}。

论文 4.2 节参数
---------------
- 采样率 16 kHz，窗口 4 s（64,000 samples）
- f_min = 32.7 Hz（C1）
- 84 个频率 bin，每八度 12 个 bin
- hop_length = 320 samples（帧率 50 Hz，天然对齐 WavLM 的 50 Hz 帧率）
- 前端实现：nnAudio

实现说明
--------
优先使用 nnAudio.CQT1992v2（GPU on-the-fly，可端到端训练）。
显式选择 prefer_nnaudio=False 时使用 `TorchCQT` 诊断后端：预计算复数核 + conv1d。
该后端未经 nnAudio 数值等价验证，不能用于声称严格前端复现。默认缺 nnAudio 即报错。
两条路径都返回 complex64 张量 (B, K, T)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文 4.2 节默认配置
DEFAULT_SR = 16000
DEFAULT_FMIN = 32.7           # C1
DEFAULT_N_BINS = 84
DEFAULT_BINS_PER_OCTAVE = 12
DEFAULT_HOP = 320


@dataclass
class CQTConfig:
    sr: int = DEFAULT_SR
    fmin: float = DEFAULT_FMIN
    n_bins: int = DEFAULT_N_BINS
    bins_per_octave: int = DEFAULT_BINS_PER_OCTAVE
    hop_length: int = DEFAULT_HOP
    window: str = "hann"
    # Q 因子：Q = 1 / (2^(1/bins_per_octave) − 1)，为标准 Constant-Q 定义
    norm: bool = True


class TorchCQT(nn.Module):
    """纯 PyTorch 的复数 Constant-Q Transform（nnAudio 的回退实现）。

    对每个 bin k：
        f_k    = fmin · 2^(k / bins_per_octave)
        N_k    = ceil(Q · sr / f_k)
        h_k[n] = w(N_k)[n] · exp(−2πi f_k n / sr) / N_k,  n = 0 … N_k−1

    所有核左侧补零对齐到最长核后合并为一个 conv1d 权重，因此各 bin 的
    时间中心一致，帧间相位差才有物理意义（这正是 PC-CQT 的前提）。
    """

    def __init__(self, cfg: CQTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        q = 1.0 / (2.0 ** (1.0 / cfg.bins_per_octave) - 1.0)
        freqs = cfg.fmin * 2.0 ** (np.arange(cfg.n_bins) / cfg.bins_per_octave)
        lengths = np.ceil(q * cfg.sr / freqs).astype(int)
        max_len = int(lengths.max())

        real_k = np.zeros((cfg.n_bins, max_len), dtype=np.float32)
        imag_k = np.zeros((cfg.n_bins, max_len), dtype=np.float32)
        for k, (fk, nk) in enumerate(zip(freqs, lengths)):
            if cfg.window == "hann":
                win = np.hanning(nk + 2)[1:-1]
            elif cfg.window == "hamming":
                win = np.hamming(nk)
            else:
                win = np.ones(nk)
            t = np.arange(nk)
            ph = -2.0 * np.pi * fk * t / cfg.sr
            scale = 1.0 / nk if cfg.norm else 1.0
            real_k[k, :nk] = np.cos(ph) * win * scale
            imag_k[k, :nk] = np.sin(ph) * win * scale

        kernel = np.concatenate([real_k, imag_k], axis=0)  # (2K, max_len)
        self.register_buffer("kernel", torch.from_numpy(kernel[:, None, :].copy()))
        self.max_len = max_len
        self.padding = max_len // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N) 波形 → (B, K, T) complex64。"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)                                   # (B, 1, N)
        out = F.conv1d(x, self.kernel, stride=self.cfg.hop_length,
                       padding=self.padding)                 # (B, 2K, T)
        k = self.cfg.n_bins
        real, imag = out[:, :k], out[:, k:]
        return torch.complex(real, imag)


class NnAudioCQT(nn.Module):
    """nnAudio 封装，返回 complex64 的 (B, K, T)。"""

    def __init__(self, cfg: CQTConfig, verbose: bool = False) -> None:
        super().__init__()
        from nnAudio import features  # noqa: WPS433（延迟导入，仅在需要时失败）

        self.cfg = cfg
        self.cqt = features.CQT1992v2(
            sr=cfg.sr,
            hop_length=cfg.hop_length,
            fmin=cfg.fmin,
            n_bins=cfg.n_bins,
            bins_per_octave=cfg.bins_per_octave,
            verbose=bool(verbose),
            output_format="Complex",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = self.cqt(x)                    # (B, K, T, 2)
        real, imag = out[..., 0], out[..., 1]
        return torch.complex(real, imag)


def build_cqt(cfg: CQTConfig, prefer_nnaudio: bool = True, verbose: bool = False) -> nn.Module:
    """默认要求 nnAudio；仅显式关闭时采用近似 TorchCQT。"""
    if prefer_nnaudio:
        try:
            return NnAudioCQT(cfg, verbose=verbose)
        except ImportError as exc:
            raise ImportError("Paper frontend requires nnAudio; install it, or explicitly select prefer_nnaudio=False for an approximate diagnostic backend") from exc
    return TorchCQT(cfg)


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """把角度 wrap 到 (−π, π]，对应论文公式 (2) 的 wrap(·)。"""
    wrapped = torch.remainder(x + torch.pi, 2.0 * torch.pi) - torch.pi
    return torch.where(wrapped <= -torch.pi, wrapped + 2.0 * torch.pi, wrapped)


class PCCQTFrontend(nn.Module):
    """Phase-Consistent CQT 前端。

    Parameters
    ----------
    use_phase: 是否拼接相位导数通道。
        True  → X_pc ∈ R^{2×K×T}（PC-CQT）
        False → X ∈ R^{1×K×T}（magnitude-only CQT，用于 Table 2 消融）

    Returns
    -------
    torch.Tensor: (B, C_in, K, T)，C_in = 2 或 1
    """

    def __init__(
        self,
        sr: int = DEFAULT_SR,
        fmin: float = DEFAULT_FMIN,
        n_bins: int = DEFAULT_N_BINS,
        bins_per_octave: int = DEFAULT_BINS_PER_OCTAVE,
        hop_length: int = DEFAULT_HOP,
        use_phase: bool = True,
        eps: float = 1e-8,
        prefer_nnaudio: bool = True,
        n_frames: Optional[int] = None,
    ) -> None:
        super().__init__()
        cfg = CQTConfig(sr=sr, fmin=fmin, n_bins=n_bins,
                        bins_per_octave=bins_per_octave, hop_length=hop_length)
        self.cfg = cfg
        self.cqt = build_cqt(cfg, prefer_nnaudio=prefer_nnaudio)
        self.use_phase = use_phase
        self.eps = eps
        self.n_frames = n_frames
        self.out_channels = 2 if use_phase else 1

    @property
    def n_bins(self) -> int:
        return self.cfg.n_bins

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, N) → (B, C_in, K, T)。"""
        x = self.cqt(wav)                                   # (B, K, T) complex
        if self.n_frames is not None and x.shape[-1] > self.n_frames:
            x = x[..., :self.n_frames]

        mag = torch.log1p(torch.abs(x))                     # S_mag = log(1 + |X|)
        feats = [mag]

        if self.use_phase:
            phase = torch.angle(x)                           # φ(k, n)
            dphase = phase[..., 1:] - phase[..., :-1]        # φ(k,n) − φ(k,n−1)
            dphase = wrap_to_pi(dphase)
            # 首帧无前驱，补零保持时间维对齐
            dphase = F.pad(dphase, (1, 0), mode="constant", value=0.0)
            feats.append(dphase)

        out = torch.stack(feats, dim=1)                      # (B, C_in, K, T)
        return out

    def extra_repr(self) -> str:
        return (f"fmin={self.cfg.fmin}, n_bins={self.cfg.n_bins}, "
                f"bins_per_octave={self.cfg.bins_per_octave}, "
                f"hop={self.cfg.hop_length}, use_phase={self.use_phase}")


def expected_frames(num_samples: int, hop_length: int = DEFAULT_HOP) -> int:
    """4 s @16 kHz、hop 320 → 200 帧（50 Hz）。"""
    return int(np.ceil(num_samples / hop_length))
