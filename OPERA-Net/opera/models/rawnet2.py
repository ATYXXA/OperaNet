"""RawNet2 baseline（论文 Table 1）。

参考 Tak et al., "End-to-end audio deepfake detection with RawNet2", ICASSP 2021：
SincNet 参数化前端 → 6 个残差块 → GRU（取最后隐状态）→ 全连接 → 2 类。

输入：原始波形（论文 Table 1 标注为 "Raw Waveform (Sinc)"）。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """参数化 Sinc 滤波器组（SincNet）。

    只用两个可学习参数（低/高频截止）生成带通滤波器，参数量极小。
    """

    def __init__(
        self,
        out_channels: int = 20,
        kernel_size: int = 251,
        sample_rate: int = 16000,
        min_low_hz: int = 50,
        min_band_hz: int = 50,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30.0
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel = torch.linspace(self._to_mel(low_hz), self._to_mel(high_hz), out_channels + 1)
        hz = self._to_hz(mel)

        self.low_hz = nn.Parameter(hz[:-1].view(-1, 1))
        self.band_hz = nn.Parameter(torch.diff(hz).view(-1, 1))

        n = (kernel_size - 1) / 2.0
        self.register_buffer("window", 0.54 - 0.46 * torch.cos(
            2 * torch.pi * torch.arange(kernel_size).float() / (kernel_size - 1)))
        self.register_buffer("n", torch.arange(-n, n + 1).view(1, -1) / sample_rate)

    @staticmethod
    def _to_mel(hz):
        return 2595 * torch.log10(1 + hz / 700.0)

    @staticmethod
    def _to_hz(mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, N) → (B, out_channels, N')。"""
        low = self.min_low_hz + torch.abs(self.low_hz)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz),
                           self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t = torch.matmul(low, self.n)
        low_pass1 = 2 * low * torch.sinc(2 * low * self.n)
        low_pass2 = 2 * high * torch.sinc(2 * high * self.n)
        band_pass = low_pass2 - low_pass1
        # 保证带通滤波器能量归一化
        band_pass = band_pass / (2 * band[:, None] + 1e-8)

        filters = (band_pass * self.window).view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


class ResidualBlock(nn.Module):
    """RawNet2 的残差块：两个 1-D 卷积 + BatchNorm + 跳跃连接。"""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.LeakyReLU(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


class RawNet2(nn.Module):
    """RawNet2 端到端伪造检测模型。

    默认配置对齐 RawNet2 论文的 6-block 结构，适配 4 s @16 kHz 输入。
    """

    def __init__(
        self,
        num_classes: int = 2,
        sinc_out: int = 20,
        sinc_kernel: int = 251,
        res_channels: List[int] = (20, 128, 128, 256, 256, 512),
        gru_hidden: int = 1024,
        gru_layers: int = 3,
        fc_hidden: int = 1024,
        sample_rate: int = 16000,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sinc = SincConv(sinc_out, sinc_kernel, sample_rate)
        self.bn0 = nn.BatchNorm1d(sinc_out)

        blocks = []
        in_ch = sinc_out
        for ch in res_channels:
            blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch),
                nn.LeakyReLU(0.3),
                ResidualBlock(ch),
                nn.MaxPool1d(3),
            ))
            in_ch = ch
        self.blocks = nn.Sequential(*blocks)

        self.gru = nn.GRU(
            input_size=in_ch,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(gru_hidden, fc_hidden)
        self.out = nn.Linear(fc_hidden, num_classes)
        self.act = nn.LeakyReLU(0.3)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, N) → logits (B, 2)。"""
        x = wav.unsqueeze(1)                 # (B, 1, N)
        x = self.act(self.bn0(self.sinc(x)))
        x = self.blocks(x)                   # (B, C, T)
        x = x.transpose(1, 2)                # (B, T, C)
        out, _ = self.gru(x)
        x = out[:, -1, :]                    # 取最后时间步
        x = self.act(self.fc(x))
        return self.out(x)

    @torch.no_grad()
    def predict_score(self, wav: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(wav)
        return logits[:, 1] - logits[:, 0]
