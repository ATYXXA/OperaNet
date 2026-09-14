"""信号流编码器：轻量 ResNet-18，把 PC-CQT 图编码成 F_sig ∈ R^{C×T}。

论文 3.1 节："This dual-channel tensor is processed by a lightweight ResNet-18
encoder to extract the signal-level embedding F_sig ∈ R^{C×T}."

输入 (B, 2, 84, 200)（频率 84 × 时间 200）。
下采样策略：频率维逐步压缩到 1，时间维只下采样 2 倍（200 → 100），
以便与 WavLM 侧对齐（本地 4 秒输入原生 199 帧，自适应池化到 100）。
"""

from __future__ import annotations

from typing import Callable, List, Optional, Type

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet 基础残差块（两层 3×3 卷积）。"""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=(stride, 1), padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()

        self.downsample: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=(stride, 1), bias=False),
                norm_layer(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        out = out + self.downsample(x)
        return self.relu(out)


class ResNet18Encoder(nn.Module):
    """轻量 ResNet-18（[2, 2, 2, 2] blocks）用于 PC-CQT 编码。

    Parameters
    ----------
    in_channels: 输入通道数，PC-CQT 为 2（幅度 + 相位导数），magnitude-only 为 1。
    feat_dim: 输出通道 C。
    freq_strides: 频率维各 stage 的 stride，默认把 84 压到 ~6 后自适应池化成 1。
    time_stride: 时间维 stride，默认 stem 处下采样 2 倍（200 → 100）。
    """

    def __init__(
        self,
        in_channels: int = 2,
        feat_dim: int = 256,
        base_channels: int = 64,
        blocks: List[int] = (2, 2, 2, 2),
        block: Type[BasicBlock] = BasicBlock,
        freq_strides: tuple = (1, 2, 2, 2),
        time_stride: int = 2,
        drop_rate: float = 0.0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        self.in_planes = base_channels

        # stem：时间维一次性下采样 2 倍，频率维保持
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7,
                      stride=(1, time_stride), padding=3, bias=False),
            norm_layer(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(1, 0)),
        )

        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        strides = freq_strides
        self.layer1 = self._make_layer(block, channels[0], blocks[0], stride=strides[0],
                                       norm_layer=norm_layer, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, channels[1], blocks[1], stride=strides[1],
                                       norm_layer=norm_layer, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, channels[2], blocks[2], stride=strides[2],
                                       norm_layer=norm_layer, drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, channels[3], blocks[3], stride=strides[3],
                                       norm_layer=norm_layer, drop_rate=drop_rate)

        self.out_channels = channels[3]
        # 通道对齐到 feat_dim，并把频率维压成 1
        self.proj = nn.Sequential(
            nn.Conv2d(self.out_channels, feat_dim, kernel_size=1, bias=False),
            norm_layer(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # (B, feat_dim, 1, T)
        self.feat_dim = feat_dim

        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer, drop_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s,
                                norm_layer=norm_layer, drop_rate=drop_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, K, T) → F_sig: (B, feat_dim, T)。"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        x = self.freq_pool(x)          # (B, feat_dim, 1, T)
        return x.squeeze(2)            # (B, feat_dim, T)
