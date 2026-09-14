"""WavLM-Linear baseline（论文 Table 1）。

论文 4.3 节定义：
    "A strong self-supervised learning (SSL) baseline, constructed by optimizing
    a linear classification head over the frozen latent representations extracted
    from the WavLM Base+ encoder."

即：WavLM 完全冻结，只训练一个线性分类头。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .wavlm_stream import WavLMStream


class WavLMLinear(nn.Module):
    """冻结的 WavLM Base+ + 线性分类头。

    与 OPERA-Net 语义流保持一致的时间对齐设置（默认池化到 100 帧），
    池化后再做 mean pooling 送入线性层。
    """

    def __init__(
        self,
        pretrained: str = "microsoft/wavlm-base-plus",
        feat_dim: int = 256,
        num_classes: int = 2,
        target_frames: Optional[int] = 100,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        # finetune=False → 整个 WavLM（含 feature encoder 与所有层）全部冻结
        self.encoder = WavLMStream(
            pretrained=pretrained,
            feat_dim=feat_dim,
            n_frozen_layers=0,
            freeze_feature_encoder=True,
            target_frames=target_frames,
            finetune=False,
        )
        self.pooling = pooling
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, N) → logits (B, 2)。"""
        f = self.encoder(wav)                 # (B, C, T)，冻结特征
        if self.pooling == "mean":
            x = f.mean(dim=-1)
        elif self.pooling == "stats":
            x = torch.cat([f.mean(dim=-1), f.std(dim=-1, unbiased=False)], dim=-1)
            x = self._reduce(x)
        else:
            x = f.max(dim=-1).values
        return self.head(x)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        """stats pooling 时把 2C 维压回 C 维（线性头保持"线性"这一约束）。"""
        if not hasattr(self, "_reduce_layer"):
            self._reduce_layer = nn.Linear(x.size(-1), self.head.in_features).to(x.device)
        return self._reduce_layer(x)

    @torch.no_grad()
    def predict_score(self, wav: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(wav)
        return logits[:, 1] - logits[:, 0]
