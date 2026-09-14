"""Semantic-Guided Gating 融合 + 分类头。

论文 3.3 节
-----------
背景音乐在非人声区主导 CQT 谱，直接拼接 F_sig 与 F_sem 会引入大量噪声。
做法是用 WavLM 的语义先验生成一个软注意力掩码来"门控"信号特征：

    G      = σ( MLP(F_sem) )                    G ∈ [0, 1]^{C×T}      (3)
    F̃_sig  = F_sig ⊙ G                                                (4)
    y      = Softmax( Linear( Concat[F_sem, F̃_sig] ) )                (5)

物理含义：WavLM 判定为人声概率低的片段（纯伴奏段）其 CQT 特征被抑制，
人声丰富区域的特征被增强。

分类器头为两层全连接（论文 3.3 节末）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SemanticGuidedGating(nn.Module):
    """由语义特征生成软门控掩码。

    MLP 用两个 1×1 卷积实现（等价于逐时间步的通道 MLP），
    保持时间分辨率不变，因此 G ∈ [0,1]^{C×T}。
    """

    def __init__(self, feat_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or feat_dim
        self.mlp = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, feat_dim, kernel_size=1),
        )

    def forward(self, f_sem: torch.Tensor) -> torch.Tensor:
        """f_sem: (B, C, T) → G: (B, C, T)，取值 [0, 1]。"""
        return torch.sigmoid(self.mlp(f_sem))


class GatedFusion(nn.Module):
    """门控融合 + 两层全连接分类头。

    Parameters
    ----------
    feat_dim: C
    num_classes: 2（bonafide / spoof）
    use_gating: False 时退化为朴素拼接（Table 2 消融的 "+ PC-CQT (w/ Phase)"）
    pooling: "stats"（mean+std）或 "mean" / "attentive"
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int = 2,
        use_gating: bool = True,
        pooling: str = "stats",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
        in_streams: int = 2,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.use_gating = use_gating
        self.pooling = pooling
        # in_streams=2 → Concat[F_sem, F̃_sig]；in_streams=1 → 只有 F_sem
        # （Table 2 的 "WavLM Stream (Fine-tuned)" 变体走这一支）
        self.in_streams = in_streams
        self.gating = SemanticGuidedGating(feat_dim) if use_gating else None

        pooled_dim = feat_dim * in_streams if pooling in ("mean", "attentive") \
            else feat_dim * in_streams * 2
        hidden_dim = hidden_dim or feat_dim

        if pooling == "attentive":
            self.attn = nn.Sequential(
                nn.Conv1d(feat_dim * in_streams, feat_dim, kernel_size=1),
                nn.Tanh(),
                nn.Conv1d(feat_dim, 1, kernel_size=1),
            )
        else:
            self.attn = None

        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        f_sig: Optional[torch.Tensor],
        f_sem: torch.Tensor,
        return_gate: bool = False,
    ):
        """f_sig: (B, C, T) 或 None；f_sem: (B, C, T)。

        Returns
        -------
        logits: (B, num_classes)
        gate（可选）: (B, C, T)
        """
        if f_sig is not None:
            if self.use_gating and self.gating is not None:
                gate = self.gating(f_sem)
                f_sig_tilde = f_sig * gate                 # 公式 (4)
            else:
                gate = torch.ones_like(f_sig)
                f_sig_tilde = f_sig
            fused = torch.cat([f_sem, f_sig_tilde], dim=1)  # 公式 (5)
        else:
            gate = None
            fused = f_sem

        if self.pooling == "stats":
            mean = fused.mean(dim=-1)
            std = fused.std(dim=-1, unbiased=False)
            pooled = torch.cat([mean, std], dim=-1)
        elif self.pooling == "attentive" and self.attn is not None:
            w = torch.softmax(self.attn(fused), dim=-1)     # (B, 1, T)
            pooled = (fused * w).sum(dim=-1)
        else:
            pooled = fused.mean(dim=-1)

        logits = self.head(pooled)
        if return_gate:
            return logits, gate
        return logits
