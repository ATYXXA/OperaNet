"""损失函数：论文 3.3 节使用的加权交叉熵（处理类别不平衡）。

CtrSVDD 官方训练子集极度不平衡（bonafide 12,169 vs deepfake 72,235，约 1:5.9），
因此按类别频率反比加权。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    labels: Sequence[int] | Iterable[int],
    num_classes: int = 2,
    mode: str = "inverse",
) -> torch.Tensor:
    """由标签统计计算类别权重。

    mode
    ----
    inverse: w_c = N / (num_classes · n_c)（sklearn 的 'balanced' 策略）
    sqrt   : w_c = sqrt(N / n_c)（更温和，样本极少时更稳）
    """
    labels = list(labels)
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for l in labels:
        counts[int(l)] += 1
    counts = torch.clamp(counts, min=1.0)
    total = counts.sum()
    if mode == "sqrt":
        w = torch.sqrt(total / counts)
    else:
        w = total / (num_classes * counts)
    return w.float()


class WeightedCrossEntropy(nn.Module):
    """带类别权重的交叉熵，可选 label smoothing。"""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


class FocalLoss(nn.Module):
    """可选替代：Focal Loss，进一步缓解易分样本主导梯度的问题。"""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none",
                             weight=self.alpha.to(logits.device) if self.alpha is not None else None)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(name: str = "weighted_ce", **kwargs) -> nn.Module:
    if name == "weighted_ce":
        return WeightedCrossEntropy(**kwargs)
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "focal":
        return FocalLoss(**kwargs)
    raise ValueError(f"未知损失 {name}")
