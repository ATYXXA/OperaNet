"""模型注册表：按配置名构造论文中的 4 个系统。

    opera_net      → OPERA-Net（含 4 种消融变体）
    rawnet2        → RawNet2 baseline
    aasist         → AASIST baseline
    wavlm_linear   → WavLM-Linear baseline
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .aasist import AASIST
from .opera_net import OPeraNet, OperaNetConfig, ablation_variant
from .rawnet2 import RawNet2
from .wavlm_linear import WavLMLinear


def build_model(name: str, cfg: Dict[str, Any] | None = None) -> nn.Module:
    """工厂函数。cfg 为 yaml 中对应 model 段的字典。"""
    cfg = dict(cfg or {})

    if name == "opera_net":
        variant = cfg.pop("variant", "full")
        model_cfg = ablation_variant(variant, **cfg)
        return OPeraNet(model_cfg)

    if name == "rawnet2":
        return RawNet2(**cfg)

    if name == "aasist":
        return AASIST(**cfg)

    if name == "wavlm_linear":
        return WavLMLinear(**cfg)

    raise ValueError(f"未知模型 {name}，可选：opera_net / rawnet2 / aasist / wavlm_linear")


__all__ = [
    "AASIST",
    "RawNet2",
    "WavLMLinear",
    "OPeraNet",
    "OperaNetConfig",
    "ablation_variant",
    "build_model",
]
