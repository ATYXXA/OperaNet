"""OPERA-Net 主模型（含 Table 2 的四种消融变体开关）。

论文 3 节的三段式结构：
    PC-CQT signal stream  →  F_sig
    WavLM semantic stream →  F_sem
    Semantic-Guided Gating fusion → y

消融开关（对应论文 Table 2）
----------------------------
| 变体                          | use_signal | use_phase | use_gating |
|-------------------------------|-----------|-----------|------------|
| WavLM Stream (Fine-tuned)     | False     | -         | -          |
| + CQT (Magnitude-only)        | True      | False     | False      |
| + PC-CQT (w/ Phase)           | True      | True      | False      |
| OPERA-Net (Full w/ Gating)    | True      | True      | True       |

输出约定
--------
forward 返回 logits (B, 2)，index 0 = bonafide，index 1 = spoof（deepfake）。
推理分数取 logits[:, 1] − logits[:, 0]（越大越像伪造），该约定贯穿 evaluate.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..cqt import PCCQTFrontend
from .fusion import GatedFusion
from .resnet18 import ResNet18Encoder
from .wavlm_stream import WavLMStream


@dataclass
class OperaNetConfig:
    # ---- 音频 / 前端 ----
    sample_rate: int = 16000
    num_samples: int = 64000          # 4 s
    fmin: float = 32.7                # C1
    n_bins: int = 84
    bins_per_octave: int = 12
    hop_length: int = 320
    n_frames: int = 200               # 64000 / 320
    prefer_nnaudio: bool = True

    # ---- 语义流 ----
    wavlm_path: str = "microsoft/wavlm-base-plus"
    feat_dim: int = 256
    n_frozen_layers: int = 6
    freeze_feature_encoder: bool = True
    target_frames: int = 100          # WavLM 200 帧 → 池化到 100
    finetune_wavlm: bool = True

    # ---- 信号流 ----
    resnet_base_channels: int = 64
    resnet_drop_rate: float = 0.0

    # ---- 消融开关 ----
    use_signal_stream: bool = True
    use_phase: bool = True
    use_gating: bool = True

    # ---- 分类头 ----
    pooling: str = "stats"
    dropout: float = 0.2
    num_classes: int = 2

    variant_name: str = field(default="OPERA-Net (Full w/ Gating)")


class OPeraNet(nn.Module):
    """OPERA-Net 双流架构。"""

    def __init__(self, cfg: OperaNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ---------- 信号流：PC-CQT + ResNet-18 ----------
        self.signal_stream: Optional[nn.Module] = None
        if cfg.use_signal_stream:
            in_channels = 2 if cfg.use_phase else 1
            self.frontend = PCCQTFrontend(
                sr=cfg.sample_rate,
                fmin=cfg.fmin,
                n_bins=cfg.n_bins,
                bins_per_octave=cfg.bins_per_octave,
                hop_length=cfg.hop_length,
                use_phase=cfg.use_phase,
                prefer_nnaudio=cfg.prefer_nnaudio,
                n_frames=cfg.n_frames,
            )
            self.encoder = ResNet18Encoder(
                in_channels=in_channels,
                feat_dim=cfg.feat_dim,
                base_channels=cfg.resnet_base_channels,
                drop_rate=cfg.resnet_drop_rate,
            )
            self.signal_stream = nn.Sequential(self.frontend, self.encoder)

        # ---------- 语义流：WavLM Base+ ----------
        self.semantic_stream = WavLMStream(
            pretrained=cfg.wavlm_path,
            feat_dim=cfg.feat_dim,
            n_frozen_layers=cfg.n_frozen_layers,
            freeze_feature_encoder=cfg.freeze_feature_encoder,
            target_frames=cfg.target_frames,
            finetune=cfg.finetune_wavlm,
        )

        # ---------- 融合与分类 ----------
        self.fusion = GatedFusion(
            feat_dim=cfg.feat_dim,
            num_classes=cfg.num_classes,
            use_gating=cfg.use_gating,
            pooling=cfg.pooling,
            dropout=cfg.dropout,
            # 关闭信号流时只有 F_sem 一支进入分类头
            in_streams=2 if cfg.use_signal_stream else 1,
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        wav: torch.Tensor,
        return_gate: bool = False,
    ):
        """wav: (B, N) 16 kHz 定长波形 → logits (B, 2)。"""
        f_sem = self.semantic_stream(wav)          # (B, C, T)
        f_sig = None
        if self.signal_stream is not None:
            x = self.frontend(wav)                 # (B, C_in, K, T)
            f_sig = self.encoder(x)                # (B, C, T)
        return self.fusion(f_sig, f_sem, return_gate=return_gate)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_score(self, wav: torch.Tensor) -> torch.Tensor:
        """推理分数：越大越像 spoof。"""
        self.eval()
        logits = self.forward(wav)
        return logits[:, 1] - logits[:, 0]

    # ------------------------------------------------------------------ #
    def param_groups(
        self,
        lr_head: float,
        lr_ssl: float,
        ssl_decay: float = 0.85,
        weight_decay: float = 0.01,
    ) -> list[dict]:
        """论文 4.2 的分层学习率：

        - 随机初始化的 PC-CQT 分支与分类头 → 较大 lr（lr_head）
        - 预训练 WavLM 各层 → 更小 lr，且按层递减（lr_ssl · decay^i）
        """
        groups: list[dict] = []

        # 1) 信号流（前端 + ResNet-18）
        if self.signal_stream is not None:
            signal_params = list(self.frontend.parameters()) + list(self.encoder.parameters())
            signal_params = [p for p in signal_params if p.requires_grad]
            if signal_params:
                groups.append({
                    "params": signal_params,
                    "lr": lr_head,
                    "weight_decay": weight_decay,
                    "name": "signal_stream",
                })

        # 2) 分类头
        head_params = [p for p in self.fusion.parameters() if p.requires_grad]
        groups.append({
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay,
            "name": "classifier_head",
        })

        # 3) WavLM 逐层
        groups.append({"params": [p for p in self.semantic_stream.proj.parameters() if p.requires_grad],
                       "lr": lr_head, "weight_decay": weight_decay, "name": "semantic_projection"})
        groups.extend(
            self.semantic_stream.layerwise_param_groups(
                base_lr=lr_ssl, decay=ssl_decay, weight_decay=weight_decay
            )
        )
        return groups

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, int]:
        from ..utils import count_parameters
        info = {
            "total_params": count_parameters(self, trainable_only=False),
            "trainable_params": count_parameters(self, trainable_only=True),
        }
        if self.signal_stream is not None:
            info["signal_stream"] = count_parameters(self.signal_stream)
            info["frontend"] = count_parameters(self.frontend)
            info["encoder"] = count_parameters(self.encoder)
        info["semantic_stream"] = count_parameters(self.semantic_stream)
        info["fusion"] = count_parameters(self.fusion)
        return info


def build_opera_net(cfg: OperaNetConfig) -> OPeraNet:
    return OPeraNet(cfg)


def ablation_variant(name: str, **overrides) -> OperaNetConfig:
    """按 Table 2 的变体名构造配置。"""
    presets = {
        "wavlm_only": dict(use_signal_stream=False, use_phase=False, use_gating=False,
                           variant_name="WavLM Stream (Fine-tuned)"),
        "cqt_mag": dict(use_signal_stream=True, use_phase=False, use_gating=False,
                        variant_name="+ CQT (Magnitude-only)"),
        "pc_cqt": dict(use_signal_stream=True, use_phase=True, use_gating=False,
                       variant_name="+ PC-CQT (w/ Phase)"),
        "full": dict(use_signal_stream=True, use_phase=True, use_gating=True,
                     variant_name="OPERA-Net (Full w/ Gating)"),
    }
    if name not in presets:
        raise KeyError(f"未知变体 {name}，可选：{list(presets)}")
    cfg = OperaNetConfig(**presets[name])
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
