"""语义流：WavLM Base+。

论文 3.2 节
-----------
- 使用 WavLM Base+（94k 小时预训练）
- 冻结底部 6 层 transformer，微调顶部 6 层（防止灾难性遗忘）
- 最后一层隐状态经线性层投影到特征维 C，得到 F_sem ∈ R^{C×T}
- 对时间维下采样，使其与 F_sig 的时间分辨率对齐

时间分辨率
----------
WavLM 的 CNN feature encoder 总 stride 为 320，因此 16 kHz 下帧率为 50 Hz：
本地卷积配置的 4 s 输入输出 199 帧；目标 T=100 时使用自适应平均池化。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:  # transformers 为可选依赖，仅在需要语义流时才必须
    from transformers import WavLMModel
except ImportError:  # pragma: no cover
    WavLMModel = None


class WavLMStream(nn.Module):
    """WavLM Base+ 语义编码器。

    Parameters
    ----------
    pretrained: 本地权重目录（如 C:/Code/SVDD/wavlm-base-plus）或 HF model id。
    feat_dim: 投影后的特征维 C。
    n_frozen_layers: 冻结的底部 transformer 层数（论文为 6）。
    freeze_feature_encoder: 是否冻结 CNN feature encoder（推荐 True）。
    target_frames: 对齐后的时间帧数；None 表示不做时间下采样（保持 200）。
    finetune: False 时整个 WavLM 冻结（用于 WavLM-Linear baseline）。
    """

    def __init__(
        self,
        pretrained: str = "microsoft/wavlm-base-plus",
        feat_dim: int = 256,
        n_frozen_layers: int = 6,
        freeze_feature_encoder: bool = True,
        target_frames: Optional[int] = 100,
        finetune: bool = True,
    ) -> None:
        super().__init__()
        if WavLMModel is None:
            raise ImportError("需要 transformers：pip install transformers")

        self.model = WavLMModel.from_pretrained(pretrained)
        hidden = self.model.config.hidden_size              # Base+ : 768
        self.hidden_size = hidden
        self.target_frames = target_frames
        self.finetune = finetune
        self.freeze_feature_encoder = freeze_feature_encoder
        self.n_frozen_layers = n_frozen_layers
        self.n_layers = self.model.config.num_hidden_layers  # Base+ : 12

        if freeze_feature_encoder:
            for p in self.model.feature_extractor.parameters():
                p.requires_grad = False
            for p in self.model.feature_projection.parameters():
                p.requires_grad = False

        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            # 底部 n_frozen_layers 层冻结，顶部其余层微调
            for i, layer in enumerate(self.model.encoder.layers):
                requires = i >= n_frozen_layers
                for p in layer.parameters():
                    p.requires_grad = requires

        self.proj = nn.Linear(hidden, feat_dim)
        if target_frames is not None:
            self.downsample = nn.AdaptiveAvgPool1d(target_frames)
        else:
            self.downsample = nn.Identity()

    @property
    def downstream_frames(self) -> int:
        return self.target_frames if self.target_frames is not None else 200

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, N) 16 kHz → F_sem: (B, feat_dim, T)。"""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        out = self.model(wav).last_hidden_state              # (B, T', 768)
        out = self.proj(out)                                 # (B, T', C)
        out = out.transpose(1, 2)                            # (B, C, T')
        return self.downsample(out)                          # (B, C, T)

    def layerwise_param_groups(
        self,
        base_lr: float,
        decay: float = 0.85,
        weight_decay: float = 0.01,
    ) -> list[dict]:
        """按层构造学习率递减的参数组。

        当前 PDF 指定 layerwise decay，但未给方向或系数。本复现选择
        lr_i = base_lr * decay^(num_layers-1-i)，越靠底层越小。
        """
        groups = []
        for i, layer in enumerate(self.model.encoder.layers):
            params = [p for p in layer.parameters() if p.requires_grad]
            if not params:
                continue
            groups.append({
                "params": params,
                "lr": base_lr * (decay ** (self.n_layers - 1 - i)),
                "weight_decay": weight_decay,
                "name": f"wavlm.layer{i}",
            })
        included = {id(p) for g in groups for p in g["params"]}
        rest = [p for p in self.model.parameters() if p.requires_grad and id(p) not in included]
        if rest:
            groups.append({"params": rest, "lr": base_lr * decay ** self.n_layers,
                           "weight_decay": weight_decay, "name": "wavlm.other"})
        return groups

    def train(self, mode: bool = True):
        """冻结的模块始终保持在 eval 态（避免其归一化统计量被更新）。"""
        super().train(mode)
        if not self.finetune:
            self.model.eval()
        elif mode:
            self.model.feature_extractor.eval()
            if self.freeze_feature_encoder:
                self.model.feature_projection.eval()
            for layer in self.model.encoder.layers[:self.n_frozen_layers]:
                layer.eval()
        return self
