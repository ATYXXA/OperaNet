"""AASIST baseline（论文 Table 1）。

参考 Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
Spectro-Temporal Graph Attention Networks", ICASSP 2022。

结构
----
    原始波形
      → SincConv 参数化前端            (B, 1, F, T)
      → 4 段 2-D 残差编码               (B, 64, F', T')
      → 自适应池化到 (16, 16) 网格
      → 把每个 (freq, time) 格子当作图节点（256 个节点，特征维 64）
      → 异构图注意力（GAT）+ 可微图池化 ×4
      → Max ⊕ Mean 图读出
      → 全连接分类头

Table 1 中标为 "Raw Waveform (Sinc)"，即与 RawNet2 同用 Sinc 前端。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rawnet2 import SincConv


class ResBlock(nn.Module):
    """2-D 残差块，保持形状不变。"""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class GraphAttentionLayer(nn.Module):
    """图注意力层（AASIST 的 heterogeneous graph attention 简化实现）。

        e_ij = LeakyReLU( a^T [W h_i ‖ W h_j] )
        α_ij = softmax_j( e_ij / temperature )
        h_i' = ELU( Σ_j α_ij · W h_j )
    """

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 2.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        # 把 a^T [Wh_i ‖ Wh_j] 拆成两个独立的线性投影：
        #   e_ij = a_l^T (Wh_i) + a_r^T (Wh_j)
        # 数值上等价于 Linear(2·out_dim, 1)，但避免显式构造 (B, N, N, D) 的中间张量，
        # 显存从 O(B·N²·D) 降到 O(B·N²)。
        self.attn_l = nn.Linear(out_dim, 1, bias=False)
        self.attn_r = nn.Linear(out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D_in)，adj: (B, N, N) → (B, N, D_out)。"""
        h = self.linear(x)                                   # (B, N, D_out)
        e = self.attn_l(h) + self.attn_r(h).transpose(1, 2)   # (B,N,1) + (B,1,N) → (B, N, N)
        e = self.leaky(e).masked_fill(adj <= 0, -1e9)
        alpha = torch.softmax(e / self.temperature, dim=-1)
        return F.elu(torch.bmm(alpha, h))


class GraphPool(nn.Module):
    """可微 top-k 图池化：用学得的节点得分保留前 k 个节点。"""

    def __init__(self, in_dim: int, ratio: float = 0.5) -> None:
        super().__init__()
        self.ratio = ratio
        self.score = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N, D)，adj: (B, N, N) → (B, k, D), (B, k, k)。"""
        b, n, d = x.shape
        k = max(1, int(round(n * self.ratio)))
        s = self.score(x).squeeze(-1)
        _, idx = torch.topk(s, k, dim=-1)                      # (B, k)

        idx_feat = idx.unsqueeze(-1).expand(b, k, d)
        x_pool = torch.gather(x, 1, idx_feat)                  # (B, k, D)

        idx_row = idx.unsqueeze(2).expand(b, k, n)
        adj_row = torch.gather(adj, 1, idx_row)                # (B, k, N)
        idx_col = idx.unsqueeze(1).expand(b, k, k)
        adj_pool = torch.gather(adj_row, 2, idx_col)           # (B, k, k)
        return x_pool, adj_pool


class AASIST(nn.Module):
    """AASIST 检测模型。

    Parameters
    ----------
    filts: [sinc_kernel, [in,out], [in,out], [in,out], [in,out]]
        官方默认 [70, [1,32], [32,32], [32,64], [64,64]]
    gat_dims: GAT 层维度
    pool_ratios / temperatures: 4 个 GAT 块后的池化比例与注意力温度
    grid: 自适应池化后的 (freq, time) 网格，决定图节点数 grid[0]*grid[1]
    """

    def __init__(
        self,
        num_classes: int = 2,
        filts: Optional[List] = None,
        gat_dims: List[int] = (64, 32),
        pool_ratios: List[float] = (0.5, 0.7, 0.5, 0.5),
        temperatures: List[float] = (2.0, 2.0, 100.0, 100.0),
        sample_rate: int = 16000,
        grid: Tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        filts = list(filts) if filts is not None else [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        sinc_kernel, b1, b2, b3, b4 = filts
        sinc_ch = 32  # SincConv 输出通道数（对应 b1[0] 的配套频率维）

        self.sinc = SincConv(sinc_ch, int(sinc_kernel), sample_rate)
        self.first_bn = nn.BatchNorm2d(1)
        self.act = nn.LeakyReLU(0.3)

        self.block1 = self._block(b1[0], b1[1])
        self.block2 = self._block(b2[0], b2[1])
        self.block3 = self._block(b3[0], b3[1])
        self.block4 = nn.Sequential(
            nn.Conv2d(b4[0], b4[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(b4[1]),
            nn.LeakyReLU(0.3),
        )
        self.grid = grid
        self.pool2d = nn.AdaptiveAvgPool2d(grid)
        node_dim = b4[1]

        # 图分支
        self.gat1 = GraphAttentionLayer(node_dim, gat_dims[0], temperatures[0])
        self.pool1 = GraphPool(gat_dims[0], pool_ratios[0])
        self.gat2 = GraphAttentionLayer(gat_dims[0], gat_dims[1], temperatures[1])
        self.pool2 = GraphPool(gat_dims[1], pool_ratios[1])
        self.gat3 = GraphAttentionLayer(gat_dims[1], gat_dims[1], temperatures[2])
        self.pool3 = GraphPool(gat_dims[1], pool_ratios[2])
        self.gat4 = GraphAttentionLayer(gat_dims[1], gat_dims[1], temperatures[3])
        self.pool4 = GraphPool(gat_dims[1], pool_ratios[3])

        self.out = nn.Linear(gat_dims[1] * 2, num_classes)   # Max ⊕ Mean

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.3),
            ResBlock(out_ch),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),   # 时间维下采样
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_adj(batch: int, n: int, device: torch.device) -> torch.Tensor:
        """全连接含自环的邻接矩阵 (B, N, N)。"""
        return torch.ones(batch, n, n, device=device)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, N) → logits (B, 2)。"""
        x = wav.unsqueeze(1)                          # (B, 1, N)
        x = self.sinc(x)                              # (B, C_sinc, T)
        x = x.unsqueeze(1)                            # (B, 1, C_sinc, T)：通道=1，频率=C_sinc
        x = self.act(self.first_bn(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)                            # (B, 64, F', T')
        x = self.pool2d(x)                            # (B, 64, grid_f, grid_t)

        b, d, f, t = x.shape
        nodes = x.permute(0, 2, 3, 1).reshape(b, f * t, d)   # (B, N=F*T, D)

        adj = self._build_adj(b, nodes.size(1), nodes.device)
        h = self.gat1(nodes, adj)
        h, adj = self.pool1(h, adj)
        h = self.gat2(h, adj)
        h, adj = self.pool2(h, adj)
        h = self.gat3(h, adj)
        h, adj = self.pool3(h, adj)
        h = self.gat4(h, adj)
        h, adj = self.pool4(h, adj)

        mx = h.max(dim=1).values                      # (B, D)
        mean = h.mean(dim=1)                          # (B, D)
        return self.out(torch.cat([mx, mean], dim=-1))

    @torch.no_grad()
    def predict_score(self, wav: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(wav)
        return logits[:, 1] - logits[:, 0]
