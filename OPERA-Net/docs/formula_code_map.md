# 论文公式 ↔ 代码对照

当前 PDF（Interspeech 2026 投稿版）共 5 个编号公式，另有多处维度声明与协议约定。
下表逐条给出代码位置与行号。

> 行号对应本仓库当前版本；改动代码后请用
> `grep -n "log1p\|wrap_to_pi\|sigmoid\|f_sig \* gate" opera/**/*.py` 复核。

---

## 一、编号公式

### 式 (1)：复 CQT 谱
$$X(k,n) = M(k,n)\cdot e^{j\phi(k,n)}$$

| 项 | 代码位置 | 行号 |
|---|---|---|
| 复谱 $X(k,n)$ 的生成 | `opera/cqt.py` → `NnAudioCQT.forward`（nnAudio `CQT1992v2`，`output_format="Complex"`） | `cqt.py:130-135` |
| 同上的诊断后端 | `opera/cqt.py` → `TorchCQT.forward`（复数 C-Q 卷积核 + `conv1d`） | `cqt.py:100-109` |
| 幅度 $M(k,n)$ | `torch.abs(x)` | `cqt.py:200` |
| 相位 $\phi(k,n)$ | `torch.angle(x)` | `cqt.py:204` |

**实现说明**：代码不显式构造 $M$ 与 $\phi$ 再合成 $X$，而是直接让 CQT 后端返回
`complex64` 张量 $(B,K,T)$，之后用 `abs` / `angle` 取幅度与相位。数学上等价。

### 正文声明：对数幅度谱
$$S_{mag} = \log(1 + M(k,n))$$

| 代码位置 | 行号 |
|---|---|
| `mag = torch.log1p(torch.abs(x))` | `cqt.py:200` |

`log1p(x)` 即 $\ln(1+x)$，在 $x \to 0$ 时数值更稳。

### 式 (2)：瞬时频率（相位导数）
$$S_{phase}(k,n) = \mathrm{wrap}\big(\phi(k,n) - \phi(k,n-1)\big)$$

| 步骤 | 代码位置 | 行号 |
|---|---|---|
| 相位差分 $\phi(k,n) - \phi(k,n-1)$ | `dphase = phase[..., 1:] - phase[..., :-1]` | `cqt.py:205` |
| $\mathrm{wrap}(\cdot) \to (-\pi, \pi]$ | `dphase = wrap_to_pi(dphase)` | `cqt.py:206` |
| `wrap_to_pi` 本体 | `opera/cqt.py::wrap_to_pi`（`remainder` 实现，$-\pi$ 映射为 $+\pi$） | `cqt.py:148-151` |
| 首帧补齐 | `F.pad(dphase, (1, 0), mode="constant", value=0.0)` | `cqt.py:208` |

> ⚠️ **论文未规定的实现约定**：$n=0$ 没有前驱帧，论文式 (2) 未说明如何处理。
> 代码把首帧补 0，使时间维与 $S_{mag}$ 保持 $T=200$ 对齐。

### 正文声明：PC-CQT 输入张量
$$X_{pc} \in \mathbb{R}^{2\times K\times N}$$

| 项 | 代码位置 | 行号 |
|---|---|---|
| 通道拼接（ch0 = $S_{mag}$，ch1 = $S_{phase}$） | `out = torch.stack(feats, dim=1)` | `cqt.py:211` |
| 通道数由 `use_phase` 决定（2 或 1） | `self.out_channels = 2 if use_phase else 1` | `cqt.py:188` |
| 幅度-only 消融分支 | `feats = [mag]`（不 append 相位） | `cqt.py:201-209` |

实测形状：`(B, 2, 84, 200)`，由 `scripts/verify_paper_params.py` 核对。

### 式 (3)：语义引导门控掩码
$$G = \sigma\big(\mathrm{MLP}(F_{sem})\big)$$

| 项 | 代码位置 | 行号 |
|---|---|---|
| $\sigma$ 与 MLP 的前向 | `SemanticGuidedGating.forward` → `return torch.sigmoid(self.mlp(f_sem))` | `fusion.py:44` |
| MLP 结构（两个 1×1 `Conv1d` + ReLU） | `SemanticGuidedGating.__init__` | `fusion.py:36-40` |
| $G \in [0,1]^{C\times T}$ | sigmoid 逐元素作用，形状与 `f_sem` 相同 | `fusion.py:43-44` |

两级 1×1 卷积等价于对每个时间步独立作用的通道 MLP，因此不改变时间分辨率。

### 式 (4)：信号特征重加权
$$\tilde{F}_{sig} = F_{sig} \odot G$$

| 代码位置 | 行号 |
|---|---|
| `gate = self.gating(f_sem)` → `f_sig_tilde = f_sig * gate` | `fusion.py:112-113` |
| 消融支路（不做门控，`use_gating=False`） | `gate = torch.ones_like(f_sig)` / `f_sig_tilde = f_sig` | `fusion.py:114-116` |

### 式 (5)：分类输出
$$y = \mathrm{Softmax}\big(\mathrm{Linear}(\mathrm{Concat}[F_{sem}, \tilde{F}_{sig}])\big)$$

| 步骤 | 代码位置 | 行号 |
|---|---|---|
| 拼接 $\mathrm{Concat}[F_{sem}, \tilde{F}_{sig}]$ | `fused = torch.cat([f_sem, f_sig_tilde], dim=1)` | `fusion.py:117` |
| `Linear` 两层全连接头 | `self.head = nn.Sequential(Linear, ReLU, Dropout, Linear)` | `fusion.py:90-95` |
| 时间维池化（mean+std） | `mean/std` → `torch.cat` | `fusion.py:122-125` |
| 推理分数 | `logits[:, 1] - logits[:, 0]`（越大越像 spoof） | `opera_net.py:144` |

> ⚠️ **论文写了 Softmax 但代码里没有显式 Softmax 层**：`self.head` 输出 **logits**，
> 交叉熵的 `Softmax` 隐含在 `F.cross_entropy` 内部（做 log-softmax + NLL）。
> 推理时取两类 logit 之差，与 Softmax 后取对数几率等价。若要与式 (5) 字面一致，
> 可在 `head` 后接 `nn.Softmax(dim=-1)`，但不影响训练与排序指标。

---

## 二、编码器与维度声明

| 论文声明 | 代码位置 | 行号 |
|---|---|---|
| $F_{sig} \in \mathbb{R}^{C\times T}$（ResNet-18 编码器） | `opera/models/resnet18.py::ResNet18Encoder` | `resnet18.py:87-110` |
| stem 处时间下采样 2×（200 → 100 帧） | `stride=(1, time_stride)`，`time_stride=2` | `resnet18.py:89` |
| 频率维压缩到 1 | `freq_strides=(1,2,2,2)` + 1×1 `Conv2d` + 自适应池化 | `resnet18.py:77, 107-110` |
| $F_{sem} \in \mathbb{R}^{C\times T}$（WavLM Base+） | `opera/models/wavlm_stream.py::WavLMStream.forward` | `wavlm_stream.py:90` |
| 末层隐状态投影到 $C$ | `self.proj = nn.Linear(hidden, feat_dim)` | `wavlm_stream.py:80` |
| 冻结底部 6 层、微调顶部 6 层 | `requires = i >= n_frozen_layers`（`n_frozen_layers=6`） | `wavlm_stream.py:46, 76` |
| $F_{sem}$ 时间下采样以对齐 $F_{sig}$ | `nn.AdaptiveAvgPool1d(target_frames)` | `wavlm_stream.py:81-82, 88` |
| 对齐帧数声明 | `target_frames: int = 100` | `opera_net.py:54`（传入点 `opera_net.py:109`） |
| 双流组装与前向 | `OPERA_Net.forward`：前端 → ResNet → 语义流 → 融合 | `opera_net.py:125-136` |

时间对齐链条：WavLM 的 CNN feature encoder 总 stride 为 320，16 kHz 下帧率 50 Hz，
4 s 输入 → 200 帧；PC-CQT 的 `hop_length=320` 同样给出 200 帧；ResNet-18 在 stem
把时间维下采样 2× → 100 帧；语义流再池化到 `target_frames=100`。
**两侧统一到 $T=100$ 后，式 (4) 的逐元素相乘才成立。**

---

## 三、训练侧

| 论文声明 | 代码位置 | 行号 |
|---|---|---|
| 加权交叉熵（处理类别不平衡） | `opera/losses.py::WeightedCrossEntropy`、`compute_class_weights` | `losses.py` |
| AdamW | `train.py` 优化器构造 | — |
| 分层学习率衰减（WavLM 顶层 lr 远小于随机分支） | `OPERA_Net.param_groups`（`lr_ssl · decay^i`） | `opera_net.py:147-180` |

---

## 四、论文 4.2 节前端超参

| 论文值 | 代码位置 | 行号 |
|---|---|---|
| 16 kHz | `DEFAULT_SR = 16000` | `cqt.py:41` |
| $f_{min} = 32.7$ Hz（C1） | `DEFAULT_FMIN = 32.7` | `cqt.py:42` |
| 84 个频率 bin | `DEFAULT_N_BINS = 84` | `cqt.py:43` |
| 每八度 12 bin | `DEFAULT_BINS_PER_OCTAVE = 12` | `cqt.py:44` |
| hop length 320 | `DEFAULT_HOP = 320` | `cqt.py:45` |
| 4 s 定长窗（裁剪/补零） | `num_samples = 64000`，`opera/audio.py` | 配置项 |

---

## 五、实验协议（非公式，但决定数值能否对上）

**pooled EER 的 cohort 定义**（`opera/metrics/eer.py::ctrsvdd_metrics`）

论文 3.1 节：*the primary ranking metric is the pooled EER across attacks A09 to A13*。

| 口径 | 定义 | B01 pooled | B02 pooled | 对应 |
|---|---|---|---|---|
| `bonafide_cohort="all"`（**默认，论文口径**） | 排除 attack A14 的 spoof，保留全部 bonafide（含 acesinger 的 2,845 条） | **11.37%** | **10.39%** | 论文 Table 1 ✓ |
| `bonafide_cohort="non_acesinger"` | 额外剔除 `source=acesinger` 的全部样本 | 12.03% | 11.16% | 官方 `results.csv` |

用 `scripts/reproduce_table1_rows.py` 实测：论文口径下重算值 11.3697 / 10.3851，
四舍五入到两位小数后与论文 Table 1 的 11.37 / 10.39 **完全一致**。

差异根源：ACESinger 既有 spoof（A14）也有 2,845 条 bonafide。把后者一并剔除会
改变 FAR 的分母（10,751 vs 13,596），从而系统性抬高 pooled EER 约 0.7–0.8 个百分点。

`bonafide_cohort="non_acesinger"` 仅在需要与官方 `results.csv` 对账时使用
（见 `scripts/verify_local_scores.py`）。

---

## 六、小结：三处「论文表述 vs 代码实现」的差异

| # | 论文 | 代码 | 影响 |
|---|---|---|---|
| 1 | 式 (5) 显式写出 `Softmax` | head 输出 logits，Softmax 在 `cross_entropy` 内部 | 无（数学等价；推理用 logit 差） |
| 2 | 式 (2) 未规定首帧 | 首帧相位差补 0 | 无（仅 1/200 帧，且与幅度通道对齐） |
| 3 | 式 (1) 显式写成 $M \cdot e^{j\phi}$ | 后端直接返回复数张量，再用 `abs`/`angle` 取用 | 无（同一对象的不同表示） |

这三处都属于实现细节层面的等价表述，不影响任何数值结果。
