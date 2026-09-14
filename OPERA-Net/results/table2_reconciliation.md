# Table 2 对账

### Table 2（论文报告值）

| Architecture Variant | CtrSVDD EER (%) | CtrSVDD min-tDCF | SingFake EER (%) | SingFake min-tDCF |
|---|---|---|---|---|
| WavLM Stream (Fine-tuned) | 2.85 | 0.088 | 5.12 | 0.155 |
| + CQT (Magnitude-only) | 2.24 | 0.065 | 4.45 | 0.130 |
| + PC-CQT (w/ Phase) | 1.82 | 0.052 | 3.95 | 0.108 |
| OPERA-Net (Full w/ Gating) | 1.54 | 0.045 | 3.68 | 0.092 |

### Table 2（本仓库实测值）

| Architecture Variant | CtrSVDD EER (%) | CtrSVDD min-tDCF | SingFake EER (%) | SingFake min-tDCF |
|---|---|---|---|---|
| WavLM Stream (Fine-tuned) | — | — | — | — |
| + CQT (Magnitude-only) | — | — | — | — |
| + PC-CQT (w/ Phase) | — | — | — | — |
| OPERA-Net (Full w/ Gating) | — | — | — | — |

### Table 2（逐格对照：`论文值 → 实测值 (Δ)`）

| Architecture Variant | CtrSVDD EER (%) | CtrSVDD min-tDCF | SingFake EER (%) | SingFake min-tDCF |
|---|---|---|---|---|
| WavLM Stream (Fine-tuned) | 2.85 → 未跑 | 0.088 → 未跑 | 5.12 → 未跑 | 0.155 → 未跑 |
| + CQT (Magnitude-only) | 2.24 → 未跑 | 0.065 → 未跑 | 4.45 → 未跑 | 0.130 → 未跑 |
| + PC-CQT (w/ Phase) | 1.82 → 未跑 | 0.052 → 未跑 | 3.95 → 未跑 | 0.108 → 未跑 |
| OPERA-Net (Full w/ Gating) | 1.54 → 未跑 | 0.045 → 未跑 | 3.68 → 未跑 | 0.092 → 未跑 |

### 数值出处（论文报告值 ← 本仓库计算路径）

| 项目 | 论文口径 | 本仓库实现 |
|---|---|---|
| 训练语料 | CtrSVDD，32,312 bonafide + 188,486 deepfake（220,798 条） | `opera/data/ctrsvdd.py` 解析 `dataset/{train,dev,test}.txt`；规模由 `scripts/verify_paper_params.py` 实测核对 |
| 训练子集 | 官方 train + dev，128,029 条 | `configs/*.yaml` 的 `data.train.splits`，`paper.train_splits` 记录 |
| CtrSVDD 评估集 | 官方 test，92,769 条（论文 'CtrSVDD evaluation set'） | `data.eval.splits`，`evaluate.py --eval_kind ctrsvdd --eval_splits test` |
| SingFake 评估集 | T01 + T02，377 条曲目（in-the-wild，社交媒体采集） | `data.cross_eval.subsets`，`evaluate.py --eval_kind singfake --eval_subsets T01 T02 --track_level` |
| EER | FAR 与 FRR 相等处的错误率 | `opera/metrics/metrics.py::compute_eer`，与 ASVspoof 官方实现同构 |
| min-tDCF | ASVspoof 2019 归一化 tandem 检测代价，ASV 子系统取 ASVspoof 2019 LA | `opera/metrics/metrics.py::compute_min_tdcf`（version=2019）；需 ASV 分数，见 `docs/metrics_coverage.md` 缺口 1 |
| PC-CQT 前端 | 16 kHz / 4 s / f_min 32.7 Hz / 84 bins / 12 bins-per-octave / hop 320 | `opera/cqt.py`，输出 `X_pc ∈ R^{2×84×200}`；参数由 `scripts/verify_paper_params.py` 逐项核对 |
| 语义流 | WavLM Base+，冻结底部 6 层、微调顶部 6 层 | `opera/models/wavlm_stream.py::n_frozen_layers` |
| 门控融合 | G = σ(MLP(F_sem))，F̃_sig = F_sig ⊙ G，Concat → 两层 FC | `opera/models/fusion.py` |
| 损失函数 | 加权交叉熵（按类别频率反比，CtrSVDD 约 5.83 : 1） | `opera/losses.py::compute_class_weights` |
| 优化 | AdamW + 分层学习率衰减（WavLM 顶层 lr 远小于随机分支） | `configs/opera_net.yaml` 的 `lr_head=1e-4` / `lr_ssl=1e-5` / `ssl_lr_decay=0.85` |