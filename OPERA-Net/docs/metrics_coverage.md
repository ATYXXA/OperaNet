# 参数、指标与复现边界

论文对应当前 PDF 第 2–4 页；逐字段依据见 [paper_reference.json](paper_reference.json)。

| 内容 | 已有依据与实现 | 不能认作原始实验设置的部分 |
|---|---|---|
| 16 kHz / 4 s | 论文明确；audio.py | 随机/中心裁剪、尾段处理、VAD 与预处理版本未给 |
| nnAudio CQT | 84 bins，12/octave，32.7 Hz，hop 320 | CQT1992v2、center/padding/normalization 未给 |
| PC-CQT | log(1+abs(X)) 和 wrapped 帧间相位差 | 200 帧截断和首帧零相位是选择 |
| WavLM Base+ | 下 6 层冻结，上 6 层微调 | CNN、projection 冻结及 dropout/layerdrop 细则未给 |
| 时间对齐 | 原生 CQT 201 帧 / WavLM 199 帧，实际检查后对齐 | C=256、T=100 是选择，不能称为论文数值 |
| ResNet-18 | [2,2,2,2] 残差结构，已修复时间维误下采样 | 具体宽度、stride 与池化论文未给 |
| 门控、分类头 | sigmoid(MLP(F_sem))、逐元素乘、concat 和两层 FC | 隐层大小、池化和 dropout 未给 |
| 优化 | AdamW、加权 CE、layerwise decay | lr、衰减方向/系数、epoch、batch、seed、scheduler 未给 |
| CtrSVDD | train/dev 59/55 singer_id 与 PDF 一致 | PDF eval 48，与本地全量 51/主协议 12 不一致，待溯源 |
| 主 EER | 合并 A09–A13 并排除 A14 spoof；**论文 Table 1 口径保留全部 bonafide**（含 acesinger），实测 B01/B02 = 11.37/10.39 命中论文；官方 `analysis/results.csv` 口径额外剔除 source=acesinger，同分数下为 12.03/11.16 | 原始 trial list 与原始预测缺失 |
| EER 计算 | 两类检查、完整端点、同分组处理，真实 B01/B02 回归通过 | 极端同分时与上游逐条排序的门限约定有差别；不自动翻转分数来降低 EER |
| SingFake | T01/T02/T03 按独立子集索引，缺失即报错 | T03 缺失，segment/track 与 Overall 规则未给 |
| min-tDCF | 当前路径不计算，旧模块仅为历史保留 | 当前 PDF 未报告该指标，不借其它 ASV 分数补造 |
| 门控图 | 可从模型返回 gate | 原有 Feature 图为代理门控，不能证明已经得到训练模型 gate |
| 论文主结果 | 精确转录、单独列为参考 | 无 OPERA-Net 训练 checkpoint/逐条分数，实测列留空 |

数值与张量验证输出：[model_validation.json](../../restored_data/current_pdf/model_validation.json)。该检查读取两条真实训练音频，使用本地 WavLM 权重，完成前向、反向与一次更新，不代表完整训练。
