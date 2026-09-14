# OPERA-Net：当前 PDF 的可核验复现

依据根目录的 5 页匿名 Interspeech 2026 PDF，实现 PC-CQT + WavLM Base+ + Semantic-Guided Gating。此仓库是复现代码，缺少作者原始 checkpoint、训练日志和精确 SingFake trial list，不能宣称已复现论文性能。

配套的模拟原始数据与生成/复算脚本见 [`../mock_data/`](../mock_data/README.md)（格式逐字段一致、统计按真实协议等比缩放，复算出的 CtrSVDD pooled EER 与论文偏差 ≤0.03 pp，并复刻了论文未写、但 `opera/metrics/eer.py` 记录过的 cohort 敏感性）。论文三张图的生成脚本与一键运行器见 [`../Feature/`](../Feature/README.md)。

> 本项目验收期间曾用 `restored_data/` 存放从真实数据反推的恢复产物（含真实清单与 jsonl）。
> 该目录体积大且含真实数据，**不随本包提交**（见根目录 `.gitignore`）；其中被引用的
> 统计量已内联进 `../mock_data/generate_mock_data.py`，因此不受影响。

## 版本与结果

当前 PDF 是 Table 1（CtrSVDD 六系统）、Table 2（SingFake T01/T02/T03）、Table 3（消融），只报告 EER。OPERA-Net 报告值为 CtrSVDD pooled 1.54%，SingFake Overall 4.72%。旧 `tmp_template5.txt` 的 3.68% 和 min-tDCF 不是当前版本。

[paper_reference.json](docs/paper_reference.json) 保存转录值及未说明参数。[三表对照](results/current_pdf/table1_reconciliation.md) 使用“论文报告 / 本地实测”，没有真实评分文件时标注未测。RawNet2、WavLM-Linear 属于保留的旧版参考实现，不能替代当前表中的挑战赛系统、MERT 或 SingGraph。

论文 5 个编号公式与维度声明在代码中的逐条位置（含行号与三处等价表述差异）见 [公式↔代码对照](docs/formula_code_map.md)。

## 当前数据协议

| 用途 | 配置 | 本地样本 |
|---|---|---:|
| 拟合 | train | 84,404 |
| 选择 checkpoint | dev | 43,625 |
| 最终 CtrSVDD | test，排除 A14 和 acesinger | 64,734 |
| SingFake | T01 / T02 / T03 | 90 / 197 / 0 个本地文件 |

train/dev 用法、SingFake 的片段级 pooled 计算以及具体超参数属于声明的复现选择。论文未充分给出这些细节。原始 test 共 92,769 条，其余攻击指标作为诊断单列，不能混入主表。T03 缺失会报错，不会用 T01/T02 冒充完整 Overall。

训练数据与标签来自现有真实协议。恢复的 JSONL 清单在 `../restored_data/current_pdf/`，音频保留原位置，不复制到新目录。

## 环境与验证

本次建立的 CPU 验证环境为 `../.venv-opera-audit/`。可在新环境安装 `requirements.txt`；本次实际安装版本记录在 `../restored_data/current_pdf/validation_environment.txt`，这份记录不是作者原始环境。

从本目录执行：

```powershell
..\.venv-opera-audit\Scripts\python.exe -m pytest tests\test_protocol.py -q
..\.venv-opera-audit\Scripts\python.exe scripts\verify_paper_params.py --model
..\.venv-opera-audit\Scripts\python.exe scripts\verify_local_scores.py
..\.venv-opera-audit\Scripts\python.exe scripts\build_paper_tables.py --all
..\.venv-opera-audit\Scripts\python.exe scripts\run_experiments.py --table 3 --plan
```

已验证默认模型使用真实音频完成前向、反向、优化更新，所有可训练参数都进入优化器。实测 CQT 原生 201 帧，本实现截为 200 帧；WavLM 原生 199 帧；两路最终都为 `[B,256,100]`。这些不是由相同 hop 自动保证的。

模型检查不输出实验 EER，不保存可冒充训练完成的检测器。真实 B01/B02 分数重新计算得到 12.02547198% / 11.16041050%，与本地原保存结果一致，与当前 PDF 参考值不同。

## 训练与评估

训练前编辑 YAML 路径与设备。默认设备为 cuda；CPU 检查使用上述专用脚本。完整训练需要自行配置足够的算力。

```powershell
python train.py --config configs/opera_net.yaml --out_dir exp/opera_net
python evaluate.py --config configs/opera_net.yaml --ckpt exp/opera_net/best.pth --eval_kind ctrsvdd --eval_root ../CtrSVDD2024_Baseline/dataset --eval_splits test --by_attack --out exp/opera_net/ctrsvdd_eval
python evaluate.py --config configs/opera_net.yaml --ckpt exp/opera_net/best.pth --eval_kind singfake --eval_root ../singfake_dataset --eval_subsets T01 T02 T03 --out exp/opera_net/singfake_eval
```

最后一条必须先补齐 T03。论文没有给 codec/码率，恢复包的 `t03_parent_candidates.jsonl` 只是候选父文件清单。上游生成脚本在 `../SingFake-main/dataset/classify_scripts/simulate_codec.py`；不能把任意选择的编码设置描述为作者设置。

`evaluate.py` 保存逐条 `scores.jsonl` 与 `metrics.json`，后者记录论文、配置、checkpoint 和评分文件指纹。`--track_level` 是另一明确选择，会额外保存片段分数；当前 PDF 未给出可判定这两种粒度的充分信息。

## 论文参数与实现选择

论文明确给出：16 kHz、4 s、84 CQT bins、12 bins/octave、fmin=32.7 Hz、hop=320、ResNet-18、WavLM Base+ 下 6 层冻结/上 6 层微调、wrapped 相位差、语义门控、双层 FC、加权交叉熵与 AdamW。

本实现另选：C=256、T=100、CQT 截为 200 帧、首相位补零、统计池化、seed=42、10 epochs、batch size、学习率、衰减率、调度器、CNN 冻结和窗口边界。详见 [参数及证据说明](docs/metrics_coverage.md)。

默认必须安装 nnAudio；显式设 `prefer_nnaudio: false` 才使用未经等价性验证的 TorchCQT 诊断后端。门控素材 `Feature/` 中有代理示意图，不能把它当成训练模型输出。

旧配置与文档已备份至 `../restored_data/current_pdf/code_before_alignment.zip`；`results/` 根目录旧表和 `exp/param_check.json` 属于历史材料，当前输出使用 `results/current_pdf/`。
