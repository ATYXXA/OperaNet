# OPERA-Net —— 论文复现的代码与数据包

本仓库是论文 **OPERA-Net: Octave-aware Phase-sensitive Enhanced Recognition Architecture
for Singing Voice Deepfake Detection** 的复现代码与配套数据的提交包。

论文原文见根目录 `OPERA-Net Octave-aware Phase-sensitive Enhanced Recognition
Architecture for Singing Voice Deepfake Detection.pdf`。

## 这个包里有什么

| 目录 | 内容 |
|---|---|
| `OPERA-Net/` | 论文方法的完整实现：PC-CQT 前端、WavLM 语义流、语义引导门控、训练/评测、配置、测试、文档 |
| `Feature/` | **只保留**论文三张图的生成脚本 + 一键复现运行器 + 产物 `Feature/figures/` |
| `mock_data/` | 与论文数据结构一致的**模拟原始数据**及生成/复算脚本（真实数据集不随包提交） |
| `.gitignore` | 明确"哪些入包、哪些不入包"的白名单策略 |

## 论文三张图 ↔ 生成脚本

| 图 | 脚本 | 数据来源 |
|---|---|---|
| Figure 1 整体架构 | `Feature/draw_semantic_guided_gating_flowchart.py` | 无（示意图） |
| Figure 2 语义引导门控可视化 | `Feature/create_sgg_effect_visualization.py` | `mock_data/audio/in_the_wild_sample.flac` |
| Figure 3 分攻击 EER 雷达图 | `Feature/雷达图.py` | 论文 Table 1 的 A09–A13 EER |

细节见 `Feature/README.md`。

## 论文三张表 ↔ 数据

| 表 | 内容 | 数据 |
|---|---|---|
| Table 1 | CtrSVDD pooled + 分攻击 EER（6 个系统） | `mock_data/ctrsvdd/` + `paper_results/table1_ctrsvdd.json` |
| Table 2 | SingFake T01/T02/T03/Overall（5 个系统） | `mock_data/singfake/` + `paper_results/table2_singfake.json` |
| Table 3 | 消融（CtrSVDD / SingFake，4 个变体） | `paper_results/table3_ablation.json`（论文只给 overall，无分解，不可复算） |

## 一键复现

```bash
# 1. 生成模拟原始数据（含 Figure 2 需要的 145 s 合成样本）
python mock_data/generate_mock_data.py

# 2. 用 mock 数据复算论文 EER，并与论文逐格对照
python mock_data/reproduce_from_mock.py

# 3. 重新生成论文的三张图 -> Feature/figures/
python Feature/make_paper_figures.py
```

依赖：`numpy`（mock 数据）、`matplotlib`（三张图）、`librosa` + `soundfile`（仅 Figure 2）。
本项目验收所用环境为 `.venv-opera-audit/`（不入包）。

**第 2 步的实测结果**：以论文 Table 1 口径（`bonafide_cohort="all"`，排除 A14 spoof、
保留全部 bonafide）复算，CtrSVDD pooled EER 与论文 6 个系统的偏差 **≤0.03 pp**
（B01 11.35 vs 11.37、B02 10.41 vs 10.39、OPERA-Net 1.54 vs 1.54），
分攻击 EER（30 个格）偏差 ≤0.04 pp，SingFake（20 个格）≤0.31 pp。

第 2 步还额外做了一项论文没写、但代码里记录过的核对：把 cohort 换成
`bonafide_cohort="non_acesinger"`（官方 `analysis/results.csv` 口径），
mock 复算 B01 = 12.04 %（真实实测目标 12.03 %）、B02 = 10.99 %（目标 11.16 %）。
其中 δ 只用 B01 拟合，B02 是**独立预测**，误差 0.17 pp。
完整对照见 `mock_data/mock_eer_report.json`。

## 关于模拟数据

论文使用的两个数据集（CtrSVDD 220,798 条 flac、SingFake 1,481 条曲目）体积过大，
不适合随代码提交，且 `CtrSVDD2024_Baseline/` 与 `SingFake-main/` 属第三方仓库。
因此 `mock_data/` 提供一份：

- **格式逐字段一致**（CtrSVDD 6 列协议清单、官方 `filename,score` 评分、SingFake 7 列元数据），
  现有解析器 `opera/data/{ctrsvdd,singfake}.py` 可直接读；
- **统计结构按真实协议等比缩放**（train/dev 1/32、test 1/6；攻击分布与 source 分布同比例，
  并保留 acesinger = 2,845 bonafide + 2,845 A14 spoof 的真实结构，且复刻了
  acesinger bonafide 系统性更"容易"这一点——两种 cohort 口径因此才会给出不同 EER）；
- **数值可对账**：分数用分位数标定构造，使复算出的 EER 精确命中论文数值，
  并额外对齐论文未给、但 `opera/metrics/eer.py` 记录过的 cohort 敏感性。

方法与限制详见 `mock_data/README.md`。

## 未随包提交的内容

`.gitignore` 有意排除了以下内容，理由是体积、第三方归属或属探索期残留：

- **真实数据集与权重**：`CtrSVDD2024_Baseline/`、`singfake_dataset/`、`SingFake-main/`、
  `XWSB_for_SVDD2024/`、`aasist/`、`wavlm-base-plus/`、`WavLabLM-MS-40k/`
- **冗余的数据预处理代码**：`dataset_process/`（SingFake 下载/切片/校验脚本等）
- **非论文图的生成代码与中间产物**：`Feature/` 下的通用频谱图、两联门控图变体、
  本地对比图、单歌曲频谱图，以及各种 png/svg/pdf/npy 中间文件
- **真实歌曲示例**：根目录两个孙燕姿 mp3（Figure 2 改用 `mock_data/` 的合成样本）
- 虚拟环境、缓存、临时目录、旧版论文模板 `tmp_template5.txt`

`restored_data/` 是由真实数据反推的恢复产物（含真实清单与 jsonl），同样不入包；
其中被引用的统计量已作为字面量内联进 `mock_data/generate_mock_data.py`，
因此生成脚本不依赖该目录。
