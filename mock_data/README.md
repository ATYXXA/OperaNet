# mock_data/ —— 与论文数据结构一致的模拟原始数据

论文依赖两个大数据集（CtrSVDD 220,798 条 flac、SingFake 1,481 条曲目），
不可能随代码提交。本目录提供一份**格式完全一致、统计结构按真实协议等比缩放**的
模拟数据，使提交包在没有真实数据集时依然能跑通代码并复现论文数值。

- 全部内容由脚本生成，不含任何真实音频或真实用户数据。
- 真实规模、分布与论文数值的来源在 `generate_mock_data.py` 顶部以字面量内联，
  因此生成过程不依赖已忽略的 `restored_data/`。

## 目录

```
mock_data/
├── generate_mock_data.py        生成全部模拟数据（唯一入口）
├── reproduce_from_mock.py       用 mock 数据复算论文 EER 并与论文对照
├── mock_config.json             生成参数与真实分布
├── mock_ctrsvdd_calibration.json   每个系统二分搜索到的分布展宽 s
├── mock_singfake_calibration.json  同上（SingFake）
├── mock_eer_report.json         reproduce_from_mock.py 的输出
│
├── ctrsvdd/
│   ├── train.txt               6 列协议清单（2,638 条）
│   ├── dev.txt                 6 列协议清单（1,407 条）
│   ├── test.txt                6 列协议清单（13,664 条，含 A14 与 acesinger）
│   ├── scores/                 6 个论文系统的官方 `filename,score` 提交格式
│   └── audio/{train,dev,test}_set/*.flac   每 split 8 条 4 s 示例音频
│
├── singfake/
│   ├── singfake.csv            7 列元数据（Set,Bonafide Or Spoof,Language,Singer,Title,Model,Url）
│   └── scores/                 5 个论文系统 × T01/T02/T03 track 级分数
│
├── audio/
│   └── in_the_wild_sample.flac 145 s 合成样本，供 Figure 2 使用
│
└── paper_results/
    ├── table1_ctrsvdd.json     论文 Table 1（CtrSVDD）
    ├── table2_singfake.json    论文 Table 2（SingFake）
    ├── table3_ablation.json    论文 Table 3（消融）
    ├── attackwise_radar.csv    论文 Figure 3 的数据源
    └── paper_params.json       论文明确给出的实现参数
```

## 格式与真实数据的对应

| 本项目文件 | 真实位置 | 格式 |
|---|---|---|
| `ctrsvdd/{train,dev,test}.txt` | `CtrSVDD2024_Baseline/dataset/` | UTF-8 无表头，空格分隔 **6 列**：`source_corpus singer_id utt_id - attack_id label` |
| `ctrsvdd/scores/*.csv` | `CtrSVDD2024_Baseline/analysis/baselines_csv/` | `filename,score`，**score 越高越像 bona fide**（详见下节） |
| `ctrsvdd/audio/*_set/<utt_id>.flac` | 同左 | 16 kHz 单声道 flac，4 s |
| `singfake/singfake.csv` | `singfake_dataset/singfake.csv` | **7 列**带表头 |
| `singfake/scores/*.csv` | 无对应真实文件 | track 级分数：`set,track_id,trial_index,score`，**越高越像 spoof** |
| `audio/in_the_wild_sample.flac` | 根目录的示例歌曲 mp3 | 16 kHz 单声道 |

解析器：`OPERA-Net/opera/data/ctrsvdd.py`（`parse_list_file`，严格校验 6 列）、
`OPERA-Net/opera/data/singfake.py`（`read_singfake_csv`）。

## 分数（`scores/`）到底是什么意思

`scores/` 里是**判决分数**，不是概率也不是标签：每行给一条样本（CtrSVDD 是 utterance 级、
SingFake 是 4 秒片段级）一个实数，表示"这条被判为某一类"的强弱。分数**没有物理量纲**，
只有**相对大小**有意义 —— EER 只依赖 bonafide 与 spoof 两个分数分布的相对位置，
所以可以直接把任意单调变换后的分数拿来算 EER。

| 文件 | 每行含义 | 方向（越大代表什么） | 参考范围 |
|---|---|---|---|
| `ctrsvdd/scores/B01_LFCC.csv` 等 6 个 | `filename,score`：该 utterance 的判决分数 | **越大越像 bona fide** | 约 −4 … +7 |
| `singfake/scores/*.csv` 5 个 | `set,track_id,trial_index,score`：该片段/曲目的判决分数 | **越大越像 spoof** | 约 −5 … +7 |

### ⚠️ 两个家族的方向是相反的，必须显式取负

**CtrSVDD 这 6 个文件模拟的是官方 baseline 提交格式**
（`CtrSVDD2024_Baseline/analysis/baselines_csv/B0x.csv`）。该文件的方向是
**score 越高越像 bonafide**（官方 baseline 训练时把标签翻转了，bonafide 记为 0）。

而本项目自己的 `OPERA-Net/opera/metrics/eer.py` 约定 **`0=bonafide, 1=spoof;
higher score = spoof`**。两者方向相反，所以送入指标前**必须取负**。

这不是猜的，是实测出来的（用项目内官方 B01/B02 分数直接算）：

| 处理 | B01 pooled | B02 pooled |
|---|---:|---:|
| 直接送入指标 | 88.63 % | 89.61 % |
| **取负后送入** | **11.37 %** | **10.39 %** |
| 论文 Table 1 | 11.37 | 10.39 |

`OPERA-Net/scripts/reproduce_table1_rows.py:141` 与 `mock_data/reproduce_from_mock.py`
都在送入指标前显式取负（前者的注释是 `# 官方文件 higher=bonafide`），
本包的 mock 分数复刻了这一方向，因此**复算流程与真实数据完全同构**。

SingFake 没有官方分数文件，`singfake/scores/*.csv` 是本包自定义格式，
直接采用项目内 `eer.py` 的方向（越高越像 spoof），无需取负。

### 分数的顺序不含任何信息

分位数构造天然是升序的，如果按清单顺序直接取用，文件里每个分组都会呈现
"相邻分数递增"的痕迹（实测会到 100 % 递增，而真实 B01 只有约 50 %）。
因此生成器在写入前**按分组打散**：bonafide 的分位数网格、每个攻击的分位数网格
各自做一次确定性置换，SingFake 的 bonafide/spoof 行也交错排列。

- 打散只改变"哪条样本拿到哪个分数"，不改变分数的多重集，**EER 完全不变**；
- 打散后分组内的相邻递增率约 50 %，与真实分数文件一致（实测 B01 46.5 k/92.8 k ≈ 50 %）。

### 这些分数是构造的，不是模型输出

分数由 `generate_mock_data.py` 用分位数标定生成（见下节），
**没有任何模型参与推理**。它们的用途是让 `scores/` 这一层的数据链路
（格式 → join → cohort 过滤 → EER）可跑通、且统计量与论文对齐，
不代表论文作者的真实预测。

## 规模缩放

| split | 真实条数 | mock 条数 | 因子 |
|---|---:|---:|---|
| train | 84,404 | 2,638 | 1/32 |
| dev | 43,625 | 1,407 | 1/32 |
| test | 92,769 | 13,664 | 1/6（A14 额外封顶到 2,400） |

每个攻击格的比例与真实协议一致；`test` 额外保持真实的 acesinger 结构
（5,690 = 2,845 bonafide + 2,845 A14 spoof，按同一因子缩放），
这样 `ctrsvdd_metrics(bonafide_cohort=...)` 的两种口径才会真正产生差异。
acesinger bonafide 占总 bonafide 的 **20.9 %**（mock 里 474/2,266，与真实一致，
因为分子分母同因子缩放）。

## bonafide 的两种来源：acesinger 结构

`ctrsvdd_metrics` 的 `bonafide_cohort` 参数只做一件事——是否把 `source=acesinger`
的 bonafide 一起剔除。这个开关只有在**两种来源的 bonafide 分数分布本来就不同**时
才有意义，所以 mock 必须复刻这个差异，否则换口径算出来的 EER 几乎不变。

真实数据的实测差异记录在 `OPERA-Net/opera/metrics/eer.py` 的模块文档串里
（用官方 baseline 分数在两个口径下各算一次）：

| 口径 | B01 pooled | B02 pooled |
|---|---:|---:|
| `all`（论文 Table 1，保留全部 bonafide） | 11.37 % | 10.39 % |
| `non_acesinger`（官方 `analysis/results.csv`，剔除 acesinger） | 12.03 % | 11.16 % |

即：acesinger 的 bonafide **系统性地更"容易"**，把它们剔除后剩下的 bonafide 更难，
FAR 变差、pooled EER 上升约 0.7–0.8 pp。

mock 的复刻方式：bonafide 分数**按来源分两条流水**——非 acesinger 取 `Q`，
acesinger 取 `−δ + Q`（在 `eer.py` 的"越高越像 spoof"方向下整体左移 δ）。
δ 是**数据属性**（与系统无关），用 **B01 的 12.03 % 单点拟合**得到
δ = 0.680；B02 的 11.16 % **不参与拟合**，作为独立预测接受检验（见复现结果一节）。

## 模拟分数的构造方法

不训练模型，而是用**分位数构造**合成可以直接算 EER 的分数序列，使统计量精确可控：

1. bonafide 分数取标准正态的等概率分位点，并**按来源分两条**：
   非 acesinger 用 `z_i = Q((i+0.5)/n_other)`，acesinger 用 `−δ + Q((j+0.5)/n_ace)`；
2. 攻击 a 的 spoof 分数取 `μ_a + s·Q((j+0.5)/n_a)`；
3. `μ_a` 由**二分求解**，使该攻击的 EER 精确等于论文的 per-attack EER。
   δ = 0 时退化为闭式解 `μ_a = (1+s)·Q⁻¹(1−EER_a)`；δ ≠ 0 时 FAR 变成
   两种 bonafide 的混合，闭式解不再成立，必须数值求解；
4. `s` 是全局"分布展宽"，只影响跨攻击合并后的 pooled EER。
   **δ = 0 时 pooled 关于 s 单调递减**，直接二分即可；**δ ≠ 0 时曲线呈 U 形**
   （FAR 有尾部地板，s 过大会让 spoof 左尾重新进入交叉区），所以先在粗网格上
   定位极小值点、取 `[0.05, 极小值点]` 这段单调下降段再二分；
5. 写入前**按分组打散**（分位数网格是升序的，不打散会留下"分数递增"的合成痕迹）；
6. CtrSVDD 的 6 个文件落盘时再**取负**，以复刻官方提交格式的方向（越高越像 bona fide）。

结果：6 个数字（pooled + A09–A13）同时对齐。`s` 与 δ 的取值记录在
`mock_ctrsvdd_calibration.json`（含 `_bonafide_structure` 一节）/
`mock_singfake_calibration.json`，各家族的分数方向记录在 `mock_config.json` 的
`score_direction` 字段。δ 的拟合只需纯数值计算、不涉及随机数，因此可确定性复现
（本机实测耗时约 14 s，大部分花在 δ 的嵌套二分上）。

SingFake 的 `Overall` 按 T01+T02+T03 并集复算（三个 split 等样本量）。

## 复现结果

```bash
python mock_data/generate_mock_data.py      # 生成（含 145 s 合成音频）
python mock_data/reproduce_from_mock.py     # 复算并对照论文
python Feature/make_paper_figures.py        # 重新生成论文三张图
```

`reproduce_from_mock.py` 调用的是 `OPERA-Net/opera/metrics/eer.py::ctrsvdd_metrics`，
与 `OPERA-Net/scripts/reproduce_table1_rows.py` **同一份口径代码**，
因此 cohort 规则（排除 A14、保留全部 bonafide）没有被重新解释。

实测结果（CtrSVDD，`bonafide_cohort="all"`，即论文 Table 1 口径）：

| system | 论文 pooled | mock 复算 | 偏差 | 分攻击（A09…A13） |
|---|---:|---:|---:|---|
| B01 (LFCC) | 11.37 | 11.35 | −0.02 | 5.35 2.90 5.84 29.49 3.66 |
| B02 (AASIST) | 10.39 | 10.41 | +0.02 | 6.71 0.98 3.61 26.82 0.98 |
| I2R-ASTAR | 2.22 | 2.21 | −0.01 | 0.66 0.49 2.47 4.54 0.61 |
| NBUMISL | 2.00 | 1.99 | −0.01 | 0.12 0.10 0.93 5.17 0.10 |
| Fosafer Speech | 1.65 | 1.63 | −0.02 | 0.22 0.39 0.05 4.18 0.05 |
| **OPERA-Net (Ours)** | **1.54** | **1.54** | **0.00** | 0.22 0.27 0.10 3.83 0.05 |

- pooled：6 个系统最大绝对偏差 **0.024 pp**；分攻击 30 个格最大绝对偏差 **0.033 pp**。
- SingFake：5 个系统 20 个格最大绝对偏差 **0.31 pp**（每 split 150 条 bonafide，
  FAR 分辨率 0.67 %）。
- 换 EER 定义（线性插值代替"最近交叉点"）：6 个系统偏差 ≤ **0.008 pp**，结论对定义不敏感。

换 cohort 的核对（与真实数据的实测口径值对照）：

| system | `all` | `non_acesinger` mock | 实测目标 | 目标差 | 角色 |
|---|---:|---:|---:|---:|---|
| B01 (LFCC) | 11.35 | **12.04** | 12.03 | +0.01 | 拟合点（δ 由它定） |
| B02 (AASIST) | 10.41 | **10.99** | 11.16 | −0.17 | 独立预测 |
| I2R-ASTAR | 2.21 | 2.40 | — | — | 无实测目标 |
| NBUMISL | 1.99 | 2.17 | — | — | 无实测目标 |
| Fosafer Speech | 1.63 | 1.79 | — | — | 无实测目标 |
| OPERA-Net (Ours) | 1.54 | 1.68 | — | — | 无实测目标 |

即：mock 不只对齐了论文写出来的数字，还复刻了**论文没写、但代码里记录了**的一条
数据性质（cohort 敏感性），且单参数 δ 模型对未参与拟合的 B02 预测误差仅 0.17 pp。

完整对照见 `mock_eer_report.json` 与 `reproduction_check.md`。

## 已知限制

1. **分辨率**：FAR 分辨率 = 1/n_bonafide。CtrSVDD 为 0.044%、SingFake 为 0.67%，
   因此论文中 <0.1% 的分攻击 EER（A11/A13）会落在分辨率下限附近。
2. **Table 3 不可复算**：论文只给了各消融变体在两个数据集上的 overall EER，
   没有给分攻击分解，无法从 mock 分数重建；脚本按原值列出并标注来源，
   **不伪造分解数据**。
3. **A14 分数是构造值**：A14 被主指标排除，mock 里给它的分数与 A13 同质，
   仅用于让 `keep = (attacks != "A14")` 这条协议过滤真正起作用。
4. **SingFake 评估集被扩大到 150/150**：真实 T01 只有 29 条 bonafide，
   FAR 分辨率 3.4% 无法分辨 3.12% 的 EER；扩大是为了让复算有意义，
   比例与真实不同，已在 `mock_config.json` 中标注。
5. **示例音频是合成信号**（谐波堆叠 + 和弦 + 打击），不是真实演唱，
   只保证尺寸、采样率、时长与代码路径一致。
6. **cohort 敏感性只用了一个自由度**：acesinger 的 bonafide 被建模为"整体平移 δ"，
   而真实差异可能还包含方差、难易分层等结构。δ 由 B01 单点拟合，
   对 B02 的预测误差 0.17 pp 说明这个单参数近似可用，但不应被当作精确重建。
