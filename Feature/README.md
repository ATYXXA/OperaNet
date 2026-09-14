# Feature/ —— 论文三张图的生成代码

本目录**只保留**能生成论文三张图的脚本。其余出图脚本（通用频谱图、两联门控图变体、
本地对比图、单歌曲频谱图等）属于探索性中间产物，已在 `.gitignore` 第 5 节整目录忽略。

## 图 ↔ 脚本 ↔ 数据

| 论文图 | 脚本 | 数据来源 | 产物 |
|---|---|---|---|
| **Figure 1** Overall architecture of OPERA-Net | `draw_semantic_guided_gating_flowchart.py` | 无（架构示意图，全部为常量） | `figures/semantic_guided_gating_flowchart.{png,pdf,svg}` |
| **Figure 2** Visualization of the Semantic-Guided Gating mechanism | `create_sgg_effect_visualization.py` | 145 s "in-the-wild" 样本（`mock_data/audio/in_the_wild_sample.flac`） | `figures/semantic_guided_gating_effect_5s.{png,pdf,svg}` |
| **Figure 3** Radar chart of EER (%) across A09–A13 | `雷达图.py` | 论文 Table 1 的 A09–A13 EER（脚本内常量，与 `mock_data/paper_results/attackwise_radar.csv` 一致） | `figures/ctrsvdd_radar_eer.{png,pdf,svg}` |

论文正文的三处图注（`tmp/pdfs/opera_paper_extracted.txt`）：

- Figure 1：*"Overall architecture of OPERA-Net."* —— 对应脚本里的
  `Input Audio Waveform → PC-CQT Front-End / WavLM Encoder → … → Channel-wise Gate G
  → Element-wise Gating → Fusion → Classifier`，即论文 2.1–2.3 节的三大组件。
- Figure 2：*"…From top to bottom: (1) the raw PC-CQT spectrogram heavily masked by
  continuous instrumental harmonics; (2) the semantic guidance score derived from the
  WavLM stream; (3) the estimated 2D soft attention mask."* —— 与脚本
  `draw_figure()` 的前 3 个面板逐项对应：
  Raw PC-CQT Input → Semantic Guidance Score → Estimated Channel-wise Gate Map。
  脚本另外多画了第 4 个面板 *"After Semantic-Guided Gating"*（门控后的谱），
  用于直观看门控效果，属于论文三段描述的严格超集；原脚本即如此，未作改动。
- Figure 3：*"Radar chart illustrating the EER (%) breakdown across five distinct
  unseen attack types (A09-A13) on the CtrSVDD dataset."* —— 与 `雷达图.py` 的
  `labels = ["A09", …, "A13", "EER(A09-A13)"]` 一致。

## 一键复现

```bash
python Feature/make_paper_figures.py          # 三张图全部重新生成
python Feature/make_paper_figures.py --only 3 # 只跑 Figure 3
```

依赖：`matplotlib`（三张图）、`librosa` + `soundfile`（仅 Figure 2 需要读音频）。
Figure 2 需要一段**长于 141 秒**的音频，因为原脚本固定截取 `[136, 141] s`；
提交包用 `mock_data/generate_mock_data.py` 合成的 145 秒样本替代真实歌曲。

## 为什么运行器要 override 路径

这三个脚本是**原样保留**的论文出图代码，没有做任何修改，
以保持"论文图 = 该脚本产物"的可追溯性。`make_paper_figures.py` 通过
`importlib` 加载脚本、再把模块级常量（`OUT_DIR` / `OUTPUT_DIR`）和
`choose_audio()` 指向 `mock_data/`，因此：

- 原脚本逐字未改，任何一处改动都不会混进来；
- 不需要根目录那两个真实的孙燕姿 mp3（约 4–5 MB，已在 `.gitignore` 中忽略）；
- 输出统一落到 `figures/`，不与探索期产物混在一起。
