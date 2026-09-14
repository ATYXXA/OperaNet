# 当前 PDF 的结果依据

唯一当前参考源是根目录同名 5 页 PDF。完整机器可读转录见 [paper_reference.json](paper_reference.json)，PDF 指纹见 [source_hashes.json](../../restored_data/current_pdf/source_hashes.json)。

- Table 1，第 3 页：CtrSVDD 六系统，A09–A13 pooled 及分攻击 EER。
- Table 2，第 4 页：SingFake 五系统，T01、T02、T03 和 Overall。
- Table 3，第 4 页：四变体消融，CtrSVDD / SingFake EER。

旧版 3.68%、6.42% 和 min-tDCF 不属于当前 PDF。当前主模型的 SingFake Overall 为 4.72%。不得把它解释为由旧数据重新计算得到的结果。

生成对照：`python scripts/build_paper_tables.py --all`。输出在 `results/current_pdf/`。未测项为 null/未测，不从论文值回填。测量项须具有当前 PDF 指纹、可验证的 scores.jsonl 及对应协议；SingFake Overall 要求 T01/T02/T03 完整存在。

表 1 的外部挑战赛系统、表 2 的 MERT/SingGraph 等没有在本项目实现为完整可复现系统。对应 config 为 null，不用其他 baseline 冒名填充。只支持 OPERA-Net 和三种消融的实验计划。

本地 B01/B02 分数重算分别为 12.0254719792% / 11.1604104954%，与保存的本地结果一致，与当前 PDF 中 11.37% / 10.39% 不同。详见 [完整审计](../../restored_data/current_pdf/README.md)。差异来源尚不确定。
