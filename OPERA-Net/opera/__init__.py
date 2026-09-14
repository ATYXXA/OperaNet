"""OPERA-Net 复现代码包。

模块导航
--------
- audio: 波形读取与 4 秒定长处理
- cqt: PC-CQT 前端（Phase-Consistent CQT）
- models: OPERA-Net 与三个 baseline（RawNet2 / AASIST / WavLM-Linear）
- data: CtrSVDD 与 SingFake 数据集
- metrics: EER 与 min-tDCF
- losses: 加权交叉熵
- trainer: 训练循环（AdamW + 分层学习率）
- utils: 通用工具
"""

__version__ = "0.1.0"
