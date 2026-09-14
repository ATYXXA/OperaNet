"""数据集构建入口：按配置名返回 (dataset, collate_fn)。"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from torch.utils.data import Dataset

from .ctrsvdd import (
    CtrSVDDDataset,
    Utterance,
    build_ctrsvdd_index,
    collate_fn as ctrsvdd_collate,
)
from .singfake import SingFakeDataset, collate_fn as singfake_collate


def build_dataset(kind: str, **kwargs) -> Dataset:
    """kind: 'ctrsvdd' | 'singfake'。"""
    if kind == "ctrsvdd":
        root = kwargs.pop("root")
        splits = kwargs.pop("splits", ("train",))
        check_exists = kwargs.pop("check_exists", False)
        index = build_ctrsvdd_index(root, splits=splits, check_exists=check_exists)
        return CtrSVDDDataset(index, **kwargs)

    if kind == "singfake":
        root = kwargs.pop("root")
        return SingFakeDataset(root, **kwargs)

    raise ValueError(f"未知数据集 {kind}")


def get_collate(kind: str):
    return {"ctrsvdd": ctrsvdd_collate, "singfake": singfake_collate}[kind]


__all__ = [
    "build_dataset",
    "get_collate",
    "CtrSVDDDataset",
    "SingFakeDataset",
    "build_ctrsvdd_index",
    "Utterance",
]
