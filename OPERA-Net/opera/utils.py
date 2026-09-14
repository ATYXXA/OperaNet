"""基础工具：随机种子、配置加载、计时、checkpoint 读写。"""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if str(path).endswith((".yml", ".yaml")):
            return yaml.safe_load(f)
        return json.load(f)


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if path.suffix in (".yml", ".yaml"):
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        else:
            json.dump(cfg, f, ensure_ascii=False, indent=2)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(state: Dict[str, Any], path: str | Path, keep_last: int = 3) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    # 只保留最近 keep_last 个 epoch checkpoint，best.pth 单独保存不受影响
    if keep_last > 0:
        ckpts = sorted(path.parent.glob("epoch_*.pth"),
                       key=lambda p: int(p.stem.split("_")[-1]))
        for old in ckpts[:-keep_last]:
            old.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    device: torch.device | None = None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device or "cpu")
    state = ckpt.get("model", ckpt)
    # 兼容多卡训练保存的 module. 前缀
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


@contextmanager
def timer(name: str = "", verbose: bool = True):
    t0 = time.time()
    yield
    if verbose:
        print(f"[timer] {name} {time.time() - t0:.2f}s")


class AverageMeter:
    """训练过程中的滑动平均统计。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def writable(path: str | Path) -> bool:
    """用于长路径/只读目录的兜底检查。"""
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".probe"
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def get_env_info() -> Dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(get_env_info())
