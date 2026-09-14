"""OPERA-Net 训练入口。

用法示例
--------
# OPERA-Net（当前 PDF Table 1/2 主结果 / Table 3 的 Full 变体）
python train.py --config configs/opera_net.yaml --out_dir exp/opera_net

# Table 3 消融：只跑 WavLM 流
python train.py --config configs/ablation_wavlm_only.yaml --out_dir exp/ab_wavlm

# RawNet2 baseline
python train.py --config configs/baseline_rawnet2.yaml --out_dir exp/rawnet2

数据集默认使用 CtrSVDD 官方训练子集 train.txt（84,404 条）。
dev 用于选择 checkpoint，test 只用于最终评估。这是声明的复现协议；
当前 PDF 没有提供可唯一还原训练样本划分的 trial list。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from opera.data import build_dataset, get_collate                 # noqa: E402
from opera.losses import compute_class_weights, build_loss        # noqa: E402
from opera.models import build_model                              # noqa: E402
from opera.trainer import Trainer                                 # noqa: E402
from opera.utils import (count_parameters, get_env_info,          # noqa: E402
                         load_config, save_config, set_seed)


# ---------------------------------------------------------------------- #
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.05,
):
    """带 warmup 的余弦学习率调度（避免从零开始就大步长更新 WavLM）。"""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """按论文 4.2 构造分层学习率的 AdamW。"""
    opt = cfg["optim"]
    lr_head = float(opt["lr_head"])
    lr_ssl = float(opt.get("lr_ssl", lr_head * 0.1))
    decay = float(opt.get("ssl_lr_decay", 0.85))
    wd = float(opt.get("weight_decay", 0.01))

    if hasattr(model, "param_groups"):
        groups = model.param_groups(lr_head=lr_head, lr_ssl=lr_ssl,
                                   ssl_decay=decay, weight_decay=wd)
        print("[optim] 使用分层学习率：")
        for g in groups:
            print(f"    {g.get('name', 'group'):<20} lr={g['lr']:.3e}")
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": lr_head, "weight_decay": wd, "name": "all"}]
        print(f"[optim] 单一学习率 lr={lr_head:.3e}")

    return torch.optim.AdamW(groups, betas=tuple(opt.get("betas", (0.9, 0.999))),
                             eps=float(opt.get("eps", 1e-8)))


def build_loaders(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """构造训练 / 评估 DataLoader。"""
    dcfg = cfg["data"]
    train_splits = set(dcfg["train"].get("splits", []))
    eval_splits = set(dcfg.get("eval", {}).get("splits", []))
    if "test" in train_splits or "test" in eval_splits:
        raise ValueError("test is reserved for final evaluation; use train for fitting and dev for model selection")
    if train_splits & eval_splits:
        raise ValueError("Training and validation splits overlap")
    train_ds = build_dataset(**dcfg["train"])
    eval_ds = build_dataset(**dcfg["eval"]) if "eval" in dcfg else None

    num_workers = int(cfg.get("num_workers", 4))
    pin = bool(cfg.get("pin_memory", True))
    bs_train = int(cfg["train_batch_size"])
    bs_eval = int(cfg.get("eval_batch_size", bs_train))

    loaders = {
        "train": DataLoader(train_ds, batch_size=bs_train, shuffle=True,
                            num_workers=num_workers, collate_fn=get_collate(dcfg["train"]["kind"]),
                            pin_memory=pin, drop_last=True,
                            persistent_workers=num_workers > 0),
    }
    if eval_ds is not None:
        loaders["eval"] = DataLoader(
            eval_ds, batch_size=bs_eval, shuffle=False,
            num_workers=num_workers, collate_fn=get_collate(dcfg["eval"]["kind"]),
            pin_memory=pin, drop_last=False, persistent_workers=num_workers > 0)
    return {"train": loaders["train"], "eval": loaders.get("eval"),
            "n_train": len(train_ds), "n_eval": len(eval_ds) if eval_ds is not None else 0,
            # 类别权重只依赖训练集的标签分布，提前取出以免被 Subset 包装后取不到
            "labels": train_ds.labels()}


# ---------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser("OPERA-Net 训练")
    ap.add_argument("--config", type=str, required=True, help="yaml 配置文件")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--model", type=str, default=None, help="覆盖配置中的模型名")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr_head", type=float, default=None)
    ap.add_argument("--lr_ssl", type=float, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--train_splits", nargs="+", default=None,
                    help="CtrSVDD 训练划分，如 train dev")
    ap.add_argument("--eval_split", type=str, default=None)
    ap.add_argument("--wavlm_path", type=str, default=None)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--amp", action="store_true", help="开启混合精度")
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_eval_samples", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None, help="从 checkpoint 继续训练")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # ---- 命令行覆盖 ----
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["train_batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.device:
        cfg["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.lr_head:
        cfg["optim"]["lr_head"] = args.lr_head
    if args.lr_ssl:
        cfg["optim"]["lr_ssl"] = args.lr_ssl
    if args.amp:
        cfg["amp"] = True
    if args.eval_every:
        cfg["eval_every"] = args.eval_every
    if args.train_splits:
        cfg["data"]["train"]["splits"] = args.train_splits
    if args.eval_split:
        cfg["data"]["eval"]["splits"] = [args.eval_split]
    if args.wavlm_path and cfg["model"]["name"] in ("opera_net", "wavlm_linear"):
        cfg["model"].setdefault("params", {})["wavlm_path"] = args.wavlm_path
        cfg["model"].setdefault("params", {})["pretrained"] = args.wavlm_path
    if args.data_root:
        for key in ("train", "eval"):
            if key in cfg["data"]:
                cfg["data"][key]["root"] = args.data_root

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = Path(cfg.get("out_dir", "exp/default"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    print("[env]", json.dumps(get_env_info(), ensure_ascii=False))
    print(f"[out] {out_dir}")

    # ---- 数据 ----
    data = build_loaders(cfg)
    print(f"[data] train={data['n_train']}  eval={data['n_eval']}")

    if args.max_train_samples:
        from torch.utils.data import Subset
        base = data["train"].dataset
        data["train"] = DataLoader(
            Subset(base, range(min(args.max_train_samples, len(base)))),
            batch_size=data["train"].batch_size, shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            collate_fn=data["train"].collate_fn)
    if args.max_eval_samples and data["eval"] is not None:
        from torch.utils.data import Subset
        base = data["eval"].dataset
        data["eval"] = DataLoader(
            Subset(base, range(min(args.max_eval_samples, len(base)))),
            batch_size=data["eval"].batch_size, shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            collate_fn=data["eval"].collate_fn)

    # ---- 模型 ----
    model = build_model(cfg["model"]["name"], cfg["model"].get("params", {}))
    model.to(device)
    print(f"[model] {cfg['model']['name']}  "
          f"total={count_parameters(model, False):,}  trainable={count_parameters(model, True):,}")

    # ---- 损失（加权交叉熵，权重由训练集标签统计得到；类别极不平衡 ~1:5.9）----
    labels = data["labels"]
    counts = {0: labels.count(0), 1: labels.count(1)}
    print(f"[data] 类别分布 bonafide={counts[0]}  spoof={counts[1]}")
    weights = compute_class_weights(labels, num_classes=2,
                                    mode=cfg.get("class_weight_mode", "inverse"))
    print(f"[loss] class weights = {weights.tolist()}")
    criterion = build_loss(cfg.get("loss", "weighted_ce"), weight=weights,
                           label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    criterion = criterion.to(device)

    # ---- 优化器与调度 ----
    optimizer = build_optimizer(model, cfg)
    epochs = int(cfg.get("epochs", 10))
    steps_per_epoch = max(1, len(data["train"]))
    total_steps = epochs * steps_per_epoch
    warmup = int(cfg["optim"].get("warmup_ratio", 0.03) * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup, total_steps,
        min_lr_ratio=float(cfg["optim"].get("min_lr_ratio", 0.05)))

    if args.resume:
        from opera.utils import load_checkpoint
        load_checkpoint(args.resume, model, optimizer, device)
        print(f"[resume] {args.resume}")

    # ---- 训练 ----
    trainer = Trainer(
        model=model,
        train_loader=data["train"],
        eval_loader=data["eval"],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        out_dir=out_dir,
        scheduler=scheduler,
        epochs=epochs,
        amp=bool(cfg.get("amp", False)),
        grad_clip=float(cfg.get("grad_clip", 5.0)),
        log_interval=int(cfg.get("log_interval", 100)),
        eval_every=int(cfg.get("eval_every", 1)),
        track_level_eval=bool(cfg.get("track_level_eval", False)),
        early_stop_patience=int(cfg.get("early_stop_patience", 0)),
    )
    result = trainer.fit()
    print(f"\n[done] best EER = {result['best_eer']:.4f}%")


if __name__ == "__main__":
    main()
