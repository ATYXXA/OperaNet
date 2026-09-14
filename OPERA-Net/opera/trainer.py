"""训练与评估循环。

训练配置（论文 4.2 节）
----------------------
- 优化器：AdamW
- 分层学习率衰减：预训练 WavLM 的学习率远小于随机初始化的 PC-CQT 分支与分类头，
  且 WavLM 内部按层递减（见 models/wavlm_stream.py::layerwise_param_groups）
- 损失：加权交叉熵（处理 bonafide / deepfake 类别不平衡）
- 输入：16 kHz、4 s 定长窗口
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_eer, compute_min_tdcf, evaluate_all
from .utils import AverageMeter, ensure_dir, save_checkpoint


# ---------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    return_meta: bool = True,
) -> Dict[str, Any]:
    """对整个 loader 做推理，返回分数、标签与元信息。

    推理分数 = logit[:, 1] − logit[:, 0]（越大越像 spoof），
    与 metrics 模块的约定一致。
    """
    model.eval()
    all_scores: List[float] = []
    all_labels: List[int] = []
    all_ids: List[str] = []
    all_attacks: List[str] = []
    all_tracks: List[str] = []
    all_sources: List[str] = []

    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy().tolist()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(wav)
            if isinstance(logits, tuple):      # 兼容 return_gate=True 的模型输出
                logits = logits[0]
        scores = (logits[:, 1] - logits[:, 0]).float().cpu().numpy().tolist()

        all_scores.extend(scores)
        all_labels.extend(labels)
        all_ids.extend(batch.get("utt_id", [""] * len(labels)))
        all_attacks.extend(batch.get("attack", ["-"] * len(labels)))
        all_sources.extend(batch.get("source", [""] * len(labels)))
        if "track_id" in batch:
            all_tracks.extend(batch["track_id"])

    out: Dict[str, Any] = {
        "scores": np.asarray(all_scores, dtype=np.float64),
        "labels": np.asarray(all_labels, dtype=np.int32),
    }
    if return_meta:
        out["utt_id"] = all_ids
        out["attack"] = all_attacks
        out["source"] = all_sources
        if all_tracks:
            out["track_id"] = all_tracks
    return out


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    asv_scores: Optional[Sequence[float]] = None,
    asv_labels: Optional[Sequence[int]] = None,
    track_level: bool = False,
) -> Dict[str, Any]:
    """推理 + 计算 EER / min-tDCF。

    track_level=True 时先把片段级分数按曲目聚合（SingFake 长音频推荐开启）。
    """
    res = run_inference(model, loader, device, amp_dtype=amp_dtype)
    scores, labels = res["scores"], res["labels"]

    if track_level and "track_id" in res:
        from .metrics import track_level_aggregate
        scores, labels = track_level_aggregate(scores, res["track_id"], labels)

    metrics = evaluate_all(scores, labels, asv_scores, asv_labels)
    metrics["scores"] = scores
    metrics["labels"] = labels
    metrics["utt_id"] = res.get("utt_id", [])
    metrics["attack"] = res.get("attack", [])
    return metrics


def evaluate_by_attack(metrics: Dict[str, Any], attacks: Sequence[str]) -> Dict[str, float]:
    """按攻击类型分组统计 EER（对应论文雷达图 A09–A13 的口径，A14 单独看）。"""
    scores = np.asarray(metrics["scores"])
    labels = np.asarray(metrics["labels"])
    atk = np.asarray(metrics.get("attack", []), dtype=object)
    out = {}
    for a in attacks:
        mask = (atk == a) | (labels == 0)
        if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
            continue
        eer, _ = compute_eer(scores[mask], labels[mask])
        out[a] = round(eer * 100, 4)
    return out


# ---------------------------------------------------------------------- #
class Trainer:
    """最小可用的训练器：训练 / 每轮评估 / 保存 best 与 last。"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        out_dir: str | Path,
        scheduler: Optional[Any] = None,
        epochs: int = 10,
        amp: bool = False,
        grad_clip: float = 5.0,
        log_interval: int = 100,
        eval_every: int = 1,
        track_level_eval: bool = False,
        asv_scores: Optional[Sequence[float]] = None,
        asv_labels: Optional[Sequence[int]] = None,
        early_stop_patience: int = 0,
        use_compile: bool = False,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.out_dir = ensure_dir(out_dir)
        self.scheduler = scheduler
        self.epochs = epochs
        self.amp = amp
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.eval_every = eval_every
        self.track_level_eval = track_level_eval
        self.asv_scores = asv_scores
        self.asv_labels = asv_labels
        self.early_stop_patience = early_stop_patience
        self.amp_dtype = torch.float16 if amp and device.type == "cuda" else None
        self.scaler = self._make_grad_scaler(device, self.amp_dtype is not None)
        if use_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(model)
            except Exception:
                pass

        self.best_eer = float("inf")
        self.history: List[Dict[str, Any]] = []
        self.log_path = self.out_dir / "metric_log.txt"

    @staticmethod
    def _make_grad_scaler(device: torch.device, enabled: bool):
        """兼容新旧 PyTorch 的 GradScaler 构造方式。"""
        try:  # PyTorch >= 2.4
            from torch.amp import GradScaler
            return GradScaler(device.type, enabled=enabled)
        except (ImportError, TypeError):  # pragma: no cover
            return torch.cuda.amp.GradScaler(enabled=enabled)

    # ------------------------------------------------------------------ #
    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        losses = AverageMeter()
        t0 = time.time()
        n_batches = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, start=1):
            wav = batch["wav"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.amp_dtype is not None):
                logits = self.model(wav)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = self.criterion(logits, labels)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            losses.update(float(loss.item()), wav.size(0))
            if step % self.log_interval == 0 or step == n_batches:
                lr_now = self.optimizer.param_groups[0]["lr"]
                print(f"  epoch {epoch:>3} | {step:>6}/{n_batches} "
                      f"| loss {losses.avg:.4f} | lr {lr_now:.2e} "
                      f"| {time.time() - t0:.0f}s")

        return {"loss": losses.avg, "epoch_time": time.time() - t0}

    # ------------------------------------------------------------------ #
    def _eval(self, epoch: int) -> Dict[str, Any]:
        m = evaluate_model(
            self.model, self.eval_loader, self.device,
            amp_dtype=self.amp_dtype,
            asv_scores=self.asv_scores, asv_labels=self.asv_labels,
            track_level=self.track_level_eval,
        )
        m["epoch"] = epoch
        return m

    # ------------------------------------------------------------------ #
    def fit(self) -> Dict[str, Any]:
        best_path = self.out_dir / "best.pth"
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            print(f"\n===== Epoch {epoch}/{self.epochs} =====")
            tr = self._train_one_epoch(epoch)

            record = {"epoch": epoch, "train_loss": round(tr["loss"], 6)}
            do_eval = (self.eval_every > 0) and (epoch % self.eval_every == 0 or epoch == self.epochs)
            if do_eval and self.eval_loader is not None:
                m = self._eval(epoch)
                record.update({k: v for k, v in m.items()
                               if k not in ("scores", "labels", "utt_id", "attack")})
                print(f"  [dev] EER {m['EER_%']:.4f}%")

                if m["EER_%"] < self.best_eer:
                    self.best_eer = m["EER_%"]
                    save_checkpoint({
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "metrics": record,
                    }, best_path, keep_last=0)
                    no_improve = 0
                    print(f"  [eval] 新的最优，已保存到 {best_path}")
                else:
                    no_improve += 1
            else:
                save_checkpoint({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "metrics": record,
                }, self.out_dir / f"epoch_{epoch:03d}.pth")

            self.history.append(record)
            self._log(record)

            if self.early_stop_patience and no_improve >= self.early_stop_patience:
                print(f"[early stop] {self.early_stop_patience} 轮无提升，停止训练")
                break

        self._save_history()
        return {"best_eer": self.best_eer, "history": self.history}

    # ------------------------------------------------------------------ #
    def _log(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _save_history(self) -> None:
        with open(self.out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump({"best_eer": self.best_eer, "history": self.history},
                      f, ensure_ascii=False, indent=2)
