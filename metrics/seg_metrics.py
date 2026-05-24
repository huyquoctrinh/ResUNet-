from typing import Dict, Optional, Sequence

import numpy as np
import torch


def _confusion_counts(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    """Return per-class intersection, pred volume, target volume."""
    pred = pred.view(-1)
    target = target.view(-1)
    intersect = torch.zeros(num_classes, dtype=torch.float64, device=pred.device)
    pred_sum = torch.zeros(num_classes, dtype=torch.float64, device=pred.device)
    tgt_sum = torch.zeros(num_classes, dtype=torch.float64, device=pred.device)
    for c in range(num_classes):
        p = pred == c
        t = target == c
        intersect[c] = (p & t).sum()
        pred_sum[c] = p.sum()
        tgt_sum[c] = t.sum()
    return intersect, pred_sum, tgt_sum


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_background: bool = True, smooth: float = 1e-6):
    inter, p, t = _confusion_counts(pred, target, num_classes)
    dice = (2 * inter + smooth) / (p + t + smooth)
    if ignore_background:
        dice = dice[1:]
    return dice, dice.mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_background: bool = True, smooth: float = 1e-6):
    inter, p, t = _confusion_counts(pred, target, num_classes)
    union = p + t - inter
    iou = (inter + smooth) / (union + smooth)
    if ignore_background:
        iou = iou[1:]
    return iou, iou.mean()


class MetricTracker:
    """Accumulates per-class intersection / pred / target counts across batches."""

    def __init__(self, num_classes: int, class_names: Optional[Sequence[str]] = None, ignore_background: bool = True):
        self.num_classes = num_classes
        self.class_names = list(class_names) if class_names is not None else [f"class_{i}" for i in range(num_classes)]
        self.ignore_background = ignore_background
        self.reset()

    def reset(self):
        self.intersect = torch.zeros(self.num_classes, dtype=torch.float64)
        self.pred_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        self.tgt_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        self._sample_dice = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # pred, target: same shape, integer labels
        inter, p, t = _confusion_counts(pred.cpu(), target.cpu(), self.num_classes)
        self.intersect += inter
        self.pred_sum += p
        self.tgt_sum += t
        # Per-sample mean dice for averaging style
        dice = (2 * inter + 1e-6) / (p + t + 1e-6)
        if self.ignore_background:
            dice = dice[1:]
        self._sample_dice.append(dice.mean().item())

    def compute(self) -> Dict:
        inter, p, t = self.intersect, self.pred_sum, self.tgt_sum
        dice = (2 * inter + 1e-6) / (p + t + 1e-6)
        iou = (inter + 1e-6) / (p + t - inter + 1e-6)
        if self.ignore_background:
            per_class_dice = dice[1:].numpy()
            per_class_iou = iou[1:].numpy()
            names = self.class_names[1:]
        else:
            per_class_dice = dice.numpy()
            per_class_iou = iou.numpy()
            names = self.class_names
        return {
            "class_names": names,
            "per_class_dice": per_class_dice,
            "per_class_iou": per_class_iou,
            "mean_dice": float(per_class_dice.mean()),
            "mean_iou": float(per_class_iou.mean()),
            "sample_mean_dice": float(np.mean(self._sample_dice)) if self._sample_dice else 0.0,
        }

    def format_table(self, title: str = "Results") -> str:
        res = self.compute()
        lines = []
        bar = "=" * 54
        lines.append(bar)
        lines.append(f"{title:^54}")
        lines.append(bar)
        lines.append(f"{'Class':<20}| {'Dice (%)':<12}| {'IoU (%)':<12}")
        lines.append("-" * 54)
        for name, d, i in zip(res["class_names"], res["per_class_dice"], res["per_class_iou"]):
            lines.append(f"{name:<20}| {d*100:<12.2f}| {i*100:<12.2f}")
        lines.append("-" * 54)
        tag = "Mean (excl. BG)" if self.ignore_background else "Mean"
        lines.append(f"{tag:<20}| {res['mean_dice']*100:<12.2f}| {res['mean_iou']*100:<12.2f}")
        lines.append(bar)
        return "\n".join(lines)
