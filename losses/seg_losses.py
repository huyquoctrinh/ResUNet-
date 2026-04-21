import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    # target: (B, *spatial) long → (B, K, *spatial)
    if target.dim() == 3:
        return F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    if target.dim() == 4:
        return F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
    raise ValueError(f"target must be 3D or 4D, got {target.shape}")


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, K, *spatial); target: (B, *spatial) long
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        tgt = _one_hot(target.clamp(min=0), num_classes)
        reduce_dims = tuple(range(2, logits.dim()))
        intersection = (probs * tgt).sum(dim=reduce_dims)
        denom = probs.sum(dim=reduce_dims) + tgt.sum(dim=reduce_dims)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ce(logits, target) + (1 - self.alpha) * self.dice(logits, target)
