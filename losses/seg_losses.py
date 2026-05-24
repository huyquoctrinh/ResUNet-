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
    """Soft Dice loss — matches PraNet-V2/MIST formulation:
        dice = (2·∑ s·t + ε) / (∑ s² + ∑ t² + ε), averaged over all classes.
    """

    def __init__(self, smooth: float = 1e-5, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, K, *spatial); target: (B, *spatial) long
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        tgt = _one_hot(target.clamp(min=0), num_classes)
        reduce_dims = tuple(range(2, logits.dim()))
        intersect = (probs * tgt).sum(dim=reduce_dims)
        s_sum = (probs * probs).sum(dim=reduce_dims)
        t_sum = (tgt * tgt).sum(dim=reduce_dims)
        dice = (2 * intersect + self.smooth) / (s_sum + t_sum + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.alpha * self.ce(logits, target)
            + (1 - self.alpha) * self.dice(logits, target)
        )


class DeepSupervisionLoss(nn.Module):
    """Apply `CombinedLoss` at every decoder level with decreasing weights.

    Expects `logits_list` = [main, aux1, aux2, ...] where `main` is the
    finest-resolution output and subsequent entries are progressively coarser
    aux outputs (already upsampled to the input spatial size).

    Weights are normalized so the loss scale matches single-output training.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        weights=(1.0, 0.5, 0.25, 0.125),
        ignore_index: int = -100,
    ):
        super().__init__()
        self.base = CombinedLoss(alpha=alpha, ignore_index=ignore_index)
        self.weights = tuple(weights)

    def forward(self, logits_list, target: torch.Tensor) -> torch.Tensor:
        if isinstance(logits_list, torch.Tensor):
            return self.base(logits_list, target)
        n = len(logits_list)
        w = self.weights[:n]
        total_w = sum(w)
        loss = 0.0
        for weight, logits in zip(w, logits_list):
            loss = loss + (weight / total_w) * self.base(logits, target)
        return loss
