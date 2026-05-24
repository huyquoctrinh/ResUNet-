from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationLoss(nn.Module):
    """Per-stage feature distillation loss between student and teacher
    feature maps of matching spatial shape `(B, D, gh, gw)`.

    Modes:
      * mse:       raw MSE on unnormalized features (FitNets-style)
      * mse_norm:  MSE on L2-normalized features
      * cosine:    1 - mean pixel-wise cosine similarity (L2-normalized)
      * smooth_l1: raw smooth L1 (no normalization)
    """

    def __init__(self, mode: str = "mse"):
        super().__init__()
        assert mode in {"mse", "mse_norm", "cosine", "smooth_l1"}, mode
        self.mode = mode

    def forward(
        self,
        student_feats: Sequence[torch.Tensor],
        teacher_feats: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[float]]:
        assert len(student_feats) == len(teacher_feats)
        total = 0.0
        per_level = []
        for s, t in zip(student_feats, teacher_feats):
            t = t.detach()
            if s.shape[-2:] != t.shape[-2:]:
                t = F.interpolate(
                    t, size=s.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            if self.mode == "mse":
                loss_i = F.mse_loss(s, t)
            elif self.mode == "mse_norm":
                s_n = F.normalize(s, dim=1)
                t_n = F.normalize(t, dim=1)
                loss_i = F.mse_loss(s_n, t_n)
            elif self.mode == "cosine":
                s_n = F.normalize(s, dim=1)
                t_n = F.normalize(t, dim=1)
                sim = (s_n * t_n).sum(dim=1)          # (B, gh, gw)
                loss_i = 1.0 - sim.mean()
            else:
                loss_i = F.smooth_l1_loss(s, t)
            per_level.append(loss_i.detach().item())
            total = total + loss_i
        return total / max(1, len(student_feats)), per_level
