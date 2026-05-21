import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationLoss(nn.Module):
    """MSE-based feature distillation loss.

    Projects student features to the teacher's dimension via a learned 1x1 conv,
    then computes MSE between the projected student features and the (frozen)
    teacher features. Spatial sizes are aligned via bilinear interpolation when
    they differ (e.g. student at stride 16 vs teacher at a different resolution).
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(student_dim, teacher_dim, kernel_size=1)
        self.mse = nn.MSELoss()

    def forward(self, student_feat: torch.Tensor,
                teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_feat: (B, student_dim, Hs, Ws)
            teacher_feat: (B, teacher_dim, Ht, Wt)  — detached / no grad.

        Returns:
            Scalar MSE loss.
        """
        proj = self.proj(student_feat)  # (B, teacher_dim, Hs, Ws)

        if proj.shape[2:] != teacher_feat.shape[2:]:
            proj = F.interpolate(
                proj, size=teacher_feat.shape[2:],
                mode='bilinear', align_corners=False,
            )

        return self.mse(proj, teacher_feat.detach())
