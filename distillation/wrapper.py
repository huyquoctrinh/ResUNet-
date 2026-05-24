from typing import Optional

import torch
import torch.nn as nn

from .dino_teacher import DINOv3Teacher
from .feature_projector import MultiScaleProjector
from .kd_loss import FeatureDistillationLoss


class DINOKDModule(nn.Module):
    """Holds the DINO teacher + per-stage projector + KD loss.

    The teacher is frozen; only projector parameters are trainable and must
    be added to the optimizer. Call `compute_loss(image, student_features,
    student_grids)` during the training step to obtain the encoder-feature
    KD loss.
    """

    def __init__(
        self,
        student_dims,
        dino_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        extract_blocks=(2, 5, 8, 11),
        img_size: int = 224,
        kd_loss_mode: str = "cosine",
        spatial_dim: int = 2,
    ):
        super().__init__()
        n_stages = len(student_dims)
        if len(extract_blocks) != n_stages:
            raise ValueError(
                f"extract_blocks ({len(extract_blocks)}) must match "
                f"number of student stages ({n_stages})"
            )
        self.teacher = DINOv3Teacher(
            model_name=dino_model,
            extract_blocks=extract_blocks,
            img_size=img_size,
        )
        self.projector = MultiScaleProjector(
            student_dims=student_dims,
            dino_dim=self.teacher.hidden_size,
            spatial_dim=spatial_dim,
        )
        self.loss_fn = FeatureDistillationLoss(mode=kd_loss_mode)

    def compute_loss(
        self,
        image: torch.Tensor,
        student_features,
        student_grids,
    ):
        teacher_feats, dino_grid = self.teacher(image)
        projected = self.projector(student_features, student_grids, dino_grid)
        loss, per_level = self.loss_fn(projected, teacher_feats)
        return loss, per_level

    def trainable_parameters(self):
        return self.projector.parameters()


def build_dino_kd(cfg, student_dims, spatial_dim: int) -> Optional[DINOKDModule]:
    """Factory: returns a DINOKDModule if cfg.distillation is enabled,
    otherwise None. Keeps training code conditional-free."""
    dcfg = getattr(cfg, "distillation", None)
    if dcfg is None:
        return None
    if not bool(getattr(dcfg, "enabled", False)):
        return None
    if "encoder_feature" not in list(getattr(dcfg, "strategies", [])):
        return None
    img_size = int(getattr(dcfg, "dino_img_size", 224))
    return DINOKDModule(
        student_dims=student_dims,
        dino_model=str(getattr(dcfg, "dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")),
        extract_blocks=tuple(getattr(dcfg, "extract_blocks", (2, 5, 8, 11))),
        img_size=img_size,
        kd_loss_mode=str(getattr(dcfg, "kd_loss_mode", "cosine")),
        spatial_dim=spatial_dim,
    )
