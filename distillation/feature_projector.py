from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiScaleProjector(nn.Module):
    """Project each student encoder stage to the DINO feature space.

    For each stage:
      tokens (B, N_i, C_i) → spatial (B, C_i, *grid_i)
                           → interpolate to (B, C_i, *target_grid)
                           → 1×1 conv + GroupNorm → (B, D_dino, *target_grid)

    One projector per student stage. Supports 2D student only (DINO is 2D).
    """

    def __init__(
        self,
        student_dims: Sequence[int],
        dino_dim: int,
        spatial_dim: int = 2,
    ):
        super().__init__()
        if spatial_dim != 2:
            raise ValueError("DINO KD only supports 2D student encoders")
        self.spatial_dim = spatial_dim
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(d, dino_dim, kernel_size=1),
                    nn.GroupNorm(
                        num_groups=min(32, dino_dim),
                        num_channels=dino_dim,
                    ),
                )
                for d in student_dims
            ]
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        grid_sizes: Sequence[Tuple[int, int]],
        target_grid: Tuple[int, int],
    ) -> list:
        projected = []
        for feat, grid, proj in zip(features, grid_sizes, self.projs):
            h, w = grid
            x = rearrange(feat, "b (h w) c -> b c h w", h=h, w=w)
            if (h, w) != tuple(target_grid):
                x = F.interpolate(
                    x, size=tuple(target_grid),
                    mode="bilinear", align_corners=False,
                )
            x = proj(x)
            projected.append(x)
        return projected
