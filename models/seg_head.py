import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SegHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, scale_factor: int, spatial_dim: int = 2):
        """scale_factor: spatial upsample needed to reach the input resolution."""
        super().__init__()
        self.scale_factor = scale_factor
        self.spatial_dim = spatial_dim
        if spatial_dim == 2:
            self.proj = nn.Conv2d(dim, num_classes, kernel_size=1)
        elif spatial_dim == 3:
            self.proj = nn.Conv3d(dim, num_classes, kernel_size=1)
        else:
            raise ValueError(spatial_dim)

    def forward(self, x, grid_size):
        # x: (B, N, C)
        if self.spatial_dim == 2:
            h, w = grid_size
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            x = self.proj(x)                                      # (B, K, h, w)
            x = F.interpolate(
                x, scale_factor=self.scale_factor,
                mode="bilinear", align_corners=False,
            )
        else:
            d, h, w = grid_size
            x = rearrange(x, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
            x = self.proj(x)                                      # (B, K, d, h, w)
            x = F.interpolate(
                x, scale_factor=self.scale_factor,
                mode="trilinear", align_corners=False,
            )
        return x
