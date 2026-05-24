from typing import Sequence

import torch
import torch.nn as nn
from einops import rearrange


class ConvExpert(nn.Module):
    def __init__(self, dim: int, kernel_size: int, spatial_dim: int = 2):
        super().__init__()
        padding = kernel_size // 2
        if spatial_dim == 2:
            self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
            self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        elif spatial_dim == 3:
            self.conv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
            self.pw = nn.Conv3d(dim, dim, kernel_size=1)
        else:
            raise ValueError(f"spatial_dim must be 2 or 3, got {spatial_dim}")
        self.act = nn.GELU()
        self.spatial_dim = spatial_dim

    def forward(self, x, grid_size):
        # x: (B, N, C)
        if self.spatial_dim == 2:
            h, w = grid_size
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            x = self.pw(self.act(self.conv(x)))
            x = rearrange(x, "b c h w -> b (h w) c")
        else:
            d, h, w = grid_size
            x = rearrange(x, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
            x = self.pw(self.act(self.conv(x)))
            x = rearrange(x, "b c d h w -> b (d h w) c")
        return x                                      # (B, N, C)


class ExpertBank(nn.Module):
    def __init__(self, dim: int, kernel_sizes: Sequence[int] = (3, 5, 7), spatial_dim: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([ConvExpert(dim, k, spatial_dim) for k in kernel_sizes])

    def forward(self, x, grid_size):
        # x: (B, N, C)
        outs = [e(x, grid_size) for e in self.experts]  # list of (B, N, C)
        return torch.stack(outs, dim=-1)                 # (B, N, C, E)
