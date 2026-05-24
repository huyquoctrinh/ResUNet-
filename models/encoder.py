from typing import Sequence

import torch
import torch.nn as nn
from einops import rearrange

from .moconv_ssm_block import MoConvSSMBlock
from .patch_embed import build_patch_embed


class PatchMerging(nn.Module):
    """Stride-2 spatial conv that halves spatial dims and doubles channels."""

    def __init__(self, dim_in: int, dim_out: int, spatial_dim: int = 2):
        super().__init__()
        if spatial_dim == 2:
            self.down = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        elif spatial_dim == 3:
            self.down = nn.Conv3d(dim_in, dim_out, kernel_size=2, stride=2)
        else:
            raise ValueError(spatial_dim)
        self.norm = nn.LayerNorm(dim_out)
        self.spatial_dim = spatial_dim

    def forward(self, x, grid_size):
        # x: (B, N, C)
        if self.spatial_dim == 2:
            h, w = grid_size
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            x = self.down(x)
            new_grid = (x.shape[-2], x.shape[-1])
            x = rearrange(x, "b c h w -> b (h w) c")
        else:
            d, h, w = grid_size
            x = rearrange(x, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
            x = self.down(x)
            new_grid = (x.shape[-3], x.shape[-2], x.shape[-1])
            x = rearrange(x, "b c d h w -> b (d h w) c")
        x = self.norm(x)
        return x, new_grid


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_dim: int = 48,
        depths: Sequence[int] = (1, 1, 1, 1),
        patch_size: int = 4,
        spatial_dim: int = 2,
        **block_kwargs,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.num_stages = len(depths)
        self.patch_embed = build_patch_embed(in_channels, base_dim, patch_size, spatial_dim)

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.stage_dims = []

        for i, depth in enumerate(depths):
            dim = base_dim * (2 ** i)
            self.stage_dims.append(dim)
            blocks = nn.ModuleList(
                [
                    MoConvSSMBlock(dim=dim, spatial_dim=spatial_dim, **block_kwargs)
                    for _ in range(depth)
                ]
            )
            self.stages.append(blocks)
            if i < len(depths) - 1:
                self.downsamples.append(PatchMerging(dim, dim * 2, spatial_dim))

    def forward(self, x):
        # x: (B, C_in, *spatial)
        features = []
        grid_sizes = []

        x, grid = self.patch_embed(x)                 # (B, N, C)

        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x, grid)
            features.append(x)
            grid_sizes.append(grid)
            if i < self.num_stages - 1:
                x, grid = self.downsamples[i](x, grid)

        return features, grid_sizes
