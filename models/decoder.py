import torch
import torch.nn as nn
from einops import rearrange


class Upsample(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, spatial_dim: int = 2):
        super().__init__()
        if spatial_dim == 2:
            self.up = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2)
        elif spatial_dim == 3:
            self.up = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=2, stride=2)
        else:
            raise ValueError(spatial_dim)
        self.spatial_dim = spatial_dim

    def forward(self, x, grid_size):
        # x: (B, N, C)
        if self.spatial_dim == 2:
            h, w = grid_size
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            x = self.up(x)
            new_grid = (x.shape[-2], x.shape[-1])
            x = rearrange(x, "b c h w -> b (h w) c")
        else:
            d, h, w = grid_size
            x = rearrange(x, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
            x = self.up(x)
            new_grid = (x.shape[-3], x.shape[-2], x.shape[-1])
            x = rearrange(x, "b c d h w -> b (d h w) c")
        return x, new_grid


class DecoderBlock(nn.Module):
    def __init__(self, dim_in: int, dim_skip: int, dim_out: int, spatial_dim: int = 2):
        super().__init__()
        self.upsample = Upsample(dim_in, dim_out, spatial_dim)
        self.fuse = nn.Linear(dim_out + dim_skip, dim_out)
        self.refine = nn.Sequential(
            nn.Linear(dim_out, dim_out * 2),
            nn.GELU(),
            nn.Linear(dim_out * 2, dim_out),
        )
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x, grid_from, skip, grid_to):
        # x: (B, N_in, C_in)
        x, new_grid = self.upsample(x, grid_from)                 # (B, N_up, C_out)
        assert new_grid == grid_to, f"grid mismatch {new_grid} vs {grid_to}"
        x = torch.cat([x, skip], dim=-1)                          # (B, N_up, C_out+C_skip)
        x = self.fuse(x)                                          # (B, N_up, C_out)
        x = self.norm(x + self.refine(x))                         # (B, N_up, C_out)
        return x


class Decoder(nn.Module):
    def __init__(self, stage_dims, spatial_dim: int = 2):
        """
        stage_dims: [C0, C1, C2, C3] (encoder channels per stage, shallow→deep).
        Decoder fuses features[3]→features[2]→features[1]→features[0].
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(stage_dims) - 1, 0, -1):
            dim_in = stage_dims[i]
            dim_skip = stage_dims[i - 1]
            dim_out = stage_dims[i - 1]
            self.blocks.append(DecoderBlock(dim_in, dim_skip, dim_out, spatial_dim))

    def forward(self, features, grid_sizes):
        # features: [f0, f1, f2, f3] shallow→deep; same order for grid_sizes
        x = features[-1]
        grid = grid_sizes[-1]
        intermediates = []  # list of (feat, grid) after each decoder block
        for i, block in enumerate(self.blocks):
            skip_idx = len(features) - 2 - i
            x = block(x, grid, features[skip_idx], grid_sizes[skip_idx])
            grid = grid_sizes[skip_idx]
            intermediates.append((x, grid))
        return x, grid, intermediates
