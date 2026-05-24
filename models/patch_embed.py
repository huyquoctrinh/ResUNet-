import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C_in, H, W)
        x = self.proj(x)                              # (B, C, H/p, W/p)
        grid_size = (x.shape[-2], x.shape[-1])
        x = rearrange(x, "b c h w -> b (h w) c")      # (B, N, C)
        x = self.norm(x)
        return x, grid_size


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C_in, D, H, W)
        x = self.proj(x)                              # (B, C, D/p, H/p, W/p)
        grid_size = (x.shape[-3], x.shape[-2], x.shape[-1])
        x = rearrange(x, "b c d h w -> b (d h w) c")  # (B, N, C)
        x = self.norm(x)
        return x, grid_size


def build_patch_embed(in_channels: int, embed_dim: int, patch_size: int, spatial_dim: int):
    if spatial_dim == 2:
        return PatchEmbed2D(in_channels, embed_dim, patch_size)
    if spatial_dim == 3:
        return PatchEmbed3D(in_channels, embed_dim, patch_size)
    raise ValueError(f"spatial_dim must be 2 or 3, got {spatial_dim}")
