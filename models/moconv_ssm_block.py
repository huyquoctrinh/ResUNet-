from typing import Sequence

import torch
import torch.nn as nn

from .expert_convs import ExpertBank
from .mamba_layer import MambaLayer
from .router import Router


class MoConvSSMBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_experts: int = 3,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
        ffn_ratio: int = 4,
        spatial_dim: int = 2,
        router_mode: str = "soft",
    ):
        super().__init__()
        self.mamba = MambaLayer(dim, d_state=ssm_d_state, d_conv=ssm_d_conv, expand=ssm_expand)
        self.router = Router(dim, n_experts=n_experts, mode=router_mode)
        self.experts = ExpertBank(dim, kernel_sizes=kernel_sizes, spatial_dim=spatial_dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(dim * ffn_ratio, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim * 2, dim)

    def forward(self, x, grid_size):
        # x: (B, N, C)
        mamba_out = self.mamba(x)                                 # (B, N, C)

        gates = self.router(mamba_out)                            # (B, N, E)
        expert_outs = self.experts(mamba_out, grid_size)          # (B, N, C, E)

        gates_expanded = gates.unsqueeze(2)                       # (B, N, 1, E)
        gated = (expert_outs * gates_expanded).sum(dim=-1)        # (B, N, C)

        moconv_out = self.norm(gated + self.ffn(gated))           # (B, N, C)

        fused = torch.cat([mamba_out, moconv_out], dim=-1)        # (B, N, 2C)
        out = self.out_proj(fused)                                # (B, N, C)

        return out + x
