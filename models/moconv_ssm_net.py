from typing import Sequence

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .seg_head import SegHead


class MoConvSSMNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_dim: int = 48,
        depths: Sequence[int] = (1, 1, 1, 1),
        patch_size: int = 4,
        spatial_dim: int = 2,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        n_experts: int = 3,
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
        ffn_ratio: int = 4,
        router_mode: str = "soft",
        deep_supervision: bool = False,
        encoder_type: str = "moconv_ssm",
        pvt_variant: str = "b2",
        pvt_pretrained: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder_type = encoder_type

        if encoder_type == "pvt_v2":
            from .pvt_v2_encoder import PvtV2Encoder
            self.encoder = PvtV2Encoder(
                variant=pvt_variant,
                in_channels=in_channels,
                pretrained=pvt_pretrained,
            )
            # PVT v2 uses stride-4 patch embed like the default
            self.patch_size = 4
            patch_size = 4
        else:
            self.patch_size = patch_size
            block_kwargs = dict(
                n_experts=n_experts,
                kernel_sizes=tuple(kernel_sizes),
                ssm_d_state=ssm_d_state,
                ssm_d_conv=ssm_d_conv,
                ssm_expand=ssm_expand,
                ffn_ratio=ffn_ratio,
                router_mode=router_mode,
            )
            self.encoder = Encoder(
                in_channels=in_channels,
                base_dim=base_dim,
                depths=tuple(depths),
                patch_size=patch_size,
                spatial_dim=spatial_dim,
                **block_kwargs,
            )

        base_dim_dec = self.encoder.stage_dims[0]
        self.decoder = Decoder(
            stage_dims=self.encoder.stage_dims, spatial_dim=spatial_dim
        )
        self.seg_head = SegHead(base_dim_dec, num_classes, patch_size, spatial_dim)

        # Deep supervision: aux heads for each decoder intermediate EXCEPT the
        # finest-resolution one (which is already fed to `self.seg_head`).
        # Decoder produces `len(depths)-1` intermediates; the first N-1 come
        # from coarser decoder stages and get aux heads. Each aux output is
        # upsampled to the full input resolution.
        self.aux_heads = nn.ModuleList()
        if deep_supervision:
            stage_dims = self.encoder.stage_dims
            # intermediate i corresponds to decoder block i:
            #   block_0 output has dim=stage_dims[-2], grid=grid_sizes[-2]  → factor=patch_size*2^(len-2)
            #   block_1 output has dim=stage_dims[-3], grid=grid_sizes[-3]  → factor=patch_size*2^(len-3)
            #   ...
            #   block_{N-2} output is the finest (same as final) → main head
            num_blocks = len(stage_dims) - 1
            for i in range(num_blocks - 1):
                stage_idx = num_blocks - 1 - i   # encoder stage index of this aux
                dim = stage_dims[stage_idx]
                scale = patch_size * (2 ** stage_idx)
                self.aux_heads.append(
                    SegHead(dim, num_classes, scale, spatial_dim)
                )

    def aux_losses(self):
        """Sum of router load-balancing losses across blocks."""
        total = 0.0
        count = 0
        for m in self.modules():
            if hasattr(m, "aux_loss") and isinstance(m.aux_loss, torch.Tensor):
                total = total + m.aux_loss
                count += 1
        if count == 0:
            return torch.tensor(0.0)
        return total / count

    def forward(self, x):
        # x: (B, C_in, *spatial)
        features, grid_sizes = self.encoder(x)
        decoded, top_grid, intermediates = self.decoder(features, grid_sizes)
        main_logits = self.seg_head(decoded, top_grid)

        if not (self.deep_supervision and self.training):
            return main_logits

        # intermediates[:-1] → coarser-to-finer; last is the finest (=decoded).
        aux_logits = []
        for head, (feat, grid) in zip(self.aux_heads, intermediates[:-1]):
            aux_logits.append(head(feat, grid))

        # Return deepest-to-shallowest: [main, aux (mid), aux (coarse), ...]
        # The main output (finest resolution) is first; aux outputs follow.
        return [main_logits] + aux_logits
