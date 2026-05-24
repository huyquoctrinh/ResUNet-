"""PVT v2 encoder wrapper that exposes the same interface as the MoConvSSM Encoder.

Uses HuggingFace transformers PvtV2Model under the hood.
Output format:  features (list of (B, N, C)), grid_sizes (list of (H, W)).
"""

import math
from typing import Sequence

import torch
import torch.nn as nn
from transformers import PvtV2Config, PvtV2Model


# Map variant name -> HuggingFace model id
PVT_V2_VARIANTS = {
    "b0": "OpenGVLab/pvt_v2_b0",
    "b1": "OpenGVLab/pvt_v2_b1",
    "b2": "OpenGVLab/pvt_v2_b2",
    "b3": "OpenGVLab/pvt_v2_b3",
    "b4": "OpenGVLab/pvt_v2_b4",
    "b5": "OpenGVLab/pvt_v2_b5",
}


class PvtV2Encoder(nn.Module):
    """Wraps HuggingFace PvtV2Model to match the Encoder interface used by
    MoConvSSMNet (features as list of (B,N,C), grid_sizes as list of (H,W)).
    """

    def __init__(
        self,
        variant: str = "b2",
        in_channels: int = 1,
        pretrained: bool = True,
    ):
        super().__init__()
        model_id = PVT_V2_VARIANTS[variant]

        if pretrained:
            self.pvt = PvtV2Model.from_pretrained(model_id)
        else:
            config = PvtV2Config.from_pretrained(model_id)
            self.pvt = PvtV2Model(config)

        # Expose stage channel dims for the decoder
        self.stage_dims = list(self.pvt.config.hidden_sizes)

        # Adapt first patch embedding if in_channels != 3
        if in_channels != 3:
            old_proj = self.pvt.encoder.layers[0].patch_embedding.proj
            new_proj = nn.Conv2d(
                in_channels,
                old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
            )
            if pretrained:
                # Average RGB weights into new channel dim
                with torch.no_grad():
                    new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
                    new_proj.bias.copy_(old_proj.bias)
            self.pvt.encoder.layers[0].patch_embedding.proj = new_proj

        # Spatial reduction factors per stage (cumulative strides)
        strides = self.pvt.config.strides  # [4, 2, 2, 2]
        self._cum_strides = []
        s = 1
        for st in strides:
            s *= st
            self._cum_strides.append(s)

    def forward(self, x):
        # x: (B, C_in, H, W)
        outputs = self.pvt(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states: tuple of (B, C, H, W) for each stage (0-indexed)
        all_hidden = outputs.hidden_states

        features = []
        grid_sizes = []
        for i in range(len(self.stage_dims)):
            feat = all_hidden[i]             # (B, C, H, W)
            h, w = feat.shape[-2], feat.shape[-1]
            # Reshape to (B, N, C) to match decoder interface
            feat = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            features.append(feat)
            grid_sizes.append((h, w))

        return features, grid_sizes
