from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DINOv3Teacher(nn.Module):
    """Frozen DINOv3 ViT teacher that extracts patch-token features at a
    chosen set of transformer blocks.

    * Handles 1→3 channel expansion for grayscale medical images.
    * Applies ImageNet mean/std normalization (DINOv3 was trained on LVD
      with ImageNet-style stats).
    * Resizes input to a fixed `img_size` (default 224) to match training
      resolution and avoid position-embedding interpolation issues.
    * Returns a list of spatial feature maps `(B, D, gh, gw)` — one per
      requested block index — at DINO's native patch grid.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        extract_blocks: Sequence[int] = (2, 5, 8, 11),
        img_size: int = 224,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for DINOv3 KD: pip install transformers"
            ) from e

        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        cfg = self.model.config
        self.patch_size = getattr(cfg, "patch_size", 16)
        self.hidden_size = getattr(cfg, "hidden_size", 384)
        self.num_layers = getattr(
            cfg, "num_hidden_layers", getattr(cfg, "num_layers", 12)
        )
        self.extract_blocks = tuple(extract_blocks)
        for b in self.extract_blocks:
            if not (0 <= b <= self.num_layers):
                raise ValueError(
                    f"extract_block {b} out of range [0, {self.num_layers}]"
                )

        self.img_size = img_size
        self.grid = img_size // self.patch_size

        # ImageNet normalization; input is assumed already scaled to [0, 1].
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        """x: (B, C_in, H, W). Returns (features, grid).

        features: list of (B, hidden_size, grid, grid)
        grid: (gh, gw) tuple
        """
        B, C, H, W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        elif C != 3:
            raise ValueError(f"DINOv3 expects 1 or 3 input channels, got {C}")

        x = (x - self.mean) / self.std
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )

        out = self.model(pixel_values=x, output_hidden_states=True)
        hidden = out.hidden_states  # tuple of length num_layers + 1

        grid = self.grid
        expected = grid * grid
        features = []
        for b in self.extract_blocks:
            h = hidden[b]                  # (B, N_tokens, D)
            # Drop leading special tokens (CLS, register tokens); keep last
            # `expected` patch tokens.
            if h.shape[1] < expected:
                raise RuntimeError(
                    f"block {b}: expected ≥{expected} patch tokens, got {h.shape[1]}"
                )
            h = h[:, -expected:, :]
            h = rearrange(h, "b (gh gw) d -> b d gh gw", gh=grid, gw=grid)
            features.append(h.contiguous())

        return features, (grid, grid)
