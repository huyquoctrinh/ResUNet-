import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba  # type: ignore
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False


class _FallbackSSM(nn.Module):
    """Bidirectional GRU fallback when mamba-ssm isn't available."""

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        hidden = d_model * expand // 2
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.out_proj = nn.Linear(2 * hidden, d_model)

    def forward(self, x):
        # x: (B, N, C)
        y, _ = self.gru(x)              # (B, N, 2*hidden)
        return self.out_proj(y)         # (B, N, C)


class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA:
            self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mixer = _FallbackSSM(d_model=d_model, expand=expand)

    def forward(self, x):
        # x: (B, N, C)
        return x + self.mixer(self.norm(x))  # (B, N, C)
