import torch
import torch.nn as nn


class EdgeCrossAttention(nn.Module):
    """Multi-head cross-attention: Q = encoder features, K = V = edge features."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, enc_feat, edge_feat):
        """
        enc_feat:  (B, C, H, W) encoder features -> Q
        edge_feat: (B, C, H, W) edge features    -> K, V
        Returns:   (B, C, H, W) attended + residual
        """
        B, C, H, W = enc_feat.shape
        N = H * W

        q = self.norm_q(enc_feat.flatten(2).transpose(1, 2))
        kv = self.norm_kv(edge_feat.flatten(2).transpose(1, 2))

        q = self.q_proj(q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.out_proj(out))

        return out.transpose(1, 2).reshape(B, C, H, W) + enc_feat
