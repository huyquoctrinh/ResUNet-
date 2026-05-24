import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, dim: int, n_experts: int = 3, mode: str = "soft", top_k: int = 2):
        super().__init__()
        assert mode in {"soft", "top_k"}
        self.mode = mode
        self.top_k = top_k
        self.n_experts = n_experts
        self.gate = nn.Linear(dim, n_experts)
        self.aux_loss = torch.tensor(0.0)

    def _load_balance_loss(self, gates):
        # Classic MoE balance term: n_experts * mean(gate_i) * mean(one_hot(argmax_i))
        with torch.no_grad():
            top1 = gates.argmax(dim=-1)
            mask = F.one_hot(top1, num_classes=self.n_experts).float()
            fraction = mask.mean(dim=(0, 1))           # (E,)
        prob = gates.mean(dim=(0, 1))                  # (E,)
        return self.n_experts * (fraction * prob).sum()

    def forward(self, x):
        # x: (B, N, C)
        logits = self.gate(x)                          # (B, N, E)
        if self.mode == "soft":
            gates = F.softmax(logits, dim=-1)
        else:
            top_vals, top_idx = logits.topk(self.top_k, dim=-1)
            sparse = torch.full_like(logits, float("-inf"))
            sparse.scatter_(-1, top_idx, top_vals)
            gates = F.softmax(sparse, dim=-1)
        self.aux_loss = self._load_balance_loss(gates)
        return gates                                   # (B, N, E)
