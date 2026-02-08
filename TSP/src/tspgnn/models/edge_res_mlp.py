from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h


class EdgeResMLP(nn.Module):
    """
    Residual MLP over edge features.

    This keeps the same input/output contract as EdgeMLPAny:
    - input: [num_edges, in_dim]
    - output: [num_edges] logits
    """

    def __init__(self, in_dim: int = 10, hidden: int = 128, dropout: float = 0.1, depth: int = 4):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.input_act = nn.GELU()
        self.input_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_ResBlock(hidden=hidden, dropout=dropout) for _ in range(max(1, int(depth)))])
        self.head_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.input_act(x)
        x = self.input_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_norm(x)
        return self.head(x).squeeze(-1)
