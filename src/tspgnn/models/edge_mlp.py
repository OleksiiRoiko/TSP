from __future__ import annotations
import torch.nn as nn

class EdgeMLPAny(nn.Module):
    def __init__(self, in_dim=10, hidden=128, dropout=0.0, depth=2):
        super().__init__()
        in_dim = int(in_dim)
        hidden = int(hidden)
        depth = int(depth)
        dropout = float(dropout)
        if in_dim < 1:
            raise ValueError("in_dim must be >= 1")
        if hidden < 1:
            raise ValueError("hidden must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        layers = []
        dims = [in_dim] + [hidden] * depth
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
