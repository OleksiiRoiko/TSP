from __future__ import annotations
import torch.nn as nn

class EdgeMLPAny(nn.Module):
    def __init__(self, in_dim=10, hidden=128, dropout=0.0, depth=2):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * depth
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
