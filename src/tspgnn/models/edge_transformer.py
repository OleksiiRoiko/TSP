from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class EdgeTransformer(nn.Module):
    """
    Graph-aware edge scorer:
    1) encode nodes from coordinates with Transformer blocks
    2) score each undirected edge (i, j) from node embeddings + edge features
    """

    # Used by train/eval/viz to provide graph context (coords + edge_counts).
    requires_graph_context: bool = True

    def __init__(
        self,
        in_dim: int = 10,
        hidden: int = 128,
        dropout: float = 0.1,
        depth: int = 3,
        n_heads: int = 4,
        ff_mult: int = 4,
    ):
        super().__init__()
        in_dim = int(in_dim)
        hidden = int(hidden)
        depth = int(depth)
        n_heads = int(n_heads)
        ff_mult = int(ff_mult)
        dropout = float(dropout)

        if in_dim < 1:
            raise ValueError("in_dim must be >= 1")
        if hidden < 1:
            raise ValueError("hidden must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if hidden % n_heads != 0:
            raise ValueError("hidden must be divisible by n_heads")
        if ff_mult < 1:
            raise ValueError("ff_mult must be >= 1")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0)")

        self.in_dim = in_dim
        self.hidden = hidden
        self.depth = depth
        self.n_heads = n_heads
        self.ff_mult = ff_mult

        self.node_embed = nn.Linear(2, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

        pair_in = (hidden * 3) + in_dim
        self.edge_head = nn.Sequential(
            nn.Linear(pair_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        self._edge_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def _complete_edges(self, n: int, dev: torch.device) -> torch.Tensor:
        key = (int(n), str(dev))
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached
        ij = torch.triu_indices(n, n, offset=1, device=dev)
        e = ij.t().contiguous()  # [m,2]
        self._edge_cache[key] = e
        return e

    @staticmethod
    def _n_from_m(m: int) -> int:
        disc = 1 + 8 * int(m)
        s = int(math.isqrt(disc))
        if s * s != disc:
            raise ValueError(f"Invalid edge count for complete graph: m={m}")
        n = (1 + s) // 2
        if n * (n - 1) // 2 != m:
            raise ValueError(f"Invalid edge count for complete graph: m={m}")
        return int(n)

    def forward(
        self,
        x: torch.Tensor,
        *,
        edge_counts: List[int] | None = None,
        coords: List[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected edge feature tensor [m, d], got shape={tuple(x.shape)}")
        if edge_counts is None or coords is None:
            raise ValueError("edge_transformer requires edge_counts and coords")
        if len(edge_counts) != len(coords):
            raise ValueError("edge_counts and coords length mismatch")

        outs: List[torch.Tensor] = []
        off = 0
        for m_raw, c in zip(edge_counts, coords):
            m = int(m_raw)
            n = int(c.shape[0])
            if n < 2:
                raise ValueError(f"Invalid n={n} for graph")
            m_expected = n * (n - 1) // 2
            if m != m_expected:
                # Fallback to edge-count inference for robustness.
                n_inferred = self._n_from_m(m)
                if n_inferred != n:
                    raise ValueError(f"edge/coords mismatch: m={m}, n={n}, inferred_n={n_inferred}")
            xb = x[off : off + m]
            off += m

            # [n, hidden]
            h = self.node_embed(c)
            h = self.encoder(h.unsqueeze(0)).squeeze(0)

            e = self._complete_edges(n, x.device)  # [m,2]
            hi = h.index_select(0, e[:, 0])
            hj = h.index_select(0, e[:, 1])
            pair = torch.cat([hi, hj, torch.abs(hi - hj), xb], dim=1)
            out = self.edge_head(pair).squeeze(-1)
            outs.append(out)

        if off != x.shape[0]:
            raise ValueError(f"Unconsumed edges: consumed={off}, total={x.shape[0]}")
        return torch.cat(outs, dim=0) if outs else x.new_zeros((0,))
