from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    # Prefer weights_only for safer deserialization when supported.
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain a state_dict mapping: {path}")
    return state


def predict_logits(
    model: torch.nn.Module,
    edge_feats: np.ndarray,
    coords: np.ndarray,
    dev: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(edge_feats).float().to(dev)
    if bool(getattr(model, "requires_graph_context", False)):
        c = torch.from_numpy(coords).float().to(dev)
        y = model(x, edge_counts=[int(x.shape[0])], coords=[c])
    else:
        y = model(x)
    return y.detach().cpu().numpy()
