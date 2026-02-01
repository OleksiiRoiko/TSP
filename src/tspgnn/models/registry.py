from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path
import re
import torch

# Use the single, configurable-depth MLP
from .edge_mlp import EdgeMLPAny


# -----------------------------
# Public builders
# -----------------------------

def build_model(name: str, overrides: Dict[str, Any] | None = None) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Create a model by name, with sensible defaults, allowing overrides.

    Names (kept for backward compatibility):
      - "edge_mlp"      -> depth=2 hidden layers
      - "edge_mlp_deep" -> depth=3 hidden layers
      - "deep"          -> alias of edge_mlp_deep
    """
    name = (name or "edge_mlp").lower()
    params: Dict[str, Any] = dict(in_dim=10, hidden=128, dropout=0.0, depth=2)
    if name in ("edge_mlp_deep", "deep"):
        params.update(dict(hidden=256, dropout=0.1, depth=3))
    if overrides:
        params.update(overrides)

    model = EdgeMLPAny(**params)
    return model, params


def ensure_models_dir(path: str | Path = "runs/models") -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# -----------------------------
# Inference from checkpoint
# -----------------------------

def infer_edge_mlp_params_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Infer (model_name, in_dim, hidden, depth) from a saved state_dict produced by EdgeMLPAny.

    Assumptions:
      - The module stores layers in a nn.Sequential named 'net'
      - Linear weights appear as keys 'net.<idx>.weight'
      - The number of Linear layers = depth (hidden layers) + 1 (final 1-unit head)
    """
    linear_layers: list[tuple[int, torch.Size]] = []
    for k, v in state.items():
        m = re.match(r"^net\.(\d+)\.weight$", k)
        if m and v.ndim == 2:
            linear_layers.append((int(m.group(1)), v.shape))  # (index, (out, in))

    if not linear_layers:
        # Fallback: find any 2D weight and guess
        for v in state.values():
            if hasattr(v, "shape") and isinstance(v.shape, torch.Size) and len(v.shape) == 2:
                out_f, in_f = int(v.shape[0]), int(v.shape[1])
                # conservative defaults
                return {"model_name": "edge_mlp_deep", "in_dim": in_f, "hidden": out_f, "depth": 3, "dropout": 0.1}
        # Nothing reasonable found
        return {"model_name": "edge_mlp_deep", "in_dim": 10, "hidden": 256, "depth": 3, "dropout": 0.1}

    linear_layers.sort(key=lambda x: x[0])
    first_out, first_in = linear_layers[0][1]
    num_linear = len(linear_layers)          # = depth_hidden + 1 (final head)
    depth_hidden = max(1, num_linear - 1)    # at least 1 hidden layer

    model_name = "edge_mlp_deep" if depth_hidden >= 3 else "edge_mlp"
    dropout = 0.1 if depth_hidden >= 3 else 0.0

    return {
        "model_name": model_name,
        "in_dim": int(first_in),
        "hidden": int(first_out),
        "depth": int(depth_hidden),
        "dropout": float(dropout),
    }


def build_model_from_state(
    state: Dict[str, torch.Tensor],
    prefer_name: str | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Build EdgeMLPAny to match a checkpoint. You can override any inferred param (e.g., in_dim).
    """
    inferred = infer_edge_mlp_params_from_state(state)

    # Start from inferred params
    params: Dict[str, Any] = dict(
        in_dim=inferred["in_dim"],
        hidden=inferred["hidden"],
        dropout=inferred["dropout"],
        depth=inferred["depth"],
    )

    # Respect explicit overrides
    if overrides:
        params.update(overrides)

    # Name is mostly informational now; pick prefer_name if provided
    depth = int(params.get("depth", inferred["depth"]))
    name = (prefer_name or ("edge_mlp_deep" if depth >= 3 else "edge_mlp")).lower()

    model = EdgeMLPAny(**params)
    return model, dict(model_name=name, **params)


# -----------------------------
# Flexible weight loading
# -----------------------------

def load_weights_flex(model: torch.nn.Module, state: Dict[str, torch.Tensor], logger=None) -> None:
    """
    Load only parameters whose names AND shapes match; skip the rest.
    Prevents size-mismatch crashes even with strict=False.
    """
    cur = model.state_dict()
    keep: Dict[str, torch.Tensor] = {}
    skipped = []

    for k, v in state.items():
        if k in cur and hasattr(v, "shape") and hasattr(cur[k], "shape") and tuple(v.shape) == tuple(cur[k].shape):
            keep[k] = v
        else:
            skipped.append(k)

    msg = f"Loading {len(keep)}/{len(state)} tensors (filtered by name+shape)."
    if logger:
        logger.info(msg)
    else:
        print(msg)

    missing, unexpected = model.load_state_dict(keep, strict=False)
    if logger:
        if missing:
            logger.info(f"Missing keys (not in checkpoint or filtered): {len(missing)}")
        if unexpected:
            logger.info(f"Unexpected keys (ignored): {len(unexpected)}")
    else:
        if missing:
            print("Missing keys:", len(missing))
        if unexpected:
            print("Unexpected keys:", len(unexpected))
