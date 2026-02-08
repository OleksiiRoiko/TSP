from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import torch

from .edge_mlp import EdgeMLPAny
from .edge_res_mlp import EdgeResMLP


def build_model(name: str, overrides: Dict[str, Any] | None = None) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Create a model by name with defaults, then apply overrides.

    Supported names:
    - edge_mlp
    - edge_mlp_deep (alias: deep)
    - edge_res_mlp (aliases: edge_res, res_mlp, res)
    """
    name = (name or "edge_mlp").lower()

    if name in ("edge_mlp_deep", "deep"):
        canonical = "edge_mlp_deep"
        params: Dict[str, Any] = dict(in_dim=10, hidden=256, dropout=0.1, depth=3)
        cls = EdgeMLPAny
    elif name in ("edge_res_mlp", "edge_res", "res_mlp", "res"):
        canonical = "edge_res_mlp"
        params = dict(in_dim=10, hidden=128, dropout=0.1, depth=4)
        cls = EdgeResMLP
    else:
        canonical = "edge_mlp"
        params = dict(in_dim=10, hidden=128, dropout=0.0, depth=2)
        cls = EdgeMLPAny

    if overrides:
        params.update(overrides)

    model = cls(**params)
    return model, dict(model_name=canonical, **params)


def _infer_edge_res_params_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any] | None:
    input_w = state.get("input.weight", None)
    head_w = state.get("head.weight", None)
    if input_w is None or head_w is None:
        return None
    if input_w.ndim != 2 or head_w.ndim != 2:
        return None

    in_dim = int(input_w.shape[1])
    hidden = int(input_w.shape[0])
    depth = 0
    for k in state.keys():
        m = re.match(r"^blocks\.(\d+)\.fc1\.weight$", k)
        if m:
            depth = max(depth, int(m.group(1)) + 1)
    depth = max(1, depth)

    return {
        "model_name": "edge_res_mlp",
        "in_dim": in_dim,
        "hidden": hidden,
        "depth": depth,
        "dropout": 0.1,
    }


def _infer_edge_mlp_params_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    linear_layers: list[tuple[int, torch.Size]] = []
    for k, v in state.items():
        m = re.match(r"^net\.(\d+)\.weight$", k)
        if m and v.ndim == 2:
            linear_layers.append((int(m.group(1)), v.shape))

    if not linear_layers:
        for v in state.values():
            if hasattr(v, "shape") and isinstance(v.shape, torch.Size) and len(v.shape) == 2:
                out_f, in_f = int(v.shape[0]), int(v.shape[1])
                return {
                    "model_name": "edge_mlp_deep",
                    "in_dim": in_f,
                    "hidden": out_f,
                    "depth": 3,
                    "dropout": 0.1,
                }
        return {"model_name": "edge_mlp_deep", "in_dim": 10, "hidden": 256, "depth": 3, "dropout": 0.1}

    linear_layers.sort(key=lambda x: x[0])
    first_out, first_in = linear_layers[0][1]
    num_linear = len(linear_layers)
    depth_hidden = max(1, num_linear - 1)
    model_name = "edge_mlp_deep" if depth_hidden >= 3 else "edge_mlp"
    dropout = 0.1 if depth_hidden >= 3 else 0.0

    return {
        "model_name": model_name,
        "in_dim": int(first_in),
        "hidden": int(first_out),
        "depth": int(depth_hidden),
        "dropout": float(dropout),
    }


def infer_model_params_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    # Detect residual architecture first (keys are distinctive).
    edge_res = _infer_edge_res_params_from_state(state)
    if edge_res is not None:
        return edge_res
    # Otherwise fall back to legacy edge_mlp inference.
    return _infer_edge_mlp_params_from_state(state)


def build_model_from_state(
    state: Dict[str, torch.Tensor],
    prefer_name: str | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    inferred = infer_model_params_from_state(state)

    params: Dict[str, Any] = dict(
        in_dim=inferred["in_dim"],
        hidden=inferred["hidden"],
        dropout=inferred["dropout"],
        depth=inferred["depth"],
    )
    if overrides:
        params.update(overrides)

    model_name = (prefer_name or str(inferred["model_name"])).lower()
    model, built = build_model(model_name, params)
    return model, built


def load_weights_flex(model: torch.nn.Module, state: Dict[str, torch.Tensor], logger=None) -> None:
    """
    Load only parameters whose names and shapes match.
    """
    cur = model.state_dict()
    keep: Dict[str, torch.Tensor] = {}

    for k, v in state.items():
        if k in cur and hasattr(v, "shape") and hasattr(cur[k], "shape") and tuple(v.shape) == tuple(cur[k].shape):
            keep[k] = v

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
