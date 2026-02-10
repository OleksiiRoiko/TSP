from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import torch

from .edge_mlp import EdgeMLPAny
from .edge_res_mlp import EdgeResMLP
from .edge_transformer import EdgeTransformer
from .names import canonical_model_name


def build_model(name: str, overrides: Dict[str, Any] | None = None) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Create a model by name with defaults, then apply overrides.

    Supported names:
    - edge_mlp
    - edge_mlp_deep (alias: deep)
    - edge_res_mlp (aliases: edge_res, res_mlp, res)
    - edge_transformer (aliases: edge_tf, transformer)
    """
    canonical = canonical_model_name(name)

    if canonical == "edge_mlp_deep":
        canonical = "edge_mlp_deep"
        params: Dict[str, Any] = dict(in_dim=10, hidden=256, dropout=0.1, depth=3)
        cls = EdgeMLPAny
    elif canonical == "edge_transformer":
        canonical = "edge_transformer"
        params = dict(in_dim=10, hidden=128, dropout=0.1, depth=3, n_heads=4, ff_mult=4)
        cls = EdgeTransformer
    elif canonical == "edge_res_mlp":
        canonical = "edge_res_mlp"
        params = dict(in_dim=10, hidden=128, dropout=0.1, depth=4)
        cls = EdgeResMLP
    elif canonical == "edge_mlp":
        canonical = "edge_mlp"
        params = dict(in_dim=10, hidden=128, dropout=0.0, depth=2)
        cls = EdgeMLPAny
    else:
        raise ValueError(f"Unknown canonical model '{canonical}'.")

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


def _infer_edge_transformer_params_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any] | None:
    node_w = state.get("node_embed.weight", None)
    head0_w = state.get("edge_head.0.weight", None)
    if node_w is None or head0_w is None:
        return None
    if node_w.ndim != 2 or head0_w.ndim != 2:
        return None

    hidden = int(node_w.shape[0])
    pair_in = int(head0_w.shape[1])
    in_dim = pair_in - (hidden * 3)
    if in_dim < 1:
        return None

    depth = 0
    for k in state.keys():
        m = re.match(r"^encoder\.layers\.(\d+)\.self_attn\.in_proj_weight$", k)
        if m:
            depth = max(depth, int(m.group(1)) + 1)
    depth = max(1, depth)

    return {
        "model_name": "edge_transformer",
        "in_dim": in_dim,
        "hidden": hidden,
        "depth": depth,
        "dropout": 0.1,
        "n_heads": 4,
        "ff_mult": 4,
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
    # Detect graph transformer first (distinctive keys).
    edge_tf = _infer_edge_transformer_params_from_state(state)
    if edge_tf is not None:
        return edge_tf
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

    params: Dict[str, Any] = {k: v for k, v in inferred.items() if k != "model_name"}
    if overrides:
        params.update(overrides)

    model_name = (prefer_name or str(inferred["model_name"])).lower()
    model, built = build_model(model_name, params)
    return model, built


def load_weights_flex(
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    logger=None,
    *,
    require_all_matched: bool = False,
) -> None:
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

    if len(keep) == 0:
        raise ValueError("No matching tensors found between checkpoint and model state.")

    missing, unexpected = model.load_state_dict(keep, strict=False)
    if require_all_matched and (missing or unexpected):
        raise ValueError(
            f"Checkpoint/model mismatch: missing={len(missing)} unexpected={len(unexpected)}. "
            "Refusing partial load in strict mode."
        )
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
