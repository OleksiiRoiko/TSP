from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..utils.run_paths import resolve_model_path
from .registry import build_model_from_state, load_weights_flex


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


def _extract_model_overrides(meta_path: Path) -> tuple[str | None, dict | None]:
    if not meta_path.exists():
        return None, None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None

    model_params = data.get("model_params", None)
    if not isinstance(model_params, dict):
        return None, None

    params = dict(model_params)
    prefer_name_raw = params.pop("model_name", data.get("model", None))
    prefer_name = str(prefer_name_raw).lower() if prefer_name_raw else None
    return prefer_name, (params or None)


def model_overrides_from_metadata(model_path_cfg: Path, resolved_model_path: Path) -> tuple[str | None, dict | None]:
    if model_path_cfg.suffix.lower() == ".json":
        pref, ov = _extract_model_overrides(model_path_cfg)
        if pref is not None or ov is not None:
            return pref, ov

    if resolved_model_path.exists():
        pref, ov = _extract_model_overrides(resolved_model_path.parent / "meta.json")
        if pref is not None or ov is not None:
            return pref, ov

    return None, None


def load_model_for_inference(
    model_path_cfg: Path,
    *,
    logger=None,
    require_all_matched: bool = True,
) -> tuple[torch.nn.Module, dict[str, object], Path]:
    mp = resolve_model_path(model_path_cfg)
    if not mp.exists():
        raise FileNotFoundError(f"model_path invalid: {model_path_cfg}")

    state = load_state_dict(mp)
    prefer_name, overrides = model_overrides_from_metadata(model_path_cfg, mp)
    model, mparams = build_model_from_state(state, prefer_name=prefer_name, overrides=overrides)
    load_weights_flex(model, state, logger=logger, require_all_matched=require_all_matched)
    return model, mparams, mp


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
