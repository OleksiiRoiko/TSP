from __future__ import annotations

import json
from pathlib import Path


def _resolve_embedded_path(anchor_file: Path, raw_path: str) -> Path:
    """
    Resolve a path stored inside latest.json/meta.json.

    We support two common styles:
    - project-root relative (e.g. runs/experiments/.../best.pt)
    - json-parent relative (e.g. 20260210-123456/best.pt or best.pt)
    """
    p = Path(str(raw_path))
    if p.is_absolute():
        return p

    # Prefer project-root-relative path first (train currently writes this style).
    direct = p.resolve()
    if direct.exists():
        return direct

    # Backward/alternative style: path relative to the json location.
    anchored = (anchor_file.parent / p).resolve()
    if anchored.exists():
        return anchored

    # Keep current working-dir semantics when target does not exist yet.
    return direct


def resolve_model_path(p: Path) -> Path:
    if p.is_dir():
        candidate = p / "latest.json"
        if candidate.exists():
            p = candidate
        else:
            runs = sorted([d for d in p.iterdir() if d.is_dir()])
            if runs:
                best = runs[-1] / "best.pt"
                if best.exists():
                    return best
    if p.suffix.lower() == ".json" and p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "path" in data:
                return _resolve_embedded_path(p, str(data["path"]))
        except Exception:
            pass
    if not p.exists() and p.name == "latest.json":
        exp_root = p.parent
        if exp_root.exists():
            runs = sorted([d for d in exp_root.iterdir() if d.is_dir()])
            if runs:
                best = runs[-1] / "best.pt"
                if best.exists():
                    return best
    return p


def infer_run_dir(model_path_cfg: Path, resolved_model: Path) -> Path | None:
    if model_path_cfg.suffix.lower() == ".json" and model_path_cfg.exists():
        try:
            data = json.loads(model_path_cfg.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if "run_dir" in data:
                    return _resolve_embedded_path(model_path_cfg, str(data["run_dir"]))
                if "path" in data:
                    return _resolve_embedded_path(model_path_cfg, str(data["path"])).parent
        except Exception:
            pass
    if model_path_cfg.is_dir():
        latest_json = model_path_cfg / "latest.json"
        if latest_json.exists():
            return infer_run_dir(latest_json, resolved_model)
    if resolved_model.exists():
        return resolved_model.parent
    return None


def dataset_tag(data_root: Path) -> str:
    s = str(data_root).lower()
    name = data_root.name.lower()
    if "tsplib" in s:
        return "tsplib"
    if "synthetic" in s and name in ("train", "val", "test"):
        return f"synthetic_{name}"
    return name or "data"
