from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, field_validator


# -----------------------------
# Config schemas (lean version)
# -----------------------------

class GenerateCfg(BaseModel):
    out_root: str = "runs/data/synthetic"
    n_list: List[int] = [20, 30, 40, 50]
    per_size_train: int = 2000
    per_size_val: int = 400
    per_size_test: int = 400
    elkai_frac: float = 0.2
    dist_names: List[str] = ["uniform", "clustered", "ring", "grid_jitter"]
    dist_probs: List[float] = [0.4, 0.2, 0.2, 0.2]
    seed: int = 0

    @field_validator("n_list")
    @classmethod
    def _n_ok(cls, v: List[int]) -> List[int]:
        assert all(isinstance(x, int) and x >= 3 for x in v), "n_list must contain ints >= 3"
        return v

    @field_validator("dist_probs")
    @classmethod
    def _probs_ok(cls, v: List[float]) -> List[float]:
        assert all(p >= 0 for p in v), "dist_probs must be non-negative"
        return v


class TsplibCfg(BaseModel):
    raw_root: str = "runs/data/tsplib/raw"
    out_root: str = "runs/data/tsplib/processed"
    download: bool = True
    names: List[str] = ["berlin52", "eil76", "pr76", "kroA100"]


class TrainCfg(BaseModel):
    exp_id: str = "edge_mlp_v1"
    train_root: str = "runs/data/synthetic/train"
    val_root: str = "runs/data/synthetic/val"
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 0
    model_name: str = "edge_mlp"   # from registry
    hidden: Optional[int] = 128
    dropout: Optional[float] = 0.0
    depth: int = 2   # NEW: number of hidden layers


class EvalCfg(BaseModel):
    model_path: str = "runs/experiments/edge_mlp_v1/latest.json"
    data_root: str = "runs/data/tsplib/processed"
    data_roots: Optional[List[str]] = None
    device: str = "cpu"
    save_json: Optional[str] = "auto"
    run_twoopt: bool = True
    seed: int = 0


class VisualizeTargetCfg(BaseModel):
    npz_dir: str
    limit: Optional[int] = None
    out_dir: Optional[str] = None
    mode: Optional[str] = None


class VisualizeCfg(BaseModel):
    mode: str = "predict"               # "dataset" or "predict"
    npz_dir: str = "runs/data/tsplib/processed"
    model: str = "runs/experiments/edge_mlp_v1/latest.json"  # only used in predict mode
    out_dir: str = "auto"
    figsize: List[float] = [11.0, 5.5]
    dpi: int = 150
    device: str = "cpu"
    limit: int = 0
    targets: Optional[List[VisualizeTargetCfg]] = None


class QACfg(BaseModel):
    root: str = "runs/data"
    check_gt: bool = True
    coverage: bool = True
    lengths: bool = True
    csv: Optional[str] = "runs/qa_report.csv"


class AppCfg(BaseModel):
    generate: GenerateCfg = GenerateCfg()
    tsplib: TsplibCfg = TsplibCfg()
    train: TrainCfg = TrainCfg()
    eval: EvalCfg = EvalCfg()
    visualize: VisualizeCfg = VisualizeCfg()
    qa: QACfg = QACfg()


# -----------------------------
# Loader
# -----------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _replace_placeholders(data: Any, exp_id: str | None) -> Any:
    if exp_id is None:
        return data
    if isinstance(data, dict):
        return {k: _replace_placeholders(v, exp_id) for k, v in data.items()}
    if isinstance(data, list):
        return [_replace_placeholders(v, exp_id) for v in data]
    if isinstance(data, str):
        return data.replace("{exp_id}", exp_id)
    return data


def load_config_data(path: str | Path = "config.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    base_path = data.pop("base", None)
    if base_path:
        base_file = (p.parent / str(base_path)).resolve()
        if not base_file.exists():
            raise FileNotFoundError(f"Base config not found: {base_file}")
        base_data = yaml.safe_load(base_file.read_text(encoding="utf-8")) or {}
        merged = _deep_merge(base_data, data)
    else:
        merged = data
    exp_id = None
    if isinstance(merged, dict):
        exp_id = merged.get("train", {}).get("exp_id")
    return _replace_placeholders(merged, exp_id)


def load_config(path: str | Path = "config.yaml") -> AppCfg:
    data = load_config_data(path)
    return AppCfg(**(data or {}))
