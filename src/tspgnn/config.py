from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, field_validator

from .models.names import allowed_model_names, canonical_model_name


# -----------------------------
# Config schemas (lean version)
# -----------------------------

class GenerateCfg(BaseModel):
    out_root: str = "runs/data/synthetic"
    n_list: List[int] = [20, 30, 40, 50]
    per_size_train: int = 2000
    per_size_val: int = 400
    per_size_test: int = 400
    # Generation workers (None -> auto cap)
    workers: Optional[int] = None
    # Tour solver: "auto" (elkai if available, else NN+2opt),
    # "concorde" (exact, requires concorde in PATH), "elkai", or "nn2opt".
    tour_solver: str = "auto"
    elkai_frac: float = 0.2
    # Concorde options (used when tour_solver="concorde")
    concorde_cmd: str = "concorde"
    concorde_scale: int = 10000
    concorde_timeout_sec: int = 60
    concorde_keep_tmp: bool = False
    # If true, fail generation when Concorde output does not confirm optimal proof.
    concorde_require_optimal_proof: bool = False
    dist_names: List[str] = ["uniform", "clustered", "ring", "grid_jitter"]
    dist_probs: List[float] = [0.4, 0.2, 0.2, 0.2]
    seed: int = 0

    @field_validator("n_list")
    @classmethod
    def _n_ok(cls, v: List[int]) -> List[int]:
        if not all(isinstance(x, int) and x >= 3 for x in v):
            raise ValueError("n_list must contain ints >= 3")
        return v

    @field_validator("dist_probs")
    @classmethod
    def _probs_ok(cls, v: List[float]) -> List[float]:
        if not all(p >= 0 for p in v):
            raise ValueError("dist_probs must be non-negative")
        return v

    @field_validator("tour_solver")
    @classmethod
    def _solver_ok(cls, v: str) -> str:
        v = str(v).lower()
        if v not in ("auto", "concorde", "elkai", "nn2opt"):
            raise ValueError("tour_solver must be auto|concorde|elkai|nn2opt")
        return v

    @field_validator("concorde_scale")
    @classmethod
    def _scale_ok(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError("concorde_scale must be >= 1")
        return int(v)

    @field_validator("workers")
    @classmethod
    def _workers_ok(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if int(v) < 1:
            raise ValueError("generate.workers must be >= 1 when set")
        return int(v)


class TsplibCfg(BaseModel):
    raw_root: str = "runs/data/tsplib/raw"
    out_root: str = "runs/data/tsplib/processed"
    download: bool = True
    names: List[str] = [
        "ulysses16", "ulysses22", "att48", "eil51", "berlin52", "st70",
        "eil76", "pr76", "gr96", "kroA100", "kroC100", "kroD100", "rd100", "lin105",
    ]


class TrainCfg(BaseModel):
    exp_id: str = "edge_mlp_v1"
    train_root: str = "runs/data/synthetic/train"
    val_root: str = "runs/data/synthetic/val"
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    # Learning-rate scheduler (optional)
    lr_scheduler: str = "plateau"  # "none" | "plateau"
    lr_factor: float = 0.6
    lr_patience: int = 2
    lr_min: float = 1e-6
    # Run validation every N epochs (always runs on final epoch)
    val_every: int = 1
    # Early stopping enabled by default:
    # stop when validation does not improve for `early_patience` checks.
    early_stop: bool = True
    # Number of consecutive validation checks without improvement before stop.
    early_patience: int = 2
    early_min_delta: float = 0.0
    seed: int = 0
    model_name: str = "edge_mlp"   # from registry
    hidden: Optional[int] = 128
    dropout: Optional[float] = 0.0
    depth: int = 2   # NEW: number of hidden layers

    @field_validator("model_name")
    @classmethod
    def _model_name_ok(cls, v: str) -> str:
        vv = str(v).lower()
        try:
            canonical_model_name(vv)
        except ValueError:
            raise ValueError(f"model_name must be one of {allowed_model_names()}")
        return vv

    @field_validator("lr_scheduler")
    @classmethod
    def _lr_sched_ok(cls, v: str) -> str:
        vv = str(v).lower()
        allowed = {"none", "", "null", "plateau", "reduce_on_plateau", "rop"}
        if vv not in allowed:
            raise ValueError(f"lr_scheduler must be one of {sorted(allowed)}")
        return vv

class EvalCfg(BaseModel):
    model_path: str = "runs/experiments/edge_mlp_v1/latest.json"
    data_roots: List[str] = ["runs/data/tsplib/processed"]
    device: str = "cpu"
    save_json: Optional[str] = "auto"
    save_pred_tour: bool = False
    run_twoopt: bool = True
    seed: int = 0


class VisualizeTargetCfg(BaseModel):
    npz_dir: str
    limit: Optional[int] = None
    out_dir: Optional[str] = None
    mode: Optional[str] = None


class VisualizeCfg(BaseModel):
    model: str = "runs/experiments/edge_mlp_v1/latest.json"  # only used in predict mode
    out_dir: str = "auto"
    figsize: List[float] = [11.0, 5.5]
    dpi: int = 150
    device: str = "cpu"
    targets: List[VisualizeTargetCfg] = []


class QACfg(BaseModel):
    root: str = "runs/data"
    check_gt: bool = True
    lengths: bool = True
    # Detect duplicate coordinate sets across train/val/test splits.
    check_split_overlap: bool = True
    # If set, enforce that synthetic files have this label_source (e.g., "concorde").
    require_label_source: Optional[str] = None
    # Validate concorde-specific fields/lengths when label_source == "concorde".
    check_concorde: bool = True
    # Require concorde_optimal_proved == true for concorde-labeled rows.
    check_concorde_optimal_proof: bool = False
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


def load_config_data(path: str | Path) -> Dict[str, Any]:
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


def load_config(path: str | Path) -> AppCfg:
    data = load_config_data(path)
    return AppCfg(**(data or {}))

