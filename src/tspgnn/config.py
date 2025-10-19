from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, field_validator


# -----------------------------
# Config schemas (lean version)
# -----------------------------

class GenerateCfg(BaseModel):
    out_root: str = "runs/data/synthetic"
    n_list: List[int] = [20, 30, 40, 50]
    per_size_train: int = 100
    per_size_val: int = 20
    per_size_test: int = 20
    elkai_frac: float = 0.1
    seed: int = 0

    @field_validator("n_list")
    @classmethod
    def _n_ok(cls, v: List[int]) -> List[int]:
        assert all(isinstance(x, int) and x >= 3 for x in v), "n_list must contain ints >= 3"
        return v


class TsplibCfg(BaseModel):
    raw_root: str = "runs/data/tsplib/raw"
    out_root: str = "runs/data/tsplib/processed"
    download: bool = True
    names: List[str] = ["berlin52", "eil76"]


class TrainCfg(BaseModel):
    exp_id: str = "edge_mlp_v1"
    train_root: str = "runs/data/synthetic/train"
    val_root: str = "runs/data/synthetic/val"
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    seed: int = 0
    model_name: str = "edge_mlp_deep"   # from registry
    hidden: Optional[int] = 256
    dropout: Optional[float] = 0.1
    feature_dim: int = 10               # any int; features are padded/truncated
    depth: int = 3   # NEW: number of hidden layers


class EvalCfg(BaseModel):
    model_path: str = "runs/models/edge_mlp_v1_best.pt"
    data_root: str = "runs/data/tsplib/processed"
    device: str = "cpu"
    save_json: Optional[str] = "runs/evals/eval.json"
    run_twoopt: bool = True
    feature_dim: Optional[int] = None   # None → infer from checkpoint; else any int
    seed: int = 0


class VisualizeCfg(BaseModel):
    mode: str = "predict"               # "dataset" or "predict"
    npz_dir: str = "runs/data/tsplib/processed"
    model: str = "runs/models/edge_mlp_v1_best.pt"  # only used in predict mode
    out_dir: str = "runs/figs/tsplib"
    figsize: List[float] = [11.0, 5.5]
    dpi: int = 150
    feature_dim: Optional[int] = None   # None → infer from checkpoint
    device: str = "cpu"
    limit: int = 0


class QACfg(BaseModel):
    root: str = "runs/data/tsplib/processed"
    check_gt: bool = True
    coverage: bool = True
    k: int = 20                         # k used for QA coverage only
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

def load_config(path: str | Path = "config.yaml") -> AppCfg:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return AppCfg(**(data or {}))
