from __future__ import annotations
from pathlib import Path
import time, json, os, sys
from contextlib import nullcontext
from typing import Any, Dict, cast
import numpy as np
import torch
try:
    # New API (PyTorch >= 2.0)
    from torch.amp.grad_scaler import GradScaler as _GradScaler
    from torch.amp.autocast_mode import autocast as _autocast
    _AMP_NEW = True
except Exception:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast
    _AMP_NEW = False
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from ..config import TrainCfg
from ..data.datasets import NPZTSPDataset, collate_edge_batches
from ..models.registry import build_model


def _bce_balanced(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Class-balanced BCE to cope with negative/positive edge imbalance."""
    pos = (targets == 1).float().sum().clamp(min=1.0)
    neg = (targets == 0).float().sum().clamp(min=1.0)
    wp = neg / (pos + neg)
    wn = pos / (pos + neg)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    w = torch.where(targets > 0.5, wp, wn)
    return (loss * w).mean()


def _maybe_prime_cache(ds: NPZTSPDataset, desc: str):
    """
    If on-disk feature cache is missing/incomplete, iterate once with a progress bar
    so users see visible work before the first epoch. Skips if already filled.
    """
    cache_root = getattr(ds, "cache_dir", None)
    if cache_root is None:
        return
    bucket = cache_root / str(ds.feature_dim) / "complete"
    bucket.mkdir(parents=True, exist_ok=True)
    cached = list(bucket.glob("*.npz"))
    if len(cached) >= len(ds):
        return
    for _ in tqdm(range(len(ds)), ncols=100, desc=f"{desc} | priming cache", leave=True):
        _ = ds[_]  # compute + save to cache


def _git_hash(repo_root: Path) -> str | None:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return None
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref: "):
        ref_path = repo_root / ".git" / ref.split(" ", 1)[1]
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
    return ref or None


def run(cfg: TrainCfg, logger, *, full_config: Dict[str, Any] | None = None, config_path: str | None = None):
    # ---- setup and validation ----
    train_root = Path(cfg.train_root)
    val_root = Path(cfg.val_root)
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError("train/val roots invalid")

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- datasets (complete graph, fixed 10-dim features) ----
    tr_ds = NPZTSPDataset(str(train_root), feature_dim=10)
    va_ds = NPZTSPDataset(str(val_root),   feature_dim=10)

    # optional: make the first-time compute visible
    _maybe_prime_cache(tr_ds, "train")
    _maybe_prime_cache(va_ds, "val")

    # ---- dataloaders (fast defaults) ----
    num_workers = max(0, min(4, (os.cpu_count() or 1) - 1))
    tr = DataLoader(
        tr_ds, batch_size=int(cfg.batch_size), shuffle=True,
        collate_fn=collate_edge_batches, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
    va = DataLoader(
        va_ds, batch_size=int(cfg.batch_size), shuffle=False,
        collate_fn=collate_edge_batches, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
    logger.info(f"Train graphs: {len(tr_ds)} | Val graphs: {len(va_ds)} | workers: {num_workers}")

    # ---- model ----
    overrides: dict[str, Any] = {"in_dim": 10}
    if cfg.hidden is not None:  overrides["hidden"]  = int(cfg.hidden)
    if cfg.dropout is not None: overrides["dropout"] = float(cfg.dropout)
    if hasattr(cfg, "depth") and cfg.depth is not None:  # depth from config.yaml
        overrides["depth"] = int(cfg.depth)

    model, mparams = build_model(cfg.model_name, overrides or None)
    model.to(device)
    logger.info(f"Model: {cfg.model_name} | Params: {mparams}")

    # ---- optim + AMP (new API) ----
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    use_amp = (device.type == "cuda")

    _GradScalerAny = cast(Any, _GradScaler)
    _autocastAny = cast(Any, _autocast)

    def _amp_setup(device_type: str, enabled: bool):
        if not enabled:
            return _GradScalerAny(enabled=False), nullcontext

        if _AMP_NEW:
            # GradScaler may or may not accept device_type depending on torch build.
            try:
                scaler = _GradScalerAny(device_type=device_type, enabled=True)
            except TypeError:
                scaler = _GradScalerAny(enabled=True)

            def _ctx():
                # torch.amp.autocast usually requires device_type positional.
                try:
                    return _autocastAny(device_type, enabled=True)
                except TypeError:
                    return _autocastAny(enabled=True)
        else:
            scaler = _GradScalerAny(enabled=True)

            def _ctx():
                return _autocastAny(enabled=True)

        return scaler, _ctx

    scaler, _amp_ctx = _amp_setup(device.type, use_amp)

    best = float("inf")
    best_ep = 0
    # run directory and artifacts
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs/experiments") / cfg.exp_id / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best.pt"
    meta_run = run_dir / "meta.json"
    latest_ptr = Path("runs/experiments") / cfg.exp_id / "latest.json"

    config_snapshot = run_dir / "config.yaml"
    if full_config:
        config_snapshot.write_text(
            yaml.safe_dump(full_config, sort_keys=False),
            encoding="utf-8",
        )

    meta_base = {
        "exp_id": cfg.exp_id,
        "model": cfg.model_name,
        "in_dim": mparams["in_dim"],
        "hidden": mparams["hidden"],
        "dropout": mparams["dropout"],
        "depth": mparams.get("depth"),
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "train_root": str(train_root),
        "val_root": str(val_root),
        "train_files": len(tr_ds),
        "val_files": len(va_ds),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "git_hash": _git_hash(Path(".")),
        "config_path": config_path,
        "config_snapshot": str(config_snapshot) if config_snapshot.exists() else None,
        "run_dir": str(run_dir),
    }

    # ---- epochs ----
    for ep in range(1, int(cfg.epochs) + 1):
        # TRAIN
        model.train(); tot = 0.0; cnt = 0
        with tqdm(total=len(tr), ncols=100, desc=f"epoch {ep:02d}/{cfg.epochs} [train]", leave=False) as pbar:
            for b in tr:
                if b["labels"] is None:
                    raise ValueError("training data missing labels")
                x = b["edge_feats"].to(device, non_blocking=True)
                y = b["labels"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with _amp_ctx():
                    logits = model(x)
                    loss = _bce_balanced(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += float(loss.item()) * y.numel(); cnt += y.numel()
                pbar.update(1)
        train_loss = tot / max(cnt, 1)

        # VAL
        model.eval(); vtot = 0.0; vcnt = 0
        with torch.no_grad(), tqdm(total=len(va), ncols=100, desc=f"epoch {ep:02d}/{cfg.epochs} [val]  ", leave=False) as pbar:
            for b in va:
                if b["labels"] is None:
                    raise ValueError("validation data missing labels")
                x = b["edge_feats"].to(device, non_blocking=True)
                y = b["labels"].to(device, non_blocking=True)
                with _amp_ctx():
                    logits = model(x)
                    loss = _bce_balanced(logits, y)
                vtot += float(loss.item()) * y.numel(); vcnt += y.numel()
                pbar.update(1)
        val_loss = vtot / max(vcnt, 1)

        # summary line (outside bars)
        tqdm.write(f"Epoch {ep:02d}/{cfg.epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best:
            best = val_loss; best_ep = ep
            torch.save(model.state_dict(), ckpt)
            payload = dict(meta_base)
            payload.update({
                "val_bce": round(float(val_loss), 6),
                "saved_at": stamp,
                "path": str(ckpt),
            })
            meta_run.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            latest_ptr.parent.mkdir(parents=True, exist_ok=True)
            latest_ptr.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tqdm.write(f"  SAVED -> {ckpt} (and updated {latest_ptr.name})")

    logger.info(f"Best Val {best:.6f} @ epoch {best_ep}\nSaved: {ckpt}")
