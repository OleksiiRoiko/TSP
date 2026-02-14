from __future__ import annotations
from pathlib import Path
import time, json, os, sys
from contextlib import nullcontext
from typing import Any, Dict, cast
import csv
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
    pos = ((targets == 1).float().sum()).clamp(min=1.0)
    neg = ((targets == 0).float().sum()).clamp(min=1.0)
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


def _forward_logits(model: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    x = batch["edge_feats"].to(device, non_blocking=True)
    if bool(getattr(model, "requires_graph_context", False)):
        edge_counts = batch.get("edge_counts", None)
        coords = batch.get("coords", None)
        if edge_counts is None or coords is None:
            raise ValueError("Model requires graph context, but batch is missing edge_counts/coords")
        coords_dev = [c.to(device, non_blocking=True) for c in coords]
        return model(x, edge_counts=edge_counts, coords=coords_dev)
    return model(x)


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
    if device.type == "cuda":
        # Use fast Tensor Core math where possible (small numeric drift is expected).
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ---- datasets (complete graph, fixed 10-dim features) ----
    tr_ds = NPZTSPDataset(str(train_root), feature_dim=10)
    va_ds = NPZTSPDataset(str(val_root),   feature_dim=10)

    # optional: make the first-time compute visible
    _maybe_prime_cache(tr_ds, "train")
    _maybe_prime_cache(va_ds, "val")

    # ---- dataloaders (fast defaults) ----
    # use maximum available CPU workers, but fall back if multiprocessing is blocked
    def _make_loaders(nw: int):
        tr_loader = DataLoader(
            tr_ds, batch_size=int(cfg.batch_size), shuffle=True,
            collate_fn=collate_edge_batches, num_workers=nw,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=2 if nw > 0 else None,
            persistent_workers=(nw > 0),
        )
        va_loader = DataLoader(
            va_ds, batch_size=int(cfg.batch_size), shuffle=False,
            collate_fn=collate_edge_batches, num_workers=nw,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=2 if nw > 0 else None,
            persistent_workers=(nw > 0),
        )
        return tr_loader, va_loader

    num_workers = max(0, min(8, (os.cpu_count() or 1)))
    tr, va = _make_loaders(num_workers)

    def _iter_loader(kind: str):
        nonlocal num_workers, tr, va
        while True:
            loader = tr if kind == "train" else va
            try:
                return iter(loader)
            except PermissionError:
                if num_workers == 0:
                    raise
                logger.warning("DataLoader multiprocessing blocked; falling back to num_workers=0")
                num_workers = 0
                tr, va = _make_loaders(num_workers)

    logger.info(f"Train graphs: {len(tr_ds)} | Val graphs: {len(va_ds)} | workers: {num_workers}")

    # ---- model ----
    overrides: dict[str, Any] = {"in_dim": 10}
    if cfg.hidden is not None:  overrides["hidden"]  = int(cfg.hidden)
    if cfg.dropout is not None: overrides["dropout"] = float(cfg.dropout)
    if hasattr(cfg, "depth") and cfg.depth is not None:  # depth from train config
        overrides["depth"] = int(cfg.depth)
    if getattr(cfg, "n_heads", None) is not None:
        overrides["n_heads"] = int(getattr(cfg, "n_heads"))
    if getattr(cfg, "ff_mult", None) is not None:
        overrides["ff_mult"] = int(getattr(cfg, "ff_mult"))
    if getattr(cfg, "edge_feat_mode", None) is not None:
        overrides["edge_feat_mode"] = str(getattr(cfg, "edge_feat_mode")).lower()

    model, mparams = build_model(cfg.model_name, overrides or None)
    model = cast(torch.nn.Module, model)
    model.to(device)
    logger.info(f"Model: {cfg.model_name} | Params: {mparams}")

    # ---- optim + AMP (new API) ----
    lr_value = float(cfg.lr)
    wd_value = float(getattr(cfg, "weight_decay", 0.0))
    use_fused_opt = (device.type == "cuda")
    if use_fused_opt:
        try:
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=lr_value,
                weight_decay=wd_value,
                fused=True,
            )
            logger.info("Optimizer: AdamW(fused=True)")
        except Exception:
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=lr_value,
                weight_decay=wd_value,
            )
            logger.info("Optimizer: AdamW(fused unsupported, using default)")
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr_value,
            weight_decay=wd_value,
        )
        logger.info("Optimizer: AdamW")
    sched_name = str(getattr(cfg, "lr_scheduler", "none") or "none").lower()
    scheduler = None
    if sched_name not in ("none", "", "null"):
        if sched_name in ("plateau", "reduce_on_plateau", "rop"):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=float(getattr(cfg, "lr_factor", 0.5)),
                patience=int(getattr(cfg, "lr_patience", 5)),
                min_lr=float(getattr(cfg, "lr_min", 1e-6)),
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {sched_name}")
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
    val_every = max(1, int(getattr(cfg, "val_every", 1)))
    early_stop = bool(getattr(cfg, "early_stop", False))
    early_patience = int(getattr(cfg, "early_patience", 10))
    early_min_delta = float(getattr(cfg, "early_min_delta", 0.0))
    no_improve = 0
    # run directory and artifacts
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs/experiments") / cfg.exp_id / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best.pt"
    meta_run = run_dir / "meta.json"
    latest_ptr = Path("runs/experiments") / cfg.exp_id / "latest.json"
    history_csv = run_dir / "train_history.csv"

    config_snapshot = run_dir / "config.yaml"
    if full_config:
        config_snapshot.write_text(
            yaml.safe_dump(full_config, sort_keys=False),
            encoding="utf-8",
        )

    meta_base = {
        "exp_id": cfg.exp_id,
        "model": cfg.model_name,
        "model_params": dict(mparams),
        "in_dim": mparams["in_dim"],
        "hidden": mparams["hidden"],
        "dropout": mparams["dropout"],
        "depth": mparams.get("depth"),
        "n_heads": mparams.get("n_heads"),
        "ff_mult": mparams.get("ff_mult"),
        "edge_feat_mode": mparams.get("edge_feat_mode"),
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": getattr(cfg, "weight_decay", 0.0),
        "lr_scheduler": sched_name,
        "lr_factor": getattr(cfg, "lr_factor", 0.5),
        "lr_patience": getattr(cfg, "lr_patience", 5),
        "lr_min": getattr(cfg, "lr_min", 1e-6),
        "val_every": val_every,
        "early_stop": early_stop,
        "early_patience": early_patience,
        "early_min_delta": early_min_delta,
        "tf32": (device.type == "cuda"),
        "fused_optimizer": use_fused_opt,
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
        "history_csv": str(history_csv),
    }

    # ---- epochs ----
    # prepare history log
    with history_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])

    for ep in range(1, int(cfg.epochs) + 1):
        ep_start = time.time()
        # TRAIN
        model.train(); tot = 0.0; cnt = 0
        with tqdm(total=len(tr), ncols=100, desc=f"epoch {ep:02d}/{cfg.epochs} [train]", leave=False) as pbar:
            for b in _iter_loader("train"):
                if b["labels"] is None:
                    raise ValueError("training data missing labels")
                y = b["labels"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with _amp_ctx():
                    logits = _forward_logits(model, b, device)
                    loss = _bce_balanced(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += float(loss.item()) * y.numel(); cnt += y.numel()
                pbar.update(1)
        train_loss = tot / max(cnt, 1)

        do_val = (ep % val_every == 0) or (ep == int(cfg.epochs))
        val_loss = float("nan")
        if do_val:
            model.eval(); vtot = 0.0; vcnt = 0
            with torch.no_grad(), tqdm(total=len(va), ncols=100, desc=f"epoch {ep:02d}/{cfg.epochs} [val]  ", leave=False) as pbar:
                for b in _iter_loader("val"):
                    if b["labels"] is None:
                        raise ValueError("validation data missing labels")
                    y = b["labels"].to(device, non_blocking=True)
                    with _amp_ctx():
                        logits = _forward_logits(model, b, device)
                        loss = _bce_balanced(logits, y)
                    vtot += float(loss.item()) * y.numel(); vcnt += y.numel()
                    pbar.update(1)
            val_loss = vtot / max(vcnt, 1)
        ep_time = time.time() - ep_start

        # scheduler step (after val)
        cur_lr = float(opt.param_groups[0]["lr"])
        if scheduler is not None and do_val:
            prev_lr = cur_lr
            scheduler.step(val_loss)
            cur_lr = float(opt.param_groups[0]["lr"])
            if cur_lr < prev_lr - 1e-12:
                logger.info(f"LR reduced: {prev_lr:.2e} -> {cur_lr:.2e}")

        # append history row
        with history_csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                ep,
                round(train_loss, 6),
                "" if not do_val else round(val_loss, 6),
                f"{cur_lr:.6e}",
                round(ep_time, 3),
            ])

        # summary line (outside bars)
        if do_val:
            tqdm.write(f"Epoch {ep:02d}/{cfg.epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | LR {cur_lr:.2e}")
        else:
            tqdm.write(f"Epoch {ep:02d}/{cfg.epochs} | Train {train_loss:.4f} | Val skipped | LR {cur_lr:.2e}")

        if do_val:
            improved = val_loss < (best - early_min_delta)
            if improved:
                best = val_loss; best_ep = ep
                no_improve = 0
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
            else:
                no_improve += 1

            if early_stop and no_improve >= early_patience:
                logger.info(f"Early stop at epoch {ep} (no improvement for {no_improve} validation checks).")
                break

    logger.info(f"Best Val {best:.6f} @ epoch {best_ep}\nSaved: {ckpt}")
