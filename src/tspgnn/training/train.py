from __future__ import annotations
from pathlib import Path
import time, json, shutil, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainCfg
from ..data.datasets import NPZTSPDataset, collate_edge_batches
from ..models.registry import build_model, ensure_models_dir


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
    bucket = cache_root / str(ds.feature_dim) / ds.candidate_mode
    bucket.mkdir(parents=True, exist_ok=True)
    cached = list(bucket.glob("*.npz"))
    if len(cached) >= len(ds):
        return
    for _ in tqdm(range(len(ds)), ncols=100, desc=f"{desc} | priming cache", leave=True):
        _ = ds[_]  # compute + save to cache


def run(cfg: TrainCfg, logger):
    # ---- setup and validation ----
    train_root = Path(cfg.train_root)
    val_root = Path(cfg.val_root)
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError("train/val roots invalid")

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- datasets (complete graph, feature_dim from cfg) ----
    tr_ds = NPZTSPDataset(str(train_root), candidate_mode="complete", k=0, feature_dim=int(cfg.feature_dim))
    va_ds = NPZTSPDataset(str(val_root),   candidate_mode="complete", k=0, feature_dim=int(cfg.feature_dim))

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
    overrides = {"in_dim": int(cfg.feature_dim)}
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
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best = float("inf")
    best_ep = 0
    ensure_models_dir()

    # dynamic, informative filenames + 'latest' pointer + meta json
    tag = (
        f"{cfg.model_name}_in{mparams['in_dim']}_h{mparams['hidden']}"
        f"_d{mparams.get('depth', 'NA')}_bs{cfg.batch_size}_lr{cfg.lr}_ep{cfg.epochs}_seed{cfg.seed}"
    )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base  = f"{cfg.exp_id}_{tag}_{stamp}"
    ckpt  = Path(f"runs/models/{base}_best.pt")
    meta  = Path(f"runs/models/{base}_meta.json")
    latest      = Path(f"runs/models/{cfg.exp_id}_latest.pt")
    latest_meta = Path(f"runs/models/{cfg.exp_id}_latest.json")

    # ---- epochs ----
    for ep in range(1, int(cfg.epochs) + 1):
        # TRAIN
        model.train(); tot = 0.0; cnt = 0
        with tqdm(total=len(tr), ncols=100, desc=f"epoch {ep:02d}/{cfg.epochs} [train]", leave=False) as pbar:
            for b in tr:
                x = b["edge_feats"].to(device, non_blocking=True)
                y = b["labels"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    logits = model(x)
                    loss = _bce_balanced(logits, y)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                tot += float(loss.item()) * y.numel(); cnt += y.numel()
                pbar.update(1)
        train_loss = tot / max(cnt, 1)

        # VAL
        model.eval(); vtot = 0.0; vcnt = 0
        with torch.no_grad(), tqdm(total=len(va), ncols=100, desc=f"epoch {ep:02d}/{cfg.epochs} [val]  ", leave=False) as pbar:
            for b in va:
                x = b["edge_feats"].to(device, non_blocking=True)
                y = b["labels"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
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
            meta.write_text(json.dumps({
                "exp_id": cfg.exp_id, "model": cfg.model_name,
                "in_dim": mparams["in_dim"], "hidden": mparams["hidden"],
                "dropout": mparams["dropout"], "depth": mparams.get("depth"),
                "batch_size": cfg.batch_size, "lr": cfg.lr, "epochs": cfg.epochs, "seed": cfg.seed,
                "val_bce": round(float(val_loss), 6), "saved_at": stamp, "path": str(ckpt)
            }, indent=2), encoding="utf-8")
            shutil.copy2(ckpt, latest)
            latest_meta.write_text(meta.read_text(encoding="utf-8"), encoding="utf-8")
            tqdm.write(f"  ✓ SAVED → {ckpt.name} (and updated {latest.name})")

    logger.info(f"Best Val {best:.6f} @ epoch {best_ep}\nSaved: {ckpt}")
