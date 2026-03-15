from __future__ import annotations
from pathlib import Path
from typing import cast
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..config import VisualizeCfg
from ..utils.io import load_npz
from ..utils.geom import complete_edges, edge_features
from ..utils.tour import decode_tour_from_edge_scores
from ..utils.run_paths import infer_run_dir, dataset_tag
from ..models.inference import load_model_for_inference, predict_logits


def _render(C, gt, pred, Ebg, out_path=None, figsize=(11.0, 5.5), dpi=150):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(float(figsize[0]), float(figsize[1])))
    for ax, title, T, color in [(axL, "Ground Truth", gt, "#1f77b4"), (axR, "Prediction", pred, "#d62728")]:
        ax.set_aspect("equal"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.axis("off"); ax.set_title(title, fontsize=12)
        if Ebg is not None:
            subset = Ebg if len(Ebg) < 5000 else Ebg[:5000]
            for a, b in subset:
                ax.plot([C[a, 0], C[b, 0]], [C[a, 1], C[b, 1]], "-", lw=0.5, alpha=0.25, color="#666", zorder=1)
        ax.scatter(C[:, 0], C[:, 1], s=12, c="#202428", zorder=3)
        n = len(T)
        for i in range(n):
            a, b = int(T[i]), int(T[(i + 1) % n])
            ax.plot([C[a, 0], C[b, 0]], [C[a, 1], C[b, 1]], "-", lw=1.8, color=color, zorder=4)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    else:
        plt.show()


def run(cfg: VisualizeCfg, logger):
    def _iter_targets():
        for t in cfg.targets:
            mode = (t.mode or "predict").lower()
            npz_dir = t.npz_dir
            limit = 0 if t.limit is None else int(t.limit)
            out_dir = t.out_dir if t.out_dir is not None else cfg.out_dir
            yield mode, npz_dir, limit, out_dir

    targets = list(_iter_targets())
    if not targets:
        logger.error("no visualization targets configured")
        return

    # Load model once if any target requires prediction
    model = None
    mparams = None
    dev = None
    model_path_cfg = None
    mp = None
    run_dir = None
    if any(mode == "predict" for mode, _, _, _ in targets):
        model_path_cfg = Path(cfg.model)
        model, mparams, mp = load_model_for_inference(model_path_cfg, logger=logger, require_all_matched=True)
        logger.info(f"Viz model params: {mparams}")
        dev = torch.device("cpu" if cfg.device == "cpu" or not torch.cuda.is_available() else "cuda")
        model.to(dev).eval()
        run_dir = infer_run_dir(model_path_cfg, mp)

    for mode, npz_dir, limit, out_dir_cfg in targets:
        files = sorted(Path(npz_dir).rglob("*.npz"))
        if limit and limit > 0:
            files = files[:limit]
        if not files:
            logger.error(f"no files in {npz_dir}")
            continue

        out_dir = Path(out_dir_cfg)
        if str(out_dir_cfg).lower() == "auto":
            tag = dataset_tag(Path(npz_dir))
            if mode == "predict" and run_dir is not None:
                out_dir = run_dir / "figs" / tag
            else:
                out_dir = Path("runs/figs") / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        if mode == "dataset":
            for f in tqdm(files, ncols=100):
                d = load_npz(f)
                C = d["coords"].astype(np.float32)
                gt = d.get("label_tour", None)
                fig, ax = plt.subplots(1, 1, figsize=(float(cfg.figsize[0]), float(cfg.figsize[1])))
                ax.set_aspect("equal"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.axis("off")
                ax.scatter(C[:, 0], C[:, 1], s=12, c="#202428")
                if gt is not None:
                    T = gt.astype(np.int64); n = T.shape[0]
                    for i in range(n):
                        a, b = int(T[i]), int(T[(i + 1) % n])
                        ax.plot([C[a, 0], C[b, 0]], [C[a, 1], C[b, 1]], "-", lw=1.8, color="#1f77b4")
                fig.savefig(out_dir / f"{f.stem}.png", dpi=int(cfg.dpi), bbox_inches="tight", pad_inches=0.02)
                plt.close(fig)
            logger.info(f"done (dataset mode) -> {out_dir}")
            continue

        # mode == "predict"
        if model is None or mparams is None or dev is None:
            logger.error("predict mode requested but model is not loaded")
            continue
        for idx, f in enumerate(tqdm(files, ncols=100)):
            try:
                d = load_npz(f)
                C = d["coords"].astype(np.float32)
                gt = d.get("label_tour", None)
                if gt is None:
                    continue
                gt = gt.astype(np.int64)

                # always complete graph
                Ebg = complete_edges(C.shape[0])
                in_dim = int(cast(int, mparams["in_dim"]))
                F = edge_features(C, Ebg, feature_dim=in_dim)
                with torch.no_grad():
                    s = predict_logits(model, F, C, dev)
                pred = decode_tour_from_edge_scores(
                    C,
                    Ebg,
                    s,
                    run_twoopt=True,
                    twoopt_passes=int(getattr(cfg, "decode_twoopt_passes", 20)),
                    multistart=int(getattr(cfg, "decode_multistart", 1)),
                    noise_std=float(getattr(cfg, "decode_noise_std", 0.0)),
                    seed=int(getattr(cfg, "seed", 0)) + (idx * 9973),
                )

                _render(C, gt, pred, Ebg, out_dir / f"{f.stem}.png", figsize=cfg.figsize, dpi=int(cfg.dpi))
            except Exception as e:
                logger.error(f"[{f.name}] {e}")
