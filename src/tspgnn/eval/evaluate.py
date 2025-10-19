from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm

from ..config import EvalCfg, QACfg
from ..utils.io import load_npz
from ..utils.geom import complete_edges, edge_features, knn_edges
from ..utils.tour import tour_edges_undirected, greedy_cycle_from_edges, two_opt, tour_length, verify_tour, tour_length_tsplib
from ..models.registry import build_model_from_state, load_weights_flex



def run(cfg: EvalCfg, logger):
    mp = Path(cfg.model_path)
    dr = Path(cfg.data_root)
    if not mp.exists() or not dr.exists():
        raise FileNotFoundError("eval: model_path or data_root invalid")

    # Build model from checkpoint (auto infer), optionally override input dim
    state = torch.load(mp, map_location="cpu")
    overrides = {}
    if cfg.feature_dim is not None:
        overrides["in_dim"] = int(cfg.feature_dim)
    model, mparams = build_model_from_state(state, prefer_name=None, overrides=overrides or None)
    logger.info(f"Eval model params: {mparams}")
    load_weights_flex(model, state, logger=logger)

    dev = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    files = sorted(dr.glob("*.npz"))
    if not files:
        raise FileNotFoundError("eval: no .npz files")
    logger.info(f"Evaluating {len(files)} instances...")

    results = []
    with torch.no_grad():
        for f in tqdm(files, ncols=100):
            d = load_npz(f)
            C = d["coords"].astype(np.float32)
            n = C.shape[0]
            gt = d.get("label_tour", None)
            gt = gt.astype(np.int64) if gt is not None else None

            # complete graph
            E = complete_edges(n)
            F = edge_features(C, E, feature_dim=mparams["in_dim"])
            logits = model(torch.from_numpy(F).float().to(dev)).cpu().numpy()
            pred = greedy_cycle_from_edges(n, E, logits)
            if cfg.run_twoopt:
                pred = two_opt(C, pred, max_passes=20)

            # normalized lengths (always available)
            pred_len_norm = float(tour_length(C, pred))
            gt_len_norm = float("nan") if gt is None else float(tour_length(C, gt))

            # TSPLIB lengths (if available in npz)
            metric = d.get("metric", None)
            coords_orig = d.get("coords_orig", None)
            if coords_orig is not None and metric is not None and gt is not None:
                pred_len_tsplib = tour_length_tsplib(coords_orig, pred, str(metric))
                gt_len_tsplib   = tour_length_tsplib(coords_orig, gt,   str(metric))
                # Use TSPLIB length for gap if present
                gap = ((pred_len_tsplib - gt_len_tsplib) / gt_len_tsplib * 100.0) if gt_len_tsplib > 0 else float("nan")
            else:
                pred_len_tsplib = float("nan")
                gt_len_tsplib   = float("nan")
                gap = ((pred_len_norm - gt_len_norm) / gt_len_norm * 100.0) if (not np.isnan(gt_len_norm) and gt_len_norm > 0) else float("nan")

            results.append({
                "instance": f.name,
                "n": int(n),
                "pred_tour": pred.tolist(),
                "pred_len_norm": pred_len_norm,
                "gt_len_norm": gt_len_norm,
                "pred_len_tsplib": pred_len_tsplib,
                "gt_len_tsplib": gt_len_tsplib,
                "gap_pct": gap
            })

    if cfg.save_json:
        p = Path(cfg.save_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logger.info(f"✓ saved {p}")


# -----------------------------
# QA flow (unchanged behavior)
# -----------------------------

def run_qa(cfg: QACfg, logger):
    root = Path(cfg.root)
    files = sorted(root.glob("*.npz"))
    if not files:
        logger.error("qa: no .npz files")
        return
    logger.info(f"QA on {len(files)} files")

    if cfg.check_gt:
        bad = []
        for f in files:
            d = load_npz(f)
            n = int(d["coords"].shape[0])
            t = d.get("label_tour", None)
            if not verify_tour(t, n):
                bad.append(f.name)
        logger.info("[check_gt] OK" if not bad else f"[check_gt] invalid: {len(bad)}")

    if cfg.coverage:
        covs = []
        for f in files:
            d = load_npz(f)
            C = d["coords"].astype(np.float32)
            t = d.get("label_tour", None)
            if t is None:
                continue
            # QA coverage uses kNN by design (diagnostic)
            E = knn_edges(C, int(cfg.k))
            gt = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in tour_edges_undirected(t))
            cand = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in E)
            covs.append(len(gt & cand) / len(gt))
        if covs:
            logger.info(f"[coverage] mean={np.mean(covs):.4f} min={np.min(covs):.4f} max={np.max(covs):.4f}")

    if cfg.lengths:
        bad = []
        for f in files:
            d = load_npz(f)
            C = d["coords"].astype(np.float32)
            t = d.get("label_tour", None)
            if t is None:
                continue
            Ls = float(d.get("label_len_norm", -1.0))
            Lc = float(tour_length(C, t))
            if abs(Ls - Lc) > 1e-6:
                bad.append(f.name)
        logger.info("[lengths] OK" if not bad else f"[lengths] mismatch: {len(bad)}")
