from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm

from ..config import EvalCfg, QACfg
from ..utils.io import load_npz
from ..utils.geom import complete_edges, edge_features
from ..utils.tour import tour_edges_undirected, greedy_cycle_from_edges, two_opt, tour_length, verify_tour, tour_length_tsplib
from ..models.registry import build_model_from_state, load_weights_flex



def run(cfg: EvalCfg, logger):
    mp = Path(cfg.model_path)
    dr = Path(cfg.data_root)
    if not mp.exists() or not dr.exists():
        raise FileNotFoundError("eval: model_path or data_root invalid")

    # Build model from checkpoint (auto infer), optionally override input dim
    state = torch.load(mp, map_location="cpu")
    # Infer input dim from checkpoint; fixed features are 10D when training from this repo
    model, mparams = build_model_from_state(state, prefer_name=None, overrides=None)
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
        logger.info(f"saved {p}")


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

    rows = []
    covs = []
    gt_bad = []
    len_bad = []

    for f in files:
        d = load_npz(f)
        C = d["coords"].astype(np.float32)
        t = d.get("label_tour", None)
        n = int(C.shape[0])

        gt_ok = verify_tour(t, n) if t is not None else False
        cov = None
        lengths_ok = True

        if cfg.coverage and gt_ok:
            # Use complete graph for QA to expect coverage==1 under full candidates
            E = complete_edges(n)
            gt_edges = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in tour_edges_undirected(t))
            cand_edges = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in E)
            cov = len(gt_edges & cand_edges) / len(gt_edges) if gt_edges else float("nan")
            covs.append(cov)

        if cfg.lengths and gt_ok:
            Ls = float(d.get("label_len_norm", -1.0))
            Lc = float(tour_length(C, t))
            lengths_ok = abs(Ls - Lc) <= 1e-6
            if not lengths_ok:
                len_bad.append(f.name)

        if cfg.check_gt and not gt_ok:
            gt_bad.append(f.name)

        rows.append({
            "instance": f.name,
            "n": n,
            "gt_valid": gt_ok,
            "coverage": cov,
            "lengths_ok": lengths_ok,
        })

    if cfg.check_gt:
        logger.info("[check_gt] OK" if not gt_bad else f"[check_gt] invalid: {len(gt_bad)}")
    if cfg.coverage and covs:
        logger.info(f"[coverage] mean={np.mean(covs):.4f} min={np.min(covs):.4f} max={np.max(covs):.4f}")
    if cfg.lengths:
        logger.info("[lengths] OK" if not len_bad else f"[lengths] mismatch: {len(len_bad)}")

    if cfg.csv:
        p = Path(cfg.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=["instance", "n", "gt_valid", "coverage", "lengths_ok"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        logger.info(f"[csv] wrote {p}")
