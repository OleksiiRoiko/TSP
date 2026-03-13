from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import BaselineEvalCfg
from ..utils.geom import complete_edges
from ..utils.io import load_npz
from ..utils.run_paths import dataset_tag
from ..utils.tour import (
    decode_tour_from_edge_scores,
    nearest_neighbor_multistart,
    tour_length,
    tour_length_tsplib,
    two_opt,
)


def _compute_lengths_and_gap(d: dict, pred: np.ndarray, coords: np.ndarray) -> tuple[float, float, float, float, float]:
    gt = d.get("label_tour", None)
    gt = gt.astype(np.int64) if gt is not None else None

    pred_len_norm = float(tour_length(coords, pred))
    gt_len_norm = float("nan") if gt is None else float(tour_length(coords, gt))

    metric = d.get("metric", None)
    coords_orig = d.get("coords_orig", None)
    if coords_orig is not None and metric is not None and gt is not None:
        pred_len_tsplib = tour_length_tsplib(coords_orig, pred, str(metric))
        gt_len_tsplib = tour_length_tsplib(coords_orig, gt, str(metric))
        gap = ((pred_len_tsplib - gt_len_tsplib) / gt_len_tsplib * 100.0) if gt_len_tsplib > 0 else float("nan")
    else:
        pred_len_tsplib = float("nan")
        gt_len_tsplib = float("nan")
        gap = ((pred_len_norm - gt_len_norm) / gt_len_norm * 100.0) if (not np.isnan(gt_len_norm) and gt_len_norm > 0) else float("nan")

    return pred_len_norm, gt_len_norm, pred_len_tsplib, gt_len_tsplib, gap


def _predict_baseline_tour(
    baseline: str,
    coords: np.ndarray,
    *,
    run_twoopt: bool,
    multistart: int,
    noise_std: float,
    twoopt_passes: int,
    seed: int,
) -> np.ndarray:
    n = int(coords.shape[0])
    if baseline == "nn2opt":
        pred = nearest_neighbor_multistart(coords, multistart=max(1, int(multistart)), seed=int(seed))
        if run_twoopt:
            pred = two_opt(coords, pred, max_passes=max(1, int(twoopt_passes)))
        return pred

    if baseline == "dist_greedy2opt":
        E = complete_edges(n)
        a = E[:, 0]
        b = E[:, 1]
        D = coords[b] - coords[a]
        dist = np.linalg.norm(D, axis=1)
        scores = -dist
        return decode_tour_from_edge_scores(
            coords,
            E,
            scores,
            run_twoopt=bool(run_twoopt),
            twoopt_passes=max(1, int(twoopt_passes)),
            multistart=max(1, int(multistart)),
            noise_std=max(0.0, float(noise_std)),
            seed=int(seed),
        )

    raise ValueError(f"Unsupported baseline: {baseline}")


def run(cfg: BaselineEvalCfg, logger):
    roots = cfg.data_roots
    if not roots:
        raise FileNotFoundError("baseline: data_roots is empty")

    save_root = Path(cfg.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for baseline in cfg.names:
        baseline_dir = save_root / baseline
        baseline_dir.mkdir(parents=True, exist_ok=True)

        for root in roots:
            dr = Path(root)
            if not dr.exists():
                logger.error(f"baseline: data_root invalid: {dr}")
                continue

            files = sorted(dr.rglob("*.npz"))
            if not files:
                logger.error(f"baseline: no .npz files in {dr}")
                continue

            logger.info(f"[{baseline}] evaluating {len(files)} instances from {dr}")
            results: list[dict] = []
            for idx, f in enumerate(tqdm(files, ncols=100)):
                d = load_npz(f)
                coords = d["coords"].astype(np.float32)
                pred = _predict_baseline_tour(
                    baseline,
                    coords,
                    run_twoopt=bool(cfg.run_twoopt),
                    multistart=int(cfg.decode_multistart),
                    noise_std=float(cfg.decode_noise_std),
                    twoopt_passes=int(cfg.decode_twoopt_passes),
                    seed=int(cfg.seed) + (idx * 9973),
                )

                pred_len_norm, gt_len_norm, pred_len_tsplib, gt_len_tsplib, gap = _compute_lengths_and_gap(d, pred, coords)
                results.append(
                    {
                        "instance": f.name,
                        "n": int(coords.shape[0]),
                        "pred_len_norm": pred_len_norm,
                        "gt_len_norm": gt_len_norm,
                        "pred_len_tsplib": pred_len_tsplib,
                        "gt_len_tsplib": gt_len_tsplib,
                        "gap_pct": gap,
                    }
                )

            tag = dataset_tag(dr)
            out_json = baseline_dir / f"eval_{tag}.json"
            out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
            logger.info(f"[{baseline}] saved {out_json}")

            gaps = [float(r["gap_pct"]) for r in results if np.isfinite(float(r.get("gap_pct", np.nan)))]
            mean_gap = float(np.mean(gaps)) if gaps else float("nan")
            med_gap = float(np.median(gaps)) if gaps else float("nan")
            summary_rows.append(
                {
                    "baseline": baseline,
                    "dataset": tag,
                    "instances": len(results),
                    "mean_gap_pct": mean_gap,
                    "median_gap_pct": med_gap,
                    "run_twoopt": bool(cfg.run_twoopt),
                    "decode_multistart": int(cfg.decode_multistart),
                    "decode_noise_std": float(cfg.decode_noise_std),
                    "decode_twoopt_passes": int(cfg.decode_twoopt_passes),
                    "seed": int(cfg.seed),
                    "result_json": str(out_json),
                }
            )

    summary_csv = save_root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        if summary_rows:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        else:
            fh.write("baseline,dataset,instances,mean_gap_pct,median_gap_pct,run_twoopt,decode_multistart,decode_noise_std,decode_twoopt_passes,seed,result_json\n")
    logger.info(f"[baseline] summary saved {summary_csv}")
