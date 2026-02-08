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
from ..utils.run_paths import resolve_model_path, infer_run_dir, dataset_tag
from ..models.registry import build_model_from_state, load_weights_flex


def _eval_files(
    files: list[Path],
    model: torch.nn.Module,
    dev: torch.device,
    in_dim: int,
    run_twoopt: bool,
):
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
            F = edge_features(C, E, feature_dim=in_dim)
            logits = model(torch.from_numpy(F).float().to(dev)).cpu().numpy()
            pred = greedy_cycle_from_edges(n, E, logits)
            if run_twoopt:
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
    return results


def _resolve_save_path(save_cfg: str | None, tag: str, run_dir: Path | None, multi: bool) -> Path | None:
    if save_cfg is None:
        return None
    if isinstance(save_cfg, str) and save_cfg.lower() in ("", "none", "null"):
        return None
    if isinstance(save_cfg, str) and save_cfg.lower() == "auto":
        if run_dir is not None:
            out_dir = run_dir / "evals"
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / f"eval_{tag}.json"
        p = Path("runs/evals") / f"eval_{tag}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    p = Path(str(save_cfg))
    if "{dataset}" in str(save_cfg):
        p = Path(str(save_cfg).replace("{dataset}", tag))
    elif multi:
        suffix = p.suffix or ".json"
        p = p.with_name(f"{p.stem}_{tag}{suffix}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def run(cfg: EvalCfg, logger):
    model_path_cfg = Path(cfg.model_path)
    mp = resolve_model_path(model_path_cfg)
    if not mp.exists():
        raise FileNotFoundError("eval: model_path invalid")

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    # Build model from checkpoint (auto infer), optionally override input dim
    state = torch.load(mp, map_location="cpu")
    # Infer input dim from checkpoint; fixed features are 10D when training from this repo
    model, mparams = build_model_from_state(state, prefer_name=None, overrides=None)
    logger.info(f"Eval model params: {mparams}")
    load_weights_flex(model, state, logger=logger)

    dev = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    roots = cfg.data_roots
    if not roots:
        raise FileNotFoundError("eval: data_roots is empty")
    multi = len(roots) > 1
    run_dir = infer_run_dir(model_path_cfg, mp)

    for root in roots:
        dr = Path(root)
        if not dr.exists():
            logger.error(f"eval: data_root invalid: {dr}")
            continue
        files = sorted(dr.rglob("*.npz"))
        if not files:
            logger.error(f"eval: no .npz files in {dr}")
            continue
        logger.info(f"Evaluating {len(files)} instances from {dr}...")

        results = _eval_files(files, model, dev, mparams["in_dim"], cfg.run_twoopt)

        tag = dataset_tag(dr)
        p = _resolve_save_path(cfg.save_json, tag, run_dir, multi)
        if p is not None:
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)
            logger.info(f"saved {p}")


# -----------------------------
# QA flow (unchanged behavior)
# -----------------------------

def run_qa(cfg: QACfg, logger):
    root = Path(cfg.root)
    files = sorted(root.rglob("*.npz"))
    if not files:
        logger.error("qa: no .npz files")
        return
    logger.info(f"QA on {len(files)} files")

    rows = []
    gt_bad = []
    len_bad = []
    len_tsplib_bad = []
    concorde_bad = []
    require_bad = []
    missing_key_files = []
    dist_counts: dict[str, int] = {}
    size_counts: dict[int, int] = {}
    source_counts: dict[str, int] = {}
    label_source_counts: dict[str, int] = {}

    def _as_str(v) -> str | None:
        if v is None:
            return None
        try:
            if hasattr(v, "shape") and v.shape == ():
                return str(v.item())
        except Exception:
            pass
        try:
            return str(v)
        except Exception:
            return None

    for f in files:
        d = load_npz(f)
        missing = [k for k in ("coords", "n") if k not in d]
        if missing:
            missing_key_files.append(f.name)
            rows.append({
                "instance": f.name,
                "n": -1,
                "gt_valid": False,
                "lengths_ok": False,
                "len_tsplib_ok": None,
                "distribution": None,
                "source": None,
                "missing_keys": ",".join(missing),
            })
            continue

        C = d["coords"].astype(np.float32)
        t = d.get("label_tour", None)
        t = t.astype(np.int64) if t is not None else None
        n = int(d.get("n", C.shape[0]))

        dist = _as_str(d.get("distribution", None))
        src = _as_str(d.get("source", None))
        label_src = _as_str(d.get("label_source", None))
        if dist:
            dist_counts[dist] = dist_counts.get(dist, 0) + 1
        if src:
            source_counts[src] = source_counts.get(src, 0) + 1
        if label_src:
            label_source_counts[label_src] = label_source_counts.get(label_src, 0) + 1
        size_counts[n] = size_counts.get(n, 0) + 1

        gt_ok = verify_tour(t, n) if t is not None else False
        lengths_ok = True
        len_tsplib_ok = None
        concorde_ok = None
        concorde_reason = ""
        require_ok = None

        if cfg.lengths and gt_ok and t is not None:
            Ls = float(d.get("label_len_norm", -1.0))
            Lc = float(tour_length(C, t))
            lengths_ok = abs(Ls - Lc) <= 1e-6
            if not lengths_ok:
                len_bad.append(f.name)

            coords_orig = d.get("coords_orig", None)
            metric = d.get("metric", None)
            if "label_len_tsplib" in d and coords_orig is not None and metric is not None:
                Lt = float(d.get("label_len_tsplib", -1.0))
                Lc_t = float(tour_length_tsplib(coords_orig, t, str(metric)))
                len_tsplib_ok = abs(Lt - Lc_t) <= 1e-6
                if not len_tsplib_ok:
                    len_tsplib_bad.append(f.name)

        # Concorde-specific checks
        if cfg.check_concorde and label_src == "concorde":
            missing = []
            if d.get("coords_orig", None) is None:
                missing.append("coords_orig")
            if d.get("metric", None) is None:
                missing.append("metric")
            if "label_len_tsplib" not in d:
                missing.append("label_len_tsplib")
            if missing:
                concorde_ok = False
                concorde_reason = "missing:" + ",".join(missing)
                concorde_bad.append(f.name)
            else:
                concorde_ok = True if len_tsplib_ok is True else False
                if concorde_ok is False:
                    concorde_reason = "len_tsplib_mismatch"
                    concorde_bad.append(f.name)

        # Enforce label_source for synthetic data (optional)
        if cfg.require_label_source:
            req = str(cfg.require_label_source).lower()
            if src == "synthetic":
                require_ok = (label_src == req)
                if not require_ok:
                    require_bad.append(f.name)

        if cfg.check_gt and not gt_ok:
            gt_bad.append(f.name)

        rows.append({
            "instance": f.name,
            "n": n,
            "gt_valid": gt_ok,
            "lengths_ok": lengths_ok,
            "len_tsplib_ok": len_tsplib_ok,
            "distribution": dist,
            "source": src,
            "label_source": label_src,
            "concorde_ok": concorde_ok,
            "require_ok": require_ok,
            "concorde_reason": concorde_reason,
            "missing_keys": "",
        })

    if cfg.check_gt:
        logger.info("[check_gt] OK" if not gt_bad else f"[check_gt] invalid: {len(gt_bad)}")
    if cfg.lengths:
        logger.info("[lengths] OK" if not len_bad else f"[lengths] mismatch: {len(len_bad)}")
        if len_tsplib_bad:
            logger.info(f"[lengths_tsplib] mismatch: {len(len_tsplib_bad)}")
    if cfg.check_concorde:
        logger.info("[concorde] OK" if not concorde_bad else f"[concorde] issues: {len(concorde_bad)}")
    if cfg.require_label_source:
        logger.info("[label_source] OK" if not require_bad else f"[label_source] mismatch: {len(require_bad)}")
    if missing_key_files:
        logger.info(f"[missing_keys] {len(missing_key_files)} files missing required fields")

    if dist_counts:
        logger.info(f"[distributions] {dict(sorted(dist_counts.items()))}")
    if size_counts:
        logger.info(f"[sizes] {dict(sorted(size_counts.items()))}")
    if source_counts:
        logger.info(f"[sources] {dict(sorted(source_counts.items()))}")
    if label_source_counts:
        logger.info(f"[label_sources] {dict(sorted(label_source_counts.items()))}")

    if cfg.csv:
        p = Path(cfg.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(
                fh,
                fieldnames=[
                    "instance",
                    "n",
                    "gt_valid",
                    "lengths_ok",
                    "len_tsplib_ok",
                    "distribution",
                    "source",
                    "label_source",
                    "concorde_ok",
                    "require_ok",
                    "concorde_reason",
                    "missing_keys",
                ],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        logger.info(f"[csv] wrote {p}")
