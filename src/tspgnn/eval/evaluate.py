from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
from tqdm import tqdm

from ..config import EvalCfg, QACfg
from ..utils.io import load_npz
from ..utils.geom import complete_edges, edge_features
from ..utils.tour import greedy_cycle_from_edges, two_opt, tour_length, verify_tour, tour_length_tsplib
from ..utils.run_paths import resolve_model_path, infer_run_dir, dataset_tag
from ..models.registry import build_model_from_state, load_weights_flex
from ..models.inference import load_state_dict, predict_logits


def _eval_files(
    files: list[Path],
    model: torch.nn.Module,
    dev: torch.device,
    in_dim: int,
    run_twoopt: bool,
    save_pred_tour: bool,
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
            logits = predict_logits(model, F, C, dev)
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

            row = {
                "instance": f.name,
                "n": int(n),
                "pred_len_norm": pred_len_norm,
                "gt_len_norm": gt_len_norm,
                "pred_len_tsplib": pred_len_tsplib,
                "gt_len_tsplib": gt_len_tsplib,
                "gap_pct": gap
            }
            if save_pred_tour:
                row["pred_tour"] = pred.tolist()
            results.append(row)
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
    state = load_state_dict(mp)
    # Infer input dim from checkpoint; fixed features are 10D when training from this repo
    model, mparams = build_model_from_state(state, prefer_name=None, overrides=None)
    logger.info(f"Eval model params: {mparams}")
    load_weights_flex(model, state, logger=logger, require_all_matched=True)

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

        results = _eval_files(
            files,
            model,
            dev,
            mparams["in_dim"],
            cfg.run_twoopt,
            bool(getattr(cfg, "save_pred_tour", False)),
        )

        tag = dataset_tag(dr)
        p = _resolve_save_path(cfg.save_json, tag, run_dir, multi)
        if p is not None:
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)
            logger.info(f"saved {p}")


# -----------------------------
# QA flow
# -----------------------------

_QA_FIELDNAMES = [
    "instance",
    "n",
    "gt_valid",
    "lengths_ok",
    "len_tsplib_ok",
    "distribution",
    "source",
    "label_source",
    "concorde_optimal_proved",
    "concorde_ok",
    "require_ok",
    "concorde_reason",
    "missing_keys",
]


@dataclass
class QAStats:
    gt_bad: list[str] = field(default_factory=list)
    len_bad: list[str] = field(default_factory=list)
    len_tsplib_bad: list[str] = field(default_factory=list)
    concorde_bad: list[str] = field(default_factory=list)
    proof_bad: list[str] = field(default_factory=list)
    require_bad: list[str] = field(default_factory=list)
    missing_key_files: list[str] = field(default_factory=list)
    coord_hash_splits: dict[str, set[str]] = field(default_factory=dict)
    coord_hash_counts: dict[str, int] = field(default_factory=dict)
    split_overlap_examples: list[str] = field(default_factory=list)
    split_labeled_files: int = 0
    dist_counts: dict[str, int] = field(default_factory=dict)
    size_counts: dict[int, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    label_source_counts: dict[str, int] = field(default_factory=dict)


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


def _as_bool(v) -> bool | None:
    if v is None:
        return None
    try:
        if hasattr(v, "shape") and v.shape == ():
            v = v.item()
    except Exception:
        pass
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y"):
            return True
        if s in ("0", "false", "no", "n"):
            return False
    return None


def _bump(counter: dict, key) -> None:
    counter[key] = counter.get(key, 0) + 1


def _split_name(path: Path) -> str | None:
    for part in path.parts:
        part_l = part.lower()
        if part_l in ("train", "val", "test"):
            return part_l
    return None


def _process_qa_file(path: Path, cfg: QACfg, rows: list[dict], stats: QAStats) -> None:
    d = load_npz(path)
    missing = [k for k in ("coords", "n") if k not in d]
    if missing:
        stats.missing_key_files.append(path.name)
        rows.append(
            {
                "instance": path.name,
                "n": -1,
                "gt_valid": False,
                "lengths_ok": False,
                "len_tsplib_ok": None,
                "distribution": None,
                "source": None,
                "label_source": None,
                "concorde_optimal_proved": None,
                "concorde_ok": None,
                "require_ok": None,
                "concorde_reason": "",
                "missing_keys": ",".join(missing),
            }
        )
        return

    coords = d["coords"].astype(np.float32)
    tour = d.get("label_tour", None)
    tour = tour.astype(np.int64) if tour is not None else None
    n = int(d.get("n", coords.shape[0]))

    if cfg.check_split_overlap:
        split_name = _split_name(path)
        if split_name is not None:
            stats.split_labeled_files += 1
            h = hashlib.blake2b(coords.tobytes(), digest_size=12).hexdigest()
            _bump(stats.coord_hash_counts, h)
            prev = stats.coord_hash_splits.get(h)
            if prev is None:
                stats.coord_hash_splits[h] = {split_name}
            else:
                if split_name not in prev and len(stats.split_overlap_examples) < 8:
                    stats.split_overlap_examples.append(f"{path.name}: {sorted(prev)} -> {split_name}")
                prev.add(split_name)

    dist = _as_str(d.get("distribution", None))
    src = _as_str(d.get("source", None))
    label_src = _as_str(d.get("label_source", None))
    concorde_optimal_proved = _as_bool(d.get("concorde_optimal_proved", None))
    if dist:
        _bump(stats.dist_counts, dist)
    if src:
        _bump(stats.source_counts, src)
    if label_src:
        _bump(stats.label_source_counts, label_src)
    _bump(stats.size_counts, n)

    gt_ok = verify_tour(tour, n) if tour is not None else False
    lengths_ok = True
    len_tsplib_ok = None
    concorde_ok = None
    concorde_reason = ""
    require_ok = None

    if cfg.lengths and gt_ok and tour is not None:
        label_len_norm = float(d.get("label_len_norm", -1.0))
        calc_len_norm = float(tour_length(coords, tour))
        lengths_ok = abs(label_len_norm - calc_len_norm) <= 1e-6
        if not lengths_ok:
            stats.len_bad.append(path.name)

        coords_orig = d.get("coords_orig", None)
        metric = d.get("metric", None)
        if "label_len_tsplib" in d and coords_orig is not None and metric is not None:
            label_len_tsplib = float(d.get("label_len_tsplib", -1.0))
            calc_len_tsplib = float(tour_length_tsplib(coords_orig, tour, str(metric)))
            len_tsplib_ok = abs(label_len_tsplib - calc_len_tsplib) <= 1e-6
            if not len_tsplib_ok:
                stats.len_tsplib_bad.append(path.name)

    if cfg.check_concorde and label_src == "concorde":
        missing = []
        if d.get("coords_orig", None) is None:
            missing.append("coords_orig")
        if d.get("metric", None) is None:
            missing.append("metric")
        if "label_len_tsplib" not in d:
            missing.append("label_len_tsplib")
        if cfg.check_concorde_optimal_proof and concorde_optimal_proved is None:
            missing.append("concorde_optimal_proved")
        if missing:
            concorde_ok = False
            concorde_reason = "missing:" + ",".join(missing)
            stats.concorde_bad.append(path.name)
        else:
            concorde_ok = True if len_tsplib_ok is True else False
            if concorde_ok is False:
                concorde_reason = "len_tsplib_mismatch"
                stats.concorde_bad.append(path.name)
            elif cfg.check_concorde_optimal_proof and concorde_optimal_proved is not True:
                concorde_ok = False
                concorde_reason = "optimal_proof_false"
                stats.concorde_bad.append(path.name)
                stats.proof_bad.append(path.name)

    if cfg.require_label_source:
        req = str(cfg.require_label_source).lower()
        if src == "synthetic":
            require_ok = label_src == req
            if not require_ok:
                stats.require_bad.append(path.name)

    if cfg.check_gt and not gt_ok:
        stats.gt_bad.append(path.name)

    rows.append(
        {
            "instance": path.name,
            "n": n,
            "gt_valid": gt_ok,
            "lengths_ok": lengths_ok,
            "len_tsplib_ok": len_tsplib_ok,
            "distribution": dist,
            "source": src,
            "label_source": label_src,
            "concorde_optimal_proved": concorde_optimal_proved,
            "concorde_ok": concorde_ok,
            "require_ok": require_ok,
            "concorde_reason": concorde_reason,
            "missing_keys": "",
        }
    )


def _log_qa_summary(cfg: QACfg, logger, stats: QAStats) -> None:
    if cfg.check_gt:
        logger.info("[check_gt] OK" if not stats.gt_bad else f"[check_gt] invalid: {len(stats.gt_bad)}")
    if cfg.lengths:
        logger.info("[lengths] OK" if not stats.len_bad else f"[lengths] mismatch: {len(stats.len_bad)}")
        if stats.len_tsplib_bad:
            logger.info(f"[lengths_tsplib] mismatch: {len(stats.len_tsplib_bad)}")
    if cfg.check_concorde:
        logger.info("[concorde] OK" if not stats.concorde_bad else f"[concorde] issues: {len(stats.concorde_bad)}")
        if cfg.check_concorde_optimal_proof:
            logger.info("[optimal_proof] OK" if not stats.proof_bad else f"[optimal_proof] issues: {len(stats.proof_bad)}")
    if cfg.require_label_source:
        logger.info("[label_source] OK" if not stats.require_bad else f"[label_source] mismatch: {len(stats.require_bad)}")
    if stats.missing_key_files:
        logger.info(f"[missing_keys] {len(stats.missing_key_files)} files missing required fields")

    if cfg.check_split_overlap:
        if stats.split_labeled_files == 0:
            logger.info("[split_overlap] skipped (no train/val/test paths under qa.root)")
        else:
            overlap_hashes = [h for h, splits in stats.coord_hash_splits.items() if len(splits) > 1]
            if overlap_hashes:
                overlap_files = sum(stats.coord_hash_counts[h] for h in overlap_hashes)
                logger.warning(
                    f"[split_overlap] duplicate coordinates across splits: "
                    f"{len(overlap_hashes)} unique hashes, {overlap_files} files"
                )
                if stats.split_overlap_examples:
                    logger.warning(f"[split_overlap_examples] {stats.split_overlap_examples}")
            else:
                logger.info("[split_overlap] OK")

    if stats.dist_counts:
        logger.info(f"[distributions] {dict(sorted(stats.dist_counts.items()))}")
    if stats.size_counts:
        logger.info(f"[sizes] {dict(sorted(stats.size_counts.items()))}")
    if stats.source_counts:
        logger.info(f"[sources] {dict(sorted(stats.source_counts.items()))}")
    if stats.label_source_counts:
        logger.info(f"[label_sources] {dict(sorted(stats.label_source_counts.items()))}")


def _write_qa_csv(path: Path, rows: list[dict], logger) -> None:
    import csv as _csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=_QA_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"[csv] wrote {path}")


def run_qa(cfg: QACfg, logger):
    root = Path(cfg.root)
    files = sorted(root.rglob("*.npz"))
    if not files:
        logger.error("qa: no .npz files")
        return
    logger.info(f"QA on {len(files)} files")

    rows: list[dict] = []
    stats = QAStats()
    for path in files:
        _process_qa_file(path, cfg, rows, stats)

    _log_qa_summary(cfg, logger, stats)
    if cfg.csv:
        _write_qa_csv(Path(cfg.csv), rows, logger)
