from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AnalyzeCfg, load_config
from ..utils.run_paths import dataset_tag
from .report_plots import generate_analysis_plots


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _resolve_embedded_path(anchor_file: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    p = Path(str(raw_path))
    if p.is_absolute():
        return p
    direct = p.resolve()
    if direct.exists():
        return direct
    anchored = (anchor_file.parent / p).resolve()
    if anchored.exists():
        return anchored
    return direct


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_history_stats(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "epochs_run": 0,
        "train_loss_first": None,
        "train_loss_last": None,
        "train_loss_delta": None,
        "val_loss_best": None,
        "val_loss_last": None,
        "val_loss_best_epoch": None,
        "total_epoch_seconds": None,
    }
    if not path.exists():
        return out

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    if not rows:
        return out

    out["epochs_run"] = len(rows)
    tr_vals = [_safe_float(r.get("train_loss")) for r in rows]
    val_vals = [_safe_float(r.get("val_loss")) for r in rows]
    sec_vals = [_safe_float(r.get("seconds")) for r in rows]

    tr_clean = [x for x in tr_vals if x is not None]
    val_clean = [x for x in val_vals if x is not None]
    sec_clean = [x for x in sec_vals if x is not None]

    if tr_clean:
        out["train_loss_first"] = tr_clean[0]
        out["train_loss_last"] = tr_clean[-1]
        out["train_loss_delta"] = tr_clean[-1] - tr_clean[0]

    if val_clean:
        out["val_loss_last"] = val_clean[-1]
        best_val = min(val_clean)
        out["val_loss_best"] = best_val
        # Identify epoch index in original rows (1-based).
        for i, v in enumerate(val_vals, start=1):
            if v is not None and abs(v - best_val) <= 1e-12:
                out["val_loss_best_epoch"] = i
                break

    if sec_clean:
        out["total_epoch_seconds"] = float(sum(sec_clean))
    return out


def _load_eval_stats(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "exists": path.exists(),
        "rows": 0,
        "gap_count": 0,
        "gap_mean": None,
        "gap_median": None,
        "gap_p90": None,
        "gap_min": None,
        "gap_max": None,
        "gap_std": None,
    }
    if not path.exists():
        return out

    data = _read_json(path)
    if not isinstance(data, list):
        return out
    out["rows"] = len(data)

    gaps: list[float] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        g = _safe_float(row.get("gap_pct"))
        if g is not None:
            gaps.append(g)
    if not gaps:
        return out

    arr = np.asarray(gaps, dtype=np.float64)
    out["gap_count"] = int(arr.size)
    out["gap_mean"] = float(np.mean(arr))
    out["gap_median"] = float(np.median(arr))
    out["gap_p90"] = float(np.percentile(arr, 90))
    out["gap_min"] = float(np.min(arr))
    out["gap_max"] = float(np.max(arr))
    out["gap_std"] = float(np.std(arr))
    return out


def _load_instance_gaps(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    data = _read_json(path)
    if not isinstance(data, list):
        return out
    for row in data:
        if not isinstance(row, dict):
            continue
        inst = row.get("instance")
        g = _safe_float(row.get("gap_pct"))
        if isinstance(inst, str) and g is not None:
            out[inst] = g
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sanitize_eval_key(filename: str) -> str:
    return filename.replace(".json", "").replace(".", "_")


def _infer_train_data_group(train_root: str) -> str:
    root = str(train_root).replace("\\", "/").lower()
    if "synthetic_concorde_v1" in root:
        return "synthetic_concorde_exact"
    if "synthetic" in root:
        return "synthetic_heuristic"
    if "tsplib" in root:
        return "tsplib"
    return "other"


def _infer_eval_profile_label(eval_name: str) -> str:
    name = str(eval_name).lower()
    if "optimized" in name:
        return "optimized"
    if "baseline" in name:
        return "baseline"
    return "primary"


def _eval_json_equal(a: Path, b: Path) -> bool:
    if not a.exists() or not b.exists():
        return False
    if a.resolve() == b.resolve():
        return True
    da = _read_json(a)
    db = _read_json(b)
    return da == db and da is not None


def _resolve_eval_profile_for_run(eval_dir: Path, primary_eval: str) -> str:
    primary = eval_dir / primary_eval
    base_label = _infer_eval_profile_label(primary_eval)
    if not primary.exists():
        return base_label
    optimized = eval_dir / "eval_tsplib_optimized.json"
    baseline = eval_dir / "eval_tsplib_baseline.json"
    if primary.name != optimized.name and _eval_json_equal(primary, optimized):
        return "optimized_alias"
    if primary.name != baseline.name and _eval_json_equal(primary, baseline):
        return "baseline_alias"
    return base_label


def _build_main_summary(cfg: AnalyzeCfg, logger) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(cfg.experiments_root)
    if not root.exists():
        raise FileNotFoundError(f"analyze.experiments_root does not exist: {root}")

    exp_dirs = sorted(root.glob(cfg.exp_glob))
    exp_dirs = [p for p in exp_dirs if p.is_dir()]
    logger.info(f"[analyze] matched experiments: {len(exp_dirs)}")

    rows: list[dict[str, Any]] = []
    eval_long_rows: list[dict[str, Any]] = []

    for exp_dir in exp_dirs:
        latest_path = exp_dir / "latest.json"
        if not latest_path.exists():
            continue

        latest = _read_json(latest_path)
        if not isinstance(latest, dict):
            continue

        run_dir = _resolve_embedded_path(latest_path, latest.get("run_dir"))
        if run_dir is None:
            continue

        model_name = str(latest.get("model", "")) if latest.get("model") is not None else ""
        edge_feat_mode = ""
        mp = latest.get("model_params")
        if isinstance(mp, dict) and mp.get("edge_feat_mode") is not None:
            edge_feat_mode = str(mp.get("edge_feat_mode"))

        history_path = _resolve_embedded_path(latest_path, latest.get("history_csv"))
        if history_path is None:
            history_path = run_dir / "train_history.csv"
        hist = _load_history_stats(history_path)
        train_root = ""
        val_root = ""
        try:
            run_cfg = load_config(run_dir / "config.yaml")
            train_root = str(run_cfg.train.train_root)
            val_root = str(run_cfg.train.val_root)
        except Exception:
            pass

        row: dict[str, Any] = {
            "exp": exp_dir.name,
            "run": Path(str(run_dir)).name,
            "run_dir": str(run_dir),
            "model": model_name,
            "edge_feat_mode": edge_feat_mode,
            "checkpoint": str(_resolve_embedded_path(latest_path, latest.get("path")) or ""),
            "train_root": train_root,
            "val_root": val_root,
            "train_data_group": _infer_train_data_group(train_root),
            "primary_eval_file": cfg.primary_eval,
            "primary_eval_profile": _resolve_eval_profile_for_run(run_dir / "evals", cfg.primary_eval),
            "epochs_run": hist["epochs_run"],
            "train_loss_first": hist["train_loss_first"],
            "train_loss_last": hist["train_loss_last"],
            "train_loss_delta": hist["train_loss_delta"],
            "val_loss_best": hist["val_loss_best"],
            "val_loss_last": hist["val_loss_last"],
            "val_loss_best_epoch": hist["val_loss_best_epoch"],
            "total_epoch_seconds": hist["total_epoch_seconds"],
        }

        eval_dir = run_dir / "evals"
        for eval_name in cfg.eval_files:
            p = eval_dir / eval_name
            st = _load_eval_stats(p)
            key = _sanitize_eval_key(eval_name)
            row[f"{key}_exists"] = st["exists"]
            row[f"{key}_rows"] = st["rows"]
            row[f"{key}_gap_mean"] = st["gap_mean"]
            row[f"{key}_gap_median"] = st["gap_median"]
            row[f"{key}_gap_p90"] = st["gap_p90"]
            row[f"{key}_gap_std"] = st["gap_std"]
            row[f"{key}_gap_min"] = st["gap_min"]
            row[f"{key}_gap_max"] = st["gap_max"]

            if st["exists"]:
                eval_long_rows.append(
                    {
                        "exp": exp_dir.name,
                        "run": Path(str(run_dir)).name,
                        "model": model_name,
                        "eval_file": eval_name,
                        "rows": st["rows"],
                        "gap_count": st["gap_count"],
                        "gap_mean": st["gap_mean"],
                        "gap_median": st["gap_median"],
                        "gap_p90": st["gap_p90"],
                        "gap_std": st["gap_std"],
                        "gap_min": st["gap_min"],
                        "gap_max": st["gap_max"],
                    }
                )

        rows.append(row)
    return rows, eval_long_rows


def _build_profile_compare(rows: list[dict[str, Any]], baseline_name: str, optimized_name: str) -> list[dict[str, Any]]:
    bkey = f"{_sanitize_eval_key(baseline_name)}_gap_mean"
    okey = f"{_sanitize_eval_key(optimized_name)}_gap_mean"
    out: list[dict[str, Any]] = []
    for row in rows:
        b = _safe_float(row.get(bkey))
        o = _safe_float(row.get(okey))
        if b is None or o is None:
            continue
        out.append(
            {
                "exp": row["exp"],
                "run": row["run"],
                "model": row["model"],
                "baseline_mean_gap_pct": b,
                "optimized_mean_gap_pct": o,
                "delta_pct": o - b,
            }
        )
    out.sort(key=lambda r: float(r["optimized_mean_gap_pct"]))
    return out


def _build_instance_matrix(rows: list[dict[str, Any]], primary_eval: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Map model id -> instance -> gap
    by_model: dict[str, dict[str, float]] = {}
    model_meta: dict[str, tuple[str, str, str]] = {}  # exp, run, model
    row_by_exp: dict[str, dict[str, Any]] = {}
    all_instances: set[str] = set()

    for row in rows:
        exp = str(row["exp"])
        run = str(row["run"])
        model = str(row["model"])
        run_dir = Path(str(row.get("run_dir", Path("runs/experiments") / exp / run)))
        eval_path = run_dir / "evals" / primary_eval
        gaps = _load_instance_gaps(eval_path)
        if not gaps:
            continue
        by_model[exp] = gaps
        model_meta[exp] = (exp, run, model)
        row_by_exp[exp] = row
        all_instances.update(gaps.keys())

    if not by_model:
        return [], []

    exps = sorted(by_model.keys())
    instances = sorted(all_instances)

    matrix_rows: list[dict[str, Any]] = []
    for inst in instances:
        r: dict[str, Any] = {"instance": inst}
        vals = []
        for exp in exps:
            g = by_model[exp].get(inst)
            r[exp] = g
            if g is not None:
                vals.append(g)
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            r["mean_gap_pct"] = float(np.mean(arr))
            r["best_gap_pct"] = float(np.min(arr))
            r["worst_gap_pct"] = float(np.max(arr))
            r["spread_pct"] = float(np.max(arr) - np.min(arr))
        matrix_rows.append(r)

    # Per-model ranking on shared instances.
    rank_sum: dict[str, float] = defaultdict(float)
    rank_cnt: dict[str, int] = defaultdict(int)
    win_cnt: dict[str, int] = defaultdict(int)
    for inst in instances:
        pairs = [(exp, by_model[exp].get(inst)) for exp in exps]
        pairs = [(exp, gap) for exp, gap in pairs if gap is not None]
        if not pairs:
            continue
        pairs.sort(key=lambda x: x[1])
        best_gap = pairs[0][1]
        pos = 0
        while pos < len(pairs):
            end = pos
            while end + 1 < len(pairs) and abs(float(pairs[end + 1][1]) - float(pairs[pos][1])) <= 1e-12:
                end += 1
            avg_rank = (float(pos + 1) + float(end + 1)) / 2.0
            for i in range(pos, end + 1):
                exp, gap = pairs[i]
                rank_sum[exp] += avg_rank
                rank_cnt[exp] += 1
                if abs(float(gap) - float(best_gap)) <= 1e-12:
                    win_cnt[exp] += 1
            pos = end + 1

    ranking_rows: list[dict[str, Any]] = []
    for exp in exps:
        exp_name, run, model = model_meta[exp]
        cnt = rank_cnt.get(exp, 0)
        mean_rank = (rank_sum[exp] / cnt) if cnt > 0 else None
        mean_gap = _safe_float(row_by_exp[exp].get(f"{_sanitize_eval_key(primary_eval)}_gap_mean"))
        ranking_rows.append(
            {
                "exp": exp_name,
                "run": run,
                "model": model,
                "instances_compared": cnt,
                "mean_rank": mean_rank,
                "wins": win_cnt.get(exp, 0),
                "primary_mean_gap_pct": mean_gap,
            }
        )
    ranking_rows.sort(key=lambda r: (float(r["mean_rank"]) if r["mean_rank"] is not None else 1e9))
    return matrix_rows, ranking_rows


def _load_baseline_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.append(dict(row))
    return out


def _build_train_group_summary(rows: list[dict[str, Any]], primary_eval: str) -> list[dict[str, Any]]:
    primary_key = f"{_sanitize_eval_key(primary_eval)}_gap_mean"
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        g = _safe_float(row.get(primary_key))
        if g is None:
            continue
        grouped[(str(row.get("train_data_group", "")), str(row.get("model", "")))].append(g)

    out: list[dict[str, Any]] = []
    for (train_group, model), vals in grouped.items():
        arr = np.asarray(vals, dtype=np.float64)
        out.append(
            {
                "train_data_group": train_group,
                "model": model,
                "experiments": int(arr.size),
                "mean_gap_pct": float(np.mean(arr)),
                "median_gap_pct": float(np.median(arr)),
                "best_gap_pct": float(np.min(arr)),
                "worst_gap_pct": float(np.max(arr)),
            }
        )
    out.sort(key=lambda r: (str(r["train_data_group"]), float(r["mean_gap_pct"])))
    return out


def _build_data_quality_pairs(rows: list[dict[str, Any]], primary_eval: str) -> list[dict[str, Any]]:
    primary_key = f"{_sanitize_eval_key(primary_eval)}_gap_mean"
    by_exp = {str(row.get("exp", "")): row for row in rows}
    out: list[dict[str, Any]] = []
    for exp, row in sorted(by_exp.items()):
        if exp.endswith("_ccv1"):
            continue
        paired_exp = f"{exp}_ccv1"
        paired = by_exp.get(paired_exp)
        if paired is None:
            continue
        gap_base = _safe_float(row.get(primary_key))
        gap_ccv1 = _safe_float(paired.get(primary_key))
        if gap_base is None or gap_ccv1 is None:
            continue
        out.append(
            {
                "base_exp": exp,
                "ccv1_exp": paired_exp,
                "model": str(row.get("model", "")),
                "base_train_group": str(row.get("train_data_group", "")),
                "ccv1_train_group": str(paired.get("train_data_group", "")),
                "base_gap_pct": gap_base,
                "ccv1_gap_pct": gap_ccv1,
                "delta_gap_pct": gap_ccv1 - gap_base,
            }
        )
    out.sort(key=lambda r: float(r["delta_gap_pct"]))
    return out


def run(cfg: AnalyzeCfg, logger):
    rows, eval_long_rows = _build_main_summary(cfg, logger)
    if not rows:
        logger.warning("[analyze] no experiment rows found")
        return

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main wide summary.
    main_fields = list(rows[0].keys())
    main_csv = out_dir / "experiments_summary.csv"
    _write_csv(main_csv, rows, main_fields)
    logger.info(f"[analyze] wrote {main_csv}")

    # Eval long table.
    if eval_long_rows:
        eval_fields = list(eval_long_rows[0].keys())
        eval_csv = out_dir / "eval_summary_long.csv"
        _write_csv(eval_csv, eval_long_rows, eval_fields)
        logger.info(f"[analyze] wrote {eval_csv}")

    # Baseline vs optimized comparison if both names are present.
    prof_rows: list[dict[str, Any]] = []
    if "eval_tsplib_baseline.json" in cfg.eval_files and "eval_tsplib_optimized.json" in cfg.eval_files:
        prof_rows = _build_profile_compare(rows, "eval_tsplib_baseline.json", "eval_tsplib_optimized.json")
        if prof_rows:
            prof_csv = out_dir / "eval_profile_compare.csv"
            _write_csv(prof_csv, prof_rows, list(prof_rows[0].keys()))
            logger.info(f"[analyze] wrote {prof_csv}")

    # Per-instance matrix + model ranking on primary eval.
    matrix_rows, ranking_rows = _build_instance_matrix(rows, cfg.primary_eval)
    if matrix_rows:
        matrix_csv = out_dir / "instance_gap_matrix.csv"
        _write_csv(matrix_csv, matrix_rows, list(matrix_rows[0].keys()))
        logger.info(f"[analyze] wrote {matrix_csv}")
    if ranking_rows:
        ranking_csv = out_dir / "model_ranking.csv"
        _write_csv(ranking_csv, ranking_rows, list(ranking_rows[0].keys()))
        logger.info(f"[analyze] wrote {ranking_csv}")

    train_group_rows = _build_train_group_summary(rows, cfg.primary_eval)
    if train_group_rows:
        train_group_csv = out_dir / "train_group_summary.csv"
        _write_csv(train_group_csv, train_group_rows, list(train_group_rows[0].keys()))
        logger.info(f"[analyze] wrote {train_group_csv}")

    data_quality_rows = _build_data_quality_pairs(rows, cfg.primary_eval)
    if data_quality_rows:
        data_quality_csv = out_dir / "data_quality_pairs.csv"
        _write_csv(data_quality_csv, data_quality_rows, list(data_quality_rows[0].keys()))
        logger.info(f"[analyze] wrote {data_quality_csv}")

    # Optional baseline-vs-models board.
    board_rows: list[dict[str, Any]] = []
    baseline_csv_cfg = cfg.baseline_summary_csv
    if baseline_csv_cfg and str(baseline_csv_cfg).lower() not in ("", "none", "null"):
        baseline_rows = _load_baseline_summary(Path(str(baseline_csv_cfg)))
        if baseline_rows:
            model_profile = _infer_eval_profile_label(cfg.primary_eval)
            baseline_profiles = {
                str(r.get("eval_profile", "")).strip().lower()
                for r in baseline_rows
                if str(r.get("eval_profile", "")).strip()
            }
            if baseline_profiles and (
                len(baseline_profiles) != 1 or model_profile not in baseline_profiles
            ):
                logger.warning(
                    "[analyze] baseline_vs_models mixes model primary eval "
                    f"({cfg.primary_eval}) with baseline summary settings; compare decode settings before ranking."
                )
            primary_key = f"{_sanitize_eval_key(cfg.primary_eval)}_gap_mean"
            for r in rows:
                g = _safe_float(r.get(primary_key))
                if g is None:
                    continue
                board_rows.append(
                    {
                        "type": "model",
                        "name": str(r.get("exp", "")),
                        "model": str(r.get("model", "")),
                        "run": str(r.get("run", "")),
                        "dataset": dataset_tag(Path("runs/data/tsplib/processed")),
                        "mean_gap_pct": g,
                        "eval_file": str(r.get("primary_eval_file", cfg.primary_eval)),
                        "eval_profile": str(r.get("primary_eval_profile", _infer_eval_profile_label(cfg.primary_eval))),
                        "run_twoopt": "",
                        "decode_multistart": "",
                        "decode_noise_std": "",
                        "decode_twoopt_passes": "",
                        "seed": "",
                    }
                )
            for r in baseline_rows:
                g = _safe_float(r.get("mean_gap_pct"))
                if g is None:
                    continue
                board_rows.append(
                    {
                        "type": "baseline",
                        "name": str(r.get("baseline", "")),
                        "model": "",
                        "run": "",
                        "dataset": str(r.get("dataset", "")),
                        "mean_gap_pct": g,
                        "eval_file": str(r.get("result_json", "")),
                        "eval_profile": str(r.get("eval_profile", "baseline_cfg")),
                        "run_twoopt": str(r.get("run_twoopt", "")),
                        "decode_multistart": str(r.get("decode_multistart", "")),
                        "decode_noise_std": str(r.get("decode_noise_std", "")),
                        "decode_twoopt_passes": str(r.get("decode_twoopt_passes", "")),
                        "seed": str(r.get("seed", "")),
                    }
                )
            if board_rows:
                board_rows.sort(key=lambda x: float(x["mean_gap_pct"]))
                board_csv = out_dir / "baseline_vs_models.csv"
                _write_csv(board_csv, board_rows, list(board_rows[0].keys()))
                logger.info(f"[analyze] wrote {board_csv}")

    generate_analysis_plots(cfg, logger, rows, ranking_rows, prof_rows, matrix_rows, board_rows)
