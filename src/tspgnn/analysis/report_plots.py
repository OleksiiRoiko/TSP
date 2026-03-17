from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AnalyzeCfg


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _short_label(name: str) -> str:
    s = str(name).strip()
    if s.startswith("exp_edge_"):
        s = s[len("exp_edge_") :]
    return s


def _family_color(model: str, row_type: str = "model") -> str:
    if row_type == "baseline":
        return "#6b7280"
    model = str(model)
    if model == "edge_transformer":
        return "#1d4ed8"
    if model == "edge_res_mlp":
        return "#059669"
    if model == "edge_mlp_deep":
        return "#d97706"
    if model == "edge_mlp":
        return "#b45309"
    return "#374151"


def _load_history_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    out: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            epoch = _safe_float(row.get("epoch"))
            train_loss = _safe_float(row.get("train_loss"))
            val_loss = _safe_float(row.get("val_loss"))
            if epoch is None:
                continue
            out.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss if train_loss is not None else float("nan"),
                    "val_loss": val_loss if val_loss is not None else float("nan"),
                }
            )
    return out


def _plot_model_ranking(plot_dir: Path, ranking_rows: list[dict[str, Any]], logger) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in ranking_rows if _safe_float(r.get("primary_mean_gap_pct")) is not None]
    if not rows:
        return
    rows.sort(key=lambda r: float(r["primary_mean_gap_pct"]))
    labels = [_short_label(str(r["exp"])) for r in rows]
    values = [float(r["primary_mean_gap_pct"]) for r in rows]
    colors = [_family_color(str(r.get("model", ""))) for r in rows]

    fig_h = max(5.5, 0.38 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    y = np.arange(len(rows), dtype=np.float64)
    ax.barh(y, values, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean gap [%]")
    ax.set_title("Model ranking by mean TSPLIB gap")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    for yi, v in zip(y, values):
        ax.text(v + 0.08, yi, f"{v:.2f}", va="center", ha="left", fontsize=8)
    out = plot_dir / "primary_gap_ranking.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[analyze] wrote {out}")


def _plot_baseline_board(plot_dir: Path, board_rows: list[dict[str, Any]], logger) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in board_rows if _safe_float(r.get("mean_gap_pct")) is not None]
    if not rows:
        return
    rows.sort(key=lambda r: float(r["mean_gap_pct"]))
    labels = [_short_label(str(r["name"])) for r in rows]
    values = [float(r["mean_gap_pct"]) for r in rows]
    colors = [_family_color(str(r.get("model", "")), str(r.get("type", "model"))) for r in rows]

    fig_h = max(4.5, 0.34 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    y = np.arange(len(rows), dtype=np.float64)
    ax.barh(y, values, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean gap [%]")
    ax.set_title("Models vs heuristic baselines on TSPLIB")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    for yi, v in zip(y, values):
        ax.text(v + 0.08, yi, f"{v:.2f}", va="center", ha="left", fontsize=8)
    out = plot_dir / "baseline_vs_models.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[analyze] wrote {out}")


def _plot_decode_profile(plot_dir: Path, prof_rows: list[dict[str, Any]], logger) -> None:
    import matplotlib.pyplot as plt

    rows = [
        r
        for r in prof_rows
        if _safe_float(r.get("baseline_mean_gap_pct")) is not None
        and _safe_float(r.get("optimized_mean_gap_pct")) is not None
    ]
    if not rows:
        return
    rows.sort(key=lambda r: float(r["optimized_mean_gap_pct"]))

    labels = [_short_label(str(r["exp"])) for r in rows]
    baseline = np.asarray([float(r["baseline_mean_gap_pct"]) for r in rows], dtype=np.float64)
    optimized = np.asarray([float(r["optimized_mean_gap_pct"]) for r in rows], dtype=np.float64)
    y = np.arange(len(rows), dtype=np.float64)

    fig_h = max(4.5, 0.5 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    for yi, b, o in zip(y, baseline, optimized):
        ax.plot([b, o], [yi, yi], color="#94a3b8", lw=1.4, zorder=1)
    ax.scatter(baseline, y, s=36, color="#6b7280", label="Baseline decode", zorder=2)
    ax.scatter(optimized, y, s=36, color="#1d4ed8", label="Optimized decode", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean gap [%]")
    ax.set_title("Effect of optimized decoding across experiments")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.legend(loc="lower right", frameon=False)
    ax.set_axisbelow(True)
    out = plot_dir / "decode_profile_compare.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[analyze] wrote {out}")


def _plot_instance_heatmap(
    plot_dir: Path,
    matrix_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    top_k: int,
    logger,
) -> None:
    import matplotlib.pyplot as plt

    ranked = [r for r in ranking_rows if str(r.get("exp", "")).strip()]
    if not ranked or not matrix_rows:
        return
    ranked.sort(
        key=lambda r: (
            float(r["mean_rank"]) if _safe_float(r.get("mean_rank")) is not None else 1e9,
            float(r["primary_mean_gap_pct"]) if _safe_float(r.get("primary_mean_gap_pct")) is not None else 1e9,
        )
    )
    exps = [str(r["exp"]) for r in ranked[: max(1, top_k)]]

    rows = sorted(
        matrix_rows,
        key=lambda r: float(r["mean_gap_pct"]) if _safe_float(r.get("mean_gap_pct")) is not None else -1.0,
        reverse=True,
    )
    instances = [str(r["instance"]).replace(".npz", "") for r in rows]
    data = np.full((len(rows), len(exps)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, exp in enumerate(exps):
            val = _safe_float(row.get(exp))
            if val is not None:
                data[i, j] = val
    if not np.isfinite(data).any():
        return

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#f3f4f6")

    fig_w = max(8.5, 1.1 * len(exps) + 2.5)
    fig_h = max(5.0, 0.42 * len(instances) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(exps)))
    ax.set_xticklabels([_short_label(x) for x in exps], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(instances)))
    ax.set_yticklabels(instances, fontsize=8)
    ax.set_title(f"Per-instance TSPLIB gap for top {len(exps)} models")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Gap [%]")
    out = plot_dir / "instance_gap_heatmap_top_models.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[analyze] wrote {out}")


def _plot_best_history(plot_dir: Path, rows: list[dict[str, Any]], primary_eval: str, logger) -> None:
    import matplotlib.pyplot as plt

    key = primary_eval.replace(".json", "").replace(".", "_") + "_gap_mean"
    best_row = None
    best_gap = None
    for row in rows:
        gap = _safe_float(row.get(key))
        if gap is None:
            continue
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_row = row
    if best_row is None:
        return

    run_dir = Path(str(best_row.get("run_dir", "")))
    hist_path = run_dir / "train_history.csv"
    history = _load_history_rows(hist_path)
    if not history:
        return

    epochs = np.asarray([r["epoch"] for r in history], dtype=np.float64)
    train_loss = np.asarray([r["train_loss"] for r in history], dtype=np.float64)
    val_loss = np.asarray([r["val_loss"] for r in history], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(epochs, train_loss, color="#1d4ed8", lw=1.8, label="Train loss")
    ax.plot(epochs, val_loss, color="#dc2626", lw=1.8, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training curve for best model: {_short_label(str(best_row.get('exp', '')))}")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    out = plot_dir / "best_model_history.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[analyze] wrote {out}")


def generate_analysis_plots(
    cfg: AnalyzeCfg,
    logger,
    rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    prof_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    board_rows: list[dict[str, Any]],
) -> None:
    if not bool(cfg.make_plots):
        return

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as _  # noqa: F401
    except Exception as exc:
        logger.warning(f"[analyze] plots skipped: matplotlib unavailable ({exc})")
        return

    plot_dir = Path(cfg.plots_dir) if cfg.plots_dir else Path(cfg.out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _plot_model_ranking(plot_dir, ranking_rows, logger)
    _plot_baseline_board(plot_dir, board_rows, logger)
    _plot_decode_profile(plot_dir, prof_rows, logger)
    _plot_instance_heatmap(plot_dir, matrix_rows, ranking_rows, int(cfg.heatmap_top_k), logger)
    _plot_best_history(plot_dir, rows, cfg.primary_eval, logger)
