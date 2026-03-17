from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "thesis" / "figures" / "generated"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _short_name(name: str) -> str:
    s = str(name).strip()
    if s.startswith("exp_edge_"):
        s = s[len("exp_edge_") :]
    return s


def _family_color(row_type: str, model: str) -> str:
    if row_type == "baseline":
        return "#6b7280"
    if model == "edge_transformer":
        return "#1d4ed8"
    if model == "edge_res_mlp":
        return "#059669"
    if model == "edge_mlp_deep":
        return "#d97706"
    if model == "edge_mlp":
        return "#b45309"
    return "#374151"


def _plot_optimized_ranking() -> None:
    rows = _read_csv(ROOT / "runs" / "analysis" / "fair_optimized" / "baseline_vs_models.csv")
    rows.sort(key=lambda r: _f(r, "mean_gap_pct"))

    labels = [_short_name(r["name"]) for r in rows]
    values = np.asarray([_f(r, "mean_gap_pct") for r in rows], dtype=np.float64)
    colors = [_family_color(r["type"], r.get("model", "")) for r in rows]

    fig_h = max(5.2, 0.34 * len(rows) + 1.3)
    fig, ax = plt.subplots(figsize=(10.8, fig_h))
    y = np.arange(len(rows), dtype=np.float64)
    ax.barh(y, values, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Priemerná relatívna odchýlka [%]")
    ax.set_title("Poradie modelov a heuristických báz v optimalizovanom profile")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    for yi, v in zip(y, values):
        ax.text(v + 0.05, yi, f"{v:.2f}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ranking_optimized_sk.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_decode_profile_compare() -> None:
    baseline_rows = _read_csv(ROOT / "runs" / "analysis" / "fair_baseline" / "baseline_vs_models.csv")
    optimized_rows = _read_csv(ROOT / "runs" / "analysis" / "fair_optimized" / "baseline_vs_models.csv")

    base_map = {r["name"]: r for r in baseline_rows}
    opt_map = {r["name"]: r for r in optimized_rows}

    joined: list[dict[str, Any]] = []
    for name, b in base_map.items():
        o = opt_map.get(name)
        if o is None:
            continue
        joined.append(
            {
                "name": name,
                "baseline": _f(b, "mean_gap_pct"),
                "optimized": _f(o, "mean_gap_pct"),
                "improvement": _f(b, "mean_gap_pct") - _f(o, "mean_gap_pct"),
                "type": b["type"],
                "model": b.get("model", ""),
            }
        )
    joined.sort(key=lambda r: r["optimized"])

    labels = [_short_name(r["name"]) for r in joined]
    baseline = np.asarray([r["baseline"] for r in joined], dtype=np.float64)
    optimized = np.asarray([r["optimized"] for r in joined], dtype=np.float64)
    y = np.arange(len(joined), dtype=np.float64)

    fig_h = max(5.5, 0.42 * len(joined) + 1.2)
    fig, ax = plt.subplots(figsize=(10.8, fig_h))
    for yi, b, o in zip(y, baseline, optimized):
        ax.plot([b, o], [yi, yi], color="#94a3b8", lw=1.5, zorder=1)
    ax.scatter(baseline, y, s=38, color="#6b7280", label="Základný profil", zorder=2)
    ax.scatter(optimized, y, s=38, color="#1d4ed8", label="Optimalizovaný profil", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Priemerná relatívna odchýlka [%]")
    ax.set_title("Vplyv dekódovacieho profilu na výslednú kvalitu trás")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "decode_profile_compare_sk.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_data_quality_pairs() -> None:
    rows = _read_csv(ROOT / "runs" / "analysis" / "fair_optimized" / "data_quality_pairs.csv")
    rows.sort(key=lambda r: _f(r, "delta_gap_pct"))

    labels = [_short_name(r["ccv1_exp"]).replace("_ccv1", "") for r in rows]
    delta = np.asarray([_f(r, "delta_gap_pct") for r in rows], dtype=np.float64)
    improvement = -delta
    colors = ["#059669" if x >= 0 else "#dc2626" for x in improvement]

    fig_h = max(4.6, 0.7 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(9.6, fig_h))
    y = np.arange(len(rows), dtype=np.float64)
    ax.barh(y, improvement, color=colors, edgecolor="none")
    ax.axvline(0.0, color="#374151", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Zlepšenie pri použití presných štítkov [p. b.]")
    ax.set_title("Vplyv presnej referenčnej vetvy na kvalitu modelov")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    for yi, v in zip(y, improvement):
        ha = "left" if v >= 0 else "right"
        dx = 0.04 if v >= 0 else -0.04
        ax.text(v + dx, yi, f"{v:.2f}", va="center", ha=ha, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_quality_pairs_sk.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_instance_difficulty() -> None:
    rows = _read_csv(ROOT / "runs" / "analysis" / "fair_optimized" / "instance_gap_matrix.csv")
    rows.sort(key=lambda r: _f(r, "mean_gap_pct"), reverse=True)

    labels = [str(r["instance"]).replace(".npz", "") for r in rows]
    values = np.asarray([_f(r, "mean_gap_pct") for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    x = np.arange(len(rows), dtype=np.float64)
    ax.bar(x, values, color="#1d4ed8", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Priemerná relatívna odchýlka [%]")
    ax.set_title("Náročnosť jednotlivých inštancií TSPLIB v optimalizovanom profile")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "instance_difficulty_sk.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_best_history() -> None:
    rows = _read_csv(ROOT / "runs" / "analysis" / "fair_optimized" / "experiments_summary.csv")
    rows = [r for r in rows if r.get("eval_tsplib_optimized_gap_mean")]
    best = min(rows, key=lambda r: _f(r, "eval_tsplib_optimized_gap_mean"))

    hist_path = Path(best["run_dir"]) / "train_history.csv"
    hist_rows = _read_csv(hist_path)
    epochs = np.asarray([float(r["epoch"]) for r in hist_rows], dtype=np.float64)
    train_loss = np.asarray([float(r["train_loss"]) for r in hist_rows], dtype=np.float64)
    val_loss = np.asarray(
        [float(r["val_loss"]) if r["val_loss"].strip() else np.nan for r in hist_rows],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(epochs, train_loss, color="#1d4ed8", lw=2.0, label="Trénovacia strata")
    ax.plot(epochs, val_loss, color="#dc2626", lw=2.0, label="Validačná strata")
    ax.set_xlabel("Epocha")
    ax.set_ylabel("Strata")
    ax.set_title(f"Priebeh učenia najlepšieho modelu: {_short_name(best['exp'])}")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_model_history_sk.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11

    _plot_optimized_ranking()
    _plot_decode_profile_compare()
    _plot_data_quality_pairs()
    _plot_instance_difficulty()
    _plot_best_history()


if __name__ == "__main__":
    main()
