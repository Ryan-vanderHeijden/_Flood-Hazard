from __future__ import annotations

"""
Generate the validation figure pack for the ungauged flood-flow model.

Reads the spatial-CV out-of-fold predictions, the metrics/baseline/importance
CSVs produced by ``evaluate.py`` and the per-site coordinates, and writes PNGs
into this directory:

  * ``fig_skill_by_rp.png``       — log-R^2 / NSE / KGE vs return period, with baseline,
  * ``fig_pred_vs_obs_q10.png``   — out-of-fold predicted vs observed log10 Q10,
  * ``fig_spatial_cv_map.png``    — leave-HUC2-out Q10 residuals at each gauge,
  * ``fig_feature_importance.png``— top LightGBM gain importances,
  * ``fig_interval_width.png``    — conformal interval half-width vs return period.

Example
-------
    python make_figures.py
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ML_DIR = Path.home() / "data" / "flood_hazard" / "ml"
REPORT_DIR = Path(__file__).resolve().parent
RPS = [2, 5, 10, 25, 50, 100, 200, 500]


def _load() -> dict:
    return {
        "cv": pd.read_parquet(ML_DIR / "cv_predictions.parquet"),
        "targets": pd.read_parquet(ML_DIR / "targets.parquet"),
        "metrics": pd.read_csv(REPORT_DIR / "metrics_by_rp.csv", index_col=0),
        "baseline": pd.read_csv(REPORT_DIR / "baseline_comparison.csv", index_col=0),
        "importance": pd.read_csv(REPORT_DIR / "feature_importance.csv"),
        "offsets": json.loads((ML_DIR / "conformal_offsets.json").read_text()),
    }


def fig_skill_by_rp(d: dict) -> None:
    m, b = d["metrics"], d["baseline"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(RPS))
    ax.plot(x, m["log_r2"], "o-", label="model log-R²", color="#1b6ca8")
    ax.plot(x, m["log_nse"], "s--", label="model log-NSE", color="#5aa9e6", alpha=0.8)
    ax.plot(x, m["cfs_kge"], "^:", label="model KGE (cfs)", color="#f4a259")
    ax.plot(x, b["baseline_log_r2"], "d-", label="StreamStats reg. log-R²", color="#b23a48")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Q{rp}" for rp in RPS])
    ax.set_ylim(0, 1)
    ax.set_xlabel("return period")
    ax.set_ylabel("skill")
    ax.set_title("Leave-HUC2-out skill by return period")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fig_skill_by_rp.png", dpi=140)
    plt.close(fig)


def fig_pred_vs_obs(d: dict) -> None:
    cv = d["cv"]
    o, p = cv["obs_log_q10"], cv["pred_log_q10"]
    fig, ax = plt.subplots(figsize=(5, 5))
    hb = ax.hexbin(o, p, gridsize=45, cmap="viridis", mincnt=1)
    lim = [min(o.min(), p.min()), max(o.max(), p.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("observed log₁₀ Q10 (cfs)")
    ax.set_ylabel("predicted log₁₀ Q10 (cfs)")
    ax.set_title("Out-of-fold Q10 (leave-HUC2-out)")
    fig.colorbar(hb, ax=ax, label="gauges")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fig_pred_vs_obs_q10.png", dpi=140)
    plt.close(fig)


def fig_spatial_map(d: dict) -> None:
    cv = d["cv"].merge(
        d["targets"][["site_no", "latitude", "longitude"]], on="site_no", how="left"
    )
    resid = cv["pred_log_q10"] - cv["obs_log_q10"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sc = ax.scatter(
        cv["longitude"], cv["latitude"], c=resid, cmap="RdBu_r",
        vmin=-0.5, vmax=0.5, s=12, edgecolor="none",
    )
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Leave-HUC2-out Q10 residual (pred − obs, log₁₀) at training gauges")
    fig.colorbar(sc, ax=ax, label="log₁₀ residual")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fig_spatial_cv_map.png", dpi=140)
    plt.close(fig)


def fig_importance(d: dict) -> None:
    imp = d["importance"].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp["feature"], imp["gain_fraction"], color="#1b6ca8")
    ax.set_xlabel("gain importance (fraction)")
    ax.set_title("Top 15 features (LightGBM gain, summed over components)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fig_feature_importance.png", dpi=140)
    plt.close(fig)


def fig_interval_width(d: dict) -> None:
    off = d["offsets"]["offsets"]
    width = [off[f"q{rp}"]["median_width_log10"] for rp in RPS]
    cov = [off[f"q{rp}"]["empirical_coverage"] for rp in RPS]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(len(RPS)), width, color="#5aa9e6")
    ax.set_xticks(range(len(RPS)))
    ax.set_xticklabels([f"Q{rp}" for rp in RPS])
    ax.set_ylabel("90% interval width (log₁₀ cfs)")
    ax.set_title(f"Conformal interval width (empirical coverage ≈ {np.mean(cov):.2f})")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fig_interval_width.png", dpi=140)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    d = _load()
    for fn in (fig_skill_by_rp, fig_pred_vs_obs, fig_spatial_map, fig_importance, fig_interval_width):
        fn(d)
        logger.info("wrote %s", fn.__name__)
    logger.info("Figures written to %s", REPORT_DIR)


if __name__ == "__main__":
    main()
