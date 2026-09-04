from __future__ import annotations

"""
Assemble ``validation_report.ipynb`` from the metrics CSVs and figure pack.

The notebook is built with markdown narrative, metrics tables rendered from the
CSVs, and the pre-generated PNGs referenced by relative path, so it renders
without needing to be executed.  Run ``evaluate.py`` and ``make_figures.py``
first.

Example
-------
    python build_report.py
"""

import logging
from pathlib import Path

import nbformat as nbf
import pandas as pd

logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).resolve().parent
ML_DIR = Path.home() / "data" / "flood_hazard" / "ml"


def _md_table(csv: Path, floatfmt: str = "{:.3f}") -> str:
    """Render a CSV as a GitHub-flavoured markdown table (no tabulate dependency)."""
    df = pd.read_csv(csv, index_col=0)
    fmt = df.map(lambda v: floatfmt.format(v) if isinstance(v, float) else str(v))
    header = "| " + " | ".join([df.index.name or ""] + list(fmt.columns)) + " |"
    sep = "| " + " | ".join(["---"] * (len(fmt.columns) + 1)) + " |"
    rows = [
        "| " + " | ".join([str(idx)] + list(r)) + " |"
        for idx, r in zip(fmt.index, fmt.to_numpy())
    ]
    return "\n".join([header, sep, *rows])


def build() -> Path:
    metrics = _md_table(REPORT_DIR / "metrics_by_rp.csv")
    baseline = _md_table(REPORT_DIR / "baseline_comparison.csv")
    n_conus = pd.read_parquet(ML_DIR / "conus_predictions.parquet", columns=["COMID"]).shape[0]

    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# Ungauged Return-Period Flood-Flow Model — Validation Report\n\n"
            "Predicts LP3 flood quantiles **Q2–Q500** at any CONUS NHDPlus reach from "
            "catchment attributes, then applies the model CONUS-wide.\n\n"
            "**Design.** Target = at-site LP3 quantiles (regenerated with the corrected "
            "peak-code mapping). Features = 55 NHDPlus/COMID *total-upstream* attributes "
            "(Wieczorek *Select Attributes*: drainage area, precip, PET, temperature, BFI, "
            "runoff, recharge, soils, land cover, dam storage). Trained on **1,443 unregulated, "
            "QC-passed gauges** (natural-flow model). Engine = **LightGBM** with an "
            "**index-flood decomposition** (predict log Q2 + non-negative consecutive "
            "log-increments) that guarantees monotone curves Q2 ≤ … ≤ Q500. Validation = "
            "**leave-HUC2-out** spatial cross-validation. Uncertainty = split-conformal bands "
            "calibrated on the out-of-region residuals."
        ),
        nbf.v4.new_markdown_cell(
            "## Skill by return period (leave-HUC2-out)\n\n" + metrics + "\n\n"
            "log-R²/NSE in log₁₀ space; KGE/NSE(cfs) in real space. Skill decays monotonically "
            "toward rarer events, as expected.\n\n"
            "![skill](fig_skill_by_rp.png)\n\n"
            "![pred vs obs](fig_pred_vs_obs_q10.png)"
        ),
        nbf.v4.new_markdown_cell(
            "## Baseline: USGS StreamStats regional regressions\n\n" + baseline + "\n\n"
            "On the ~570 sites with published StreamStats regression estimates, the "
            "state-specific USGS equations still lead (e.g. Q10 log-R² ≈ 0.88 vs ≈ 0.80). "
            "The value here is a single, nationally consistent model with calibrated uncertainty "
            "and full CONUS coverage — including reaches StreamStats regression does not serve."
        ),
        nbf.v4.new_markdown_cell(
            "## Spatial cross-validation residuals\n\n"
            "![spatial](fig_spatial_cv_map.png)\n\n"
            "Residuals are small on average but regionally structured (e.g. Gulf-coast "
            "under-prediction), consistent with prior findings — a target for future "
            "region-aware refinement."
        ),
        nbf.v4.new_markdown_cell(
            "## Feature importance (physical sanity)\n\n"
            "![importance](fig_feature_importance.png)\n\n"
            "The model leans on catchment size (stream length, drainage area), mean-annual "
            "precipitation, base-flow index and runoff — the expected physical drivers."
        ),
        nbf.v4.new_markdown_cell(
            "## Uncertainty\n\n"
            "![interval](fig_interval_width.png)\n\n"
            "90% conformal intervals; empirical out-of-fold coverage ≈ 0.90 by construction. "
            "Bands widen for rarer return periods."
        ),
        nbf.v4.new_markdown_cell(
            f"## CONUS product\n\n"
            f"`ml/conus_predictions.parquet` — **{n_conus:,} COMIDs**, columns "
            "`q{rp}_cfs` / `q{rp}_lo_cfs` / `q{rp}_hi_cfs` for rp ∈ {{2,5,10,25,50,100,200,500}}, "
            "plus `TOT_BASIN_AREA` and `has_upstream_dam`.\n\n"
            "**Caveats.** (1) The model is trained on unregulated gauges, so predictions are "
            "*natural* flood potential; reaches flagged `has_upstream_dam` should be read as such. "
            "(2) Most CONUS reaches are far smaller headwater catchments than the training gauges, "
            "so those predictions are extrapolation — the widened conformal bands express this but "
            "were calibrated at gauge scale.\n\n"
            "**Mapping.** The table is delivered COMID-keyed (no geometry). To map it, join "
            "`COMID` to the **FEATUREID** field of any NHDPlusV2 catchment layer (or `COMID` of the "
            "flowlines) in QGIS/ArcGIS. The national NHDPlus catchment *raster* is not directly "
            "usable — its cells store GRIDCODE, whose crosswalk to COMID ships only in EPA's 7.8 GB "
            "Seamless Geodatabase."
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "conus = pd.read_parquet('~/data/flood_hazard/ml/conus_predictions.parquet',\n"
            "                        columns=['COMID','q10_cfs','q10_lo_cfs','q10_hi_cfs','TOT_BASIN_AREA'])\n"
            "conus.describe()"
        ),
    ]
    nb["cells"] = cells
    out = REPORT_DIR / "validation_report.ipynb"
    nbf.write(nb, out)
    logger.info("Wrote %s", out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build()


if __name__ == "__main__":
    main()
