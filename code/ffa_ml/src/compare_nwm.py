from __future__ import annotations

"""
Benchmark the ungauged ML model against the NWM v3.0 retrospective at gauge reaches.

At every USGS gauge reach we have three independent estimates of the standard
return-period discharges Q2…Q500:

1. **At-site LP3** on the observed USGS annual instantaneous peaks
   (``ffa/streamstats_validation.parquet``) — the training target, treated as truth.
2. **This ML model** — the *out-of-fold* leave-HUC2-out CV predictions
   (``ml/cv_predictions.parquet``), so the model is scored as a genuine ungauged
   predictor (never on gauges it trained on), matching how NWM is inherently
   out-of-sample here.  (The in-sample ``conus_predictions`` would give a misleading
   R² ≈ 0.996 at these very training reaches.)
3. **NWM retrospective** (``nwm/nwm_streamflow.parquet``) — 45 years (1979–2023) of
   *daily mean* simulated streamflow; we take the annual maxima and fit the same LP3
   (log-Pearson III, method of moments) to obtain NWM-implied Q2…Q500.

The module streams the 68.8M-row NWM daily table row-group by row-group (bounded
memory), builds a water-year annual-max series per reach, fits LP3, and writes
``ml/nwm_return_periods.parquet``.  It then joins all three sources on the common
reaches and reports log-space skill (R²/bias/RMSE) of NWM-vs-truth and ML-vs-truth
per return period, plus an obs-vs-estimate scatter figure.

Caveat: NWM retrospective is *daily mean* flow, so its annual maxima systematically
under-represent instantaneous peaks (most in small/flashy basins).  A low bias for
NWM is expected and is reported, not corrected.

Example
-------
    python compare_nwm.py
    python compare_nwm.py --min-years 20
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

logger = logging.getLogger(__name__)

DATA_DIR = Path.home() / "data" / "flood_hazard"
NWM_PATH = DATA_DIR / "nwm" / "nwm_streamflow.parquet"
OBS_PATH = DATA_DIR / "ffa" / "streamstats_validation.parquet"
CV_PRED_PATH = DATA_DIR / "ml" / "cv_predictions.parquet"
REG_PATH = DATA_DIR / "metadata" / "regulation.parquet"
FF_PATH = DATA_DIR / "ffa" / "flood_frequency.parquet"
PPCC_PATH = DATA_DIR / "ffa" / "ppcc.parquet"

OUT_NWM = DATA_DIR / "ml" / "nwm_return_periods.parquet"
OUT_COMPARE = DATA_DIR / "ml" / "nwm_comparison.parquet"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
METRICS_CSV = REPORT_DIR / "nwm_comparison_metrics.csv"
FIG_PATH = REPORT_DIR / "fig_nwm_comparison.png"

RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 200, 500]


def nwm_annual_max(nwm_path: Path) -> pd.DataFrame:
    """Stream the NWM daily table → water-year annual-max streamflow per reach.

    Returns a long frame ``[reach_id, wy, qmax_cfs]`` plus attaches, via the return
    value's ``.attrs['pairs']``, the reach_id→site_no crosswalk observed in the file.
    """
    pf = pq.ParquetFile(nwm_path)
    maxes: dict[tuple[int, int], float] = {}
    pairs: dict[int, str] = {}
    cols = ["site_no", "reach_id", "date", "streamflow_cfs"]
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=cols).to_pandas()
        d = pd.to_datetime(df["date"])
        df["wy"] = d.dt.year + (d.dt.month >= 10).astype("int32")
        g = df.groupby(["reach_id", "wy"], sort=False)["streamflow_cfs"].max()
        for key, val in g.items():
            if key not in maxes or val > maxes[key]:
                maxes[key] = float(val)
        for rid, sno in df[["reach_id", "site_no"]].drop_duplicates().itertuples(index=False):
            pairs.setdefault(int(rid), sno)
        logger.info("  row group %d/%d processed", i + 1, pf.num_row_groups)

    idx = pd.MultiIndex.from_tuples(maxes.keys(), names=["reach_id", "wy"])
    out = pd.DataFrame({"qmax_cfs": list(maxes.values())}, index=idx).reset_index()
    out.attrs["pairs"] = pd.Series(pairs, name="site_no").rename_axis("reach_id").reset_index()
    logger.info("NWM annual-max: %d reach-years across %d reaches",
                len(out), out["reach_id"].nunique())
    return out


def lp3_quantiles(qmax: np.ndarray, return_periods: list[int]) -> dict[int, float] | None:
    """Log-Pearson III quantiles (method of moments) from an annual-max series."""
    q = qmax[np.isfinite(qmax) & (qmax > 0)]
    if q.size < 2:
        return None
    x = np.log10(q)
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    if std == 0:
        return None
    skew = float(stats.skew(x, bias=False))
    out: dict[int, float] = {}
    for t in return_periods:
        xp = stats.pearson3.ppf(1.0 - 1.0 / t, skew, loc=mean, scale=std)
        out[t] = float(10.0 ** xp)
    return out


def fit_nwm_return_periods(ams: pd.DataFrame, min_years: int) -> pd.DataFrame:
    """Fit LP3 per reach on the NWM annual-max series."""
    rows: list[dict] = []
    for rid, grp in ams.groupby("reach_id", sort=False):
        n = grp["qmax_cfs"].notna().sum()
        if n < min_years:
            continue
        qs = lp3_quantiles(grp["qmax_cfs"].to_numpy(), RETURN_PERIODS)
        if qs is None:
            continue
        row = {"reach_id": int(rid), "n_years": int(n)}
        row.update({f"nwm_q{t}_cfs": qs[t] for t in RETURN_PERIODS})
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.merge(ams.attrs["pairs"], on="reach_id", how="left")
    logger.info("Fitted NWM LP3 at %d reaches (min_years=%d)", len(out), min_years)
    return out


def _qc_sites() -> pd.Index:
    """site_no passing the same QC gate used for training targets."""
    ff = pd.read_parquet(FF_PATH)
    ppcc = pd.read_parquet(PPCC_PATH, columns=["site_no", "ppcc_ok"])
    ff = ff.merge(ppcc, on="site_no", how="left")
    ok = (
        ff.get("record_ok", True)
        & ~ff.get("degenerate_fit", False)
        & ff["ppcc_ok"].fillna(False)
        & ~ff.get("high_censoring", False)
    )
    return pd.Index(ff.loc[ok, "site_no"].unique())


def build_comparison(nwm_rp: pd.DataFrame) -> pd.DataFrame:
    """Join NWM, observed at-site LP3, and ML predictions on common reaches."""
    obs = pd.read_parquet(
        OBS_PATH, columns=["site_no"] + [f"q{t}_cfs" for t in RETURN_PERIODS] + ["q2_check_failed"]
    ).rename(columns={f"q{t}_cfs": f"obs_q{t}_cfs" for t in RETURN_PERIODS})
    # ML side: out-of-fold spatial-CV predictions (log10) → cfs; join on site_no.
    cv = pd.read_parquet(
        CV_PRED_PATH, columns=["site_no"] + [f"pred_log_q{t}" for t in RETURN_PERIODS]
    )
    for t in RETURN_PERIODS:
        cv[f"ml_q{t}_cfs"] = 10.0 ** cv[f"pred_log_q{t}"]
    pred = cv[["site_no"] + [f"ml_q{t}_cfs" for t in RETURN_PERIODS]]
    reg = pd.read_parquet(REG_PATH, columns=["site_no", "is_regulated"])

    df = nwm_rp.merge(obs, on="site_no", how="inner").merge(pred, on="site_no", how="left")
    df = df.merge(reg, on="site_no", how="left")
    df["is_regulated"] = df["is_regulated"].fillna(False)
    df["qc_ok"] = df["site_no"].isin(_qc_sites()) & ~df["q2_check_failed"].fillna(True)
    logger.info("Comparison frame: %d reaches (%d QC-ok, %d unregulated QC-ok)",
                len(df), int(df["qc_ok"].sum()),
                int((df["qc_ok"] & ~df["is_regulated"]).sum()))
    return df


def _skill(obs: np.ndarray, est: np.ndarray) -> dict[str, float]:
    """Log10-space skill of an estimate against observed."""
    m = np.isfinite(obs) & np.isfinite(est) & (obs > 0) & (est > 0)
    lo, le = np.log10(obs[m]), np.log10(est[m])
    resid = le - lo
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((lo - lo.mean()) ** 2))
    return {
        "n": int(m.sum()),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "bias_dex": float(resid.mean()),
        "rmse_dex": float(np.sqrt((resid ** 2).mean())),
        "median_ratio": float(10.0 ** np.median(resid)),
    }


def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-RP skill for NWM and ML against the at-site LP3 truth (unregulated QC set)."""
    sub = df[df["qc_ok"] & ~df["is_regulated"]]
    rows: list[dict] = []
    for t in RETURN_PERIODS:
        obs = sub[f"obs_q{t}_cfs"].to_numpy()
        for name in ("nwm", "ml"):
            s = _skill(obs, sub[f"{name}_q{t}_cfs"].to_numpy())
            rows.append({"return_period": t, "source": name, **s})
    out = pd.DataFrame(rows).set_index(["return_period", "source"])
    return out


def make_figure(df: pd.DataFrame, out: Path) -> None:
    """Obs-vs-estimate scatter (Q10 and Q100) for NWM and ML on the unregulated QC set."""
    sub = df[df["qc_ok"] & ~df["is_regulated"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, t in zip(axes, (10, 100)):
        obs = sub[f"obs_q{t}_cfs"].to_numpy()
        for name, color in (("nwm", "#d1495b"), ("ml", "#2e86ab")):
            est = sub[f"{name}_q{t}_cfs"].to_numpy()
            m = np.isfinite(obs) & np.isfinite(est) & (obs > 0) & (est > 0)
            ax.scatter(obs[m], est[m], s=5, alpha=0.35, c=color, linewidths=0,
                       label=f"{name.upper()} (n={m.sum()})")
        lims = [np.nanmin(obs[obs > 0]), np.nanmax(obs)]
        ax.plot(lims, lims, "k-", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"At-site LP3 Q{t} (cfs)")
        ax.set_ylabel(f"Estimated Q{t} (cfs)")
        ax.set_title(f"Q{t}: NWM vs ML against at-site LP3")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-years", type=int, default=15,
                        help="minimum NWM annual-max years to fit LP3 (default 15)")
    parser.add_argument("--nwm", type=Path, default=NWM_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if OUT_NWM.exists():
        logger.info("Reusing cached %s", OUT_NWM)
        nwm_rp = pd.read_parquet(OUT_NWM)
    else:
        ams = nwm_annual_max(args.nwm)
        nwm_rp = fit_nwm_return_periods(ams, args.min_years)
        nwm_rp.to_parquet(OUT_NWM, index=False)
        logger.info("Wrote %s", OUT_NWM)

    df = build_comparison(nwm_rp)
    df.to_parquet(OUT_COMPARE, index=False)
    logger.info("Wrote %s", OUT_COMPARE)

    metrics = metrics_table(df)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_CSV)
    logger.info("Wrote %s\n%s", METRICS_CSV, metrics.to_string())
    make_figure(df, FIG_PATH)


if __name__ == "__main__":
    main()
