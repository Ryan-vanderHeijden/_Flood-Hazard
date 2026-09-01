"""Validate the LP3/EMA pipeline against USGS-published peak-flow statistics.

The StreamStats gage service publishes, for most long-record gages, the USGS's
own at-site flood frequency estimates (``PK*AEP``). Those come from the same
kind of Bulletin 17C analysis this pipeline performs, on largely the same peak
records, so they are a genuine reference: our fitted quantiles should reproduce
them to within the sampling and methodological noise of the two analyses.

Two checks are run, and they are deliberately different in kind:

**External** — our LP3 quantile against the published station estimate at the
same AEP, in log10 space, for the eight AEPs StreamStats publishes. Reported as
R², RMSE, median ratio and the P10–P90 ratio spread. Disagreement here is
two-sided: the published analysis was frozen at some earlier date with its own
record period, skew and censoring choices, so a mismatch does not by itself
convict our fit.

**Internal** — the fitted 2-year flow against the *empirical median annual
peak* at the same site. By definition those are the same quantity, so this
needs no external reference and no assumption about anyone's method. It is the
check that isolates fit failures from ordinary methodological disagreement, and
it covers the ~40% of sites with no published estimate to compare against.

Writes ``data/ffa/streamstats_validation.parquet`` — one row per site, holding
both checks plus the record and fit descriptors needed to stratify them.

**Known issue this validation found.** The report is currently dominated by a
peak-code mapping error in ``compute_flood_frequency.py``: NWIS code 6 is
"Discharge affected by Regulation or Diversion", but the fitting code treats it
as a left-censored observation. That collapses the lower tail at every site with
code-6 peaks and accounts for 94% of the total squared Q2 error against the
published estimates. See ``code/ffa_analysis/peak_cd_notes.md``. Until the
mapping is fixed and the pipeline re-run, read the Q2 row of the external table
as a measurement of that defect rather than of the method.

    python code/ffa_analysis/src/validate_against_streamstats.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_standard_quantiles import compute_standard_quantiles  # noqa: E402

DATA_DIR = Path.home() / "data" / "flood_hazard"
FFA_DIR = DATA_DIR / "ffa"
META_DIR = DATA_DIR / "metadata"
OUT_PATH = FFA_DIR / "streamstats_validation.parquet"

# The AEPs StreamStats publishes, as return periods. Our LP3 parameters are
# evaluated at exactly these so the comparison is like-for-like.
RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 200, 500]

# Peak qualification codes excluded from the empirical median: 1 is a maximum
# daily average rather than an instantaneous peak, 8 is stage with no discharge.
# This mirrors the exclusions the fitting code applies.
DROP_CODES = ("1", "8")

# A fitted 2-year flow this far from the empirical median annual peak is a fit
# failure, not a difference of opinion: 0.1 in log10 is a factor of 1.26.
Q2_TOLERANCE_LOG = 0.10

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _pad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["site_no"] = df["site_no"].astype(str).str.zfill(8)
    return df


def empirical_median_peak(peaks: pd.DataFrame) -> pd.DataFrame:
    """Median observed annual peak per site — the empirical 2-year flood.

    Code-6 (left-censored) peaks are kept: their recorded value is an upper
    bound on that year's peak, so including them biases the median slightly
    high, which is the conservative direction for a check that looks for
    fitted quantiles falling too low.
    """
    cd = peaks["peak_cd"].fillna("").astype(str).str.upper()
    keep = peaks["peak_va"].notna() & (peaks["peak_va"] > 0)
    for code in DROP_CODES:
        keep &= ~cd.str.contains(code, regex=False)
    out = (
        peaks[keep]
        .groupby("site_no")["peak_va"]
        .agg(emp_median_peak="median", n_peaks_observed="size")
        .reset_index()
    )
    return _pad(out)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison(
    ffa: pd.DataFrame,
    streamstats: pd.DataFrame,
    peaks: pd.DataFrame,
    return_periods: list[int] = RETURN_PERIODS,
) -> pd.DataFrame:
    """One row per QC-passed site: our quantiles, published quantiles, checks."""
    ffa = _pad(ffa)
    fitted = _pad(compute_standard_quantiles(ffa, return_periods))

    keep = [
        "site_no", "n_peaks", "n_censored", "n_pilf", "n_hist", "n_dropped",
        "lp3_skew", "lp3_weighted_skew", "lp3_scale", "lp3_loc",
        "high_censoring", "perception_threshold_cfs", "state_cd",
    ]
    df = fitted.merge(ffa[[c for c in keep if c in ffa.columns]], on="site_no")

    ss_cols = ["site_no"] + [
        c for c in streamstats.columns if c.startswith(("ss_station_", "ss_regression_"))
    ]
    df = df.merge(_pad(streamstats)[ss_cols], on="site_no", how="left")
    df = df.merge(empirical_median_peak(peaks), on="site_no", how="left")

    # Share of the effective record that the EMA had to treat as censored.
    df["frac_censored"] = df["n_censored"] / (df["n_peaks"] + df["n_censored"])

    # External check, per return period, as a log10 residual (ours − published).
    for rp in return_periods:
        ours, pub = df[f"q{rp}_cfs"], df.get(f"ss_station_q{rp}_cfs")
        if pub is None:
            continue
        ok = ours.notna() & pub.notna() & (ours > 0) & (pub > 0)
        df[f"resid_q{rp}"] = np.where(ok, np.log10(ours) - np.log10(pub), np.nan)

    # Internal check: fitted Q2 against the empirical median annual peak.
    ok = df["emp_median_peak"].gt(0) & df["q2_cfs"].gt(0)
    df["q2_vs_empirical"] = np.where(ok, df["q2_cfs"] / df["emp_median_peak"], np.nan)
    df["q2_check_failed"] = (
        np.log10(df["q2_vs_empirical"]).abs() > Q2_TOLERANCE_LOG
    ).fillna(False)

    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def agreement(ours: pd.Series, published: pd.Series) -> dict:
    """Log-space agreement between two sets of discharge estimates."""
    ok = ours.notna() & published.notna() & (ours > 0) & (published > 0)
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    o, p = np.log10(ours[ok]), np.log10(published[ok])
    resid = o - p
    ratio = ours[ok] / published[ok]
    return {
        "n": int(ok.sum()),
        "r2": float(1 - (resid**2).sum() / ((o - o.mean()) ** 2).sum()),
        "rmse_log": float(np.sqrt((resid**2).mean())),
        "median_ratio": float(ratio.median()),
        "p10_ratio": float(ratio.quantile(0.10)),
        "p90_ratio": float(ratio.quantile(0.90)),
        "within_10pct": float((ratio.sub(1).abs() < 0.10).mean()),
        "within_25pct": float((ratio.sub(1).abs() < 0.25).mean()),
    }


def agreement_table(
    df: pd.DataFrame,
    kind: str = "station",
    return_periods: list[int] = RETURN_PERIODS,
) -> pd.DataFrame:
    """Agreement metrics at every published return period."""
    rows = []
    for rp in return_periods:
        pub = f"ss_{kind}_q{rp}_cfs"
        if pub not in df.columns:
            continue
        rows.append({"return_period_yr": rp, **agreement(df[f"q{rp}_cfs"], df[pub])})
    return pd.DataFrame(rows).set_index("return_period_yr")


def stratified(df: pd.DataFrame, by: pd.Series, rp: int = 2) -> pd.DataFrame:
    """Q2 agreement and the internal check, split by an arbitrary grouping."""
    g = df.assign(_g=by).groupby("_g", observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "rmse_log": g[f"resid_q{rp}"].apply(lambda x: float(np.sqrt((x**2).mean()))),
        "median_resid": g[f"resid_q{rp}"].median(),
        "q2_vs_empirical": g["q2_vs_empirical"].median(),
        "pct_check_failed": g["q2_check_failed"].mean() * 100,
    })


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(df: pd.DataFrame) -> None:
    n_pub = df.filter(like="ss_station_q").notna().any(axis=1).sum()
    log.info("QC-passed sites: %d (%d with a published station estimate)", len(df), n_pub)

    print("\n=== External: our LP3 quantiles vs USGS published station estimates ===")
    print(agreement_table(df).round(3).to_string())

    print("\n=== Internal: fitted Q2 vs empirical median annual peak ===")
    ratio = df["q2_vs_empirical"].dropna()
    print(f"  n = {len(ratio)}   median = {ratio.median():.3f}   "
          f"P10 = {ratio.quantile(.1):.3f}   P90 = {ratio.quantile(.9):.2f}")
    print(f"  fitted Q2 below half the empirical median: {(ratio < 0.5).sum()} sites")
    print(f"  outside +/-{Q2_TOLERANCE_LOG} log10 (a factor of "
          f"{10**Q2_TOLERANCE_LOG:.2f}): {int(df['q2_check_failed'].sum())} sites "
          f"({df['q2_check_failed'].mean() * 100:.1f}%)")

    print("\n=== Where the disagreement sits ===")
    scale_bin = pd.cut(df["lp3_scale"], [0, 0.6, np.inf], labels=["<=0.6", ">0.6"])
    cens_bin = np.where(df["frac_censored"].fillna(0) > 0, "censored", "uncensored")
    print(stratified(df, scale_bin.astype(str) + " / " + cens_bin).round(3).to_string())

    print("\n=== Q2 error concentration ===")
    e = df["resid_q2"].dropna()
    for cut in (0.6, 1.0):
        bad = df.loc[e.index, "lp3_scale"] > cut
        print(f"  lp3_scale > {cut}: {bad.sum():>4} of {len(e)} compared sites "
              f"hold {(e[bad]**2).sum() / (e**2).sum() * 100:.0f}% of the squared error")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    ffa = pd.read_parquet(args.data_dir / "ffa" / "flood_frequency.parquet")
    peaks = pd.read_parquet(args.data_dir / "ffa" / "annual_peaks.parquet")
    ss = pd.read_parquet(args.data_dir / "metadata" / "streamstats_peaks.parquet")

    df = build_comparison(ffa, ss, peaks)
    report(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("Saved %d rows -> %s", len(df), args.out)


if __name__ == "__main__":
    main()
