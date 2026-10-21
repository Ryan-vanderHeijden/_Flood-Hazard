"""Refit LP3 on the recent record at unregulated gages that are not stationary.

The unregulated counterpart of refit_post_cutoff.py. Unregulated gages with a
significant trend or step change (compute_peak_trends.py) have no dam dates,
so the start of the recent record is

  - the **Pettitt change year**, where the step change is significant; the
    changes here fall mid-record (median 1979), where Pettitt locates them
    well (its late placement affects changes near the record start; see
    nonstationarity_notes.md);
  - otherwise (trend only) the **most recent** ``RECENT_YEARS`` systematic
    years. Under a steady trend this estimates conditions about half a
    window ago, still far closer to the present than the full-record mean.

Sites with at least ``MIN_POST`` systematic years from the cutoff are refit
with the pipeline's own worker and re-screened with the same Mann–Kendall
test; a surviving trend (BH FDR q = ``FDR_Q`` across these refits) is flagged
``gradual_change``. Step sites with fewer years left are "too_short" and
flagged ``nonstationary_full_record``: a 30-year window would reach back
across the change.

**Role.** Bulletin 17C treats urbanization as a watershed change, for which
the recent record is the right basis, so at urbanizing basins (GAGES-II
developed land ≥ ``URBAN_DEV``%, NLCD 2006) the refit is the **primary**
present-day value. Elsewhere the cause is unattributed (climate, drainage,
groundwater, channel or rating changes); Bulletin 17C keeps the stationary
fit, and the refit is a **sensitivity** value only.

Writes ``data/ffa/unregulated_recent_fits.parquet`` — one row per
unregulated site with a significant result.

    python code/ffa_analysis/src/refit_unregulated_recent.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_flood_frequency import (  # noqa: E402
    _LEVELS, _RATING_MAX_FLOW, _THRESHOLD_FLOWS, _apply_weighted_skew, _load_site_info,
)
from compute_peak_trends import FDR_Q, bh_reject, systematic_peaks  # noqa: E402
from refit_post_cutoff import MIN_POST, refit  # noqa: E402

DATA_DIR = Path.home() / "data" / "flood_hazard"
OUT_PATH = DATA_DIR / "ffa" / "unregulated_recent_fits.parquet"
GAGESII_LC = DATA_DIR / "gagesii" / "csv" / "conterm_lc06_basin.txt"

RECENT_YEARS = 30     # window for trend-only sites
URBAN_DEV = 20        # % developed land for "urbanizing" (as in make_nonstationarity_figures)

log = logging.getLogger(__name__)


def developed_pct() -> pd.Series:
    lc = pd.read_csv(GAGESII_LC, dtype={"STAID": str}, usecols=["STAID", "DEVNLCD06"],
                     encoding="latin1")
    return lc.assign(site_no=lc["STAID"].str.zfill(8)).set_index("site_no")["DEVNLCD06"]


def choose_cutoffs(sites: pd.DataFrame, sys_wy: pd.Series) -> pd.DataFrame:
    """cutoff_wy (first year kept), cutoff_source and systematic years kept."""
    recent = sites["site_no"].map(
        lambda s: sys_wy[s][-RECENT_YEARS] if len(sys_wy[s]) >= RECENT_YEARS else np.nan)
    cutoff = sites["change_wy"].where(sites["pettitt_sig"], recent)
    src = np.where(sites["pettitt_sig"], "pettitt", "recent30")
    n_post = [int((sys_wy[s] >= c).sum()) for s, c in zip(sites["site_no"], cutoff)]
    return sites.assign(cutoff_wy=cutoff, cutoff_source=src, n_sys_post=n_post)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    ffa_dir, meta_dir = args.data_dir / "ffa", args.data_dir / "metadata"

    trends = pd.read_parquet(ffa_dir / "peak_trends.parquet")
    sites = trends[trends["tested"] & trends["is_regulated"].eq(False)
                   & (trends["mk_sig"] | trends["pettitt_sig"])].copy()
    sites["kind"] = np.select([sites["mk_sig"] & sites["pettitt_sig"], sites["mk_sig"]],
                              ["trend + step", "trend only"], "step only")
    size = sites["change_pct"].where(sites["pettitt_sig"], sites["sen_pct_decade"])
    sites["direction"] = np.where(size > 0, "up", "down")
    sites["developed_pct"] = sites["site_no"].map(developed_pct())
    sites["urbanizing"] = sites["developed_pct"] >= URBAN_DEV
    sites["role"] = np.where(sites["urbanizing"], "primary", "sensitivity")

    peaks = pd.read_parquet(ffa_dir / "annual_peaks.parquet")
    peaks = peaks[peaks["site_no"].isin(sites["site_no"])]
    sys_wy = systematic_peaks(peaks).groupby("site_no")["wy"].apply(lambda w: np.sort(w.to_numpy()))
    sites = choose_cutoffs(sites, sys_wy)
    sites["status"] = np.where(sites["n_sys_post"] >= MIN_POST, "refit", "too_short")
    sites["nonstationary_full_record"] = sites["status"].eq("too_short")

    fs = pd.read_parquet(meta_dir / "flood_stages.parquet")
    keep = ["site_no"] + _THRESHOLD_FLOWS + [c for c in fs.columns
                                             if c.endswith("_source") or c == _RATING_MAX_FLOW]
    fs = fs[keep].drop_duplicates("site_no")
    fs_idx = fs.set_index("site_no")
    by_site = dict(tuple(peaks.groupby("site_no")))

    fits, screens = [], []
    for r in sites[sites["status"] == "refit"].itertuples():
        flows = tuple(float(fs_idx.at[r.site_no, c]) if r.site_no in fs_idx.index
                      and pd.notna(fs_idx.at[r.site_no, c]) else float("nan")
                      for c in _THRESHOLD_FLOWS)
        f, s = refit(r.site_no, by_site[r.site_no], int(r.cutoff_wy), flows)
        fits.append(f)
        screens.append(s)
    log.info("Refit %d of %d unregulated sites with a significant result", len(fits), len(sites))

    res = _apply_weighted_skew(pd.DataFrame(fits), fs, _load_site_info(meta_dir),
                               pd.DataFrame(columns=["state_cd", "regional_skew_g",
                                                     "mse_regional_g"]), "station")
    res["degenerate_fit"] = ((res["lp3_loc"] < 0.5) | (res["lp3_scale"] > 2.0)
                             | (res["lp3_skew"].abs() > 3.0)).fillna(False)
    rp_cols = ([f"{lv}_return_period_yr" for lv in _LEVELS]
               + [f"{lv}_return_period_lb_yr" for lv in _LEVELS])
    fit_cols = ["n_peaks", "lp3_skew", "lp3_loc", "lp3_scale", "degenerate_fit"]
    res = res[["site_no"] + fit_cols + rp_cols].rename(
        columns={c: f"post_{c}" for c in fit_cols + rp_cols})
    scr = pd.DataFrame(screens)
    scr["gradual_change"] = bh_reject(scr["post_mk_p"], FDR_Q)

    full = pd.read_parquet(ffa_dir / "flood_frequency.parquet", columns=["site_no"] + rp_cols)
    out = (sites.merge(res, on="site_no", how="left").merge(scr, on="site_no", how="left")
           .merge(full.rename(columns={c: f"full_{c}" for c in rp_cols}), on="site_no",
                  how="left"))
    report(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    log.info("Saved %d rows -> %s", len(out), args.out)


def report(df: pd.DataFrame) -> None:
    log.info("Sites: %d; role %s; cutoff %s; status %s", len(df),
             df["role"].value_counts().to_dict(), df["cutoff_source"].value_counts().to_dict(),
             df["status"].value_counts().to_dict())
    r = df[(df["status"] == "refit") & ~df["post_degenerate_fit"].fillna(False).astype(bool)]
    log.info("Refit: median %d years kept; %d degenerate; %d gradual_change",
             r["n_post"].median(), df["post_degenerate_fit"].fillna(False).astype(bool).sum(),
             r["gradual_change"].sum())
    for (role, dirn), g in r.groupby(["role", "direction"]):
        parts = []
        for lv in _LEVELS:
            a, b = g[f"post_{lv}_return_period_yr"], g[f"full_{lv}_return_period_yr"]
            ok = a.gt(0) & b.gt(0) & np.isfinite(a) & np.isfinite(b)
            parts.append(f"{lv} {(a[ok] / b[ok]).median():.2f}")
        log.info("  %-11s %-4s n=%3d  median present/full RP: %s", role, dirn, len(g),
                 ", ".join(parts))


if __name__ == "__main__":
    main()
