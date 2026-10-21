"""Refit LP3 on the present-day-regulation record at regulated gages.

Bulletin 17C's answer to a record changed by regulation is to fit only the
homogeneous part of it, the years under present-day conditions. For each
tested regulated site (compute_peak_trends.py), the start of that period is

  1. the later of two physical dates, where either exists:
     - **storage**: the completion year of the largest upstream NID dam
       (by storage, among dams with a completion year), the way USGS station
       reports date a regulated period from closure of the main reservoir;
       counted only where dams completed during the record add DOR ≥
       ``DOR_MIN`` (a small late dam should not cut the record). The cutoff
       is the year after, the first full water year with the dam in place.
       ``--storage-frac F`` instead uses the year dated storage reached F of
       today's (the sensitivity case; 0.9 gives nearly the same cutoffs, see
       nonstationarity_notes.md)
     - **code 6**: the first water year NWIS flags a peak as affected by
       regulation or diversion, counted only when it is more than
       ``MID_RECORD`` years into the record (validated against NID dam years
       in nonstationarity_notes.md);
  2. otherwise the Pettitt change year, where the change is significant;
  3. otherwise no cutoff: the full-record fit stands.

A date within ``MID_RECORD`` years of the record start means the whole record
is already post-regulation, so it does not cut anything.

Sites with at least ``MIN_POST`` systematic years after the cutoff are refit
with the pipeline's own worker (same peak classification, MGBT, EMA, station
skew and above-rating lower bounds as compute_flood_frequency.py). The post-
cutoff record is then re-screened with the same Mann–Kendall test; a trend
that survives (BH FDR q = ``FDR_Q`` across refit sites) means the change is
gradual and the cutoff did not reach a homogeneous period.

Writes ``data/ffa/post_cutoff_fits.parquet`` — one row per tested regulated
site, with the cutoff, its source, the refit return periods and the full-
record ones beside them.

    python code/ffa_analysis/src/refit_post_cutoff.py
    python code/ffa_analysis/src/refit_post_cutoff.py --storage-frac 0.9 --out /tmp/x.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attribute_change_points import DOR_MIN, dams_in_basins, load_dams  # noqa: E402
from compute_flood_frequency import (  # noqa: E402
    _LEVELS, _RATING_MAX_FLOW, _THRESHOLD_FLOWS, _apply_weighted_skew,
    _fit_site_worker, _load_site_info, _MIN_PEAKS, _parse_peak_cd,
)
from compute_peak_trends import FDR_Q, bh_reject, mann_kendall_hr, systematic_peaks  # noqa: E402

DATA_DIR = Path.home() / "data" / "flood_hazard"
OUT_PATH = DATA_DIR / "ffa" / "post_cutoff_fits.parquet"

MID_RECORD = 5        # a date this close to the record start cuts nothing
MIN_POST = 30         # systematic years needed after the cutoff to refit

log = logging.getLogger(__name__)


def water_year(dt: pd.Series) -> pd.Series:
    return dt.dt.year + (dt.dt.month >= 10)


def storage_year(pairs: pd.DataFrame, frac: float | None = None) -> pd.Series:
    """Per site, the completion year of the largest dated dam, or with frac the
    year by which that share of dated storage existed."""
    d = (pairs.dropna(subset=["yearCompleted"]).reset_index(drop=True)
              .sort_values(["site_no", "yearCompleted"]))
    if frac is None:
        return d.loc[d.groupby("site_no")["nidStorage"].idxmax()].set_index("site_no")["yearCompleted"]
    cum = d.groupby("site_no")["nidStorage"].cumsum()
    tot = d.groupby("site_no")["nidStorage"].transform("sum")
    return d[cum >= frac * tot].groupby("site_no")["yearCompleted"].min()


def code6_onset(peaks: pd.DataFrame) -> pd.Series:
    codes = peaks["peak_cd"].apply(_parse_peak_cd)
    p = peaks[codes.apply(lambda c: 6 in c) & ~codes.apply(lambda c: 1 in c or 7 in c)]
    return water_year(p["datetime"]).groupby(p["site_no"]).min()


def choose_cutoffs(sites: pd.DataFrame) -> pd.DataFrame:
    """cutoff_wy (first year kept) and cutoff_source per site."""
    late = sites["first_wy"] + MID_RECORD
    stor = (sites["storage_wy"] + 1).where(
        (sites["dor_during_record"] >= DOR_MIN) & (sites["storage_wy"] + 1 > late))
    c6 = sites["c6_onset_wy"].where(sites["c6_onset_wy"] > late)
    phys = pd.concat([stor, c6], axis=1).max(axis=1)
    src = np.select([stor.notna() & c6.notna() & (stor >= c6),
                     stor.notna() & c6.notna(), stor.notna(), c6.notna()],
                    ["storage", "code6", "storage", "code6"], "")
    pett = sites["change_wy"].where(sites["pettitt_sig"] & phys.isna())
    out = sites.assign(cutoff_wy=phys.fillna(pett),
                       cutoff_source=np.where(phys.notna(), src,
                                              np.where(pett.notna(), "pettitt", "none")))
    return out


def refit(site: str, peaks: pd.DataFrame, cutoff: int, flows: tuple) -> tuple[dict, dict]:
    """Pipeline fit on peaks from cutoff on, plus the MK re-screen of that record."""
    post = peaks[water_year(peaks["datetime"]) >= cutoff].reset_index(drop=True)
    fit = _fit_site_worker((site, post, flows, _MIN_PEAKS))
    codes = post["peak_cd"].apply(_parse_peak_cd)
    s = post[~codes.apply(lambda c: 1 in c or 7 in c) & post["peak_va"].gt(0)]
    s = s.assign(wy=water_year(s["datetime"])).drop_duplicates("wy").sort_values("wy")
    _, z, p = mann_kendall_hr(s["wy"].to_numpy(float), np.log10(s["peak_va"].to_numpy(float)))
    return fit, {"site_no": site, "n_post": len(s), "post_mk_z": z, "post_mk_p": p}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--storage-frac", type=float, default=None,
                    help="date storage by this share of today's instead of the largest dam")
    args = ap.parse_args()
    ffa_dir, meta_dir = args.data_dir / "ffa", args.data_dir / "metadata"

    trends = pd.read_parquet(ffa_dir / "peak_trends.parquet")
    attr = pd.read_parquet(ffa_dir / "change_attribution.parquet",
                           columns=["site_no", "dor_during_record", "dam_attributed"])
    sites = (trends[trends["tested"] & trends["is_regulated"].eq(True)]
             .merge(attr, on="site_no", how="left"))

    basins = gpd.read_parquet(meta_dir / "gage_basins.parquet")
    pairs = dams_in_basins(load_dams(args.data_dir / "nid" / "nid_dams.parquet"),
                           basins[basins["site_no"].isin(sites["site_no"])])
    peaks = pd.read_parquet(ffa_dir / "annual_peaks.parquet")
    peaks = peaks[peaks["site_no"].isin(sites["site_no"])]

    sites = sites.join(storage_year(pairs, args.storage_frac).rename("storage_wy"), on="site_no")
    sites = sites.join(code6_onset(peaks).rename("c6_onset_wy"), on="site_no")
    sites = choose_cutoffs(sites)
    sp = systematic_peaks(peaks).merge(sites[["site_no", "cutoff_wy"]], on="site_no")
    sites["n_sys_post"] = (sp[sp["wy"] >= sp["cutoff_wy"]].groupby("site_no").size()
                           .reindex(sites["site_no"]).fillna(0).astype(int).to_numpy())
    sites["status"] = np.select(
        [sites["cutoff_source"].eq("none"), sites["n_sys_post"] < MIN_POST],
        ["no_cutoff", "too_short"], "refit")

    fs = pd.read_parquet(meta_dir / "flood_stages.parquet")
    keep = ["site_no"] + _THRESHOLD_FLOWS + [c for c in fs.columns
                                             if c.endswith("_source") or c == _RATING_MAX_FLOW]
    fs = fs[keep].drop_duplicates("site_no")
    fs_idx = fs.set_index("site_no")
    by_site = dict(tuple(peaks.groupby("site_no")))

    fits, screens = [], []
    todo = sites[sites["status"] == "refit"]
    for r in todo.itertuples():
        flows = tuple(float(fs_idx.at[r.site_no, c]) if r.site_no in fs_idx.index
                      and pd.notna(fs_idx.at[r.site_no, c]) else float("nan")
                      for c in _THRESHOLD_FLOWS)
        f, s = refit(r.site_no, by_site[r.site_no], int(r.cutoff_wy), flows)
        fits.append(f)
        screens.append(s)
    log.info("Refit %d sites", len(fits))

    res = _apply_weighted_skew(pd.DataFrame(fits), fs, _load_site_info(meta_dir),
                               pd.DataFrame(columns=["state_cd", "regional_skew_g",
                                                     "mse_regional_g"]), "station")
    res["degenerate_fit"] = ((res["lp3_loc"] < 0.5) | (res["lp3_scale"] > 2.0)
                             | (res["lp3_skew"].abs() > 3.0)).fillna(False)
    rp_cols = ([f"{lv}_return_period_yr" for lv in _LEVELS]
               + [f"{lv}_return_period_lb_yr" for lv in _LEVELS])
    res = res[["site_no", "n_peaks", "lp3_skew", "lp3_loc", "lp3_scale", "degenerate_fit"]
              + rp_cols].rename(columns={c: f"post_{c}" for c in
                                         ["n_peaks", "lp3_skew", "lp3_loc", "lp3_scale",
                                          "degenerate_fit"] + rp_cols})
    scr = pd.DataFrame(screens)
    scr["post_trend_sig"] = bh_reject(scr["post_mk_p"], FDR_Q)

    full = pd.read_parquet(ffa_dir / "flood_frequency.parquet", columns=["site_no"] + rp_cols)
    out = (sites.merge(res, on="site_no", how="left").merge(scr, on="site_no", how="left")
           .merge(full.rename(columns={c: f"full_{c}" for c in rp_cols}), on="site_no",
                  how="left"))
    report(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    log.info("Saved %d rows -> %s", len(out), args.out)


def report(df: pd.DataFrame) -> None:
    log.info("Regulated tested sites: %d", len(df))
    log.info("Cutoff source: %s", df["cutoff_source"].value_counts().to_dict())
    log.info("Status: %s", df["status"].value_counts().to_dict())
    r = df[df["status"] == "refit"]
    log.info("Refit: median %d post-cutoff years; %d degenerate; %d keep a significant "
             "post-cutoff trend", r["n_post"].median(), r["post_degenerate_fit"].sum(),
             r["post_trend_sig"].sum())
    for lv in _LEVELS:
        a, b = r[f"post_{lv}_return_period_yr"], r[f"full_{lv}_return_period_yr"]
        ok = a.gt(0) & b.gt(0) & np.isfinite(a) & np.isfinite(b)
        ratio = a[ok] / b[ok]
        log.info("  %-8s n=%4d  median post/full RP %.2f  (P10 %.2f, P90 %.2f)  longer at %.0f%%",
                 lv, ok.sum(), ratio.median(), ratio.quantile(0.1), ratio.quantile(0.9),
                 100 * (ratio > 1).mean())


if __name__ == "__main__":
    main()
