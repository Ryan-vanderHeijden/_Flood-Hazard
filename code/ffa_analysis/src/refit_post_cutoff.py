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
cutoff record is then re-screened with the same Mann–Kendall test (BH FDR
q = ``FDR_Q`` across refit sites).

**Second pass.** Where a post-cutoff trend survives, storage often kept
growing after the largest dam. Those sites are refit once more from the year
after dated storage reached ``SECOND_PASS_FRAC`` of today's, if that is later
than the first cutoff and leaves ``MIN_POST`` years; the new fit is kept when
its trend falls below the first pass's BH threshold (cutoff_source
"storage90"). Sites still trending after that are flagged
``gradual_change``: no single cutoff reaches a homogeneous record, so their
present-day values should be used with care.

**Flag for unrefit records.** ``nonstationary_full_record`` marks regulated
sites that keep their full-record fit despite evidence of change: a
significant trend or step change at a site whose cutoff leaves fewer than
``MIN_POST`` years ("too_short") or that has no cutoff date ("no_cutoff"),
and, at sites with fewer than ``MIN_POST`` years in all ("short_record",
never screened), a physical cutoff date (dam or code 6) mid-record. Their
reported return periods mix pre- and post-regulation floods.

Writes ``data/ffa/post_cutoff_fits.parquet`` — one row per regulated QC
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
from attribute_change_points import DOR_MIN, M3_PER_ACRE_FT, dams_in_basins, load_dams  # noqa: E402
from compute_flood_frequency import (  # noqa: E402
    _LEVELS, _RATING_MAX_FLOW, _THRESHOLD_FLOWS, _apply_weighted_skew,
    _fit_site_worker, _load_site_info, _MIN_PEAKS, _parse_peak_cd,
)
from compute_peak_trends import FDR_Q, bh_reject, mann_kendall_hr, systematic_peaks  # noqa: E402

DATA_DIR = Path.home() / "data" / "flood_hazard"
OUT_PATH = DATA_DIR / "ffa" / "post_cutoff_fits.parquet"

MID_RECORD = 5        # a date this close to the record start cuts nothing
MIN_POST = 30         # systematic years needed after the cutoff to refit
SECOND_PASS_FRAC = 0.90   # storage share for the second-pass cutoff

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


def dor_during_record(pairs: pd.DataFrame, sites: pd.DataFrame, basins: pd.DataFrame,
                      runoff_mm: pd.Series) -> pd.Series:
    """DOR of dams completed within each site's systematic record (as in
    attribute_change_points.py, but for every site, screened or not)."""
    area = basins.set_index("site_no")["area_km2"]
    runoff_af = runoff_mm.reindex(area.index) * 1e-3 * area * 1e6 / M3_PER_ACRE_FT
    span = sites.set_index("site_no")[["first_wy", "last_wy"]]
    d = pairs.dropna(subset=["yearCompleted"]).join(span, on="site_no")
    d = d[d["yearCompleted"].between(d["first_wy"], d["last_wy"])]
    built = d.groupby("site_no")["nidStorage"].sum()
    return (built / runoff_af.reindex(built.index)).reindex(span.index).fillna(0.0)


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
    sites = trends[trends["is_regulated"].eq(True)].copy()   # screened or not
    for c in ("mk_sig", "pettitt_sig"):
        sites[c] = sites[c].fillna(False).astype(bool)

    basins = gpd.read_parquet(meta_dir / "gage_basins.parquet")
    basins = basins[basins["site_no"].isin(sites["site_no"])]
    pairs = dams_in_basins(load_dams(args.data_dir / "nid" / "nid_dams.parquet"), basins)
    runoff = (pd.read_parquet(meta_dir / "regulation.parquet", columns=["site_no", "RUNAVE7100"])
              .drop_duplicates("site_no").set_index("site_no")["RUNAVE7100"])
    sites["dor_during_record"] = sites["site_no"].map(
        dor_during_record(pairs, sites, basins, runoff))
    peaks = pd.read_parquet(ffa_dir / "annual_peaks.parquet")
    peaks = peaks[peaks["site_no"].isin(sites["site_no"])]

    sites = sites.join(storage_year(pairs, args.storage_frac).rename("storage_wy"), on="site_no")
    sites = sites.join(code6_onset(peaks).rename("c6_onset_wy"), on="site_no")
    sites = choose_cutoffs(sites)
    sp = systematic_peaks(peaks).merge(sites[["site_no", "cutoff_wy"]], on="site_no")
    sites["n_sys_post"] = (sp[sp["wy"] >= sp["cutoff_wy"]].groupby("site_no").size()
                           .reindex(sites["site_no"]).fillna(0).astype(int).to_numpy())
    sites["status"] = np.select(
        [~sites["tested"], sites["cutoff_source"].eq("none"), sites["n_sys_post"] < MIN_POST],
        ["short_record", "no_cutoff", "too_short"], "refit")
    changed = sites["mk_sig"] | sites["pettitt_sig"]
    sites["nonstationary_full_record"] = (
        (sites["status"].isin(["too_short", "no_cutoff"]) & changed)
        | (sites["status"].eq("short_record") & sites["cutoff_source"].isin(["storage", "code6"])))

    fs = pd.read_parquet(meta_dir / "flood_stages.parquet")
    keep = ["site_no"] + _THRESHOLD_FLOWS + [c for c in fs.columns
                                             if c.endswith("_source") or c == _RATING_MAX_FLOW]
    fs = fs[keep].drop_duplicates("site_no")
    fs_idx = fs.set_index("site_no")
    by_site = dict(tuple(peaks.groupby("site_no")))

    def flows(site: str) -> tuple:
        return tuple(float(fs_idx.at[site, c]) if site in fs_idx.index
                     and pd.notna(fs_idx.at[site, c]) else float("nan")
                     for c in _THRESHOLD_FLOWS)

    fits, screens = {}, {}
    for r in sites[sites["status"] == "refit"].itertuples():
        fits[r.site_no], screens[r.site_no] = refit(
            r.site_no, by_site[r.site_no], int(r.cutoff_wy), flows(r.site_no))
    log.info("Refit %d sites", len(fits))

    # First-pass BH: a p-value is significant iff it is <= the largest rejected one.
    p1 = pd.Series({k: v["post_mk_p"] for k, v in screens.items()})
    rejected = bh_reject(p1, FDR_Q)
    p_crit = p1[rejected].max() if rejected.any() else -np.inf

    # Second pass: a later, storage-share cutoff for sites still trending.
    s2 = storage_year(pairs, SECOND_PASS_FRAC)
    sites["second_pass"] = False
    sys_wy = sp.groupby("site_no")["wy"].apply(np.asarray)
    idx = sites.set_index("site_no")
    n_tried = 0
    for site in p1.index[rejected]:
        r = idx.loc[site]
        c2 = s2.get(site, np.nan) + 1
        if not (np.isfinite(c2) and r["dor_during_record"] >= DOR_MIN
                and c2 > r["first_wy"] + MID_RECORD and c2 > r["cutoff_wy"]
                and (sys_wy[site] >= c2).sum() >= MIN_POST):
            continue
        n_tried += 1
        f, scr2 = refit(site, by_site[site], int(c2), flows(site))
        if scr2["post_mk_p"] > p_crit:
            fits[site], screens[site] = f, scr2
            m = sites["site_no"] == site
            sites.loc[m, ["cutoff_wy", "cutoff_source", "second_pass"]] = [c2, "storage90", True]
            sites.loc[m, "n_sys_post"] = int((sys_wy[site] >= c2).sum())
    log.info("Second pass: %d of %d trending sites tried a later cutoff; %d resolved",
             n_tried, int(rejected.sum()), int(sites["second_pass"].sum()))
    fits, screens = list(fits.values()), list(screens.values())

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
    scr["post_trend_sig"] = scr["post_mk_p"] <= p_crit
    scr["gradual_change"] = scr["post_trend_sig"]

    full = pd.read_parquet(ffa_dir / "flood_frequency.parquet", columns=["site_no"] + rp_cols)
    out = (sites.merge(res, on="site_no", how="left").merge(scr, on="site_no", how="left")
           .merge(full.rename(columns={c: f"full_{c}" for c in rp_cols}), on="site_no",
                  how="left"))
    report(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    log.info("Saved %d rows -> %s", len(out), args.out)


def report(df: pd.DataFrame) -> None:
    log.info("Regulated QC sites: %d (%d screened)", len(df), df["tested"].sum())
    log.info("Cutoff source: %s", df["cutoff_source"].value_counts().to_dict())
    log.info("Status: %s", df["status"].value_counts().to_dict())
    log.info("Full record kept despite evidence of change (nonstationary_full_record): %s",
             df.loc[df["nonstationary_full_record"], "status"].value_counts().to_dict())
    r = df[df["status"] == "refit"]
    log.info("Refit: median %d post-cutoff years; %d degenerate; %d keep a significant "
             "post-cutoff trend (gradual_change)", r["n_post"].median(),
             r["post_degenerate_fit"].sum(), r["post_trend_sig"].sum())
    for lv in _LEVELS:
        a, b = r[f"post_{lv}_return_period_yr"], r[f"full_{lv}_return_period_yr"]
        ok = a.gt(0) & b.gt(0) & np.isfinite(a) & np.isfinite(b)
        ratio = a[ok] / b[ok]
        log.info("  %-8s n=%4d  median post/full RP %.2f  (P10 %.2f, P90 %.2f)  longer at %.0f%%",
                 lv, ok.sum(), ratio.median(), ratio.quantile(0.1), ratio.quantile(0.9),
                 100 * (ratio > 1).mean())


if __name__ == "__main__":
    main()
