"""Attribute the peak-flow change points to upstream dam construction.

The attribution step after compute_peak_trends.py. Dams from the National
Inventory of Dams (NID) are placed in each gage's GAGES-II drainage basin
(extract_gagesii_basins.py), and
their storage is expressed as a degree of regulation (DOR): storage over the
basin's mean annual runoff volume (GAGES-II RUNAVE7100 × basin area), the
same measure compute_regulation.py uses for ``is_regulated`` (DOR ≥ 0.10).

A significant Pettitt change point is **dam-attributed** when the dams
completed in the window ``change_wy − WIN_BEFORE`` … ``change_wy + WIN_AFTER``
add at least ``DOR_MIN`` of DOR. The window is asymmetric because a dam
finished in year Y can shift peaks from water year Y onward, and the Pettitt
location is itself uncertain by a few years.

As a control, the same window test is applied to tested sites *without* a
significant change point, centred on their record midpoint. The share of
those that pass is the rate at which a dam lands in a window by chance at a
site of that kind; attribution only means something well above it.

Storage is NID ``nidStorage`` (acre-feet; the larger of normal and maximum
storage, as in GAGES-II STOR_NID). Dams with no completion year are counted
in the basin totals but cannot be placed in time; their share is reported.

Writes ``data/ffa/change_attribution.parquet`` — one row per tested site.

    python code/ffa_analysis/src/attribute_change_points.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

DATA_DIR = Path.home() / "data" / "flood_hazard"
OUT_PATH = DATA_DIR / "ffa" / "change_attribution.parquet"

M3_PER_ACRE_FT = 1233.48
WIN_BEFORE = 5       # dam completed up to this many years before change_wy …
WIN_AFTER = 1        # … or this many after
DOR_MIN = 0.05       # DOR added in the window to attribute the change

log = logging.getLogger(__name__)


def load_dams(path: Path) -> gpd.GeoDataFrame:
    d = pd.read_parquet(path, columns=["nidId", "name", "latitude", "longitude",
                                       "yearCompleted", "nidStorage"])
    d = d[d["nidStorage"].gt(0) & d["latitude"].notna() & d["longitude"].notna()]
    d["yearCompleted"] = d["yearCompleted"].where(d["yearCompleted"].between(1700, 2030))
    return gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d["longitude"], d["latitude"]),
                            crs="EPSG:4326")


def dams_in_basins(dams: gpd.GeoDataFrame, basins: gpd.GeoDataFrame) -> pd.DataFrame:
    """(site_no, dam) pairs for every dam inside a gage's basin; nested basins share dams."""
    j = gpd.sjoin(dams.to_crs(basins.crs), basins[["site_no", "geometry"]],
                  predicate="within", how="inner")
    return pd.DataFrame(j.drop(columns=["geometry", "index_right"]))


def window_dor(sd: pd.DataFrame, lo: float, hi: float, runoff_af: float) -> tuple[float, str]:
    """DOR added by dams completed in [lo, hi], and the largest of them."""
    w = sd[sd["yearCompleted"].between(lo, hi)]
    if w.empty or not np.isfinite(runoff_af) or runoff_af <= 0:
        return 0.0, ""
    top = w.loc[w["nidStorage"].idxmax()]
    return float(w["nidStorage"].sum() / runoff_af), f"{top['name']} ({int(top['yearCompleted'])})"


def attribute(trends: pd.DataFrame, basins: gpd.GeoDataFrame, dams: gpd.GeoDataFrame,
              runoff_mm: pd.Series) -> pd.DataFrame:
    area_km2 = basins.set_index("site_no")["area_km2"]
    runoff_af = (runoff_mm.reindex(area_km2.index) * 1e-3 * area_km2 * 1e6) / M3_PER_ACRE_FT
    pairs = dams_in_basins(dams, basins)
    by_site = dict(tuple(pairs.groupby("site_no")))

    rows = []
    for r in trends[trends["tested"]].itertuples():
        sd = by_site.get(r.site_no, pairs.iloc[:0])
        v = runoff_af.get(r.site_no, np.nan)
        known = sd[sd["yearCompleted"].notna()]
        row = {
            "site_no": r.site_no,
            "basin_km2": area_km2.get(r.site_no, np.nan),
            "n_dams": len(sd),
            "dor_nid": sd["nidStorage"].sum() / v if v > 0 else np.nan,
            "unknown_year_storage_frac": (1 - known["nidStorage"].sum() / sd["nidStorage"].sum()
                                          if len(sd) else np.nan),
            "dor_before_record": (known.loc[known["yearCompleted"] < r.first_wy, "nidStorage"].sum() / v
                                  if v > 0 else np.nan),
            "dor_during_record": (known.loc[known["yearCompleted"].between(r.first_wy, r.last_wy),
                                            "nidStorage"].sum() / v if v > 0 else np.nan),
        }
        if r.pettitt_sig:
            centre, is_control = r.change_wy, False
        else:
            centre, is_control = round((r.first_wy + r.last_wy) / 2), True
        dor_w, top = window_dor(sd, centre - WIN_BEFORE, centre + WIN_AFTER, v)
        row.update(window_centre_wy=centre, is_control=is_control,
                   window_dor=dor_w, window_top_dam=top,
                   dam_in_window=dor_w >= DOR_MIN)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["dam_attributed"] = ~out["is_control"] & out["dam_in_window"]
    return out


def report(df: pd.DataFrame) -> None:
    log.info("Basins with >= 1 NID dam: %d of %d; median unknown-year storage share %.0f%%",
             (df["n_dams"] > 0).sum(), len(df),
             100 * df.loc[df["n_dams"] > 0, "unknown_year_storage_frac"].median())
    for reg, g in df.groupby("is_regulated"):
        cp, ctl = g[~g["is_control"]], g[g["is_control"]]
        log.info("%s: %d change points, %d (%.0f%%) dam-attributed; "
                 "control rate %.1f%% of %d",
                 "regulated" if reg else "unregulated", len(cp), cp["dam_attributed"].sum(),
                 100 * cp["dam_attributed"].mean() if len(cp) else np.nan,
                 100 * ctl["dam_in_window"].mean(), len(ctl))
    down = df[~df["is_control"] & (df["change_pct"] <= 0)]
    up = df[~df["is_control"] & (df["change_pct"] > 0)]
    log.info("Dam-attributed: %.0f%% of step drops (%d), %.0f%% of step increases (%d)",
             100 * down["dam_attributed"].mean(), len(down),
             100 * up["dam_attributed"].mean(), len(up))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    trends = pd.read_parquet(args.data_dir / "ffa" / "peak_trends.parquet")
    basins = gpd.read_parquet(args.data_dir / "metadata" / "gage_basins.parquet")
    dams = load_dams(args.data_dir / "nid" / "nid_dams.parquet")
    reg = (pd.read_parquet(args.data_dir / "metadata" / "regulation.parquet",
                           columns=["site_no", "RUNAVE7100", "dor"])
           .drop_duplicates("site_no").set_index("site_no"))

    df = attribute(trends, basins, dams, reg["RUNAVE7100"])
    df = df.merge(trends[["site_no", "is_regulated", "pettitt_sig", "change_wy", "change_pct",
                          "trend_dir"]], on="site_no", how="left")
    df = df.merge(reg[["dor"]].rename(columns={"dor": "dor_gagesii"}),
                  left_on="site_no", right_index=True, how="left")
    report(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("Saved %d rows -> %s", len(df), args.out)


if __name__ == "__main__":
    main()
