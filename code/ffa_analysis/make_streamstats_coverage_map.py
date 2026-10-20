"""
Map of which QC sites have a published StreamStats station estimate.

A site counts as covered when USGS publishes at least one at-site (PK*AEP)
flood quantile for it (streamstats_validation.parquet, ss_station_q*_cfs);
those are the sites behind the external check in validate_against_streamstats.py.
Covered sites are filled dots; the rest are open rings drawn on top, so dense
clusters of either stay visible.  Styling follows make_rp_maps.py.

Usage:
  python make_streamstats_coverage_map.py

Writes streamstats_coverage.{png,pdf} to FIG_DIR.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from make_rp_maps import (
    COLORS, CONUS_EXTENT, FFA_DIR, FIG_DIR, INK_2, LAND, PROJ, STATE, TITLE_FONT,
    _conus_geoms, load_sites,
)

COVERED = COLORS[3]      # teal, the deck's accent
UNCOVERED = INK_2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--out", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    sites = load_sites(False)
    sv = pd.read_parquet(FFA_DIR / "streamstats_validation.parquet")
    station_cols = [c for c in sv.columns if c.startswith("ss_station_q")]
    covered = set(sv.loc[sv[station_cols].notna().any(axis=1), "site_no"])
    sites["covered"] = sites["site_no"].isin(covered)
    yes, no = sites[sites["covered"]], sites[~sites["covered"]]

    fig, ax = plt.subplots(figsize=(7.0, 4.6), subplot_kw={"projection": PROJ})
    ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    ax.add_geometries(_conus_geoms(), crs=ccrs.PlateCarree(), facecolor=LAND,
                      edgecolor=STATE, linewidth=0.3, zorder=0)
    ax.spines["geo"].set_visible(False)

    ax.scatter(yes["longitude"], yes["latitude"], s=9, color=COVERED,
               edgecolors=LAND, linewidths=0.2, transform=ccrs.PlateCarree(), zorder=3)
    ax.scatter(no["longitude"], no["latitude"], s=10, facecolors="none",
               edgecolors=UNCOVERED, linewidths=0.5, transform=ccrs.PlateCarree(), zorder=4)

    ax.set_title("Sites with a published StreamStats station estimate", loc="left",
                 pad=4, fontdict=TITLE_FONT)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.5, markerfacecolor=COVERED,
               markeredgecolor=LAND, markeredgewidth=0.3,
               label=f"Station estimate published ({len(yes):,})"),
        Line2D([], [], marker="o", linestyle="none", markersize=4.5, markerfacecolor="none",
               markeredgecolor=UNCOVERED, markeredgewidth=0.7,
               label=f"No station estimate ({len(no):,})"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7,
              handletextpad=0.3, borderaxespad=0.2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"streamstats_coverage.{ext}", facecolor="white")
    print(f"{len(yes):,} covered / {len(no):,} not → {args.out / 'streamstats_coverage'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
