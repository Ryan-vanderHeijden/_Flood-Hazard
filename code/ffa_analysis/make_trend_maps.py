"""
Non-stationarity screen of the annual peaks — maps and summary panels.

Reads peak_trends.parquet (src/compute_peak_trends.py).  Top: CONUS maps of
the Mann–Kendall trends (coloured by Sen slope, % per decade) and the Pettitt
change points (coloured by the after/before change in median peak), each
significant at a Benjamini–Hochberg FDR of 5%; triangles point in the
direction of change, and tested sites with no significant result are small
gray dots.  Bottom: the share of tested sites with each result by regulation
class, and the years of the significant change points.

Usage:
  python make_trend_maps.py

Writes peak_trends.{png,pdf} to FIG_DIR.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from make_rp_maps import (
    CONUS_EXTENT, FFA_DIR, FIG_DIR, INK, INK_2, LAND, PROJ, STATE, TITLE_FONT,
    _conus_geoms, load_sites,
)

# Diverging, ordered from decrease to increase: warm arm = smaller peaks,
# blue arm = larger (blue reads as "more water").  Each arm validated
# as an ordinal ramp on LAND (light ends clear 2:1); the light ends separate
# under CVD (dE 16), and triangle direction repeats the sign.
DIV_COLORS = ["#8f2a14", "#d4582f", "#ef9a7a", "#86b6ef", "#2a78d6", "#104281"]
DIV_CMAP = ListedColormap(DIV_COLORS)
TREND_BREAKS = [-20, -10, 0, 10, 20]          # Sen slope, % per decade
CHANGE_BREAKS = [-50, -25, 0, 25, 50]         # median after / before, %
NS_GRAY = "#c2c1bc"
REG_INK = ["#9a9994", INK]                    # unregulated, regulated


def class_labels(breaks: list[int], unit: str) -> list[str]:
    b = breaks
    def f(v):
        return f"−{-v}" if v < 0 else f"{v}"
    return ([f"≤ {f(b[0])}{unit}"] + [f"{f(lo)} to {f(hi)}" for lo, hi in zip(b[:-1], b[1:])]
            + [f"≥ +{b[-1]}{unit}"])


def draw_map(ax, geoms, tested, sig, value, breaks, title, unit) -> None:
    ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor=LAND,
                      edgecolor=STATE, linewidth=0.3, zorder=0)
    ax.spines["geo"].set_visible(False)
    ns = tested[~sig]
    ax.scatter(ns["longitude"], ns["latitude"], s=2.5, color=NS_GRAY, linewidths=0,
               transform=ccrs.PlateCarree(), zorder=2)
    norm = BoundaryNorm([-np.inf] + breaks + [np.inf], DIV_CMAP.N)
    s = tested[sig].assign(v=value[sig]).sort_values("v", key=np.abs)
    for marker, part in (("^", s[s["v"] > 0]), ("v", s[s["v"] <= 0])):
        ax.scatter(part["longitude"], part["latitude"], s=13, marker=marker,
                   c=part["v"], cmap=DIV_CMAP, norm=norm, edgecolors=LAND,
                   linewidths=0.25, transform=ccrs.PlateCarree(), zorder=3)
    up, down = int((s["v"] > 0).sum()), int((s["v"] <= 0).sum())
    ax.set_title(title, loc="left", pad=2, fontdict=TITLE_FONT)
    ax.text(0.0, -0.01, f"{up} up · {down} down · {len(tested) - len(s):,} not significant",
            transform=ax.transAxes, ha="left", va="top", fontsize=6, color=INK_2)
    handles = [Line2D([], [], marker="v" if i < 3 else "^", linestyle="none",
                      markersize=4.5, markerfacecolor=c, markeredgewidth=0, label=l)
               for i, (c, l) in enumerate(zip(DIV_COLORS, class_labels(breaks, unit)))]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.07), ncol=3,
              frameon=False, fontsize=5.5, handletextpad=0.1, columnspacing=0.8)


def strip(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, width=0.4, length=2, labelsize=6)


def draw_shares(ax, t: pd.DataFrame) -> None:
    """Grouped bars: % of tested sites per result, unregulated vs regulated."""
    results = [("Trend up", t["trend_dir"].eq("up")),
               ("Trend down", t["trend_dir"].eq("down")),
               ("Change up", t["pettitt_sig"] & (t["change_pct"] > 0)),
               ("Change down", t["pettitt_sig"] & (t["change_pct"] <= 0))]
    x = np.arange(len(results))
    w = 0.38
    for k, (reg, label) in enumerate(((False, "Unregulated"), (True, "Regulated"))):
        g = t["is_regulated"].eq(reg)
        pct = [100 * m[g].mean() for _, m in results]
        ax.bar(x + (k - 0.5) * w, pct, width=w - 0.04, color=REG_INK[k], linewidth=0,
               label=f"{label} ({int(g.sum()):,})")
        for xi, v in zip(x + (k - 0.5) * w, pct):
            ax.text(xi, v + 0.4, f"{v:.0f}", ha="center", va="bottom", fontsize=5.5,
                    color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([r for r, _ in results], fontsize=6, color=INK_2)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    strip(ax)
    ax.set_title("% of tested sites", loc="left", pad=2, fontdict=TITLE_FONT)
    ax.legend(loc="upper left", frameon=False, fontsize=6, handlelength=1.0,
              handletextpad=0.4, bbox_to_anchor=(0.0, 0.98))


def draw_change_years(ax, t: pd.DataFrame) -> None:
    cp = t[t["pettitt_sig"]]
    bins = np.arange(1925, 2016, 5)
    parts = [cp.loc[cp["is_regulated"].eq(reg), "change_wy"] for reg in (False, True)]
    counts, _, _ = ax.hist(parts, bins=bins, stacked=True, color=REG_INK, rwidth=0.88,
                           linewidth=0, label=["Unregulated", "Regulated"])
    ax.set_ylim(0, counts[-1].max() * 1.25)
    ax.legend(loc="upper right", frameon=False, fontsize=6, handlelength=1.0,
              handletextpad=0.4)
    strip(ax)
    ax.set_yticks([])
    ax.set_xlim(bins[0], bins[-1])
    ax.set_title(f"Year of significant change points ({len(cp)})", loc="left", pad=2,
                 fontdict=TITLE_FONT)
    med = cp.groupby("is_regulated")["change_wy"].median()
    ax.text(0.0, 0.98, f"median: unregulated {med.get(False, np.nan):.0f} · "
            f"regulated {med.get(True, np.nan):.0f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6, color=INK_2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--out", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    tr = pd.read_parquet(FFA_DIR / "peak_trends.parquet")
    sites = load_sites(False)[["site_no", "latitude", "longitude"]]
    t = tr[tr["tested"]].merge(sites, on="site_no", how="inner")
    geoms = _conus_geoms()

    fig = plt.figure(figsize=(7.0, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1], hspace=0.42, wspace=0.06,
                          left=0.01, right=0.99, top=0.97, bottom=0.06)
    a = fig.add_subplot(gs[0, 0], projection=PROJ)
    b = fig.add_subplot(gs[0, 1], projection=PROJ)
    draw_map(a, geoms, t, t["mk_sig"], t["sen_pct_decade"], TREND_BREAKS,
             "Trend (Mann–Kendall)", "%/decade")
    draw_map(b, geoms, t, t["pettitt_sig"], t["change_pct"], CHANGE_BREAKS,
             "Step change (Pettitt)", "%")
    c = fig.add_subplot(gs[1, 0])
    d = fig.add_subplot(gs[1, 1])
    draw_shares(c, t)
    draw_change_years(d, t)
    # The bottom panels have axis labels; pull them in from the map margins.
    for ax in (c, d):
        p = ax.get_position()
        ax.set_position([p.x0 + 0.03, p.y0, p.width - 0.05, p.height])

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"peak_trends.{ext}", facecolor="white")
    print(f"tested {len(t):,} sites → {args.out / 'peak_trends'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
