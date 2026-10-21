"""
Period-of-record figures for the sites behind the LP3 fits.

Record length here is the number of systematic water years with an annual
peak, the rows the EMA fit sees after code-1 (daily mean) peaks are dropped;
code-7 historical peaks are counted separately.  It includes censored and
PILF years, so it can exceed flood_frequency's n_peaks.  QC sites only.

Three figures, styled like make_rp_maps.py:
  record_timeline     sites reporting each water year (top) over a raster of
                      every site's record, sorted by first year (bottom)
  record_length_map   CONUS dots coloured by record length, inset histogram
  record_vs_rp        threshold return period against record length, one
                      panel per level.  Record length here adds the
                      historical period the EMA fit uses (hist_H, or the
                      count of code-7 peaks where no period is defined, as in
                      compute_flood_frequency's n_eff).  Above the 1:1 line
                      the return period exceeds the record, so it leans
                      increasingly on the fitted LP3 tail rather than on
                      observed floods; on the line, the chance the record
                      holds at least one such flood is about 63%

Usage:
  python make_record_figures.py
  python make_record_figures.py --regulation unregulated   # or: regulated

Writes <figure>[_<regulation>].{png,pdf} to FIG_DIR.
"""

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from make_rp_maps import (
    CMAP, CONUS_EXTENT, FFA_DIR, FIG_DIR, INK_2, LAND, LEVELS,
    NORM, PROJ, REGULATION, STATE, TITLE_FONT, _conus_geoms, label_subset, load_sites,
    split_sites, subset_suffix,
)

sys.path.insert(0, str(Path(__file__).parent / "src"))
from compute_flood_frequency import _parse_peak_cd  # noqa: E402

# Record-length classes (yr).  A single-hue blue ramp, so record length never
# reads as a return period; validated as an ordinal ramp on LAND (the light
# end clears 2:1).
REC_BREAKS = [30, 50, 70, 90]
REC_LABELS = ["10–30", "30–50", "50–70", "70–90", "≥ 90"]
REC_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
REC_NORM   = BoundaryNorm([0] + REC_BREAKS + [np.inf], len(REC_COLORS))
REC_CMAP   = ListedColormap(REC_COLORS)

SYS_INK  = REC_COLORS[3]
HIST_INK = "#d9741e"      # code-7 historical peaks: the one warm accent
LAST_WY  = 2025           # water year 2026 is partial (27 sites so far)
FIRST_WY = 1850           # timeline x-axis start; earlier historical peaks are counted in a note


def site_records(site_nos: pd.Series) -> pd.DataFrame:
    """One row per (site, water year) with a peak, flagged systematic or historical."""
    p = pd.read_parquet(FFA_DIR / "annual_peaks.parquet",
                        columns=["site_no", "datetime", "peak_cd"])
    p = p[p["site_no"].isin(site_nos)]
    codes = p["peak_cd"].apply(_parse_peak_cd)   # the fit's parser: "7" and "7.0" alike
    p = p[~codes.apply(lambda c: 1 in c)]
    codes = codes.loc[p.index]
    wy = p["datetime"].dt.year + (p["datetime"].dt.month >= 10)
    return (pd.DataFrame({"site_no": p["site_no"], "wy": wy,
                          "hist": codes.apply(lambda c: 7 in c)})
            .drop_duplicates(["site_no", "wy"]))


def record_summary(rec: pd.DataFrame) -> pd.DataFrame:
    sys_ = rec[~rec["hist"]]
    g = sys_.groupby("site_no")["wy"].agg(first="min", last="max", years="nunique")
    g["gap"] = g["last"] - g["first"] + 1 - g["years"]
    g["hist_years"] = rec[rec["hist"]].groupby("site_no").size()
    return g.fillna({"hist_years": 0}).reset_index()


def strip_axes(ax, keep=("bottom",)) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
        ax.spines[side].set_linewidth(0.4)
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, width=0.4, length=2, labelsize=6)


def fig_timeline(rec: pd.DataFrame, summ: pd.DataFrame):
    order = summ.sort_values(["first", "years"], ascending=[True, False])["site_no"]
    row = pd.Series(np.arange(len(order)), index=order.to_numpy())
    y0 = int(rec["wy"].min())
    grid = np.zeros((len(order), LAST_WY - y0 + 1), np.uint8)
    r = rec[(rec["wy"] <= LAST_WY) & ~rec["hist"]]
    grid[row[r["site_no"]].to_numpy(), (r["wy"] - y0).to_numpy()] = 1
    hist = rec[rec["hist"]]

    fig, (top, bot) = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08})
    per_year = (grid == 1).sum(0)
    years = np.arange(y0, LAST_WY + 1)
    top.fill_between(years, per_year, step="mid", color=SYS_INK, linewidth=0)
    top.set_ylim(0, per_year.max() * 1.15)
    top.set_yticks([0, 1000, 2000, 3000])
    top.set_yticklabels(["0", "1,000", "2,000", "3,000"])
    strip_axes(top, keep=("bottom", "left"))
    top.set_title("QC sites with a systematic annual peak", loc="left", pad=4,
                  fontdict=TITLE_FONT)
    k = int(per_year.argmax())
    top.annotate(f"{per_year[k]:,} in {years[k]}", (years[k], per_year[k]),
                 xytext=(-6, 2), textcoords="offset points", ha="right",
                 va="bottom", fontsize=6, color=INK_2)

    # Antialias in RGBA so rows thinner than a pixel blend instead of alias.
    bot.imshow(grid, aspect="auto", interpolation="antialiased",
               interpolation_stage="rgba", cmap=ListedColormap([LAND, SYS_INK]),
               vmin=0, vmax=1, extent=(y0 - 0.5, LAST_WY + 0.5, len(order), 0))
    # Historical peaks are too sparse to survive the raster: draw them as ticks.
    bot.scatter(hist["wy"], row[hist["site_no"]] + 0.5, s=4, marker="|",
                color=HIST_INK, linewidths=0.6, zorder=3)
    strip_axes(bot, keep=("bottom", "left"))
    bot.set_ylabel("Sites, by first systematic year", fontsize=6.5, color=INK_2)
    bot.set_yticks([0, 1000, 2000, 3000])
    bot.set_yticklabels(["0", "1,000", "2,000", "3,000"])
    bot.set_xlabel("Water year", fontsize=6.5, color=INK_2)
    bot.set_xlim(FIRST_WY - 0.5, LAST_WY + 0.5)
    bot.legend(handles=[
        Line2D([], [], marker="s", linestyle="none", markersize=5,
               markerfacecolor=SYS_INK, markeredgewidth=0, label="Systematic peak"),
        Line2D([], [], marker="s", linestyle="none", markersize=5,
               markerfacecolor=HIST_INK, markeredgewidth=0,
               label=f"Historical peak, code 7 ({int((summ['hist_years'] > 0).sum())} sites;"
                     f" {int((hist['wy'] < FIRST_WY).sum())} peaks before {FIRST_WY} not shown)"),
    ], loc="lower left", frameon=False, fontsize=6, handletextpad=0.3)
    med = summ[["first", "years", "gap"]].median()
    top.text(0.0, 0.97,
             f"Median: first year {med['first']:.0f} · {med['years']:.0f} years of record"
             f" · {int((summ['gap'] > 0).sum()):,} sites with gaps",
             transform=top.transAxes, ha="left", va="top", fontsize=6, color=INK_2)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.08)
    return fig


def fig_length_map(sites: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.0, 4.6), subplot_kw={"projection": PROJ})
    ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    ax.add_geometries(_conus_geoms(), crs=ccrs.PlateCarree(), facecolor=LAND,
                      edgecolor=STATE, linewidth=0.3, zorder=0)
    ax.spines["geo"].set_visible(False)
    # Long records first, so the short ones (the weaker fits) sit on top.
    s = sites.sort_values("years", ascending=False)
    ax.scatter(s["longitude"], s["latitude"], s=8, c=s["years"], cmap=REC_CMAP,
               norm=REC_NORM, edgecolors=LAND, linewidths=0.2,
               transform=ccrs.PlateCarree(), zorder=3)
    ax.set_title("Years of systematic peak record", loc="left", pad=4,
                 fontdict=TITLE_FONT)

    share = 100 * np.bincount(REC_NORM(sites["years"]), minlength=len(REC_COLORS)) / len(sites)
    ins = ax.inset_axes([0.0, 0.03, 0.24, 0.24])
    x = np.arange(len(REC_COLORS))
    ins.bar(x, share, width=0.82, color=REC_COLORS, linewidth=0)
    for xi, v in zip(x, share):
        ins.text(xi, v + share.max() * 0.04, f"{v:.0f}", ha="center", va="bottom",
                 fontsize=4.5, color=INK_2)
    ins.set_ylim(0, share.max() * 1.3)
    ins.set_xlim(-0.6, len(REC_COLORS) - 0.4)
    ins.set_xticks([])
    ins.set_yticks([])
    strip_axes(ins)
    ins.patch.set_alpha(0.0)
    ins.text(0.0, 1.0, "% of sites", transform=ins.transAxes, ha="left",
             va="top", fontsize=5, color=INK_2)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                      markerfacecolor=c, markeredgewidth=0, label=l)
               for c, l in zip(REC_COLORS, REC_LABELS)]
    fig.legend(handles=handles, title="Record length (yr)", ncol=5, loc="lower left",
               bbox_to_anchor=(0.06, 0.0), frameon=False, fontsize=6.5,
               title_fontsize=7, handletextpad=0.2, columnspacing=1.0,
               alignment="left")
    ax.text(1.0, -0.005, f"{len(sites):,} sites · median {sites['years'].median():.0f} yr",
            transform=ax.transAxes, ha="right", va="top", fontsize=6, color=INK_2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.10)
    return fig


def fig_vs_rp(sites: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.10})
    # Effective record: systematic years plus the historical period, as the
    # EMA fit counts it (hist_H, else one year per code-7 peak).
    sites = sites.assign(years=sites["years"] + np.where(
        sites["hist_H"] > 0, sites["hist_H"], sites["n_hist"]))
    xmax = sites["years"].max() * 1.05
    for ax, (level, title) in zip(axes, LEVELS):
        pts, bnd = split_sites(sites, level)
        pts = pts.sort_values("v")
        ax.plot([10, xmax], [10, xmax], color=INK_2, linewidth=0.6, linestyle=(0, (3, 2)),
                zorder=1)
        ax.scatter(pts["years"], pts["v"], s=3, c=pts["v"], cmap=CMAP, norm=NORM,
                   linewidths=0, zorder=2, rasterized=True)
        ax.scatter(bnd["years"], bnd["v"], s=6, marker="^", c=bnd["v"], cmap=CMAP,
                   norm=NORM, linewidths=0, zorder=3, rasterized=True)
        # Bounds above the line certainly exceed the record length; bounds
        # below it could be either, so they count only toward the denominator.
        beyond = (pts["v"] > pts["years"]).sum() + (bnd["v"] > bnd["years"]).sum()
        n = len(pts) + len(bnd)
        ax.set_title(title, loc="left", pad=11, fontdict=TITLE_FONT)
        ax.text(0.0, 1.015, f"≥ {100 * beyond / n:.0f}% exceed record length",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=6, color=INK_2)
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.set_ylim(1, 1e4)
        ax.set_xlim(0, xmax)
        strip_axes(ax, keep=("bottom", "left"))
        ax.grid(axis="y", color=STATE, linewidth=0.3)
        ax.set_axisbelow(True)
    axes[0].set_yticks([1, 10, 100, 1000, 10000])
    axes[0].set_yticklabels(["1", "10", "100", "1,000", "10,000"])
    axes[0].set_ylabel("Return period (yr)", fontsize=6.5, color=INK_2)
    fig.supxlabel("Years of record (systematic + historical period)", fontsize=6.5, color=INK_2, y=0.10)
    fig.legend(handles=[
        Line2D([], [], color=INK_2, linewidth=0.6, linestyle=(0, (3, 2)),
               label="Return period = record length"),
        Line2D([], [], marker="^", linestyle="none", markersize=3.5,
               markerfacecolor=INK_2, markeredgewidth=0, label="Lower bound"),
    ], loc="lower right", bbox_to_anchor=(0.99, 0.0), ncol=2, frameon=False,
        fontsize=6, handletextpad=0.4)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.83, bottom=0.27)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--regulation", choices=list(REGULATION), default="all")
    ap.add_argument("--out", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    sites = load_sites(False, args.regulation)
    rec = site_records(sites["site_no"])
    summ = record_summary(rec)
    sites = sites.merge(summ, on="site_no", how="inner")

    args.out.mkdir(parents=True, exist_ok=True)
    for stem, fig in (("record_timeline", fig_timeline(rec, summ)),
                      ("record_length_map", fig_length_map(sites)),
                      ("record_vs_rp", fig_vs_rp(sites))):
        label_subset(fig, args.regulation, len(sites))
        stem += subset_suffix(args.regulation)
        for ext in ("png", "pdf"):
            fig.savefig(args.out / f"{stem}.{ext}", facecolor="white")
        plt.close(fig)
        print(f"→ {args.out / stem}.{{png,pdf}}")
    print(f"QC sites: {len(sites):,} · median {sites['years'].median():.0f} yr of record")


if __name__ == "__main__":
    main()
