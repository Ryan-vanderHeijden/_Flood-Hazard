"""
Publication maps — LP3 return periods of the NWS flood thresholds, CONUS.

One panel per threshold (action / minor / moderate / major).  Each QC-passing
site is a dot coloured by its return-period class.  Thresholds whose stage is
above the USGS rating table have no flow, only a lower bound on the return
period (<level>_return_period_lb_yr); they are drawn as up-triangles ("at
least") coloured by the class of that bound so they are not silently dropped (at the major level they
are ~30% of sites, and dropping them biases the map toward short return
periods).

Each panel carries an inset histogram of the same classes, as a share of the
panel's sites; lower bounds are the hatched top of each bar, counted in the
class of the bound (their true class can only be the same or rarer).

Return periods use lp3_skew_used (station skew; see compute_flood_frequency).

Usage:
  python make_rp_maps.py               # QC = record_ok & ~degenerate_fit
  python make_rp_maps.py --ppcc        # additionally require ppcc_ok
  python make_rp_maps.py --regulation unregulated   # or: regulated

Writes rp_maps_conus[_ppcc][_<regulation>].{png,pdf} to FIG_DIR.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.io import shapereader
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DATA_DIR = Path("/home/ryan/data/flood_hazard")
FFA_DIR  = DATA_DIR / "ffa"
META_DIR = DATA_DIR / "metadata"
FIG_DIR  = DATA_DIR / "figures"

LEVELS = [
    ("action",   "Action"),
    ("minor",    "Minor"),
    ("moderate", "Moderate"),
    ("major",    "Major"),
]

# Return-period classes (yr).  Darker = rarer.  Six-step multi-hue sequential
# ramp built in OKLCH (L 0.93 -> 0.26, hue 100 -> 300): lightness is monotone
# and every adjacent pair clears dE 15 (normal vision) and dE 10 (protan /
# deutan / tritan), so the two darkest classes differ in hue (navy vs purple),
# not just lightness.  Light steps sit below 2:1 on the land fill, so dots get
# a thin DOT_EDGE outline.
BREAKS  = [2, 5, 10, 25, 100]
LABELS  = ["< 2", "2–5", "5–10", "10–25", "25–100", "≥ 100"]
COLORS  = ["#fcea7f", "#8fd181", "#00ab96", "#007a9c", "#1a4386", "#2f0b54"]
CMAP    = ListedColormap(COLORS)
NORM    = BoundaryNorm([0] + BREAKS + [np.inf], CMAP.N)

LAND    = "#fcfcfb"
STATE   = "#c9c8c4"
INK     = "#2b2b29"
INK_2   = "#6b6a66"
DOT_EDGE = "#55544f"
HATCH    = "//////"
# Hatch ink per class: dark lines on the pale bars, light lines on the dark ones.
HATCH_INK = ["#55544f", "#55544f", "#fcfcfb", "#fcfcfb", "#fcfcfb", "#fcfcfb"]

CONUS_EXTENT = [-125, -66.7, 24.3, 49.6]
PROJ = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5)

DOT_SIZE = 7     # pt^2
TRI_SIZE = 11

plt.rcParams.update({
    "font.family": "Fira Sans",
    "font.size": 7,
    "axes.titlesize": 8,
    "text.color": INK,
    "pdf.fonttype": 42,     # editable text in Illustrator
    "savefig.dpi": 300,
})

# Panel titles in Literata SemiBold (the static faces ship without a Bold),
# loaded from the shared assets repo; fall back to Fira Sans if it is absent.
_LITERATA = Path.home() / "data" / "assets" / "fonts" / "Literata"
for _ttf in sorted(_LITERATA.glob("Literata-*.ttf")):
    fm.fontManager.addfont(str(_ttf))
TITLE_FONT = (
    {"family": "Literata", "weight": "semibold", "size": 9}
    if any(f.name == "Literata" for f in fm.fontManager.ttflist)
    else {"family": "Fira Sans", "weight": "bold", "size": 8}
)


def _conus_geoms():
    shp = shapereader.natural_earth(resolution="50m", category="cultural",
                                    name="admin_1_states_provinces_lakes")
    return [
        rec.geometry for rec in shapereader.Reader(shp).records()
        if rec.attributes["admin"] == "United States of America"
        and rec.attributes["name"] not in {"Alaska", "Hawaii"}
    ]


# Regulation subsets, from regulation.parquet's is_regulated (moderate or major:
# degree of regulation >= 0.10, escalated by NWIS peak codes 5/6; see
# compute_regulation.py).  Sites without a regulation profile fall in neither.
REGULATION = {
    "all": None,
    "unregulated": ("Unregulated gages", False),
    "regulated": ("Regulated gages", True),
}


def load_sites(apply_ppcc: bool, regulation: str = "all") -> pd.DataFrame:
    ffa = pd.read_parquet(FFA_DIR / "flood_frequency.parquet")
    qc = ffa["record_ok"] & ~ffa["degenerate_fit"]
    if REGULATION[regulation] is not None:
        reg = pd.read_parquet(META_DIR / "regulation.parquet",
                              columns=["site_no", "is_regulated"]).drop_duplicates("site_no")
        ffa = ffa.merge(reg, on="site_no", how="left")
        qc &= ffa["is_regulated"].eq(REGULATION[regulation][1])
    if apply_ppcc:
        ppcc = pd.read_parquet(FFA_DIR / "ppcc.parquet", columns=["site_no", "ppcc_ok"])
        ffa = ffa.merge(ppcc, on="site_no", how="left")
        qc &= ffa["ppcc_ok"].fillna(False)
    info = (
        pd.read_parquet(META_DIR / "site_info.parquet",
                        columns=["site_no", "latitude", "longitude"])
        .drop_duplicates("site_no")
    )
    return ffa[qc].merge(info, on="site_no", how="inner").dropna(
        subset=["latitude", "longitude"]
    )


def km_median(values: np.ndarray, censored: np.ndarray) -> float:
    """Kaplan–Meier median with right-censored values (lower bounds).

    Dropping the lower bounds biases the median low, because they are the
    sites whose threshold is above the rating, i.e. the rarer floods.
    """
    order = np.lexsort((censored, values))  # events before censored at ties
    v, c = values[order], censored[order]
    at_risk = len(v) - np.arange(len(v))
    surv = np.cumprod(np.where(c, 1.0, 1.0 - 1.0 / at_risk))
    below = np.nonzero(surv <= 0.5)[0]
    return float(v[below[0]]) if len(below) else np.nan


def split_sites(sites: pd.DataFrame, level: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point estimates and lower-bound-only sites, each with the value in v."""
    rp = sites[f"{level}_return_period_yr"].replace(np.inf, np.nan)
    lb = sites[f"{level}_return_period_lb_yr"]
    pts = sites[rp.notna()].assign(v=rp)
    bnd = sites[lb.notna() & rp.isna()].assign(v=lb)
    return pts, bnd


def class_shares(pts: pd.DataFrame, bnd: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Percent of the panel's sites in each class: (point estimates, bounds)."""
    n = max(len(pts) + len(bnd), 1)
    k = len(COLORS)
    return (100 * np.bincount(NORM(pts["v"]), minlength=k) / n,
            100 * np.bincount(NORM(bnd["v"]), minlength=k) / n)


def draw_hist(ax, pts: pd.DataFrame, bnd: pd.DataFrame, ymax: float) -> None:
    """Class histogram inset over the Pacific / Baja corner of the map."""
    ins = ax.inset_axes([0.0, 0.02, 0.24, 0.24])
    p_pts, p_bnd = class_shares(pts, bnd)
    x = np.arange(len(COLORS))
    ins.bar(x, p_pts, width=0.82, color=COLORS, edgecolor=DOT_EDGE, linewidth=0.2)
    # matplotlib 3.10 draws hatching in the edge colour, so hatch with
    # edgecolor=HATCH_INK and add the outline in a separate unfilled pass.
    ins.bar(x, p_bnd, bottom=p_pts, width=0.82, color=COLORS,
            edgecolor=HATCH_INK, linewidth=0, hatch=HATCH)
    ins.bar(x, p_bnd, bottom=p_pts, width=0.82, fill=False,
            edgecolor=DOT_EDGE, linewidth=0.2)
    for xi, tot in zip(x, p_pts + p_bnd):
        if tot >= 0.5:
            ins.text(xi, tot + ymax * 0.03, f"{tot:.0f}", ha="center", va="bottom",
                     fontsize=4.5, color=INK_2)
    ins.set_ylim(0, ymax)
    ins.set_xlim(-0.6, len(COLORS) - 0.4)
    ins.set_xticks([])
    ins.set_yticks([])
    for side in ("top", "right", "left"):
        ins.spines[side].set_visible(False)
    ins.spines["bottom"].set_linewidth(0.4)
    ins.spines["bottom"].set_color(INK_2)
    ins.patch.set_alpha(0.0)
    ins.text(0.0, 1.0, "% of sites", transform=ins.transAxes, ha="left",
             va="top", fontsize=5, color=INK_2)


def draw_panel(ax, sites: pd.DataFrame, level: str, title: str, geoms,
               hist_ymax: float) -> None:
    ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor=LAND,
                      edgecolor=STATE, linewidth=0.3, zorder=0)
    ax.spines["geo"].set_visible(False)

    pts, bnd = split_sites(sites, level)

    # Sort ascending so the rarer (darker, fewer) sites are drawn on top.
    pts = pts.sort_values("v")
    ax.scatter(pts["longitude"], pts["latitude"], s=DOT_SIZE, c=pts["v"],
               cmap=CMAP, norm=NORM, edgecolors=DOT_EDGE, linewidths=0.2,
               transform=ccrs.PlateCarree(), zorder=3)

    # Lower bounds on top of the dots: shape says "at least", colour gives the
    # class of the bound.
    bnd = bnd.sort_values("v")
    ax.scatter(bnd["longitude"], bnd["latitude"], s=TRI_SIZE, marker="^",
               c=bnd["v"], cmap=CMAP, norm=NORM,
               edgecolors=DOT_EDGE, linewidths=0.2,
               transform=ccrs.PlateCarree(), zorder=4)

    med = km_median(
        np.concatenate([pts["v"].to_numpy(), bnd["v"].to_numpy()]),
        np.r_[np.zeros(len(pts), bool), np.ones(len(bnd), bool)],
    )
    ax.set_title(title, loc="left", pad=2, fontdict=TITLE_FONT)
    ax.text(0.0, -0.02,
            f"{len(pts) + len(bnd):,} sites ({len(bnd):,} lower bounds)"
            f" · median {med:.1f} yr",
            transform=ax.transAxes, ha="left", va="top", color=INK_2, fontsize=6)
    draw_hist(ax, pts, bnd, hist_ymax)


def subset_suffix(regulation: str) -> str:
    return "" if regulation == "all" else f"_{regulation}"


def label_subset(fig, regulation: str, n_sites: int) -> None:
    """Name the subset top right, so a single figure is never mistaken for all gages."""
    if REGULATION[regulation] is None:
        return
    fig.text(0.99, 0.995, f"{REGULATION[regulation][0]} · {n_sites:,} QC sites",
             ha="right", va="top", fontsize=7, color=INK_2)


def legend_handles():
    swatch = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.5,
               markerfacecolor=c, markeredgecolor=DOT_EDGE,
               markeredgewidth=0.3, label=l)
        for c, l in zip(COLORS, LABELS)
    ]
    kind = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.5,
               markerfacecolor=INK_2, markeredgecolor=DOT_EDGE,
               markeredgewidth=0.3, label="LP3 return period"),
        (
            Line2D([], [], marker="^", linestyle="none", markersize=5,
                   markerfacecolor=INK_2, markeredgecolor=DOT_EDGE,
                   markeredgewidth=0.3),
            Patch(facecolor=INK_2, edgecolor=LAND, linewidth=0, hatch=HATCH),
        ),
    ]
    return swatch, kind


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--ppcc", action="store_true",
                    help="also require the PPCC goodness-of-fit test to pass")
    ap.add_argument("--regulation", choices=list(REGULATION), default="all")
    ap.add_argument("--out", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    sites = load_sites(args.ppcc, args.regulation)
    geoms = _conus_geoms()

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9),
                             subplot_kw={"projection": PROJ},
                             gridspec_kw={"hspace": 0.12, "wspace": 0.03})
    # One y-scale for all four histograms so the panels compare directly.
    # Headroom above the tallest bar for its value label and "% of sites".
    hist_ymax = 1.5 * max(
        sum(class_shares(*split_sites(sites, level))).max() for level, _ in LEVELS
    )
    for ax, (level, title) in zip(axes.flat, LEVELS):
        draw_panel(ax, sites, level, title, geoms, hist_ymax)

    swatch, kind = legend_handles()
    leg1 = fig.legend(handles=swatch, title="Return period (yr)", ncol=6,
                      loc="lower left", bbox_to_anchor=(0.06, 0.0),
                      frameon=False, handletextpad=0.2, columnspacing=1.0,
                      title_fontsize=7, alignment="left")
    fig.legend(handles=kind,
               labels=["LP3 return period",
                       "Lower bound (stage above rating)"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
               ncol=1, loc="lower right", bbox_to_anchor=(0.97, 0.0),
               frameon=False, handlelength=3.0, handletextpad=0.5)
    fig.add_artist(leg1)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.11)

    args.out.mkdir(parents=True, exist_ok=True)
    label_subset(fig, args.regulation, len(sites))
    stem = "rp_maps_conus" + ("_ppcc" if args.ppcc else "") + subset_suffix(args.regulation)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"{stem}.{ext}", facecolor="white")
    print(f"QC sites: {len(sites):,} → {args.out / stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
