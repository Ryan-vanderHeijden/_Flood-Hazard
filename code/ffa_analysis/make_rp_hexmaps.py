"""
Hexagon-binned companion to make_rp_maps.py — regional LP3 return periods.

Sites are snapped to the nearest centre of a pointy-top hexagonal grid in the
map's Albers coordinates (`--hex-km` centre to centre, as in the regime hex
map in backgrounds/ideas.py).  Each hexagon is coloured by the Kaplan–Meier
median return period of its sites, with lower bounds treated as right-
censored, using the same classes and colours as the site maps.

Lightness encodes return period here, so sparse hexagons are not faded (that
would read as a shorter return period); instead a hexagon's size grows with
its site count, from half size at `--min-sites` to full size at
`--full-sites`.  Hexagons with fewer than `--min-sites` sites are skipped.
Where more than half of a hexagon's sites are lower bounds the median is
itself only a lower bound: the hexagon takes the class of the bound and is
hatched, like the lower-bound tops of the site-map histograms.

Usage:
  python make_rp_hexmaps.py                    # 150 km hexagons
  python make_rp_hexmaps.py --hex-km 200 --ppcc

Writes rp_hexmaps_conus_<km>km.{png,pdf} to FIG_DIR.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, RegularPolygon

from make_rp_maps import (
    COLORS, CONUS_EXTENT, DOT_EDGE, FIG_DIR, HATCH, HATCH_INK, INK_2, LAND,
    LABELS, LEVELS, NORM, PROJ, STATE, TITLE_FONT, _conus_geoms, km_median, load_sites,
    split_sites,
)


def hex_cells(x: np.ndarray, y: np.ndarray, h: float, x0: float, y0: float) -> np.ndarray:
    """(i, j) of the nearest pointy-top hexagon centre, spacing h, origin (x0, y0)."""
    rh = h * np.sqrt(3) / 2
    best_d, best = np.full(x.size, np.inf), np.zeros((x.size, 2), int)
    j0 = np.round((y - y0) / rh).astype(int)
    for dj in (-1, 0, 1):
        j = j0 + dj
        i = np.round((x - x0) / h - 0.5 * (j % 2)).astype(int)
        dist = np.hypot(x - (x0 + h * (i + 0.5 * (j % 2))), y - (y0 + rh * j))
        better = dist < best_d
        best_d[better], best[better] = dist[better], np.c_[i, j][better]
    return best


def hex_summary(sites: pd.DataFrame, level: str, h: float, x0: float, y0: float) -> pd.DataFrame:
    """One row per hexagon: centre, site count, KM median and whether it is a bound."""
    pts, bnd = split_sites(sites, level)
    both = pd.concat([pts.assign(cens=False), bnd.assign(cens=True)])
    cells = hex_cells(both["x"].to_numpy(), both["y"].to_numpy(), h, x0, y0)
    both["i"], both["j"] = cells[:, 0], cells[:, 1]
    rows = []
    for (i, j), g in both.groupby(["i", "j"]):
        v, c = g["v"].to_numpy(), g["cens"].to_numpy()
        med = km_median(v, c)
        is_bound = np.isnan(med)   # survival never reaches 0.5: median > every event
        if is_bound:
            # Each true value is >= its recorded value, so the true median is
            # >= the plain median of the recorded values.
            med = float(np.median(v))
        rows.append({
            "cx": x0 + h * (i + 0.5 * (j % 2)), "cy": y0 + h * np.sqrt(3) / 2 * j,
            "n": len(g), "median": med, "is_bound": is_bound,
        })
    return pd.DataFrame(rows)


def draw_panel(ax, hexes: pd.DataFrame, title: str, h: float, min_sites: int,
               full_sites: int, geoms) -> None:
    ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor=LAND,
                      edgecolor=STATE, linewidth=0.3, zorder=0)
    ax.spines["geo"].set_visible(False)

    shown = hexes[hexes["n"] >= min_sites]
    radius = h / np.sqrt(3)   # pointy-top circumradius that tiles at spacing h
    for row in shown.itertuples():
        k = int(NORM(row.median))
        frac = np.clip((row.n - min_sites) / max(full_sites - min_sites, 1), 0, 1)
        r = radius * np.sqrt(0.25 + 0.75 * frac)   # area from 1/4 to full
        ax.add_patch(RegularPolygon(
            (row.cx, row.cy), numVertices=6, radius=r, orientation=0,
            facecolor=COLORS[k], edgecolor=HATCH_INK[k] if row.is_bound else DOT_EDGE,
            hatch=HATCH if row.is_bound else None, linewidth=0 if row.is_bound else 0.2,
            transform=PROJ, zorder=3,
        ))
        if row.is_bound:   # 3.10 hatches in the edge colour: outline in a second pass
            ax.add_patch(RegularPolygon(
                (row.cx, row.cy), numVertices=6, radius=r, orientation=0,
                fill=False, edgecolor=DOT_EDGE, linewidth=0.2,
                transform=PROJ, zorder=3,
            ))

    ax.set_title(title, loc="left", pad=2, fontdict=TITLE_FONT)
    n_bound = int(shown["is_bound"].sum())
    ax.text(0.0, -0.02,
            f"{int(shown['n'].sum()):,} sites · {len(shown):,} hexagons"
            f" · {n_bound} with a lower-bound median",
            transform=ax.transAxes, ha="left", va="top", color=INK_2, fontsize=6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--hex-km", type=float, default=150)
    ap.add_argument("--min-sites", type=int, default=3)
    ap.add_argument("--full-sites", type=int, default=10)
    ap.add_argument("--ppcc", action="store_true",
                    help="also require the PPCC goodness-of-fit test to pass")
    ap.add_argument("--out", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    sites = load_sites(args.ppcc)
    xy = PROJ.transform_points(ccrs.PlateCarree(), sites["longitude"].to_numpy(),
                               sites["latitude"].to_numpy())
    sites["x"], sites["y"] = xy[:, 0], xy[:, 1]
    h = args.hex_km * 1000
    x0, y0 = sites["x"].min(), sites["y"].min()   # one grid for all four panels
    geoms = _conus_geoms()

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9),
                             subplot_kw={"projection": PROJ},
                             gridspec_kw={"hspace": 0.12, "wspace": 0.03})
    for ax, (level, title) in zip(axes.flat, LEVELS):
        hexes = hex_summary(sites, level, h, x0, y0)
        draw_panel(ax, hexes, title, h, args.min_sites, args.full_sites, geoms)

    swatch = [
        Line2D([], [], marker="h", linestyle="none", markersize=6,
               markerfacecolor=c, markeredgecolor=DOT_EDGE,
               markeredgewidth=0.3, label=l)
        for c, l in zip(COLORS, LABELS)
    ]
    leg1 = fig.legend(handles=swatch, title="Median return period (yr)", ncol=6,
                      loc="lower left", bbox_to_anchor=(0.06, 0.0),
                      frameon=False, handletextpad=0.2, columnspacing=1.0,
                      title_fontsize=7, alignment="left")
    size_key = tuple(
        Line2D([], [], marker="h", linestyle="none", markersize=ms,
               markerfacecolor=INK_2, markeredgecolor=DOT_EDGE, markeredgewidth=0.3)
        for ms in (3.5, 5, 7)
    )
    fig.legend(handles=[size_key,
                        Patch(facecolor=INK_2, edgecolor=LAND, linewidth=0, hatch=HATCH)],
               labels=[f"{args.min_sites} → {args.full_sites}+ sites",
                       "Median is a lower bound"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
               ncol=1, loc="lower right", bbox_to_anchor=(0.97, 0.0),
               frameon=False, handlelength=3.0, handletextpad=0.5)
    fig.add_artist(leg1)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.11)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"rp_hexmaps_conus_{args.hex_km:g}km" + ("_ppcc" if args.ppcc else "")
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"{stem}.{ext}", facecolor="white")
    print(f"QC sites: {len(sites):,} → {args.out / stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
