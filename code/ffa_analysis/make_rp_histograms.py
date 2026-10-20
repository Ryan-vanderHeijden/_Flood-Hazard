"""
Histograms of LP3 return-period classes at the NWS thresholds.

A standalone version of the inset histograms in make_rp_maps.py: one panel per
threshold (action / minor / moderate / major) on a shared y-scale, bars as the
share of the panel's QC sites in each return-period class, coloured like the
maps.  Lower bounds (thresholds above the rating table) are the hatched top of
each bar, counted in the class of the bound (their true class can only be the
same or rarer).  Each panel notes its site count and Kaplan–Meier median.

Usage:
  python make_rp_histograms.py
  python make_rp_histograms.py --regulation unregulated   # or: regulated
  python make_rp_histograms.py --ppcc

Writes rp_histograms[_ppcc][_<regulation>].{png,pdf} to FIG_DIR.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from make_rp_maps import (
    COLORS, DOT_EDGE, FIG_DIR, HATCH, HATCH_INK, INK_2, LABELS, LAND, LEVELS,
    REGULATION, TITLE_FONT, class_shares, km_median, label_subset, load_sites,
    split_sites, subset_suffix,
)


def draw_panel(ax, sites, level: str, title: str, ymax: float) -> None:
    pts, bnd = split_sites(sites, level)
    p_pts, p_bnd = class_shares(pts, bnd)
    x = np.arange(len(COLORS))
    ax.bar(x, p_pts, width=0.82, color=COLORS, edgecolor=DOT_EDGE, linewidth=0.3)
    # matplotlib 3.10 draws hatching in the edge colour: hatch, then outline.
    ax.bar(x, p_bnd, bottom=p_pts, width=0.82, color=COLORS,
           edgecolor=HATCH_INK, linewidth=0, hatch=HATCH)
    ax.bar(x, p_bnd, bottom=p_pts, width=0.82, fill=False,
           edgecolor=DOT_EDGE, linewidth=0.3)
    for xi, tot in zip(x, p_pts + p_bnd):
        if tot >= 0.5:
            ax.text(xi, tot + ymax * 0.02, f"{tot:.0f}", ha="center", va="bottom",
                    fontsize=6, color=INK_2)

    med = km_median(
        np.concatenate([pts["v"].to_numpy(), bnd["v"].to_numpy()]),
        np.r_[np.zeros(len(pts), bool), np.ones(len(bnd), bool)],
    )
    ax.set_title(title, loc="left", pad=4, fontdict=TITLE_FONT)
    ax.text(0.0, 1.0, f"{len(pts) + len(bnd):,} sites · median {med:.1f} yr",
            transform=ax.transAxes, ha="left", va="top", fontsize=6, color=INK_2)

    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, len(COLORS) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=5.5, color=INK_2)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(INK_2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--ppcc", action="store_true",
                    help="also require the PPCC goodness-of-fit test to pass")
    ap.add_argument("--regulation", choices=list(REGULATION), default="all")
    ap.add_argument("--out", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    sites = load_sites(args.ppcc, args.regulation)
    # One y-scale for all panels, with headroom for the tallest bar's label
    # and the panel note.
    ymax = 1.3 * max(sum(class_shares(*split_sites(sites, lv))).max() for lv, _ in LEVELS)

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.4), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, (level, title) in zip(axes, LEVELS):
        draw_panel(ax, sites, level, title, ymax)

    fig.text(0.01, 0.5, "% of sites", rotation=90, ha="left", va="center",
             fontsize=7, color=INK_2)
    fig.supxlabel("Return period (yr)", fontsize=7, color=INK_2, y=0.08)
    fig.legend(handles=[Patch(facecolor=INK_2, edgecolor=LAND, linewidth=0, hatch=HATCH)],
               labels=["Lower bounds (stage above rating)"], loc="lower right",
               bbox_to_anchor=(0.99, 0.0), frameon=False, fontsize=6,
               handlelength=2.0, handletextpad=0.5)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.25)
    label_subset(fig, args.regulation, len(sites))

    args.out.mkdir(parents=True, exist_ok=True)
    stem = "rp_histograms" + ("_ppcc" if args.ppcc else "") + subset_suffix(args.regulation)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"{stem}.{ext}", facecolor="white")
    print(f"QC sites: {len(sites):,} → {args.out / stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
