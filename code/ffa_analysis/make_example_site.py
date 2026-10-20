"""
Example LP3 frequency curve for one gage, with its NWS thresholds.

Annual peaks (systematic, as classified for the EMA fit) at Weibull plotting
positions on a normal-probability AEP axis, the fitted LP3 curve
(lp3_skew_used / lp3_loc / lp3_scale), and each NWS threshold flow read across
to its return period.  Styling follows make_rp_maps.py.

Usage:
  python make_example_site.py                  # 01031500, Piscataquis R. nr Dover-Foxcroft, ME
  python make_example_site.py --site 01054500

Writes example_site_<site>.{png,pdf} to FIG_DIR.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import norm, pearson3

from make_rp_maps import COLORS, FFA_DIR, FIG_DIR, INK, INK_2, META_DIR, TITLE_FONT

sys.path.insert(0, str(Path(__file__).parent / "src"))
from compute_flood_frequency import _classify_peaks  # noqa: E402

LEVELS = [("action", "Action"), ("minor", "Minor"), ("moderate", "Moderate"), ("major", "Major")]
AEP_TICKS = [0.5, 0.2, 0.1, 0.04, 0.02, 0.01, 0.002]   # 2 … 500 yr
CURVE = COLORS[3]
GRID = "#e6e5e1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--site", default="01031500")
    ap.add_argument("--out", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    ffa = pd.read_parquet(FFA_DIR / "flood_frequency.parquet").set_index("site_no").loc[args.site]
    fs = pd.read_parquet(META_DIR / "flood_stages.parquet").set_index("site_no").loc[args.site]
    name = (pd.read_parquet(META_DIR / "site_info.parquet", columns=["site_no", "station_name"])
            .drop_duplicates("site_no").set_index("site_no").at[args.site, "station_name"])
    peaks = pd.read_parquet(FFA_DIR / "annual_peaks.parquet")
    q = np.sort(_classify_peaks(peaks[peaks["site_no"] == args.site].reset_index(drop=True))["sys_peaks"])[::-1]

    skew, loc, scale = ffa["lp3_skew_used"], ffa["lp3_loc"], ffa["lp3_scale"]
    x = lambda aep: norm.ppf(1 - np.asarray(aep))     # probability axis, rarer to the right

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    aep_curve = np.logspace(np.log10(0.995), np.log10(0.0015), 300)
    ax.plot(x(aep_curve), 10 ** pearson3.ppf(1 - aep_curve, skew, loc, scale),
            color=CURVE, lw=1.6, zorder=3)
    pp = np.arange(1, len(q) + 1) / (len(q) + 1)       # Weibull plotting position = AEP
    ax.scatter(x(pp), q, s=10, color=INK_2, edgecolors="none", zorder=4)

    # Each threshold: horizontal read-across to the curve, drop to the AEP axis.
    y_floor = q.min() * 0.5
    for lvl, label in LEVELS:
        flow, rp = fs[f"{lvl}_flow_cfs"], ffa[f"{lvl}_return_period_yr"]
        if not (np.isfinite(flow) and np.isfinite(rp)):
            continue
        xr = x(1 / rp)
        ax.plot([x(0.995), xr], [flow, flow], color=INK, lw=0.6, ls=(0, (3, 2)), zorder=2)
        ax.plot([xr, xr], [flow, y_floor], color=INK, lw=0.6, ls=(0, (3, 2)), zorder=2)
        ax.text(x(0.97), flow * 1.04, f"{label}  ·  {rp:.1f} yr", fontsize=7,
                color=INK, va="bottom", ha="left")

    ax.set_yscale("log")
    ax.set_ylim(y_floor, None)
    ax.set_xlim(x(0.995), x(0.0015))
    ax.set_xticks(x(AEP_TICKS))
    ax.set_xticklabels([f"{1 / a:g}" for a in AEP_TICKS])
    ax.set_xlabel("Return period (yr)", color=INK_2)
    ax.set_ylabel("Annual peak flow (cfs)", color=INK_2)
    ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=7)
    ax.set_title(f"{name}  ({args.site})", loc="left", pad=6, fontdict=TITLE_FONT)
    # Direct labels instead of a legend (the threshold lines cross every corner).
    a_lab = 0.008   # right of a ~90-yr major line, left of the plot edge
    ax.text(x(a_lab), 10 ** pearson3.ppf(1 - a_lab, skew, loc, scale) * 0.9,
            "LP3 fit\n(EMA, station skew)", fontsize=7, color=CURVE, ha="left", va="top")
    a_pk = 0.8      # empty space below the curve, left of the action line
    ax.text(x(a_pk), 10 ** pearson3.ppf(1 - a_pk, skew, loc, scale) * 0.62,
            f"annual peaks\n(n = {len(q)})", fontsize=7, color=INK_2, ha="center", va="top")
    fig.tight_layout()

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"example_site_{args.site}"
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"{stem}.{ext}", facecolor="white")
    print(f"→ {args.out / stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
