from __future__ import annotations

"""
Render a CONUS overview PNG of predicted Q{rp} from the catchment geoparquet.

Iterates the per-VPU geoparquet parts (bounded memory — one region at a time),
uses catchment centroids coloured by log10 of the predicted discharge, and
writes ``fig_conus_q{rp}_catchments.png``.  This is a quick visual check of the
mapped product; full polygon rendering belongs in a GIS.

Example
-------
    python map_geoparquet.py --rp 10
"""

import argparse
import logging
import warnings
from pathlib import Path
import colorcet as cc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import geopandas as gpd

logger = logging.getLogger(__name__)

PARTS_DIR = Path.home() / "data" / "flood_hazard" / "ml" / "catchment_parts"
REPORT_DIR = Path(__file__).resolve().parent


def build(rp: int = 10) -> Path:
    qcol = f"q{rp}_cfs"
    xs, ys, vs = [], [], []
    for part in sorted(PARTS_DIR.glob("catchments_*.parquet")):
        g = gpd.read_parquet(part, columns=[qcol, "geometry"])
        g = g[g[qcol] > 0]
        with warnings.catch_warnings():  # centroid on geographic CRS — fine for a viz
            warnings.simplefilter("ignore")
            c = g.geometry.centroid
        xs.append(c.x.to_numpy())
        ys.append(c.y.to_numpy())
        vs.append(g[qcol].to_numpy())
        logger.info("  %s: %d catchments", part.stem, len(g))
    x = np.concatenate(xs); y = np.concatenate(ys); v = np.concatenate(vs)
    logger.info("Total plotted: %d catchments", len(v))
    logger.info("Q%d range: %.1f – %.1f cfs", rp, v.min(), v.max())

    fig, ax = plt.subplots(figsize=(11, 6.6))
    sc = ax.scatter(x, y, c=v, cmap=cc.cm.kbc_r, s=0.6,
                    norm=LogNorm(vmin=v.min(), vmax=v.max()), linewidths=0)
    ax.set_xlim(-125, -66); ax.set_ylim(24, 50)
    ax.set_aspect(1.25); ax.set_axis_off()
    ax.set_title(f"Predicted {rp}-year flood (Q{rp}) — {len(v):,} NHDPlus catchments", fontsize=12)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cbar.set_label(f"Q{rp} (cfs)")
    fig.tight_layout()
    out = REPORT_DIR / f"fig_conus_q{rp}_catchments.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rp", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(args.rp)


if __name__ == "__main__":
    main()
