"""New England HAND maps (native EPSG:5070): return-period stage + per-reach modeled bias.

Layout shared across panels: state boundaries + the NHD network coloured per reach,
line width by stream order.  Stage panels use colorcet kbc_r; the bias panel colours
each reach by its expected HAND stage bias, estimated from the per-stream-order median
gauge residual (major category, 97 New England gauges) — a size-based bias model, not a
per-reach measurement.
"""
import glob, os, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import colorcet as cc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm

FIM_DIR = Path.home() / "data" / "flood_hazard" / "fim_hand"   # data (not in git)
REPORT_DIR = Path(__file__).resolve().parent                    # figures live here

STAGE = pd.read_parquet(FIM_DIR / "newengland_tier2_stage.parquet").set_index("feature_id")

# per-order major-stage bias from the validation (predicted - observed rise)
val = pd.read_parquet(FIM_DIR / "newengland_validation.parquet")
val["res_major"] = (val["h2_major"]-val["h2_action"]) - (val["major_stage_ft"]-val["action_stage_ft"])
BIAS_BY_ORDER = val.groupby("stream_order")["res_major"].median()

# states — US Census 1:500k cartographic boundaries (crisp), NE neighbourhood, in Albers
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _s = gpd.read_file(f"zip://{FIM_DIR / 'cb_states_500k.zip'}")
    states = _s[_s["STUSPS"].isin(["VT","NH","ME","MA","CT","RI","NY","NJ","PA"])].to_crs(5070)
    VT = _s[_s["STUSPS"] == "VT"].to_crs(5070)              # for the Vermont zoom
    VT_BOUNDS = VT.total_bounds

def load_network():
    segs, q10, q100, order = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in sorted(glob.glob(str(FIM_DIR / "ne_streams" / "*.gpkg"))):
            g = gpd.read_file(f, columns=["ID","order_"])
            g = g[g["ID"].isin(STAGE.index)]
            if g.empty: continue
            g = g.join(STAGE[["q10_stage_ft","q100_stage_ft"]], on="ID")
            for geom, a, b, od in zip(g.geometry, g.q10_stage_ft, g.q100_stage_ft, g["order_"]):
                if geom is None or geom.geom_type != "LineString" or not np.isfinite(a): continue
                segs.append(np.asarray(geom.coords)); q10.append(a); q100.append(b); order.append(od)
    return segs, np.array(q10), np.array(q100), np.array(order)

segs, q10, q100, order = load_network()
print(f"segments: {len(segs):,}")
lw = 0.15 + 0.30*np.clip(order-1,0,7)
idx = np.argsort(order)                       # big rivers drawn on top
xmin,ymin = np.min([s.min(0) for s in segs],0); xmax,ymax = np.max([s.max(0) for s in segs],0)
pad = 0.04*(xmax-xmin)

def base(ax):
    states.boundary.plot(ax=ax, color="0.55", linewidth=0.7, zorder=1)
    states.plot(ax=ax, color="0.96", zorder=0)
    ax.set_xlim(xmin-pad, xmax+pad); ax.set_ylim(ymin-pad, ymax+pad)
    ax.set_aspect("equal"); ax.set_axis_off()

def stage_panel(vals, title, fname):
    fig, ax = plt.subplots(figsize=(8.5,9)); base(ax)
    norm = Normalize(vmin=np.nanpercentile(vals,2), vmax=np.nanpercentile(vals,98))
    lc = LineCollection([segs[i] for i in idx], array=vals[idx], cmap=cc.cm.kbc_r,
                        norm=norm, linewidths=lw[idx], zorder=3)
    ax.add_collection(lc)
    ax.set_title(f"Northeast (New England + Champlain) — predicted {title} stage above channel (ft)\n"
                 f"{len(segs):,} NHD reaches · HAND synthetic rating curves", fontsize=12)
    cb = fig.colorbar(lc, ax=ax, shrink=0.55, pad=0.01); cb.set_label(f"{title} stage (ft)")
    fig.tight_layout(); fig.savefig(REPORT_DIR / fname, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("wrote", fname)

stage_panel(q10, "Q10", "fig_ne_stage_map_q10.png")
stage_panel(q100, "Q100", "fig_ne_stage_map_q100.png")

# --- Vermont zoom (both CT-River and Champlain sides) ---
def vt_zoom(vals, title, fname):
    x0,y0,x1,y1 = VT_BOUNDS; px=0.06*(x1-x0); py=0.06*(y1-y0)
    fig, ax = plt.subplots(figsize=(7,9))
    states.plot(ax=ax, color="0.96", zorder=0)
    states.boundary.plot(ax=ax, color="0.55", linewidth=0.8, zorder=1)
    VT.boundary.plot(ax=ax, color="0.15", linewidth=1.4, zorder=2)
    norm = Normalize(vmin=np.nanpercentile(vals,2), vmax=np.nanpercentile(vals,98))
    lc = LineCollection([segs[i] for i in idx], array=vals[idx], cmap=cc.cm.kbc_r,
                        norm=norm, linewidths=(lw[idx]+0.25), zorder=3)
    ax.add_collection(lc)
    ax.set_xlim(x0-px, x1+px); ax.set_ylim(y0-py, y1+py)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(f"Vermont — predicted {title} stage above channel (ft)\n"
                 "Connecticut R. (east) + Lake Champlain (west) basins", fontsize=12)
    cb = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.01); cb.set_label(f"{title} stage (ft)")
    fig.tight_layout(); fig.savefig(REPORT_DIR / fname, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", fname)

vt_zoom(q10, "Q10", "fig_vt_stage_map_q10.png")

# per-reach modeled bias (major) by stream order
bias = np.array([BIAS_BY_ORDER.get(o, np.nan) for o in order])
fig, ax = plt.subplots(figsize=(8.5,9)); base(ax)
vmin, vmax = np.nanmin(bias), max(np.nanmax(bias), 0.01)
norm = TwoSlopeNorm(vmin=min(vmin,-0.01), vcenter=0.0, vmax=vmax)
lc = LineCollection([segs[i] for i in idx], array=bias[idx], cmap="RdBu",
                    norm=norm, linewidths=lw[idx], zorder=3)
ax.add_collection(lc)
ax.set_title("Northeast (New England + Champlain) — modeled HAND major-stage bias (ft)\n"
             "per reach from per-stream-order median gauge residual (red = under-predicts)", fontsize=11)
cb = fig.colorbar(lc, ax=ax, shrink=0.55, pad=0.01); cb.set_label("expected predicted − observed rise (ft)")
fig.tight_layout(); fig.savefig(REPORT_DIR / "fig_ne_bias_map.png", dpi=160, bbox_inches="tight"); plt.close(fig)
print("wrote fig_ne_bias_map.png")
print("bias by order (ft):", {int(k):round(v,2) for k,v in BIAS_BY_ORDER.items()})
