"""Diagnose HAND stage-rise bias vs stream order / drainage area / slope, + a gauge-residual map."""
import glob, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pyproj import Transformer

FIM_DIR = Path.home() / "data" / "flood_hazard" / "fim_hand"
REPORT_DIR = Path(__file__).resolve().parent

v = pd.read_parquet(FIM_DIR / "newengland_validation.parquet")
si = pd.read_parquet("/home/ryan/data/flood_hazard/metadata/site_info.parquet",
                     columns=["site_no","latitude","longitude","drainage_area_sqmi"])
v = v.merge(si, on="site_no", how="left")

# HAND stage-rise residual (predicted - observed), per category
for cat in ["flood","moderate","major"]:
    v[f"res_{cat}"] = (v[f"h2_{cat}"]-v["h2_action"]) - (v[f"{cat}_stage_ft"]-v["action_stage_ft"])

fig = plt.figure(figsize=(15,10)); gs = fig.add_gridspec(2,3)

# --- A: residual vs stream order (box, major) ---
axA = fig.add_subplot(gs[0,0])
orders = sorted(v["stream_order"].dropna().unique())
data = [v.loc[v.stream_order==o,"res_major"].dropna() for o in orders]
axA.boxplot(data, tick_labels=[int(o) for o in orders], showfliers=False)
axA.axhline(0,color="r",lw=0.8); axA.set_xlabel("stream order"); axA.set_ylabel("HAND major-stage residual (ft)")
axA.set_title("Residual vs stream order"); axA.grid(alpha=0.3)

# --- B: residual vs drainage area ---
axB = fig.add_subplot(gs[0,1])
for cat,c in [("flood","#2e86ab"),("major","#d1495b")]:
    m=v["drainage_area_sqmi"].notna()&v[f"res_{cat}"].notna()
    axB.scatter(v.loc[m,"drainage_area_sqmi"],v.loc[m,f"res_{cat}"],s=16,alpha=0.6,c=c,label=cat)
axB.set_xscale("log"); axB.axhline(0,color="k",lw=0.6); axB.set_xlabel("drainage area (sq mi)")
axB.set_ylabel("residual (ft)"); axB.set_title("Residual vs drainage area"); axB.legend(fontsize=8); axB.grid(alpha=0.3)

# --- C: residual vs reach slope ---
axC = fig.add_subplot(gs[0,2])
for cat,c in [("flood","#2e86ab"),("major","#d1495b")]:
    m=(v["slope"]>0)&v[f"res_{cat}"].notna()
    axC.scatter(v.loc[m,"slope"],v.loc[m,f"res_{cat}"],s=16,alpha=0.6,c=c,label=cat)
axC.set_xscale("log"); axC.axhline(0,color="k",lw=0.6); axC.set_xlabel("reach slope (m/m)")
axC.set_ylabel("residual (ft)"); axC.set_title("Residual vs reach slope"); axC.legend(fontsize=8); axC.grid(alpha=0.3)

# --- D: map of gauge residual (major) over faint network ---
axD = fig.add_subplot(gs[1,:])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for f in sorted(glob.glob(str(FIM_DIR / "ne_streams" / "*.gpkg"))):
        g=gpd.read_file(f,columns=["order_"])
        segs=[np.asarray(ge.coords) for ge in g.geometry if ge is not None and ge.geom_type=="LineString"]
        axD.add_collection(LineCollection(segs,colors="0.82",linewidths=0.2))
tr=Transformer.from_crs(4326,5070,always_xy=True)
m=v["latitude"].notna()&v["res_major"].notna()
x,y=tr.transform(v.loc[m,"longitude"].values,v.loc[m,"latitude"].values)
res=v.loc[m,"res_major"].values
sc=axD.scatter(x,y,c=res,cmap="RdBu",vmin=-6,vmax=6,s=55,edgecolor="k",linewidth=0.4,zorder=5)
axD.autoscale(); axD.set_aspect("equal"); axD.set_axis_off()
axD.set_title(f"Gauge HAND major-stage residual (ft) — blue = HAND under-predicts  (n={m.sum()})")
fig.colorbar(sc,ax=axD,shrink=0.6,pad=0.01,label="predicted − observed rise (ft)")

fig.suptitle("New England HAND stage-rise diagnostics",fontsize=14,y=1.0)
fig.tight_layout(); fig.savefig(REPORT_DIR / "fig_ne_diagnostics.png",dpi=150,bbox_inches="tight")

# print correlations of residual with drivers
print("Spearman corr of MAJOR residual with drivers:")
for col in ["stream_order","drainage_area_sqmi","slope","BANKFULL_WIDTH"]:
    s=v[["res_major",col]].dropna()
    print(f"  {col:20s} rho={s['res_major'].corr(s[col],method='spearman'):+.2f}  (n={len(s)})")
print("\nMedian residual by order (major):")
print(v.groupby('stream_order')['res_major'].agg(['median','count']).round(2).to_string())
