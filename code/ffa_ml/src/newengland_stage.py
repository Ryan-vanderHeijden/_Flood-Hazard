"""Scale the HAND-SRC stage test to New England (NHD region 01, 58 HUC8s).

Builds one HAND synthetic rating curve per reach (median discharge across the reach's
HydroIDs at each shared stage level), interpolates our predicted Q2..Q500 -> stage, and
validates datum-independently (predicted vs observed stage-RISE from the NWS action
level) against the Tier-1.5 compound-Manning recipe on every gauge in the region.
"""
import glob, os
import numpy as np, pandas as pd
from scipy.optimize import brentq

FIM_DIR = os.path.expanduser("~/data/flood_hazard/fim_hand")  # raw + derived data (not in git)
HT_DIR = os.path.join(FIM_DIR, "ne_hydrotables")
CFS2CMS = 0.0283168; M2FT = 3.28084
RPS = [2,5,10,25,50,100,200,500]
CATS = [("action","action_stage_ft","action_flow_cfs"),
        ("flood","flood_stage_ft","flood_flow_cfs"),
        ("moderate","moderate_stage_ft","moderate_flow_cfs"),
        ("major","major_stage_ft","major_flow_cfs")]
USECOLS = ["feature_id","branch_id","stage","discharge_cms","default_discharge_cms","order_"]

def load_srcs():
    srcs = {}; order = {}
    for f in sorted(sorted(glob.glob(os.path.join(HT_DIR,"01*.csv"))+glob.glob(os.path.join(HT_DIR,"0430*.csv")))):
        df = pd.read_csv(f, usecols=USECOLS, low_memory=False)
        df = df[df["branch_id"]==0].copy()
        df["q"] = df["discharge_cms"].fillna(df["default_discharge_cms"])
        agg = df.groupby(["feature_id","stage"], sort=True)["q"].median().reset_index()
        ordr = df.groupby("feature_id")["order_"].first()
        for fid, g in agg.groupby("feature_id", sort=False):
            g = g.sort_values("stage")
            q = g["q"].to_numpy(); s = g["stage"].to_numpy()
            keep = np.concatenate([[True], np.diff(q) > 0])
            if keep.sum() < 3: continue
            srcs[int(fid)] = (q[keep], s[keep]); order[int(fid)] = int(ordr.loc[fid])
    return srcs, order

def stage_at(q_cms, src):
    q,s = src
    if q_cms <= q[0]:  return float(s[0])
    if q_cms >= q[-1]: return np.nan
    return float(np.interp(q_cms, q, s))

# Tier-1.5 compound Manning (kfp=10, floodplain n x2)
N=0.045
def _mq(h,W,D,S,kfp=10.0,nfp=2.0,n=N):
    if h<=0: return 0.0
    if h<=D: A=W*h;P=W+2*h;R=A/P;return (1/n)*A*R**(2/3)*np.sqrt(S)
    hf=h-D;A0=W*D;P0=W+2*D;R0=A0/P0;Qmc=(1/n)*A0*R0**(2/3)*np.sqrt(S)
    Wf=(kfp-1)*W;Af=Wf*hf;Pf=Wf+hf;Rf=Af/Pf;return Qmc+(1/(nfp*n))*Af*Rf**(2/3)*np.sqrt(S)
def invert(Qcfs,W,D,S):
    Q=Qcfs*CFS2CMS
    if not(Q>0 and W>0 and D>0 and S>0): return np.nan
    try: return brentq(lambda h:_mq(h,W,D,S)-Q,1e-4,200.0,maxiter=200)*M2FT
    except Exception: return np.nan

def skill(obs,pred):
    m=np.isfinite(obs)&np.isfinite(pred)&(obs>0); o,p=obs[m],pred[m]
    if len(o)<5: return (len(o),np.nan,np.nan,np.nan,np.nan)
    corr=np.corrcoef(o,p)[0,1]; r2=1-np.sum((o-p)**2)/np.sum((o-o.mean())**2)
    return (int(m.sum()),corr,r2,float(np.median(p-o)),float(np.mean(p-o)))

if __name__ == "__main__":
    print("Loading HAND SRCs (region 01) ...")
    srcs, order = load_srcs()
    print(f"  reaches with usable SRC: {len(srcs):,}")

    # return-period stage product
    pred = pd.read_parquet(os.path.expanduser("~/data/flood_hazard/ml/conus_predictions.parquet"),
                           columns=["COMID"]+[f"q{t}_cfs" for t in RPS])
    pred = pred[pred["COMID"].isin(srcs)].copy()
    rows=[]
    for r in pred.itertuples(index=False):
        fid=int(r.COMID); src=srcs[fid]; rec={"feature_id":fid,"order_":order[fid]}
        over=False
        for t in RPS:
            q=getattr(r,f"q{t}_cfs"); st=stage_at(q*CFS2CMS,src) if np.isfinite(q) else np.nan
            rec[f"q{t}_stage_ft"]=st*M2FT if np.isfinite(st) else np.nan
            if not np.isfinite(st): over=True
        rec["src_ceiling_ft"]=src[1][-1]*M2FT; rec["exceeds_src"]=over; rows.append(rec)
    stage=pd.DataFrame(rows); stage.to_parquet(os.path.join(FIM_DIR,"newengland_tier2_stage.parquet"),index=False)
    print(f"  reaches with predicted stage: {len(stage):,}  "
          f"(Q100 above SRC ceiling: {stage['q100_stage_ft'].isna().mean()*100:.2f}%)")

    # validation at in-region gauges
    proto = pd.read_parquet(os.path.join(FIM_DIR,"src_gauge_proto.parquet"))
    proto = proto[proto["COMID"].isin(srcs)].copy()
    for cat,scol,fcol in CATS:
        proto[f"h2_{cat}"]=[stage_at(q*CFS2CMS,srcs[int(c)])*M2FT if (np.isfinite(q) and q>0) else np.nan
                            for q,c in zip(proto[fcol],proto["COMID"])]
        proto[f"h1_{cat}"]=[invert(q,W,D,S) for q,W,D,S in
                            zip(proto[fcol],proto["BANKFULL_WIDTH"],proto["BANKFULL_DEPTH"],proto["slope"])]
    proto.to_parquet(os.path.join(FIM_DIR,"newengland_validation.parquet"),index=False)
    print(f"\nIn-region gauges: {len(proto)}")
    print(f"{'tier':7s} {'cat':9s} {'n':>4s} {'corr':>6s} {'R2':>7s} {'medBias':>8s} {'meanBias':>9s}")
    for cat,scol,fcol in CATS[1:]:
        obs=(proto[scol]-proto["action_stage_ft"]).to_numpy()
        for tier,pre in [("HAND","h2_"),("Tier1.5","h1_")]:
            n,corr,r2,mb,mnb=skill(obs,(proto[f"{pre}{cat}"]-proto[f"{pre}action"]).to_numpy())
            print(f"{tier:7s} {cat:9s} {n:4d} {corr:6.3f} {r2:7.2f} {mb:+8.2f} {mnb:+9.2f}")
        print()
