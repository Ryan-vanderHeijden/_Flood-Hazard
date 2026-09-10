"""Tier-1 SRC ceiling test: can a floodplain/compound section rescue Manning+Bieger?

Datum-independent validation: compare predicted vs observed STAGE RISE from the
'action' category up to flood/moderate/major, across section models. If a calibrated
floodplain-widening term lifts corr from ~0.3 toward ~0.7, Tier-1.5 is viable; if not,
the limit is per-reach geometry (needs HAND / Tier 2).
"""
import os
import numpy as np, pandas as pd
from scipy.optimize import brentq

FIM_DIR = os.path.expanduser("~/data/flood_hazard/fim_hand")
df = pd.read_parquet(os.path.join(FIM_DIR, "src_gauge_proto.parquet"))
N = 0.045; CFS2CMS = 0.0283168; M2FT = 3.28084
CATS = [("action","action_stage_ft","action_flow_cfs"),
        ("flood","flood_stage_ft","flood_flow_cfs"),
        ("moderate","moderate_stage_ft","moderate_flow_cfs"),
        ("major","major_stage_ft","major_flow_cfs")]

def manning_Q(h, Wbf, Dbf, S, kfp, n=N, nfp_mult=1.0):
    """Discharge (cms) at depth h(m). Compound: rectangular main channel width Wbf up to
    Dbf; above Dbf a floodplain of extra width (kfp-1)*Wbf with roughness nfp_mult*n."""
    if h <= 0: return 0.0
    if h <= Dbf or kfp <= 1.0:
        A = Wbf*h; P = Wbf + 2*h
        R = A/P
        return (1.0/n)*A*R**(2/3)*np.sqrt(S)
    # main channel (full) + floodplain overbank block
    hf = h - Dbf
    A_mc = Wbf*Dbf; P_mc = Wbf + 2*Dbf; R_mc = A_mc/P_mc
    Q_mc = (1.0/n)*A_mc*R_mc**(2/3)*np.sqrt(S)
    Wfp = (kfp-1.0)*Wbf
    A_fp = Wfp*hf; P_fp = Wfp + hf; R_fp = A_fp/P_fp
    Q_fp = (1.0/(nfp_mult*n))*A_fp*R_fp**(2/3)*np.sqrt(S)
    return Q_mc + Q_fp

def invert(Qcms, Wbf, Dbf, S, kfp, nfp_mult=1.0):
    if not (Qcms>0 and Wbf>0 and Dbf>0 and S>0): return np.nan
    f = lambda h: manning_Q(h,Wbf,Dbf,S,kfp,nfp_mult=nfp_mult)-Qcms
    try:
        return brentq(f, 1e-4, 200.0, maxiter=200)
    except Exception:
        return np.nan

d = df.dropna(subset=["BANKFULL_WIDTH","BANKFULL_DEPTH","slope"]).copy()
d = d[(d.BANKFULL_WIDTH>0)&(d.BANKFULL_DEPTH>0)&(d.slope>0)]
print(f"gauges usable: {len(d)}")

def evaluate(kfp, nfp_mult=1.0):
    # depth (ft) at each category
    depth = {}
    for cat,scol,fcol in CATS:
        q = d[fcol].to_numpy()*CFS2CMS
        W=d.BANKFULL_WIDTH.to_numpy(); D=d.BANKFULL_DEPTH.to_numpy(); S=d.slope.to_numpy()
        depth[cat] = np.array([invert(qi,Wi,Di,Si,kfp,nfp_mult)*M2FT if np.isfinite(qi) and qi>0 else np.nan
                               for qi,Wi,Di,Si in zip(q,W,D,S)])
    rows=[]
    for cat,scol,fcol in CATS[1:]:
        obs = d[scol].to_numpy() - d["action_stage_ft"].to_numpy()
        pred = depth[cat] - depth["action"]
        m = np.isfinite(obs)&np.isfinite(pred)&(obs>0)
        o,p = obs[m],pred[m]
        corr = np.corrcoef(o,p)[0,1]
        ss_res=np.sum((o-p)**2); ss_tot=np.sum((o-o.mean())**2)
        r2 = 1-ss_res/ss_tot
        rows.append((cat,m.sum(),corr,r2,np.median(p-o),np.mean(p-o)))
    return rows

print(f"\n{'model':28s} {'cat':9s} {'n':>5s} {'corr':>6s} {'R2':>8s} {'medBias':>8s} {'meanBias':>9s}")
configs = [("rectangular (kfp=1)",1.0,1.0),
           ("compound kfp=5",5.0,1.0),
           ("compound kfp=10",10.0,1.0),
           ("compound kfp=20",20.0,1.0),
           ("compound kfp=10, nfp x2",10.0,2.0),
           ("compound kfp=40",40.0,1.0)]
for name,kfp,nfp in configs:
    for cat,n,corr,r2,mb,mnb in evaluate(kfp,nfp):
        print(f"{name:28s} {cat:9s} {n:5d} {corr:6.3f} {r2:8.2f} {mb:+8.2f} {mnb:+9.2f}")
    print()
