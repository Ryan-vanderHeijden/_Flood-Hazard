"""Screen the annual peak records behind the LP3 fits for non-stationarity.

Bulletin 17C assumes a stationary record and asks that trends and abrupt
changes be tested for and attributed before the frequency curve is trusted
(USACE ETL 1100-2-3 lays out the same detect → attribute sequence). This is
the detection step, run on every QC site's systematic record:

**Monotonic trend** — Mann–Kendall on the annual peaks, with the Hamed & Rao
(1998) variance correction for serial correlation, and the Sen (Theil–Sen)
slope of log10 peak against water year, reported as % change per decade.

**Abrupt change** — Pettitt's test for a single change point in the median,
with the first water year after the change and the after/before ratio of
median peaks.

Both are rank tests on the same peaks the EMA fit sees: code-1 (daily mean)
and code-7 (historical) peaks are dropped. Censored years (codes 4 and 8) are
ranked at their recorded value, an approximation that only matters where
censoring is common. Sites with fewer than ``MIN_YEARS`` systematic years are
not tested. Significance is controlled across sites with the Benjamini–
Hochberg false discovery rate (q = ``FDR_Q``), because at ~3,300 tests a
plain 5% level flags ~165 sites by chance alone.

For a first attribution, each site's **regulation onset** is the first water
year NWIS flags a peak with code 5 or 6 (affected by regulation or
diversion). A Pettitt change within ``ONSET_WINDOW`` years of it is marked
as coinciding with regulation, the case Bulletin 17C handles by fitting the
post-change record only.

Writes ``data/ffa/peak_trends.parquet`` — one row per QC site.

    python code/ffa_analysis/src/compute_peak_trends.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_flood_frequency import _parse_peak_cd  # noqa: E402

DATA_DIR = Path.home() / "data" / "flood_hazard"
FFA_DIR = DATA_DIR / "ffa"
META_DIR = DATA_DIR / "metadata"
OUT_PATH = FFA_DIR / "peak_trends.parquet"

MIN_YEARS = 30       # systematic years needed to test a site
FDR_Q = 0.05         # Benjamini–Hochberg false discovery rate
ONSET_WINDOW = 5     # years between a change point and regulation onset
HR_MAX_LAG = 3       # autocorrelation lags in the Hamed–Rao correction

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def sen_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Median of all pairwise slopes (Theil–Sen); t need not be evenly spaced."""
    i, j = np.triu_indices(len(t), k=1)
    return float(np.median((y[j] - y[i]) / (t[j] - t[i])))


def mann_kendall_hr(t: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Mann–Kendall (tau, z, two-sided p) with the Hamed–Rao variance correction.

    The correction inflates the variance of S by the effective-sample-size
    ratio n/n*, from the significant lag autocorrelations of the ranks of the
    Sen-detrended series.  Only lags 1..``HR_MAX_LAG`` enter, and the ratio
    is floored at 1: with all n − 3 lags, chance "significant" negative
    autocorrelations shrink the variance and white noise is rejected about
    twice as often as the nominal 5%.
    """
    n = len(x)
    s = float(np.sign(x[None, :] - x[:, None])[np.triu_indices(n, k=1)].sum())
    _, ties = np.unique(x, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5)
             - (ties * (ties - 1) * (2 * ties + 5)).sum()) / 18.0

    r = rankdata(x - sen_slope(t, x) * t)
    r = r - r.mean()
    denom = (r ** 2).sum()
    k = np.arange(1, min(HR_MAX_LAG, n - 3) + 1)
    rho = np.array([(r[:-kk] * r[kk:]).sum() / denom for kk in k])
    sig = np.abs(rho) > norm.ppf(0.975) / np.sqrt(n)
    ratio = 1 + 2 / (n * (n - 1) * (n - 2)) * (
        ((n - k) * (n - k - 1) * (n - k - 2) * rho)[sig].sum())
    var_s *= max(ratio, 1.0)

    z = 0.0 if s == 0 else (s - np.sign(s)) / np.sqrt(var_s)
    tau = s / (n * (n - 1) / 2)
    return tau, float(z), float(2 * norm.sf(abs(z)))


def pettitt(x: np.ndarray) -> tuple[int, float]:
    """Pettitt change point: (index of the first value after the change, p)."""
    n = len(x)
    u = np.cumsum(np.sign(x[:, None] - x[None, :]).sum(1))[:-1]
    t = int(np.abs(u).argmax())
    k = float(np.abs(u[t]))
    p = min(1.0, 2 * np.exp(-6 * k ** 2 / (n ** 3 + n ** 2)))
    return t + 1, p


def bh_reject(p: pd.Series, q: float) -> pd.Series:
    """Benjamini–Hochberg rejections at FDR q; NaN p-values are not rejected."""
    ok = p.dropna().sort_values()
    m = len(ok)
    below = ok.to_numpy() <= q * np.arange(1, m + 1) / m
    cut = np.nonzero(below)[0].max() + 1 if below.any() else 0
    out = pd.Series(False, index=p.index)
    out.loc[ok.index[:cut]] = True
    return out


# ---------------------------------------------------------------------------
# Per-site screen
# ---------------------------------------------------------------------------

def systematic_peaks(peaks: pd.DataFrame) -> pd.DataFrame:
    """Systematic peaks as the fit sees them, with water year and regulation flag."""
    p = peaks[peaks["peak_va"].notna() & (peaks["peak_va"] > 0)].copy()
    codes = p["peak_cd"].apply(_parse_peak_cd)
    keep = ~codes.apply(lambda c: 1 in c or 7 in c)
    p, codes = p[keep], codes[keep]
    p["wy"] = p["datetime"].dt.year + (p["datetime"].dt.month >= 10)
    p["regulated_pk"] = codes.apply(lambda c: 5 in c or 6 in c)
    return (p.sort_values(["site_no", "wy"])
             .drop_duplicates(["site_no", "wy"])[["site_no", "wy", "peak_va", "regulated_pk"]])


def screen_site(g: pd.DataFrame) -> dict:
    t = g["wy"].to_numpy(float)
    y = np.log10(g["peak_va"].to_numpy(float))
    reg = g.loc[g["regulated_pk"], "wy"]
    row = {"n_years": len(g), "first_wy": int(t[0]), "last_wy": int(t[-1]),
           "reg_onset_wy": int(reg.min()) if len(reg) else np.nan}
    if len(g) < MIN_YEARS:
        return row
    tau, z, p = mann_kendall_hr(t, y)
    k, p_cp = pettitt(y)
    row.update(
        mk_tau=tau, mk_z=z, mk_p=p,
        sen_pct_decade=100 * (10 ** (10 * sen_slope(t, y)) - 1),
        pettitt_p=p_cp, change_wy=int(t[k]),
        change_pct=100 * (10 ** (np.median(y[k:]) - np.median(y[:k])) - 1),
    )
    return row


def screen(peaks: pd.DataFrame, sites: pd.Index) -> pd.DataFrame:
    sp = systematic_peaks(peaks[peaks["site_no"].isin(sites)])
    rows = [{"site_no": s, **screen_site(g)} for s, g in sp.groupby("site_no")]
    df = pd.DataFrame(rows)
    df["tested"] = df["n_years"] >= MIN_YEARS
    df["mk_sig"] = bh_reject(df["mk_p"], FDR_Q)
    df["pettitt_sig"] = bh_reject(df["pettitt_p"], FDR_Q)
    df["trend_dir"] = np.select(
        [df["mk_sig"] & (df["mk_z"] > 0), df["mk_sig"] & (df["mk_z"] < 0)],
        ["up", "down"], "none")
    df["change_near_reg_onset"] = (
        df["pettitt_sig"]
        & ((df["change_wy"] - df["reg_onset_wy"]).abs() <= ONSET_WINDOW))
    return df


def report(df: pd.DataFrame) -> None:
    t = df[df["tested"]]
    log.info("Tested %d of %d QC sites (>= %d systematic years)",
             len(t), len(df), MIN_YEARS)
    log.info("Mann-Kendall (BH q=%.2f): %d up, %d down, %d not significant",
             FDR_Q, (t["trend_dir"] == "up").sum(), (t["trend_dir"] == "down").sum(),
             (t["trend_dir"] == "none").sum())
    log.info("Pettitt change points: %d significant; %d within %d yr of regulation onset",
             t["pettitt_sig"].sum(), t["change_near_reg_onset"].sum(), ONSET_WINDOW)
    if "is_regulated" in t:
        for reg, g in t.groupby("is_regulated"):
            log.info("  %s: %d sites · %.0f%% up · %.0f%% down · %.0f%% change point",
                     "regulated" if reg else "unregulated", len(g),
                     100 * (g["trend_dir"] == "up").mean(),
                     100 * (g["trend_dir"] == "down").mean(),
                     100 * g["pettitt_sig"].mean())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    ffa = pd.read_parquet(args.data_dir / "ffa" / "flood_frequency.parquet",
                          columns=["site_no", "record_ok", "degenerate_fit"])
    qc = pd.Index(ffa.loc[ffa["record_ok"] & ~ffa["degenerate_fit"], "site_no"])
    peaks = pd.read_parquet(args.data_dir / "ffa" / "annual_peaks.parquet",
                            columns=["site_no", "datetime", "peak_va", "peak_cd"])
    reg = (pd.read_parquet(args.data_dir / "metadata" / "regulation.parquet",
                           columns=["site_no", "is_regulated"])
           .drop_duplicates("site_no"))

    df = screen(peaks, qc).merge(reg, on="site_no", how="left")
    report(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("Saved %d rows -> %s", len(df), args.out)


if __name__ == "__main__":
    main()
