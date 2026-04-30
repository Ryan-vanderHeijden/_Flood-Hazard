from __future__ import annotations

"""
Flood Frequency Analysis — Log-Pearson Type III (LP3), Bulletin 17C.

Implements the Expected Moments Algorithm (EMA) following Cohn et al. (1997)
and USGS Bulletin 17C (2019). Key features vs. a plain MLE fit:
  - Method of Moments (not MLE) parameterization
  - Left-censored observations (NWIS qualification code 6) incorporated via
    EMA: their expected contribution below the perception threshold is added
    to the moment sums rather than discarding them or treating them as real peaks
  - Historical peaks (NWIS qualification code 7) incorporated with a weighted
    historical period: years in the historical period that exceed no known
    threshold are treated as censored below the minimum historical peak

For each streamgage with at least one defined flood flow threshold, this
module:
  1. Fetches the USGS NWIS annual instantaneous peak flow record.
  2. Classifies peaks by qualification code (systematic / censored / historical).
  3. Fits an LP3 distribution via EMA.
  4. Evaluates the fitted distribution at each NWS flood flow threshold to
     produce the annual exceedance probability (AEP) and return period.

Outputs (written to out_path):
  annual_peaks.parquet    — raw NWIS peak records, all sites, long format
  flood_frequency.parquet — LP3 fit parameters + AEP/return period per
                            threshold, plus EMA metadata columns
"""

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import pearson3

import dataretrieval.nwis as nwis

logger = logging.getLogger(__name__)

_FFA_MAX_WORKERS = 6
_MIN_PEAKS       = 10   # minimum non-censored systematic peaks for record_ok
_EMA_MAX_ITER    = 50
_EMA_TOL         = 1e-6

_THRESHOLD_FLOWS = [
    "action_flow_cfs",
    "flood_flow_cfs",
    "moderate_flow_cfs",
    "major_flow_cfs",
]
_LEVELS = ["action", "flood", "moderate", "major"]


# ---------------------------------------------------------------------------
# Qualification-code parsing
# ---------------------------------------------------------------------------

def _parse_peak_cd(val) -> frozenset:
    """Parse an NWIS peak_cd field to a frozenset of normalized codes.

    Handles None, "None", single values ("6", "C"), float-strings ("6.0"),
    and comma-separated lists ("1,6", "6,C").
    """
    if val is None or val == "None" or (isinstance(val, float) and np.isnan(val)):
        return frozenset()
    codes: set = set()
    for part in str(val).replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        try:
            codes.add(int(float(part)))
        except ValueError:
            codes.add(part)
    return frozenset(codes)


def _classify_peaks(df: pd.DataFrame) -> dict:
    """Classify annual peak records for a single site.

    NWIS qualification codes used here:
      6 — discharge is less than the minimum recordable value (left-censored)
      7 — historical peak (pre-gauging or outside systematic record)

    Dropped before classification:
      1 — max daily average (not instantaneous); biases LP3 fit downward
      8 — stage only, no discharge determined; peak_va is not a flow

    Returns a dict:
      sys_peaks                : np.ndarray  — non-censored systematic flows (cfs)
      cens_peaks               : np.ndarray  — code-6 perception-threshold values (cfs)
      hist_peaks               : np.ndarray  — code-7 historical flows (cfs)
      n_sys_years              : int         — systematic record rows (censored + non-censored)
      hist_H                   : int         — historical period length in years (0 if none)
      n_censored               : int         — count of code-6 observations
      perception_threshold_cfs : float       — min code-6 value (global perception threshold)
      n_dropped                : int         — peaks dropped due to codes 1 or 8
    """
    _empty = dict(
        sys_peaks=np.array([]), cens_peaks=np.array([]),
        hist_peaks=np.array([]), n_sys_years=0, hist_H=0,
        n_censored=0, perception_threshold_cfs=np.nan, n_dropped=0,
    )
    if df.empty or "peak_va" not in df.columns:
        return _empty

    codes = df["peak_cd"].apply(_parse_peak_cd)

    # Drop daily-average (1) and stage-only (8) peaks before any further use
    drop_mask = codes.apply(lambda c: 1 in c or 8 in c)
    n_dropped = int(drop_mask.sum())
    if n_dropped:
        df    = df[~drop_mask].copy()
        codes = codes[~drop_mask]

    is_hist = codes.apply(lambda c: 7 in c)
    is_cens = codes.apply(lambda c: 6 in c) & ~is_hist

    # Systematic rows = everything that is not a historical peak
    sys_df      = df[~is_hist].copy()
    n_sys_years = len(sys_df)

    cens_mask = is_cens.loc[sys_df.index]
    cens_df   = sys_df[cens_mask]
    good_df   = sys_df[~cens_mask]

    def _positive_vals(frame: pd.DataFrame) -> np.ndarray:
        v = frame["peak_va"].dropna()
        return v[v > 0].values.astype(float)

    cens_vals = _positive_vals(cens_df)
    good_vals = _positive_vals(good_df)

    hist_df   = df[is_hist].copy()
    hist_vals = _positive_vals(hist_df)

    # Historical period length H
    # hist_start = earliest year with a code-7 peak
    # hist_end   = max(year_last_pk) across code-7 rows if available;
    #              otherwise (start of systematic record) - 1
    hist_H = 0
    if len(hist_vals) > 0:
        try:
            hist_dates = pd.to_datetime(
                df.loc[is_hist, "datetime"], errors="coerce", utc=True
            )
            sys_dates = pd.to_datetime(
                df.loc[~is_hist, "datetime"], errors="coerce", utc=True
            )
            hist_years = hist_dates.dt.year.dropna()
            hist_start = int(hist_years.min()) if len(hist_years) > 0 else None

            hist_end = None
            if "year_last_pk" in hist_df.columns:
                ylp = pd.to_numeric(hist_df["year_last_pk"], errors="coerce").dropna()
                if len(ylp) > 0:
                    hist_end = int(ylp.max())
            if hist_end is None:
                sys_years = sys_dates.dt.year.dropna()
                if len(sys_years) > 0:
                    hist_end = int(sys_years.min()) - 1

            if (hist_start is not None and hist_end is not None
                    and hist_end >= hist_start):
                hist_H = hist_end - hist_start + 1
        except Exception:
            pass

    return dict(
        sys_peaks=good_vals,
        cens_peaks=cens_vals,
        hist_peaks=hist_vals,
        n_sys_years=n_sys_years,
        hist_H=hist_H,
        n_censored=len(cens_vals),
        perception_threshold_cfs=(
            float(cens_vals.min()) if len(cens_vals) > 0 else np.nan
        ),
        n_dropped=n_dropped,
    )


# ---------------------------------------------------------------------------
# Truncated Pearson III conditional moments (core EMA building block)
# ---------------------------------------------------------------------------

def _trunc_moments(
    skew: float, mu: float, sigma: float, threshold: float, below: bool = True
) -> tuple[float, float, float]:
    """Compute E[Y], E[Y²], E[Y³] for Y ~ Pearson3(skew, mu, sigma) conditioned
    on Y < threshold (below=True) or Y > threshold (below=False).

    Uses numerical integration so it is correct for any skew value.
    Returns (nan, nan, nan) when the conditioning probability is negligible.
    """
    if below:
        prob = float(pearson3.cdf(threshold, skew, mu, sigma))
        try:
            lo = float(pearson3.ppf(1e-10, skew, mu, sigma))
        except Exception:
            lo = mu - 10.0 * sigma
        hi = threshold
    else:
        prob = float(pearson3.sf(threshold, skew, mu, sigma))
        lo = threshold
        try:
            hi = float(pearson3.ppf(1 - 1e-10, skew, mu, sigma))
        except Exception:
            hi = mu + 10.0 * sigma

    if prob < 1e-10 or not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        return np.nan, np.nan, np.nan

    try:
        def f(y: float, k: int) -> float:
            return (y ** k) * pearson3.pdf(y, skew, mu, sigma)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress IntegrationWarning on difficult tails
            e1 = quad(f, lo, hi, args=(1,), limit=200)[0] / prob
            e2 = quad(f, lo, hi, args=(2,), limit=200)[0] / prob
            e3 = quad(f, lo, hi, args=(3,), limit=200)[0] / prob

        if not (np.isfinite(e1) and np.isfinite(e2) and np.isfinite(e3)):
            return np.nan, np.nan, np.nan
        return float(e1), float(e2), float(e3)
    except Exception:
        return np.nan, np.nan, np.nan


# ---------------------------------------------------------------------------
# EMA fitting
# ---------------------------------------------------------------------------

def _fit_lp3_ema(
    sys_peaks:  np.ndarray,
    cens_peaks: np.ndarray,
    hist_peaks: np.ndarray,
    hist_H:     int,
    max_iter:   int = _EMA_MAX_ITER,
    tol:        float = _EMA_TOL,
) -> tuple[float, float, float] | None:
    """Fit LP3 via the Expected Moments Algorithm (B17C / Cohn et al. 1997).

    Parameters
    ----------
    sys_peaks  : non-censored systematic peak flows (cfs)
    cens_peaks : code-6 peak values, each used as a per-observation perception
                 threshold (cfs); the minimum is used as a single-site PT
    hist_peaks : code-7 historical peak flows (cfs)
    hist_H     : historical period length in years (0 = no historical record)

    Returns
    -------
    (skew, loc, scale) in log10-space for scipy.stats.pearson3, or None.
    AEP = pearson3.sf(log10(Q_threshold), skew, loc, scale).
    """
    if len(sys_peaks) + len(hist_peaks) < 2:
        return None

    # --- log10-transform ---
    log_sys  = np.log10(sys_peaks)  if len(sys_peaks)  > 0 else np.array([])
    log_hist = np.log10(hist_peaks) if len(hist_peaks) > 0 else np.array([])
    log_cens = np.log10(cens_peaks) if len(cens_peaks) > 0 else np.array([])

    n_s = len(log_sys)
    n_c = len(log_cens)
    n_h = len(log_hist)

    # Single perception threshold per site: minimum code-6 value (conservative)
    PT_6 = float(log_cens.min()) if n_c > 0 else None

    # Historical below-threshold years: H - n_h years with peaks below the
    # minimum historical peak (PT_H = min historical log10 flow)
    PT_H = float(log_hist.min()) if n_h > 0 else None
    H_c  = max(hist_H - n_h, 0)

    # Total effective record length
    N = n_s + n_c + (hist_H if hist_H > 0 else 0)
    if N < 2:
        return None

    # --- Initial estimate: MOM on all non-censored observations ---
    init = np.concatenate([log_sys, log_hist])
    if len(init) < 2:
        return None

    mu    = float(np.mean(init))
    sigma = float(np.std(init, ddof=1))
    if sigma < 1e-10:
        return None

    n_i  = len(init)
    skew = (
        float(n_i * np.sum((init - mu) ** 3) /
              ((n_i - 1) * (n_i - 2) * sigma ** 3))
        if n_i >= 3 else 0.0
    )

    # --- EMA iteration ---
    for _ in range(max_iter):
        mu_old, sigma_old, skew_old = mu, sigma, skew
        if sigma < 1e-10:
            break

        # Sums from uncensored observations (systematic + historical)
        S1 = float(np.sum(log_sys)) + float(np.sum(log_hist))
        S2 = float(np.sum(log_sys ** 2)) + float(np.sum(log_hist ** 2))
        S3 = float(np.sum(log_sys ** 3)) + float(np.sum(log_hist ** 3))
        W  = float(n_s + n_h)

        # Contribution from censored systematic peaks (code 6, below PT_6)
        if n_c > 0 and PT_6 is not None:
            e1, e2, e3 = _trunc_moments(skew, mu, sigma, PT_6, below=True)
            if not any(np.isnan(x) for x in (e1, e2, e3)):
                S1 += n_c * e1
                S2 += n_c * e2
                S3 += n_c * e3
                W  += n_c

        # Contribution from historical below-threshold years (below PT_H)
        if H_c > 0 and PT_H is not None:
            e1, e2, e3 = _trunc_moments(skew, mu, sigma, PT_H, below=True)
            if not any(np.isnan(x) for x in (e1, e2, e3)):
                S1 += H_c * e1
                S2 += H_c * e2
                S3 += H_c * e3
                W  += H_c

        if W < 2:
            break

        # Update moments (B17C Eq. 6-6 – 6-8)
        mu     = S1 / W
        sigma2 = max((S2 - W * mu ** 2) / (W - 1), 1e-12)
        sigma  = float(np.sqrt(sigma2))
        skew   = (
            float(W * (S3 - 3 * mu * S2 + 2 * W * mu ** 3) /
                  ((W - 1) * (W - 2) * sigma ** 3))
            if W >= 3 else skew_old
        )

        if (abs(mu - mu_old) < tol and
                abs(sigma - sigma_old) < tol and
                abs(skew - skew_old) < tol):
            break

    return float(skew), float(mu), float(sigma)


# ---------------------------------------------------------------------------
# AEP / return period from fitted parameters
# ---------------------------------------------------------------------------

def _threshold_stats(
    threshold_flow: float,
    skew: float,
    loc: float,
    scale: float,
) -> tuple[float, float]:
    """Return (AEP, return_period_yr) for a single threshold flow."""
    if not np.isfinite(threshold_flow) or threshold_flow <= 0:
        return np.nan, np.nan
    aep = float(pearson3.sf(np.log10(threshold_flow), skew, loc, scale))
    if aep <= 0:
        return aep, np.nan   # store NaN instead of inf
    return aep, 1.0 / aep


# ---------------------------------------------------------------------------
# Peak fetching (unchanged)
# ---------------------------------------------------------------------------

def _fetch_peaks_site(site_no: str) -> pd.DataFrame | None:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format")
            df, _ = nwis.get_discharge_peaks(sites=site_no)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        if "site_no" not in df.columns:
            df.insert(0, "site_no", site_no)
        return df
    except Exception as exc:
        logger.warning("get_discharge_peaks failed for %s: %s", site_no, exc)
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_flood_frequency(
    flood_stages_path: Path,
    out_path: Path,
    min_peaks: int = _MIN_PEAKS,
) -> pd.DataFrame:
    """Fetch annual peak flows and fit LP3 (EMA) for all sites with defined
    flood flow thresholds.

    Parameters
    ----------
    flood_stages_path : Path
        Directory containing flood_stages.parquet.
    out_path : Path
        Directory where annual_peaks.parquet and flood_frequency.parquet
        are written.
    min_peaks : int
        Minimum non-censored systematic peaks for record_ok = True.

    Returns
    -------
    DataFrame with LP3 fit results and threshold AEPs / return periods.

    New columns vs. previous version
    ---------------------------------
    n_censored               : int   — code-6 peaks excluded from direct fitting
    n_hist                   : int   — code-7 historical peaks
    hist_H                   : int   — historical period length (years)
    perception_threshold_cfs : float — minimum code-6 peak_va (site PT)
    high_censoring           : bool  — >25 % of effective record is censored
    """
    fs_file = flood_stages_path / "flood_stages.parquet"
    if not fs_file.exists():
        raise FileNotFoundError(fs_file)

    fs = pd.read_parquet(fs_file, columns=["site_no"] + _THRESHOLD_FLOWS)
    has_any = fs[_THRESHOLD_FLOWS].notna().any(axis=1)
    fs = fs[has_any].copy()
    site_ids = fs["site_no"].tolist()
    logger.info(
        "%d sites with at least one flow threshold — fetching annual peaks",
        len(site_ids),
    )

    # Fetch peaks in parallel
    peak_frames: list[pd.DataFrame] = []
    n_empty, n_total, n_done = 0, len(site_ids), 0
    with ThreadPoolExecutor(max_workers=_FFA_MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_peaks_site, s): s for s in site_ids}
        for future in as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                peak_frames.append(result)
            else:
                n_empty += 1
            n_done += 1
            if n_done % 100 == 0 or n_done == n_total:
                logger.info("  fetched %d/%d sites", n_done, n_total)

    logger.info(
        "Received peak data for %d/%d sites (%d returned nothing)",
        len(peak_frames), len(site_ids), n_empty,
    )

    if peak_frames:
        all_peaks = pd.concat(peak_frames, ignore_index=True)
    else:
        all_peaks = pd.DataFrame(columns=["site_no"])

    # Normalize mixed-type object columns so pyarrow can serialize them.
    for col in all_peaks.select_dtypes(include="object").columns:
        all_peaks[col] = all_peaks[col].where(
            all_peaks[col].isna(), all_peaks[col].astype(str)
        )

    out_path.mkdir(parents=True, exist_ok=True)
    peaks_file = out_path / "annual_peaks.parquet"
    all_peaks.to_parquet(peaks_file, index=False)
    logger.info("Saved annual peaks → %s  (%d rows)", peaks_file, len(all_peaks))

    # Per-site full DataFrame lookup (needed for code classification)
    all_peaks_by_site: dict[str, pd.DataFrame] = {
        site: grp.reset_index(drop=True)
        for site, grp in all_peaks.groupby("site_no")
    }

    # Fit LP3 (EMA) and compute threshold return periods per site
    records = []
    for _, row in fs.iterrows():
        site = row["site_no"]

        cl = (
            _classify_peaks(all_peaks_by_site[site])
            if site in all_peaks_by_site
            else dict(
                sys_peaks=np.array([]), cens_peaks=np.array([]),
                hist_peaks=np.array([]), n_sys_years=0, hist_H=0,
                n_censored=0, perception_threshold_cfs=np.nan, n_dropped=0,
            )
        )

        n_sys      = len(cl["sys_peaks"])
        n_censored = cl["n_censored"]
        n_hist     = len(cl["hist_peaks"])
        hist_H     = cl["hist_H"]
        n_dropped  = cl["n_dropped"]
        n_eff      = n_sys + n_censored + (hist_H if hist_H > 0 else n_hist)

        base: dict = {
            "site_no":  site,
            "n_peaks":  n_sys,       # non-censored systematic peaks
            "n_censored": n_censored,
            "n_hist":   n_hist,
            "hist_H":   hist_H,
            "n_dropped": n_dropped,
            "perception_threshold_cfs": cl["perception_threshold_cfs"],
            "high_censoring": (
                n_censored > 0 and n_censored / max(n_eff, 1) > 0.25
            ),
            "record_ok": n_sys >= min_peaks,
            "lp3_skew":  np.nan,
            "lp3_loc":   np.nan,
            "lp3_scale": np.nan,
        }
        for lvl in _LEVELS:
            base[f"{lvl}_aep"]              = np.nan
            base[f"{lvl}_return_period_yr"] = np.nan

        if n_sys + n_hist >= 2:
            fit = _fit_lp3_ema(
                cl["sys_peaks"],
                cl["cens_peaks"],
                cl["hist_peaks"],
                hist_H,
            )
            if fit is not None:
                skew, loc, scale = fit
                base.update({"lp3_skew": skew, "lp3_loc": loc, "lp3_scale": scale})
                for lvl, col in zip(_LEVELS, _THRESHOLD_FLOWS):
                    aep, rp = _threshold_stats(row[col], skew, loc, scale)
                    base[f"{lvl}_aep"]              = aep
                    base[f"{lvl}_return_period_yr"] = rp

        records.append(base)

    result = pd.DataFrame(records)

    n_short     = int((result["n_peaks"] < min_peaks).sum())
    n_high_cens = int(result["high_censoring"].sum())
    n_dropped   = int(result["n_dropped"].sum())
    logger.info(
        "EMA fit complete: %d sites, %d short record (<%d non-censored peaks), "
        "%d high-censoring (>25%% code-6), %d peaks dropped (codes 1/8)",
        len(result), n_short, min_peaks, n_high_cens, n_dropped,
    )

    ffa_file = out_path / "flood_frequency.parquet"
    result.to_parquet(ffa_file, index=False)
    logger.info("Saved flood frequency results → %s", ffa_file)

    return result
