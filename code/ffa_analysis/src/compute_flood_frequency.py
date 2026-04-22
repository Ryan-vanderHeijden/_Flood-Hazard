from __future__ import annotations

"""
Flood Frequency Analysis — Log-Pearson Type III (LP3).

For each streamgage with at least one defined flood flow threshold, this
module:
  1. Fetches the USGS NWIS annual instantaneous peak flow record.
  2. Fits a Log-Pearson Type III distribution (scipy.stats.pearson3 on
     log10-transformed peaks, MLE).
  3. Evaluates the fitted distribution at each NWS flood flow threshold to
     produce the annual exceedance probability (AEP) and return period.

Outputs (written to out_path):
  annual_peaks.parquet  — raw NWIS peak records, all sites, long format
  flood_frequency.parquet — LP3 fit parameters + AEP/return period per threshold
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearson3

import dataretrieval.nwis as nwis

logger = logging.getLogger(__name__)

_FFA_MAX_WORKERS = 16
_MIN_PEAKS = 10

_THRESHOLD_FLOWS = [
    "action_flow_cfs",
    "flood_flow_cfs",
    "moderate_flow_cfs",
    "major_flow_cfs",
]
_LEVELS = ["action", "flood", "moderate", "major"]


def _fetch_peaks_site(site_no: str) -> pd.DataFrame | None:
    try:
        df, _ = nwis.get_peaks(sites=site_no)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df.insert(0, "site_no", site_no)
        return df
    except Exception as exc:
        logger.warning("get_peaks failed for %s: %s", site_no, exc)
        return None


def _fit_lp3(peaks: np.ndarray) -> tuple[float, float, float] | None:
    """Fit LP3 to log10-transformed peaks. Returns (skew, loc, scale) or None."""
    log_peaks = np.log10(peaks)
    try:
        skew, loc, scale = pearson3.fit(log_peaks)
        return float(skew), float(loc), float(scale)
    except Exception as exc:
        logger.debug("pearson3.fit failed: %s", exc)
        return None


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
        return aep, np.inf
    return aep, 1.0 / aep


def compute_flood_frequency(
    flood_stages_path: Path,
    out_path: Path,
    min_peaks: int = _MIN_PEAKS,
) -> pd.DataFrame:
    """
    Fetch annual peak flows and fit LP3 distributions for all sites with
    defined flood flow thresholds.

    Parameters
    ----------
    flood_stages_path : Path
        Directory containing flood_stages.parquet.
    out_path : Path
        Directory where annual_peaks.parquet and flood_frequency.parquet
        are written.
    min_peaks : int
        Minimum number of annual peaks required; sites below this threshold
        have record_ok = False (fit still attempted if n_peaks >= 2).

    Returns
    -------
    DataFrame with LP3 fit results and threshold AEPs/return periods.
    """
    fs_file = flood_stages_path / "flood_stages.parquet"
    if not fs_file.exists():
        raise FileNotFoundError(fs_file)

    fs = pd.read_parquet(fs_file, columns=["site_no"] + _THRESHOLD_FLOWS)
    has_any = fs[_THRESHOLD_FLOWS].notna().any(axis=1)
    fs = fs[has_any].copy()
    site_ids = fs["site_no"].tolist()
    logger.info("%d sites with at least one flow threshold — fetching annual peaks", len(site_ids))

    # Fetch peaks in parallel
    peak_frames: list[pd.DataFrame] = []
    n_empty = 0
    with ThreadPoolExecutor(max_workers=_FFA_MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_peaks_site, s): s for s in site_ids}
        for future in as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                peak_frames.append(result)
            else:
                n_empty += 1

    logger.info(
        "Received peak data for %d/%d sites (%d returned nothing)",
        len(peak_frames), len(site_ids), n_empty,
    )

    if peak_frames:
        all_peaks = pd.concat(peak_frames, ignore_index=True)
    else:
        all_peaks = pd.DataFrame(columns=["site_no"])

    out_path.mkdir(parents=True, exist_ok=True)
    peaks_file = out_path / "annual_peaks.parquet"
    all_peaks.to_parquet(peaks_file, index=False)
    logger.info("Saved annual peaks → %s  (%d rows)", peaks_file, len(all_peaks))

    # Build per-site peak lookup
    if not all_peaks.empty and "peak_va" in all_peaks.columns:
        site_peaks = (
            all_peaks.dropna(subset=["peak_va"])
            .query("peak_va > 0")
            .groupby("site_no")["peak_va"]
            .apply(np.array)
            .to_dict()
        )
    else:
        site_peaks = {}

    # Fit LP3 and compute threshold return periods per site
    records = []
    for _, row in fs.iterrows():
        site = row["site_no"]
        peaks = site_peaks.get(site, np.array([]))
        n = len(peaks)
        ok = n >= min_peaks

        base = {"site_no": site, "n_peaks": n, "record_ok": ok,
                "lp3_skew": np.nan, "lp3_loc": np.nan, "lp3_scale": np.nan}
        for lvl in _LEVELS:
            base[f"{lvl}_aep"] = np.nan
            base[f"{lvl}_return_period_yr"] = np.nan

        if n >= 2:
            fit = _fit_lp3(peaks)
            if fit is not None:
                skew, loc, scale = fit
                base.update({"lp3_skew": skew, "lp3_loc": loc, "lp3_scale": scale})
                for lvl, col in zip(_LEVELS, _THRESHOLD_FLOWS):
                    aep, rp = _threshold_stats(row[col], skew, loc, scale)
                    base[f"{lvl}_aep"] = aep
                    base[f"{lvl}_return_period_yr"] = rp

        records.append(base)

    result = pd.DataFrame(records)
    n_short = int((result["n_peaks"] < min_peaks).sum())
    logger.info(
        "LP3 fit complete: %d sites total, %d flagged short record (< %d peaks)",
        len(result), n_short, min_peaks,
    )

    ffa_file = out_path / "flood_frequency.parquet"
    result.to_parquet(ffa_file, index=False)
    logger.info("Saved flood frequency results → %s", ffa_file)

    return result
