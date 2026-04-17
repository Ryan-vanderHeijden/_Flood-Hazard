from __future__ import annotations

"""
Service 6: Compute the non-exceedance percentile of each NWS flood flow
threshold within the USGS observed daily discharge distribution.

For each streamgage, the historical discharge record is treated as an
empirical CDF.  Each threshold flow (action / flood / moderate / major) is
located on that CDF: the resulting value is the fraction of valid observed
days on which discharge was at or below the threshold.

Algorithm:
  1. Load streamflow.parquet — keep only site_no + discharge_cfs.
  2. Drop NaN and negative values; keep zeros (valid low-flow observations).
  3. Require >= _MIN_VALID_DAYS non-null days per site; flag shorter records.
  4. Sort each site's flow array once, then use binary search for all four
     thresholds (O(n log n) sort + O(log n) per threshold).
  5. Merge with flood_stages.parquet on site_no and write output.

Output: data/metadata/flood_threshold_percentiles.parquet
Columns: site_no, n_valid_days, record_ok,
         action_flow_pct, flood_flow_pct, moderate_flow_pct, major_flow_pct
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_VALID_DAYS = 3_650  # ~10 years

_THRESHOLD_COLS = [
    "action_flow_cfs",
    "flood_flow_cfs",
    "moderate_flow_cfs",
    "major_flow_cfs",
]
_PERCENTILE_COLS = [c.replace("_cfs", "_pct") for c in _THRESHOLD_COLS]


def _pct_of_score(sorted_arr: np.ndarray, score: float) -> float:
    """Non-exceedance percentile (0–100): fraction of values <= score."""
    count = np.searchsorted(sorted_arr, score, side="right")
    return float(count) / len(sorted_arr) * 100.0


def compute_flood_percentiles(
    streamflow_path: Path,
    flood_stages_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    """
    Compute non-exceedance percentiles for flood flow thresholds.

    Parameters
    ----------
    streamflow_path : Path
        Directory containing streamflow.parquet.
    flood_stages_path : Path
        Directory containing flood_stages.parquet.
    out_path : Path
        Directory where flood_threshold_percentiles.parquet is written.

    Returns
    -------
    DataFrame with columns: site_no, n_valid_days, record_ok,
        action_flow_pct, flood_flow_pct, moderate_flow_pct, major_flow_pct.
    """
    sf_file = streamflow_path / "streamflow.parquet"
    fs_file = flood_stages_path / "flood_stages.parquet"

    if not sf_file.exists():
        raise FileNotFoundError(sf_file)
    if not fs_file.exists():
        raise FileNotFoundError(fs_file)

    logger.info("Loading discharge from %s", sf_file)
    sf = pd.read_parquet(sf_file, columns=["site_no", "discharge_cfs"])

    logger.info("Loading flood stages from %s", fs_file)
    fs = pd.read_parquet(fs_file, columns=["site_no"] + _THRESHOLD_COLS)

    # Keep only gages that have at least one defined flow threshold
    has_any = fs[_THRESHOLD_COLS].notna().any(axis=1)
    fs = fs[has_any].copy()
    logger.info("%d gages have at least one flow threshold", len(fs))

    # Drop invalid discharge rows; keep zeros
    sf = sf[sf["discharge_cfs"].notna() & (sf["discharge_cfs"] >= 0.0)]

    # Build per-site sorted arrays
    grouped = sf.groupby("site_no")["discharge_cfs"].apply(
        lambda s: np.sort(s.to_numpy(dtype=np.float64))
    )

    records = []
    n_short = 0
    for _, row in fs.iterrows():
        site = row["site_no"]
        arr = grouped.get(site)

        if arr is None or len(arr) == 0:
            records.append(
                {"site_no": site, "n_valid_days": 0, "record_ok": False,
                 **{c: np.nan for c in _PERCENTILE_COLS}}
            )
            continue

        n = len(arr)
        ok = n >= _MIN_VALID_DAYS
        if not ok:
            n_short += 1

        pcts = {}
        for t_col, p_col in zip(_THRESHOLD_COLS, _PERCENTILE_COLS):
            thresh = row[t_col]
            pcts[p_col] = _pct_of_score(arr, thresh) if pd.notna(thresh) else np.nan

        records.append({"site_no": site, "n_valid_days": n, "record_ok": ok, **pcts})

    result = pd.DataFrame(records)

    logger.info(
        "Computed percentiles for %d gages (%d flagged short record < %d days)",
        len(result), n_short, _MIN_VALID_DAYS,
    )

    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "flood_threshold_percentiles.parquet"
    result.to_parquet(out_file, index=False)
    logger.info("Saved → %s", out_file)

    return result
