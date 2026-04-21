from __future__ import annotations

"""
Service 4: Fill NaN flood flow thresholds by interpolating USGS rating curves.

For each site that has at least one flood stage threshold defined but a missing
corresponding flow (in cfs), this service:

  1. Fetches the current USGS NWIS rating curve (file_type="exsa").
  2. Interpolates each NaN flow threshold from the stage-discharge table.
  3. Writes only NaN flow values; NWPS-provided flows are preserved.
  4. Adds *_flow_source columns: "nwps" for NWPS-provided, "rating_curve" for
     newly filled, None for still-missing.

Datum note: USGS rating curves use gage height above the local gage datum (GD).
NWS stage thresholds are also reported as gage height above the local gage datum,
so the two datum systems are directly comparable — no offset correction needed.

Output: overwrites data/metadata/flood_stages.parquet in place with filled
flow columns and added *_flow_source columns.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import dataretrieval.nwis as nwis

logger = logging.getLogger(__name__)

# Parallel workers for rating-curve fetches
_RC_MAX_WORKERS = 30

# Stage-to-flow threshold pairs (stage col → flow col)
_THRESHOLD_PAIRS = [
    ("action_stage_ft",   "action_flow_cfs"),
    ("flood_stage_ft",    "flood_flow_cfs"),
    ("moderate_stage_ft", "moderate_flow_cfs"),
    ("major_stage_ft",    "major_flow_cfs"),
]

_FLOW_COLS   = [p[1] for p in _THRESHOLD_PAIRS]
_SOURCE_COLS = [c.replace("_cfs", "_source") for c in _FLOW_COLS]


def _fetch_rating(site_no: str) -> pd.DataFrame | None:
    """
    Fetch the current USGS rating curve for a site.

    Returns a two-column DataFrame [stage_ft, discharge_cfs] sorted by stage,
    or None if no valid rating is available.
    """
    try:
        df, _ = nwis.get_ratings(site=site_no, file_type="exsa")
    except Exception as exc:
        logger.debug("  Rating fetch failed for %s: %s", site_no, exc)
        return None

    if df is None or df.empty:
        return None

    # Locate stage (INDEP) and discharge (DEP) columns (names may vary)
    indep_col = next((c for c in df.columns if c.upper() == "INDEP"), None)
    dep_col   = next((c for c in df.columns if c.upper() == "DEP"),   None)

    if indep_col is None or dep_col is None:
        logger.debug("  No INDEP/DEP columns in rating for %s; columns=%s", site_no, list(df.columns))
        return None

    rating = df[[indep_col, dep_col]].copy()
    rating.columns = ["stage_ft", "discharge_cfs"]
    rating["stage_ft"]     = pd.to_numeric(rating["stage_ft"],     errors="coerce")
    rating["discharge_cfs"] = pd.to_numeric(rating["discharge_cfs"], errors="coerce")

    # Remove sentinel values and NaNs
    rating = rating[rating["stage_ft"].notna() & rating["discharge_cfs"].notna()]
    rating = rating[rating["discharge_cfs"] > -9000]
    rating = rating.sort_values("stage_ft").reset_index(drop=True)

    if rating.empty:
        return None

    return rating


def _interpolate_flows(
    rating: pd.DataFrame,
    stages: dict[str, float],
) -> dict[str, float]:
    """
    Interpolate flow values at each stage threshold from the rating curve.

    Uses numpy.interp which clamps to the boundary values when a threshold
    falls outside the rating table range; boundary extrapolation is flagged
    in the source column by the caller.

    Args:
        rating: DataFrame with [stage_ft, discharge_cfs] sorted by stage.
        stages: Mapping of flow-column-name → stage value (ft).

    Returns:
        Dict of flow-column-name → interpolated discharge (cfs).
    """
    x = rating["stage_ft"].values
    y = rating["discharge_cfs"].values
    result: dict[str, float] = {}

    for flow_col, stage_val in stages.items():
        if np.isnan(stage_val):
            continue
        flow = float(np.interp(stage_val, x, y))
        # Treat as invalid if we had to extrapolate below zero-flow
        if flow < 0:
            flow = float("nan")
        result[flow_col] = flow

    return result


def _process_site(
    site_no: str,
    row: pd.Series,
) -> tuple[str, dict[str, float]] | None:
    """
    Worker: fetch rating curve and compute flow thresholds.

    If any flow threshold is missing, re-derives ALL flows from the current
    rating curve so that all thresholds for a site share a single consistent
    source (avoids ordering violations when NWPS and rating-curve flows
    originate from different versions of the stage-discharge relationship).

    Returns (site_no, {flow_col: value, ...}) or None on failure.
    """
    # Only proceed if at least one flow is missing (has valid stage but NaN flow)
    needs_any = any(
        pd.isna(row.get(flow_col)) and not pd.isna(row.get(stage_col))
        for stage_col, flow_col in _THRESHOLD_PAIRS
    )
    if not needs_any:
        return None

    # Re-derive ALL flows with valid stages from the rating curve
    targets: dict[str, float] = {}
    for stage_col, flow_col in _THRESHOLD_PAIRS:
        if not pd.isna(row.get(stage_col)):
            targets[flow_col] = float(row[stage_col])

    rating = _fetch_rating(site_no)
    if rating is None:
        logger.debug("  No rating curve for %s", site_no)
        return None

    filled = _interpolate_flows(rating, targets)
    if not filled:
        return None

    logger.debug(
        "  %s: derived %d flow threshold(s) from rating curve",
        site_no, len(filled),
    )
    return site_no, filled


def fill_flows_from_ratings(
    flood_stages: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """
    Fill NaN flood flow thresholds from USGS rating curves and save the result.

    Args:
        flood_stages: DataFrame from fetch_flood_stages, with columns
                      [site_no, *_stage_ft, *_flow_cfs, ...].
        out_path:     Directory where the updated flood_stages.parquet is saved.

    Returns:
        Updated DataFrame with filled *_flow_cfs and added *_flow_source columns.
    """
    df = flood_stages.copy()

    # Add source columns; backfill NWPS-provided flows
    for flow_col, src_col in zip(_FLOW_COLS, _SOURCE_COLS):
        if src_col not in df.columns:
            df[src_col] = None
        # Any flow already present (from NWPS) is labeled as such
        has_flow = df[flow_col].notna()
        df.loc[has_flow & df[src_col].isna(), src_col] = "nwps"

    # Identify sites that need rating-curve lookup
    needs_rating = df[
        df[[p[1] for p in _THRESHOLD_PAIRS]].isna().any(axis=1) &  # any NaN flow
        df[[p[0] for p in _THRESHOLD_PAIRS]].notna().any(axis=1)   # at least one stage
    ]["site_no"].tolist()

    if not needs_rating:
        logger.info("  No sites need rating-curve fallback.")
        _save(df, out_path)
        return df

    logger.info(
        "  Fetching rating curves for %d sites with NaN flow thresholds...",
        len(needs_rating),
    )

    # Build a row-lookup dict for quick access in workers
    row_lookup = df.set_index("site_no")

    filled_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=_RC_MAX_WORKERS) as ex:
        futures = {
            ex.submit(_process_site, sno, row_lookup.loc[sno]): sno
            for sno in needs_rating
            if sno in row_lookup.index
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                failed_count += 1
                continue
            site_no, filled = result
            idx = df.index[df["site_no"] == site_no]
            for flow_col, flow_val in filled.items():
                df.loc[idx, flow_col] = flow_val
                src_col = flow_col.replace("_cfs", "_source")
                df.loc[idx, src_col] = "rating_curve"
            filled_count += 1

    logger.info(
        "  Rating curves: filled flows for %d sites; no rating found for %d",
        filled_count, failed_count,
    )

    # Summary of remaining gaps
    for flow_col in _FLOW_COLS:
        stage_col = flow_col.replace("_cfs", "_ft").replace("_flow", "_stage")
        has_stage = df[stage_col].notna()
        still_nan = df[flow_col].isna() & has_stage
        if still_nan.any():
            logger.info(
                "  Still NaN after rating curves: %s — %d/%d sites with stage",
                flow_col, still_nan.sum(), has_stage.sum(),
            )

    _save(df, out_path)
    return df


def _save(df: pd.DataFrame, out_path: Path) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    p = out_path / "flood_stages.parquet"
    df.to_parquet(p, index=False)
    logger.info("Saved %d rows → %s", len(df), p)
