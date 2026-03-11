"""
Service 3: Fetch NWS flood stage thresholds for each gage.

Thresholds retrieved (where available):
  action_stage    — Stage at which NWS issues an action alert (ft)
  flood_stage     — Stage at which minor flooding begins (ft)
  moderate_stage  — Stage at which moderate flooding begins (ft)
  major_stage     — Stage at which major flooding begins (ft)
  action_flow     — Discharge at action stage (cfs)
  flood_flow      — Discharge at minor flood stage (cfs)
  moderate_flow   — Discharge at moderate flood stage (cfs)
  major_flow      — Discharge at major flood stage (cfs)
  *_impact        — NWS impact statement for each flood category

Data source: NWS National Water Prediction Service (NWPS) API
  https://api.water.noaa.gov/nwps/v1/gauges

Strategy:
  1. Fetch USGS→NWS LID crosswalk from NOAA HADS
     (https://hads.ncep.noaa.gov/USGS/ALL_USGS-HADS_SITES.txt).
  2. Direct lookup: site_no → LID(s). No spatial matching needed.
  3. Parallel-fetch individual NWPS records for matched LIDs.
  4. Verify the NWPS usgsId field matches the USGS site number
     (handles the ~37 sites that map to multiple LIDs).
  5. Extract flood category stage/flow values and impact statements.

Note: Only sites with observed stage data need flood stage thresholds;
  the pipeline passes only those sites (has_stage_data == True in
  data_coverage.parquet).

Outputs:
  data/metadata/flood_stages.parquet  — stage/flow thresholds + impact statements
  data/metadata/gauge_map.parquet     — USGS site_no ↔ NWS LID ↔ NWM reachId
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

NWPS_BASE = "https://api.water.noaa.gov/nwps/v1/gauges"
HADS_URL  = "https://hads.ncep.noaa.gov/USGS/ALL_USGS-HADS_SITES.txt"

# Concurrent workers for individual NWPS gauge record fetches.
MAX_WORKERS = 50


def _fetch_hads_crosswalk() -> pd.DataFrame:
    """
    Fetch the NOAA HADS crosswalk of USGS site numbers to NWS LIDs.

    Returns a DataFrame with columns [site_no, lid].  Sites with multiple
    LIDs appear as multiple rows.
    """
    resp = requests.get(HADS_URL, timeout=30)
    resp.raise_for_status()

    rows = []
    for line in resp.text.splitlines()[4:]:   # skip 4-line header
        if len(line) < 20 or line.startswith("---"):
            continue
        lid  = line[0:5].strip()
        usgs = line[6:21].strip()
        if lid and usgs:
            rows.append({"site_no": usgs, "lid": lid})

    return pd.DataFrame(rows)


def _fetch_nwps_gauge(lid: str) -> dict | None:
    """Fetch a single NWPS gauge record; return None on any failure."""
    try:
        resp = requests.get(f"{NWPS_BASE}/{lid}", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _extract_stages(record: dict) -> dict:
    """Extract flood category stage/flow values and impact statements from a NWPS gauge record."""
    cats = record.get("flood", {}).get("categories", {})

    def _val(key: str, field: str) -> float:
        v = cats.get(key, {}).get(field)
        if v is None:
            return float("nan")
        try:
            f = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return float("nan") if f <= -9000 else f

    # Build stage→impact statement lookup from the impacts array.
    stage_to_impact: dict[float, str] = {}
    for item in record.get("flood", {}).get("impacts", []):
        s = item.get("stage")
        stmt = item.get("statement", "").strip()
        if s is not None and stmt:
            stage_to_impact[float(s)] = stmt

    def _impact(key: str) -> str | None:
        stage_val = _val(key, "stage")
        if np.isnan(stage_val):
            return None
        return stage_to_impact.get(stage_val)

    return {
        "action_stage_ft":   _val("action",   "stage"),
        "flood_stage_ft":    _val("minor",    "stage"),
        "moderate_stage_ft": _val("moderate", "stage"),
        "major_stage_ft":    _val("major",    "stage"),
        "action_flow_cfs":   _val("action",   "flow"),
        "flood_flow_cfs":    _val("minor",    "flow"),
        "moderate_flow_cfs": _val("moderate", "flow"),
        "major_flow_cfs":    _val("major",    "flow"),
        "action_impact":     _impact("action"),
        "flood_impact":      _impact("minor"),
        "moderate_impact":   _impact("moderate"),
        "major_impact":      _impact("major"),
    }


def fetch_flood_stages(
    gage_ids: list[str],
    out_path: Path,
) -> pd.DataFrame:
    """
    Fetch NWS flood stage thresholds and save to Parquet.

    Args:
        gage_ids:  USGS site numbers to process (should be sites with
                   observed stage data).
        out_path:  Directory where flood_stages.parquet will be written.

    Returns:
        DataFrame with columns [site_no, action_stage_ft, flood_stage_ft,
        moderate_stage_ft, major_stage_ft, action_flow_cfs, flood_flow_cfs,
        moderate_flow_cfs, major_flow_cfs, action_impact, flood_impact,
        moderate_impact, major_impact].  Rows are present for every entry
        in gage_ids; missing values are NaN / None.

        Also writes gauge_map.parquet [site_no, lid, reach_id] to out_path.
    """
    stage_cols = [
        "action_stage_ft", "flood_stage_ft", "moderate_stage_ft", "major_stage_ft",
        "action_flow_cfs",  "flood_flow_cfs",  "moderate_flow_cfs",  "major_flow_cfs",
        "action_impact",    "flood_impact",    "moderate_impact",    "major_impact",
    ]
    empty = pd.DataFrame(columns=["site_no"] + stage_cols)

    if not gage_ids:
        return empty

    logger.info("Fetching NWPS flood stages for %d gages", len(gage_ids))

    # ------------------------------------------------------------------
    # Step 1: Fetch USGS → NWS LID crosswalk from HADS
    # ------------------------------------------------------------------
    try:
        crosswalk = _fetch_hads_crosswalk()
    except requests.RequestException as exc:
        logger.error("Failed to fetch HADS crosswalk: %s", exc)
        return empty

    logger.info("  HADS crosswalk: %d entries", len(crosswalk))

    # ------------------------------------------------------------------
    # Step 2: Direct site_no → LID lookup
    # ------------------------------------------------------------------
    targets = crosswalk[crosswalk["site_no"].isin(gage_ids)].copy()

    n_matched  = targets["site_no"].nunique()
    n_no_match = len(set(gage_ids)) - n_matched
    logger.info(
        "  Matched %d/%d gages to NWS LIDs via HADS crosswalk",
        n_matched, len(set(gage_ids)),
    )
    if n_no_match:
        logger.info(
            "  %d gages not in HADS crosswalk (no NWS forecast point)",
            n_no_match,
        )

    if targets.empty:
        logger.warning("No LID matches found in HADS crosswalk.")
        return empty

    # ------------------------------------------------------------------
    # Step 3: Parallel-fetch individual NWPS records
    # ------------------------------------------------------------------
    lids = targets["lid"].unique().tolist()
    logger.info("  Fetching individual records for %d NWPS gauges...", len(lids))

    lid_records: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_nwps_gauge, lid): lid for lid in lids}
        for fut in as_completed(futures):
            lid = futures[fut]
            record = fut.result()
            if record:
                lid_records[lid] = record

    logger.info("  Retrieved %d/%d NWPS records", len(lid_records), len(lids))

    # ------------------------------------------------------------------
    # Step 4: Verify usgsId and extract stages
    # For sites with multiple LIDs, use the first that passes verification.
    # ------------------------------------------------------------------
    rows = []
    gauge_map_rows = []
    seen_sites: set[str] = set()

    for _, row in targets.iterrows():
        site_no = row["site_no"]
        if site_no in seen_sites:
            continue  # already resolved via an earlier LID for this site

        record = lid_records.get(row["lid"])
        if record is None:
            continue

        usgs_id = record.get("usgsId", "")
        if not usgs_id:
            logger.debug(
                "  LID %s has no usgsId (candidate for %s); skipping",
                row["lid"], site_no,
            )
            continue

        if usgs_id.zfill(8) != str(site_no).zfill(8):
            logger.debug(
                "  LID %s: usgsId %s != site_no %s; skipping",
                row["lid"], usgs_id, site_no,
            )
            continue

        stages = _extract_stages(record)
        rows.append({"site_no": site_no, **stages})

        reach_id = record.get("reachId", "")
        if isinstance(reach_id, str):
            reach_id = reach_id.strip() or None
        gauge_map_rows.append({
            "site_no":  site_no,
            "lid":      row["lid"],
            "reach_id": reach_id,
        })

        seen_sites.add(site_no)

    # ------------------------------------------------------------------
    # Step 5: Build output — all gage_ids present, unmatched → NaN
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows) if rows else empty.copy()
    df = pd.DataFrame({"site_no": gage_ids}).merge(df, on="site_no", how="left")
    for col in stage_cols:
        if col not in df.columns:
            df[col] = float("nan")

    n_flood  = df["flood_stage_ft"].notna().sum()
    n_action = df["action_stage_ft"].notna().sum()
    logger.info(
        "  Flood stage defined: %d/%d | Action stage defined: %d/%d",
        n_flood, len(df), n_action, len(df),
    )

    out_path.mkdir(parents=True, exist_ok=True)

    parquet_file = out_path / "flood_stages.parquet"
    df.to_parquet(parquet_file, index=False)
    logger.info("Saved %d rows → %s", len(df), parquet_file)

    gauge_map_df = (
        pd.DataFrame(gauge_map_rows)
        if gauge_map_rows
        else pd.DataFrame(columns=["site_no", "lid", "reach_id"])
    )
    gauge_map_file = out_path / "gauge_map.parquet"
    gauge_map_df.to_parquet(gauge_map_file, index=False)
    logger.info("Saved %d rows → %s", len(gauge_map_df), gauge_map_file)

    return df
