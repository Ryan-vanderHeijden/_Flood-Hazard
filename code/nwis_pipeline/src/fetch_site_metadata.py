from __future__ import annotations

"""
Service 2: Fetch basic site attributes from USGS NWIS for each gage.

Attributes retrieved:
  station_nm    — Station name
  dec_lat_va    → latitude
  dec_long_va   → longitude
  alt_va        → elevation_ft (NAVD 88 or NGVD 29)
  drain_area_va → drainage_area_sqmi
  huc_cd        — HUC-8 watershed code
  state_cd      — FIPS state code

Output: data/metadata/site_info.parquet
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import dataretrieval.nwis as nwis

logger = logging.getLogger(__name__)

# NWIS site service has practical URL-length limits; batch to be safe
BATCH_SIZE = 100

# Parallel workers for batch requests
_META_MAX_WORKERS = 10

# NWIS → readable column names
RENAME = {
    "station_nm": "station_name",
    "dec_lat_va": "latitude",
    "dec_long_va": "longitude",
    "alt_va": "elevation_ft",
    "drain_area_va": "drainage_area_sqmi",
    "huc_cd": "huc8",
    "state_cd": "state_cd",
}


def _fetch_info_batch(batch: list[str]) -> pd.DataFrame | None:
    """Fetch basic site attributes for one batch."""
    try:
        df, _ = nwis.get_info(sites=batch)
        return df if not df.empty else None
    except Exception as exc:
        logger.error("get_info failed for batch starting %s: %s", batch[0], exc)
        return None


def _fetch_catalog_batch(batch: list[str]) -> pd.DataFrame | None:
    """Fetch series catalog (period-of-record) for one batch."""
    try:
        df, _ = nwis.get_info(sites=batch, seriesCatalogOutput=True)
        return df if not df.empty else None
    except Exception as exc:
        logger.error("get_info (catalog) failed for batch starting %s: %s", batch[0], exc)
        return None


def fetch_site_metadata(gage_ids: list[str], out_path: Path) -> pd.DataFrame:
    """
    Fetch site metadata for all gages and save to Parquet.

    Args:
        gage_ids: List of USGS site numbers (zero-padded strings).
        out_path: Directory where site_info.parquet will be written.

    Returns:
        DataFrame with readable column names indexed by site_no.
    """
    logger.info("Fetching site metadata for %d gages (workers=%d)", len(gage_ids), _META_MAX_WORKERS)

    batches = [gage_ids[i : i + BATCH_SIZE] for i in range(0, len(gage_ids), BATCH_SIZE)]
    logger.info("  Using %d batches of up to %d sites", len(batches), BATCH_SIZE)

    # --- Basic site attributes (parallelized) ---
    info_frames = []
    with ThreadPoolExecutor(max_workers=_META_MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_info_batch, batch): batch for batch in batches}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                info_frames.append(result)

    if not info_frames:
        logger.warning("No site metadata returned.")
        return pd.DataFrame()

    df = pd.concat(info_frames).reset_index()

    # Keep site_no and renamed attribute columns
    keep = ["site_no"] + [c for c in RENAME if c in df.columns]
    df = df[keep].rename(columns=RENAME)

    # --- Discharge (00060) period of record (parallelized) ---
    catalog_frames = []
    with ThreadPoolExecutor(max_workers=_META_MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_catalog_batch, batch): batch for batch in batches}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                catalog_frames.append(result)

    if catalog_frames:
        catalog = pd.concat(catalog_frames).reset_index()
        # Use the union of discharge (00060) and stage (00065) periods so that
        # sites with a longer stage record are not silently truncated.
        por = (
            catalog[catalog["parm_cd"].isin(["00060", "00065"])]
            .groupby("site_no")
            .agg(begin_date=("begin_date", "min"), end_date=("end_date", "max"))
            .reset_index()
        )
        df = df.merge(por, on="site_no", how="left")
    else:
        logger.warning("No series catalog data retrieved; period-of-record will be missing.")
        df["begin_date"] = pd.NaT
        df["end_date"] = pd.NaT

    # Coerce numeric columns
    for col in ("latitude", "longitude", "elevation_ft", "drainage_area_sqmi"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("  Retrieved metadata for %d sites", len(df))

    missing = set(gage_ids) - set(df["site_no"].astype(str))
    if missing:
        logger.warning("No metadata returned for: %s", sorted(missing))

    out_path.mkdir(parents=True, exist_ok=True)
    parquet_file = out_path / "site_info.parquet"
    df.to_parquet(parquet_file, index=False)
    logger.info("Saved %d rows → %s", len(df), parquet_file)

    return df
