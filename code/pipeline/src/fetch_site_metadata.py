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
from pathlib import Path

import pandas as pd
import dataretrieval.nwis as nwis

logger = logging.getLogger(__name__)

# NWIS site service has practical URL-length limits; batch to be safe
BATCH_SIZE = 100

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


def fetch_site_metadata(gage_ids: list[str], out_path: Path) -> pd.DataFrame:
    """
    Fetch site metadata for all gages and save to Parquet.

    Args:
        gage_ids: List of USGS site numbers (zero-padded strings).
        out_path: Directory where site_info.parquet will be written.

    Returns:
        DataFrame with readable column names indexed by site_no.
    """
    logger.info("Fetching site metadata for %d gages", len(gage_ids))

    batches = [gage_ids[i : i + BATCH_SIZE] for i in range(0, len(gage_ids), BATCH_SIZE)]
    logger.info("  Using %d batches of up to %d sites", len(batches), BATCH_SIZE)

    # --- Basic site attributes ---
    info_frames = []
    for batch in batches:
        try:
            batch_df, _ = nwis.get_info(sites=batch)
            if not batch_df.empty:
                info_frames.append(batch_df)
        except Exception as exc:
            logger.error("get_info failed for batch starting %s: %s", batch[0], exc)

    if not info_frames:
        logger.warning("No site metadata returned.")
        return pd.DataFrame()

    df = pd.concat(info_frames).reset_index()

    # Keep site_no and renamed attribute columns
    keep = ["site_no"] + [c for c in RENAME if c in df.columns]
    df = df[keep].rename(columns=RENAME)

    # --- Discharge (00060) period of record ---
    catalog_frames = []
    for batch in batches:
        try:
            cat_batch, _ = nwis.get_info(sites=batch, seriesCatalogOutput=True)
            if not cat_batch.empty:
                catalog_frames.append(cat_batch)
        except Exception as exc:
            logger.error("get_info (catalog) failed for batch starting %s: %s", batch[0], exc)

    if catalog_frames:
        catalog = pd.concat(catalog_frames).reset_index()
        discharge_por = (
            catalog[catalog["parm_cd"] == "00060"]
            .groupby("site_no")
            .agg(begin_date=("begin_date", "min"), end_date=("end_date", "max"))
            .reset_index()
        )
        df = df.merge(discharge_por, on="site_no", how="left")
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
