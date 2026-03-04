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

    df, _ = nwis.get_info(sites=gage_ids)

    if df.empty:
        logger.warning("No site metadata returned.")
        return df

    df = df.reset_index()

    # Keep site_no and renamed attribute columns
    keep = ["site_no"] + [c for c in RENAME if c in df.columns]
    df = df[keep].rename(columns=RENAME)

    # Fetch discharge (00060) period of record per site
    catalog, _ = nwis.get_info(sites=gage_ids, seriesCatalogOutput=True)
    catalog = catalog.reset_index()
    discharge_por = (
        catalog[catalog["parm_cd"] == "00060"]
        .groupby("site_no")
        .agg(begin_date=("begin_date", "min"), end_date=("end_date", "max"))
        .reset_index()
    )
    df = df.merge(discharge_por, on="site_no", how="left")

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
