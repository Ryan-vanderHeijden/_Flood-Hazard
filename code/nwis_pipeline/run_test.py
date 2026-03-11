"""
Quick test run of the NWIS pipeline on the first 10 gages.
Writes outputs to data/test/ to avoid touching production data.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data" / "test"
sys.path.insert(0, str(BASE_DIR / "src"))

from fetch_streamflow import fetch_streamflow
from fetch_site_metadata import fetch_site_metadata
from fetch_flood_stages import fetch_flood_stages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_pipeline")

GAGES = [
    "01011000", "01013500", "01015800", "01016500", "01017000",
    "01017060", "01017290", "01017550", "01017960", "01018000",
]

def main():
    logger.info("Test run: %d gages → %s", len(GAGES), DATA_DIR)

    logger.info("=" * 50)
    logger.info("SERVICE 1: Site metadata")
    logger.info("=" * 50)
    site_info = fetch_site_metadata(gage_ids=GAGES, out_path=DATA_DIR / "metadata")

    site_dates = {
        row["site_no"]: (str(row["begin_date"]), str(row["end_date"]))
        for _, row in site_info.iterrows()
        if pd.notna(row.get("begin_date")) and pd.notna(row.get("end_date"))
    }

    logger.info("=" * 50)
    logger.info("SERVICE 2: Daily streamflow + stage")
    logger.info("=" * 50)
    streamflow_df = fetch_streamflow(site_dates=site_dates, out_path=DATA_DIR / "streamflow")

    logger.info("=" * 50)
    logger.info("SERVICE 3: Flood stage thresholds")
    logger.info("=" * 50)
    if not streamflow_df.empty and "stage_ft" in streamflow_df.columns:
        has_stage = streamflow_df.groupby("site_no")["stage_ft"].apply(lambda s: s.notna().any())
        stage_gage_ids = has_stage[has_stage].index.tolist()
    else:
        stage_gage_ids = []
    logger.info("%d/%d gages have stage data", len(stage_gage_ids), len(GAGES))

    flood_stages = fetch_flood_stages(
        gage_ids=stage_gage_ids,
        out_path=DATA_DIR / "metadata",
        cache_path=DATA_DIR / "metadata" / "hads_crosswalk.parquet",
    )

    logger.info("=" * 50)
    logger.info("DONE — outputs in %s", DATA_DIR)
    logger.info("=" * 50)
    logger.info("site_info:     %d rows", len(site_info))
    logger.info("streamflow_df: %d rows", len(streamflow_df))
    logger.info("flood_stages:  %d rows", len(flood_stages))

if __name__ == "__main__":
    main()
