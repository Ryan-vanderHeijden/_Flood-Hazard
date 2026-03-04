"""
Data acquisition pipeline — CIROH Flood Hazard Thresholds Project.

Services run in sequence:
  1. Site metadata  — lat, lon, elevation, drainage area, period of record
  2. Streamflow     — daily discharge + stage from USGS NWIS (per-site date ranges)
  3. Flood stages   — action / flood / moderate / major thresholds

Configuration:
  Edit CONFIG_DIR / DATA_DIR as needed.
  Add or remove gage IDs in config/gages.csv.

Usage:
  python run_pipeline.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR   = BASE_DIR / "data"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Service imports (after path is set up)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(BASE_DIR / "src"))
from fetch_streamflow import fetch_streamflow
from fetch_site_metadata import fetch_site_metadata
from fetch_flood_stages import fetch_flood_stages


def load_gage_ids(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, dtype={"site_no": str})
    ids = df["site_no"].str.strip().str.zfill(8).tolist()
    logger.info("Loaded %d gage IDs from %s", len(ids), csv_path)
    return ids


def _summarize_coverage(
    gage_ids: list[str],
    streamflow_df: pd.DataFrame,
    flood_stages: pd.DataFrame,
    out_path: Path,
) -> None:
    """Log and save a per-site data coverage summary."""
    # Stage data availability from streamflow
    if not streamflow_df.empty:
        stage_coverage = (
            streamflow_df.groupby("site_no")["stage_ft"]
            .apply(lambda s: s.notna().any())
            .rename("has_stage_data")
            .reset_index()
        )
    else:
        stage_coverage = pd.DataFrame({"site_no": gage_ids, "has_stage_data": False})

    # Threshold availability from flood stages
    threshold_coverage = flood_stages[["site_no"]].copy()
    threshold_coverage["has_flood_stage"] = flood_stages["flood_stage_ft"].notna()
    threshold_coverage["has_action_stage"] = flood_stages["action_stage_ft"].notna()
    threshold_coverage["has_moderate_stage"] = flood_stages["moderate_stage_ft"].notna()
    threshold_coverage["has_major_stage"] = flood_stages["major_stage_ft"].notna()

    coverage = pd.DataFrame({"site_no": gage_ids}).merge(
        stage_coverage, on="site_no", how="left"
    ).merge(threshold_coverage, on="site_no", how="left")

    coverage["has_stage_data"] = coverage["has_stage_data"].fillna(False)
    for col in ("has_flood_stage", "has_action_stage", "has_moderate_stage", "has_major_stage"):
        coverage[col] = coverage[col].fillna(False)

    # Log warnings for sites missing data
    no_stage = coverage.loc[~coverage["has_stage_data"], "site_no"].tolist()
    no_flood = coverage.loc[~coverage["has_flood_stage"], "site_no"].tolist()
    no_action = coverage.loc[~coverage["has_action_stage"], "site_no"].tolist()

    if no_stage:
        logger.warning("No stage (gage height) data:   %s", no_stage)
    if no_flood:
        logger.warning("No flood stage threshold:      %s", no_flood)
    if no_action:
        logger.warning("No action stage threshold:     %s", no_action)

    out_path.mkdir(parents=True, exist_ok=True)
    parquet_file = out_path / "data_coverage.parquet"
    coverage.to_parquet(parquet_file, index=False)
    logger.info("Saved coverage summary → %s", parquet_file)


def main():
    gage_ids = load_gage_ids(CONFIG_DIR / "gages.csv")

    logger.info("=" * 60)
    logger.info("SERVICE 1: Site metadata")
    logger.info("=" * 60)
    site_info = fetch_site_metadata(
        gage_ids=gage_ids,
        out_path=DATA_DIR / "metadata",
    )

    # Build per-site date ranges from NWIS period-of-record
    site_dates = {
        row["site_no"]: (str(row["begin_date"]), str(row["end_date"]))
        for _, row in site_info.iterrows()
        if pd.notna(row.get("begin_date")) and pd.notna(row.get("end_date"))
    }

    logger.info("=" * 60)
    logger.info("SERVICE 2: Daily streamflow + stage")
    logger.info("=" * 60)
    streamflow_df = fetch_streamflow(
        site_dates=site_dates,
        out_path=DATA_DIR / "streamflow",
    )

    logger.info("=" * 60)
    logger.info("SERVICE 3: Flood stage thresholds")
    logger.info("=" * 60)
    flood_stages = fetch_flood_stages(
        gage_ids=gage_ids,
        out_path=DATA_DIR / "metadata",
    )

    logger.info("=" * 60)
    logger.info("DATA COVERAGE SUMMARY")
    logger.info("=" * 60)
    _summarize_coverage(gage_ids, streamflow_df, flood_stages, DATA_DIR / "metadata")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
