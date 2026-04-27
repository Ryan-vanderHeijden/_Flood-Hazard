"""
Data acquisition pipeline — CIROH Flood Hazard Thresholds Project.

Services run in sequence:
  1. Site metadata    — lat, lon, elevation, drainage area, period of record
  2. Streamflow       — daily discharge + stage from USGS NWIS (per-site date ranges)
  3. Flood stages     — action / flood / moderate / major thresholds (NWS NWPS)
  4. Rating curves    — fill NaN flood flow thresholds via USGS stage-discharge ratings
  5. Bankfull width   — estimate bankfull channel width via StreamStats regional regressions
  6. Percentiles      — non-exceedance percentile of each flood flow threshold in observed record

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
DATA_DIR   = Path("/home/ryan/data/flood_hazard")


# ---------------------------------------------------------------------------
# Logging — console + rotating log file
# ---------------------------------------------------------------------------
_log_file = BASE_DIR / "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Service imports (after path is set up)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(BASE_DIR / "src"))
from fetch_streamflow import fetch_streamflow
from fetch_site_metadata import fetch_site_metadata
from fetch_flood_stages import fetch_flood_stages
from fetch_rating_curves import fill_flows_from_ratings
from fetch_bankfull_width import fetch_bankfull_width
from compute_flood_percentiles import compute_flood_percentiles


def load_gage_ids(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, dtype={"site_no": str})
    ids = df["site_no"].str.strip().str.zfill(8).tolist()
    logger.info("Loaded %d gage IDs from %s", len(ids), csv_path)
    return ids


def _summarize_coverage(
    gage_ids: list[str],
    streamflow_df: pd.DataFrame,
    flood_stages: pd.DataFrame,
    gauge_map: pd.DataFrame,
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

    # Threshold availability from flood stages (after rating-curve fill)
    threshold_coverage = flood_stages[["site_no"]].copy()
    threshold_coverage["has_flood_stage"] = flood_stages["flood_stage_ft"].notna()
    threshold_coverage["has_action_stage"] = flood_stages["action_stage_ft"].notna()
    threshold_coverage["has_moderate_stage"] = flood_stages["moderate_stage_ft"].notna()
    threshold_coverage["has_major_stage"] = flood_stages["major_stage_ft"].notna()
    threshold_coverage["has_flood_flow"] = flood_stages["flood_flow_cfs"].notna()

    coverage = pd.DataFrame({"site_no": gage_ids}).merge(
        stage_coverage, on="site_no", how="left"
    ).merge(threshold_coverage, on="site_no", how="left")

    coverage["has_stage_data"] = coverage["has_stage_data"].fillna(False)
    for col in ("has_flood_stage", "has_action_stage", "has_moderate_stage", "has_major_stage", "has_flood_flow"):
        coverage[col] = coverage[col].fillna(False)

    # Reach ID availability from gauge_map (includes NLDI-resolved sites)
    if not gauge_map.empty:
        reach_coverage = (
            gauge_map[gauge_map["reach_id"].notna()][["site_no"]]
            .drop_duplicates()
            .assign(has_reach_id=True)
        )
        coverage = coverage.merge(reach_coverage, on="site_no", how="left")
    else:
        coverage["has_reach_id"] = False
    coverage["has_reach_id"] = coverage["has_reach_id"].fillna(False)

    out_path.mkdir(parents=True, exist_ok=True)
    parquet_file = out_path / "data_coverage.parquet"
    coverage.to_parquet(parquet_file, index=False)
    logger.info("Saved coverage summary → %s", parquet_file)


def _validate_outputs(
    gage_ids: list[str],
    site_info: pd.DataFrame,
    streamflow_df: pd.DataFrame,
    flood_stages: pd.DataFrame,
    gauge_map: pd.DataFrame,
) -> None:
    """Log output consistency checks and a pipeline-funnel summary."""
    n_total = len(gage_ids)

    n_meta = site_info["site_no"].nunique() if not site_info.empty else 0

    if not streamflow_df.empty and "discharge_cfs" in streamflow_df.columns:
        n_discharge = int(
            streamflow_df.groupby("site_no")["discharge_cfs"]
            .apply(lambda s: s.notna().any())
            .sum()
        )
    else:
        n_discharge = 0

    if not streamflow_df.empty and "stage_ft" in streamflow_df.columns:
        n_stage = int(
            streamflow_df.groupby("site_no")["stage_ft"]
            .apply(lambda s: s.notna().any())
            .sum()
        )
    else:
        n_stage = 0

    n_flood_threshold = int(flood_stages["flood_stage_ft"].notna().sum()) if not flood_stages.empty else 0
    n_reach = int(gauge_map["reach_id"].notna().sum()) if not gauge_map.empty else 0

    logger.info(
        "Pipeline funnel: %d total → %d metadata → %d discharge → "
        "%d stage → %d flood threshold → %d reach_id",
        n_total, n_meta, n_discharge, n_stage, n_flood_threshold, n_reach,
    )

    # Streamflow sites should be a subset of site_info
    if not streamflow_df.empty and not site_info.empty:
        flow_sites = set(streamflow_df["site_no"].unique())
        meta_sites = set(site_info["site_no"].astype(str).unique())
        orphan = flow_sites - meta_sites
        if orphan:
            logger.warning(
                "%d streamflow sites missing from site_info: %s",
                len(orphan), sorted(orphan),
            )

    # Flag any gauge_map rows where reach_id is still null after NLDI fallback
    if not gauge_map.empty:
        null_reach = gauge_map["reach_id"].isna().sum()
        if null_reach:
            logger.warning("%d sites in gauge_map have null reach_id", null_reach)


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
    # Only fetch flood stages for sites that have observed stage (gage height) data.
    if not streamflow_df.empty and "stage_ft" in streamflow_df.columns:
        has_stage = (
            streamflow_df.groupby("site_no")["stage_ft"]
            .apply(lambda s: s.notna().any())
        )
        stage_gage_ids = has_stage[has_stage].index.tolist()
    else:
        stage_gage_ids = []
    logger.info("%d/%d gages have stage data", len(stage_gage_ids), len(gage_ids))

    flood_stages = fetch_flood_stages(
        gage_ids=stage_gage_ids,
        out_path=DATA_DIR / "metadata",
        cache_path=DATA_DIR / "metadata" / "hads_crosswalk.parquet",
    )

    # Read gauge_map back from disk (written by fetch_flood_stages).
    gauge_map_path = DATA_DIR / "metadata" / "gauge_map.parquet"
    gauge_map = (
        pd.read_parquet(gauge_map_path)
        if gauge_map_path.exists()
        else pd.DataFrame(columns=["site_no", "lid", "reach_id"])
    )

    logger.info("=" * 60)
    logger.info("SERVICE 4: Fill flood flows from USGS rating curves")
    logger.info("=" * 60)
    flood_stages = fill_flows_from_ratings(
        flood_stages=flood_stages,
        out_path=DATA_DIR / "metadata",
    )

    logger.info("=" * 60)
    logger.info("SERVICE 5: Bankfull width from StreamStats regional regressions")
    logger.info("=" * 60)
    fetch_bankfull_width(
        site_info=site_info,
        flood_stages=flood_stages,
        out_path=DATA_DIR / "metadata",
    )

    logger.info("=" * 60)
    logger.info("SERVICE 6: Flood flow threshold percentiles")
    logger.info("=" * 60)
    compute_flood_percentiles(
        streamflow_path=DATA_DIR / "streamflow",
        flood_stages_path=DATA_DIR / "metadata",
        out_path=DATA_DIR / "metadata",
    )

    logger.info("=" * 60)
    logger.info("VALIDATION")
    logger.info("=" * 60)
    _validate_outputs(gage_ids, site_info, streamflow_df, flood_stages, gauge_map)

    logger.info("=" * 60)
    logger.info("DATA COVERAGE SUMMARY")
    logger.info("=" * 60)
    _summarize_coverage(gage_ids, streamflow_df, flood_stages, gauge_map, DATA_DIR / "metadata")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
