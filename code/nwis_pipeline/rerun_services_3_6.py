"""
Re-run Services 3-6 of the NWIS pipeline against the existing streamflow.

Use this after changing flood-stage / gage_map logic when Services 1-2 do not
need to change.  run_pipeline.py re-downloads all streamflow from NWIS
whenever the Service 2 checkpoints are missing; this script reuses
data/streamflow/streamflow.parquet instead.

Steps:
  3. Flood stages + gage_map (incl. NLDI reach_id fallback)
  4. Rating-curve flood flows
  5. Bankfull width            — rewrites channel_geometry.parquet
  6. Flood flow percentiles
  then validation + data_coverage.parquet, and the two standalone steps
  that Service 5 invalidates:
     NHDPlus slope             — re-adds nhd_slope_ft_ft to channel_geometry
     specific stream power     — rewrites stream_power.parquet

Per-site stage/discharge availability is read from streamflow.parquet one row
group at a time, so the full 177M-row table is never loaded here (Service 6
still loads site_no + discharge_cfs, ~10 GB peak).

Usage:
  cd code/nwis_pipeline
  python rerun_services_3_6.py
"""

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

import run_pipeline as rp  # logging → pipeline.log, adds src/ to sys.path
from fetch_NHDPlus_slope import fetch_nhd_slope
from compute_specific_stream_power import compute_specific_stream_power

logger = rp.logger
DATA_DIR = rp.DATA_DIR
META_DIR = DATA_DIR / "metadata"
SF_FILE  = DATA_DIR / "streamflow" / "streamflow.parquet"


def _sites_with(col: str) -> set[str]:
    """Sites with at least one non-null, non-NaN value in ``col``."""
    pf = pq.ParquetFile(SF_FILE)
    sites: set[str] = set()
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["site_no", col])
        valid = pc.and_(pc.is_valid(t[col]), pc.invert(pc.is_nan(t[col])))
        sites.update(pc.unique(t["site_no"].filter(valid)).to_pylist())
    return sites


def _section(title: str) -> None:
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def main():
    gage_ids = rp.load_gage_ids(rp.CONFIG_DIR / "gages.csv")
    site_info = rp.fetch_site_metadata(gage_ids=gage_ids, out_path=META_DIR)  # cached

    _section(f"RERUN (services 3-6) — reusing {SF_FILE}")
    stage_sites = _sites_with("stage_ft")
    discharge_sites = _sites_with("discharge_cfs")
    stage_gage_ids = sorted(stage_sites)
    logger.info("%d/%d gages have stage data", len(stage_gage_ids), len(gage_ids))

    # One row per site with the same any()-semantics the validation and
    # coverage helpers apply to the full streamflow table.
    all_sites = sorted(stage_sites | discharge_sites)
    sf_proxy = pd.DataFrame({
        "site_no": all_sites,
        "discharge_cfs": [1.0 if s in discharge_sites else float("nan") for s in all_sites],
        "stage_ft": [1.0 if s in stage_sites else float("nan") for s in all_sites],
    })

    _section("SERVICE 3: Flood stage thresholds")
    flood_stages = rp.fetch_flood_stages(
        gage_ids=stage_gage_ids,
        out_path=META_DIR,
        cache_path=META_DIR / "hads_crosswalk.parquet",
    )
    gage_map = pd.read_parquet(META_DIR / "gage_map.parquet")

    _section("SERVICE 4: Fill flood flows from USGS rating curves")
    flood_stages = rp.fill_flows_from_ratings(flood_stages=flood_stages, out_path=META_DIR)

    _section("SERVICE 5: Bankfull width from StreamStats regional regressions")
    rp.fetch_bankfull_width(site_info=site_info, flood_stages=flood_stages, out_path=META_DIR)

    _section("SERVICE 6: Flood flow threshold percentiles")
    rp.compute_flood_percentiles(
        streamflow_path=DATA_DIR / "streamflow",
        flood_stages_path=META_DIR,
        out_path=META_DIR,
    )

    _section("NHDPlus slope (restores nhd_slope_ft_ft after Service 5)")
    fetch_nhd_slope(gage_map, META_DIR)

    _section("Specific stream power")
    ssp = compute_specific_stream_power(
        pd.read_parquet(META_DIR / "channel_geometry.parquet"),
        pd.read_parquet(META_DIR / "flood_stages.parquet"),
    )
    ssp.to_parquet(META_DIR / "stream_power.parquet", index=False)
    logger.info("Saved %d rows → %s", len(ssp), META_DIR / "stream_power.parquet")

    _section("VALIDATION")
    rp._validate_outputs(gage_ids, site_info, sf_proxy, flood_stages, gage_map)

    _section("DATA COVERAGE SUMMARY")
    rp._summarize_coverage(gage_ids, sf_proxy, flood_stages, gage_map, META_DIR)

    logger.info("Rerun complete.")


if __name__ == "__main__":
    main()
