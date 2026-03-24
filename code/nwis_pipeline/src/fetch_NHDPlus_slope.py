from __future__ import annotations

"""
Standalone script: fetch NHDPlus channel slope and add to channel_geometry.parquet.

Reads gage_map.parquet (site_no → reach_id / NHDPlus COMID), queries NHDPlus
flowline attributes via pynhd, and adds `nhd_slope_ft_ft` to
data/metadata/channel_geometry.parquet.

NHDPlus slope is the reach-average channel slope in ft/ft derived from the
3DEP elevation profile (NHDPlus Value Added Attributes, field `slope`).

Usage (standalone):
    cd code/nwis_pipeline
    python src/fetch_NHDPlus_slope.py

Inputs:
    data/metadata/gage_map.parquet         — site_no, reach_id (NWM / COMID)
    data/metadata/channel_geometry.parquet  — written by Service 5 (optional;
                                              created if absent)

Output:
    data/metadata/channel_geometry.parquet  — updated in place with
                                              nhd_slope_ft_ft column

Dependencies: pynhd, pandas, pyarrow
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from pynhd import WaterData

# ---------------------------------------------------------------------------
# Paths  (relative to this script's location: src/)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent
DATA_DIR = _SCRIPT_DIR.parent / "data"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fetch_nhd_slope")

# COMIDs per WaterData request — safe range: 50–200
_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_slope_batch(comids: list[int], wd: WaterData) -> pd.DataFrame:
    """
    Query NHDPlus flowline attributes for a batch of COMIDs.

    Returns a DataFrame with columns [comid (int64), nhd_slope_ft_ft].
    On failure, returns NaN slope for the entire batch.
    """
    try:
        gdf = wd.byid("comid", comids)
    except Exception as exc:
        logger.warning("  WaterData query failed for batch of %d: %s", len(comids), exc)
        return pd.DataFrame({"comid": comids, "nhd_slope_ft_ft": float("nan")})

    if gdf is None or gdf.empty:
        logger.debug("  No data returned for batch of %d COMIDs", len(comids))
        return pd.DataFrame({"comid": comids, "nhd_slope_ft_ft": float("nan")})

    # Drop geometry, work with plain DataFrame
    df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name, errors="ignore"))

    # Locate slope column (case-insensitive; NHDPlus VAA field is "slope", ft/ft)
    slope_col = next((c for c in df.columns if c.lower() == "slope"), None)
    if slope_col is None:
        logger.warning(
            "  'slope' column not found in WaterData response. "
            "Available columns: %s",
            sorted(df.columns.tolist()),
        )
        df["nhd_slope_ft_ft"] = float("nan")
    else:
        df = df.rename(columns={slope_col: "nhd_slope_ft_ft"})

    df["comid"] = df["comid"].astype("int64")
    return df[["comid", "nhd_slope_ft_ft"]]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def fetch_nhd_slope(
    gage_map: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """
    Fetch NHDPlus channel slope for all gages in gage_map with a reach_id
    and add `nhd_slope_ft_ft` to channel_geometry.parquet.

    Args:
        gage_map: DataFrame with [site_no, reach_id] columns. reach_id is
                   the NWM reach ID, which equals the NHDPlus COMID.
        out_path:  Directory containing channel_geometry.parquet (written by
                   Service 5). Created if absent.

    Returns:
        Updated channel_geometry DataFrame with nhd_slope_ft_ft column.
    """
    # Load existing channel_geometry (optional — created by Service 5)
    cg_path = out_path / "channel_geometry.parquet"
    if cg_path.exists():
        channel_geometry = pd.read_parquet(cg_path)
        logger.info("Loaded channel_geometry: %d rows", len(channel_geometry))
    else:
        logger.warning(
            "channel_geometry.parquet not found at %s — will write slope-only output",
            cg_path,
        )
        channel_geometry = pd.DataFrame(columns=["site_no"])

    # Restrict to sites with a reach_id
    has_reach = gage_map["reach_id"].notna()
    gm = gage_map.loc[has_reach, ["site_no", "reach_id"]].copy()
    gm["comid"] = gm["reach_id"].astype(str).str.strip().astype("int64")

    logger.info("Sites with reach_id: %d / %d", len(gm), len(gage_map))

    if gm.empty:
        logger.warning("No sites with reach_id — nothing to fetch")
        return channel_geometry

    # Fetch slope in batches
    wd = WaterData("nhdflowline_network")
    comids = gm["comid"].tolist()
    batch_frames: list[pd.DataFrame] = []

    for i in range(0, len(comids), _BATCH_SIZE):
        batch = comids[i : i + _BATCH_SIZE]
        logger.info(
            "  Querying COMIDs %d – %d / %d",
            i + 1, min(i + _BATCH_SIZE, len(comids)), len(comids),
        )
        batch_frames.append(_fetch_slope_batch(batch, wd))

    slope_vaa = (
        pd.concat(batch_frames, ignore_index=True).drop_duplicates("comid")
        if batch_frames
        else pd.DataFrame({"comid": comids, "nhd_slope_ft_ft": float("nan")})
    )

    # Map comid → site_no
    gm_slope = gm.merge(slope_vaa, on="comid", how="left")[["site_no", "nhd_slope_ft_ft"]]

    n_with_slope = gm_slope["nhd_slope_ft_ft"].notna().sum()
    logger.info("Slope retrieved: %d / %d sites with reach_id", n_with_slope, len(gm_slope))

    # Drop stale slope column if re-running
    if "nhd_slope_ft_ft" in channel_geometry.columns:
        channel_geometry = channel_geometry.drop(columns=["nhd_slope_ft_ft"])

    result = channel_geometry.merge(gm_slope, on="site_no", how="left")

    # Sites with reach_id but no flood stage are not in channel_geometry — append them
    extra = gm_slope[~gm_slope["site_no"].isin(result["site_no"])]
    if not extra.empty:
        logger.info(
            "  Appending %d reach_id sites absent from channel_geometry (no flood stage)",
            len(extra),
        )
        result = pd.concat([result, extra], ignore_index=True)

    # Save
    out_path.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cg_path, index=False)
    logger.info("Saved %d rows → %s", len(result), cg_path)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gage_map_path = DATA_DIR / "metadata" / "gage_map.parquet"
    if not gage_map_path.exists():
        logger.error("gage_map.parquet not found at %s", gage_map_path)
        sys.exit(1)

    gage_map = pd.read_parquet(gage_map_path)
    logger.info("Loaded gage_map: %d sites", len(gage_map))

    fetch_nhd_slope(gage_map, DATA_DIR / "metadata")
