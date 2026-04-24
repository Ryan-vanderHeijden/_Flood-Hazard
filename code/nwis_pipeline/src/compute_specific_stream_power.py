import logging
import sys
from pathlib import Path

import pandas as pd


CFS_TO_CMS = 0.028316831999
FT_TO_M = 0.3048

# ---------------------------------------------------------------------------
# Paths  (relative to this script's location: src/)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
DATA_DIR = _SCRIPT_DIR.parent.parent.parent / "data"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("compute_ssp")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def specific_stream_power(gamma: float = 9800, Q: float = None, S: float = None, w: float = None) -> float:
    """
    Compute specific stream power (W/m²).

        gamma   unit weight of water (N/m³), default 9800
        Q       discharge (m³/s)
        S       energy slope / channel slope (m/m, dimensionless)
        w       channel width (m)

    Returns omega = (gamma * Q * S) / w  in W/m².
    """
    return (gamma * Q * S) / w


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def compute_specific_stream_power(
    channel_geometry: pd.DataFrame,
    flood_stages: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute specific stream power at each NWS flood threshold for every gage.

    Inputs
    ------
    channel_geometry : DataFrame with columns
        site_no, bankfull_width_ft, nhd_slope_ft_ft
    flood_stages : DataFrame with columns
        site_no, action_flow_cfs, flood_flow_cfs, moderate_flow_cfs, major_flow_cfs

    Returns
    -------
    DataFrame with columns:
        site_no,
        action_ssp_wm2, flood_ssp_wm2, moderate_ssp_wm2, major_ssp_wm2
    """
    # Merge on site_no
    df = channel_geometry[["site_no", "bankfull_width_ft", "nhd_slope_ft_ft"]].merge(
        flood_stages[["site_no", "action_flow_cfs", "flood_flow_cfs",
                      "moderate_flow_cfs", "major_flow_cfs"]],
        on="site_no",
        how="inner",
    )

    # Unit conversions
    # slope: ft/ft is dimensionless — same numeric value in m/m, no conversion
    df["width_m"] = df["bankfull_width_ft"] * FT_TO_M
    for col in ["action_flow_cfs", "flood_flow_cfs", "moderate_flow_cfs", "major_flow_cfs"]:
        df[col.replace("_cfs", "_cms")] = df[col] * CFS_TO_CMS

    # Compute SSP for each threshold
    for threshold in ("action", "flood", "moderate", "major"):
        df[f"{threshold}_ssp_wm2"] = df.apply(
            lambda row, t=threshold: specific_stream_power(
                Q=row[f"{t}_flow_cms"],
                S=row["nhd_slope_ft_ft"],
                w=row["width_m"],
            )
            if pd.notna(row[f"{t}_flow_cms"])
            and pd.notna(row["nhd_slope_ft_ft"])
            and pd.notna(row["width_m"])
            and row["width_m"] > 0
            else float("nan"),
            axis=1,
        )

    n_total = len(df)
    for threshold in ("action", "flood", "moderate", "major"):
        n_valid = df[f"{threshold}_ssp_wm2"].notna().sum()
        logger.info("  %s: %d / %d sites with valid SSP", threshold, n_valid, n_total)

    out_cols = ["site_no",
                "action_ssp_wm2", "flood_ssp_wm2",
                "moderate_ssp_wm2", "major_ssp_wm2"]
    return df[out_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    meta_dir = DATA_DIR / "metadata"

    channel_geometry = pd.read_parquet(meta_dir / "channel_geometry.parquet")
    logger.info("Loaded channel_geometry: %d rows", len(channel_geometry))

    flood_stages = pd.read_parquet(meta_dir / "flood_stages.parquet")
    logger.info("Loaded flood_stages: %d rows", len(flood_stages))

    result = compute_specific_stream_power(channel_geometry, flood_stages)
    logger.info("Computed SSP for %d sites", len(result))

    out_path = meta_dir / "stream_power.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Saved → %s", out_path)
