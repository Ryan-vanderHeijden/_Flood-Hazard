"""
Service 1: Fetch daily mean discharge and stage from USGS NWIS.

Parameters fetched:
  00060 — Discharge (ft³/s)
  00065 — Gage height / stage (ft)  [not all gages have this sensor]

Output: data/streamflow/streamflow.parquet
Columns: site_no, date, discharge_cfs, discharge_cd, stage_ft, stage_cd
"""

import logging
from pathlib import Path

import pandas as pd
import dataretrieval.nwis as nwis

logger = logging.getLogger(__name__)

PARAM_DISCHARGE = "00060"
PARAM_STAGE = "00065"


def _normalize_site_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Rename NWIS columns and ensure expected columns exist."""
    df = raw_df.reset_index()

    col_map = {}
    for col in df.columns:
        if col in ("site_no", "datetime"):
            continue
        if PARAM_DISCHARGE in col and "_cd" not in col:
            col_map[col] = "discharge_cfs"
        elif PARAM_DISCHARGE in col and "_cd" in col:
            col_map[col] = "discharge_cd"
        elif PARAM_STAGE in col and "_cd" not in col:
            col_map[col] = "stage_ft"
        elif PARAM_STAGE in col and "_cd" in col:
            col_map[col] = "stage_cd"

    df = df.rename(columns=col_map).rename(columns={"datetime": "date"})

    # NWIS may return multiple sensors for the same parameter; keep first occurrence
    dupes = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
    if dupes:
        site_no = df["site_no"].iloc[0]
        logger.warning(
            "  %s: multiple sensors detected for %s — keeping first occurrence only",
            site_no, dupes,
        )
    df = df.loc[:, ~df.columns.duplicated()]

    for col in ("discharge_cfs", "discharge_cd", "stage_ft", "stage_cd"):
        if col not in df.columns:
            df[col] = pd.NA

    df = df[["site_no", "date", "discharge_cfs", "discharge_cd", "stage_ft", "stage_cd"]]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def fetch_streamflow(
    site_dates: dict[str, tuple[str, str]],
    out_path: Path,
) -> pd.DataFrame:
    """
    Fetch daily discharge and stage for all gages and save to Parquet.

    Args:
        site_dates: Mapping of site_no → (begin_date, end_date), e.g.
                    {"01144000": ("1960-10-01", "2025-03-03")}.
        out_path:   Directory where streamflow.parquet will be written.

    Returns:
        DataFrame with columns [site_no, date, discharge_cfs, discharge_cd,
        stage_ft, stage_cd].
    """
    logger.info("Fetching daily discharge + stage for %d gages (per-site date ranges)", len(site_dates))

    frames = []
    for site_no, (start, end) in site_dates.items():
        logger.info("  %s: %s to %s", site_no, start, end)
        raw, _ = nwis.get_dv(
            sites=[site_no],
            parameterCd=[PARAM_DISCHARGE, PARAM_STAGE],
            start=start,
            end=end,
        )
        if raw.empty:
            logger.warning("  %s: no data returned", site_no)
            continue
        frames.append(_normalize_site_df(raw))

    if not frames:
        logger.warning("No data returned for any gage.")
        return pd.DataFrame(columns=["site_no", "date", "discharge_cfs", "discharge_cd", "stage_ft", "stage_cd"])

    df = pd.concat(frames, ignore_index=True)

    # Per-gage record counts
    counts = df.groupby("site_no").size()
    for site, n in counts.items():
        has_stage = df.loc[df["site_no"] == site, "stage_ft"].notna().any()
        logger.info(
            "  %s: %d records | stage data: %s", site, n, "yes" if has_stage else "no"
        )

    missing = set(site_dates) - set(df["site_no"].unique())
    if missing:
        logger.warning("No data returned for: %s", sorted(missing))

    out_path.mkdir(parents=True, exist_ok=True)
    parquet_file = out_path / "streamflow.parquet"
    df.to_parquet(parquet_file, index=False)
    logger.info("Saved %d rows → %s", len(df), parquet_file)

    return df
