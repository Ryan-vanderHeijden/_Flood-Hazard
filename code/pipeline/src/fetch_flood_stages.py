"""
Service 3: Fetch USGS flood stage thresholds for each gage.

Thresholds retrieved (where available):
  action_stage    — Stage at which NWS/USGS issues an action alert (ft)
  flood_stage     — Stage at which minor flooding begins (ft)
  moderate_stage  — Stage at which moderate flooding begins (ft)
  major_stage     — Stage at which major flooding begins (ft)

These are fetched from the NWIS site service with siteOutput=expanded.
Not all gages have assigned thresholds; missing values are left as NaN.

Output: data/metadata/flood_stages.parquet
"""

import io
import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NWIS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"

# Mapping from NWIS RDB column names → readable names.
# The exact column names vary; we try several known variants.
STAGE_COLS = {
    "action_stage": "action_stage_ft",
    "flood_stage": "flood_stage_ft",
    "moderate_flood_stage": "moderate_stage_ft",
    "major_flood_stage": "major_stage_ft",
}

# NWIS allows up to ~100 sites per request; batch to be safe
BATCH_SIZE = 50


def _parse_rdb(text: str) -> pd.DataFrame:
    """Parse NWIS RDB (tab-separated with two-line header) into a DataFrame."""
    lines = text.splitlines()
    data_lines = [l for l in lines if not l.startswith("#")]
    if len(data_lines) < 2:
        return pd.DataFrame()
    # First non-comment line = headers, second = type codes (skip), rest = data
    header = data_lines[0]
    body = "\n".join([header] + data_lines[2:])
    return pd.read_csv(io.StringIO(body), sep="\t", dtype=str, low_memory=False)


def _fetch_batch(site_nos: list[str]) -> pd.DataFrame:
    params = {
        "format": "rdb",
        "sites": ",".join(site_nos),
        "siteOutput": "expanded",
    }
    resp = requests.get(NWIS_SITE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_rdb(resp.text)


def fetch_flood_stages(gage_ids: list[str], out_path: Path) -> pd.DataFrame:
    """
    Fetch flood stage thresholds for all gages and save to Parquet.

    Args:
        gage_ids: List of USGS site numbers (zero-padded strings).
        out_path: Directory where flood_stages.parquet will be written.

    Returns:
        DataFrame with columns [site_no, action_stage_ft, flood_stage_ft,
        moderate_stage_ft, major_stage_ft].
    """
    logger.info("Fetching flood stage thresholds for %d gages", len(gage_ids))

    batches = [
        gage_ids[i : i + BATCH_SIZE] for i in range(0, len(gage_ids), BATCH_SIZE)
    ]
    frames = []
    for batch in batches:
        try:
            frames.append(_fetch_batch(batch))
        except requests.RequestException as exc:
            logger.error("NWIS request failed for batch %s: %s", batch, exc)

    if not frames:
        logger.warning("No flood stage data retrieved.")
        return pd.DataFrame(columns=["site_no"] + list(STAGE_COLS.values()))

    raw = pd.concat(frames, ignore_index=True)

    # Normalise column names (lowercase, strip whitespace)
    raw.columns = raw.columns.str.strip().str.lower()

    # site_no column may be called 'site_no' or 'agency_cd'+'site_no'
    if "site_no" not in raw.columns:
        logger.error("Could not find site_no column in NWIS response.")
        return pd.DataFrame(columns=["site_no"] + list(STAGE_COLS.values()))

    keep = ["site_no"] + [c for c in STAGE_COLS if c in raw.columns]
    df = raw[keep].copy().rename(columns=STAGE_COLS)

    # Ensure all threshold columns exist
    for col in STAGE_COLS.values():
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce to numeric
    for col in STAGE_COLS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_with_flood = df["flood_stage_ft"].notna().sum()
    n_with_action = df["action_stage_ft"].notna().sum()
    logger.info(
        "  Flood stage defined: %d/%d gages | Action stage defined: %d/%d gages",
        n_with_flood,
        len(df),
        n_with_action,
        len(df),
    )

    out_path.mkdir(parents=True, exist_ok=True)
    parquet_file = out_path / "flood_stages.parquet"
    df.to_parquet(parquet_file, index=False)
    logger.info("Saved %d rows → %s", len(df), parquet_file)

    return df
