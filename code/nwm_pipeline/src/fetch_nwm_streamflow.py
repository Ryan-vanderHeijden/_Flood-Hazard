"""
fetch_nwm_streamflow.py
-----------------------
Fetch NWM Retrospective v3.0 daily-mean streamflow from the public AWS S3 Zarr
store for every gauge that has a NWM reach_id in gauge_map.parquet.

Data source
-----------
s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr
  Variable : streamflow  (m³/s, NWM native units)
  Time span: ~Feb 1979 – Dec 2023 (hourly; resampled to daily mean here)
  Access   : anonymous (public bucket, us-east-1)

Output
------
<out_path>/nwm_streamflow.parquet
    site_no        str    USGS 8-digit site number
    reach_id       int64  NWM feature_id
    date           date   UTC date of daily mean
    streamflow_cms float32 daily mean streamflow in m³/s

Checkpointing
-------------
One parquet per calendar year is written to <out_path>/checkpoints/{year}.parquet.
Re-runs skip years whose checkpoint already exists, so the job can be safely
interrupted and restarted.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

logger = logging.getLogger(__name__)

_BUCKET_URI = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
_REGION = "us-east-1"
_VARIABLE = "streamflow"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_nwm_streamflow(gauge_map_path: Path, out_path: Path) -> None:
    """Fetch NWM v3.0 daily streamflow and write to ``out_path/nwm_streamflow.parquet``.

    Parameters
    ----------
    gauge_map_path:
        Path to ``gauge_map.parquet`` produced by the NWIS pipeline.
        Required columns: ``site_no`` (str), ``reach_id`` (str or numeric).
    out_path:
        Output directory.  Created if absent.  Per-year checkpoints are written
        to ``out_path/checkpoints/``.
    """
    out_path = Path(out_path)
    checkpoint_dir = out_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load gauge_map, extract valid reach_ids
    # ------------------------------------------------------------------
    logger.info("Loading gauge map from %s", gauge_map_path)
    gmap = pd.read_parquet(gauge_map_path, columns=["site_no", "reach_id"])

    # Drop rows with missing reach_id
    gmap = gmap.dropna(subset=["reach_id"])
    gmap = gmap[gmap["reach_id"].astype(str).str.strip() != ""]

    # Cast to int64 — NWPS returns reach_id as a string
    try:
        gmap["reach_id"] = gmap["reach_id"].astype(np.int64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not cast reach_id column to int64 in {gauge_map_path}. "
            "Inspect the file for non-numeric values."
        ) from exc

    # Deduplicate (keep first site_no per reach_id in case of duplicates)
    gmap = gmap.drop_duplicates(subset=["reach_id"])

    feature_id_to_site_no: dict[int, str] = dict(
        zip(gmap["reach_id"].tolist(), gmap["site_no"].tolist())
    )
    requested_ids = sorted(feature_id_to_site_no.keys())

    logger.info("%d sites have a NWM reach_id", len(requested_ids))

    # ------------------------------------------------------------------
    # Step 2: Open S3 Zarr (lazy — no data downloaded yet)
    # ------------------------------------------------------------------
    logger.info("Opening NWM v3.0 Zarr store: %s", _BUCKET_URI)
    s3 = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": _REGION})
    s3store = s3fs.S3Map(root=_BUCKET_URI, s3=s3, check=False)
    ds = xr.open_zarr(s3store, consolidated=True)

    if _VARIABLE not in ds:
        raise KeyError(
            f"Variable '{_VARIABLE}' not found in Zarr store. "
            f"Available variables: {list(ds.data_vars)}"
        )

    # Determine available feature_ids and warn about any missing ones
    zarr_ids = set(ds["feature_id"].values.tolist())
    available_ids = [fid for fid in requested_ids if fid in zarr_ids]
    missing_ids = [fid for fid in requested_ids if fid not in zarr_ids]

    if missing_ids:
        missing_sites = [feature_id_to_site_no[fid] for fid in missing_ids]
        logger.warning(
            "%d reach_ids not found in Zarr (site_nos: %s ...); they will be skipped.",
            len(missing_ids),
            ", ".join(missing_sites[:10]),
        )
    logger.info(
        "%d/%d reach_ids found in Zarr; fetching streamflow.",
        len(available_ids),
        len(requested_ids),
    )

    if not available_ids:
        raise ValueError("None of the requested reach_ids were found in the Zarr store.")

    # Select only the features we need (lazy)
    ds_filtered = ds[[_VARIABLE]].sel(feature_id=available_ids)

    # Determine full year range from Zarr time coordinate
    times = pd.to_datetime(ds["time"].values)
    start_year = int(times.min().year)
    end_year   = int(times.max().year)
    logger.info("Zarr time range: %d – %d", start_year, end_year)

    # ------------------------------------------------------------------
    # Step 3: Year-by-year processing with checkpointing
    # ------------------------------------------------------------------
    total_years = end_year - start_year + 1
    completed_years = 0

    for year in range(start_year, end_year + 1):
        ckpt = checkpoint_dir / f"{year}.parquet"
        if ckpt.exists():
            logger.debug("Year %d: checkpoint exists, skipping.", year)
            completed_years += 1
            continue

        logger.info("Year %d/%d: fetching and resampling …", year, end_year)

        ds_year = ds_filtered.sel(
            time=slice(f"{year}-01-01", f"{year}-12-31T23:59:59")
        )

        if ds_year.sizes["time"] == 0:
            logger.warning("Year %d: no time steps found in Zarr, skipping.", year)
            # Write empty checkpoint so we don't retry
            pd.DataFrame(
                columns=["site_no", "reach_id", "date", "streamflow_cms"]
            ).to_parquet(ckpt, index=False)
            completed_years += 1
            continue

        # Resample hourly → daily mean (compute triggers actual S3 reads)
        daily = ds_year.resample(time="1D").mean(skipna=True).compute()

        df = (
            daily
            .to_dataframe()
            .reset_index()
            .rename(columns={_VARIABLE: "streamflow_cms", "time": "date"})
        )

        # Convert timestamp → date
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Map feature_id → site_no
        df["site_no"] = df["feature_id"].map(feature_id_to_site_no)
        df["reach_id"] = df["feature_id"].astype(np.int64)

        # Drop rows where all streamflow is NaN (no data for this feature/year)
        df = df.dropna(subset=["streamflow_cms"])

        # Cast to float32 to match NWM source precision
        df["streamflow_cms"] = df["streamflow_cms"].astype(np.float32)

        df = df[["site_no", "reach_id", "date", "streamflow_cms"]]
        df.to_parquet(ckpt, index=False)
        completed_years += 1

        logger.debug(
            "Year %d: %d rows written to %s", year, len(df), ckpt.name
        )

    logger.info(
        "All %d years processed (%d checkpoints written/found).",
        total_years,
        completed_years,
    )

    # ------------------------------------------------------------------
    # Step 4: Consolidate checkpoints → final parquet
    # ------------------------------------------------------------------
    _consolidate(checkpoint_dir, out_path, start_year, end_year)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _consolidate(
    checkpoint_dir: Path,
    out_path: Path,
    start_year: int,
    end_year: int,
) -> None:
    """Concatenate all yearly checkpoints into a single sorted parquet."""
    logger.info("Consolidating yearly checkpoints …")

    parts = []
    for year in range(start_year, end_year + 1):
        ckpt = checkpoint_dir / f"{year}.parquet"
        if ckpt.exists():
            parts.append(pd.read_parquet(ckpt))

    if not parts:
        logger.warning("No checkpoint files found — output will be empty.")
        df_final = pd.DataFrame(
            columns=["site_no", "reach_id", "date", "streamflow_cms"]
        )
    else:
        df_final = pd.concat(parts, ignore_index=True)
        df_final = df_final.sort_values(["site_no", "date"]).reset_index(drop=True)

    out_file = out_path / "nwm_streamflow.parquet"
    df_final.to_parquet(out_file, index=False)

    n_sites = df_final["site_no"].nunique() if not df_final.empty else 0
    n_rows = len(df_final)
    logger.info(
        "Saved %d rows across %d sites → %s", n_rows, n_sites, out_file
    )
