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
    site_no        str     USGS 8-digit site number
    reach_id       int64   NWM feature_id
    date           date    UTC date of daily mean
    streamflow_cms float32 daily mean streamflow in m³/s

Checkpointing
-------------
One parquet per calendar year is written to <out_path>/checkpoints/{year}.parquet.
Re-runs skip years whose checkpoint already exists, so the job can be safely
interrupted and restarted. Delete a year file to force re-fetch of that year.

Parallelism
-----------
Years are processed concurrently using ThreadPoolExecutor. Each worker opens
its own S3 connection to avoid event-loop conflicts in s3fs. The default
_YEAR_WORKERS = 4 balances S3 throughput against memory (~250 MB peak per
concurrent year at 7,000 sites). Raise it on machines with more RAM and a
fast connection to AWS us-east-1.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

logger = logging.getLogger(__name__)

_BUCKET_URI = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
_REGION = "us-east-1"
_VARIABLE = "streamflow"
_YEAR_WORKERS = 4  # concurrent years; each holds ~250 MB peak at 7,000 sites


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

    # Deduplicate: keep first site_no per reach_id
    gmap = gmap.drop_duplicates(subset=["reach_id"])

    feature_id_to_site_no: dict[int, str] = dict(
        zip(gmap["reach_id"].tolist(), gmap["site_no"].tolist())
    )
    requested_ids = sorted(feature_id_to_site_no.keys())
    logger.info("%d sites have a NWM reach_id", len(requested_ids))

    # ------------------------------------------------------------------
    # Step 2: Open S3 Zarr once to discover available feature_ids + year range
    # ------------------------------------------------------------------
    logger.info("Opening NWM v3.0 Zarr store: %s", _BUCKET_URI)
    s3_meta = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": _REGION})
    ds_meta = xr.open_zarr(
        s3fs.S3Map(root=_BUCKET_URI, s3=s3_meta, check=False),
        consolidated=True,
    )

    if _VARIABLE not in ds_meta:
        raise KeyError(
            f"Variable '{_VARIABLE}' not found in Zarr store. "
            f"Available variables: {list(ds_meta.data_vars)}"
        )

    zarr_ids = set(ds_meta["feature_id"].values.tolist())
    available_ids = [fid for fid in requested_ids if fid in zarr_ids]
    missing_ids   = [fid for fid in requested_ids if fid not in zarr_ids]

    if missing_ids:
        missing_sites = [feature_id_to_site_no[fid] for fid in missing_ids]
        logger.warning(
            "%d reach_ids not found in Zarr (e.g. %s); they will be skipped.",
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

    times = pd.to_datetime(ds_meta["time"].values)
    start_year = int(times.min().year)
    end_year   = int(times.max().year)
    logger.info("Zarr time range: %d – %d", start_year, end_year)

    ds_meta.close()  # metadata connection no longer needed

    # ------------------------------------------------------------------
    # Step 3: Parallel year processing with checkpointing
    # ------------------------------------------------------------------
    years = list(range(start_year, end_year + 1))
    years_to_fetch = [y for y in years if not (checkpoint_dir / f"{y}.parquet").exists()]
    years_cached   = len(years) - len(years_to_fetch)

    if years_cached:
        logger.info(
            "%d/%d years already checkpointed; fetching remaining %d.",
            years_cached, len(years), len(years_to_fetch),
        )

    if years_to_fetch:
        logger.info(
            "Fetching %d years with %d concurrent workers.",
            len(years_to_fetch), _YEAR_WORKERS,
        )
        with ThreadPoolExecutor(max_workers=_YEAR_WORKERS) as executor:
            futures = {
                executor.submit(
                    _fetch_year,
                    year, available_ids, feature_id_to_site_no, checkpoint_dir,
                ): year
                for year in years_to_fetch
            }
            for future in as_completed(futures):
                year = futures[future]
                try:
                    n_rows = future.result()
                    logger.info("Year %d complete: %d rows.", year, n_rows)
                except Exception as exc:
                    logger.error("Year %d failed: %s", year, exc)

    logger.info("All %d years processed.", len(years))

    # ------------------------------------------------------------------
    # Step 4: Consolidate checkpoints → final parquet
    # ------------------------------------------------------------------
    _consolidate(checkpoint_dir, out_path, start_year, end_year)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_year(
    year: int,
    available_ids: list[int],
    feature_id_to_site_no: dict[int, str],
    checkpoint_dir: Path,
) -> int:
    """Fetch one calendar year of NWM streamflow and write a checkpoint parquet.

    Each call opens its own S3 connection so threads do not share an asyncio
    event loop (the safe pattern for concurrent s3fs usage).

    Returns the number of rows written.
    """
    ckpt = checkpoint_dir / f"{year}.parquet"

    # Re-check inside the worker in case another thread beat us to it
    if ckpt.exists():
        return 0

    s3 = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": _REGION})
    ds = xr.open_zarr(
        s3fs.S3Map(root=_BUCKET_URI, s3=s3, check=False),
        consolidated=True,
    )
    ds_filtered = ds[[_VARIABLE]].sel(feature_id=available_ids)

    ds_year = ds_filtered.sel(
        time=slice(f"{year}-01-01", f"{year}-12-31T23:59:59")
    )

    if ds_year.sizes["time"] == 0:
        pd.DataFrame(
            columns=["site_no", "reach_id", "date", "streamflow_cms"]
        ).to_parquet(ckpt, index=False)
        return 0

    # Resample hourly → daily mean (.compute() triggers the S3 reads)
    daily = ds_year.resample(time="1D").mean(skipna=True).compute()

    df = (
        daily
        .to_dataframe()
        .reset_index()
        .rename(columns={_VARIABLE: "streamflow_cms", "time": "date"})
    )

    df["date"]        = pd.to_datetime(df["date"]).dt.date
    df["site_no"]     = df["feature_id"].map(feature_id_to_site_no)
    df["reach_id"]    = df["feature_id"].astype(np.int64)
    df["streamflow_cms"] = df["streamflow_cms"].astype(np.float32)

    df = df[["site_no", "reach_id", "date", "streamflow_cms"]]
    df = df.dropna(subset=["streamflow_cms"])

    df.to_parquet(ckpt, index=False)
    ds.close()
    return len(df)


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
    logger.info(
        "Saved %d rows across %d sites → %s", len(df_final), n_sites, out_file
    )
