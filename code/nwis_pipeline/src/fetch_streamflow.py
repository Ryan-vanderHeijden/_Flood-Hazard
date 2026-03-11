from __future__ import annotations

"""
Service 1: Fetch daily mean discharge and stage from USGS NWIS.

Parameters fetched:
  00060 — Discharge (ft³/s)
  00065 — Gage height / stage (ft)  [not all gages have this sensor]

Stage fetch strategy (two-pass):
  1. DV (daily values): NWIS pre-computed daily means.  Only available for
     older/legacy gauges (~542 of 1 946 in current config).
  2. IV fallback (instantaneous values): modern gauges record 15-min stage but
     NWIS does not store a DV mean.  For sites with no DV stage, we pull the
     raw IV series and compute daily means ourselves.  These rows are marked
     stage_cd = "iv_mean" to distinguish them from native DV means.

Performance notes:
  - Pass 1 and Pass 2 are both parallelized with ThreadPoolExecutor.
  - Checkpoint files (streamflow_dv_checkpoint.parquet,
    streamflow_iv_checkpoint.parquet) are written to out_path so a re-run
    can skip already-fetched sites without repeating API calls.
  - Sites that returned no data are tracked in *_no_data.txt so they are
    also skipped on re-run.

Output: data/streamflow/streamflow.parquet
Columns: site_no, date, discharge_cfs, discharge_cd, stage_ft, stage_cd
"""

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import dataretrieval.nwis as nwis

logger = logging.getLogger(__name__)

PARAM_DISCHARGE = "00060"
PARAM_STAGE = "00065"

# DV fetch tuning
_DV_MAX_WORKERS = 20        # parallel threads for DV fetch
_DV_CHECKPOINT_EVERY = 200  # save DV checkpoint every N completed fetches

# IV fetch tuning
_IV_BATCH_SIZE = 25      # sites per get_iv() call
_IV_YEAR_CHUNK = 10      # years per get_iv() call
_IV_MAX_WORKERS = 10     # parallel threads for IV batch × window calls (reduced to avoid rate-limiting)
_IV_MIN_START = "2000-01-01"  # NWIS IV data not reliably available before ~2000
_IV_MAX_RETRIES = 5      # retry attempts on connection errors
_IV_RETRY_BASE = 4.0     # base seconds for exponential backoff (× 2^attempt + jitter)

_COLS = ["site_no", "date", "discharge_cfs", "discharge_cd", "stage_ft", "stage_cd"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
        site_no = df["site_no"].iloc[0] if "site_no" in df.columns and len(df) > 0 else "unknown"
        logger.debug(
            "  %s: multiple sensors detected for %s — keeping first occurrence only",
            site_no, dupes,
        )
    df = df.loc[:, ~df.columns.duplicated()]

    for col in ("discharge_cfs", "discharge_cd", "stage_ft", "stage_cd"):
        if col not in df.columns:
            df[col] = pd.NA

    df = df[_COLS]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Replace USGS no-data sentinel (-999999) with NaN
    for col in ("discharge_cfs", "stage_ft"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] <= -999000, col] = float("nan")

    return df


def _fetch_dv_single(site_no: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch DV data for one site. Returns a normalized DataFrame or None."""
    try:
        raw, _ = nwis.get_dv(
            sites=[site_no],
            parameterCd=[PARAM_DISCHARGE, PARAM_STAGE],
            start=start,
            end=end,
        )
        if raw.empty:
            return None
        return _normalize_site_df(raw)
    except Exception as exc:
        logger.warning("  %s: DV fetch failed: %s", site_no, exc)
        return None


def _fetch_iv_batch_window(
    batch: list[str], win_start: str, win_end: str
) -> pd.DataFrame | None:
    """
    Fetch IV stage for one (batch × window) pair and resample to daily means.
    Returns a DataFrame with columns [site_no, date, stage_ft] or None.
    Retries up to _IV_MAX_RETRIES times on connection errors.
    """
    last_exc: Exception | None = None
    for attempt in range(_IV_MAX_RETRIES):
        try:
            raw, _ = nwis.get_iv(
                sites=batch,
                parameterCd=PARAM_STAGE,
                start=win_start,
                end=win_end,
            )
            break  # success
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc)
            # Only retry on connection-type errors; empty-response errors are expected
            # for windows with no data and won't succeed on retry.
            is_connection_err = (
                "Connection" in exc_str
                or "ConnectionReset" in exc_str
                or "timeout" in exc_str.lower()
                or "RemoteDisconnected" in exc_str
            )
            if is_connection_err and attempt < _IV_MAX_RETRIES - 1:
                delay = _IV_RETRY_BASE * (2 ** attempt) + random.uniform(0, 2)
                logger.debug(
                    "  IV call (%s → %s) attempt %d failed: %s — retrying in %.1fs",
                    win_start, win_end, attempt + 1, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.warning("  IV call failed (%s → %s): %s", win_start, win_end, exc)
                return None
    else:
        # All retries exhausted
        logger.warning("  IV call failed (%s → %s): %s", win_start, win_end, last_exc)
        return None

    if raw.empty:
        return None

    iv = raw.reset_index()
    iv["date"] = pd.to_datetime(iv["datetime"]).dt.normalize()

    stage_col = next(
        (c for c in iv.columns if PARAM_STAGE in c and "_cd" not in c), None
    )
    if stage_col is None:
        return None

    # Replace USGS no-data sentinel before resampling; averaging sentinels
    # with real values would produce fractional garbage (e.g. -791665.4).
    iv[stage_col] = pd.to_numeric(iv[stage_col], errors="coerce")
    iv.loc[iv[stage_col] <= -999000, stage_col] = float("nan")

    daily = (
        iv.groupby(["site_no", "date"])[stage_col]
        .mean()
        .reset_index()
        .rename(columns={stage_col: "stage_ft"})
    )
    return daily if not daily.empty else None


def _fetch_iv_stage(site_dates: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """
    Fetch stage from instantaneous values (15-min) and resample to daily means.

    All (batch × window) pairs are dispatched concurrently via ThreadPoolExecutor.

    Args:
        site_dates: site_no → (begin_date, end_date) for sites to query.

    Returns:
        DataFrame with columns [site_no, date, stage_ft] (daily means).
        stage_cd is NOT set here; the caller sets it to "iv_mean".
    """
    if not site_dates:
        return pd.DataFrame(columns=["site_no", "date", "stage_ft"])

    sites = list(site_dates.keys())

    # Global date window (union of all per-site ranges, floored at IV data availability)
    global_start = max(
        min(pd.Timestamp(v[0]) for v in site_dates.values()),
        pd.Timestamp(_IV_MIN_START),
    )
    global_end   = max(pd.Timestamp(v[1]) for v in site_dates.values())

    windows: list[tuple[str, str]] = []
    cur = global_start
    while cur <= global_end:
        win_end = min(
            cur + pd.DateOffset(years=_IV_YEAR_CHUNK) - pd.Timedelta(days=1),
            global_end,
        )
        windows.append((cur.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")))
        cur = win_end + pd.Timedelta(days=1)

    batches = [sites[i : i + _IV_BATCH_SIZE] for i in range(0, len(sites), _IV_BATCH_SIZE)]
    total_calls = len(batches) * len(windows)
    logger.info(
        "  IV fetch: %d sites → %d batches × %d time windows = %d API calls (workers=%d)",
        len(sites), len(batches), len(windows), total_calls, _IV_MAX_WORKERS,
    )

    frames: list[pd.DataFrame] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=_IV_MAX_WORKERS) as ex:
        future_map = {
            ex.submit(_fetch_iv_batch_window, batch, ws, we): (batch, ws, we)
            for batch in batches
            for ws, we in windows
        }
        for fut in as_completed(future_map):
            completed += 1
            if completed % 50 == 0:
                logger.info("  IV progress: %d/%d calls done", completed, total_calls)
            result = fut.result()
            if result is not None:
                frames.append(result)

    if not frames:
        return pd.DataFrame(columns=["site_no", "date", "stage_ft"])

    return pd.concat(frames, ignore_index=True)


def _load_no_data_sites(path: Path) -> set[str]:
    """Load the list of sites previously confirmed to have no data."""
    if path.exists():
        return set(path.read_text().splitlines())
    return set()


def _save_no_data_sites(sites: set[str], path: Path) -> None:
    path.write_text("\n".join(sorted(sites)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_streamflow(
    site_dates: dict[str, tuple[str, str]],
    out_path: Path,
) -> pd.DataFrame:
    """
    Fetch daily discharge and stage for all gages and save to Parquet.

    Stage is fetched via two passes:
      1. DV (daily values) — parallelized; covers gages with pre-computed daily means.
      2. IV fallback — for sites with no DV stage, pulls 15-min instantaneous
         values and computes daily means.  Marked stage_cd = "iv_mean".

    Checkpoint files written to out_path enable re-runs to skip already-fetched
    sites:
      streamflow_dv_checkpoint.parquet — DV data accumulated so far
      streamflow_dv_no_data.txt        — sites confirmed to have no DV data
      streamflow_iv_checkpoint.parquet — IV stage data accumulated so far
      streamflow_iv_no_data.txt        — sites confirmed to have no IV data

    Args:
        site_dates: Mapping of site_no → (begin_date, end_date).
        out_path:   Directory where streamflow.parquet will be written.

    Returns:
        DataFrame with columns [site_no, date, discharge_cfs, discharge_cd,
        stage_ft, stage_cd].
    """
    logger.info(
        "Fetching daily discharge + stage for %d gages (workers=%d)",
        len(site_dates), _DV_MAX_WORKERS,
    )

    out_path.mkdir(parents=True, exist_ok=True)
    dv_ckpt_file    = out_path / "streamflow_dv_checkpoint.parquet"
    dv_nodata_file  = out_path / "streamflow_dv_no_data.txt"
    iv_ckpt_file    = out_path / "streamflow_iv_checkpoint.parquet"
    iv_nodata_file  = out_path / "streamflow_iv_no_data.txt"

    # ------------------------------------------------------------------
    # Pass 1: Parallel DV fetch (discharge + stage) with checkpointing
    # ------------------------------------------------------------------

    # Determine already-processed sites (data returned or confirmed empty)
    if dv_ckpt_file.exists():
        dv_checkpoint = pd.read_parquet(dv_ckpt_file)
        done_sites = set(dv_checkpoint["site_no"].unique())
    else:
        dv_checkpoint = pd.DataFrame(columns=_COLS)
        done_sites = set()

    done_sites |= _load_no_data_sites(dv_nodata_file)

    remaining = {s: d for s, d in site_dates.items() if s not in done_sites}
    logger.info(
        "  DV pass: %d sites to fetch, %d already done (checkpoint/no-data)",
        len(remaining), len(site_dates) - len(remaining),
    )

    frames: list[pd.DataFrame] = [dv_checkpoint] if not dv_checkpoint.empty else []
    no_data_sites: set[str] = set()
    completed = 0

    with ThreadPoolExecutor(max_workers=_DV_MAX_WORKERS) as ex:
        future_map = {
            ex.submit(_fetch_dv_single, site_no, start, end): site_no
            for site_no, (start, end) in remaining.items()
        }
        for fut in as_completed(future_map):
            site_no = future_map[fut]
            result = fut.result()
            if result is not None:
                frames.append(result)
            else:
                no_data_sites.add(site_no)
            completed += 1
            if completed % _DV_CHECKPOINT_EVERY == 0:
                partial = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_COLS)
                partial.to_parquet(dv_ckpt_file, index=False)
                all_no_data = _load_no_data_sites(dv_nodata_file) | no_data_sites
                _save_no_data_sites(all_no_data, dv_nodata_file)
                logger.info(
                    "  DV checkpoint: %d/%d sites processed (%d with data, %d no data)",
                    completed, len(remaining),
                    len(partial["site_no"].unique()) if not partial.empty else 0,
                    len(all_no_data),
                )

    # Save final DV checkpoint and no-data list
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(dv_ckpt_file, index=False)
    else:
        logger.warning("No DV data returned for any gage.")
        return pd.DataFrame(columns=_COLS)

    all_no_data = _load_no_data_sites(dv_nodata_file) | no_data_sites
    _save_no_data_sites(all_no_data, dv_nodata_file)

    logger.info(
        "  DV pass complete: %d records, %d sites with data, %d sites no data",
        len(df), df["site_no"].nunique(), len(all_no_data),
    )

    # ------------------------------------------------------------------
    # Pass 2: IV fallback for sites with no DV stage
    # ------------------------------------------------------------------
    dv_stage_sites = (
        df.groupby("site_no")["stage_ft"]
        .apply(lambda s: s.notna().any())
    )
    no_dv_stage = dv_stage_sites[~dv_stage_sites].index.tolist()

    if no_dv_stage:
        logger.info(
            "Pass 2: %d/%d sites lack DV stage — attempting IV fallback",
            len(no_dv_stage), len(dv_stage_sites),
        )

        # Sites already handled in a previous IV run
        iv_done_sites = set()
        if iv_ckpt_file.exists():
            iv_checkpoint = pd.read_parquet(iv_ckpt_file)
            iv_done_sites = set(iv_checkpoint["site_no"].unique())
        else:
            iv_checkpoint = pd.DataFrame(columns=["site_no", "date", "stage_ft"])

        iv_done_sites |= _load_no_data_sites(iv_nodata_file)

        iv_remaining = {
            s: site_dates[s]
            for s in no_dv_stage
            if s in site_dates and s not in iv_done_sites
        }
        logger.info(
            "  IV pass: %d sites to fetch, %d already done (checkpoint/no-data)",
            len(iv_remaining), len(no_dv_stage) - len(iv_remaining),
        )

        if iv_remaining:
            iv_new = _fetch_iv_stage(iv_remaining)
        else:
            iv_new = pd.DataFrame(columns=["site_no", "date", "stage_ft"])

        # Merge new IV data with checkpoint
        iv_stage_parts = [p for p in [iv_checkpoint, iv_new] if not p.empty]
        iv_stage = pd.concat(iv_stage_parts, ignore_index=True) if iv_stage_parts else pd.DataFrame(columns=["site_no", "date", "stage_ft"])

        # Track sites with no IV data
        if not iv_new.empty:
            iv_fetched = set(iv_new["site_no"].unique())
        else:
            iv_fetched = set()
        iv_no_data = {s for s in iv_remaining if s not in iv_fetched}
        all_iv_no_data = _load_no_data_sites(iv_nodata_file) | iv_no_data

        if not iv_stage.empty:
            iv_stage.to_parquet(iv_ckpt_file, index=False)
        _save_no_data_sites(all_iv_no_data, iv_nodata_file)

        if not iv_stage.empty:
            iv_stage_renamed = iv_stage.rename(columns={"stage_ft": "_iv_stage"})
            df = df.merge(iv_stage_renamed, on=["site_no", "date"], how="left")
            fill_mask = df["stage_ft"].isna() & df["_iv_stage"].notna()
            df.loc[fill_mask, "stage_ft"] = df.loc[fill_mask, "_iv_stage"]
            df.loc[fill_mask, "stage_cd"] = "iv_mean"
            df = df.drop(columns=["_iv_stage"])
            logger.info(
                "  IV fallback filled stage for %d site-days across %d sites",
                fill_mask.sum(),
                df.loc[fill_mask, "site_no"].nunique(),
            )
        else:
            logger.warning("  IV fallback returned no data.")
    else:
        logger.info("All sites have DV stage data — IV fallback not needed.")

    # Ensure numeric dtypes (stage_ft can become object when pd.NA mixes with
    # float values filled in from IV data via .loc assignment)
    df["stage_ft"] = pd.to_numeric(df["stage_ft"], errors="coerce")
    df["discharge_cfs"] = pd.to_numeric(df["discharge_cfs"], errors="coerce")

    # ------------------------------------------------------------------
    # Summary logging (vectorized — avoids O(n²) per-site DataFrame scans)
    # ------------------------------------------------------------------
    summary = (
        df.groupby("site_no")
        .agg(
            n_records=("date", "count"),
            has_stage=("stage_ft", lambda s: s.notna().any()),
            iv_count=("stage_cd", lambda s: (s == "iv_mean").sum()),
        )
        .reset_index()
    )

    for _, row in summary.iterrows():
        stage_src = ""
        if row["has_stage"]:
            stage_src = " (iv_mean)" if row["iv_count"] > 0 else " (dv)"
        logger.debug(
            "  %s: %d records | stage: %s%s",
            row["site_no"], row["n_records"],
            "yes" if row["has_stage"] else "no",
            stage_src,
        )

    n_with_stage = int(summary["has_stage"].sum())
    n_iv = int((summary["iv_count"] > 0).sum())
    logger.info(
        "  Stage summary: %d/%d sites have stage (%d via IV fallback, %d via DV only)",
        n_with_stage, len(summary), n_iv, n_with_stage - n_iv,
    )

    missing = set(site_dates) - set(df["site_no"].unique())
    if missing:
        logger.warning("No data returned for: %s", sorted(missing))

    parquet_file = out_path / "streamflow.parquet"
    df.to_parquet(parquet_file, index=False)
    logger.info("Saved %d rows → %s", len(df), parquet_file)

    return df
