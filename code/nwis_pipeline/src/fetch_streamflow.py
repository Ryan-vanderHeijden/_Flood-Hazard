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

# IV fetch tuning
_IV_BATCH_SIZE = 10   # sites per get_iv() call
_IV_YEAR_CHUNK = 5    # years per get_iv() call (limits response size)


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


def _fetch_iv_stage(site_dates: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """
    Fetch stage from instantaneous values (15-min) and resample to daily means.

    Requests are batched (_IV_BATCH_SIZE sites) and chunked into _IV_YEAR_CHUNK-year
    windows to keep individual response sizes manageable.

    Args:
        site_dates: site_no → (begin_date, end_date) for sites to query.

    Returns:
        DataFrame with columns [site_no, date, stage_ft] (daily means).
        stage_cd is NOT set here; the caller sets it to "iv_mean".
    """
    if not site_dates:
        return pd.DataFrame(columns=["site_no", "date", "stage_ft"])

    sites = list(site_dates.keys())

    # Global date window (union of all per-site ranges)
    global_start = min(pd.Timestamp(v[0]) for v in site_dates.values())
    global_end   = max(pd.Timestamp(v[1]) for v in site_dates.values())

    # Build time windows
    windows: list[tuple[str, str]] = []
    cur = global_start
    while cur <= global_end:
        win_end = min(cur + pd.DateOffset(years=_IV_YEAR_CHUNK) - pd.Timedelta(days=1), global_end)
        windows.append((cur.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")))
        cur = win_end + pd.Timedelta(days=1)

    batches = [sites[i : i + _IV_BATCH_SIZE] for i in range(0, len(sites), _IV_BATCH_SIZE)]
    total_calls = len(batches) * len(windows)
    logger.info(
        "  IV fetch: %d sites → %d batches × %d time windows = %d API calls",
        len(sites), len(batches), len(windows), total_calls,
    )

    frames: list[pd.DataFrame] = []
    call_num = 0

    for batch in batches:
        for win_start, win_end in windows:
            call_num += 1
            logger.info(
                "  IV call %d/%d: %d sites, %s → %s",
                call_num, total_calls, len(batch), win_start, win_end,
            )
            try:
                raw, _ = nwis.get_iv(
                    sites=batch,
                    parameterCd=PARAM_STAGE,
                    start=win_start,
                    end=win_end,
                )
            except Exception as exc:
                logger.warning("  IV call failed (%s → %s): %s", win_start, win_end, exc)
                continue

            if raw.empty:
                continue

            iv = raw.reset_index()
            iv["date"] = pd.to_datetime(iv["datetime"]).dt.normalize()

            stage_col = next(
                (c for c in iv.columns if PARAM_STAGE in c and "_cd" not in c), None
            )
            if stage_col is None:
                continue

            daily = (
                iv.groupby(["site_no", "date"])[stage_col]
                .mean()
                .reset_index()
                .rename(columns={stage_col: "stage_ft"})
            )
            frames.append(daily)

    if not frames:
        return pd.DataFrame(columns=["site_no", "date", "stage_ft"])

    return pd.concat(frames, ignore_index=True)


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
      1. DV (daily values) — fast, covers gages with pre-computed daily means.
      2. IV fallback — for sites with no DV stage, pulls 15-min instantaneous
         values and computes daily means.  Marked stage_cd = "iv_mean".

    Args:
        site_dates: Mapping of site_no → (begin_date, end_date), e.g.
                    {"01144000": ("1960-10-01", "2025-03-03")}.
        out_path:   Directory where streamflow.parquet will be written.

    Returns:
        DataFrame with columns [site_no, date, discharge_cfs, discharge_cd,
        stage_ft, stage_cd].
    """
    logger.info(
        "Fetching daily discharge + stage for %d gages (per-site date ranges)",
        len(site_dates),
    )

    # ------------------------------------------------------------------
    # Pass 1: DV fetch (discharge + stage)
    # ------------------------------------------------------------------
    frames: list[pd.DataFrame] = []
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
        return pd.DataFrame(
            columns=["site_no", "date", "discharge_cfs", "discharge_cd", "stage_ft", "stage_cd"]
        )

    df = pd.concat(frames, ignore_index=True)

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
        iv_site_dates = {s: site_dates[s] for s in no_dv_stage if s in site_dates}
        iv_stage = _fetch_iv_stage(iv_site_dates)

        if not iv_stage.empty:
            iv_stage = iv_stage.rename(columns={"stage_ft": "_iv_stage"})
            df = df.merge(iv_stage, on=["site_no", "date"], how="left")
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
    # Summary logging
    # ------------------------------------------------------------------
    counts = df.groupby("site_no").size()
    for site, n in counts.items():
        has_stage = df.loc[df["site_no"] == site, "stage_ft"].notna().any()
        stage_src = ""
        if has_stage:
            iv_rows = (
                (df["site_no"] == site) & (df["stage_cd"] == "iv_mean")
            ).sum()
            stage_src = " (iv_mean)" if iv_rows > 0 else " (dv)"
        logger.info(
            "  %s: %d records | stage: %s%s",
            site, n, "yes" if has_stage else "no", stage_src,
        )

    missing = set(site_dates) - set(df["site_no"].unique())
    if missing:
        logger.warning("No data returned for: %s", sorted(missing))

    out_path.mkdir(parents=True, exist_ok=True)
    parquet_file = out_path / "streamflow.parquet"
    df.to_parquet(parquet_file, index=False)
    logger.info("Saved %d rows → %s", len(df), parquet_file)

    return df
