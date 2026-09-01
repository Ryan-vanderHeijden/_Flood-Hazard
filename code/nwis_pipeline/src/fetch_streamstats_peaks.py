"""Fetch published USGS peak-flow statistics from the StreamStats gage service.

Provides the external benchmark for the ungaged-basin work: for each gage,
USGS publishes three flood-frequency estimates per annual exceedance
probability, drawn from the state regression reports.

    PK*AEP    station estimate  — from the at-site record (compare to our LP3 fit)
    RPK*AEP   regression estimate — what the published regional equation predicts
                                    from basin characteristics alone
    WPK*AEP   weighted estimate — station and regression combined per B17C

``RPK*AEP`` is the one that matters here. It is the published answer to exactly
the question this project is asking, computed without using the gage's own
record, so it is the number a new ungaged model has to beat.

**API cost.** The service ignores repeated ``stationIDOrCode`` parameters — it
honours only one station per request, so a per-site loop would cost one request
per gage. Paging the unfiltered collection instead retrieves the whole national
table in ~200 requests: 46 pages of stations plus ~155 pages of statistics.
Requests are issued sequentially with a delay between them, and every page is
checkpointed so an interrupted run resumes rather than refetching.

Writes ``data/metadata/streamstats_peaks.parquet``.

    python code/nwis_pipeline/src/fetch_streamstats_peaks.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path.home() / "data" / "flood_hazard"
CACHE_DIR = DATA_DIR / "streamstats" / "pages"
OUT_PATH = DATA_DIR / "metadata" / "streamstats_peaks.parquet"

BASE = "https://streamstats.usgs.gov/gagestatsservices"
PEAK_FLOW_GROUP = 2

# Deliberately conservative: this is a bulk query against a public USGS
# database, and the whole job is only ~200 requests. One connection, paced.
PAGE_SIZE_STATS = 2000
PAGE_SIZE_STATIONS = 1000
DELAY_S = 1.5
MAX_RETRIES = 4
TIMEOUT_S = 300

# StreamStats encodes annual exceedance probability in the code; map to the
# return period the rest of the project speaks in.
AEP_TO_RP = {
    "50AEP": 2, "20AEP": 5, "10AEP": 10, "4AEP": 25,
    "2AEP": 50, "1AEP": 100, "0_5AEP": 200, "0_2AEP": 500,
}
# Longest prefix first so RPK/WPK are never shadowed by a PK test.
KIND_PREFIX = {"RPK": "regression", "WPK": "weighted", "PK": "station"}

# The paged collection endpoint returns unitTypeID only (the per-station
# endpoint expands it to a unitType object). 44 = ft^3/s, 13 = m^3/s; other
# ids on this group are years-of-record and log-space errors, which the
# regression-code filter already drops.
UNIT_CFS, UNIT_CMS = 44, 13

CFS_PER_CMS = 35.3146667

log = logging.getLogger(__name__)


def _get(path: str, params: dict) -> tuple[list[dict], str | None]:
    """One paged GET with retry and exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                f"{BASE}/{path}", params=params, timeout=TIMEOUT_S,
                headers={"Accept": "application/json"},
            )
            if r.status_code == 200:
                return r.json(), r.headers.get("x-usgswim-messages")
            log.warning("HTTP %s on %s page %s", r.status_code, path, params.get("page"))
        except requests.RequestException as exc:
            log.warning("%s on %s page %s", type(exc).__name__, path, params.get("page"))
        time.sleep(DELAY_S * 2 ** (attempt + 1))
    raise RuntimeError(f"{path} page {params.get('page')} failed after {MAX_RETRIES} tries")


def _total_pages(msg: str | None) -> int | None:
    """Pull the page count out of the service's x-usgswim-messages header."""
    if not msg:
        return None
    try:
        for info in json.loads(msg).get("info", []):
            if "Returning page" in info and " of " in info:
                return int(info.split(" of ")[1].strip().rstrip("."))
    except (ValueError, KeyError):
        pass
    return None


def _fetch_paged(path: str, page_size: int, extra: dict | None = None) -> list[dict]:
    """Page through a collection endpoint, checkpointing each page to disk.

    The page count is written to a sidecar on the first run so a resumed run
    knows where to stop without issuing a probe request.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    extra = extra or {}
    meta_path = CACHE_DIR / f"{path}_meta.json"
    total = json.loads(meta_path.read_text())["total_pages"] if meta_path.exists() else None

    rows: list[dict] = []
    page = 1
    while total is None or page <= total:
        cache = CACHE_DIR / f"{path}_{page:04d}.json"
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            data, msg = _get(path, {"page": page, "pageCount": page_size, **extra})
            cache.write_text(json.dumps(data))
            if total is None:
                total = _total_pages(msg)
                if total is not None:
                    meta_path.write_text(json.dumps({"total_pages": total}))
            log.info("%s page %d/%s  (+%d rows)", path, page, total or "?", len(data))
            time.sleep(DELAY_S)

        rows.extend(data)
        if not data:
            break
        page += 1

    log.info("%s: %d rows across %d pages", path, len(rows), page - 1)
    return rows


def fetch() -> tuple[pd.DataFrame, pd.DataFrame]:
    stations = pd.DataFrame(_fetch_paged("stations", PAGE_SIZE_STATIONS))
    stats = pd.DataFrame(
        _fetch_paged("statistics", PAGE_SIZE_STATS,
                     {"statisticGroups": str(PEAK_FLOW_GROUP)})
    )
    return stations, stats


def reshape(stations: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long statistics table to one row per gage."""
    st = stations[["id", "code", "isRegulated", "regionID"]].rename(
        columns={"id": "stationID", "code": "site_no",
                 "isRegulated": "ss_is_regulated", "regionID": "ss_region_id"})

    stats = stats.copy()
    stats["rt_code"] = stats["regressionType"].apply(
        lambda d: d.get("code") if isinstance(d, dict) else None)

    # Split e.g. RPK10AEP into kind=regression, return period=10.
    def split(code: str) -> tuple[str, int] | tuple[None, None]:
        for prefix, kind in KIND_PREFIX.items():
            if code and code.startswith(prefix):
                tail = code[len(prefix):]
                if tail in AEP_TO_RP:
                    return kind, AEP_TO_RP[tail]
        return None, None

    parsed = stats["rt_code"].map(split)
    stats["kind"] = [p[0] for p in parsed]
    stats["rp"] = [p[1] for p in parsed]
    stats = stats.dropna(subset=["kind", "rp"])

    # Normalise to cfs; the service carries a few metric records.
    metric = stats["unitTypeID"].eq(UNIT_CMS)
    stats.loc[metric, "value"] = stats.loc[metric, "value"] * CFS_PER_CMS
    if metric.any():
        log.info("Converted %d metric records to cfs", int(metric.sum()))
    unexpected = ~stats["unitTypeID"].isin([UNIT_CFS, UNIT_CMS])
    if unexpected.any():
        log.warning("Dropping %d records with unexpected unit ids: %s",
                    int(unexpected.sum()),
                    sorted(stats.loc[unexpected, "unitTypeID"].unique()))
        stats = stats[~unexpected]

    stats["col"] = ("ss_" + stats["kind"] + "_q"
                    + stats["rp"].astype(int).astype(str) + "_cfs")

    wide = (stats.pivot_table(index="stationID", columns="col",
                              values="value", aggfunc="first")
            .reset_index())
    out = st.merge(wide, on="stationID", how="right")
    out["site_no"] = out["site_no"].astype(str).str.strip()

    # Keep only USGS-style numeric station codes; the service also carries
    # Environment Canada and other agency gages.
    out = out[out["site_no"].str.fullmatch(r"\d{8,15}")]
    out["site_no"] = out["site_no"].str.zfill(8)
    return out.drop_duplicates("site_no").reset_index(drop=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    stations, stats = fetch()
    log.info("Fetched %d stations, %d peak-flow statistics", len(stations), len(stats))

    df = reshape(stations, stats)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    log.info("Wrote %s  (%d gages x %d columns)", OUT_PATH, *df.shape)

    ffa_path = DATA_DIR / "ffa" / "flood_frequency.parquet"
    if ffa_path.exists():
        ffa = pd.read_parquet(
            ffa_path, columns=["site_no", "record_ok", "degenerate_fit"])
        qc = ffa[ffa.record_ok & ~ffa.degenerate_fit].site_no.astype(str).str.zfill(8)
        have = df.set_index("site_no")
        matched = qc[qc.isin(have.index)]
        log.info("QC-passed FFA sites matched: %d/%d (%.1f%%)",
                 len(matched), len(qc), 100 * len(matched) / len(qc))
        for c in ["ss_regression_q10_cfs", "ss_station_q10_cfs", "ss_weighted_q10_cfs"]:
            if c in have.columns:
                n = have.loc[matched, c].notna().sum()
                log.info("  %-26s non-null for %d matched sites", c, n)


if __name__ == "__main__":
    main()
