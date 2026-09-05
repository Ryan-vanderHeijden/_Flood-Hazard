from __future__ import annotations

"""
Fetch NHDPlus v2.1 catchment attributes for the return-period flood-flow model.

The training targets are at-site Log-Pearson III flood quantiles at USGS gauges;
the predictors are catchment characteristics that must be available both at the
gauges (for training) and at every CONUS reach (for inference).  The canonical,
CONUS-complete source keyed to NHDPlus COMID is:

    Wieczorek, M.E., Jackson, S.E., and Schwarz, G.E., 2018 (rev. 2019),
    "Select Attributes for NHDPlus Version 2.1 Reach Catchments and Modified
    Network Routed Upstream Watersheds for the Conterminous United States",
    U.S. Geological Survey data release, https://doi.org/10.5066/F7765D7V
    (ScienceBase item 5669a79ee4b08895842a1d47).

Each themed child item publishes three COMID-keyed parquet files:
  * ``*_cat.parquet`` — the local reach catchment value,
  * ``*_acc.parquet`` — divergence-routed accumulation,
  * ``*_tot.parquet`` — total upstream (routed) accumulation.

For flood-frequency prediction the basin-integrated ``_tot`` (total upstream)
values are the physically relevant predictors, so this module downloads the
``_tot`` parquet for a curated set of flood-relevant themes, caches the raw
files, and merges them on ``COMID`` into a single wide attribute table.

Downloads use the ScienceBase REST file endpoint over plain HTTPS (a browser
User-Agent is required); no ``sciencebase`` client is needed.

Example
-------
    python fetch_nhdplus_attributes.py                       # fetch all curated themes
    python fetch_nhdplus_attributes.py --themes basin ppt    # a subset
    python fetch_nhdplus_attributes.py --list                # print the theme catalogue
"""

import argparse
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

logger = logging.getLogger(__name__)

_SB_FILE_URL = "https://www.sciencebase.gov/catalog/file/get/{item_id}"
_SB_ITEM_URL = "https://www.sciencebase.gov/catalog/item/{item_id}"
_USER_AGENT = "Mozilla/5.0 (research; flood-hazard ungauged FFA)"
_COMID = "COMID"

DATA_DIR = Path.home() / "data" / "flood_hazard" / "nhdplus"
RAW_DIR = DATA_DIR / "wieczorek_tot_raw"
OUT_PATH = DATA_DIR / "comid_attributes.parquet"

# Curated flood-relevant themes from the Wieczorek "Select Attributes" release,
# mapping a short local name -> (ScienceBase item id, columns to keep or None
# for every TOT_ column in the file).  Item ids were confirmed against the
# published catalogue; every listed item exposes a "<item_id>_tot.parquet".
THEMES: dict[str, tuple[str, list[str] | None]] = {
    # Topography / network geometry — includes TOT_BASIN_AREA (drainage area).
    "basin": (
        "57976a0ce4b021cadec97890",
        [
            "TOT_BASIN_AREA", "TOT_BASIN_SLOPE", "TOT_ELEV_MEAN", "TOT_ELEV_MIN",
            "TOT_ELEV_MAX", "TOT_STREAM_SLOPE", "TOT_STREAM_LENGTH", "TOT_STRM_DENS",
        ],
    ),
    # Climate.
    "ppt": ("573b70a7e4b0dae0d5e3ae85", None),          # mean annual precip (PPT7100_ANN)
    "pet": ("56f96ed1e4b0a6037df06a2d", None),          # PRISM PET 1971-2000
    "tmean": ("57054bf2e4b0d4e2b756d364", None),        # mean annual temperature
    "tmax": ("57054aebe4b0d4e2b756d1fc", None),         # mean annual max temperature
    "tmin": ("57054c9be4b0d4e2b756d465", None),         # mean annual min temperature
    "rh": ("57054a24e4b0d4e2b756d0e7", None),           # relative humidity
    "snow": ("57053dc5e4b0d4e2b756c117", None),         # snow as % of precip (PRSNOW)
    # Hydrology / water balance.
    "bfi": ("5669a8e3e4b08895842a1d4f", None),          # base flow index
    "runoff": ("56fd5bd0e4b0c07cbfa40473", None),       # McCabe-Wolock mean annual runoff
    "recharge": ("56f97577e4b0a6037df06b5a", None),     # natural groundwater recharge
    "twi": ("56f97be4e4b0a6037df06b70", None),          # topographic wetness index
    "wtdepth": ("56f97456e4b0a6037df06b50", None),      # water-table depth below surface
    "contact": ("56f96fc5e4b0a6037df06b12", None),      # subsurface contact time
    "ieof": ("56f974e2e4b0a6037df06b55", None),         # infiltration-excess overland flow %
    "satof": ("56f97acbe4b0a6037df06b6a", None),        # saturation overland flow %
    # Soils.
    "hsg": ("5728d93be4b0b13d3918a99f", None),          # STATSGO2 hydrologic soil groups
    "soiltext": ("5728dd46e4b0b13d3918a9a7", None),     # STATSGO texture / AWC / permeability
    # Land cover and anthropogenic modification.
    "nlcd11": ("5761bad4e4b04f417c2d30c5", None),       # NLCD 2011 land-cover fractions
    # NID dams: the theme carries every decade 1930-2013; keep only the latest
    # snapshot as a static predictor of upstream regulation.
    "nid": (
        "58c301f2e4b0f37a93ed915a",
        ["TOT_NDAMS2013", "TOT_NID_STORAGE2013", "TOT_NORM_STORAGE2013", "TOT_MAJOR2013"],
    ),
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _USER_AGENT})
    return s


def _tot_source(session: requests.Session, item_id: str) -> tuple[str, str]:
    """Locate the total-upstream attribute file for a ScienceBase item.

    Returns ``("parquet", name)`` when a ready ``*_tot.parquet`` exists, else
    ``("zip", name)`` for a ``*_TOT_*.zip`` bundle of per-VPU CSVs.  Raises when
    neither is present.
    """
    r = session.get(_SB_ITEM_URL.format(item_id=item_id), params={"format": "json"}, timeout=60)
    r.raise_for_status()
    names = [f["name"] for f in r.json().get("files", [])]
    tots = [n for n in names if n.endswith("_tot.parquet")]
    if tots:
        return "parquet", tots[0]
    zips = [n for n in names if n.lower().endswith(".zip") and "tot" in n.lower()]
    if zips:
        return "zip", zips[0]
    raise FileNotFoundError(f"item {item_id} has no TOT parquet or zip (files: {names[:6]}…)")


def _download(session: requests.Session, item_id: str, fname: str) -> bytes:
    r = session.get(_SB_FILE_URL.format(item_id=item_id), params={"name": fname}, timeout=600)
    r.raise_for_status()
    return r.content


def _read_tot_zip(blob: bytes) -> pd.DataFrame:
    """Read and concatenate the CSV member(s) of a Wieczorek ``*_TOT_*.zip`` bundle."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
        frames = [pd.read_csv(zf.open(n), low_memory=False) for n in csvs]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    df.columns = [c.upper() if c.upper() == _COMID else c for c in df.columns]
    return df


def _fetch_theme(
    session: requests.Session,
    name: str,
    item_id: str,
    keep: list[str] | None,
    *,
    refetch: bool = False,
) -> pd.DataFrame:
    """Download (or load from cache) one theme's total-upstream attribute table.

    Parameters
    ----------
    session : requests.Session
        Session carrying the browser User-Agent required by ScienceBase.
    name : str
        Short local theme name (used for the cache filename).
    item_id : str
        ScienceBase item id of the theme.
    keep : list[str] | None
        TOT_ columns to retain; ``None`` keeps every ``TOT_`` column.  The
        ``_NODATA`` companion flags are always dropped (missingness is encoded
        as NaN downstream by ``build_features``).
    refetch : bool
        Re-download even if a cached copy exists.

    Returns
    -------
    pd.DataFrame
        Indexed by ``COMID`` with the requested TOT_ columns.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{name}_{item_id}_tot.parquet"

    if cache.exists() and not refetch:
        logger.info("  [%s] cache hit → %s", name, cache.name)
        df = pd.read_parquet(cache)
    else:
        kind, fname = _tot_source(session, item_id)
        logger.info("  [%s] downloading %s (%s) …", name, fname, kind)
        blob = _download(session, item_id, fname)
        if kind == "parquet":
            df = pq.read_table(io.BytesIO(blob)).to_pandas()
        else:
            df = _read_tot_zip(blob)
        df.to_parquet(cache, index=False)
        logger.info("  [%s] %d rows, %d cols cached", name, len(df), df.shape[1])

    if _COMID not in df.columns:
        raise KeyError(f"theme {name} ({item_id}) has no {_COMID} column")

    tot_cols = [c for c in df.columns if c.startswith("TOT_") and not c.endswith("_NODATA")]
    if keep is not None:
        missing = [c for c in keep if c not in df.columns]
        if missing:
            logger.warning("  [%s] requested columns absent, skipping: %s", name, missing)
        tot_cols = [c for c in keep if c in df.columns]

    out = df[[_COMID, *tot_cols]].copy()
    del df
    out[_COMID] = pd.to_numeric(out[_COMID], errors="coerce").astype("Int32")
    out = out.dropna(subset=[_COMID]).drop_duplicates(subset=[_COMID]).set_index(_COMID)
    # Downcast value columns to float32 — halves the footprint of the wide
    # 2.7M-row merge without meaningfully affecting attribute precision.
    floatcols = out.select_dtypes("float64").columns
    out[floatcols] = out[floatcols].astype("float32")
    return out


def fetch_nhdplus_attributes(
    themes: list[str] | None = None,
    out_path: Path = OUT_PATH,
    *,
    refetch: bool = False,
) -> pd.DataFrame:
    """Download and merge curated NHDPlus TOT_ attribute themes into one table.

    Parameters
    ----------
    themes : list[str] | None
        Subset of :data:`THEMES` keys to fetch; ``None`` fetches all of them.
    out_path : Path
        Destination parquet for the merged, COMID-keyed attribute table.
    refetch : bool
        Force re-download of every theme rather than using the raw cache.

    Returns
    -------
    pd.DataFrame
        Wide attribute table indexed by ``COMID``, one column per TOT_ attribute.
    """
    selected = list(THEMES) if themes is None else themes
    unknown = [t for t in selected if t not in THEMES]
    if unknown:
        raise ValueError(f"unknown themes {unknown}; valid: {sorted(THEMES)}")

    session = _session()
    merged: pd.DataFrame | None = None
    failed: list[str] = []
    for name in selected:
        item_id, keep = THEMES[name]
        try:
            part = _fetch_theme(session, name, item_id, keep, refetch=refetch)
        except Exception as exc:  # noqa: BLE001 — one bad theme shouldn't sink the run
            logger.error("  [%s] FAILED (%s): %s", name, item_id, exc)
            failed.append(name)
            continue
        merged = part if merged is None else merged.join(part, how="outer")

    if merged is None:
        raise RuntimeError(f"no themes fetched successfully (failed: {failed})")
    if failed:
        logger.warning("Skipped %d theme(s): %s", len(failed), failed)
    merged = merged.sort_index()
    logger.info("Merged attribute table: %d COMIDs × %d attributes", *merged.shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.reset_index().to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--themes", nargs="*", default=None, help="subset of theme names")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="output parquet path")
    parser.add_argument("--refetch", action="store_true", help="ignore the raw cache")
    parser.add_argument("--list", action="store_true", help="list theme catalogue and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.list:
        for name, (item_id, keep) in THEMES.items():
            cols = "all TOT_" if keep is None else ", ".join(keep)
            print(f"{name:10s} {item_id}  {cols}")
        return

    fetch_nhdplus_attributes(args.themes, args.out, refetch=args.refetch)


if __name__ == "__main__":
    main()
