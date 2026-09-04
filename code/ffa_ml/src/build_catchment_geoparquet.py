from __future__ import annotations

"""
Download NHDPlusV2 simplified catchment polygons and join the model's CONUS
return-period predictions to them, producing a mapped geoparquet.

The full-resolution national catchment layer ships only inside EPA's 7.8 GB
Seamless Geodatabase; the *simplified* catchments (the ``SimplifiedCatchments``
extension) are ~824 MB across the 21 CONUS Vector Processing Units (VPUs) and
render far better at national scale, so they are used here.  Each VPU shapefile
keys on ``FEATUREID`` (= NHDPlus ``COMID``), so the join to
``ml/conus_predictions.parquet`` is a direct key merge — no gridcode crosswalk.

Processing is per-VPU (one region in memory at a time) and resumable: downloads,
extractions and per-VPU geoparquet parts are all cached.  The parts are then
stream-merged into one geoparquet with a pyarrow ``ParquetWriter`` so the full
2.6M-polygon layer is never materialised at once.

Output ``ml/conus_catchments_predictions.parquet`` (GeoParquet) — plus per-VPU
parts under ``ml/catchment_parts/``.

Example
-------
    python build_catchment_geoparquet.py
    python build_catchment_geoparquet.py --vpus MA_02 CA_18   # a subset
"""

import argparse
import logging
import os
import re
import zipfile  # noqa: F401  (py7zr handles .7z; kept for parity/debugging)
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import geopandas as gpd
import pandas as pd
import py7zr
import pyarrow.parquet as pq
import pyogrio
import requests

logger = logging.getLogger(__name__)

S3_BASE = "https://dmap-data-commons-ow.s3.amazonaws.com/"
PREFIX = "NHDPlusV21/Data/Extensions/SimplifiedCatchments/"
NONCONUS = ("_HI_20_", "_CI_21_", "_PI_22")
_UA = {"User-Agent": "Mozilla/5.0 (research; flood-hazard)"}

DATA_DIR = Path.home() / "data" / "flood_hazard"
GEOM_DIR = DATA_DIR / "nhdplus" / "simplified_catchments"
ML_DIR = DATA_DIR / "ml"
PRED_PATH = ML_DIR / "conus_predictions.parquet"
PARTS_DIR = ML_DIR / "catchment_parts"
OUT_PATH = ML_DIR / "conus_catchments_predictions.parquet"

_FEATUREID = "FEATUREID"
_VPU_RE = re.compile(r"NHDPlusV21_([A-Z]+_[0-9A-Z]+)_NHDPlusCatchment2")


def list_conus_vpus() -> dict[str, str]:
    """Return {vpu: s3_key} for the latest CONUS simplified-catchment file per VPU."""
    session = requests.Session()
    session.headers.update(_UA)
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        text = session.get(S3_BASE, params=params, timeout=60).text
        keys += re.findall(r"<Key>(.*?)</Key>", text)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", text)
        if not m:
            break
        token = m.group(1)

    out: dict[str, str] = {}
    for k in keys:
        if not k.endswith(".7z") or "NHDPlusCatchment2" not in k:
            continue
        if any(n in k for n in NONCONUS):
            continue
        vpu = _VPU_RE.search(k).group(1)
        if vpu not in out or k > out[vpu]:  # lexical max ~ latest version suffix
            out[vpu] = k
    return dict(sorted(out.items()))


def _download(session: requests.Session, key: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("  cached download %s", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(S3_BASE + key, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk_size=1 << 20):
                fh.write(block)
        tmp.rename(dest)
    logger.info("  downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def _extract_shp(archive: Path, workdir: Path) -> Path:
    """Extract a .7z and return the catchment shapefile (the one with FEATUREID)."""
    workdir.mkdir(parents=True, exist_ok=True)
    if not any(workdir.rglob("*.shp")):
        with py7zr.SevenZipFile(archive, "r") as z:
            z.extractall(path=workdir)
    for shp in workdir.rglob("*.shp"):
        try:
            cols = pyogrio.read_info(shp)["fields"]
        except Exception:  # noqa: BLE001
            continue
        if _FEATUREID in cols:
            return shp
    raise FileNotFoundError(f"no catchment shapefile with {_FEATUREID} in {workdir}")


def _load_predictions(pred_path: Path) -> pd.DataFrame:
    """Load COMID-keyed predictions (float32) for the join."""
    df = pd.read_parquet(pred_path)
    floatcols = df.select_dtypes("float64").columns
    df[floatcols] = df[floatcols].astype("float32")
    return df.set_index("COMID")


def build_part(vpu: str, key: str, preds: pd.DataFrame, session: requests.Session) -> Path:
    """Download+extract one VPU, join predictions, write and return its geoparquet part."""
    part = PARTS_DIR / f"catchments_{vpu}.parquet"
    if part.exists():
        logger.info("[%s] cached part", vpu)
        return part

    archive = GEOM_DIR / Path(key).name
    _download(session, key, archive)
    shp = _extract_shp(archive, GEOM_DIR / vpu)

    gdf = pyogrio.read_dataframe(shp, columns=[_FEATUREID])
    gdf = gdf.rename(columns={_FEATUREID: "COMID"})
    gdf["COMID"] = pd.to_numeric(gdf["COMID"], errors="coerce").astype("int64")
    gdf = gdf.join(preds, on="COMID")  # left join keeps every catchment
    matched = gdf["q10_cfs"].notna().sum()
    logger.info("[%s] %d catchments, %d joined to a prediction", vpu, len(gdf), int(matched))

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(part, index=False)
    return part


def merge_parts(parts: list[Path], out_path: Path) -> None:
    """Stream-merge per-VPU geoparquet parts into one file (bounded memory)."""
    writer: pq.ParquetWriter | None = None
    total = 0
    for p in parts:
        table = pq.read_table(p)  # retains the GeoParquet 'geo' schema metadata
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table.cast(writer.schema))
        total += table.num_rows
    if writer is not None:
        writer.close()
    logger.info("Merged %d parts → %s (%d catchments)", len(parts), out_path, total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vpus", nargs="*", default=None, help="subset of VPU codes")
    parser.add_argument("--pred", type=Path, default=PRED_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--no-merge", action="store_true", help="write parts only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    vpu_keys = list_conus_vpus()
    if args.vpus:
        vpu_keys = {v: k for v, k in vpu_keys.items() if v in set(args.vpus)}
    logger.info("Processing %d VPU(s): %s", len(vpu_keys), list(vpu_keys))

    preds = _load_predictions(args.pred)
    session = requests.Session()
    session.headers.update(_UA)

    parts = [build_part(vpu, key, preds, session) for vpu, key in vpu_keys.items()]

    if not args.no_merge:
        merge_parts(parts, args.out)


if __name__ == "__main__":
    main()
