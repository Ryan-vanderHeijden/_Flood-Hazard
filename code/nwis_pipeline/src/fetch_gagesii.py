"""Join GAGES-II basin characteristics onto the project gage set.

GAGES-II (Geospatial Attributes of Gages for Evaluating Streamflow, version II;
Falcone, 2011) supplies ~300 pre-computed basin characteristics for 9,067
conterminous-US streamgages, keyed by USGS station ID. It fills the gap the
current predictor set leaves open: climate forcing (precipitation, snow
fraction, PET), soils, baseflow index, and reservoir storage.

Downloads the USGS archive on first run, caches it under
``data/gagesii/``, and writes the merged table to
``data/metadata/gagesii_basin_chars.parquet``.

Standalone — not called by run_pipeline.py.

    python code/nwis_pipeline/src/fetch_gagesii.py
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path.home() / "data" / "flood_hazard"
GAGESII_DIR = DATA_DIR / "gagesii"
CSV_DIR = GAGESII_DIR / "csv"
OUT_PATH = DATA_DIR / "metadata" / "gagesii_basin_chars.parquet"

SOURCE_URL = "https://water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip"

log = logging.getLogger(__name__)

# Tables to merge, and the columns taken from each. None means "all columns".
# Selection favours variables with a plausible mechanistic link to flood
# frequency; the full archive stays on disk for anything else.
TABLES: dict[str, list[str] | None] = {
    "conterm_basinid": [
        "DRAIN_SQKM", "HUC02", "LAT_GAGE", "LNG_GAGE", "STATE",
    ],
    "conterm_bas_classif": [
        "CLASS", "AGGECOREGION", "HYDRO_DISTURB_INDX",
    ],
    "conterm_bas_morph": [
        "BAS_COMPACTNESS",
    ],
    # Climate — the predictors missing from the current feature set entirely.
    "conterm_climate": [
        "PPTAVG_BASIN", "T_AVG_BASIN", "T_MAX_BASIN", "T_MIN_BASIN",
        "RH_BASIN", "PET", "SNOW_PCT_PRECIP", "PRECIP_SEAS_IND",
        "WD_BASIN", "WDMAX_BASIN", "WDMIN_BASIN",
        "JAN_PPT7100_CM", "FEB_PPT7100_CM", "MAR_PPT7100_CM", "APR_PPT7100_CM",
        "MAY_PPT7100_CM", "JUN_PPT7100_CM", "JUL_PPT7100_CM", "AUG_PPT7100_CM",
        "SEP_PPT7100_CM", "OCT_PPT7100_CM", "NOV_PPT7100_CM", "DEC_PPT7100_CM",
    ],
    "conterm_topo": [
        "ELEV_MEAN_M_BASIN", "ELEV_MAX_M_BASIN", "ELEV_MIN_M_BASIN",
        "ELEV_STD_M_BASIN", "ELEV_SITE_M", "RRMEAN", "RRMEDIAN",
        "SLOPE_PCT", "ASPECT_NORTHNESS", "ASPECT_EASTNESS",
    ],
    "conterm_soils": [
        "HGA", "HGB", "HGC", "HGD", "HGVAR",
        "AWCAVE", "PERMAVE", "BDAVE", "OMAVE", "WTDEPAVE", "ROCKDEPAVE",
        "CLAYAVE", "SILTAVE", "SANDAVE", "KFACT_UP", "RFACT",
    ],
    "conterm_hydro": [
        "STREAMS_KM_SQ_KM", "STRAHLER_MAX", "MAINSTEM_SINUOUSITY",
        "ARTIFPATH_PCT", "HIRES_LENTIC_PCT", "BFI_AVE",
        "PERDUN", "PERHOR", "TOPWET", "CONTACT", "RUNAVE7100",
        "WB5100_ANN_MM",
        "PCT_1ST_ORDER", "PCT_2ND_ORDER", "PCT_3RD_ORDER",
        "PCT_4TH_ORDER", "PCT_5TH_ORDER", "PCT_6TH_ORDER_OR_MORE",
    ],
    "conterm_hydromod_dams": [
        "NDAMS_2009", "DDENS_2009", "STOR_NID_2009", "STOR_NOR_2009",
        "MAJ_NDAMS_2009", "RAW_DIS_NEAREST_DAM", "RAW_DIS_NEAREST_MAJ_DAM",
    ],
    "conterm_lc06_basin": [
        "DEVNLCD06", "FORESTNLCD06", "PLANTNLCD06", "WATERNLCD06",
        "SNOWICENLCD06", "BARRENNLCD06", "SHRUBNLCD06", "GRASSNLCD06",
        "PASTURENLCD06", "CROPSNLCD06", "WOODYWETNLCD06", "EMERGWETNLCD06",
    ],
    "conterm_landscape_pat": [
        "FRAGUN_BASIN", "HIRES_LENTIC_DENS", "HIRES_LENTIC_MEANSIZ",
    ],
    "conterm_pop_infrastr": [
        "PDEN_2000_BLOCK", "ROADS_KM_SQ_KM", "RD_STR_INTERS", "IMPNLCD06",
    ],
    "conterm_geology": [
        "GEOL_REEDBUSH_DOM", "GEOL_REEDBUSH_DOM_PCT",
    ],
}

# Columns whose absence is tolerated rather than warned about.
OPTIONAL_COLS: set[str] = set()


def download() -> None:
    """Fetch and unpack the GAGES-II archive if the CSVs are not cached."""
    if (CSV_DIR / "conterm_basinid.txt").exists():
        log.info("GAGES-II CSVs already cached at %s", CSV_DIR)
        return

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading GAGES-II from %s", SOURCE_URL)
    resp = requests.get(SOURCE_URL, timeout=600)
    resp.raise_for_status()

    outer = zipfile.ZipFile(io.BytesIO(resp.content))
    outer.extractall(GAGESII_DIR)

    inner_path = GAGESII_DIR / "spreadsheets-in-csv-format.zip"
    with zipfile.ZipFile(inner_path) as inner:
        inner.extractall(CSV_DIR)
    log.info("Extracted %d CSV tables", len(list(CSV_DIR.glob('*.txt'))))


def _read(table: str) -> pd.DataFrame:
    """Read one GAGES-II CSV with STAID normalised to a zero-padded key."""
    df = pd.read_csv(
        CSV_DIR / f"{table}.txt", dtype={"STAID": str}, encoding="latin-1",
    )
    df["STAID"] = df["STAID"].str.strip().str.zfill(8)
    return df


def build() -> pd.DataFrame:
    """Merge the selected GAGES-II tables into one site-keyed frame."""
    merged: pd.DataFrame | None = None

    for table, cols in TABLES.items():
        df = _read(table)
        if cols is not None:
            available = [c for c in cols if c in df.columns]
            missing = set(cols) - set(available) - OPTIONAL_COLS
            if missing:
                log.warning("%s: columns not found, skipped: %s", table, sorted(missing))
            df = df[["STAID"] + available]
        merged = df if merged is None else merged.merge(df, on="STAID", how="outer")
        log.info("%-28s +%3d cols  (%d rows)", table, df.shape[1] - 1, len(df))

    assert merged is not None
    merged = merged.rename(columns={"STAID": "site_no"})

    # Match the rest of the project: site_no is the plain NWIS string.
    merged["site_no"] = merged["site_no"].astype(str)
    return merged


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
    )
    download()
    df = build()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    log.info("Wrote %s  (%d sites x %d columns)", OUT_PATH, *df.shape)

    # Coverage against the FFA site set, if it exists yet.
    ffa_path = DATA_DIR / "ffa" / "flood_frequency.parquet"
    if ffa_path.exists():
        ffa = pd.read_parquet(ffa_path, columns=["site_no", "record_ok", "degenerate_fit"])
        qc = ffa[ffa.record_ok & ~ffa.degenerate_fit].site_no.astype(str).str.zfill(8)
        have = set(df.site_no)
        log.info(
            "Coverage: %d/%d QC-passed FFA sites matched (%.1f%%)",
            qc.isin(have).sum(), len(qc), 100 * qc.isin(have).mean(),
        )


if __name__ == "__main__":
    main()
