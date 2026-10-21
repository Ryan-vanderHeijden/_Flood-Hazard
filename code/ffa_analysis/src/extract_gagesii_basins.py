"""Extract GAGES-II drainage-basin polygons for the QC gages.

GAGES-II (Falcone 2011) ships its basin boundaries as one shapefile per
aggregated ecoregion, plus one for all reference gages. Every QC gage is in
GAGES-II, and these are the same boundaries its basin characteristics
(RUNAVE7100, STOR_NID_2009, and so the regulation DOR) were computed on.

Each shapefile is read on its own with an attribute filter, so only QC gages
are loaded (the full set is ~1.4 GB), and polygons are simplified to
``SIMPLIFY_M`` metres, far below anything that matters for locating dams.

Writes ``data/metadata/gage_basins.parquet`` (site_no, area_km2, geometry; in
the GAGES-II Albers projection).

    python code/ffa_analysis/src/extract_gagesii_basins.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

DATA_DIR = Path.home() / "data" / "flood_hazard"
GAGESII_BASINS = Path.home() / "data" / "usgs" / "GAGES_2" / "boundaries-shapefiles-by-aggeco"
OUT_PATH = DATA_DIR / "metadata" / "gage_basins.parquet"

SIMPLIFY_M = 100

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--src", type=Path, default=GAGESII_BASINS)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    ffa = pd.read_parquet(args.data_dir / "ffa" / "flood_frequency.parquet",
                          columns=["site_no", "record_ok", "degenerate_fit"])
    qc = set(ffa.loc[ffa["record_ok"] & ~ffa["degenerate_fit"], "site_no"])
    where = "GAGE_ID IN ({})".format(",".join(f"'{s}'" for s in sorted(qc)))

    parts = []
    for shp in sorted(args.src.glob("bas_*.shp")):
        g = pyogrio.read_dataframe(shp, columns=["GAGE_ID"], where=where)
        if g.empty:
            continue
        g["geometry"] = g.geometry.simplify(SIMPLIFY_M)
        parts.append(g.rename(columns={"GAGE_ID": "site_no"}))
        log.info("%-28s %4d QC basins", shp.name, len(g))

    basins = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    basins = basins.drop_duplicates("site_no")   # ref and non-ref files do not overlap, but be safe
    basins["area_km2"] = basins.area / 1e6
    missing = qc - set(basins["site_no"])
    log.info("%d of %d QC gages have a basin; missing: %s",
             len(basins), len(qc), sorted(missing)[:10])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    basins[["site_no", "area_km2", "geometry"]].to_parquet(args.out)
    log.info("Saved -> %s", args.out)


if __name__ == "__main__":
    main()
