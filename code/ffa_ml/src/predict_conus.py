from __future__ import annotations

"""
Apply the trained return-period flood model to every CONUS NHDPlus reach.

Loads the fitted index-flood component models, the feature spec and the conformal
offsets, streams the ~3M-row COMID attribute table in chunks (bounded memory),
predicts monotone log10 quantiles, attaches conformal lower/upper bands, and
writes a COMID-keyed prediction table:

    COMID, q{rp}_cfs, q{rp}_lo_cfs, q{rp}_hi_cfs   for rp in 2..500,
    plus TOT_BASIN_AREA (context) and has_upstream_dam (regulation flag).

Predictions on reaches with upstream regulation carry ``has_upstream_dam=True``:
the model was trained on unregulated gauges, so those values estimate *natural*
flood potential and should be read as such.

The output is keyed to COMID; NHDPlusV2 catchment geometry (a large, separate
download) can be joined at delivery time to produce a geoparquet.

Example
-------
    python predict_conus.py
    python predict_conus.py --chunksize 250000
"""

import argparse
import json
import logging
import os
import resource
from pathlib import Path

# Keep native thread pools small *before* importing numpy/lightgbm so CONUS
# inference stays within a predictable, bounded memory footprint.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_features import ATTR_PATH, iter_conus_features, load_spec
from train import BASE_COL, INCREMENTS, MODEL_DIR, RPS, predict_quantiles
from uncertainty import OFFSETS_PATH

logger = logging.getLogger(__name__)

ML_DIR = Path.home() / "data" / "flood_hazard" / "ml"
OUT_PATH = ML_DIR / "conus_predictions.parquet"


def _limit_address_space(gb: float) -> None:
    """Cap this process's virtual memory so a runaway dies cleanly instead of OOM-killing the box."""
    if gb <= 0:
        return
    nbytes = int(gb * 1024**3)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard = nbytes if hard == resource.RLIM_INFINITY else min(nbytes, hard)
    resource.setrlimit(resource.RLIMIT_AS, (nbytes, new_hard))
    logger.info("Address-space cap set to %.1f GB", gb)


def load_models(model_dir: Path = MODEL_DIR, num_threads: int = 2) -> dict:
    """Load the index-flood component models, pinning each booster to few threads."""
    names = [BASE_COL] + [f"d_{rp}" for rp, _ in INCREMENTS]
    models = {}
    for name in names:
        m = joblib.load(model_dir / f"lgbm_{name}.joblib")
        m.set_params(n_jobs=num_threads)
        models[name] = m
    return models


def _predict_chunk(chunk: pd.DataFrame, models: dict, features: list[str], offsets: dict) -> pd.DataFrame:
    """Predict monotone quantiles + conformal bands (cfs) for one COMID feature chunk."""
    log_q = predict_quantiles(models, chunk[features])  # monotone log10 quantiles
    out = pd.DataFrame({"COMID": chunk["COMID"].to_numpy()})
    off = offsets["offsets"]
    for rp in RPS:
        lg = log_q[f"log_q{rp}"].to_numpy()
        out[f"q{rp}_cfs"] = np.power(10.0, lg)
        out[f"q{rp}_lo_cfs"] = np.power(10.0, lg + off[f"q{rp}"]["lo"])
        out[f"q{rp}_hi_cfs"] = np.power(10.0, lg + off[f"q{rp}"]["hi"])
    out["TOT_BASIN_AREA"] = chunk["TOT_BASIN_AREA"].to_numpy() if "TOT_BASIN_AREA" in chunk else np.nan
    if "TOT_NDAMS2013" in chunk:
        out["has_upstream_dam"] = (chunk["TOT_NDAMS2013"].fillna(0).to_numpy() > 0)
    return out


def predict_conus(
    attr_path: Path = ATTR_PATH,
    model_dir: Path = MODEL_DIR,
    offsets_path: Path = OFFSETS_PATH,
    out_path: Path = OUT_PATH,
    chunksize: int = 200_000,
    mem_cap_gb: float = 6.0,
) -> Path:
    """Run CONUS-wide inference and write the COMID-keyed prediction parquet."""
    _limit_address_space(mem_cap_gb)
    spec = load_spec()
    features = spec["features"]
    models = load_models(model_dir)
    offsets = json.loads(Path(offsets_path).read_text())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    n_total = 0
    try:
        for i, chunk in enumerate(iter_conus_features(spec, attr_path, chunksize), 1):
            # A COMID with no usable attributes at all can't be scored; skip it.
            usable = chunk[features].notna().any(axis=1)
            chunk = chunk[usable]
            if chunk.empty:
                continue
            preds = _predict_chunk(chunk, models, features, offsets)
            table = pa.Table.from_pandas(preds, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            n_total += len(preds)
            logger.info("  chunk %d: wrote %d rows (cumulative %d)", i, len(preds), n_total)
    finally:
        if writer is not None:
            writer.close()

    logger.info("Wrote %s (%d COMIDs)", out_path, n_total)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attr", type=Path, default=ATTR_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--offsets", type=Path, default=OFFSETS_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--mem-cap-gb", type=float, default=6.0, help="0 disables the cap")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    predict_conus(args.attr, args.model_dir, args.offsets, args.out, args.chunksize, args.mem_cap_gb)


if __name__ == "__main__":
    main()
