from __future__ import annotations

"""
Assemble the training target table for the *NWS-threshold stream-power* model.

This is the parallel target track to ``build_targets.py`` (which builds the LP3
flood-quantile targets).  Here the targets are the stream power at each NWS
flood threshold — the physical re-expression of the threshold whose *return
period* proved near-unpredictable from watershed attributes.  Two families:

  * specific stream power  ω = γ·Q·S / w   (``*_ssp_wm2``, W/m²)
  * total stream power     Ω = γ·Q·S       (``*_tsp_wm``,  W/m)

for the action / flood / moderate / major thresholds, from
``metadata/stream_power.parquet``.

Rather than re-derive the FFA QC screening, this reads the already-screened
``ml/targets.parquet`` (site_no, COMID, huc2/aggecoregion groups, drainage area,
lat/lon, ``train_ok`` and all QC flags) and merges the stream-power columns onto
it by ``site_no``.  Eight log10 targets are added; a site is eligible for a given
target when it is ``train_ok`` and that stream-power value is finite and > 0.

Example
-------
    python build_streampower_targets.py
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path.home() / "data" / "flood_hazard"
META_DIR = DATA_DIR / "metadata"
ML_DIR = DATA_DIR / "ml"
TARGETS_PATH = ML_DIR / "targets.parquet"
STREAM_POWER_PATH = META_DIR / "stream_power.parquet"
OUT_PATH = ML_DIR / "streampower_targets.parquet"

THRESHOLDS = ["action", "flood", "moderate", "major"]
# raw stream-power column -> log10 target name
SP_COLS = {f"{t}_ssp_wm2": f"log_{t}_ssp" for t in THRESHOLDS}
SP_COLS.update({f"{t}_tsp_wm": f"log_{t}_tsp" for t in THRESHOLDS})
LOG_SP_COLS = list(SP_COLS.values())

# meta carried alongside the targets (subset of build_targets output actually present)
_KEEP_META = [
    "site_no", "COMID", "huc2", "aggecoregion", "state_cd",
    "drainage_area_sqmi", "latitude", "longitude", "train_ok",
]


def build_streampower_targets(
    targets_path: Path = TARGETS_PATH,
    stream_power_path: Path = STREAM_POWER_PATH,
    out_path: Path = OUT_PATH,
) -> pd.DataFrame:
    """Build and write the COMID-keyed stream-power target table."""
    targets = pd.read_parquet(targets_path)
    meta_cols = [c for c in _KEEP_META if c in targets.columns]
    tgt = targets[meta_cols].drop_duplicates("site_no").copy()

    sp = pd.read_parquet(stream_power_path).drop_duplicates("site_no")
    df = tgt.merge(sp, on="site_no", how="left")

    # log10 targets, guarding non-positive stream power (log undefined).
    for raw, log_col in SP_COLS.items():
        v = df[raw]
        df[log_col] = np.where(v > 0, np.log10(v.where(v > 0)), np.nan)

    train_ok = df["train_ok"].fillna(False) if "train_ok" in df.columns else False
    finite_any = df[LOG_SP_COLS].notna().any(axis=1)
    df["sp_train_ok"] = train_ok & finite_any

    _log_funnel(df, train_ok)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = meta_cols + [*SP_COLS.keys(), *LOG_SP_COLS, "sp_train_ok"]
    keep = list(dict.fromkeys(keep))  # de-dup, preserve order
    df[keep].to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d sites, %d sp_train_ok)", out_path, len(df), int(df["sp_train_ok"].sum()))
    return df


def _log_funnel(df: pd.DataFrame, train_ok: pd.Series) -> None:
    logger.info("Target sites: %d", len(df))
    logger.info("  train_ok (from LP3 targets):  %d", int(np.asarray(train_ok).sum()))
    for log_col in LOG_SP_COLS:
        n = int((df[log_col].notna() & train_ok).sum())
        logger.info("  train_ok & valid %-16s %d", log_col + ":", n)
    logger.info("  sp_train_ok (any target):     %d", int(df["sp_train_ok"].sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=TARGETS_PATH)
    parser.add_argument("--stream-power", type=Path, default=STREAM_POWER_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_streampower_targets(args.targets, args.stream_power, args.out)


if __name__ == "__main__":
    main()
