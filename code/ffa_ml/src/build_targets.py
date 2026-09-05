from __future__ import annotations

"""
Assemble the training target table for the ungauged return-period flood model.

The targets are the at-site Log-Pearson III flood quantiles produced by the FFA
pipeline (``ffa/streamstats_validation.parquet``: ``q2_cfs`` … ``q500_cfs`` for
the 2/5/10/25/50/100/200/500-year return periods).  This module joins those
quantiles to their fit-quality flags, screens out unreliable sites, attaches the
NHDPlus COMID (the key that links each gauge to its catchment attributes) and the
spatial grouping columns used for leave-region-out cross-validation, and adds the
log10-transformed targets the models actually fit.

Screening
---------
A site is retained for training (``train_ok``) when all of:
  * ``record_ok``       — >= 10 systematic peaks in the LP3 fit,
  * ``~degenerate_fit`` — EMA did not diverge,
  * ``ppcc_ok``         — passes the probability-plot-correlation GoF test,
  * ``~high_censoring`` — <= 25% censored/PILF peaks,
  * ``~q2_check_failed``— fitted Q2 agrees with the empirical median peak,
  * ``~is_regulated``   — not moderately/heavily regulated (regulation.parquet),
  * a non-null COMID and strictly positive quantiles.

Every site is written out with its flags retained so the screening is auditable;
``train_ok`` marks the modelling universe.  The published USGS StreamStats
regression estimates (``ss_regression_q*_cfs``) are carried through as the
baseline the model is later scored against.

Example
-------
    python build_targets.py
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path.home() / "data" / "flood_hazard"
FFA_DIR = DATA_DIR / "ffa"
META_DIR = DATA_DIR / "metadata"
ML_DIR = DATA_DIR / "ml"
OUT_PATH = ML_DIR / "targets.parquet"

RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 200, 500]
Q_COLS = [f"q{rp}_cfs" for rp in RETURN_PERIODS]
LOG_Q_COLS = [f"log_q{rp}" for rp in RETURN_PERIODS]
SS_COLS = [f"ss_regression_q{rp}_cfs" for rp in RETURN_PERIODS]


def build_targets(
    ffa_dir: Path = FFA_DIR,
    meta_dir: Path = META_DIR,
    out_path: Path = OUT_PATH,
) -> pd.DataFrame:
    """Build and write the screened, COMID-keyed target table.

    Parameters
    ----------
    ffa_dir, meta_dir : Path
        Directories holding the FFA outputs and the per-site metadata.
    out_path : Path
        Destination parquet.

    Returns
    -------
    pd.DataFrame
        One row per site with quantiles, log10 targets, QC flags, COMID,
        spatial groups, StreamStats baseline, and a ``train_ok`` flag.
    """
    sv = pd.read_parquet(
        ffa_dir / "streamstats_validation.parquet",
        columns=["site_no", *Q_COLS, *SS_COLS, "q2_check_failed"],
    )
    ff = pd.read_parquet(
        ffa_dir / "flood_frequency.parquet",
        columns=["site_no", "record_ok", "degenerate_fit", "high_censoring", "n_peaks"],
    )
    ppcc = pd.read_parquet(ffa_dir / "ppcc.parquet", columns=["site_no", "ppcc_ok"])
    reg = pd.read_parquet(
        meta_dir / "regulation.parquet",
        columns=["site_no", "is_regulated", "is_reference", "regulation_class"],
    )
    info = pd.read_parquet(
        meta_dir / "site_info.parquet",
        columns=["site_no", "latitude", "longitude", "huc8", "drainage_area_sqmi"],
    )
    gmap = pd.read_parquet(meta_dir / "gage_map.parquet", columns=["site_no", "reach_id"])
    gii = _load_ecoregion(meta_dir)

    df = sv.drop_duplicates("site_no")
    for other in (ff, ppcc, reg, info, gmap, gii):
        df = df.merge(other.drop_duplicates("site_no"), on="site_no", how="left")

    # COMID from the NWM/NHDPlus reach id.
    df["COMID"] = pd.to_numeric(df.pop("reach_id"), errors="coerce").astype("Int64")
    # HUC2 region for leave-HUC-out CV.
    df["huc2"] = df["huc8"].astype("string").str.zfill(8).str[:2]

    # log10 targets (guard non-positive quantiles).
    pos = df[Q_COLS] > 0
    for q, lq in zip(Q_COLS, LOG_Q_COLS):
        df[lq] = np.where(df[q] > 0, np.log10(df[q].where(df[q] > 0)), np.nan)

    qc = (
        df["record_ok"].fillna(False)
        & ~df["degenerate_fit"].fillna(True)
        & df["ppcc_ok"].fillna(False)
        & ~df["high_censoring"].fillna(True)
        & ~df["q2_check_failed"].fillna(True)
    )
    df["qc_pass"] = qc
    df["train_ok"] = (
        qc
        & ~df["is_regulated"].fillna(True)
        & df["COMID"].notna()
        & pos.all(axis=1)
    )

    _log_funnel(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d sites, %d train_ok)", out_path, len(df), int(df["train_ok"].sum()))
    return df


def _load_ecoregion(meta_dir: Path) -> pd.DataFrame:
    """Return site_no + GAGES-II AGGECOREGION for leave-ecoregion-out CV (empty-safe)."""
    path = meta_dir / "gagesii_basin_chars.parquet"
    try:
        gii = pd.read_parquet(path, columns=["site_no", "AGGECOREGION"])
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.warning("AGGECOREGION unavailable (%s); ecoregion CV will be skipped", exc)
        return pd.DataFrame({"site_no": pd.Series(dtype="string")})
    return gii.rename(columns={"AGGECOREGION": "aggecoregion"})


def _log_funnel(df: pd.DataFrame) -> None:
    n = len(df)
    logger.info("Target sites: %d", n)
    logger.info("  qc_pass:            %d", int(df["qc_pass"].sum()))
    logger.info("  with COMID:         %d", int(df["COMID"].notna().sum()))
    logger.info("  is_regulated:       %d", int(df["is_regulated"].fillna(False).sum()))
    logger.info("  train_ok (final):   %d", int(df["train_ok"].sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffa-dir", type=Path, default=FFA_DIR)
    parser.add_argument("--meta-dir", type=Path, default=META_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_targets(args.ffa_dir, args.meta_dir, args.out)


if __name__ == "__main__":
    main()
