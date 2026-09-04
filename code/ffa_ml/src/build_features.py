from __future__ import annotations

"""
Assemble the model-ready feature matrix from NHDPlus COMID attributes.

Reads the merged Wieczorek attribute table (``nhdplus/comid_attributes.parquet``,
one row per COMID, TOT_ = total upstream-routed values) and the screened target
table (``ml/targets.parquet``), and produces:

  * ``ml/training_features.parquet`` — one row per ``train_ok`` gauge, carrying
    the feature columns, the log10 quantile targets, the StreamStats baseline and
    the spatial-CV grouping columns; ready for ``train.py``.
  * ``ml/feature_spec.json`` — the fitted feature contract (ordered feature list
    and the NODATA sentinel rule) so CONUS inference transforms attributes
    identically.

Feature handling is deliberately light: LightGBM is invariant to monotone
transforms of individual features (splits are threshold-based) and handles NaN
natively, so features are passed through raw after two fitted screens —
sentinel-to-NaN (Wieczorek encodes NODATA as <= -9998) and dropping columns that
are all-missing or (near-)constant across the training sites.  Log transforms are
applied only to the *target* (in ``build_targets``) and, for reporting, to size
attributes when a linear baseline or interpretability plot needs them.

Example
-------
    python build_features.py
"""

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path.home() / "data" / "flood_hazard"
ATTR_PATH = DATA_DIR / "nhdplus" / "comid_attributes.parquet"
ML_DIR = DATA_DIR / "ml"
TARGETS_PATH = ML_DIR / "targets.parquet"
FEATURES_PATH = ML_DIR / "training_features.parquet"
SPEC_PATH = ML_DIR / "feature_spec.json"

SENTINEL_MAX = -9998.0  # Wieczorek NODATA (-9999); treat anything this low as missing.

# Target / grouping columns carried alongside the features into the training table.
from build_targets import LOG_Q_COLS, Q_COLS, SS_COLS  # noqa: E402  (sibling module)

_GROUP_COLS = ["huc2", "aggecoregion", "state_cd"]
_KEEP_META = ["site_no", "COMID", *_GROUP_COLS, "drainage_area_sqmi", "latitude", "longitude"]


def sanitize(attr: pd.DataFrame) -> pd.DataFrame:
    """Replace NODATA sentinels (<= -9998) with NaN across TOT_ attribute columns."""
    tot = [c for c in attr.columns if c.startswith("TOT_")]
    attr[tot] = attr[tot].mask(attr[tot] <= SENTINEL_MAX)
    return attr


def fit_feature_spec(attr_train: pd.DataFrame) -> dict:
    """Choose the feature columns from the training attribute rows.

    Drops all-missing and (near-)constant columns; returns a JSON-serialisable
    spec with the ordered feature list and the sentinel rule.
    """
    tot = [c for c in attr_train.columns if c.startswith("TOT_")]
    keep, dropped = [], {}
    for c in tot:
        s = attr_train[c]
        if s.notna().sum() == 0:
            dropped[c] = "all-missing"
        elif s.nunique(dropna=True) <= 1:
            dropped[c] = "constant"
        else:
            keep.append(c)
    if dropped:
        logger.info("Dropped %d feature(s): %s", len(dropped), dropped)
    logger.info("Selected %d features", len(keep))
    return {"features": keep, "sentinel_max": SENTINEL_MAX}


def transform(attr: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Apply the fitted spec (sentinel-clean + select features) to any COMID attribute frame."""
    attr = sanitize(attr)
    return attr.reindex(columns=spec["features"])


def _report_correlations(feat: pd.DataFrame, target: pd.Series, spec: dict) -> None:
    """Log Spearman feature-target correlations and near-collinear feature pairs (sanity check)."""
    corr = feat.corrwith(target, method="spearman").abs().sort_values(ascending=False)
    logger.info("Top |Spearman| vs log_q10: %s", corr.head(8).round(3).to_dict())
    cm = feat.corr().abs()
    pairs = [
        (a, b, round(cm.loc[a, b], 3))
        for i, a in enumerate(cm.columns)
        for b in cm.columns[i + 1 :]
        if cm.loc[a, b] > 0.95
    ]
    if pairs:
        logger.info("Near-collinear feature pairs (|r|>0.95): %s", pairs)


def build_features(
    attr_path: Path = ATTR_PATH,
    targets_path: Path = TARGETS_PATH,
    features_path: Path = FEATURES_PATH,
    spec_path: Path = SPEC_PATH,
) -> pd.DataFrame:
    """Build and write the training feature matrix and the feature spec.

    Returns
    -------
    pd.DataFrame
        The training table (features + log targets + baseline + groups).
    """
    targets = pd.read_parquet(targets_path)
    train = targets[targets["train_ok"]].copy()
    train_comids = train["COMID"].dropna().astype("int64").unique()

    attr = pd.read_parquet(attr_path).set_index("COMID")
    attr = sanitize(attr)
    attr_train = attr.reindex(train_comids)

    spec = fit_feature_spec(attr_train)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2))
    logger.info("Wrote %s", spec_path)

    feats = attr_train[spec["features"]].reset_index()  # COMID + features
    meta_cols = [c for c in _KEEP_META if c in train.columns]
    tbl = train[[*meta_cols, *Q_COLS, *LOG_Q_COLS, *SS_COLS]].merge(
        feats, on="COMID", how="left"
    )

    if "log_q10" in tbl.columns:
        _report_correlations(tbl[spec["features"]], tbl["log_q10"], spec)

    tbl.to_parquet(features_path, index=False)
    logger.info(
        "Wrote %s (%d train sites × %d features)", features_path, len(tbl), len(spec["features"])
    )
    return tbl


def iter_conus_features(
    spec: dict, attr_path: Path = ATTR_PATH, chunksize: int = 200_000
) -> Iterator[pd.DataFrame]:
    """Yield COMID feature chunks for the whole CONUS attribute table.

    Streams the parquet with pyarrow ``iter_batches`` so the full ~3M-row table
    is never materialised at once (CONUS inference must stay within a bounded
    memory footprint).  Each yielded frame has ``COMID`` as a column plus the
    spec's feature columns, sentinel-cleaned.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(attr_path)
    for batch in pf.iter_batches(batch_size=chunksize, columns=["COMID", *spec["features"]]):
        yield sanitize(batch.to_pandas())


def load_spec(spec_path: Path = SPEC_PATH) -> dict:
    return json.loads(Path(spec_path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attr", type=Path, default=ATTR_PATH)
    parser.add_argument("--targets", type=Path, default=TARGETS_PATH)
    parser.add_argument("--out", type=Path, default=FEATURES_PATH)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_features(args.attr, args.targets, args.out, args.spec)


if __name__ == "__main__":
    main()
