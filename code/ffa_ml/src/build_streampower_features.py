from __future__ import annotations

"""
Assemble the *watershed-only* feature matrix for the NWS-threshold stream-power
model.

Reuses the NHDPlus ``TOT_`` attribute machinery from ``build_features.py``
(``sanitize``, the fitted ``feature_spec.json`` column list, the NODATA sentinel
rule) but deliberately **excludes the algebraic inputs to stream power**.

Stream power is ω = γ·Q·S / w, so reach slope (S) and channel width (w) are
definitional, not predictive: letting the model see them would manufacture skill
mechanically.  We therefore drop any feature whose name matches
``EXCLUDE_PATTERNS`` (the slope proxies ``TOT_BASIN_SLOPE`` / ``TOT_STREAM_SLOPE``;
channel width is not in the Wieczorek TOT_ set, so no width column is present, but
the pattern list documents the intent and is trivial to extend).  This yields the
honest test: can *non-definitional* watershed attributes predict the threshold
stream power?

To stay within the server's memory budget the ~3M-row CONUS attribute table is
read for the surviving feature columns only, then reduced to the training COMIDs.

Example
-------
    python build_streampower_features.py
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from build_features import ATTR_PATH, SPEC_PATH, load_spec, sanitize
from build_streampower_targets import LOG_SP_COLS, SP_COLS, _KEEP_META

logger = logging.getLogger(__name__)

DATA_DIR = Path.home() / "data" / "flood_hazard"
ML_DIR = DATA_DIR / "ml"
SP_TARGETS_PATH = ML_DIR / "streampower_targets.parquet"
FEATURES_PATH = ML_DIR / "streampower_training_features.parquet"
SP_SPEC_PATH = ML_DIR / "streampower_feature_spec.json"

# Algebraic inputs to stream power (ω = γ·Q·S/w) — excluded to keep the test honest.
EXCLUDE_PATTERNS = ("SLOPE",)


def watershed_only_features(features: list[str]) -> tuple[list[str], list[str]]:
    """Split the fitted feature list into (kept, dropped) by EXCLUDE_PATTERNS."""
    dropped = [c for c in features if any(p in c.upper() for p in EXCLUDE_PATTERNS)]
    kept = [c for c in features if c not in dropped]
    return kept, dropped


def build_streampower_features(
    attr_path: Path = ATTR_PATH,
    sp_targets_path: Path = SP_TARGETS_PATH,
    base_spec_path: Path = SPEC_PATH,
    features_path: Path = FEATURES_PATH,
    spec_path: Path = SP_SPEC_PATH,
) -> pd.DataFrame:
    """Build and write the watershed-only training feature matrix + spec."""
    base = load_spec(base_spec_path)
    kept, dropped = watershed_only_features(base["features"])
    logger.info("Watershed-only: kept %d / %d features; dropped %s",
                len(kept), len(base["features"]), dropped)

    spec = {"features": kept, "sentinel_max": base["sentinel_max"],
            "excluded_stream_power_inputs": dropped}
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2))
    logger.info("Wrote %s", spec_path)

    tgt = pd.read_parquet(sp_targets_path)
    train = tgt[tgt["sp_train_ok"]].copy()
    train_comids = train["COMID"].dropna().astype("int64").unique()

    # Read only the kept feature columns for the whole CONUS table, then reduce
    # to training COMIDs (memory-bounded on an 18 GB host).
    attr = pd.read_parquet(attr_path, columns=["COMID", *kept]).set_index("COMID")
    attr = sanitize(attr)
    feats = attr.reindex(train_comids)[kept].reset_index()  # COMID + features

    meta_cols = [c for c in _KEEP_META if c in train.columns]
    carry = meta_cols + [*SP_COLS.keys(), *LOG_SP_COLS]
    carry = list(dict.fromkeys(carry))
    tbl = train[carry].merge(feats, on="COMID", how="left")

    tbl.to_parquet(features_path, index=False)
    logger.info("Wrote %s (%d train sites × %d features)", features_path, len(tbl), len(kept))
    return tbl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attr", type=Path, default=ATTR_PATH)
    parser.add_argument("--sp-targets", type=Path, default=SP_TARGETS_PATH)
    parser.add_argument("--base-spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--out", type=Path, default=FEATURES_PATH)
    parser.add_argument("--spec", type=Path, default=SP_SPEC_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_streampower_features(args.attr, args.sp_targets, args.base_spec, args.out, args.spec)


if __name__ == "__main__":
    main()
