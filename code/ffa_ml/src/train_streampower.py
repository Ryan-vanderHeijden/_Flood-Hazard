from __future__ import annotations

"""
Train the NWS-threshold stream-power model with leave-HUC2-out spatial CV.

Parallel to ``train.py`` but simpler: the eight stream-power targets
(action/flood/moderate/major × {specific ω, total Ω}, all log10) are *not* a
nested quantile curve, so there is no index-flood / monotone decomposition —
each target gets one independent LightGBM regressor.  Skill is estimated with
GroupKFold leave-HUC2-out CV (the honest analogue of prediction at an ungauged
basin), reusing the LP3 model's hyper-parameters for comparability.  Out-of-fold
predictions are written for scoring by ``evaluate_streampower.py`` and final
models are refit on all sites and saved.

Example
-------
    python train_streampower.py                    # leave-HUC2-out CV + final fit
    python train_streampower.py --group aggecoregion
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

from train import LGBM_PARAMS
from build_streampower_features import FEATURES_PATH, SP_SPEC_PATH
from build_streampower_targets import LOG_SP_COLS

logger = logging.getLogger(__name__)

ML_DIR = Path.home() / "data" / "flood_hazard" / "ml"
MODEL_DIR = ML_DIR / "streampower_models"
CV_PRED_PATH = ML_DIR / "streampower_cv_predictions.parquet"


def _cv_one_target(
    df: pd.DataFrame, features: list[str], target: str, group_col: str, n_splits: int
) -> pd.DataFrame:
    """Leave-region-out CV for a single target; return out-of-fold predictions."""
    sub = df[df[target].notna() & df[group_col].notna()].copy()
    X, y, groups = sub[features], sub[target], sub[group_col].astype(str)
    k = min(n_splits, groups.nunique())
    if k < 2 or len(sub) < 2 * k:
        logger.warning("  %s: too few groups/sites (n=%d, groups=%d) — skipped",
                       target, len(sub), groups.nunique())
        return pd.DataFrame()

    oof = pd.Series(np.nan, index=sub.index, dtype=float)
    gkf = GroupKFold(n_splits=k)
    for tr, te in gkf.split(X, groups=groups):
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof.iloc[te] = m.predict(X.iloc[te])

    o, p = y.to_numpy(), oof.to_numpy()
    ss_tot = np.sum((o - o.mean()) ** 2)
    r2 = 1 - np.sum((o - p) ** 2) / ss_tot if ss_tot > 0 else np.nan
    logger.info("  %-16s n=%4d  %d folds  leave-%s-out R2 = %.3f",
                target, len(sub), k, group_col, r2)

    return pd.DataFrame({
        "site_no": sub["site_no"].values,
        "COMID": sub["COMID"].values,
        "group": groups.values,
        "target": target,
        "obs": o,
        "pred": p,
    })


def train_final(df: pd.DataFrame, features: list[str], targets: list[str],
                model_dir: Path = MODEL_DIR) -> None:
    """Refit each target on all available sites and persist for inference."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        sub = df[df[target].notna()]
        if len(sub) < 20:
            continue
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(sub[features], sub[target])
        joblib.dump(m, model_dir / f"lgbm_{target}.joblib")
    (model_dir / "spec.json").write_text(
        json.dumps({"targets": targets, "features": features}, indent=2))
    logger.info("Saved final models → %s", model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    parser.add_argument("--spec", type=Path, default=SP_SPEC_PATH)
    parser.add_argument("--group", default="huc2", help="spatial CV grouping column")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--cv-out", type=Path, default=CV_PRED_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    features = json.loads(Path(args.spec).read_text())["features"]
    df = pd.read_parquet(args.features_path)
    before = len(df)
    df = df[df[features].notna().any(axis=1)].reset_index(drop=True)
    if len(df) < before:
        logger.info("Dropped %d site(s) with no attribute match", before - len(df))

    logger.info("Leave-%s-out CV over %d targets:", args.group, len(LOG_SP_COLS))
    parts = [_cv_one_target(df, features, t, args.group, args.n_splits) for t in LOG_SP_COLS]
    parts = [p for p in parts if not p.empty]
    cv = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    args.cv_out.parent.mkdir(parents=True, exist_ok=True)
    cv.to_parquet(args.cv_out, index=False)
    logger.info("Wrote %s", args.cv_out)

    train_final(df, features, LOG_SP_COLS, args.model_dir)


if __name__ == "__main__":
    main()
