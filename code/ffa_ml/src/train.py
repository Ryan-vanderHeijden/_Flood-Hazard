from __future__ import annotations

"""
Train the return-period flood-flow model with an index-flood decomposition and
leave-region-out spatial cross-validation.

Rather than predict the eight log10 quantiles independently (which can produce
physically impossible crossing curves, Q100 < Q50), the model predicts:

  * a *base* — log10(Q2), the index flood, and
  * seven non-negative *increments* — log10(Q_k / Q_{k-1}) between consecutive
    return periods.

Reconstructing by cumulative sum with the increments clipped at zero guarantees a
monotone frequency curve (Q2 <= Q5 <= ... <= Q500).  Each component is a LightGBM
regressor.  Skill is estimated with GroupKFold leave-HUC2-out cross-validation
(the honest analogue of prediction at an ungauged basin); out-of-fold predictions
are written for scoring by ``evaluate.py``.  Final models are refit on all
training sites and saved for CONUS inference.

Example
-------
    python train.py                       # leave-HUC2-out CV + final fit
    python train.py --group aggecoregion   # leave-ecoregion-out stress test
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

from build_features import FEATURES_PATH, SPEC_PATH, load_spec

logger = logging.getLogger(__name__)

ML_DIR = Path.home() / "data" / "flood_hazard" / "ml"
MODEL_DIR = ML_DIR / "models"
CV_PRED_PATH = ML_DIR / "cv_predictions.parquet"

RPS = [2, 5, 10, 25, 50, 100, 200, 500]
BASE_COL = "log_q2"
# consecutive (rp, prev_rp) pairs whose log-ratio increments are modelled
INCREMENTS = [(5, 2), (10, 5), (25, 10), (50, 25), (100, 50), (200, 100), (500, 200)]

LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0,
    n_jobs=-1,
    verbosity=-1,
)


def _component_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Build the base + consecutive-increment regression targets from log quantiles."""
    comp = pd.DataFrame(index=df.index)
    comp[BASE_COL] = df[BASE_COL]
    for rp, prev in INCREMENTS:
        comp[f"d_{rp}"] = df[f"log_q{rp}"] - df[f"log_q{prev}"]
    return comp


def fit_components(X: pd.DataFrame, comp: pd.DataFrame, params: dict | None = None) -> dict:
    """Fit one LightGBM regressor per decomposition component."""
    params = params or LGBM_PARAMS
    models = {}
    for col in comp.columns:
        m = LGBMRegressor(**params)
        m.fit(X, comp[col])
        models[col] = m
    return models


def predict_quantiles(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Predict monotone log10 quantiles by summing base + clipped increments."""
    out = pd.DataFrame(index=X.index)
    log_prev = models[BASE_COL].predict(X)
    out["log_q2"] = log_prev
    for rp, _prev in INCREMENTS:
        inc = np.clip(models[f"d_{rp}"].predict(X), 0.0, None)  # non-negative => monotone
        log_prev = log_prev + inc
        out[f"log_q{rp}"] = log_prev
    return out


def spatial_cv(
    df: pd.DataFrame, features: list[str], group_col: str, n_splits: int = 10
) -> pd.DataFrame:
    """Leave-region-out CV; return per-site out-of-fold predicted log quantiles."""
    df = df[df[group_col].notna()].copy()
    X, groups = df[features], df[group_col].astype(str)
    comp = _component_targets(df)
    n_splits = min(n_splits, groups.nunique())
    logger.info(
        "Spatial CV: %d folds over %d '%s' groups, %d sites",
        n_splits, groups.nunique(), group_col, len(df),
    )

    oof = pd.DataFrame(index=df.index, columns=[f"log_q{rp}" for rp in RPS], dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(gkf.split(X, groups=groups), 1):
        models = fit_components(X.iloc[tr], comp.iloc[tr])
        preds = predict_quantiles(models, X.iloc[te])
        oof.iloc[te] = preds.values
        logger.info("  fold %d/%d: train=%d test=%d", k, n_splits, len(tr), len(te))

    res = df[["site_no", "COMID", group_col]].copy()
    for rp in RPS:
        res[f"pred_log_q{rp}"] = oof[f"log_q{rp}"].values
        res[f"obs_log_q{rp}"] = df[f"log_q{rp}"].values
    return res


def train_final(df: pd.DataFrame, features: list[str], model_dir: Path = MODEL_DIR) -> dict:
    """Refit all components on every training site and persist them for inference."""
    models = fit_components(df[features], _component_targets(df))
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, m in models.items():
        joblib.dump(m, model_dir / f"lgbm_{name}.joblib")
    (model_dir / "components.json").write_text(
        json.dumps({"base": BASE_COL, "increments": INCREMENTS, "rps": RPS, "features": features}, indent=2)
    )
    logger.info("Saved %d component models → %s", len(models), model_dir)
    return models


def _quick_scores(cv: pd.DataFrame) -> None:
    """Log a quick log-space R2 per RP as an at-a-glance CV signal (full metrics in evaluate.py)."""
    for rp in RPS:
        o, p = cv[f"obs_log_q{rp}"], cv[f"pred_log_q{rp}"]
        ss_res = ((o - p) ** 2).sum()
        ss_tot = ((o - o.mean()) ** 2).sum()
        logger.info("  Q%-3d  log-space R2 = %.3f", rp, 1 - ss_res / ss_tot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--group", default="huc2", help="spatial CV grouping column")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--cv-out", type=Path, default=CV_PRED_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spec = load_spec(args.spec)
    features = spec["features"]
    df = pd.read_parquet(args.features_path)
    # Drop sites whose COMID had no attribute match (all features NaN).
    before = len(df)
    df = df[df[features].notna().any(axis=1)].reset_index(drop=True)
    if len(df) < before:
        logger.info("Dropped %d site(s) with no attribute match", before - len(df))

    cv = spatial_cv(df, features, args.group, args.n_splits)
    args.cv_out.parent.mkdir(parents=True, exist_ok=True)
    cv.to_parquet(args.cv_out, index=False)
    logger.info("Wrote %s", args.cv_out)
    _quick_scores(cv)

    train_final(df, features, args.model_dir)


if __name__ == "__main__":
    main()
