from __future__ import annotations

"""
Score the spatially cross-validated model and contextualise it against the
published USGS StreamStats regional-regression equations.

Consumes the out-of-fold predictions from ``train.py``
(``ml/cv_predictions.parquet``) and reports, per return period:

  * log-space R^2, Nash-Sutcliffe efficiency, RMSE and bias, and
  * real-space (cfs) Kling-Gupta efficiency and Nash-Sutcliffe efficiency.

The baseline is ``ss_regression_q*_cfs`` (the StreamStats answer an ungauged
model must beat), scored on the same sites against the same at-site LP3 targets.
Feature importances are aggregated (LightGBM gain) across the decomposition
components as a physical sanity check.

Outputs ``reports/metrics_by_rp.csv``, ``reports/baseline_comparison.csv`` and
``reports/feature_importance.csv``.

Example
-------
    python evaluate.py
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train import BASE_COL, CV_PRED_PATH, MODEL_DIR, RPS

logger = logging.getLogger(__name__)

ML_DIR = Path.home() / "data" / "flood_hazard" / "ml"
TARGETS_PATH = ML_DIR / "targets.parquet"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    return 1.0 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)


def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = pred.std() / obs.std()
    beta = pred.mean() / obs.mean()
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def _metrics_row(obs_log: np.ndarray, pred_log: np.ndarray) -> dict:
    """Metrics for one RP: log-space R2/NSE/RMSE/bias plus real-space KGE/NSE."""
    m = np.isfinite(obs_log) & np.isfinite(pred_log)
    o, p = obs_log[m], pred_log[m]
    ss_tot = np.sum((o - o.mean()) ** 2)
    r2 = 1.0 - np.sum((o - p) ** 2) / ss_tot
    o_cfs, p_cfs = 10.0**o, 10.0**p
    return {
        "n": int(m.sum()),
        "log_r2": r2,
        "log_nse": nse(o, p),
        "log_rmse": float(np.sqrt(np.mean((p - o) ** 2))),
        "log_bias": float(np.mean(p - o)),
        "cfs_kge": kge(o_cfs, p_cfs),
        "cfs_nse": nse(o_cfs, p_cfs),
    }


def metrics_by_rp(cv: pd.DataFrame) -> pd.DataFrame:
    rows = {
        f"Q{rp}": _metrics_row(cv[f"obs_log_q{rp}"].to_numpy(), cv[f"pred_log_q{rp}"].to_numpy())
        for rp in RPS
    }
    return pd.DataFrame(rows).T.rename_axis("return_period")


def baseline_comparison(cv: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Compare model vs StreamStats-regression baseline (log-space R2/NSE) on shared sites."""
    ss = targets.set_index("site_no")
    cvi = cv.set_index("site_no")
    rows = {}
    for rp in RPS:
        obs = cvi[f"obs_log_q{rp}"]
        model = cvi[f"pred_log_q{rp}"]
        base_cfs = ss[f"ss_regression_q{rp}_cfs"].reindex(cvi.index)
        base = np.log10(base_cfs.where(base_cfs > 0))
        both = obs.notna() & model.notna() & base.notna()
        o = obs[both].to_numpy()
        rows[f"Q{rp}"] = {
            "n_shared": int(both.sum()),
            "model_log_r2": 1 - np.sum((o - model[both].to_numpy()) ** 2) / np.sum((o - o.mean()) ** 2),
            "baseline_log_r2": 1 - np.sum((o - base[both].to_numpy()) ** 2) / np.sum((o - o.mean()) ** 2),
            "model_log_nse": nse(o, model[both].to_numpy()),
            "baseline_log_nse": nse(o, base[both].to_numpy()),
        }
    return pd.DataFrame(rows).T.rename_axis("return_period")


def feature_importance(model_dir: Path = MODEL_DIR) -> pd.DataFrame:
    """Aggregate LightGBM gain importance across all decomposition components."""
    comps = list(model_dir.glob("lgbm_*.joblib"))
    total: pd.Series | None = None
    for path in comps:
        m = joblib.load(path)
        imp = pd.Series(m.booster_.feature_importance("gain"), index=m.feature_name_)
        total = imp if total is None else total.add(imp, fill_value=0.0)
    assert total is not None
    total = (total / total.sum()).sort_values(ascending=False)
    return total.rename("gain_fraction").rename_axis("feature").reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=Path, default=CV_PRED_PATH)
    parser.add_argument("--targets", type=Path, default=TARGETS_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.report_dir.mkdir(parents=True, exist_ok=True)

    cv = pd.read_parquet(args.cv)
    targets = pd.read_parquet(args.targets)

    mrp = metrics_by_rp(cv)
    mrp.to_csv(args.report_dir / "metrics_by_rp.csv")
    logger.info("Metrics by RP:\n%s", mrp.round(3).to_string())

    bc = baseline_comparison(cv, targets)
    bc.to_csv(args.report_dir / "baseline_comparison.csv")
    logger.info("Model vs StreamStats-regression baseline:\n%s", bc.round(3).to_string())

    fi = feature_importance(args.model_dir)
    fi.to_csv(args.report_dir / "feature_importance.csv", index=False)
    logger.info("Top features by gain:\n%s", fi.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
