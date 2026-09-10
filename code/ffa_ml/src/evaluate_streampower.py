from __future__ import annotations

"""
Score the spatially cross-validated NWS-threshold stream-power model.

Consumes the out-of-fold predictions from ``train_streampower.py``
(``ml/streampower_cv_predictions.parquet``, long format: one row per
site × target with ``obs`` / ``pred`` in log10 units) and reports, per target
(4 thresholds × {specific ω, total Ω}):

  * log-space R², Nash-Sutcliffe efficiency, RMSE and bias, and
  * real-space (W/m² or W/m) Kling-Gupta and Nash-Sutcliffe efficiency.

The point of the exercise is the comparison: the LP3 flood quantiles regress
well (log-space R² ≈ 0.8+) while the NWS threshold *return periods* do not
(≈ 0).  This tells us where the stream-power re-expression of the threshold
lands between those two, using watershed-only predictors (reach slope / width
excluded, since ω = γ·Q·S/w is built from them).

Outputs ``reports/streampower_metrics.csv``.

Example
-------
    python evaluate_streampower.py
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from evaluate import _metrics_row
from train_streampower import CV_PRED_PATH
from build_streampower_targets import LOG_SP_COLS

logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def metrics_by_target(cv: pd.DataFrame) -> pd.DataFrame:
    """Per-target log- and real-space metrics, ordered as LOG_SP_COLS."""
    rows = {}
    for target, grp in cv.groupby("target"):
        rows[target] = _metrics_row(grp["obs"].to_numpy(), grp["pred"].to_numpy())
    order = [t for t in LOG_SP_COLS if t in rows]
    return pd.DataFrame({t: rows[t] for t in order}).T.rename_axis("target")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=Path, default=CV_PRED_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.report_dir.mkdir(parents=True, exist_ok=True)

    cv = pd.read_parquet(args.cv)
    if cv.empty:
        logger.warning("No CV predictions to score (%s is empty)", args.cv)
        return

    mbt = metrics_by_target(cv)
    out = args.report_dir / "streampower_metrics.csv"
    mbt.to_csv(out)
    logger.info("Stream-power skill (leave-HUC2-out, watershed-only predictors):\n%s",
                mbt.round(3).to_string())
    logger.info("Wrote %s", out)
    logger.info("Benchmark context: LP3 flood quantiles ~0.8+ log-R2 (strong); "
                "NWS threshold return periods ~0 (none).")


if __name__ == "__main__":
    main()
