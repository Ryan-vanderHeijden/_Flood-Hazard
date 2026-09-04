from __future__ import annotations

"""
Calibrate prediction intervals for the return-period flood model via split
conformal prediction on the spatial cross-validation residuals.

The out-of-fold predictions from ``train.py`` are genuine leave-HUC2-out
predictions, so their residuals ``r = obs_log - pred_log`` are an honest sample
of the error the model makes at a basin in an unseen region.  For a target
coverage ``1 - alpha`` the conformal interval in log10 space is

    [ pred + Q_{alpha/2}(r),  pred + Q_{1-alpha/2}(r) ]

evaluated per return period.  Because the calibration residuals are themselves
out-of-region, the resulting bands reflect spatial-extrapolation uncertainty
rather than in-sample noise, and their empirical coverage on the OOF set is a
faithful check.  Offsets are stored for CONUS inference.

Output ``ml/conformal_offsets.json``.

Example
-------
    python uncertainty.py --coverage 0.90
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from train import CV_PRED_PATH, RPS

logger = logging.getLogger(__name__)

ML_DIR = Path.home() / "data" / "flood_hazard" / "ml"
OFFSETS_PATH = ML_DIR / "conformal_offsets.json"


def compute_offsets(cv: pd.DataFrame, coverage: float = 0.90) -> dict:
    """Per-RP lower/upper log10 conformal offsets and their empirical OOF coverage."""
    alpha = 1.0 - coverage
    offsets: dict[str, dict[str, float]] = {}
    for rp in RPS:
        resid = (cv[f"obs_log_q{rp}"] - cv[f"pred_log_q{rp}"]).to_numpy()
        resid = resid[np.isfinite(resid)]
        lo = float(np.quantile(resid, alpha / 2))
        hi = float(np.quantile(resid, 1 - alpha / 2))
        cov = float(np.mean((resid >= lo) & (resid <= hi)))
        offsets[f"q{rp}"] = {
            "lo": lo,
            "hi": hi,
            "empirical_coverage": cov,
            "median_width_log10": hi - lo,
        }
    return {"coverage": coverage, "rps": RPS, "offsets": offsets}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=Path, default=CV_PRED_PATH)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--out", type=Path, default=OFFSETS_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cv = pd.read_parquet(args.cv)
    spec = compute_offsets(cv, args.coverage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2))

    tbl = pd.DataFrame(spec["offsets"]).T
    logger.info("Conformal offsets (%.0f%% target):\n%s", 100 * args.coverage, tbl.round(3).to_string())
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
