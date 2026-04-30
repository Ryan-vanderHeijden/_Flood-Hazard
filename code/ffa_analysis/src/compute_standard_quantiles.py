from __future__ import annotations

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearson3

logger = logging.getLogger(__name__)

_RETURN_PERIODS = [2, 5, 10, 30]


def compute_standard_quantiles(
    ffa: pd.DataFrame,
    return_periods: list[int] = _RETURN_PERIODS,
) -> pd.DataFrame:
    """Compute flow quantiles at standard return periods from fitted LP3 parameters.

    Parameters
    ----------
    ffa:
        flood_frequency.parquet — must contain record_ok, degenerate_fit,
        lp3_skew, lp3_loc, lp3_scale columns.
    return_periods:
        Return periods in years. AEP = 1 / RP.

    Returns
    -------
    DataFrame with columns: site_no, q{rp}_cfs for each RP.
    """
    eligible = ffa[(ffa["record_ok"] == True) & (~ffa["degenerate_fit"])]

    records = []
    for row in eligible.itertuples(index=False):
        rec: dict = {"site_no": row.site_no}
        for rp in return_periods:
            aep = 1.0 / rp
            log_q = pearson3.ppf(1.0 - aep, row.lp3_skew, row.lp3_loc, row.lp3_scale)
            rec[f"q{rp}_cfs"] = float(10.0 ** log_q)
        records.append(rec)

    sq = pd.DataFrame(records)

    q_cols = [f"q{rp}_cfs" for rp in return_periods]
    n_nonfinite = int((~np.isfinite(sq[q_cols])).any(axis=1).sum())
    if n_nonfinite:
        logger.warning("%d sites have non-finite quantiles", n_nonfinite)

    return sq


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compute standard LP3 flow quantiles.")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="/home/ryan/data/flood_hazard/ffa",
        help="Directory containing flood_frequency.parquet",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    logger.info("Loading data from %s", data_dir)
    ffa = pd.read_parquet(data_dir / "flood_frequency.parquet")

    logger.info("Computing standard quantiles for return periods %s yr …", _RETURN_PERIODS)
    sq = compute_standard_quantiles(ffa)

    n_sites = len(sq)
    logger.info("Done: %d eligible sites", n_sites)

    out_path = data_dir / "standard_quantiles.parquet"
    sq.to_parquet(out_path, index=False)
    logger.info("Saved → %s", out_path)
