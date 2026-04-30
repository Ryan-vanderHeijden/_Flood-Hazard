from __future__ import annotations

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# NWS threshold → LP3 standard quantile pairings (matched by median RP)
# Empirical medians: action≈1.8yr, flood≈3.4yr, moderate≈9.5yr, major≈28.2yr
_PAIRINGS = [
    ("action_flow_cfs",   "q2_cfs",  "action_q2"),
    ("flood_flow_cfs",    "q5_cfs",  "flood_q5"),
    ("moderate_flow_cfs", "q10_cfs", "moderate_q10"),
    ("major_flow_cfs",    "q30_cfs", "major_q30"),
]


def compute_change_analysis(
    stages: pd.DataFrame,
    sq: pd.DataFrame,
) -> pd.DataFrame:
    """Compare NWS threshold flows to LP3 standard quantile flows.

    Percent change = (q_lp3 - q_nws) / q_nws * 100.
    Positive = LP3 predicts more flow than the NWS threshold at that RP level.

    Parameters
    ----------
    stages : flood_stages.parquet
    sq     : standard_quantiles.parquet

    Returns
    -------
    DataFrame with NWS flows, LP3 quantile flows, and pct-change columns.
    """
    nws_cols = [p[0] for p in _PAIRINGS]
    lp3_cols = [p[1] for p in _PAIRINGS]

    df = (
        stages[["site_no"] + nws_cols]
        .merge(sq[["site_no"] + lp3_cols], on="site_no", how="inner")
    )

    for nws_col, lp3_col, name in _PAIRINGS:
        valid = (df[nws_col] > 0) & df[nws_col].notna() & df[lp3_col].notna()
        df[f"{name}_pct"] = np.where(
            valid,
            100.0 * (df[lp3_col] - df[nws_col]) / df[nws_col],
            np.nan,
        )

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="NWS threshold vs. LP3 standard quantile comparison.")
    parser.add_argument("ffa_dir",  nargs="?", default="/home/ryan/data/flood_hazard/ffa")
    parser.add_argument("meta_dir", nargs="?", default="/home/ryan/data/flood_hazard/metadata")
    args = parser.parse_args()

    ffa_dir  = Path(args.ffa_dir)
    meta_dir = Path(args.meta_dir)

    stages = pd.read_parquet(meta_dir / "flood_stages.parquet")
    sq     = pd.read_parquet(ffa_dir  / "standard_quantiles.parquet")

    df = compute_change_analysis(stages, sq)

    pct_cols = [f"{name}_pct" for _, _, name in _PAIRINGS]
    for col in pct_cols:
        med = df[col].median()
        logger.info("%s  median pct change: %+.1f%%  (n=%d)", col, med, df[col].notna().sum())

    out = ffa_dir / "change_analysis.parquet"
    df.to_parquet(out, index=False)
    logger.info("Saved → %s", out)
