"""
Flood Frequency Analysis — standalone entry point.

Reads flood_stages.parquet from the NWIS pipeline outputs, fetches USGS
annual peak flows, fits Log-Pearson Type III distributions, and writes:
  data/annual_peaks.parquet
  data/flood_frequency.parquet

Usage:
  python run_ffa.py             # reuse cached annual_peaks.parquet
  python run_ffa.py --refetch   # re-download NWIS peaks for every site
"""

import argparse
import logging
import sys
import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).parent
DATA_DIR   = Path("/home/ryan/data/flood_hazard/ffa")
NWIS_META  = Path("/home/ryan/data/flood_hazard/metadata")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "ffa.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ffa")

sys.path.insert(0, str(BASE_DIR / "src"))
from compute_flood_frequency import compute_flood_frequency
from compute_standard_quantiles import compute_standard_quantiles
from compute_change_analysis import compute_change_analysis


def main():
    ap = argparse.ArgumentParser(description="Flood frequency analysis (LP3 / EMA).")
    ap.add_argument("--refetch", action="store_true",
                    help="re-download NWIS annual peaks instead of using the cache")
    args = ap.parse_args()

    logger.info("Flood Frequency Analysis")
    logger.info("  Input : %s", NWIS_META)
    logger.info("  Output: %s", DATA_DIR)

    ffa_df = compute_flood_frequency(
        flood_stages_path=NWIS_META,
        out_path=DATA_DIR,
        refetch=args.refetch,
    )

    logger.info("Computing standard quantiles …")
    sq_df = compute_standard_quantiles(ffa_df)
    sq_path = DATA_DIR / "standard_quantiles.parquet"
    sq_df.to_parquet(sq_path, index=False)
    logger.info("Standard quantiles saved → %s", sq_path)

    logger.info("Computing change analysis …")
    stages = pd.read_parquet(NWIS_META / "flood_stages.parquet")
    ca_df = compute_change_analysis(stages, sq_df)
    ca_path = DATA_DIR / "change_analysis.parquet"
    ca_df.to_parquet(ca_path, index=False)
    logger.info("Change analysis saved → %s", ca_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
