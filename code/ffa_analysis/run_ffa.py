"""
Flood Frequency Analysis — standalone entry point.

Reads flood_stages.parquet from the NWIS pipeline outputs, fetches USGS
annual peak flows, fits Log-Pearson Type III distributions, and writes:
  data/annual_peaks.parquet
  data/flood_frequency.parquet

Usage:
  python run_ffa.py
"""

import logging
import sys
from pathlib import Path

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
NWIS_META  = BASE_DIR.parent / "nwis_pipeline" / "data" / "metadata"

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


def main():
    logger.info("Flood Frequency Analysis")
    logger.info("  Input : %s", NWIS_META)
    logger.info("  Output: %s", DATA_DIR)

    compute_flood_frequency(
        flood_stages_path=NWIS_META,
        out_path=DATA_DIR,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
