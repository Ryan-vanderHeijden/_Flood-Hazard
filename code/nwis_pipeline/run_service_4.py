"""
Run Services 4 and 5 independently (assumes Services 1–3 have already run).

  4. Rating curves    — fill NaN flood flow thresholds via USGS stage-discharge ratings
  5. Bankfull width   — estimate bankfull channel width via StreamStats regional regressions

Usage:
  python run_services_4_5.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")

sys.path.insert(0, str(BASE_DIR / "src"))
from fetch_rating_curves import fill_flows_from_ratings

flood_stages = pd.read_parquet(DATA_DIR / "metadata/flood_stages.parquet")
site_info    = pd.read_parquet(DATA_DIR / "metadata/site_info.parquet")

logger.info("=" * 60)
logger.info("SERVICE 4: Fill flood flows from USGS rating curves")
logger.info("=" * 60)
flood_stages = fill_flows_from_ratings(
    flood_stages=flood_stages,
    out_path=DATA_DIR / "metadata",
)


logger.info("Done.")
