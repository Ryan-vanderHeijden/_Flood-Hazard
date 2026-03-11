"""
run_pipeline.py  —  NWM Retrospective v3.0 streamflow pipeline
---------------------------------------------------------------
Entry point.  Reads gauge_map.parquet from the NWIS pipeline and fetches
NWM v3.0 daily streamflow for every site with a valid NWM reach_id.

Usage
-----
    python code/nwm_pipeline/run_pipeline.py

Output
------
    code/nwm_pipeline/data/nwm/nwm_streamflow.parquet
    code/nwm_pipeline/data/nwm/checkpoints/{year}.parquet  (per-year, resumable)
"""

import logging
import sys
from pathlib import Path

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from src.fetch_nwm_streamflow import fetch_nwm_streamflow

# ---------------------------------------------------------------------------
# Paths  (resolved relative to this file — works from any working directory)
# ---------------------------------------------------------------------------

_PIPELINE_DIR = Path(__file__).parent

# Input: gauge_map.parquet from NWIS pipeline
GAUGE_MAP_PATH = _PIPELINE_DIR.parent / "nwis_pipeline" / "data" / "metadata" / "gauge_map.parquet"

# Output directory for NWM data
NWM_OUT_PATH = _PIPELINE_DIR / "data" / "nwm"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("NWM Retrospective v3.0 — Streamflow Pipeline")
    logger.info("=" * 60)
    logger.info("Gauge map  : %s", GAUGE_MAP_PATH)
    logger.info("Output dir : %s", NWM_OUT_PATH)

    if not GAUGE_MAP_PATH.exists():
        logger.error(
            "gauge_map.parquet not found at %s.\n"
            "Run the NWIS pipeline first (code/nwis_pipeline/run_pipeline.py).",
            GAUGE_MAP_PATH,
        )
        sys.exit(1)

    fetch_nwm_streamflow(GAUGE_MAP_PATH, NWM_OUT_PATH)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
