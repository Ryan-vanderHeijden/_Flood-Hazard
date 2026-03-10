# CIROH Flood Hazard Thresholds

Research code for the NOAA CIROH project to define nationally consistent flood hazard severity thresholds for use with NOAA's National Water Model (NWM) flood inundation forecasts.

**Institution:** University of Vermont (UVM)
**Funding:** NOAA / Cooperative Institute for Research to Operations in Hydrology (CIROH)

---

## Project Goals

Flood forecasts communicate *where* water will go, but not always *how bad* it will be. This project develops consistent, physically grounded thresholds that classify NWM-predicted flood inundation by hazard severity — supporting clearer risk communication for both fluvial (riverine) and pluvial (rainfall-driven) flooding.

Key objectives:
- Identify and characterize historical flood events from streamflow and stage records
- Distinguish fluvial vs. pluvial flood types
- Develop hazard severity thresholds anchored to observed impacts
- Integrate thresholds into NWM flood inundation mapping workflows

---

## Repository Structure

```
code/
└── nwis_pipeline/  # Data acquisition pipeline (USGS NWIS)
```

---

## Data Pipeline (`code/nwis_pipeline/`)

Fetches and organizes the observational data needed for threshold development. All data are pulled from public APIs (USGS NWIS, NOAA) and written to local Parquet files.

### Setup

```bash
cd code/nwis_pipeline
pip install -r requirements.txt
```

### Configuration

Edit `config/gages.csv` to add or remove USGS gage site IDs.

### Run

```bash
python run_pipeline.py
```

### Services (run in sequence)

| # | Service | Source | Output |
|---|---------|--------|--------|
| 1 | Site metadata | USGS NWIS | `data/metadata/site_info.parquet` |
| 2 | Daily streamflow & stage | USGS NWIS | `data/streamflow/streamflow.parquet` |
| 3 | Flood stage thresholds | NWS NWPS API | `data/metadata/flood_stages.parquet` |
| 4 | Data coverage summary | — | `data/metadata/data_coverage.parquet` |

Date ranges are derived automatically from each gage's NWIS period of record — no hardcoded start/end dates.

**Service 2 — stage fetch strategy (two-pass, parallelized):**
NWIS only stores pre-computed daily means for stage (00065) at older/legacy gauges (~542 of ~1,946 in the current config). Modern gauges record stage as 15-minute instantaneous values (unit values, `uv`) with no daily mean stored.

To maximise stage coverage, Service 2 uses two passes — both parallelized with `ThreadPoolExecutor`:
1. **Pass 1 — daily values (`get_dv`):** 20 parallel workers; picks up all gauges with a native daily mean.
2. **Pass 2 — IV fallback (`get_iv`):** for sites still lacking stage after Pass 1, fetches the raw 15-min series and computes daily means. Requests are batched (25 sites × 10-year windows) and dispatched concurrently (20 workers). These rows are tagged `stage_cd = "iv_mean"` to distinguish them from native daily means.

Both passes write checkpoint files to `data/streamflow/` so a re-run after a crash resumes from where it left off rather than re-fetching all data. Delete `streamflow_dv_checkpoint.parquet`, `streamflow_iv_checkpoint.parquet`, and the `*_no_data.txt` files to force a full re-fetch.

Flood stage thresholds (Service 3) are fetched from the [NWS National Water Prediction Service API](https://api.water.noaa.gov/nwps/v1/gauges) using 50 parallel workers. Each USGS site is spatially matched to its nearest NWS gauge; the match is verified against the NWS `usgsId` field. Only gages with observed stage data are queried.

### Inspect outputs

Open `code/nwis_pipeline/inspect_outputs.ipynb` in Jupyter to explore the fetched data.

---

## Dependencies

- Python 3.10+
- [`dataretrieval`](https://github.com/DOI-USGS/dataretrieval-python) — USGS NWIS API wrapper
- `pandas`, `pyarrow`, `requests`, `numpy`
