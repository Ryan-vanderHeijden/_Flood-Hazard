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
└── pipeline/       # Data acquisition pipeline (USGS NWIS)
```

---

## Data Pipeline (`code/pipeline/`)

Fetches and organizes the observational data needed for threshold development. All data are pulled from public APIs (USGS NWIS, NOAA) and written to local Parquet files.

### Setup

```bash
cd code/pipeline
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
| 3 | Flood stage thresholds | NOAA / NWS | `data/metadata/flood_stages.parquet` |
| 4 | Data coverage summary | — | `data/metadata/data_coverage.parquet` |

Date ranges are derived automatically from each gage's NWIS period of record — no hardcoded start/end dates.

### Inspect outputs

Open `code/pipeline/inspect_outputs.ipynb` in Jupyter to explore the fetched data.

---

## Dependencies

- Python 3.10+
- [`dataretrieval`](https://github.com/DOI-USGS/dataretrieval-python) — USGS NWIS API wrapper
- `pandas`, `pyarrow`, `requests`
