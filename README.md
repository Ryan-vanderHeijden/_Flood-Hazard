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
├── requirements.txt       # shared dependencies for all pipelines
├── nwis_pipeline/         # Pipeline 1: USGS NWIS data acquisition
└── nwm_pipeline/          # Pipeline 2: NWM Retrospective v3.0 streamflow
```

---

## Setup

```bash
pip install -r code/requirements.txt
```

---

## Pipeline 1 — NWIS (`code/nwis_pipeline/`)

Fetches and organizes the observational data needed for threshold development. All data are pulled from public APIs (USGS NWIS, NOAA) and written to local Parquet files.

### Configuration

Edit `config/gages.csv` to add or remove USGS gage site IDs.

### Run

```bash
python code/nwis_pipeline/run_pipeline.py
```

### Services (run in sequence)

| # | Service | Source | Output |
|---|---------|--------|--------|
| 1 | Site metadata | USGS NWIS | `data/metadata/site_info.parquet` |
| 2 | Daily streamflow & stage | USGS NWIS | `data/streamflow/streamflow.parquet` |
| 3 | Flood stage thresholds | NWS NWPS API | `data/metadata/flood_stages.parquet`, `data/metadata/gauge_map.parquet` |
| 4 | Data coverage summary | — | `data/metadata/data_coverage.parquet` |

Date ranges are derived automatically from each gage's NWIS period of record, using the union of the discharge (00060) and stage (00065) catalog periods — no hardcoded start/end dates.

**Service 2 — stage fetch strategy (two-pass, parallelized):**
NWIS only stores pre-computed daily means for stage (00065) at older/legacy gauges (~542 of ~1,946 in the current config). Modern gauges record stage as 15-minute instantaneous values (unit values, `uv`) with no daily mean stored.

To maximise stage coverage, Service 2 uses two passes — both parallelized with `ThreadPoolExecutor`:
1. **Pass 1 — daily values (`get_dv`):** 20 parallel workers; picks up all gauges with a native daily mean.
2. **Pass 2 — IV fallback (`get_iv`):** for sites still lacking stage after Pass 1, fetches the raw 15-min series and computes daily means. Requests are batched (25 sites × 10-year windows) and dispatched concurrently (20 workers). These rows are tagged `stage_cd = "iv_mean"` to distinguish them from native daily means.

Both passes write checkpoint files to `data/streamflow/` so a re-run after a crash resumes from where it left off rather than re-fetching all data. Delete `streamflow_dv_checkpoint.parquet`, `streamflow_iv_checkpoint.parquet`, and the `*_no_data.txt` files to force a full re-fetch.

Flood stage thresholds (Service 3) are fetched from the [NWS National Water Prediction Service API](https://api.water.noaa.gov/nwps/v1/gauges) using 50 parallel workers. The USGS site → NWS LID mapping uses a two-step approach to maximise reach coverage:

1. **HADS crosswalk** ([NOAA HADS](https://hads.ncep.noaa.gov/USGS/ALL_USGS-HADS_SITES.txt), ~10,483 entries) — direct `site_no → LID` lookup, covers ~84% of stage gages. Matches are verified against the NWS `usgsId` field. The crosswalk is cached locally in `data/metadata/hads_crosswalk.parquet` and refreshed every 30 days.
2. **NLDI fallback** — for the remaining ~16% not in HADS, the [USGS NLDI API](https://labs.waterdata.usgs.gov/api/nldi/linked-data/nwissite) is queried to obtain the NHDPlus COMID (= NWM reach ID) directly. These sites receive NaN flood thresholds but appear in `gauge_map.parquet` so the NWM pipeline can still fetch their streamflow.

Only gages with observed stage data are queried. Three files are written:

- **`flood_stages.parquet`** — stage (ft) and flow (cfs) thresholds for action/minor/moderate/major categories, plus NWS impact statements for each category.
- **`gauge_map.parquet`** — maps each USGS `site_no` to its NWS LID and NWM `reach_id`. Sites resolved via NLDI fallback have `lid = null`.
- **`hads_crosswalk.parquet`** — cached HADS crosswalk (auto-managed; delete to force re-download).

A log file (`pipeline.log`) is written alongside `run_pipeline.py` on every run.

### Inspect outputs

Open `code/nwis_pipeline/inspect_outputs.ipynb` in Jupyter to explore the fetched data.

---

## Pipeline 2 — NWM Retrospective (`code/nwm_pipeline/`)

Fetches NWM Retrospective v3.0 daily-mean streamflow from the public AWS S3 Zarr store for every gauge that has a NWM reach_id in `gauge_map.parquet`. Run this after Pipeline 1.

**Data source:** `s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr` (anonymous public access, us-east-1). Time span: ~Feb 1979 – Dec 2023 (hourly, resampled to daily mean).

### Run

```bash
python code/nwm_pipeline/run_pipeline.py
```

### Output

| File | Description |
|------|-------------|
| `data/nwm/nwm_streamflow.parquet` | Daily mean streamflow per site, full period of record |
| `data/nwm/nwm_metadata.json` | Provenance: NWM version, source URI, fetch date, site count, year range |
| `data/nwm/checkpoints/{year}.parquet` | Per-year checkpoints — delete to re-fetch a specific year |

**Output columns:** `site_no` (USGS ID), `reach_id` (NWM feature_id), `date`, `streamflow_cms` (m³/s), `streamflow_cfs` (ft³/s).

Processing is year-by-year with checkpointing and parallel execution (`_YEAR_WORKERS = 4` concurrent years by default). If the job is interrupted, restart the same command to resume from where it left off. Each worker opens its own S3 connection; raise `_YEAR_WORKERS` in `src/fetch_nwm_streamflow.py` if you have more RAM available (~250 MB peak per concurrent year at 7,000 sites). A log file (`pipeline.log`) is written alongside `run_pipeline.py` on every run.

---

## Dependencies

- Python 3.10+
- See `code/requirements.txt` for the full list
