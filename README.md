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
├── requirements.txt           # shared dependencies for all pipelines
├── nwis_pipeline/             # Pipeline 1: USGS NWIS data acquisition
│   ├── run_pipeline.py        # entry point (Services 1–6)
│   ├── run_service_4.py       # standalone entry point for Services 4 only
│   ├── src/
│   │   ├── fetch_site_metadata.py
│   │   ├── fetch_streamflow.py
│   │   ├── fetch_flood_stages.py
│   │   ├── fetch_rating_curves.py
│   │   ├── fetch_bankfull_width.py
│   │   ├── compute_flood_percentiles.py        # Service 6 — called by run_pipeline.py
│   │   ├── fetch_NHDPlus_slope.py              # standalone — run separately
│   │   └── compute_specific_stream_power.py    # standalone — run separately
│   ├── inspect_outputs.ipynb
│   ├── inspect_stream_power.ipynb
│   └── inspect_coverage.ipynb
├── nwm_pipeline/              # Pipeline 2: NWM Retrospective v3.0 streamflow
│   ├── run_pipeline.py        # entry point
│   ├── src/
│   │   └── fetch_nwm_streamflow.py
│   └── inspect_nwm_results.ipynb
└── ffa_analysis/              # Pipeline 3: Flood Frequency Analysis (LP3/EMA)
    ├── run_ffa.py             # entry point
    ├── src/
    │   └── compute_flood_frequency.py
    └── inspect_ffa.ipynb
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
| 4 | Flood flows from rating curves | USGS NWIS ratings | `data/metadata/flood_stages.parquet` (updated in place) |
| 5 | Bankfull width | USGS StreamStats NSS API | `data/metadata/channel_geometry.parquet` |
| 6 | Flood flow threshold percentiles | observed discharge record | `data/metadata/flood_threshold_percentiles.parquet` |

Date ranges are derived automatically from each gage's NWIS period of record, using the union of the discharge (00060) and stage (00065) catalog periods — no hardcoded start/end dates.

**Service 2 — stage fetch strategy (two-pass, parallelized):**
NWIS only stores pre-computed daily means for stage (00065) at older/legacy gauges (~542 of ~1,946 in the current config). Modern gauges record stage as 15-minute instantaneous values (unit values, `uv`) with no daily mean stored.

To maximise stage coverage, Service 2 uses two passes — both parallelized with `ThreadPoolExecutor`:
1. **Pass 1 — daily values (`get_dv`):** 20 parallel workers; picks up all gauges with a native daily mean.
2. **Pass 2 — IV fallback (`get_iv`):** for sites still lacking stage after Pass 1, fetches the raw 15-min series and computes daily means. Requests are dispatched one site at a time in 2-year windows (20 parallel workers). Keeping calls small avoids NWIS connection timeouts that occur with large batched requests. These rows are tagged `stage_cd = "iv_mean"` to distinguish them from native daily means.

Both passes write checkpoint files to `data/streamflow/` so a re-run after a crash resumes from where it left off rather than re-fetching all data. Delete `streamflow_dv_checkpoint.parquet`, `streamflow_iv_checkpoint.parquet`, and the `*_no_data.txt` files to force a full re-fetch.

Flood stage thresholds (Service 3) are fetched from the [NWS National Water Prediction Service API](https://api.water.noaa.gov/nwps/v1/gauges) using 50 parallel workers. The USGS site → NWS LID mapping uses the NOAA HADS crosswalk (~10,483 entries, covers ~84% of stage gages). Matches are verified against the NWS `usgsId` field. The crosswalk is cached locally in `data/metadata/hads_crosswalk.parquet` and refreshed every 30 days.

Three files are written:

- **`flood_stages.parquet`** — stage (ft) and flow (cfs) thresholds for action/minor/moderate/major categories, plus NWS impact statements.
- **`gauge_map.parquet`** — maps each USGS `site_no` to its NWS LID and NWM `reach_id`.
- **`hads_crosswalk.parquet`** — cached HADS crosswalk (auto-managed; delete to force re-download).

**Service 4** fills `*_flow_cfs` values that are NaN after Service 3 by fetching the active USGS NWIS rating curve (`file_type="exsa"`) for each affected site (30 parallel workers). Adds `*_flow_source` columns (`"nwps"` | `"rating_curve"`) to `flood_stages.parquet`.

**Service 5** estimates bankfull channel width for every gage with a defined flood stage threshold using drainage-area-based regional regression equations from the [USGS StreamStats NSS API](https://streamstats.usgs.gov/nssservices). Only DRNAREA-only equations are applied; multi-variable equations are skipped. Output: `data/metadata/channel_geometry.parquet` (columns: `site_no`, `bankfull_width_ft`, `bkfw_equation`, `bkfw_region`, `bkfw_da_min_sqmi`, `bkfw_da_max_sqmi`, `bkfw_da_in_range`).

**Service 6** computes the non-exceedance percentile of each NWS flood flow threshold (action / flood / moderate / major) within each gage's historical daily discharge distribution using the Weibull plotting position. Sites with fewer than 3,650 valid days (~10 years) are flagged `record_ok = False`. Output: `data/metadata/flood_threshold_percentiles.parquet` (columns: `site_no`, `n_valid_days`, `record_ok`, `action_flow_pct`, `flood_flow_pct`, `moderate_flow_pct`, `major_flow_pct`).

A log file (`pipeline.log`) is written alongside `run_pipeline.py` on every run.

### Standalone analysis scripts

Two scripts run independently of the main pipeline (not called by `run_pipeline.py`):

**`src/fetch_NHDPlus_slope.py`** — fetches reach-average channel slope from NHDPlus Value Added Attributes via `pynhd`/WaterData (`nhdflowline_network`). Uses the NWM `reach_id` as the NHDPlus COMID. Queries in batches of 100, adds `nhd_slope_ft_ft` (ft/ft, dimensionless) to `channel_geometry.parquet`.

```bash
cd code/nwis_pipeline
python src/fetch_NHDPlus_slope.py
```

**`src/compute_specific_stream_power.py`** — computes specific stream power (W/m²) at each NWS flood threshold using ω = (γ·Q·S)/w, where γ = 9800 N/m³. Requires `channel_geometry.parquet` (bankfull width + NHD slope) and `flood_stages.parquet` (flood flow thresholds). Output: `data/metadata/stream_power.parquet` (columns: `site_no`, `action_ssp_wm2`, `flood_ssp_wm2`, `moderate_ssp_wm2`, `major_ssp_wm2`).

```bash
cd code/nwis_pipeline
python src/compute_specific_stream_power.py
```

### Inspect outputs

Open `code/nwis_pipeline/inspect_outputs.ipynb` in Jupyter to explore the pipeline outputs, or `code/nwis_pipeline/inspect_stream_power.ipynb` for specific stream power results.

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

## Pipeline 3 — Flood Frequency Analysis (`code/ffa_analysis/`)

Fits Log-Pearson Type III (LP3) distributions to USGS annual peak flow records for every gage with a defined NWS flood flow threshold, and evaluates each threshold's annual exceedance probability (AEP) and return period. Run this after Pipeline 1.

The implementation follows **USGS Bulletin 17C (2019)** using the **Expected Moments Algorithm (EMA)** (Cohn et al., 1997):
- Left-censored peaks (NWIS qualification code 6) are incorporated via EMA — their expected contribution below the perception threshold is added to the moment sums rather than being discarded.
- Historical peaks (qualification code 7) extend the effective record length using a weighted historical period.

### Prerequisites

Pipeline 1 (`code/nwis_pipeline/`) must have run and produced `data/metadata/flood_stages.parquet` with flow threshold columns populated.

### Run

```bash
python code/ffa_analysis/run_ffa.py
```

### Outputs

| File | Description |
|------|-------------|
| `data/annual_peaks.parquet` | Raw NWIS annual instantaneous peak flow records for all sites (long format) |
| `data/flood_frequency.parquet` | LP3 fit parameters + AEP / return period per threshold, plus EMA metadata |

**`flood_frequency.parquet` columns:**

| Column | Description |
|--------|-------------|
| `site_no` | USGS gage ID |
| `n_peaks` | Non-censored systematic peaks used in fit |
| `n_censored` | Code-6 censored peaks incorporated via EMA |
| `n_hist` | Code-7 historical peaks |
| `hist_H` | Historical period length (years) |
| `perception_threshold_cfs` | Minimum code-6 peak value (site perception threshold) |
| `high_censoring` | `True` if >25% of effective record is censored |
| `record_ok` | `True` if ≥10 non-censored systematic peaks |
| `lp3_skew`, `lp3_loc`, `lp3_scale` | LP3 distribution parameters (log10-space, `scipy.stats.pearson3`) |
| `{level}_aep` | Annual exceedance probability at each flood threshold |
| `{level}_return_period_yr` | Return period (years) at each flood threshold |

A log file (`ffa.log`) is written alongside `run_ffa.py` on every run.

### Inspect outputs

Open `code/ffa_analysis/inspect_ffa.ipynb` to explore AEP results, return period distributions, and LP3 fit diagnostics.

---

## Dependencies

- Python 3.10+
- See `code/requirements.txt` for the full list (`pynhd` required for `fetch_NHDPlus_slope.py`)
