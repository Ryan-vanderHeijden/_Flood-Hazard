from __future__ import annotations

"""
Service 5: Estimate bankfull channel width from USGS StreamStats NSS regional
regression equations.

For each streamgage that has at least one flood stage threshold defined,
bankfull width is estimated from the drainage-area-based regional regression
equations published via the USGS StreamStats NSS (National Streamflow
Statistics) API.

Algorithm:
  1. Map FIPS state codes → 2-letter state abbreviations.
  2. For each unique state, POST /scenarios/bylocation with the state centroid
     to discover which Bieger 2015 physiographic-province equations apply.
  3. For each discovered region, GET the full scenario object, fill DRNAREA=1,
     and POST /scenarios/estimate to retrieve the equation string.
  4. Select the most-specific DRNAREA-only width equation.
  5. Evaluate the power-law equation for each site and write
     data/metadata/channel_geometry.parquet.

StreamStats NSS API (base: https://streamstats.usgs.gov/nssservices):
  GET  /statisticgroups                          → list group IDs
  POST /scenarios/bylocation?statisticgroups=24  → region IDs for a location
  GET  /scenarios?statisticgroups=24&regressionregions={id}  → full scenario object
  POST /scenarios/estimate                       → compute result + equation string

Note: the POST /scenarios/estimate body must be the *full* GET response object;
a minimal body (just id + parameters) causes the server to return HTTP 500.

Limitations:
  - Sites in states without published equations receive NaN width.
  - Only DRNAREA-only equations are applied; multi-variable equations are skipped.
  - One equation is chosen per state (state centroid used for region selection),
    so sites near physiographic province boundaries may use a sub-optimal region.
  - Extrapolation beyond the equation DA range is applied but flagged with
    bkfw_da_in_range = False.

Output columns in channel_geometry.parquet:
  site_no              — USGS site number
  bankfull_width_ft    — estimated bankfull width (ft); NaN if unavailable
  bkfw_equation        — equation string as returned by StreamStats
  bkfw_region          — StreamStats regression region name
  bkfw_da_min_sqmi     — minimum DA in equation calibration range (sq mi)
  bkfw_da_max_sqmi     — maximum DA in equation calibration range (sq mi)
  bkfw_da_in_range     — True if site DA is within calibration bounds
"""

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# StreamStats NSS Services base URL
_SS_BASE = "https://streamstats.usgs.gov/nssservices"

# NSS statistic group ID for bankfull statistics (BNKF)
_BNKF_GROUP_ID = 24

# Concurrent workers for per-state equation fetches
_SS_MAX_WORKERS = 10

# Approximate state centroids (lon, lat) for bylocation queries
# Used to find which physiographic-province equation applies to each state.
_STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "AL": (-86.8, 32.8),  "AK": (-153.4, 64.2), "AZ": (-111.9, 34.3),
    "AR": (-92.4, 34.8),  "CA": (-119.5, 37.2),  "CO": (-105.5, 39.0),
    "CT": (-72.7, 41.6),  "DE": (-75.5, 39.0),   "DC": (-77.0, 38.9),
    "FL": (-81.5, 28.7),  "GA": (-83.4, 32.6),   "HI": (-157.8, 20.3),
    "ID": (-114.5, 44.4), "IL": (-89.2, 40.0),   "IN": (-86.3, 40.0),
    "IA": (-93.5, 42.0),  "KS": (-98.4, 38.5),   "KY": (-84.3, 37.5),
    "LA": (-92.0, 31.0),  "ME": (-69.4, 45.4),   "MD": (-76.6, 39.0),
    "MA": (-71.8, 42.3),  "MI": (-84.5, 44.3),   "MN": (-94.3, 46.4),
    "MS": (-89.7, 32.7),  "MO": (-92.5, 38.4),   "MT": (-110.4, 46.9),
    "NE": (-99.9, 41.5),  "NV": (-116.4, 39.3),  "NH": (-71.6, 43.7),
    "NJ": (-74.4, 40.1),  "NM": (-106.1, 34.5),  "NY": (-75.5, 42.9),
    "NC": (-79.4, 35.5),  "ND": (-100.5, 47.5),  "OH": (-82.8, 40.4),
    "OK": (-97.5, 35.5),  "OR": (-120.5, 44.0),  "PA": (-77.2, 40.9),
    "RI": (-71.5, 41.7),  "SC": (-80.9, 33.8),   "SD": (-100.2, 44.4),
    "TN": (-86.7, 35.9),  "TX": (-99.3, 31.5),   "UT": (-111.1, 39.3),
    "VT": (-72.7, 44.1),  "VA": (-79.5, 37.5),   "WA": (-120.5, 47.5),
    "WV": (-80.6, 38.6),  "WI": (-89.8, 44.8),   "WY": (-107.5, 43.0),
    "PR": (-66.5, 18.2),  "VI": (-64.8, 17.7),
}

# FIPS state code (zero-padded string) → 2-letter postal abbreviation
_FIPS_TO_STATE: dict[str, str] = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY", "72": "PR", "78": "VI",
}


# ---------------------------------------------------------------------------
# Equation parsing
# ---------------------------------------------------------------------------

def _eval_equation(equation: str, variables: dict[str, float]) -> float | None:
    """
    Evaluate a StreamStats regression equation string.

    Supported forms (case-insensitive):
      15.04 * DRNAREA ^ 0.40
      EXP(-5.06 + 1.22 * LN(DRNAREA))
      3.281 * 5.90 * (2.590 * DRNAREA) ^ 0.280

    Args:
        equation: Equation string from the StreamStats JSON response.
        variables: Dict of variable name → numeric value (e.g. {"DRNAREA": 45.2}).

    Returns:
        Evaluated float, or None on parse/math error.
    """
    expr = equation.strip()

    # Substitute variable values (case-insensitive, word-boundary safe)
    for var, val in variables.items():
        expr = re.sub(
            r"(?<![A-Za-z_])" + re.escape(var) + r"(?![A-Za-z_0-9])",
            str(val),
            expr,
            flags=re.IGNORECASE,
        )

    # Normalize operators and math functions
    expr = expr.replace("^", "**")
    expr = re.sub(r"\bLN\b",    "log",    expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bLOG10\b", "log10",  expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bEXP\b",   "exp",    expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bSQRT\b",  "sqrt",   expr, flags=re.IGNORECASE)

    _safe_math = {
        "log":   math.log,
        "log10": math.log10,
        "exp":   math.exp,
        "sqrt":  math.sqrt,
    }

    try:
        result = eval(expr, {"__builtins__": {}}, _safe_math)  # noqa: S307
        val = float(result)
        return None if (math.isnan(val) or math.isinf(val) or val <= 0) else val
    except Exception:
        return None


# ---------------------------------------------------------------------------
# NSS API helpers
# ---------------------------------------------------------------------------

def _get_bankfull_group_id() -> int:
    """
    Query the NSS statistic-groups endpoint and return the ID for the bankfull
    statistics group (code BNKF).  Falls back to the known default of 24.
    """
    try:
        resp = requests.get(f"{_SS_BASE}/statisticgroups", timeout=15)
        resp.raise_for_status()
        for group in resp.json():
            if group.get("code", "").upper() == "BNKF":
                gid = group.get("id")
                logger.info("  NSS bankfull group ID: %s (%s)", gid, group.get("name"))
                return int(gid)
    except Exception as exc:
        logger.debug("  Could not fetch NSS statistic groups: %s", exc)
    logger.debug("  Falling back to default bankfull group ID %d", _BNKF_GROUP_ID)
    return _BNKF_GROUP_ID


def _bylocation_region_ids(lon: float, lat: float, group_id: int) -> list[int]:
    """
    POST /scenarios/bylocation with a tiny bounding polygon around (lon, lat)
    and return the IDs of applicable DRNAREA-only bankfull regression regions.
    """
    d = 0.01  # ~1 km half-width
    poly = {
        "type": "Polygon",
        "coordinates": [[[lon-d, lat-d], [lon+d, lat-d],
                          [lon+d, lat+d], [lon-d, lat+d], [lon-d, lat-d]]],
    }
    try:
        resp = requests.post(
            f"{_SS_BASE}/scenarios/bylocation",
            json=poly,
            params={"statisticgroups": str(group_id), "unitsystem": "english"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("  bylocation failed for (%.3f, %.3f): %s", lon, lat, exc)
        return []

    rr_ids = []
    for scenario in data:
        for rr in scenario.get("regressionRegions", []):
            params = rr.get("parameters", [])
            codes = {p.get("code", "").upper() for p in params}
            if codes == {"DRNAREA"}:
                rr_ids.append(rr["id"])
    return rr_ids


def _equation_for_region(rr_id: int, group_id: int) -> dict | None:
    """
    Retrieve the bankfull-width equation for a given regression region ID.

    Workflow:
      1. GET /scenarios?statisticgroups={id}&regressionregions={rr_id} → full object.
      2. Confirm the region requires only DRNAREA.
      3. Fill DRNAREA=1.0 in the full object and POST /scenarios/estimate.
      4. Extract the first 'width' result.

    Returns an equation dict with keys: name, code, equation, independentVariables.
    Note: the POST body must be the *full* GET response — a minimal body fails.
    """
    # Step 1: GET full scenario object
    try:
        resp = requests.get(
            f"{_SS_BASE}/scenarios",
            params={
                "statisticgroups": str(group_id),
                "regressionregions": str(rr_id),
                "unitsystem": "english",
            },
            timeout=20,
        )
        resp.raise_for_status()
        scenario_data = resp.json()
    except Exception as exc:
        logger.debug("  NSS scenario GET failed rr=%d: %s", rr_id, exc)
        return None

    if not scenario_data:
        return None

    rr_list = scenario_data[0].get("regressionRegions", [])
    if not rr_list:
        return None

    # Confirm DRNAREA-only
    param_codes = {
        p.get("code", "").upper()
        for rr in rr_list
        for p in rr.get("parameters", [])
    }
    if param_codes != {"DRNAREA"}:
        return None

    rr_name = rr_list[0].get("name", "")
    da_param = next(
        (p for rr in rr_list for p in rr.get("parameters", [])
         if p.get("code", "").upper() == "DRNAREA"),
        None,
    )
    limits = da_param.get("limits", {}) if da_param else {}
    da_min = float(limits.get("min", float("nan")))
    da_max = float(limits.get("max", float("nan")))

    # Step 2: Fill DRNAREA=1 in the full object and POST
    for rr in rr_list:
        for p in rr.get("parameters", []):
            if p.get("code", "").upper() == "DRNAREA":
                p["value"] = 1.0

    try:
        resp2 = requests.post(
            f"{_SS_BASE}/scenarios/estimate",
            json=scenario_data,
            params={"unitsystem": "english"},
            timeout=30,
        )
        resp2.raise_for_status()
        estimate = resp2.json()
    except Exception as exc:
        logger.debug("  NSS estimate POST failed rr=%d: %s", rr_id, exc)
        return None

    # Step 3: Find bankfull-width result
    for sc in estimate:
        for rr_r in sc.get("regressionRegions", []):
            for res in rr_r.get("results", []):
                name_lower = res.get("name", "").lower()
                code = res.get("code", "").upper()
                eq_str = res.get("equation", "")
                if not eq_str:
                    continue
                if "width" in name_lower or code in {"BFWDTH", "BKFW", "BFW", "BW"}:
                    return {
                        "name":  rr_name,
                        "code":  code,
                        "equation": eq_str,
                        "independentVariables": [{
                            "code": "DRNAREA",
                            "min":  da_min,
                            "max":  da_max,
                        }],
                    }
    return None


def _fetch_state_equations(state_abbrev: str, group_id: int | str) -> list[dict] | None:
    """
    Fetch bankfull-width equations for one state from the NSS API.

    Uses the state centroid to call /scenarios/bylocation, which returns the
    physiographically-appropriate regression regions for that location.  Then
    retrieves equation strings by GETting each region's full scenario object
    and POSTing to /scenarios/estimate.

    Returns a list of one equation dict (compatible with _build_state_equations)
    or None if no equation could be retrieved.
    """
    centroid = _STATE_CENTROIDS.get(state_abbrev)
    if centroid is None:
        logger.debug("  No centroid for state %s", state_abbrev)
        return None

    lon, lat = centroid
    rr_ids = _bylocation_region_ids(lon, lat, int(group_id))
    if not rr_ids:
        logger.debug("  No DRNAREA-only regions found for %s at centroid", state_abbrev)
        return None

    # Sort: prefer more-specific (non-national USA) regions first
    def _specificity(rr_id: int) -> int:
        # We don't have names here; prefer IDs < 800 (most national is 798=USA)
        return 0 if rr_id == 798 else 1

    rr_ids_sorted = sorted(rr_ids, key=_specificity, reverse=True)

    for rr_id in rr_ids_sorted:
        eq = _equation_for_region(rr_id, int(group_id))
        if eq is not None:
            logger.debug(
                "  %s: equation from region id=%d: %s", state_abbrev, rr_id, eq["equation"]
            )
            return [eq]

    logger.debug("  No working equation found for %s", state_abbrev)
    return None


# ---------------------------------------------------------------------------
# Per-state equation cache builder
# ---------------------------------------------------------------------------

def _build_state_equations(
    states: list[str],
    group_id: int | str,
) -> dict[str, dict | None]:
    """
    Fetch and cache the best bankfull-width equation for each state abbreviation.

    Returns dict of state_abbrev → equation dict (or None).
    """
    state_eq: dict[str, dict | None] = {}

    def _worker(state: str) -> tuple[str, dict | None]:
        eq_list = _fetch_state_equations(state, group_id)
        best = eq_list[0] if eq_list else None
        return state, best

    with ThreadPoolExecutor(max_workers=_SS_MAX_WORKERS) as ex:
        futures = {ex.submit(_worker, s): s for s in states}
        for fut in as_completed(futures):
            state, eq = fut.result()
            state_eq[state] = eq

    n_found = sum(1 for v in state_eq.values() if v is not None)
    logger.info(
        "  NSS bankfull-width equations found for %d/%d states",
        n_found, len(states),
    )
    return state_eq


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def fetch_bankfull_width(
    site_info: pd.DataFrame,
    flood_stages: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """
    Estimate bankfull channel width for gages with defined flood stages.

    Args:
        site_info:    DataFrame with [site_no, drainage_area_sqmi, state_cd].
                      state_cd is a 2-digit FIPS code string (e.g. "50" for VT).
        flood_stages: DataFrame with [site_no, flood_stage_ft, ...].
                      Only sites with a non-NaN flood_stage_ft are processed.
        out_path:     Directory where channel_geometry.parquet is written.

    Returns:
        DataFrame with one row per input site (in flood_stages), columns:
        [site_no, bankfull_width_ft, bkfw_equation, bkfw_region,
         bkfw_da_min_sqmi, bkfw_da_max_sqmi, bkfw_da_in_range].
    """
    # Sites with at least one flood stage
    has_stage = flood_stages["flood_stage_ft"].notna()
    target_sites = flood_stages.loc[has_stage, "site_no"].tolist()

    logger.info(
        "Estimating bankfull width for %d sites with flood stages", len(target_sites)
    )

    if not target_sites:
        empty = pd.DataFrame(columns=[
            "site_no", "bankfull_width_ft", "bkfw_equation", "bkfw_region",
            "bkfw_da_min_sqmi", "bkfw_da_max_sqmi", "bkfw_da_in_range",
        ])
        _save(empty, out_path)
        return empty

    # Join drainage area and state
    meta = (
        site_info[["site_no", "drainage_area_sqmi", "state_cd"]]
        .copy()
        .assign(
            state_cd=lambda d: d["state_cd"].astype(str).str.strip().str.zfill(2),
        )
    )
    df = pd.DataFrame({"site_no": target_sites}).merge(meta, on="site_no", how="left")

    # Map FIPS → state abbreviation
    df["state_abbrev"] = df["state_cd"].map(_FIPS_TO_STATE)
    n_unmapped = df["state_abbrev"].isna().sum()
    if n_unmapped:
        logger.warning("  %d sites have unrecognised FIPS state code", n_unmapped)

    # Discover NSS bankfull group ID
    group_id = _get_bankfull_group_id()

    # Fetch equations for all states present in dataset
    states_needed = df["state_abbrev"].dropna().unique().tolist()
    state_equations = _build_state_equations(states_needed, group_id)

    # Evaluate equation per site
    rows = []
    n_computed = 0
    n_no_eq = 0
    n_no_da = 0

    for _, row in df.iterrows():
        site_no = row["site_no"]
        da      = row["drainage_area_sqmi"]
        state   = row["state_abbrev"]

        base = {
            "site_no":           site_no,
            "bankfull_width_ft": float("nan"),
            "bkfw_equation":     None,
            "bkfw_region":       state,
            "bkfw_da_min_sqmi":  float("nan"),
            "bkfw_da_max_sqmi":  float("nan"),
            "bkfw_da_in_range":  None,
        }

        if pd.isna(da):
            n_no_da += 1
            rows.append(base)
            continue

        eq = state_equations.get(state) if state else None
        if eq is None:
            n_no_eq += 1
            rows.append(base)
            continue

        # Extract DA limits
        da_min = da_max = float("nan")
        for ivar in (eq.get("independentVariables") or []):
            if ivar.get("code", "").upper() == "DRNAREA":
                da_min = float(ivar.get("min") or float("nan"))
                da_max = float(ivar.get("max") or float("nan"))
                break

        in_range: bool | None = None
        if not (np.isnan(da_min) or np.isnan(da_max)):
            in_range = bool(da_min <= da <= da_max)

        width = _eval_equation(eq.get("equation", ""), {"DRNAREA": da})

        record = {
            **base,
            "bankfull_width_ft": width if width is not None else float("nan"),
            "bkfw_equation":     eq.get("equation"),
            "bkfw_region":       eq.get("name", state),
            "bkfw_da_min_sqmi":  da_min,
            "bkfw_da_max_sqmi":  da_max,
            "bkfw_da_in_range":  in_range,
        }
        rows.append(record)
        if width is not None:
            n_computed += 1
        else:
            n_no_eq += 1  # equation parse failed

    result = pd.DataFrame(rows)

    logger.info(
        "  Bankfull width computed: %d sites | no equation: %d | no DA: %d",
        n_computed, n_no_eq, n_no_da,
    )

    _save(result, out_path)
    return result


def _save(df: pd.DataFrame, out_path: Path) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    p = out_path / "channel_geometry.parquet"
    df.to_parquet(p, index=False)
    logger.info("Saved %d rows → %s", len(df), p)
