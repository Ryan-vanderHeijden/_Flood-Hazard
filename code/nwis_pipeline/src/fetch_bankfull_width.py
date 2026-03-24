from __future__ import annotations

"""
Service 5: Estimate bankfull channel width from USGS StreamStats regional
regression equations.

For each streamgage that has at least one flood stage threshold defined,
bankfull width is estimated from the drainage-area-based regional regression
equations published via the USGS StreamStats Regression Services API.

Algorithm:
  1. Map FIPS state codes → 2-letter state abbreviations.
  2. For each unique state, query the StreamStats API for channel-geometry
     regression equations (bankfull width).
  3. Select the width equation that requires only drainage area (DRNAREA),
     preferring equations whose DA range brackets the site's drainage area.
  4. Evaluate the power-law or log-linear equation for each site.
  5. Write data/metadata/channel_geometry.parquet.

StreamStats API endpoint:
  https://streamstats.usgs.gov/regressionservices/equations/byregions
  ?rcode={state_abbrev}&statisticgroups={channelgeometry_id}&unitsystem=english

Limitations:
  - Not all states have published bankfull channel-geometry equations in
    StreamStats.  Sites in those states receive NaN width.
  - Multi-variable equations (requiring variables other than DRNAREA) are
    skipped; only DRNAREA-only equations are applied.
  - Extrapolation beyond the equation's DA range is applied but flagged
    with bkfw_da_in_range = False.

Output columns in channel_geometry.parquet:
  site_no              — USGS site number
  bankfull_width_ft    — estimated bankfull width (ft); NaN if unavailable
  bkfw_equation        — equation string as returned by StreamStats
  bkfw_region          — StreamStats region name / code
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

# StreamStats Regression Services base URL
_SS_BASE = "https://streamstats.usgs.gov/regressionservices"

# Concurrent workers for per-state equation fetches
_SS_MAX_WORKERS = 10

# Keywords used to identify bankfull width equations
_WIDTH_KEYWORDS = {"bankfull width", "bankfull channel width", "bkfw", "bfw", "channel width"}

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
# StreamStats API helpers
# ---------------------------------------------------------------------------

def _get_channelgeometry_group_id() -> int | None:
    """
    Query the StreamStats statistic-groups endpoint and return the numeric ID
    for the channel-geometry group.

    Returns None if the endpoint is unavailable or the group is not listed.
    """
    try:
        resp = requests.get(f"{_SS_BASE}/statisticgroups.json", timeout=15)
        resp.raise_for_status()
        for group in resp.json():
            name = group.get("name", "").lower()
            code = group.get("code", "").lower()
            if "channel" in name or "channel" in code or "geometry" in name:
                gid = group.get("id") or group.get("ID")
                logger.info("  StreamStats channel-geometry group ID: %s (%s)", gid, group.get("name"))
                return int(gid)
    except Exception as exc:
        logger.debug("  Could not fetch StreamStats statistic groups: %s", exc)
    return None


def _fetch_state_equations(state_abbrev: str, group_id: int | str) -> list[dict] | None:
    """
    Fetch channel-geometry regression equations for one state from StreamStats.

    Returns the parsed JSON list (may be empty) or None on HTTP/parse failure.
    """
    url = (
        f"{_SS_BASE}/equations/byregions"
        f"?rcode={state_abbrev}&statisticgroups={group_id}&unitsystem=english"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("  StreamStats equations fetch failed for %s: %s", state_abbrev, exc)
        return None


# ---------------------------------------------------------------------------
# Equation parsing
# ---------------------------------------------------------------------------

def _eval_equation(equation: str, variables: dict[str, float]) -> float | None:
    """
    Evaluate a StreamStats regression equation string.

    Supported forms (case-insensitive):
      0.61 * DRNAREA ^ 0.587
      EXP(-5.06 + 1.22 * LN(DRNAREA))
      exp(2.35 + 0.613*ln(DRNAREA))
      (DRNAREA^0.603) * 2.56

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


def _is_width_equation(eq: dict) -> bool:
    """True if the equation name / code looks like a bankfull-width equation."""
    name = eq.get("name", "").lower().strip()
    code = eq.get("code", "").lower().strip()
    return (
        any(k in name for k in _WIDTH_KEYWORDS)
        or code in {"bkfw", "bfw", "bw", "bankfullwidth", "bfwidth"}
    )


def _only_needs_drnarea(eq: dict) -> bool:
    """True if the equation's independent variables list contains only DRNAREA."""
    ivars = eq.get("independentVariables") or []
    if not ivars:
        # Try to infer from the equation string itself
        eqstr = eq.get("equation", "")
        tokens = re.findall(r"[A-Z]{2,}", eqstr)
        non_da = [t for t in tokens if t not in ("DRNAREA",)]
        return len(non_da) == 0
    codes = {v.get("code", "").upper() for v in ivars}
    return codes == {"DRNAREA"}


def _best_width_equation(equations: list[dict]) -> dict | None:
    """
    Select the best bankfull-width equation from a list of equations for one region.

    Preference: DRNAREA-only equations; within those, no particular ranking
    (typically there is only one per region).
    """
    candidates = [eq for eq in equations if _is_width_equation(eq) and _only_needs_drnarea(eq)]
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Per-state equation cache builder
# ---------------------------------------------------------------------------

def _build_state_equations(
    states: list[str],
    group_id: int | str,
) -> dict[str, dict | None]:
    """
    Fetch and cache bankfull-width equations for each state abbreviation.

    Returns dict of state_abbrev → best-width-equation dict (or None).
    """
    state_eq: dict[str, dict | None] = {}

    def _worker(state: str) -> tuple[str, dict | None]:
        data = _fetch_state_equations(state, group_id)
        if not data:
            return state, None
        # Response is a list of region objects; each has an "equations" array
        all_eqs: list[dict] = []
        for region in (data if isinstance(data, list) else [data]):
            all_eqs.extend(region.get("equations") or [])
        best = _best_width_equation(all_eqs)
        return state, best

    with ThreadPoolExecutor(max_workers=_SS_MAX_WORKERS) as ex:
        futures = {ex.submit(_worker, s): s for s in states}
        for fut in as_completed(futures):
            state, eq = fut.result()
            state_eq[state] = eq

    n_found = sum(1 for v in state_eq.values() if v is not None)
    logger.info(
        "  StreamStats bankfull-width equations found for %d/%d states",
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

    # Discover StreamStats channel-geometry group ID
    group_id = _get_channelgeometry_group_id()
    if group_id is None:
        logger.warning(
            "  Could not determine StreamStats channel-geometry group ID; "
            "trying 'channelgeometry' as fallback."
        )
        group_id = "channelgeometry"

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

        # Extract DA limits from the equation's independentVariables entry
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
