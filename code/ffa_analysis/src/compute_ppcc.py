from __future__ import annotations

import argparse
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from scipy.stats import pearsonr, pearson3

logger = logging.getLogger(__name__)


_PPCC_ALPHA = 0.05
_PPCC_N_SIM = 10_000
_PPCC_RNG_SEED = 3
_PPCC_WORKERS = 10


def _ppcc(log_peaks: np.ndarray, skew: float, loc: float, scale: float) -> float:
    n = len(log_peaks)
    p = np.arange(1, n + 1) / (n + 1)
    q = pearson3.ppf(p, skew, loc, scale)
    r, _ = pearsonr(np.sort(log_peaks), q)
    return float(r)


def _ppcc_critical_value(args: tuple) -> tuple[int, float]:
    """Worker: simulate lower-tail PPCC critical value for a single n. Returns (n, cv)."""
    n, alpha, n_sim, seed = args
    rng = np.random.default_rng(seed)
    rs = np.empty(n_sim)
    p = np.arange(1, n + 1) / (n + 1)
    for i in range(n_sim):
        samp = np.sort(rng.standard_normal(n))
        m1 = samp.mean()
        m2 = samp.var(ddof=1)
        m3 = ((samp - m1) ** 3).mean()
        s = np.sqrt(m2)
        skew_hat = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
        q = pearson3.ppf(p, skew_hat, m1, s)
        r, _ = pearsonr(samp, q)
        rs[i] = r
    return n, float(np.quantile(rs, alpha))


def _ppcc_site_worker(args: tuple) -> dict | None:
    """Worker: compute PPCC for a single site. Returns result dict or None."""
    site, log_peaks, skew, loc, scale, n_pilf, min_peaks = args
    if n_pilf > 0 and n_pilf < len(log_peaks):
        log_peaks = np.sort(log_peaks)[n_pilf:]
    if len(log_peaks) < min_peaks:
        return None
    r = _ppcc(log_peaks, skew, loc, scale)
    return {"site_no": site, "ppcc": r, "n_peaks": len(log_peaks)}


def compute_ppcc(
    ffa: pd.DataFrame,
    peaks: pd.DataFrame,
    alpha: float = _PPCC_ALPHA,
    n_sim: int = _PPCC_N_SIM,
    seed: int = _PPCC_RNG_SEED,
    min_peaks: int = 5,
    max_workers: int = _PPCC_WORKERS,
) -> pd.DataFrame:
    """Compute per-site PPCC goodness-of-fit for LP3 fits.

    Parameters
    ----------
    ffa:
        flood_frequency.parquet — must contain record_ok, degenerate_fit,
        lp3_skew/loc/scale, n_pilf columns.
    peaks:
        annual_peaks.parquet — must contain site_no, peak_cd, peak_va.
    alpha:
        Significance level for the MC critical value test.
    n_sim:
        Monte Carlo replicates for critical value simulation.
    seed:
        RNG seed for reproducibility.
    min_peaks:
        Minimum uncensored peaks required to compute PPCC.
    max_workers:
        Number of worker processes.

    Returns
    -------
    DataFrame with columns: site_no, ppcc, n_peaks, ppcc_cv, ppcc_ok.
    """
    eligible = ffa[(ffa["record_ok"] == True) & (~ffa["degenerate_fit"])]

    # Pre-group peaks by site to avoid repeated DataFrame filtering in workers
    usable = peaks[
        (~peaks["peak_cd"].str.contains("1|6|7|8", na=False))
        & (peaks["peak_va"] > 0)
    ].copy()
    usable["log_peak_va"] = np.log10(usable["peak_va"])
    peaks_by_site: dict[str, np.ndarray] = {
        site: grp["log_peak_va"].dropna().values
        for site, grp in usable.groupby("site_no")
    }

    # Build per-site worker args
    site_args = []
    for row in eligible.itertuples(index=False):
        log_peaks = peaks_by_site.get(row.site_no)
        if log_peaks is None or len(log_peaks) < min_peaks:
            continue
        n_pilf = int(row.n_pilf) if not np.isnan(row.n_pilf) else 0
        site_args.append((
            row.site_no, log_peaks,
            row.lp3_skew, row.lp3_loc, row.lp3_scale,
            n_pilf, min_peaks,
        ))

    # Parallel per-site PPCC
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        site_results = list(pool.map(_ppcc_site_worker, site_args, chunksize=10))

    records = [r for r in site_results if r is not None]
    ppcc_df = pd.DataFrame(records)
    if ppcc_df.empty:
        ppcc_df["ppcc_cv"] = pd.Series(dtype=float)
        ppcc_df["ppcc_ok"] = pd.Series(dtype=bool)
        return ppcc_df

    # Parallel critical value simulation — one task per unique n
    unique_ns = sorted(ppcc_df["n_peaks"].unique())
    cv_args = [(n, alpha, n_sim, seed) for n in unique_ns]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        cv_map = dict(pool.map(_ppcc_critical_value, cv_args, chunksize=1))

    ppcc_df["ppcc_cv"] = ppcc_df["n_peaks"].map(cv_map)
    ppcc_df["ppcc_ok"] = ppcc_df["ppcc"] >= ppcc_df["ppcc_cv"]
    return ppcc_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compute per-site LP3 PPCC goodness-of-fit.")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="/home/ryan/data/flood_hazard/ffa",
        help="Directory containing flood_frequency.parquet and annual_peaks.parquet",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    logger.info("Loading data from %s", data_dir)
    ffa   = pd.read_parquet(data_dir / "flood_frequency.parquet")
    peaks = pd.read_parquet(data_dir / "annual_peaks.parquet")

    logger.info("Running PPCC analysis (%d workers) …", _PPCC_WORKERS)
    ppcc_df = compute_ppcc(ffa, peaks)

    n_fail = int((~ppcc_df["ppcc_ok"]).sum())
    logger.info(
        "Done: %d sites evaluated, %d fail (%.1f%%)",
        len(ppcc_df), n_fail, 100 * n_fail / len(ppcc_df),
    )

    out_path = data_dir / "ppcc.parquet"
    ppcc_df.to_parquet(out_path, index=False)
    logger.info("Saved → %s", out_path)
