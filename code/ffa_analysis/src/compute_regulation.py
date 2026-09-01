"""Build a per-site regulation profile for the FFA gage set.

Flood frequency assumes a stationary, unregulated flood series. Reservoir
operation violates that: storage truncates the peaks the LP3 is fitted to.
Two independent signals of regulation are available, and they measure
different things:

1. **NWIS peak qualification codes 5 and 6** — a field-office annotation that a
   given annual peak was affected by regulation or diversion. Accurate where
   applied, but which of the two codes a district uses varies, so both are
   needed; absence of a code remains weak evidence of absence of regulation.

2. **GAGES-II National Inventory of Dams storage** — a physical measurement of
   impoundment upstream of the gage. Complete and consistent, but blind to
   diversions, which move water without storing it.

They agree well once the codes are read correctly (Spearman ~0.62 across all
gages), but neither subsumes the other — storage is blind to diversions, and
annotation is blind where a district does not apply it — so this module keeps
both and escalates on either.

The continuous index is the **degree of regulation**

    DOR = NID storage / mean annual runoff

which is dimensionless — the number of years of the basin's mean annual runoff
that upstream reservoirs can hold. This works because GAGES-II reports
``STOR_NID_2009`` in megalitres per square kilometre, which is numerically
millimetres of depth over the watershed, the same unit as ``RUNAVE7100``.

Writes ``data/metadata/regulation.parquet``.

    python code/ffa_analysis/src/compute_regulation.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_flood_frequency import _parse_peak_cd  # noqa: E402

DATA_DIR = Path.home() / "data" / "flood_hazard"
PEAKS_PATH = DATA_DIR / "ffa" / "annual_peaks.parquet"
GAGESII_PATH = DATA_DIR / "metadata" / "gagesii_basin_chars.parquet"
OUT_PATH = DATA_DIR / "metadata" / "regulation.parquet"

# DOR class breaks, in years of mean annual runoff impounded. The 0.02 and 0.10
# cuts follow the degree-of-regulation conventions used in the GRanD reservoir
# literature (Lehner et al., 2011); 0.50 separates seasonal from multi-year
# storage, which is where flood peaks stop being recoverable from the record.
DOR_BREAKS = [0.02, 0.10, 0.50]
CLASSES = ["unregulated", "minor", "moderate", "major"]

log = logging.getLogger(__name__)


def peak_code_signals(peaks: pd.DataFrame) -> pd.DataFrame:
    """Per-site fraction of annual peaks flagged regulated or anthropogenic.

    Codes 5 and 6 both mark regulation or diversion — 5 "to unknown degree",
    6 without that qualifier. They are near-disjoint in practice (only 40 sites
    carry both), so the choice looks like district annotation habit rather than
    a real distinction in the field.

    Code C marks a documented anthropogenic change in the watershed
    (urbanisation, channelisation), which is a stationarity concern rather than
    a regulation one and is kept separate.

    Code R means "Revised" and carries no regulation information. An earlier
    version of this module counted it as regulation, which flagged 150 sites
    whose median degree of regulation is 0.007 — among the least regulated
    gages in the file. See ``code/ffa_analysis/peak_cd_notes.md``.
    """
    codes = peaks["peak_cd"].apply(_parse_peak_cd)
    out = pd.DataFrame({
        "site_no": peaks["site_no"].astype(str),
        "is_reg": codes.apply(lambda c: 5 in c or 6 in c),
        "is_anth": codes.apply(lambda c: "C" in c),
    })
    g = out.groupby("site_no").agg(
        n_peaks_total=("is_reg", "size"),
        n_peaks_regulated=("is_reg", "sum"),
        frac_peaks_regulated=("is_reg", "mean"),
        frac_peaks_anthropogenic=("is_anth", "mean"),
    ).reset_index()
    g["site_no"] = g["site_no"].str.zfill(8)
    return g


def classify(df: pd.DataFrame) -> pd.Series:
    """Assign a regulation class from DOR, escalated by peak-code evidence.

    DOR is the primary axis because it is measured consistently everywhere.
    Peak codes then escalate sites whose regulation the storage metric misses —
    principally diversions, which remove water without impounding it.
    """
    cls = pd.cut(
        df["dor"], [-np.inf] + DOR_BREAKS + [np.inf], labels=CLASSES,
    ).astype(object)

    # Storage says quiet, the field office says otherwise: trust the annotation.
    heavy_codes = df["frac_peaks_regulated"] > 0.75
    some_codes = df["frac_peaks_regulated"] > 0.25

    rank = {c: i for i, c in enumerate(CLASSES)}
    cur = cls.map(rank)
    floor = pd.Series(0, index=df.index)
    floor[some_codes] = rank["moderate"]
    floor[heavy_codes] = rank["major"]

    escalated = np.maximum(cur.fillna(0), floor).astype(int)
    out = pd.Series([CLASSES[i] for i in escalated], index=df.index)
    out[df["dor"].isna() & (df["frac_peaks_regulated"].fillna(0) == 0)] = pd.NA
    return out


def build() -> pd.DataFrame:
    peaks = pd.read_parquet(PEAKS_PATH, columns=["site_no", "peak_cd"])
    gg = pd.read_parquet(GAGESII_PATH, columns=[
        "site_no", "STOR_NID_2009", "STOR_NOR_2009", "RUNAVE7100",
        "NDAMS_2009", "MAJ_NDAMS_2009", "DDENS_2009",
        "RAW_DIS_NEAREST_DAM", "RAW_DIS_NEAREST_MAJ_DAM",
        "CLASS", "HYDRO_DISTURB_INDX",
    ])

    df = peak_code_signals(peaks).merge(gg, on="site_no", how="outer")

    # STOR_NID_2009 (megalitres/km2) is numerically mm over the basin, so
    # dividing by mean annual runoff (mm/yr) gives years of storage.
    df["dor"] = df["STOR_NID_2009"] / df["RUNAVE7100"].replace(0, np.nan)
    df["dor_normal"] = df["STOR_NOR_2009"] / df["RUNAVE7100"].replace(0, np.nan)

    df["regulation_class"] = classify(df)
    df["is_regulated"] = df["regulation_class"].isin(["moderate", "major"])
    df["is_reference"] = df["CLASS"].eq("Ref")

    # Which source drove the call — useful when auditing a surprising site.
    df["reg_evidence"] = np.select(
        [
            (df["dor"] > DOR_BREAKS[1]) & (df["frac_peaks_regulated"] > 0.25),
            (df["dor"] > DOR_BREAKS[1]),
            (df["frac_peaks_regulated"] > 0.25),
        ],
        ["both", "storage_only", "peak_codes_only"],
        default="neither",
    )

    cols = [
        "site_no", "regulation_class", "is_regulated", "is_reference",
        "reg_evidence", "dor", "dor_normal",
        "frac_peaks_regulated", "frac_peaks_anthropogenic",
        "n_peaks_total", "n_peaks_regulated",
        "STOR_NID_2009", "RUNAVE7100", "NDAMS_2009", "MAJ_NDAMS_2009",
        "DDENS_2009", "RAW_DIS_NEAREST_DAM", "RAW_DIS_NEAREST_MAJ_DAM",
        "HYDRO_DISTURB_INDX",
    ]
    return df[cols]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
    )
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    log.info("Wrote %s  (%d sites)", OUT_PATH, len(df))

    ffa_path = DATA_DIR / "ffa" / "flood_frequency.parquet"
    if ffa_path.exists():
        ffa = pd.read_parquet(
            ffa_path, columns=["site_no", "record_ok", "degenerate_fit"])
        qc = set(ffa[ffa.record_ok & ~ffa.degenerate_fit]
                 .site_no.astype(str).str.zfill(8))
        sub = df[df.site_no.isin(qc)]
        log.info("QC-passed sites: %d", len(sub))
        for name, n in sub.regulation_class.value_counts().items():
            log.info("  %-12s %5d  (%.1f%%)", name, n, 100 * n / len(sub))
        log.info("Evidence source:")
        for name, n in sub.reg_evidence.value_counts().items():
            log.info("  %-16s %5d", name, n)


if __name__ == "__main__":
    main()
