"""
One-off migration: rename the "flood" threshold level to "minor" in existing
data files, matching the code rename (NWPS itself calls this category
"minor").

Renames, under the data root:
  - parquet columns whose name has "flood" as a level token
    (flood_stage_ft, has_flood_stage, log_flood_ssp, d_flood, …)
  - the values "flood" / "log_flood_*" in the level-like string columns
    VALUE_COLS
  - files named with the level token (ml/streampower_models/lgbm_log_flood_*.joblib)

General uses of the word are left alone: flood_stages.parquet,
flood_frequency.parquet, flood_impact_tags.parquet, the flood_hazard root.
Every changed file is copied to <root>/_backup_<date>_pre_minor_rename/
first.  Backup folders, *pre_* snapshots and raw API caches are skipped.

Usage:
  python code/migrate_flood_to_minor.py                 # dry run
  python code/migrate_flood_to_minor.py --apply
  python code/migrate_flood_to_minor.py --root /path/to/data --apply   # e.g. on the VM
"""

from __future__ import annotations   # `dict | None` on the VM's Python 3.9

import argparse
import re
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

TOKEN = re.compile(r"(?<![a-z0-9])flood(?![a-z0-9])")   # "flood" between _ / ends
VALUE_COLS = {"level", "target", "threshold", "category"}
SKIP_DIRS = ("_backup", "streamstats/pages", "cache")


def to_minor(s: str) -> str:
    return TOKEN.sub("minor", s)


def skipped(p: Path) -> bool:
    return any(k in str(p) for k in SKIP_DIRS) or p.name.startswith(("pre_",)) or ".pre_" in p.name


def plan_parquet(p: Path) -> dict | None:
    names = pq.read_schema(p).names
    renames = {c: to_minor(c) for c in names if TOKEN.search(c)}
    value_cols = [c for c in names if c in VALUE_COLS]
    vals = {}
    if value_cols:
        df = pd.read_parquet(p, columns=value_cols)
        for c in value_cols:
            s = df[c].dropna().astype(str)
            n = int(s.str.contains(TOKEN).sum())
            if n:
                vals[c] = n
    return {"renames": renames, "values": vals} if (renames or vals) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Rename the 'flood' level to 'minor' in data files.")
    ap.add_argument("--root", type=Path, default=Path("/home/ryan/data/flood_hazard"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    root = args.root
    backup = root / f"_backup_{date.today():%Y%m%d}_pre_minor_rename"

    def back_up(p: Path) -> None:
        dst = backup / p.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)

    n_files = 0
    for p in sorted(root.rglob("*.parquet")):
        if skipped(p):
            continue
        plan = plan_parquet(p)
        if not plan:
            continue
        n_files += 1
        print(f"{p.relative_to(root)}: {len(plan['renames'])} columns"
              + (f", values in {plan['values']}" if plan["values"] else ""))
        if args.apply:
            back_up(p)
            df = pd.read_parquet(p).rename(columns=plan["renames"])
            for c in plan["values"]:
                mask = df[c].notna()
                df.loc[mask, c] = df.loc[mask, c].astype(str).map(to_minor)
            df.to_parquet(p, index=False)

    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix != ".parquet" and not skipped(p) \
                and TOKEN.search(p.stem) and "lgbm_" in p.name:
            new = p.with_name(to_minor(p.name))
            n_files += 1
            print(f"{p.relative_to(root)} → {new.name}")
            if args.apply:
                back_up(p)
                p.rename(new)

    verb = "migrated" if args.apply else "would migrate (dry run; pass --apply)"
    print(f"{verb} {n_files} files" + (f"; backups in {backup}" if args.apply else ""))


if __name__ == "__main__":
    main()
