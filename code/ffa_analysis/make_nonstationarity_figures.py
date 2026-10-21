"""
Figures and tables for the non-stationarity screen, attribution and refits.

Reads peak_trends.parquet, change_attribution.parquet and
post_cutoff_fits.parquet (src/compute_peak_trends.py,
src/attribute_change_points.py, src/refit_post_cutoff.py) plus the GAGES-II
land-cover and hydro-modification tables.

Figures (FIG_DIR):
  change_attribution   significant change points coloured by their likely
                       cause (map), and change year against the completion
                       year of the largest upstream dam (scatter)
  post_cutoff_refits   full-record against present-day return period at
                       the refit regulated gages, one panel per level

Tables (TABLE_DIR, CSV, plus all of them as Markdown in
nonstationarity_tables.md):
  screen               trend / change-point results by regulation class
  attribution          change-point groups and their dam, diversion and
                       urbanization indicators
  refit_status         refit outcome by cutoff source
  refit_ratios         present-day ÷ full-record return period by level
  km_medians           Kaplan–Meier median return periods: unregulated,
                       regulated (full record) and regulated (present day)
  unregulated_recent   recent-record refits at non-stationary unregulated
                       gages (src/refit_unregulated_recent.py), by role
                       and direction

Usage:
  python make_nonstationarity_figures.py
"""

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from make_rp_maps import (
    CONUS_EXTENT, DATA_DIR, FFA_DIR, FIG_DIR, INK, INK_2, LAND, LEVELS, PROJ, STATE,
    TITLE_FONT, _conus_geoms, km_median, load_sites, split_sites,
)

sys.path.insert(0, str(Path(__file__).parent / "src"))
from attribute_change_points import dams_in_basins, load_dams  # noqa: E402

import geopandas as gpd  # noqa: E402

TABLE_DIR = DATA_DIR / "tables"
GAGESII_CSV = DATA_DIR / "gagesii" / "csv"

# Likely cause of a significant change point.  Violet, amber and aqua: three
# categorical slots that pass the all-pairs CVD check and stay clear of the
# red-blue trend colormap in make_trend_maps.py, plus "unexplained" as open
# light-gray markers (lighter and unfilled, so it never relies on hue alone).
CAUSES = [
    ("dam", "Dam completed near the change", "#6250d6"),
    ("storage_later", "Storage built, change later", "#eda100"),
    ("urban", "Urbanizing basin (≥ 20% developed)", "#1baf7a"),
    ("unexplained", "No identified cause", "#b5b4af"),
]
CAUSE_COLOR = {k: c for k, _, c in CAUSES}
CAUSE_LABEL = {k: l for k, l, _ in CAUSES}
URBAN_DEV = 20          # % developed land (NLCD 2006) for "urbanizing"
STORAGE_LATER_DOR = 0.10
CHANGED = INK           # refit sites where the screen found a change (ink, not
                        # a hue, so it never reads as the trend colormap's blue)
UNCHANGED = "#b5b4af"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def gagesii(name: str, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(GAGESII_CSV / f"conterm_{name}.txt", dtype={"STAID": str},
                     encoding="latin1", usecols=["STAID"] + cols)
    return df.rename(columns={"STAID": "site_no"}).assign(
        site_no=lambda d: d["site_no"].str.zfill(8))


def largest_dam_year(site_nos: pd.Series) -> pd.Series:
    basins = gpd.read_parquet(DATA_DIR / "metadata" / "gage_basins.parquet")
    pairs = dams_in_basins(load_dams(DATA_DIR / "nid" / "nid_dams.parquet"),
                           basins[basins["site_no"].isin(site_nos)]).reset_index(drop=True)
    d = pairs.dropna(subset=["yearCompleted"])
    return d.loc[d.groupby("site_no")["nidStorage"].idxmax()].set_index("site_no")["yearCompleted"]


def load_attribution() -> pd.DataFrame:
    a = pd.read_parquet(FFA_DIR / "change_attribution.parquet")
    a = (a.merge(gagesii("lc06_basin", ["DEVNLCD06"]), on="site_no", how="left")
          .merge(gagesii("hydromod_other", ["CANALS_PCT", "PCT_IRRIG_AG"]),
                 on="site_no", how="left"))
    cp = ~a["is_control"]
    a["cause"] = np.select(
        [cp & a["dam_attributed"],
         cp & a["is_regulated"].eq(True) & (a["dor_during_record"] >= STORAGE_LATER_DOR),
         cp & (a["DEVNLCD06"] >= URBAN_DEV),
         cp],
        ["dam", "storage_later", "urban", "unexplained"], "")
    a["largest_dam_wy"] = a["site_no"].map(largest_dam_year(a.loc[cp, "site_no"]))
    sites = load_sites(False)[["site_no", "latitude", "longitude"]]
    return a.merge(sites, on="site_no", how="left")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def strip(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.4)
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, width=0.4, length=2, labelsize=6)


def cause_marker(ax, df, key, **kw):
    open_ = key == "unexplained"
    for marker, part in (("^", df[df["change_pct"] > 0]), ("v", df[df["change_pct"] <= 0])):
        ax.scatter(part["x"], part["y"], marker=marker,
                   facecolors="none" if open_ else CAUSE_COLOR[key],
                   edgecolors=CAUSE_COLOR[key] if open_ else LAND,
                   linewidths=0.6 if open_ else 0.25, **kw)


def fig_attribution(a: pd.DataFrame):
    cp = a[a["cause"] != ""]
    fig = plt.figure(figsize=(7.0, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.12,
                          left=0.01, right=0.98, top=0.90, bottom=0.24)
    ax = fig.add_subplot(gs[0], projection=PROJ)
    ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    ax.add_geometries(_conus_geoms(), crs=ccrs.PlateCarree(), facecolor=LAND,
                      edgecolor=STATE, linewidth=0.3, zorder=0)
    ax.spines["geo"].set_visible(False)
    for key, _, _ in reversed(CAUSES):      # unexplained underneath
        g = cp[cp["cause"] == key].assign(x=lambda d: d["longitude"], y=lambda d: d["latitude"])
        cause_marker(ax, g, key, s=12, transform=ccrs.PlateCarree(), zorder=3)
    ax.set_title(f"Significant change points ({len(cp)})", loc="left", pad=2,
                 fontdict=TITLE_FONT)

    sc = fig.add_subplot(gs[1])
    lo, hi = 1900, 2015
    # Attribution window: dam completed 5 yr before to 1 yr after the change,
    # i.e. change from 1 yr before to 5 yr after the dam.
    sc.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 5, hi + 5], color="#ecebe7",
                    linewidth=0, zorder=0)
    sc.plot([lo, hi], [lo, hi], color=INK_2, linewidth=0.5, linestyle=(0, (3, 2)), zorder=1)
    reg = cp[cp["is_regulated"].eq(True) & cp["largest_dam_wy"].notna()]
    for key, _, _ in reversed(CAUSES):
        g = reg[reg["cause"] == key].assign(x=lambda d: d["largest_dam_wy"],
                                             y=lambda d: d["change_wy"])
        cause_marker(sc, g, key, s=10, zorder=3)
    sc.set_xlim(lo, hi)
    sc.set_ylim(lo, hi)
    sc.set_aspect("equal")
    strip(sc)
    sc.set_xlabel("Largest upstream dam completed", fontsize=6.5, color=INK_2)
    sc.set_ylabel("Change year (Pettitt)", fontsize=6.5, color=INK_2)
    sc.set_title(f"Regulated gages ({len(reg)})", loc="left", pad=2, fontdict=TITLE_FONT)
    sc.text(0.97, 0.04, "shaded: change 1 yr before\nto 5 yr after the dam",
            transform=sc.transAxes, ha="right", va="bottom", fontsize=5.5, color=INK_2)

    n = cp["cause"].value_counts()
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                      markerfacecolor="none" if k == "unexplained" else c,
                      markeredgecolor=c if k == "unexplained" else LAND,
                      markeredgewidth=0.7 if k == "unexplained" else 0,
                      label=f"{l} ({n.get(k, 0)})") for k, l, c in CAUSES]
    handles += [Line2D([], [], marker="^", linestyle="none", markersize=4.5,
                       markerfacecolor=INK_2, markeredgewidth=0, label="Increase"),
                Line2D([], [], marker="v", linestyle="none", markersize=4.5,
                       markerfacecolor=INK_2, markeredgewidth=0, label="Decrease")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3,
               frameon=False, fontsize=6, handletextpad=0.2, columnspacing=1.2)
    return fig


def fig_refits(r: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.35), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.10})
    changed = r["mk_sig"] | r["pettitt_sig"]
    for ax, (lv, title) in zip(axes, LEVELS):
        a, b = r[f"full_{lv}_return_period_yr"], r[f"post_{lv}_return_period_yr"]
        ok = np.isfinite(a) & np.isfinite(b) & a.gt(0) & b.gt(0)
        ax.plot([1, 1e4], [1, 1e4], color=INK_2, linewidth=0.5, linestyle=(0, (3, 2)), zorder=1)
        for m, color, z in ((~changed, UNCHANGED, 2), (changed, CHANGED, 3)):
            sel = ok & m
            ax.scatter(a[sel], b[sel], s=3, color=color, linewidths=0, zorder=z,
                       rasterized=True)
        med = (b[ok & changed] / a[ok & changed]).median()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.set_xlim(1, 1e4)
        ax.set_ylim(1, 1e4)
        ax.set_xticks([1, 10, 100, 1000, 10000])
        ax.set_xticklabels(["1", "10", "100", "1k", "10k"])
        strip(ax)
        ax.set_title(title, loc="left", pad=11, fontdict=TITLE_FONT)
        ax.text(0.0, 1.015, f"changed: median × {med:.2f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6, color=INK_2)
    axes[0].set_yticks([1, 10, 100, 1000, 10000])
    axes[0].set_yticklabels(["1", "10", "100", "1k", "10k"])
    axes[0].set_ylabel("Present-day RP (yr)", fontsize=6.5, color=INK_2)
    fig.supxlabel("Full-record return period (yr)", fontsize=6.5, color=INK_2, y=0.11)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=3.5, markerfacecolor=CHANGED,
               markeredgewidth=0, label=f"Change detected ({int(changed.sum())} sites)"),
        Line2D([], [], marker="o", linestyle="none", markersize=3.5, markerfacecolor=UNCHANGED,
               markeredgewidth=0, label=f"No change detected ({int((~changed).sum())})"),
    ], loc="lower right", bbox_to_anchor=(0.99, 0.0), ncol=2, frameon=False, fontsize=6,
        handletextpad=0.3)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.27)
    return fig


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_screen(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in (("Unregulated", t[t["is_regulated"].eq(False)]),
                     ("Regulated", t[t["is_regulated"].eq(True)]), ("All", t)):
        n = len(g)
        rows.append({
            "Group": label, "Tested": n,
            "Trend up": (g["trend_dir"] == "up").sum(),
            "Trend down": (g["trend_dir"] == "down").sum(),
            "Change up": (g["pettitt_sig"] & (g["change_pct"] > 0)).sum(),
            "Change down": (g["pettitt_sig"] & (g["change_pct"] <= 0)).sum(),
            "Any": (g["mk_sig"] | g["pettitt_sig"]).sum(),
        })
    df = pd.DataFrame(rows)
    for c in ["Trend up", "Trend down", "Change up", "Change down", "Any"]:
        df[c] = [f"{v} ({100 * v / n:.0f}%)" for v, n in zip(df[c], df["Tested"])]
    return df


def table_attribution(a: pd.DataFrame) -> pd.DataFrame:
    a = a.assign(group=np.where(a["is_regulated"].eq(True), "Regulated", "Unregulated"))
    keyed = a[a["cause"] != ""].copy()
    ctl = a[a["is_control"]].assign(cause="control")
    rows = []
    for (grp, cause), g in pd.concat([keyed, ctl]).groupby(["group", "cause"]):
        rows.append({
            "Group": grp,
            "Cause": "No change point (control)" if cause == "control" else CAUSE_LABEL[cause],
            "Sites": len(g),
            "Median change": (f"{g['change_pct'].median():+.0f}%".replace("-", "−")
                              if cause != "control" else ""),
            "Median change year": (f"{g['change_wy'].median():.0f}" if cause != "control" else ""),
            "DOR built during record": round(g["dor_during_record"].median(), 2),
            "Canals (% length)": round(g["CANALS_PCT"].median(), 2),
            "Irrigated ag (%)": round(g["PCT_IRRIG_AG"].median(), 2),
            "Developed (%)": round(g["DEVNLCD06"].median(), 1),
        })
    return pd.DataFrame(rows)


def table_refit_status(r: pd.DataFrame) -> pd.DataFrame:
    names = {"storage": "Largest dam completed", "code6": "Code-6 onset",
             "storage90": "90% of storage (second pass)",
             "pettitt": "Pettitt change year", "none": "None",
             "refit": "Refit", "too_short": "< 30 yr after cutoff", "no_cutoff": "No cutoff",
             "short_record": "< 30 yr record (not screened)"}
    tab = pd.crosstab(r["cutoff_source"].map(names), r["status"].map(names),
                      margins=True, margins_name="Total")
    tab = tab[["Refit", "< 30 yr after cutoff", "No cutoff", "< 30 yr record (not screened)",
               "Total"]]
    tab.index.name = "Cutoff source"
    return tab.reset_index()


def table_flags(r: pd.DataFrame) -> pd.DataFrame:
    why = {"too_short": "Change detected; < 30 yr after cutoff",
           "no_cutoff": "Change detected; no dam or code-6 date",
           "short_record": "< 30 yr record; dam or code-6 date mid-record"}
    f = r[r["nonstationary_full_record"]]
    out = (f["status"].map(why).value_counts().rename_axis("Reason").reset_index(name="Sites"))
    return pd.concat([out, pd.DataFrame([{"Reason": "Total", "Sites": len(f)}])],
                     ignore_index=True)


def table_refit_ratios(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    refit = r[(r["status"] == "refit") & ~r["post_degenerate_fit"].fillna(False).astype(bool)]
    changed = refit["mk_sig"] | refit["pettitt_sig"]
    for lv, title in LEVELS:
        for label, g in (("Change detected", refit[changed]),
                         ("No change detected", refit[~changed]), ("All refit", refit)):
            a, b = g[f"full_{lv}_return_period_yr"], g[f"post_{lv}_return_period_yr"]
            ok = np.isfinite(a) & np.isfinite(b) & a.gt(0) & b.gt(0)
            q = b[ok] / a[ok]
            rows.append({"Level": title, "Sites": label, "n": int(ok.sum()),
                         "Median ratio": round(q.median(), 2),
                         "P10": round(q.quantile(0.1), 2), "P90": round(q.quantile(0.9), 2),
                         "Longer (%)": round(100 * (q > 1).mean())})
    return pd.DataFrame(rows)


def table_unregulated(u: pd.DataFrame) -> pd.DataFrame:
    """Recent-record refits at unregulated gages: present ÷ full RP by role and direction."""
    r = u[(u["status"] == "refit") & ~u["post_degenerate_fit"].fillna(False).astype(bool)]
    rows = []
    for (role, dirn), g in r.groupby(["role", "direction"]):
        row = {"Role": "Primary (urbanizing)" if role == "primary" else "Sensitivity",
               "Direction": dirn, "Refit": len(g),
               "Too short": int(((u["role"] == role) & (u["direction"] == dirn)
                                 & (u["status"] == "too_short")).sum())}
        for lv, title in LEVELS:
            a, b = g[f"full_{lv}_return_period_yr"], g[f"post_{lv}_return_period_yr"]
            ok = np.isfinite(a) & np.isfinite(b) & a.gt(0) & b.gt(0)
            row[title] = round((b[ok] / a[ok]).median(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def km(sites: pd.DataFrame, lv: str) -> float:
    pts, bnd = split_sites(sites, lv)
    return km_median(np.concatenate([pts["v"].to_numpy(), bnd["v"].to_numpy()]),
                     np.r_[np.zeros(len(pts), bool), np.ones(len(bnd), bool)])


def table_km(r: pd.DataFrame) -> pd.DataFrame:
    unreg, reg = load_sites(False, "unregulated"), load_sites(False, "regulated")
    ok = r[(r["status"] == "refit") & ~r["post_degenerate_fit"].fillna(False).astype(bool)]
    present = reg.set_index("site_no")
    for lv, _ in LEVELS:
        for kind in ("return_period_yr", "return_period_lb_yr"):
            col = f"{lv}_{kind}"
            upd = ok.set_index("site_no")[f"post_{col}"]
            present.loc[present.index.intersection(upd.index), col] = upd
    present = present.reset_index()
    rows = [{"Level": title,
             f"Unregulated ({len(unreg):,})": round(km(unreg, lv), 1),
             f"Regulated, full record ({len(reg):,})": round(km(reg, lv), 1),
             f"Regulated, present day ({len(ok):,} refit)": round(km(present, lv), 1)}
            for lv, title in LEVELS]
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Pipe table; numbers right-aligned (avoids a tabulate dependency)."""
    num = [pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---:" if n else "---" for n in num) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else f"{v:,}" if isinstance(v, (int, np.integer))
                              else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--out", type=Path, default=FIG_DIR)
    ap.add_argument("--tables", type=Path, default=TABLE_DIR)
    args = ap.parse_args()

    trends = pd.read_parquet(FFA_DIR / "peak_trends.parquet")
    a = load_attribution()
    r = pd.read_parquet(FFA_DIR / "post_cutoff_fits.parquet")

    args.out.mkdir(parents=True, exist_ok=True)
    for stem, fig in (("change_attribution", fig_attribution(a)),
                      ("post_cutoff_refits",
                       fig_refits(r[(r["status"] == "refit")
                                    & ~r["post_degenerate_fit"].fillna(False).astype(bool)]))):
        for ext in ("png", "pdf"):
            fig.savefig(args.out / f"{stem}.{ext}", facecolor="white")
        plt.close(fig)
        print(f"→ {args.out / stem}.{{png,pdf}}")

    args.tables.mkdir(parents=True, exist_ok=True)
    tables = {
        "screen": ("Trend and change-point screen (BH FDR 5%, sites with ≥ 30 years)",
                   table_screen(trends[trends["tested"]])),
        "attribution": ("Change points by likely cause, with dam, diversion and "
                        "urbanization indicators (medians)", table_attribution(a)),
        "refit_status": ("Post-cutoff refits at regulated gages: outcome by cutoff source",
                         table_refit_status(r)),
        "full_record_flags": ("Regulated gages keeping a full-record fit despite evidence of "
                              "change (nonstationary_full_record)", table_flags(r)),
        "refit_ratios": ("Present-day ÷ full-record return period at refit gages",
                         table_refit_ratios(r)),
        "km_medians": ("Kaplan–Meier median return period (yr), lower bounds censored",
                       table_km(r)),
        "unregulated_recent": ("Recent-record refits at non-stationary unregulated gages: "
                               "median present ÷ full-record return period",
                               table_unregulated(pd.read_parquet(
                                   FFA_DIR / "unregulated_recent_fits.parquet"))),
    }
    md = ["# Non-stationarity tables", "",
          "Generated by `code/ffa_analysis/make_nonstationarity_figures.py`.", ""]
    for name, (caption, df) in tables.items():
        df.to_csv(args.tables / f"nonstationarity_{name}.csv", index=False)
        md += [f"## {caption}", "", to_markdown(df), ""]
    (args.tables / "nonstationarity_tables.md").write_text("\n".join(md))
    print(f"→ {args.tables}/nonstationarity_*.csv and nonstationarity_tables.md")


if __name__ == "__main__":
    main()
