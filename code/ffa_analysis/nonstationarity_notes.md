# Non-stationarity in the annual peaks

First-pass screen, 2026-10-21. Code: `src/compute_peak_trends.py` (writes
`ffa/peak_trends.parquet`) and `make_trend_maps.py` (writes `figures/peak_trends.{png,pdf}`).

## Guidance

- **Bulletin 17C** assumes a stationary record. It asks you to test for trends and
  step changes and attribute them. Where a known watershed change caused the shift
  (a dam, diversion or urbanization), fit only the homogeneous record or adjust the
  record to current conditions. It prescribes no time-varying distribution for
  climate-driven trends.
- **USACE ETL 1100-2-3** (Nonstationarity Detection Tool) follows the same sequence:
  detect, attribute, then decide. A change counts only when several tests agree,
  the result is robust and there is a physical cause.
- Fitting a distribution whose parameters drift with time (Salas & Obeysekera 2014;
  Read & Vogel 2015) is still research practice; Serinaldi & Kilsby 2015 caution
  against it.
- NWS thresholds reflect current channel conditions and impacts. So the most
  relevant return period is the one under present-day conditions, which argues for
  a recent-record or post-change fit where a change is real.

## Method

- **Peaks:** the systematic peaks the EMA fit uses: code 1 (daily mean) and code 7
  (historical) are dropped. Censored years (codes 4/8) are ranked at their recorded
  value. Only sites with at least 30 years are tested.
- **Trend:** Mann–Kendall with the Hamed–Rao variance correction, lags 1–3 only, and
  the variance only ever inflated. The standard all-lags version rejected 11% of
  white-noise series against a nominal 5%; the restricted version rejects 4%. It
  rejects 8% of AR(1) φ = 0.3 series (14% uncorrected) and detects a 0.02 sd/yr
  trend over 75 years 93% of the time. The trend size is the Sen slope of log10
  peak, in % per decade.
- **Step change:** Pettitt's test, giving the first water year after the change and
  the after/before change in median peak.
- **Multiple testing:** Benjamini–Hochberg false-discovery rate of 5% across sites,
  applied separately to each test.

## Results

2,930 of 3,312 QC sites were tested; 382 have fewer than 30 years.

| | Unregulated (1,490) | Regulated (1,440) |
|---|---:|---:|
| Trend up | 7% | 4% |
| Trend down | 2% | 17% |
| Step change up | 5% | 4% |
| Step change down | 2% | 18% |
| Any significant result | 10% (153) | 24% (352) |

- **Regulated gages hold most of the non-stationarity.** 284 regulated sites show a
  downward trend or step drop. The change points cluster in the mid-1970s (median
  1975), with a median change of −46%, which fits dam construction. They
  concentrate in the Great Plains from Texas to the Dakotas, and in the West.
- **Unregulated gages are mostly stationary**, but 116 show an upward trend or step
  increase (median trend +9% per decade). These cluster in Illinois/Indiana, the Red
  River of the North and the Northeast, consistent with published regional trend
  studies. Bulletin 17C has no adjustment for these.
- **The regulation-onset proxy is weak.** The proxy is the first year NWIS flags a
  peak with code 5 or 6. Only 70 significant change points fall within 5 years of
  it, because 565 of the 1,170 sites with those codes carry them from their first
  year (regulation predates the record).

## Dam attribution (2026-10-21)

Code: `src/extract_gagesii_basins.py` writes `metadata/gage_basins.parquet` from the
GAGES-II boundary shapefiles, which cover all 3,312 QC gages. (The NLDI basin service
rate-limited us after ~800 requests.) `src/attribute_change_points.py` writes
`ffa/change_attribution.parquet`, using NID dams from
`~/data/flood_hazard/nid/nid_dams.parquet` (the 2026 national GeoPackage).

- Basin storage from NID ÷ (RUNAVE7100 × area) reproduces the GAGES-II DOR
  (log r = 0.96, median ratio 1.02).
- **Rule:** a change point is dam-attributed when dams completed within
  [change_wy − 5, change_wy + 1] add DOR ≥ 0.05.
- **Control:** the same window, centred on the record midpoint, at sites without a
  change point.

| | Change points | Dam-attributed | Control rate |
|---|---:|---:|---:|
| Regulated | 303 | 113 (37%) | 11% |
| Unregulated | 104 | 0 | 0.3% |

Of the 190 regulated change points not attributed:
- 147 had substantial storage (DOR ≥ 0.10) built during the record, but the change
  point falls well after the storage build-out, mostly 10–30+ years later. Gradual,
  multi-dam build-out probably smears the single Pettitt split; operating rules,
  small farm ponds below the NID size threshold and groundwater pumping in the Plains
  are other candidates.
- 43 have no storage built during the record.

For comparison, 58% of regulated control sites also had DOR ≥ 0.10 built during
their record, so that alone does not single out change sites.

### Diversions and urbanization (GAGES-II, undated)

NWIS code 6 means "regulation *or diversion*", so NID storage only covers half of it.
The GAGES-II proxies below have no dates, so they can only compare groups, not time
a change. The comparison uses Mann–Whitney tests against the control sites.

- **Regulated, storage built but change not aligned (147 sites):** more canal length
  (median 0.52% vs 0.11%, p = 0.015) and roughly three times the irrigated
  agriculture (0.46% vs 0.16% of the basin, p < 0.001). This fits diversion for
  irrigation, and/or groundwater pumping, lowering peaks. Dam-attributed sites look
  like the control sites on these measures.
- **Unregulated, step increases and upward trends:** mostly urban. 49% of
  step-increase sites and 48% of upward-trend sites have ≥ 20% developed land
  (NLCD 2006), against 11% of sites with no trend or change point (p < 1e-13).
  Step-decrease sites have none. Their higher freshwater withdrawals are a
  population signal, not diversion. So the unregulated upward signal is largely
  **urbanization**, a watershed change Bulletin 17C does cover (adjust or use the
  recent record), not a climate trend.

### Code-6 onset as a start date

"Onset" is the first water year with a code-6 peak, counted only when it is more than
5 years after the record starts ("mid-record").

- **It is a physical marker.** At dam-attributed sites with a mid-record onset (81),
  the onset is within ±2 years of the NID dam year in 60% of cases (±5 years: 70%),
  with a median offset of 0.
- **It persists.** Once a site is flagged, the median share of later years flagged
  is 100%.
- **It is rarely administrative.** Only 8% of mid-record onsets share a state and
  year with 4 or more other sites.
- **Across all regulated change points with a mid-record onset,** 45% fall within
  ±5 years of it, against 17% at control sites.
- **But it explains few of the unattributed changes.** 86 of the 190 unattributed
  regulated change points have a mid-record onset. Only 17 fall within ±5 years of
  it, and 46 come more than 10 years after. The flag marks when regulation or
  diversion began, but the peaks keep changing for decades afterwards (expanding
  diversions), so the post-onset record may itself not be stationary.

**Implication for refits:** pinning a homogeneous period to the Pettitt year catches
only the clear cases. The alternative is a physically defined start of present-day
regulation, e.g. the year upstream NID storage reached 90% of today's, which
applies to every regulated site with storage built during the record.

## Post-cutoff refits (2026-10-21)

Code: `src/refit_post_cutoff.py` writes `ffa/post_cutoff_fits.parquet`. Cutoff rule,
for each regulated site tested in the screen:

1. The later of two physical dates:
   - the year after the largest upstream NID dam was completed, counted only if
     DOR ≥ 0.05 was built during the record (originally the year storage reached
     90% of today's; see the sensitivity section below);
   - the code-6 onset year.

   Either date counts only when it falls more than 5 years into the record.
2. Otherwise, the Pettitt year, if significant.
3. Otherwise, no cutoff.

Sites need ≥ 30 systematic years after the cutoff to be refit. The refit uses the
pipeline's own worker (peak classification, MGBT, EMA, station skew, above-rating
lower bounds).

- **Validation:** refitting with the cutoff at the first year reproduces
  `flood_frequency.parquet` exactly (parameter difference 0.0 at 25 sampled sites).
- **1,440 regulated sites tested:**

  | | Sites |
  |---|---:|
  | Cutoff from storage | 911 |
  | Cutoff from code 6 | 99 |
  | Cutoff from Pettitt | 35 |
  | No cutoff | 395 |
  | Refit | 863 |
  | Cutoff but < 30 years after it | 182 |

  Refit sites keep a median of 53 years after the cutoff; 4 refits are degenerate.
- **Trends removed:** 258 refit sites had a significant full-record trend. After the
  cutoff, 40 still do (BH FDR 5%), 32 of them downward. Those are gradual,
  continuing declines; the cutoff did not reach a homogeneous period, so flag them.
- **Return periods under present-day regulation are longer** (post/full ratio):

  | | Sites | Moderate | Major |
  |---|---:|---:|---:|
  | Detected change | 300 | 1.90 | 2.51 |
  | No detected change | 563 | 1.07 | 1.11 |

  Across all refits, the median ratio is 1.04 at action, 1.09 at minor, 1.16 at
  moderate and 1.22 at major, and the return period is longer at about two-thirds of
  sites at every level. The small, consistent rise at sites with no detected change
  probably reflects regulation effects too weak to detect, and possibly some
  short-record skew effect.
- **Extremes:** 86 sites have a major return period more than 5× longer. Their
  post-cutoff records are as long as average (52 years), 93% are step drops, and
  their station skew falls by a median of −0.43. The thinner upper tail is what
  lengthens the major return period. These rest on station skew from about 50
  years, so they are the least certain.

### Sensitivity of the storage cutoff

The 90% storage fraction is a heuristic, not from Bulletin 17C, which gives no rule
for when regulation is "in place". It trades homogeneity against record length.

| Storage rule | Storage cutoffs | Median cutoff year | Refit | Too short | Same as 90% | Within 5 yr of 90% |
|---|---:|---:|---:|---:|---:|---:|
| 50% of today's | 766 | 1965 | 852 | 106 | 54% | 65% |
| 75% | 857 | 1969 | 879 | 133 | 69% | 80% |
| **90%** | 911 | 1972 | 863 | 182 | — | — |
| 100% | 1,010 | 2005 | 363 | 726 | 23% | 29% |
| Largest upstream dam | 767 | 1966 | 834 | 119 | 61% | 71% |

Any rule below 100% gives the same cutoff year at most sites (median shift 0). At
100%, any small late dam moves the cutoff forward, and half the sites end up too
short to refit.

**Main rule switched to the largest upstream dam (2026-10-21).** The 90% rule is
kept as `--storage-frac 0.9`. Full refits under both:

| | Largest dam (main) | 90% storage |
|---|---:|---:|
| Refit / too short / no cutoff | 833 / 118 / 489 | 863 / 182 / 395 |
| Median years after cutoff | 58 | 53 |
| Detected-change sites (352): refit / too short / no cutoff | 304 / 33 / 15 | 300 / 41 / 11 |
| …still trending after cutoff | 57 | 35 |
| …median refit ÷ full, moderate / major | 1.89 / 2.25 | 1.90 / 2.51 |
| Sites trending after cutoff (all refits) | 62 | 40 |

At the 781 sites refit under both rules, 61% have the same cutoff, and the major
return period matches within ±10% at 74% (median ratio 1.00). The largest-dam rule
keeps more record and is easier to explain. The 90% rule removes more trends, because
storage that keeps growing after the largest dam still changes the peaks.

### Refit integrity: record-shortening control (2026-10-21)

Does fitting on a shorter, more recent record lengthen return periods by itself?

**Control:** 1,210 unregulated sites with no significant trend or change point,
each refit on its most recent years. Post-cutoff lengths are drawn from the real
refits (median 49 years kept; at least 10 years dropped, at least 30 kept). The
script is in the session scratchpad; its logic is to call `refit()` with a recent
cutoff.

| Level | Control | Regulated refits, no change detected | Regulated refits, change detected |
|---|---:|---:|---:|
| Action | 0.99 (longer at 42%) | 1.02 | 1.22 |
| Minor | 0.99 (40%) | 1.03 | 1.51 |
| Moderate | 0.97 (41%) | 1.05 | 1.89 |
| Major | 0.93 (39%) | 1.10 | 2.25 |

All values are median present-day ÷ full-record return period. Median station skew
in the control barely moves (−0.036 → −0.023).

- Shortening a stationary record does **not** lengthen return periods; it shortens
  them slightly. The regulated increases are not a method artifact. Against the
  control, they are, if anything, slightly understated.
- The control's major-level P10–P90 (0.64–1.52) is the sampling noise of a
  shortened refit. Shifts at changed sites lie far outside it.

### Refit integrity: lingering trends and skew (2026-10-21)

**Lingering trends.** 62 refit sites keep a significant post-cutoff trend (49 of
them downward). Moving their cutoff to the later of the largest dam and the year
storage reached 90% of today's:

- 32 have a later 90%-storage date; 27 of those keep ≥ 30 years.
- After refitting, 17 fall below the main run's BH threshold (p = 0.0035). The
  median new p is 0.010, so the trend weakened rather than vanished. 10 still trend.
- At the 17 sites, the major return period is a further 1.25× longer at the median.
- **45 sites remain to flag as gradual change:** 10 still trending, 5 too short,
  30 with no later storage date.

**Skew sensitivity.** Present-day return periods recomputed with the post-cutoff
location and scale but the full-record skew (median ratios):

| Level | No change detected: free / fixed skew | Change detected: free / fixed skew |
|---|---:|---:|
| Action | 1.02 / 1.02 | 1.22 / 1.24 |
| Minor | 1.03 / 1.05 | 1.50 / 1.53 |
| Moderate | 1.05 / 1.09 | 1.89 / 1.93 |
| Major | 1.10 / 1.16 | 2.32 / 2.30 |

- The shifts come from the location and scale (smaller typical floods), not from
  skew noise.
- **The 86 extreme sites** (major ratio > 5): with the full-record skew, the median
  ratio is 8.2, 55 stay above 5 and 82 above 2. Their skew changes by a median of
  −0.40, against 0.18 in absolute terms across all refits. The extremes are real
  drops in flood size, and the skew change is not what drives them.

### Why change years often lag the largest dam (2026-10-21)

This compares 299 regulated change points with the completion year of the largest
upstream dam:

| Change relative to the largest dam | Sites |
|---|---:|
| More than 2 years before | 61 |
| 0–5 years after | 80 |
| 6–15 years after | 31 |
| More than 15 years after | 127 |

For the 127 that come more than 15 years after:

- **It is the right dam.** The largest dam alone holds DOR 0.34 at the median.
- **The dam comes early in the record.** It was completed a median of 12 years into
  the record. 48% are no more than 10 years in, and 29% predate the record entirely
  (for these, the change cannot come from that dam).
- **Pettitt places early changes late.** In synthetic 80-year series with a −0.8 sd
  step:

  | True step | Detected | Median lag of estimate | Lag > 15 yr |
  |---|---:|---:|---:|
  | Year 7 | 8% | +25 yr | 73% |
  | Year 7, then continued decline | 67% | +29 yr | 89% |
  | Year 15, then continued decline | 94% | +16 yr | 53% |
  | Year 40 | 84% | +0 yr | 1% |

  A change near the start of a record is placed decades late, especially when peaks
  keep falling afterwards.
- **Storage kept growing.** 58% gained DOR ≥ 0.05 from dams built between the
  largest dam and the change.
- **Dam modifications don't explain it.** An upstream dam was modified within
  ±5 years of the change at 28% of these sites, against 12–26% in the other groups.

**Implications**

- Many "storage built, change later" sites are probably dam-driven changes that
  Pettitt has placed late, not diversions. The attribution count of 113 is a lower
  bound on dam-driven change.
- Pettitt years are not reliable dates for a change early in a record. This supports
  cutting at physical dates (dam completion, code-6 onset); Pettitt is used as the
  fallback at only 48 sites.

### Second pass built in (2026-10-21)

`refit_post_cutoff.py` now refits sites still trending after the first cutoff:

- **Second-pass cutoff:** the year after storage reached 90% of today's, when that is
  later than the first cutoff and leaves ≥ 30 years.
- **Kept** when the trend falls below the first pass's BH threshold; recorded as
  `cutoff_source = "storage90"`, `second_pass = True`.
- **Flagged** `gradual_change` when the site still trends.

Results: 27 tried, 17 resolved, **45 flagged `gradual_change`**.

Updated headline numbers:
- Changed sites, median present-day ÷ full-record: 1.23 at action, 1.56 at minor,
  1.99 at moderate and 2.42 at major.
- Regulated present-day KM medians: 2.1, 4.3, 12.8 and 50.9 yr.

### Flag for unrefit records (2026-10-21)

`post_cutoff_fits.parquet` now covers all 1,555 regulated QC sites. The 115 with
< 30 years (never screened) appear with `status = "short_record"`.

`nonstationary_full_record` marks sites that keep a full-record fit despite evidence
of change. Their values mix pre- and post-regulation floods; flag them, don't refit.

| Reason | Sites |
|---|---:|
| Change detected; < 30 yr after the cutoff | 33 |
| Change detected; no dam or code-6 date | 15 |
| < 30 yr record; dam or code-6 date mid-record | 7 |
| **Total** | **55** |

A 20-year refit was considered and rejected: station skew from so few years is too
noisy. Storage built during the record is now computed inside the refit script for
every site. It matches `change_attribution.parquet` at all 1,440 screened sites
(maximum difference ~1e-15).

## Unregulated gages: recent-record refits (2026-10-21)

Code: `src/refit_unregulated_recent.py` writes `ffa/unregulated_recent_fits.parquet`.
Scope: 153 unregulated screened gages with a significant trend or step change
(116 up, 37 down; 50 urbanizing, all up).

- **Cutoff:** the Pettitt year at step sites (104). These changes fall mid-record
  (median 1979), where Pettitt locates them well. Trend-only sites (49) use the most
  recent 30 systematic years.
- **Role (user decision):** urbanizing basins (≥ 20% developed) get the refit as
  the **primary** present-day value, since Bulletin 17C treats urbanization as a
  watershed change. The other 103 get a **sensitivity** value only.
- **Results:** 131 refit; 22 step sites have < 30 systematic years left (median 24),
  so they are `too_short` and flagged `nonstationary_full_record`. 1 refit is
  degenerate.

| Role | Direction | Refit | Action | Minor | Moderate | Major |
|---|---|---:|---:|---:|---:|---:|
| Primary (urbanizing) | up | 38 | 0.79 | 0.73 | 0.67 | 0.66 |
| Sensitivity | up | 62 | 0.84 | 0.76 | 0.68 | 0.68 |
| Sensitivity | down | 30 | 1.33 | 1.60 | 1.81 | 1.59 |

Values are median present ÷ full-record return period. All lie well outside the
stationary control (0.93–0.99).

- **Gradual change:** only 2 sites pass BH after the cutoff, but 13 have p < 0.05,
  10 of them in the 30-year windows. Post-cutoff power is low with 30–50 years and
  131 tests, so treat the absence of a flag as weak evidence.
- **Unregulated KM medians barely move** (major 48.8 → 48.8 with present-day values
  substituted at urban sites, or at all 153). The regulated vs unregulated comparison
  stands.

## Figures and tables

`make_nonstationarity_figures.py` writes the `change_attribution` and
`post_cutoff_refits` figures to `figures/`, and CSVs plus
`nonstationarity_tables.md` to `~/data/flood_hazard/tables/`.

**Headline: the regulation result in the deck was an artifact of non-stationary
records.** Kaplan–Meier medians (yr), lower bounds treated as censored:

| Level | Unregulated | Regulated, full record | Regulated, present day |
|---|---:|---:|---:|
| Action | 1.5 | 2.0 | 2.1 |
| Minor | 2.9 | 3.9 | 4.3 |
| Moderate | 10.3 | 10.4 | 12.8 |
| Major | 48.8 | 37.4 | **50.9** |

Under present-day fits, regulated gages reach major flood stage about as often as
unregulated ones (50.9 vs 48.8 yr), not more often (37.4). Full-record fits include
pre-dam floods that the reservoirs now prevent. Action and minor stay rarer at
regulated gages.

Change-point causes: 113 dam completed near the change, 147 storage built with the
change later, 47 urbanizing basin, 100 no identified cause. The 65 unregulated
sites with no identified cause have more irrigated agriculture than the control
(0.52% vs 0.09%), which could point to irrigation or groundwater pumping.

## Next steps

1. **Attribution:** join dam completion years from the National Inventory of Dams
   (upstream of each gage) to separate dam-driven change points from
   climate/land-use ones.
2. **Regulated sites with a dam-attributed change:** refit LP3 on the post-change
   record (the Bulletin 17C homogeneous-period approach). Compare the threshold
   return periods and note the record length lost, which is the trade-off.
3. **Unregulated sites with an upward trend:** these are mostly urbanizing basins
   (see the diversions and urbanization section). Treat them as a watershed change:
   add a recent-period refit as a sensitivity estimate rather than a time-varying
   model. Imperviousness over time (NLCD series) could time the change.
4. **Data release:** add trend and change-point flags (`trend_dir`, `pettitt_sig`,
   `change_wy`) and, where computed, the post-change or recent-period return period.
