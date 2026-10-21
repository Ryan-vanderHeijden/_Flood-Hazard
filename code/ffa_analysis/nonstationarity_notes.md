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
