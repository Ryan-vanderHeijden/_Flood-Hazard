# NWIS Peak Qualification Code Analysis

Analysed from `annual_peaks.parquet` (236,358 peaks across 3,332 sites).

> **Corrected 2026-09-01, and the pipeline re-run.** An earlier version of this
> file assigned the wrong meanings to codes **4, 6, 8 and R**, and
> `compute_flood_frequency.py` and `compute_regulation.py` were written against
> those wrong meanings. The definitions below are copied verbatim from the NWIS
> RDB header served with every peak file (e.g.
> `https://nwis.waterdata.usgs.gov/nwis/peak?site_no=01091500&format=rdb`).
> Both modules now use them, and every FFA output has been regenerated. The
> mistake and its measured consequences are documented at the bottom.
>
> The likely origin of the error: codes 4, 6 and 8 in the earlier table are
> close to the **gage-height** qualification codes (`gage_ht_cd`), which are a
> different code list served in the same header. `gage_ht_cd` 4 really is
> "gage height below minimum recordable elevation".

## Official `peak_cd` definitions (NWIS)

| Code | Peaks | % peaks | Sites | % sites | Official meaning |
|------|------:|--------:|------:|--------:|---------|
| 6    | 52,885 | 22.4% | 871 | 26.1% | Discharge affected by Regulation or Diversion |
| 5    | 25,800 | 10.9% | 570 | 17.1% | Discharge affected **to unknown degree** by Regulation or Diversion |
| C    |  5,907 |  2.5% | 172 |  5.2% | All or part of the record affected by Urbanization, Mining, Agricultural changes, Channelization, or other |
| 2    |  4,902 |  2.1% | 1,585 | 47.6% | Discharge is an Estimate |
| 1    |  3,622 |  1.5% | 643 | 19.3% | Discharge is a Maximum Daily Average |
| 7    |    974 |  0.4% | 575 | 17.3% | Discharge is an Historic Peak |
| 9    |    639 |  0.3% | 396 | 11.9% | Discharge due to Snowmelt, Hurricane, Ice-Jam or Debris Dam breakup |
| R    |    594 |  0.3% | 240 |  7.2% | **Revised** |
| 8    |    176 |  0.1% | 74  |  2.2% | Discharge actually **greater** than indicated value |
| F    |     91 |  0.0% | 5   |  0.2% | Peak supplied by another agency |
| 3    |     66 |  0.0% | 57  |  1.7% | Discharge affected by Dam Failure |
| 4    |     58 |  0.0% | 40  |  1.2% | Discharge **less** than indicated value, which is Minimum Recordable Discharge at this site |
| O    |     15 |  0.0% | 12  |  0.4% | Opportunistic value not from systematic data collection |

`A`, `Bd` and `Bm` (date of occurrence unknown or inexact) do not appear in this
extract and carry no hydrological meaning for FFA.

## Treatment

| Code | Treatment | Reason |
|------|-----------|--------|
| **4** | **Left-censored in EMA**, at its own recorded value | The peak that year was below the minimum recordable discharge. B17C uses a per-observation perception threshold, not a site-wide one. 58 peaks, 40 sites. |
| **8** | **Right-censored in EMA**, at its own recorded value | The peak exceeded the indicated value; the value is a lower bound. 176 peaks, 74 sites. |
| **7** | Historical peak with its own perception threshold | Large, well-documented pre-systematic events; EMA incorporates them with a weighted historical period. |
| **1** | **Drop peak** | A maximum daily average is not an instantaneous peak and biases the fit downward. It is strictly a lower bound, so right-censoring would recover the year — a possible future improvement, 3,622 peaks. |
| **5, 6** | Keep as ordinary systematic peaks; count toward regulation in `compute_regulation.py` | Both mark regulation or diversion. The peak itself is fully observed. |
| **C** | Keep; flagged separately as an anthropogenic-change (stationarity) concern | Urbanisation, channelisation, mining, agriculture. |
| **R** | Keep; carries no information for FFA | "Revised". |
| **2, 9** | Keep | Estimates and snowmelt/hurricane/ice-jam events are uncertain but not systematically biased. |
| **3** | Keep | Dam-failure peaks are arguably outside the flood population and B17C would exclude them; 66 peaks, low priority. |
| **F, O** | Keep | `O` is by definition not systematic and should arguably be excluded from the record length; 15 peaks. |

### What was wrong before

| Code | Was treated as | Scale of the error |
|------|----------------|--------------------|
| **6** | Left-censored in EMA at the site-wide minimum code-6 value | 52,885 peaks, 871 sites — 22% of every annual peak in the file |
| **4** | Ignored | 58 peaks, 40 sites |
| **8** | Dropped as "stage only" | 176 peaks, 74 sites |
| **R** | A regulation flag in `compute_regulation.py` | 594 peaks, 240 sites |

## Measured consequences of the code-6 error

Every code-6 peak was passed to EMA as "this year's peak was below the smallest
code-6 value at this site." At a site where most of the record carries code 6
that forces the fitted LP3 to place most of its mass far below the observed
peaks, which inflates the fitted scale and collapses the lower tail.

**Sites lost.** Of the 872 sites with any code-6 peak, only 278 survive the
`record_ok` / `degenerate_fit` gates — 32%, against 99.2% for sites with no
code-6 peak. Where more than 75% of the record carries code 6, 82% of sites fail
`record_ok` outright, because fewer than ten peaks are left once the rest are
moved into the censored bucket.

| Share of record with code 6 | Sites | Passing QC |
|---|---:|---:|
| none | 2,462 | 99.2% |
| 1–25% | 46 | 82.6% |
| 25–50% | 74 | 71.6% |
| 50–75% | 200 | 54.0% |
| >75% | 552 | 14.3% |

**Fits corrupted.** Among sites that did pass, those with any code-6 peak have a
median fitted `lp3_scale` of 0.77 against 0.26 for the rest, and their fitted Q2
is a median 0.38× the observed median annual peak — the two quantities are the
same thing by definition. 162 sites hold 94% of the total squared Q2 error
against USGS published station estimates.

## After the fix

Pipeline re-run 2026-09-01 (`run_ffa.py`, cached `annual_peaks.parquet`, no
refetch), then `compute_regulation.py` and `validate_against_streamstats.py`.

| | before | after |
|---|---:|---:|
| Sites clearing QC | 2,720 | **3,308** |
| — short record (<10 peaks) | 476 | **18** |
| — degenerate fit | 171 | **8** |
| — high censoring flag | 826 | **1** |
| Sites with a published estimate to compare | 1,601 | **1,786** |
| Q2 vs published station — R² (log) | 0.804 | **0.970** |
| Q2 vs published station — RMSE (log) | 0.255 | **0.099** |
| Q10 — R² | 0.956 | 0.967 |
| Q500 — R² | 0.873 | **0.917** |
| Fitted Q2 below half the empirical median | 162 sites | **3 sites** |
| Failing the ±25% Q2 gate | 266 (9.8%) | **76 (2.3%)** |

**589 gages gained, 1 lost.** The loss is 06813000, whose single code-4 peak is
now correctly censored and pushes station skew from −2.14 to −3.57, past the
`degenerate_fit` gate at |skew| > 3. It is flagged rather than silently wrong.

Of the 2,720 sites that already passed QC, 2,360 are bit-identical — they carry
no code-6 peaks. The 360 that changed have a median |Δ lp3_scale| of 0.245.

Sites with code-4 or code-8 censoring behave normally afterwards: 40 and 61
sites respectively, median fitted Q2 exactly 1.00× the empirical median annual
peak (P10 0.89).

The residual disagreement that remains sits in ~187 uncensored high-variance
sites, mostly Northern Great Plains prairie basins with strongly negative log
skew. Those look like real hydrology rather than a fit failure — their fitted Q2
is 0.93× the empirical median.

## Consequences of the code-R error

`compute_regulation.py` counted codes 5 and R as regulation. R means "revised".
It now counts codes 5 and 6. Spearman correlation between the peak-code
regulation fraction and NID-storage degree of regulation:

| Signal | Spearman vs DOR |
|---|---:|
| old, 5 or R | 0.205 |
| R alone | −0.035 |
| corrected, 5 or 6 | **0.622** |

Regulation classes on the QC-passed set move accordingly — unregulated 43.2% →
35.6%, major 16.3% → **31.9%** — partly because the labels are right and partly
because the ~590 recovered gages are disproportionately regulated.

150 sites are labelled regulated on code R alone; their median DOR is 0.007,
against 0.049 for the gage set as a whole — they are among the *least* regulated
gages in the file. The earlier conclusion that the two regulation signals "agree
only weakly" was mostly an artefact of this mapping: with the correct codes they
agree well.

## Regulation prevalence under the corrected codes

Sites by share of peaks carrying a regulation code (5 or 6):

| Threshold | Sites |
|-----------|------:|
| > 25% | 1,202 (36.1%) |
| > 50% | 1,112 (33.4%) |
| > 75% | 861 (25.8%) |
| 100%  | 504 (15.1%) |

For comparison, the current (code 5 or R) rule flags 413 sites above 25%. The
corrected rule flags nearly three times as many.

Codes 5 and 6 are near-disjoint in practice — only 40 sites carry both — which
is consistent with different USGS districts preferring one annotation over the
other rather than the two meaning genuinely different things in the field.
