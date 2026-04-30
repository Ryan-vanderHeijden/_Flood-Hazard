# NWIS Peak Qualification Code Analysis

Analysed from `annual_peaks.parquet` (236,274 peaks across 3,332 sites).

## Codes present in dataset

| Code | Peaks | % peaks | Sites | % sites | Meaning |
|------|------:|--------:|------:|--------:|---------|
| 6    | 52,885 | 22.4% | 871 | 26.1% | Discharge < minimum recordable (left-censored) |
| 5    | 25,791 | 10.9% | 570 | 17.1% | Regulated/diverted — unknown degree |
| C    |  5,905 |  2.5% | 172 |  5.2% | Anthropogenic change (urbanization, channelization, etc.) |
| 2    |  4,902 |  2.1% | 1,585 | 47.6% | Estimated discharge |
| 1    |  3,622 |  1.5% | 643 | 19.3% | Max **daily average** — not instantaneous peak |
| 7    |    974 |  0.4% | 575 | 17.3% | Historical peak (pre-gauging / outside systematic record) |
| R    |    594 |  0.3% | 240 |  7.2% | Regulated/diverted |
| 9    |    639 |  0.3% | 396 | 11.9% | Snowmelt / ice-jam / hurricane / debris dam |
| 8    |    176 |  0.1% | 74  |  2.2% | Stage only — discharge not determined |
| 3    |     66 |  0.0% | 57  |  1.7% | Dam failure / ice-jam / debris dam |
| 4    |     58 |  0.0% | 40  |  1.2% | Stage below minimum recordable |
| F    |     91 |  0.0% | 5   |  0.2% | Peak discharge estimate |
| O    |     15 |  0.0% | 12  |  0.4% | Discharge > or < indicated value |

Codes 6 and 7 are handled by the EMA fitting routine. All others were previously passed through unchanged.

## Regulation (codes 5 and R)

Sites with heavy code-5 prevalence:

| Threshold | Sites |
|-----------|------:|
| > 25% code-5 peaks | 413 (12.4%) |
| > 50% code-5 peaks | 343 |
| > 75% code-5 peaks | 256 |
| 100% code-5 peaks  | 168 |

These sites may not represent natural flood frequency. Not yet filtered — decision deferred.

## Treatment decisions

| Code | Treatment | Reason |
|------|-----------|--------|
| **1** | **Drop peak** | Daily average systematically underestimates instantaneous peak; biases LP3 fit downward |
| **8** | **Drop peak** | `peak_va` is a stage value, not a discharge; meaningless in flow-based FFA |
| **6** | **Left-censor (EMA)** | Discharge below threshold of measurement; treated as interval [0, threshold] in EMA fitting |
| **7** | **Keep with perception threshold (EMA)** | Historical peaks are large, well-documented events; EMA incorporates pre-systematic record via separate perception threshold |
| 5, R | Flag (future) | Regulation affects stationarity; extent varies by site |
| C    | Flag (future) | Nonstationarity concern |
| 2    | Keep | Estimates are uncertain but not systematically biased |
| 9    | Keep | Rare, individually valid events |
| 3    | Keep | Rare, individually valid events |
