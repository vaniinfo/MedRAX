# External validation of CheXagent p_present: Indiana → NIH ChestX-ray14

NIH random sample: 5000 images (one per patient, natural prevalence); supplement: 600 rare-finding images from other patients (enriched; used only for AUC and thresholds, never for calibration). NIH split: stratified 70/30 by image = by patient, seed 20260927. Indiana parameters: `frozen_dev_params.json`, SHA-256 verified (3c14b9441b81…). NIH parameters frozen at 2026-09-27T16:22:24 (`nih_frozen_dev_params.json`) before test scoring. 95% CIs: 2000-sample image-level bootstrap. NIH labels are NLP-mined from reports (noisy).

## Sample sizes (positives / negatives)

| Finding | NIH dev random | NIH dev suppl. | NIH test random | NIH test suppl. |
|---|---|---|---|---|
| cardiomegaly | 80 / 3420 | 115 / 305 | 35 / 1465 | 50 / 130 |
| pleural effusion | 180 / 3320 | 105 / 315 | 77 / 1423 | 45 / 135 |
| pneumothorax | 43 / 3457 | 117 / 303 | 18 / 1482 | 50 / 130 |
| consolidation | 62 / 3438 | 117 / 303 | 27 / 1473 | 50 / 130 |
| pulmonary edema | 24 / 3476 | 115 / 305 | 10 / 1490 | 49 / 131 |
| atelectasis | 218 / 3282 | 63 / 357 | 93 / 1407 | 27 / 153 |

## A / B / C at a glance

AUC and Brier score (lower is better). A: Indiana test, Indiana calibration. B: NIH random sample (all 5,000 are unseen by B), Indiana calibration. C: NIH test (random part), NIH calibration.

| Finding | A AUC | A Brier raw → cal | B AUC | B Brier raw → Indiana-cal | C Brier NIH-cal |
|---|---|---|---|---|---|
| cardiomegaly | 0.93 | 0.104 → 0.095 | 0.94 | 0.074 → 0.047 | 0.019 |
| pleural effusion | 0.96 | 0.088 → 0.067 | 0.92 | 0.062 → 0.043 | 0.035 |
| pneumothorax | 0.93 | 0.016 → 0.015 | 0.90 | 0.019 → 0.017 | 0.009 |
| consolidation | 0.89 | 0.103 → 0.110 | 0.84 | 0.050 → 0.036 | 0.017 |
| pulmonary edema | 0.92 | 0.072 → 0.056 | 0.92 | 0.023 → 0.009 | 0.005 |
| atelectasis | 0.80 | 0.195 → 0.162 | 0.89 | 0.108 → 0.064 | 0.044 |

## Calibration on the NIH test set (random part; natural prevalence)

Same images for all columns. Δ = Indiana-cal − NIH-cal (positive: NIH recalibration is better).

| Finding | Brier raw | Brier Indiana-cal (B) | Brier NIH-cal (C) | Δ Brier | ECE raw | ECE Indiana-cal | ECE NIH-cal |
|---|---|---|---|---|---|---|---|
| cardiomegaly | 0.069 [0.059–0.078] | 0.043 [0.037–0.050] | 0.019 [0.014–0.025] | 0.024 [0.017–0.031] | 0.110 | 0.075 | 0.004 |
| pleural effusion | 0.067 [0.058–0.077] | 0.048 [0.040–0.056] | 0.035 [0.028–0.042] | 0.013 [0.008–0.017] | 0.085 | 0.040 | 0.008 |
| pneumothorax | 0.018 [0.014–0.023] | 0.016 [0.011–0.021] | 0.009 [0.005–0.013] | 0.007 [0.003–0.011] | 0.044 | 0.025 | 0.006 |
| consolidation | 0.048 [0.041–0.055] | 0.034 [0.030–0.039] | 0.017 [0.011–0.023] | 0.018 [0.014–0.022] | 0.080 | 0.092 | 0.000 |
| pulmonary edema | 0.020 [0.015–0.024] | 0.007 [0.005–0.010] | 0.005 [0.002–0.009] | 0.002 [-0.000–0.003] | 0.044 | 0.019 | 0.001 |
| atelectasis | 0.110 [0.100–0.121] | 0.064 [0.058–0.071] | 0.044 [0.037–0.052] | 0.020 [0.015–0.025] | 0.176 | 0.103 | 0.009 |

ECE is biased upward under bootstrap, so its intervals are omitted; use the Brier intervals.

### Reliability by raw p_present band (NIH test, random part)

Observed positive rate / mean Indiana-calibrated / mean NIH-calibrated.

| Finding | 0.0–0.1 | 0.1–0.3 | 0.3–0.5 | 0.5–0.7 | 0.7–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|
| cardiomegaly | 0% / 1% / 0% (n=1104) | 1% / 12% / 2% (n=139) | 3% / 26% / 4% (n=89) | 11% / 40% / 6% (n=61) | 15% / 63% / 12% (n=80) | 37% / 85% / 28% (n=27) |
| pleural effusion | 1% / 1% / 0% (n=1136) | 2% / 6% / 3% (n=127) | 11% / 19% / 9% (n=72) | 5% / 37% / 16% (n=40) | 34% / 63% / 30% (n=67) | 50% / 89% / 60% (n=58) |
| pneumothorax | 0% / 1% / 0% (n=1332) | 2% / 10% / 3% (n=105) | 0% / 35% / 7% (n=27) | 12% / 60% / 12% (n=17) | 50% / 87% / 24% (n=10) | 56% / 98% / 62% (n=9) |
| consolidation | 0% / 5% / 1% (n=1164) | 3% / 21% / 3% (n=156) | 9% / 31% / 5% (n=77) | 8% / 41% / 7% (n=59) | 13% / 53% / 12% (n=39) | 20% / 70% / 21% (n=5) |
| pulmonary edema | 0% / 1% / 0% (n=1325) | 0% / 8% / 2% (n=93) | 3% / 16% / 3% (n=39) | 4% / 26% / 6% (n=24) | 25% / 43% / 12% (n=16) | 67% / 77% / 36% (n=3) |
| atelectasis | 1% / 3% / 0% (n=756) | 1% / 13% / 3% (n=282) | 6% / 25% / 7% (n=161) | 11% / 38% / 14% (n=127) | 29% / 55% / 28% (n=137) | 57% / 77% / 57% (n=37) |

## Operating points on the NIH test set (random + supplement)

Sensitivity / specificity with 95% CI. Thresholds: universal 0.50; Indiana-frozen Youden T; NIH-dev Youden T; NIH-dev thresholds reaching a target sensitivity.

| Finding | Test AUC | Rule | Threshold | Sensitivity | Specificity |
|---|---|---|---|---|---|
| cardiomegaly | 0.93 [0.90–0.95] | universal 0.50 | 0.500 | 0.80 [0.71–0.88] | 0.90 [0.88–0.91] |
| | | Indiana Youden (B) | 0.443 | 0.82 [0.74–0.90] | 0.88 [0.86–0.89] |
| | | NIH Youden (C) | 0.381 | 0.85 [0.77–0.92] | 0.86 [0.84–0.88] |
| | | NIH sens ≥ 0.80 | 0.453 | 0.82 [0.74–0.90] | 0.88 [0.86–0.89] |
| | | NIH sens ≥ 0.90 | 0.160 | 0.92 [0.85–0.97] | 0.77 [0.75–0.79] |
| | | NIH sens ≥ 0.95 | 0.043 | 0.98 [0.94–1.00] | 0.67 [0.64–0.69] |
| pleural effusion | 0.88 [0.84–0.91] | universal 0.50 | 0.500 | 0.70 [0.62–0.79] | 0.91 [0.89–0.92] |
| | | Indiana Youden (B) | 0.556 | 0.68 [0.60–0.76] | 0.92 [0.90–0.93] |
| | | NIH Youden (C) | 0.352 | 0.79 [0.71–0.86] | 0.87 [0.85–0.88] |
| | | NIH sens ≥ 0.80 | 0.401 | 0.76 [0.69–0.84] | 0.88 [0.86–0.90] |
| | | NIH sens ≥ 0.90 | 0.065 | 0.87 [0.81–0.93] | 0.70 [0.68–0.73] |
| | | NIH sens ≥ 0.95 | 0.020 | 0.92 [0.87–0.96] | 0.51 [0.49–0.54] |
| pneumothorax | 0.88 [0.83–0.93] | universal 0.50 | 0.500 | 0.49 [0.36–0.60] | 0.98 [0.98–0.99] |
| | | Indiana Youden (B) | 0.119 | 0.69 [0.58–0.80] | 0.91 [0.89–0.92] |
| | | NIH Youden (C) | 0.128 | 0.69 [0.58–0.80] | 0.91 [0.90–0.93] |
| | | NIH sens ≥ 0.80 | 0.073 | 0.75 [0.65–0.85] | 0.84 [0.82–0.86] |
| | | NIH sens ≥ 0.90 | 0.031 | 0.88 [0.80–0.95] | 0.66 [0.63–0.68] |
| | | NIH sens ≥ 0.95 | 0.020 | 0.96 [0.90–1.00] | 0.52 [0.50–0.54] |
| consolidation | 0.87 [0.83–0.91] | universal 0.50 | 0.500 | 0.53 [0.42–0.64] | 0.92 [0.91–0.93] |
| | | Indiana Youden (B) | 0.237 | 0.79 [0.70–0.88] | 0.84 [0.82–0.86] |
| | | NIH Youden (C) | 0.091 | 0.87 [0.79–0.94] | 0.74 [0.72–0.76] |
| | | NIH sens ≥ 0.80 | 0.077 | 0.88 [0.81–0.95] | 0.72 [0.70–0.74] |
| | | NIH sens ≥ 0.90 | 0.021 | 0.96 [0.92–1.00] | 0.57 [0.54–0.59] |
| | | NIH sens ≥ 0.95 | 0.004 | 0.96 [0.92–1.00] | 0.29 [0.27–0.31] |
| pulmonary edema | 0.94 [0.91–0.96] | universal 0.50 | 0.500 | 0.58 [0.44–0.70] | 0.97 [0.96–0.98] |
| | | Indiana Youden (B) | 0.368 | 0.73 [0.61–0.84] | 0.95 [0.94–0.96] |
| | | NIH Youden (C) | 0.115 | 0.85 [0.75–0.93] | 0.87 [0.85–0.89] |
| | | NIH sens ≥ 0.80 | 0.206 | 0.78 [0.67–0.88] | 0.91 [0.89–0.92] |
| | | NIH sens ≥ 0.90 | 0.068 | 0.90 [0.81–0.97] | 0.84 [0.82–0.85] |
| | | NIH sens ≥ 0.95 | 0.024 | 0.95 [0.88–1.00] | 0.75 [0.73–0.77] |
| atelectasis | 0.88 [0.85–0.91] | universal 0.50 | 0.500 | 0.82 [0.75–0.88] | 0.81 [0.79–0.83] |
| | | Indiana Youden (B) | 0.448 | 0.85 [0.79–0.91] | 0.78 [0.76–0.80] |
| | | NIH Youden (C) | 0.531 | 0.80 [0.72–0.87] | 0.82 [0.80–0.84] |
| | | NIH sens ≥ 0.80 | 0.501 | 0.82 [0.75–0.88] | 0.81 [0.79–0.83] |
| | | NIH sens ≥ 0.90 | 0.274 | 0.93 [0.88–0.97] | 0.66 [0.64–0.69] |
| | | NIH sens ≥ 0.95 | 0.114 | 0.95 [0.91–0.98] | 0.51 [0.49–0.54] |

## AUC by view (NIH test, random + supplement)

| Finding | PA AUC (pos/neg) | AP AUC (pos/neg) |
|---|---|---|
| cardiomegaly | 0.94 (62/1260) | 0.89 (23/335) |
| pleural effusion | 0.88 (83/1239) | 0.88 (39/319) |
| pneumothorax | 0.89 (42/1280) | 0.80 (26/332) |
| consolidation | 0.87 (45/1277) | 0.88 (32/326) |
| pulmonary edema | 0.94 (9/1313) | 0.84 (50/308) |
| atelectasis | 0.89 (82/1240) | 0.82 (38/320) |
