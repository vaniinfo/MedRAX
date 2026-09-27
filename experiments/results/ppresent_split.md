# Held-out test: universal 0.50 vs per-finding threshold for p_present

Split: stratified by image, 70/30, seed 20260927 (381 development / 163 test images). Thresholds (Youden's J) and Platt calibration fitted on development only and frozen at 2026-09-27T11:14:48 (`frozen_dev_params.json`) before the test set was evaluated. 95% CIs: 2000-sample image-level bootstrap of the test set.

**Enriched corpus** (studies selected per finding): AUC and sensitivity/specificity transfer; absolute positive rates are optimistic for a real clinic.

## Sample sizes

| Finding | Dev pos / neg | Test pos / neg |
|---|---|---|
| cardiomegaly | 115 / 266 | 49 / 114 |
| pleural effusion | 84 / 297 | 36 / 127 |
| pneumothorax | 20 / 361 | 8 / 155 |
| consolidation | 76 / 305 | 33 / 130 |
| pulmonary edema | 31 / 350 | 14 / 149 |
| atelectasis | 113 / 268 | 48 / 115 |

## Discrimination and the threshold comparison (test set)

| Finding | AUC | Frozen T | Sens @0.50 | Sens @T | Δ sens (T − 0.50) | Spec @0.50 | Spec @T | Δ spec (T − 0.50) | Δ Youden J |
|---|---|---|---|---|---|---|---|---|---|
| cardiomegaly | 0.93 [0.89–0.97] | 0.443 | 0.88 [0.78–0.96] | 0.92 [0.83–0.98] | 0.04 [0.00–0.11] | 0.85 [0.78–0.91] | 0.83 [0.76–0.90] | -0.02 [-0.05–0.00] | 0.02 [-0.03–0.09] |
| pleural effusion | 0.96 [0.93–0.99] | 0.556 | 0.89 [0.78–0.98] | 0.89 [0.78–0.98] | 0.00 [0.00–0.00] | 0.86 [0.80–0.92] | 0.87 [0.81–0.93] | 0.02 [0.00–0.04] | 0.02 [0.00–0.04] |
| pneumothorax | 0.93 [0.76–1.00] | 0.119 | 0.62 [0.25–1.00] | 0.75 [0.40–1.00] | 0.12 [0.00–0.43] | 1.00 [1.00–1.00] | 0.94 [0.90–0.97] | -0.06 [-0.10–-0.03] | 0.07 [-0.09–0.37] |
| consolidation | 0.89 [0.83–0.94] | 0.237 | 0.61 [0.43–0.77] | 0.85 [0.71–0.96] | 0.24 [0.10–0.41] | 0.91 [0.86–0.95] | 0.74 [0.66–0.81] | -0.17 [-0.23–-0.11] | 0.07 [-0.09–0.24] |
| pulmonary edema | 0.92 [0.85–0.97] | 0.368 | 0.64 [0.36–0.89] | 0.79 [0.54–1.00] | 0.14 [0.00–0.33] | 0.92 [0.88–0.96] | 0.87 [0.82–0.92] | -0.05 [-0.08–-0.01] | 0.10 [-0.06–0.30] |
| atelectasis | 0.80 [0.72–0.87] | 0.448 | 0.79 [0.67–0.90] | 0.85 [0.75–0.95] | 0.06 [0.00–0.14] | 0.64 [0.55–0.73] | 0.61 [0.51–0.70] | -0.03 [-0.07–-0.01] | 0.03 [-0.04–0.11] |

Δ columns are paired on the same test images; a CI excluding 0 means the per-finding threshold changes that metric beyond bootstrap noise.

## Calibration (test set; lower is better)

| Finding | Brier raw p | Brier calibrated | ECE raw p | ECE calibrated |
|---|---|---|---|---|
| cardiomegaly | 0.104 [0.070–0.139] | 0.095 [0.066–0.125] | 0.079 [0.049–0.134] | 0.010 [0.018–0.079] |
| pleural effusion | 0.088 [0.060–0.118] | 0.067 [0.044–0.093] | 0.100 [0.059–0.141] | 0.026 [0.020–0.075] |
| pneumothorax | 0.016 [0.003–0.034] | 0.015 [0.001–0.033] | 0.029 [0.011–0.046] | 0.005 [0.001–0.025] |
| consolidation | 0.103 [0.078–0.130] | 0.110 [0.086–0.134] | 0.035 [0.022–0.091] | 0.074 [0.047–0.125] |
| pulmonary edema | 0.072 [0.050–0.098] | 0.056 [0.033–0.082] | 0.083 [0.049–0.123] | 0.011 [0.009–0.053] |
| atelectasis | 0.195 [0.159–0.233] | 0.162 [0.131–0.194] | 0.148 [0.103–0.218] | 0.026 [0.024–0.105] |

## Reliability on the test set, by raw p_present band

Observed positive rate vs mean Platt-calibrated probability in each raw band.

| Finding | 0.0–0.1 | 0.1–0.3 | 0.3–0.5 | 0.5–0.7 | 0.7–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|
| cardiomegaly | 1% obs / 2% cal (n=77) | 15% obs / 13% cal (n=13) | 23% obs / 25% cal (n=13) | 46% obs / 42% cal (n=13) | 65% obs / 65% cal (n=20) | 89% obs / 89% cal (n=27) |
| pleural effusion | 0% obs / 1% cal (n=83) | 5% obs / 8% cal (n=20) | 30% obs / 18% cal (n=10) | 20% obs / 38% cal (n=10) | 47% obs / 63% cal (n=15) | 92% obs / 91% cal (n=25) |
| pneumothorax | 1% obs / 1% cal (n=145) | 0% obs / 10% cal (n=12) | 100% obs / 45% cal (n=1) | – | – | 100% obs / 100% cal (n=5) |
| consolidation | 1% obs / 7% cal (n=77) | 18% obs / 21% cal (n=33) | 29% obs / 31% cal (n=21) | 45% obs / 41% cal (n=20) | 89% obs / 51% cal (n=9) | 100% obs / 71% cal (n=3) |
| pulmonary edema | 0% obs / 1% cal (n=102) | 12% obs / 8% cal (n=25) | 13% obs / 16% cal (n=15) | 27% obs / 25% cal (n=11) | 56% obs / 45% cal (n=9) | 100% obs / 71% cal (n=1) |
| atelectasis | 5% obs / 3% cal (n=38) | 9% obs / 12% cal (n=23) | 26% obs / 25% cal (n=23) | 37% obs / 37% cal (n=35) | 52% obs / 55% cal (n=31) | 69% obs / 79% cal (n=13) |
