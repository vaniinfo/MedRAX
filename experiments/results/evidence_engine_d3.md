# Evidence Engine D3: dual-evidence REFUTING rule

Protocol: `experiments/evidence_engine_d3_design.md` (committed before this run). Empirical bands built from NIH development only and frozen to `nih_dev_empirical_bands.json` (SHA-256 f6fb7df6f27b…) before the test split was read. Platt boundaries identical to D2. Test CIs: 2,000-sample image-level bootstrap.

## Frozen empirical bands (NIH development, random + supplement)

Per band: positives / negatives, empirical LR, exact upper 95% bound (Clopper–Pearson). ✓ = upper bound ≤ 0.2 (a Platt-REFUTING case in this band can be confirmed).

| Finding | Dev pos / neg | 0–0.02 | 0.02–0.05 | 0.05–0.1 | 0.1–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–0.9 | 0.9–1 |
|---|---|---|---|---|---|---|---|---|---|---|
| cardiomegaly | 195 / 3725 | 4/2240: 0.03 ≤0.09 ✓ | 6/292: 0.39 ≤0.94 | 3/193: 0.30 ≤0.99 | 9/215: 0.80 ≤1.70 | 11/250: 0.84 ≤1.66 | 17/186: 1.75 ≤3.15 | 42/191: 4.20 ≤6.30 | 33/84: 7.50 ≤12.7 | 70/74: 18.1 ≤27.5 |
| pleural effusion | 285 / 3635 | 14/1910: 0.09 ≤0.16 ✓ | 11/515: 0.27 ≤0.52 | 8/309: 0.33 ≤0.72 | 8/237: 0.43 ≤0.95 | 15/230: 0.83 ≤1.54 | 23/148: 1.98 ≤3.44 | 41/138: 3.79 ≤5.94 | 56/91: 7.85 ≤12.2 | 109/57: 24.4 ≤37.1 |
| pneumothorax | 160 / 3760 | 8/1914: 0.10 ≤0.20 ✓ | 14/966: 0.34 ≤0.59 | 17/429: 0.93 ≤1.58 | 7/226: 0.73 ≤1.67 | 23/136: 3.97 ≤6.83 | 23/35: 15.4 ≤32.0 | 21/32: 15.4 ≤33.2 | 14/6: 54.8 ≤243.2 | 33/16: 48.5 ≤113.9 |
| consolidation | 179 / 3741 | 17/2103: 0.17 ≤0.27 | 13/408: 0.67 ≤1.22 | 12/299: 0.84 ≤1.60 | 12/235: 1.07 ≤2.07 | 19/273: 1.45 ≤2.48 | 33/180: 3.83 ≤6.00 | 54/182: 6.20 ≤8.92 | 13/42: 6.47 ≤14.9 | 6/19: 6.60 ≤23.4 |
| pulmonary edema | 139 / 3781 | 5/2679: 0.05 ≤0.12 ✓ | 2/333: 0.16 ≤0.64 | 8/215: 1.01 ≤2.22 | 12/191: 1.71 ≤3.34 | 23/163: 3.84 ≤6.45 | 26/108: 6.55 ≤11.2 | 36/58: 16.9 ≤29.1 | 15/19: 21.5 ≤56.7 | 12/15: 21.8 ≤65.7 |
| atelectasis | 281 / 3639 | 3/1012: 0.04 ≤0.12 ✓ | 7/491: 0.18 ≤0.41 | 2/351: 0.07 ≤0.29 | 10/404: 0.32 ≤0.64 | 21/495: 0.55 ≤0.90 | 29/370: 1.02 ≤1.57 | 83/389: 2.76 ≤3.63 | 81/100: 10.5 ≤15.4 | 45/27: 21.6 ≤42.6 |

## Primary: NIH test, random part (natural prevalence)

Rule: primary 5 / 0.2. REFUTING state: coverage (share of all cases), empirical LR [95% CI] (✓ point ≤ 0.2; ✓✓ upper CI ≤ 0.2), and positives classified REFUTING (false reassurance).

| Finding | Platt source | Pos / neg | D2 Platt-only REFUTING | positives in it | **D3 dual-evidence REFUTING** | positives in it |
|---|---|---|---|---|---|---|
| cardiomegaly | NIH | 35 / 1465 | 58.5%, 0.05 [0.00–0.17] ✓✓ | 1 / 35 (3%) | **58.5%, 0.05 [0.00–0.17] ✓✓** | 1 / 35 (3%) |
| cardiomegaly | Indiana | 35 / 1465 | 70.7%, 0.04 [0.00–0.14] ✓✓ | 1 / 35 (3%) | **60.6%, 0.05 [0.00–0.16] ✓✓** | 1 / 35 (3%) |
| pleural effusion | NIH | 77 / 1423 | 58.1%, 0.19 [0.08–0.32] ✓ | 9 / 77 (12%) | **52.5%, 0.19 [0.08–0.32] ✓** | 8 / 77 (10%) |
| pleural effusion | Indiana | 77 / 1423 | 74.1%, 0.22 [0.11–0.34] ✗ | 13 / 77 (17%) | **52.0%, 0.19 [0.08–0.33] ✓** | 8 / 77 (10%) |
| pneumothorax | NIH | 18 / 1482 | 16.5%, 0.33 [0.00–1.15] ✗ | 1 / 18 (6%) | **16.5%, 0.33 [0.00–1.15] ✗** | 1 / 18 (6%) |
| pneumothorax | Indiana | 18 / 1482 | 53.6%, 0.10 [0.00–0.34] ✓ | 1 / 18 (6%) | **49.8%, 0.11 [0.00–0.37] ✓** | 1 / 18 (6%) |
| consolidation | NIH | 27 / 1473 | 4.5%, 0.83 [0.00–2.88] ✗ | 1 / 27 (4%) | **0%, –** | 0 / 27 |
| consolidation | Indiana | 27 / 1473 | 21.7%, 0.00 [0.00–0.00] ✓✓ | 0 / 27 (0%) | **0%, –** | 0 / 27 |
| pulmonary edema | NIH | 10 / 1490 | 19.1%, 0.00 [0.00–0.00] ✓✓ | 0 / 10 (0%) | **19.1%, 0.00 [0.00–0.00] ✓✓** | 0 / 10 (0%) |
| pulmonary edema | Indiana | 10 / 1490 | 36.9%, 0.00 [0.00–0.00] ✓✓ | 0 / 10 (0%) | **36.9%, 0.00 [0.00–0.00] ✓✓** | 0 / 10 (0%) |
| atelectasis | NIH | 93 / 1407 | 42.3%, 0.12 [0.03–0.23] ✓ | 5 / 93 (5%) | **27.3%, 0.04 [0.00–0.12] ✓✓** | 1 / 93 (1%) |
| atelectasis | Indiana | 93 / 1407 | 34.9%, 0.15 [0.03–0.28] ✓ | 5 / 93 (5%) | **20.1%, 0.05 [0.00–0.17] ✓✓** | 1 / 93 (1%) |

## Secondary: NIH test, random + supplement

Rule: primary 5 / 0.2. REFUTING state: coverage (share of all cases), empirical LR [95% CI] (✓ point ≤ 0.2; ✓✓ upper CI ≤ 0.2), and positives classified REFUTING (false reassurance).

| Finding | Platt source | Pos / neg | D2 Platt-only REFUTING | positives in it | **D3 dual-evidence REFUTING** | positives in it |
|---|---|---|---|---|---|---|
| cardiomegaly | NIH | 85 / 1595 | 54.5%, 0.02 [0.00–0.07] ✓✓ | 1 / 85 (1%) | **54.5%, 0.02 [0.00–0.07] ✓✓** | 1 / 85 (1%) |
| cardiomegaly | Indiana | 85 / 1595 | 66.8%, 0.03 [0.00–0.09] ✓✓ | 2 / 85 (2%) | **56.5%, 0.02 [0.00–0.06] ✓✓** | 1 / 85 (1%) |
| pleural effusion | NIH | 122 / 1558 | 53.5%, 0.17 [0.08–0.27] ✓ | 12 / 122 (10%) | **48.0%, 0.16 [0.07–0.26] ✓** | 10 / 122 (8%) |
| pleural effusion | Indiana | 122 / 1558 | 69.3%, 0.21 [0.13–0.30] ✗ | 19 / 122 (16%) | **47.6%, 0.16 [0.07–0.26] ✓** | 10 / 122 (8%) |
| pneumothorax | NIH | 68 / 1612 | 15.1%, 0.19 [0.00–0.48] ✓ | 2 / 68 (3%) | **15.1%, 0.19 [0.00–0.48] ✓** | 2 / 68 (3%) |
| pneumothorax | Indiana | 68 / 1612 | 49.7%, 0.11 [0.02–0.23] ✓ | 4 / 68 (6%) | **46.0%, 0.09 [0.00–0.21] ✓** | 3 / 68 (4%) |
| consolidation | NIH | 77 / 1603 | 4.0%, 0.31 [0.00–1.07] ✗ | 1 / 77 (1%) | **0%, –** | 0 / 77 |
| consolidation | Indiana | 77 / 1603 | 19.5%, 0.06 [0.00–0.21] ✓ | 1 / 77 (1%) | **0%, –** | 0 / 77 |
| pulmonary edema | NIH | 59 / 1621 | 17.1%, 0.00 [0.00–0.00] ✓✓ | 0 / 59 (0%) | **17.1%, 0.00 [0.00–0.00] ✓✓** | 0 / 59 (0%) |
| pulmonary edema | Indiana | 59 / 1621 | 33.3%, 0.00 [0.00–0.00] ✓✓ | 0 / 59 (0%) | **33.3%, 0.00 [0.00–0.00] ✓✓** | 0 / 59 (0%) |
| atelectasis | NIH | 120 / 1560 | 38.8%, 0.10 [0.02–0.20] ✓✓ | 5 / 120 (4%) | **24.8%, 0.03 [0.00–0.11] ✓✓** | 1 / 120 (1%) |
| atelectasis | Indiana | 120 / 1560 | 32.1%, 0.12 [0.03–0.24] ✓ | 5 / 120 (4%) | **18.3%, 0.04 [0.00–0.15] ✓✓** | 1 / 120 (1%) |

## Internal reasons (NIH test, random part; NIH Platt source; primary rule)

| Finding | SUPPORTING | REFUTING | WEAK_EVIDENCE | REFUTING_NOT_EMPIRICALLY_CONFIRMED | OUTSIDE_SUPPORT | FEW_DEV_POSITIVES |
|---|---|---|---|---|---|---|
| cardiomegaly | 4.5% | 58.5% | 37.0% | 0.0% | 0.0% | 0.0% |
| pleural effusion | 7.5% | 52.5% | 34.4% | 5.5% | 0.0% | 0.0% |
| pneumothorax | 2.7% | 16.5% | 80.3% | 0.0% | 0.5% | 0.0% |
| consolidation | 0.9% | 0.0% | 94.3% | 4.5% | 0.3% | 0.0% |
| pulmonary edema | 2.7% | 19.1% | 78.1% | 0.0% | 0.2% | 0.0% |
| atelectasis | 6.5% | 27.3% | 50.7% | 15.0% | 0.5% | 0.0% |

## Sensitivity analysis: NIH test, random + supplement

Rule: sensitivity 2 / 0.5. REFUTING state: coverage (share of all cases), empirical LR [95% CI] (✓ point ≤ 0.5; ✓✓ upper CI ≤ 0.5), and positives classified REFUTING (false reassurance).

| Finding | Platt source | Pos / neg | D2 Platt-only REFUTING | positives in it | **D3 dual-evidence REFUTING** | positives in it |
|---|---|---|---|---|---|---|
| cardiomegaly | NIH | 85 / 1595 | 66.7%, 0.03 [0.00–0.09] ✓✓ | 2 / 85 (2%) | **56.5%, 0.02 [0.00–0.06] ✓✓** | 1 / 85 (1%) |
| cardiomegaly | Indiana | 85 / 1595 | 76.0%, 0.12 [0.05–0.20] ✓✓ | 8 / 85 (9%) | **56.5%, 0.02 [0.00–0.06] ✓✓** | 1 / 85 (1%) |
| pleural effusion | NIH | 122 / 1558 | 70.2%, 0.21 [0.13–0.30] ✓✓ | 19 / 122 (16%) | **48.0%, 0.16 [0.07–0.27] ✓✓** | 10 / 122 (8%) |
| pleural effusion | Indiana | 122 / 1558 | 77.6%, 0.24 [0.16–0.33] ✓✓ | 24 / 122 (20%) | **47.6%, 0.16 [0.07–0.27] ✓✓** | 10 / 122 (8%) |
| pneumothorax | NIH | 68 / 1612 | 51.0%, 0.11 [0.02–0.23] ✓✓ | 4 / 68 (6%) | **50.0%, 0.08 [0.00–0.19] ✓✓** | 3 / 68 (4%) |
| pneumothorax | Indiana | 68 / 1612 | 70.4%, 0.26 [0.14–0.40] ✓✓ | 13 / 68 (19%) | **46.0%, 0.09 [0.00–0.21] ✓✓** | 3 / 68 (4%) |
| consolidation | NIH | 77 / 1603 | 37.5%, 0.10 [0.00–0.23] ✓✓ | 3 / 77 (4%) | **37.5%, 0.10 [0.00–0.23] ✓✓** | 3 / 77 (4%) |
| consolidation | Indiana | 77 / 1603 | 52.4%, 0.09 [0.02–0.20] ✓✓ | 4 / 77 (5%) | **49.6%, 0.05 [0.00–0.13] ✓✓** | 2 / 77 (3%) |
| pulmonary edema | NIH | 59 / 1621 | 51.6%, 0.03 [0.00–0.11] ✓✓ | 1 / 59 (2%) | **51.6%, 0.03 [0.00–0.11] ✓✓** | 1 / 59 (2%) |
| pulmonary edema | Indiana | 59 / 1621 | 68.7%, 0.07 [0.00–0.16] ✓✓ | 3 / 59 (5%) | **68.7%, 0.07 [0.00–0.16] ✓✓** | 3 / 59 (5%) |
| atelectasis | NIH | 120 / 1560 | 54.0%, 0.10 [0.03–0.18] ✓✓ | 7 / 120 (6%) | **46.2%, 0.10 [0.03–0.18] ✓✓** | 6 / 120 (5%) |
| atelectasis | Indiana | 120 / 1560 | 50.7%, 0.11 [0.03–0.19] ✓✓ | 7 / 120 (6%) | **39.7%, 0.12 [0.04–0.21] ✓✓** | 6 / 120 (5%) |

## Sensitivity analysis: NIH test, random + supplement

Rule: sensitivity 10 / 0.1. REFUTING state: coverage (share of all cases), empirical LR [95% CI] (✓ point ≤ 0.1; ✓✓ upper CI ≤ 0.1), and positives classified REFUTING (false reassurance).

| Finding | Platt source | Pos / neg | D2 Platt-only REFUTING | positives in it | **D3 dual-evidence REFUTING** | positives in it |
|---|---|---|---|---|---|---|
| cardiomegaly | NIH | 85 / 1595 | 43.0%, 0.03 [0.00–0.09] ✓✓ | 1 / 85 (1%) | **43.0%, 0.03 [0.00–0.09] ✓✓** | 1 / 85 (1%) |
| cardiomegaly | Indiana | 85 / 1595 | 59.0%, 0.02 [0.00–0.07] ✓✓ | 1 / 85 (1%) | **56.5%, 0.02 [0.00–0.07] ✓✓** | 1 / 85 (1%) |
| pleural effusion | NIH | 122 / 1558 | 35.1%, 0.09 [0.02–0.18] ✓ | 4 / 122 (3%) | **0%, –** | 0 / 122 |
| pleural effusion | Indiana | 122 / 1558 | 60.5%, 0.19 [0.10–0.29] ✗ | 15 / 122 (12%) | **0%, –** | 0 / 122 |
| pneumothorax | NIH | 68 / 1612 | 1.2%, 0.00 [0.00–0.00] ✓✓ | 0 / 68 (0%) | **0%, –** | 0 / 68 |
| pneumothorax | Indiana | 68 / 1612 | 33.8%, 0.13 [0.00–0.29] ✗ | 3 / 68 (4%) | **0%, –** | 0 / 68 |
| consolidation | NIH | 77 / 1603 | 4.0%, 0.31 [0.00–1.06] ✗ | 1 / 77 (1%) | **0%, –** | 0 / 77 |
| consolidation | Indiana | 77 / 1603 | 0%, – | 0 / 77 | **0%, –** | 0 / 77 |
| pulmonary edema | NIH | 59 / 1621 | 17.1%, 0.00 [0.00–0.00] ✓✓ | 0 / 59 (0%) | **0%, –** | 0 / 59 |
| pulmonary edema | Indiana | 59 / 1621 | 17.1%, 0.00 [0.00–0.00] ✓✓ | 0 / 59 (0%) | **0%, –** | 0 / 59 |
| atelectasis | NIH | 120 / 1560 | 27.5%, 0.06 [0.00–0.14] ✓ | 2 / 120 (2%) | **0%, –** | 0 / 120 |
| atelectasis | Indiana | 120 / 1560 | 17.8%, 0.04 [0.00–0.14] ✓ | 1 / 120 (1%) | **0%, –** | 0 / 120 |
