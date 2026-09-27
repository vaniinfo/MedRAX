# Evidence Engine D1/D2 results

Protocol: `experiments/evidence_engine_design.md` (committed before this run). Inputs SHA-256 verified against `results/REPRODUCE.md`. NIH test: random part n = 1500 (natural prevalence), supplement n = 180. LR CIs from 1000 Platt refits on the source development set; reported CIs from 2000 image-level bootstrap resamples of the test set. Seed 20260927.

## D1: how much of the Indiana → NIH calibration shift is prevalence shift?

NIH test, random part. B = Indiana Platt; B′ = Indiana LR × NIH development prevalence (aggregate count only); C = NIH Platt. Brier (lower is better) with 95% CI. Share explained = (B − B′) / (B − C).

| Finding | NIH dev prevalence | Brier raw | Brier B | Brier B′ | Brier C | B − B′ (prevalence) | B′ − C (residual) | Share explained |
|---|---|---|---|---|---|---|---|---|
| cardiomegaly | 0.023 | 0.069 [0.060–0.078] | 0.043 [0.037–0.050] | 0.020 [0.014–0.026] | 0.019 [0.014–0.025] | 0.024 [0.017–0.031] | 0.000 [0.000–0.001] | 98% [96%–100%] |
| pleural effusion | 0.051 | 0.067 [0.058–0.077] | 0.048 [0.039–0.056] | 0.035 [0.028–0.043] | 0.035 [0.028–0.042] | 0.012 [0.007–0.018] | 0.000 [-0.000–0.001] | 97% [89%–102%] |
| pneumothorax | 0.012 | 0.018 [0.014–0.023] | 0.016 [0.011–0.021] | 0.010 [0.006–0.014] | 0.009 [0.005–0.013] | 0.006 [0.004–0.008] | 0.001 [-0.001–0.003] | 89% [70%–139%] |
| consolidation | 0.018 | 0.048 [0.041–0.055] | 0.034 [0.030–0.039] | 0.017 [0.011–0.023] | 0.017 [0.011–0.023] | 0.018 [0.013–0.022] | 0.000 [-0.000–0.001] | 99% [96%–100%] |
| pulmonary edema | 0.007 | 0.020 [0.015–0.025] | 0.007 [0.005–0.010] | 0.006 [0.002–0.010] | 0.005 [0.002–0.009] | 0.001 [-0.001–0.003] | 0.000 [-0.000–0.001] | 74% [-250%–101%] |
| atelectasis | 0.062 | 0.110 [0.100–0.120] | 0.064 [0.058–0.071] | 0.047 [0.038–0.055] | 0.044 [0.037–0.052] | 0.017 [0.010–0.025] | 0.003 [0.000–0.005] | 87% [68%–100%] |

Calibration diagnostics (NIH test, random part): calibration-in-the-large (mean predicted − observed; 0 is ideal) / calibration slope (1 is ideal).

| Finding | Observed prevalence | B | B′ | C |
|---|---|---|---|---|
| cardiomegaly | 0.023 | +0.075 / 0.91 | -0.010 / 0.91 | -0.004 / 1.15 |
| pleural effusion | 0.051 | +0.040 / 0.61 | -0.008 / 0.61 | +0.000 / 0.81 |
| pneumothorax | 0.012 | +0.025 / 0.62 | +0.005 / 0.62 | +0.002 / 1.07 |
| consolidation | 0.018 | +0.092 / 1.11 | -0.007 / 1.11 | -0.000 / 1.13 |
| pulmonary edema | 0.007 | +0.019 / 1.19 | -0.004 / 1.19 | -0.000 / 1.26 |
| atelectasis | 0.062 | +0.103 / 1.24 | -0.019 / 1.24 | +0.005 / 1.04 |

## D2: does the LR keep its meaning across datasets?

### State boundaries (raw p_present) under the primary rule

SUPPORTING if p ≥ p_S (lower 95% CI of LR ≥ 5); REFUTING if p ≤ p_R (upper 95% CI of LR ≤ 0.2); outside support → INSUFFICIENT DATA.

| Finding | Source | Dev positives | p_S | p_R | Support range (p) |
|---|---|---|---|---|---|
| cardiomegaly | Indiana | 115 | 0.892 | 0.0654 | 0.0000–0.987 |
| cardiomegaly | NIH | 80 | 0.801 | 0.0145 | 0.0000–0.991 |
| pleural effusion | Indiana | 84 | 0.826 | 0.0911 | 0.0010–0.981 |
| pleural effusion | NIH | 180 | 0.743 | 0.0277 | 0.0000–0.991 |
| pneumothorax | Indiana | 20 | 0.448 | 0.0230 | 0.0030–0.846 |
| pneumothorax | NIH | 43 | 0.428 | 0.0052 | 0.0010–0.941 |
| consolidation | Indiana | 76 | 0.923 | 0.0022 | 0.0010–0.845 |
| consolidation | NIH | 62 | 0.802 | 0.0006 | 0.0000–0.906 |
| pulmonary edema | Indiana | 31 | 0.899 | 0.0016 | 0.0000–0.887 |
| pulmonary edema | NIH | 24 | 0.490 | 0.0006 | 0.0000–0.938 |
| atelectasis | Indiana | 113 | 0.926 | 0.0552 | 0.0040–0.938 |
| atelectasis | NIH | 218 | 0.791 | 0.0573 | 0.0010–0.965 |

### Primary: NIH test, random part (natural prevalence)

Empirical LR = P(state | finding) / P(state | no finding), 95% bootstrap CI; coverage = share of all cases in the state. ✓ = point estimate meets the criterion (SUPPORTING ≥ 5, REFUTING ≤ 0.2); ✓✓ = the CI also meets it.

| Finding | Source | Pos / neg | SUPPORTING: coverage, LR | UNCERTAIN: coverage, LR | REFUTING: coverage, LR | INSUFFICIENT |
|---|---|---|---|---|---|---|
| cardiomegaly | Indiana | 35 / 1465 | 2%, 25.6 [12.6–51.0] ✓✓ | 27%, 2.49 [1.83–3.16] | 71%, 0.04 [0.00–0.13] ✓✓ | 0% |
| cardiomegaly | NIH | 35 / 1465 | 5%, 15.1 [9.62–22.6] ✓✓ | 37%, 1.24 [0.78–1.72] | 58%, 0.05 [0.00–0.16] ✓✓ | 0% |
| pleural effusion | Indiana | 77 / 1423 | 5%, 15.5 [10.8–23.2] ✓✓ | 20%, 1.57 [1.03–2.15] | 74%, 0.22 [0.12–0.33] ✗ | 1% |
| pleural effusion | NIH | 77 / 1423 | 8%, 14.7 [11.1–19.8] ✓✓ | 34%, 0.67 [0.39–0.96] | 58%, 0.19 [0.09–0.32] ✓ | 0% |
| pneumothorax | Indiana | 18 / 1482 | 2%, 22.2 [8.22–41.3] ✓✓ | 39%, 0.71 [0.22–1.32] | 54%, 0.10 [0.00–0.37] ✓ | 5% |
| pneumothorax | NIH | 18 / 1482 | 3%, 20.6 [9.02–36.3] ✓✓ | 80%, 0.34 [0.11–0.63] | 17%, 0.33 [0.00–1.18] ✗ | 0% |
| consolidation | Indiana | 27 / 1473 | 0%, – | 73%, 1.22 [1.04–1.38] | 22%, 0.00 [0.00–0.00] ✓✓ | 5% |
| consolidation | NIH | 27 / 1473 | 1%, 16.4 [0.00–49.0] ✓ | 94%, 0.86 [0.69–1.01] | 4%, 0.83 [0.00–2.79] ✗ | 0% |
| pulmonary edema | Indiana | 10 / 1490 | 0%, – | 63%, 1.11 [0.60–1.59] | 37%, 0.00 [0.00–0.00] ✓✓ | 0% |
| pulmonary edema | NIH | 10 / 1490 | 3%, 21.3 [7.27–41.4] ✓✓ | 78%, 0.38 [0.00–0.80] | 19%, 0.00 [0.00–0.00] ✓✓ | 0% |
| atelectasis | Indiana | 93 / 1407 | 0%, 7.56 [0.00–60.5] ✓ | 57%, 1.51 [1.34–1.66] | 35%, 0.15 [0.03–0.29] ✓ | 8% |
| atelectasis | NIH | 93 / 1407 | 6%, 11.1 [7.77–15.8] ✓✓ | 51%, 0.91 [0.69–1.11] | 42%, 0.12 [0.03–0.24] ✓ | 0% |

### Secondary: NIH test, random + supplement

Empirical LR = P(state | finding) / P(state | no finding), 95% bootstrap CI; coverage = share of all cases in the state. ✓ = point estimate meets the criterion (SUPPORTING ≥ 5, REFUTING ≤ 0.2); ✓✓ = the CI also meets it.

| Finding | Source | Pos / neg | SUPPORTING: coverage, LR | UNCERTAIN: coverage, LR | REFUTING: coverage, LR | INSUFFICIENT |
|---|---|---|---|---|---|---|
| cardiomegaly | Indiana | 85 / 1595 | 3%, 23.5 [13.7–41.4] ✓✓ | 30%, 2.36 [1.97–2.77] | 67%, 0.03 [0.00–0.09] ✓✓ | 0% |
| cardiomegaly | NIH | 85 / 1595 | 6%, 12.2 [8.58–17.0] ✓✓ | 40%, 1.36 [1.08–1.67] | 54%, 0.02 [0.00–0.07] ✓✓ | 0% |
| pleural effusion | Indiana | 122 / 1558 | 7%, 12.1 [8.96–16.7] ✓✓ | 23%, 1.57 [1.19–1.99] | 69%, 0.21 [0.13–0.30] ✗ | 1% |
| pleural effusion | NIH | 122 / 1558 | 9%, 11.0 [8.54–14.3] ✓✓ | 37%, 0.80 [0.59–1.03] | 53%, 0.17 [0.08–0.27] ✓ | 0% |
| pneumothorax | Indiana | 68 / 1612 | 3%, 17.4 [10.4–29.6] ✓✓ | 42%, 1.05 [0.78–1.36] | 50%, 0.11 [0.02–0.23] ✓ | 5% |
| pneumothorax | NIH | 68 / 1612 | 4%, 15.8 [9.88–25.3] ✓✓ | 80%, 0.57 [0.44–0.72] | 15%, 0.19 [0.00–0.49] ✓ | 1% |
| consolidation | Indiana | 77 / 1603 | 0%, – | 75%, 1.18 [1.08–1.28] | 19%, 0.06 [0.00–0.21] ✓ | 5% |
| consolidation | NIH | 77 / 1603 | 2%, 18.2 [8.65–37.1] ✓✓ | 94%, 0.83 [0.72–0.93] | 4%, 0.31 [0.00–1.13] ✗ | 1% |
| pulmonary edema | Indiana | 59 / 1621 | 0%, – | 66%, 1.35 [1.20–1.48] | 33%, 0.00 [0.00–0.00] ✓✓ | 1% |
| pulmonary edema | NIH | 59 / 1621 | 5%, 16.3 [11.2–22.8] ✓✓ | 77%, 0.47 [0.32–0.63] | 17%, 0.00 [0.00–0.00] ✓✓ | 0% |
| atelectasis | Indiana | 120 / 1560 | 0%, 5.20 [0.00–29.4] ✓ | 60%, 1.48 [1.35–1.59] | 32%, 0.12 [0.03–0.24] ✓ | 7% |
| atelectasis | NIH | 120 / 1560 | 7%, 8.26 [5.90–11.5] ✓✓ | 54%, 0.98 [0.81–1.15] | 39%, 0.10 [0.02–0.20] ✓✓ | 0% |

### Pre-registered sensitivity analyses (NIH test, random + supplement)

Coverage and empirical LR of SUPPORTING / REFUTING under each rule.

| Finding | Source | Rule | UNCERTAIN coverage | SUPPORTING: coverage, LR | REFUTING: coverage, LR |
|---|---|---|---|---|---|
| cardiomegaly | Indiana | primary 5 / 0.2 | 30% | 3%, 23.5 | 67%, 0.03 |
| cardiomegaly | Indiana | sensitivity 2 / 0.5 | 15% | 9%, 9.87 | 76%, 0.12 |
| cardiomegaly | Indiana | sensitivity 10 / 0.1 | 40% | 1%, 26.8 | 59%, 0.02 |
| cardiomegaly | NIH | primary 5 / 0.2 | 40% | 6%, 12.2 | 54%, 0.02 |
| cardiomegaly | NIH | sensitivity 2 / 0.5 | 20% | 13%, 7.62 | 67%, 0.03 |
| cardiomegaly | NIH | sensitivity 10 / 0.1 | 55% | 2%, 25.0 | 43%, 0.03 |
| pleural effusion | Indiana | primary 5 / 0.2 | 23% | 7%, 12.1 | 69%, 0.21 |
| pleural effusion | Indiana | sensitivity 2 / 0.5 | 11% | 11%, 8.83 | 78%, 0.24 |
| pleural effusion | Indiana | sensitivity 10 / 0.1 | 35% | 4%, 14.8 | 61%, 0.19 |
| pleural effusion | NIH | primary 5 / 0.2 | 37% | 9%, 11.0 | 53%, 0.17 |
| pleural effusion | NIH | sensitivity 2 / 0.5 | 16% | 14%, 7.42 | 70%, 0.21 |
| pleural effusion | NIH | sensitivity 10 / 0.1 | 59% | 6%, 12.8 | 35%, 0.09 |
| pneumothorax | Indiana | primary 5 / 0.2 | 42% | 3%, 17.4 | 50%, 0.11 |
| pneumothorax | Indiana | sensitivity 2 / 0.5 | 19% | 5%, 13.2 | 70%, 0.26 |
| pneumothorax | Indiana | sensitivity 10 / 0.1 | 59% | 2%, 30.0 | 34%, 0.13 |
| pneumothorax | NIH | primary 5 / 0.2 | 80% | 4%, 15.8 | 15%, 0.19 |
| pneumothorax | NIH | sensitivity 2 / 0.5 | 40% | 8%, 9.04 | 51%, 0.11 |
| pneumothorax | NIH | sensitivity 10 / 0.1 | 96% | 2%, 44.8 | 1%, 0.00 |
| consolidation | Indiana | primary 5 / 0.2 | 75% | 0%, – | 19%, 0.06 |
| consolidation | Indiana | sensitivity 2 / 0.5 | 34% | 8%, 6.67 | 52%, 0.09 |
| consolidation | Indiana | sensitivity 10 / 0.1 | 95% | 0%, – | 0%, – |
| consolidation | NIH | primary 5 / 0.2 | 94% | 2%, 18.2 | 4%, 0.31 |
| consolidation | NIH | sensitivity 2 / 0.5 | 44% | 18%, 4.99 | 38%, 0.10 |
| consolidation | NIH | sensitivity 10 / 0.1 | 95% | 0%, – | 4%, 0.31 |
| pulmonary edema | Indiana | primary 5 / 0.2 | 66% | 0%, – | 33%, 0.00 |
| pulmonary edema | Indiana | sensitivity 2 / 0.5 | 26% | 5%, 15.0 | 69%, 0.07 |
| pulmonary edema | Indiana | sensitivity 10 / 0.1 | 82% | 0%, – | 17%, 0.00 |
| pulmonary edema | NIH | primary 5 / 0.2 | 77% | 5%, 16.3 | 17%, 0.00 |
| pulmonary edema | NIH | sensitivity 2 / 0.5 | 38% | 10%, 9.08 | 52%, 0.03 |
| pulmonary edema | NIH | sensitivity 10 / 0.1 | 81% | 1%, 36.6 | 17%, 0.00 |
| atelectasis | Indiana | primary 5 / 0.2 | 60% | 0%, 5.20 | 32%, 0.12 |
| atelectasis | Indiana | sensitivity 2 / 0.5 | 33% | 9%, 7.15 | 51%, 0.11 |
| atelectasis | Indiana | sensitivity 10 / 0.1 | 75% | 0%, – | 18%, 0.04 |
| atelectasis | NIH | primary 5 / 0.2 | 54% | 7%, 8.26 | 39%, 0.10 |
| atelectasis | NIH | sensitivity 2 / 0.5 | 26% | 20%, 4.73 | 54%, 0.10 |
| atelectasis | NIH | sensitivity 10 / 0.1 | 70% | 2%, 13.0 | 28%, 0.06 |

### Platt LR vs empirical LR by score band (NIH test, random + supplement)

Empirical LR = P(band | finding) / P(band | no finding) with 95% CI; Platt LR = mean Platt-derived LR of the cases in the band (Indiana / NIH source). ✓ = Platt LR within the empirical CI.

| Finding | 0–0.02 | 0.02–0.05 | 0.05–0.1 | 0.1–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–0.9 | 0.9–1 |
|---|---|---|---|---|---|---|---|---|---|
| cardiomegaly | 0.02 [0.00–0.07] · 0.01✓ / 0.04✓ (n=949) | 0.14 [0.00–0.50] · 0.06✓ / 0.21✓ (n=133) | 0.43 [0.00–1.12] · 0.12✓ / 0.37✓ (n=89) | 0.74 [0.17–1.56] · 0.25✓ / 0.65✓ (n=105) | 0.88 [0.19–1.77] · 0.53✓ / 1.18✓ (n=112) | 2.50 [1.19–4.02] · 1.13✗ / 2.14✓ (n=102) | 5.63 [3.45–8.55] · 2.56✗ / 4.11✓ (n=91) | 6.41 [3.31–11.0] · 5.36✓ / 7.41✓ (n=55) | 24.7 [14.2–46.1] · 17.1✓ / 18.1✓ (n=44) |
| pleural effusion | 0.16 [0.07–0.26] · 0.01✗ / 0.04✗ (n=807) | 0.26 [0.06–0.51] · 0.03✗ / 0.16✓ (n=247) | 0.38 [0.09–0.83] · 0.08✗ / 0.30✓ (n=138) | 0.46 [0.11–1.00] · 0.19✓ / 0.56✓ (n=115) | 0.91 [0.28–1.77] · 0.53✓ / 1.21✓ (n=90) | 2.55 [1.26–4.18] · 1.22✗ / 2.27✓ (n=78) | 3.31 [1.68–5.59] · 3.42✓ / 4.92✓ (n=68) | 9.06 [5.26–15.0] · 8.56✓ / 9.84✓ (n=53) | 14.0 [9.62–21.5] · 41.4✗ / 30.9✗ (n=84) |
| pneumothorax | 0.08 [0.00–0.19] · 0.03✓ / 0.17✓ (n=841) | 0.59 [0.27–0.96] · 0.15✗ / 0.50✓ (n=410) | 0.86 [0.29–1.56] · 0.46✓ / 0.97✓ (n=199) | 0.50 [0.00–1.37] · 1.37✓ / 1.81✗ (n=96) | 6.32 [2.97–11.1] · 4.81✓ / 3.75✓ (n=57) | 2.84 [0.00–7.40] · 17.2✗ / 7.85✗ (n=28) | 36.9 [17.2–94.9] · 65.5✓ / 16.8✗ (n=23) | 29.6 [6.02–171.2] · 156.8✓ / 28.2✓ (n=9) | 56.9 [21.7–272.0] · ≥1000✗ / 210.1✓ (n=17) |
| consolidation | 0.07 [0.00–0.16] · 0.14✓ / 0.24✗ (n=902) | 0.43 [0.10–0.89] · 0.41✓ / 0.67✓ (n=196) | 0.59 [0.00–1.44] · 0.62✓ / 1.01✓ (n=109) | 0.88 [0.19–1.78] · 0.93✓ / 1.51✓ (n=123) | 2.24 [1.18–3.57] · 1.48✓ / 2.37✓ (n=134) | 3.90 [2.14–6.12] · 2.34✓ / 3.72✓ (n=95) | 5.62 [3.22–8.92] · 3.57✓ / 5.62✓ (n=80) | 18.4 [8.69–35.6] · 5.53✗ / 8.62✗ (n=32) | 5.95 [0.00–24.1] · 9.51✓ / 14.7✓ (n=9) |
| pulmonary edema | 0.07 [0.00–0.15] · 0.04✓ / 0.12✓ (n=1177) | 0.38 [0.00–1.01] · 0.24✓ / 0.64✓ (n=148) | 1.34 [0.28–2.79] · 0.44✓ / 1.16✓ (n=86) | 1.64 [0.35–3.61] · 0.80✓ / 2.03✓ (n=71) | 1.78 [0.37–3.52] · 1.47✓ / 3.63✗ (n=82) | 7.63 [3.38–14.2] · 2.82✗ / 6.70✓ (n=46) | 12.8 [5.99–22.7] · 5.44✗ / 12.4✓ (n=41) | 39.7 [18.3–105.6] · 10.6✗ / 23.3✓ (n=22) | 68.7 [12.4–≥1000] · 38.0✓ / 77.5✓ (n=7) |
| atelectasis | 0.03 [0.00–0.11] · 0.03✓ / 0.02✓ (n=416) | 0.19 [0.00–0.43] · 0.08✓ / 0.08✓ (n=207) | 0.17 [0.00–0.46] · 0.16✓ / 0.17✓ (n=153) | 0.08 [0.00–0.26] · 0.27✗ / 0.34✗ (n=173) | 0.52 [0.23–0.88] · 0.54✓ / 0.77✓ (n=232) | 1.18 [0.65–1.79] · 1.06✓ / 1.71✓ (n=192) | 3.38 [2.45–4.49] · 2.05✗ / 3.77✓ (n=189) | 7.16 [4.57–11.1] · 3.92✗ / 8.15✓ (n=76) | 14.3 [8.02–26.2] · 9.18✓ / 23.0✓ (n=42) |
