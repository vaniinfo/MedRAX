# EXPLORATORY: E2 multiplicity from the replayed pre-registered bootstrap draws

Run on 2026-10-25 after all pre-registered results were known. Method: docstring of `review_multiplicity_exact.py`.

| dataset | condition | positives | difference [replayed 95% CI] | committed | matches | resamples on far side of 0 | bootstrap p | Holm, all 29 | Holm, >= 10 positives |
|---|---|---|---|---|---|---|---|---|---|
| NIH test | cardiomegaly | 35 | +0.229 [+0.146, +0.320] | +0.229 [+0.146, +0.320] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| NIH test | pleural effusion | 77 | +0.162 [+0.098, +0.226] | +0.162 [+0.098, +0.226] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| NIH test | pneumothorax | 18 | +0.362 [+0.163, +0.540] | +0.362 [+0.163, +0.540] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| NIH test | consolidation | 27 | +0.101 [-0.004, +0.211] | +0.101 [-0.004, +0.211] | yes | 61 | 0.0620 | 0.1859 | 0.1239 |
| NIH test | pulmonary edema | 10 | +0.206 [+0.092, +0.345] | +0.206 [+0.092, +0.345] | yes | 1 | 0.0020 | 0.0290 | 0.0280 |
| NIH test | atelectasis | 93 | +0.115 [+0.079, +0.149] | +0.115 [+0.079, +0.149] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| Indiana | cardiomegaly | 164 | +0.198 [+0.154, +0.245] | +0.198 [+0.154, +0.245] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| Indiana | pleural effusion | 120 | +0.103 [+0.062, +0.153] | +0.103 [+0.062, +0.153] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| Indiana | pneumothorax | 28 | +0.349 [+0.215, +0.483] | +0.349 [+0.215, +0.483] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| Indiana | consolidation | 109 | +0.115 [+0.066, +0.169] | +0.115 [+0.066, +0.169] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| Indiana | pulmonary edema | 45 | +0.152 [+0.076, +0.230] | +0.152 [+0.076, +0.230] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| Indiana | atelectasis | 161 | +0.091 [+0.050, +0.129] | +0.091 [+0.050, +0.129] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| VinDr | cardiomegaly | 358 | +0.236 [+0.208, +0.264] | +0.236 [+0.208, +0.264] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| VinDr | pleural effusion | 120 | +0.182 [+0.132, +0.231] | +0.182 [+0.132, +0.231] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| VinDr | pneumothorax | 15 | +0.494 [+0.335, +0.648] | +0.494 [+0.335, +0.648] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| VinDr | consolidation | 25 | +0.037 [+0.016, +0.064] | +0.037 [+0.016, +0.064] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| VinDr | atelectasis | 10 | +0.203 [+0.057, +0.379] | +0.203 [+0.057, +0.379] | yes | 1 | 0.0020 | 0.0290 | 0.0280 |
| NIH test | pneumonia | 9 | +0.113 [-0.019, +0.278] | +0.113 [-0.019, +0.278] | yes | 119 | 0.1199 | 0.2399 | not counted |
| NIH test | infiltration | 185 | +0.013 [-0.016, +0.044] | +0.013 [-0.016, +0.044] | yes | 358 | 0.3588 | 0.3588 | 0.3588 |
| NIH test | emphysema | 14 | +0.341 [+0.170, +0.519] | +0.341 [+0.170, +0.519] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| NIH test | nodule | 85 | +0.114 [+0.057, +0.174] | +0.114 [+0.057, +0.174] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| NIH test | mass | 67 | +0.082 [+0.033, +0.141] | +0.082 [+0.033, +0.141] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| NIH test | fibrosis | 24 | +0.101 [+0.041, +0.166] | +0.101 [+0.041, +0.166] | yes | 1 | 0.0020 | 0.0290 | 0.0280 |
| NIH test | pleural thickening | 30 | +0.217 [+0.145, +0.294] | +0.217 [+0.145, +0.294] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| VinDr | infiltration | 44 | +0.055 [+0.033, +0.081] | +0.055 [+0.033, +0.081] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| VinDr | nodule | 90 | +0.162 [+0.120, +0.210] | +0.162 [+0.120, +0.210] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| VinDr | mass | 90 | +0.082 [+0.046, +0.123] | +0.082 [+0.046, +0.123] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| VinDr | fibrosis | 182 | +0.116 [+0.089, +0.145] | +0.116 [+0.089, +0.145] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |
| VinDr | pleural thickening | 178 | +0.277 [+0.229, +0.327] | +0.277 [+0.229, +0.327] | yes | 0 | 0.0010 | 0.0290 | 0.0280 |

Reproduced CIs matching the committed table: 29 of 29. statsmodels not installed; Holm checked only by this script's implementation.
Significant at 0.05 after Holm: 26 of 29 (all); 26 of 28 (at least 10 positives). The smallest attainable p is 0.0010, so with 29 tests Holm's first threshold is 0.05 / 29 = 0.0017.

