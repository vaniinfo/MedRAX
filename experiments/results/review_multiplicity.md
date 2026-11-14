# EXPLORATORY: multiplicity analyses requested in review

Run on 2026-10-25 after all pre-registered results were known. Not pre-registered. Methods: see the docstring of `review_multiplicity.py`.

## 1. E2: approximate Holm adjustment (p-values reconstructed from CIs)

| dataset | condition | difference [95% CI] | approx. p | Holm, all 29 | Holm, comparisons with >= 10 positives |
|---|---|---|---|---|---|
| NIH | Cardiomegaly | +0.229 [+0.146, +0.320] | 6.4e-08 | 1.3e-06 | 1.2e-06 |
| NIH | Pleural effusion | +0.162 [+0.098, +0.226] | 7e-07 | 1.3e-05 | 1.2e-05 |
| NIH | Pneumothorax | +0.362 [+0.163, +0.540] | 0.00036 | 0.0033 | 0.0029 |
| NIH | Consolidation | +0.101 [-0.004, +0.211] | 0.059 | 0.18 | 0.12 |
| NIH | Pulmonary edema | +0.206 [+0.092, +0.345] | 0.0004 | 0.0033 | 0.0029 |
| NIH | Atelectasis | +0.115 [+0.079, +0.149] | 3.8e-10 | 8.8e-09 | 8.4e-09 |
| NIH | Pneumonia | +0.113 [-0.019, +0.278] | 0.093 | 0.19 | not counted |
| NIH | Infiltration | +0.013 [-0.016, +0.044] | 0.38 | 0.38 | 0.38 |
| NIH | Emphysema | +0.341 [+0.170, +0.519] | 9.3e-05 | 0.0011 | 0.00097 |
| NIH | Nodule | +0.114 [+0.057, +0.174] | 8.9e-05 | 0.0011 | 0.00097 |
| NIH | Mass | +0.082 [+0.033, +0.141] | 0.001 | 0.0058 | 0.0048 |
| NIH | Fibrosis | +0.101 [+0.041, +0.166] | 0.00097 | 0.0058 | 0.0048 |
| NIH | Pleural thickening | +0.217 [+0.145, +0.294] | 3.5e-09 | 7.3e-08 | 7e-08 |
| Indiana | Cardiomegaly | +0.198 [+0.154, +0.245] | 0 | 0 | 0 |
| Indiana | Pleural effusion | +0.103 [+0.062, +0.153] | 8.5e-07 | 1.4e-05 | 1.4e-05 |
| Indiana | Pneumothorax | +0.349 [+0.215, +0.483] | 3.3e-07 | 6.3e-06 | 6e-06 |
| Indiana | Consolidation | +0.115 [+0.066, +0.169] | 4.2e-06 | 6.3e-05 | 5.9e-05 |
| Indiana | Pulmonary edema | +0.152 [+0.076, +0.230] | 8.9e-05 | 0.0011 | 0.00097 |
| Indiana | Atelectasis | +0.091 [+0.050, +0.129] | 1.4e-05 | 0.00018 | 0.00016 |
| VinDr | Cardiomegaly | +0.236 [+0.208, +0.264] | 0 | 0 | 0 |
| VinDr | Pleural effusion | +0.182 [+0.132, +0.231] | 9.7e-13 | 2.3e-11 | 2.2e-11 |
| VinDr | Pneumothorax | +0.494 [+0.335, +0.648] | 1.1e-09 | 2.5e-08 | 2.4e-08 |
| VinDr | Consolidation | +0.037 [+0.016, +0.064] | 0.00055 | 0.0039 | 0.0033 |
| VinDr | Atelectasis | +0.203 [+0.057, +0.379] | 0.0064 | 0.026 | 0.019 |
| VinDr | Infiltration | +0.055 [+0.033, +0.081] | 9.6e-07 | 1.5e-05 | 1.4e-05 |
| VinDr | Nodule* | +0.162 [+0.120, +0.210] | 4e-14 | 1e-12 | 9.7e-13 |
| VinDr | Mass* | +0.082 [+0.046, +0.123] | 8e-06 | 0.00011 | 0.0001 |
| VinDr | Fibrosis | +0.116 [+0.089, +0.145] | 0 | 0 | 0 |
| VinDr | Pleural thickening | +0.277 [+0.229, +0.327] | 0 | 0 | 0 |

Significant at 0.05 after Holm: 26 of 29 (all comparisons); 26 of 28 (comparisons with at least 10 positives).

## 2. E4: bootstrap p-values and Holm across the six pre-registered fusion hypotheses

2000 new image-level resamples, seed 20261026.

| setting | comparison | difference [95% CI] | resamples <= 0 | one-sided p | Holm-adjusted p |
|---|---|---|---|---|---|
| (a) base, single image | fused CheXagent minus report alone | +0.306 [+0.284, +0.326] | 0 | 0.0005 | 0.0030 |
| (a) base, single image | fused CheXagent minus fused MAIRA-2 | +0.152 [+0.121, +0.184] | 0 | 0.0005 | 0.0030 |
| (b) base, prior + current | fused CheXagent minus report alone | +0.291 [+0.264, +0.316] | 0 | 0.0005 | 0.0030 |
| (b) base, prior + current | fused CheXagent minus fused MAIRA-2 | +0.162 [+0.122, +0.202] | 0 | 0.0005 | 0.0030 |
| (c) trained, prior + current | fused CheXagent minus report alone | +0.244 [+0.218, +0.268] | 0 | 0.0005 | 0.0030 |
| (c) trained, prior + current | fused CheXagent minus fused MAIRA-2 | +0.144 [+0.110, +0.175] | 0 | 0.0005 | 0.0030 |

The smallest attainable p with 2000 resamples is 0.0005.

