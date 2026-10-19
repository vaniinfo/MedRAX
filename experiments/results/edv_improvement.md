# Does EDV improve condition detection in MedGemma reports?

Protocol: `edv_improvement_protocol.md` (committed before the reports). Base MedGemma 1.5 reports on the 1500 NIH test images, read for 13 conditions. Paired 95% CIs from 2000 image-level bootstrap resamples. Mean AUC over the 12 conditions with at least 10 positives.

## Primary: mean AUC

| system | mean AUC [95% CI] |
|---|---|
| MedGemma alone | 0.557 [0.540-0.574] |
| MedGemma + CheXagent | 0.863 [0.844-0.881] |
| MedGemma + MAIRA-2 | 0.711 [0.678-0.742] |

| hypothesis | difference [95% CI] | result |
|---|---|---|
| H1: MedGemma + CheXagent minus MedGemma alone | +0.306 [+0.284 to +0.328] | supported |
| H2: MedGemma + CheXagent minus MedGemma + MAIRA-2 | +0.152 [+0.121 to +0.184] | supported |
| (also) MedGemma + MAIRA-2 minus MedGemma alone | +0.154 [+0.123 to +0.182] | supported |

## Per condition (AUC)

| condition | positives / n | MedGemma alone | + CheXagent | + MAIRA-2 | CheXagent minus alone [95% CI] | CheXagent minus MAIRA-2 [95% CI] |
|---|---|---|---|---|---|---|
| cardiomegaly | 35 / 1500 | 0.554 | 0.934 | 0.706 | +0.381 [+0.321 to +0.435] | +0.228 [+0.147 to +0.322] |
| pleural effusion | 77 / 1500 | 0.758 | 0.883 | 0.792 | +0.125 [+0.079 to +0.174] | +0.092 [+0.037 to +0.146] |
| pneumothorax | 18 / 1500 | 0.527 | 0.894 | 0.563 | +0.366 [+0.234 to +0.462] | +0.331 [+0.118 to +0.513] |
| consolidation | 27 / 1500 | 0.510 | 0.861 | 0.759 | +0.351 [+0.270 to +0.418] | +0.102 [-0.005 to +0.210] |
| pulmonary edema | 10 / 1500 | 0.743 | 0.915 | 0.806 | +0.172 [+0.033 to +0.320] | +0.108 [+0.016 to +0.252] |
| atelectasis | 93 / 1500 | 0.566 | 0.879 | 0.781 | +0.313 [+0.268 to +0.353] | +0.098 [+0.065 to +0.133] |
| pneumonia (fewer than 10 positives; not in the mean) | 9 / 1500 | 0.500 | 0.734 | 0.622 | +0.234 [+0.029 to +0.402] | +0.113 [-0.028 to +0.290] |
| infiltration | 185 / 1500 | 0.498 | 0.599 | 0.585 | +0.101 [+0.055 to +0.148] | +0.014 [-0.016 to +0.044] |
| emphysema | 14 / 1500 | 0.500 | 0.921 | 0.580 | +0.421 [+0.361 to +0.471] | +0.341 [+0.182 to +0.524] |
| nodule | 85 / 1500 | 0.514 | 0.801 | 0.686 | +0.287 [+0.232 to +0.335] | +0.115 [+0.061 to +0.175] |
| mass | 67 / 1500 | 0.514 | 0.934 | 0.852 | +0.420 [+0.386 to +0.449] | +0.082 [+0.034 to +0.135] |
| fibrosis | 24 / 1500 | 0.500 | 0.823 | 0.723 | +0.323 [+0.233 to +0.399] | +0.100 [+0.041 to +0.164] |
| pleural thickening | 30 / 1500 | 0.500 | 0.916 | 0.699 | +0.416 [+0.374 to +0.451] | +0.217 [+0.140 to +0.295] |

## Secondary: the [Image check] rule as yes/no decisions (all 13 conditions pooled)

| system | sensitivity | specificity | balanced accuracy |
|---|---|---|---|
| MedGemma alone | 0.113 | 0.989 | 0.551 |
| MedGemma + [Image check] rule | 0.145 | 0.986 | 0.566 |
| difference [95% CI] | +0.033 [+0.020 to +0.046] | -0.003 [-0.004 to -0.002] | +0.015 [+0.008 to +0.021] |

Limitations (recorded in the protocol): MedGemma's score has three values, which caps its AUC; NIH labels are noisy; CheXagent was probably trained on NIH images; base MedGemma, not the team's model.
