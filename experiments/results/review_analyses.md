# EXPLORATORY: analyses requested in internal review

Run on 2026-10-25 after all pre-registered results were known. Not pre-registered.

## 1. Patient audit (NIH)

| quantity | value |
|---|---|
| development images / unique patients | 3500 / 3500 |
| test images / unique patients | 1500 / 1500 |
| patients in both development and test | 0 |
| test images whose patient also appears in development | 0 |
| test patients with more than one test image | 0 (max 1 images) |
| longitudinal pairs (current in test) / unique patients | 499 / 499 |
| prior images that are development images / test images / neither | 0 / 0 / 499 |
| grounding cases / images / unique patients | 947 / 847 / 699 |
| grounding patients who also have a test-split image | 42 |

## 2-3. Checker-only baselines and patient-clustered intervals

### Experiment 1: base MedGemma, single image (1,500 test images)

1500 images, 1500 patients; mean over 12 conditions with at least 10 positives.

| system | mean AUC | 95% CI (image) | 95% CI (patient-clustered) |
|---|---|---|---|
| report alone | 0.557 | 0.539-0.573 | 0.541-0.576 |
| CheXagent alone | 0.866 | 0.845-0.884 | 0.846-0.883 |
| MAIRA-2 alone | 0.695 | 0.666-0.725 | 0.664-0.727 |
| report + CheXagent | 0.863 | 0.842-0.882 | 0.843-0.881 |
| report + MAIRA-2 | 0.711 | 0.680-0.742 | 0.678-0.743 |

| comparison | difference | 95% CI (image) | 95% CI (patient-clustered) |
|---|---|---|---|
| report + CheXagent minus report alone | +0.306 | +0.285 to +0.327 | +0.284 to +0.327 |
| report + CheXagent minus report + MAIRA-2 | +0.152 | +0.123 to +0.183 | +0.121 to +0.184 |
| report + CheXagent minus CheXagent alone | -0.002 | -0.003 to -0.002 | -0.003 to -0.002 |
| report + MAIRA-2 minus MAIRA-2 alone | +0.016 | +0.005 to +0.028 | +0.005 to +0.029 |
| CheXagent alone minus MAIRA-2 alone | +0.170 | +0.140 to +0.202 | +0.138 to +0.203 |

Conditions (of the 12 averaged) where the report score is the same for every image: emphysema, fibrosis, pleural thickening.

### Experiment 2: base MedGemma, prior + current (499 pairs)

499 images, 499 patients; mean over 9 conditions with at least 10 positives.

| system | mean AUC | 95% CI (image) | 95% CI (patient-clustered) |
|---|---|---|---|
| report alone | 0.545 | 0.532-0.559 | 0.531-0.559 |
| CheXagent alone | 0.840 | 0.811-0.864 | 0.813-0.864 |
| MAIRA-2 alone | 0.667 | 0.624-0.706 | 0.628-0.705 |
| report + CheXagent | 0.835 | 0.807-0.860 | 0.808-0.860 |
| report + MAIRA-2 | 0.673 | 0.631-0.712 | 0.634-0.711 |

| comparison | difference | 95% CI (image) | 95% CI (patient-clustered) |
|---|---|---|---|
| report + CheXagent minus report alone | +0.291 | +0.264 to +0.315 | +0.264 to +0.315 |
| report + CheXagent minus report + MAIRA-2 | +0.162 | +0.123 to +0.203 | +0.124 to +0.198 |
| report + CheXagent minus CheXagent alone | -0.004 | -0.007 to -0.002 | -0.007 to -0.001 |
| report + MAIRA-2 minus MAIRA-2 alone | +0.007 | +0.001 to +0.012 | +0.001 to +0.013 |
| CheXagent alone minus MAIRA-2 alone | +0.173 | +0.132 to +0.215 | +0.134 to +0.211 |

Conditions (of the 9 averaged) where the report score is the same for every image: emphysema, pleural thickening.

### Experiment 3: trained MedGemma, prior + current (499 pairs)

499 images, 499 patients; mean over 9 conditions with at least 10 positives.

| system | mean AUC | 95% CI (image) | 95% CI (patient-clustered) |
|---|---|---|---|
| report alone | 0.593 | 0.572-0.615 | 0.571-0.615 |
| CheXagent alone | 0.840 | 0.813-0.865 | 0.813-0.866 |
| MAIRA-2 alone | 0.667 | 0.624-0.707 | 0.626-0.708 |
| report + CheXagent | 0.837 | 0.811-0.863 | 0.810-0.863 |
| report + MAIRA-2 | 0.693 | 0.655-0.731 | 0.655-0.730 |

| comparison | difference | 95% CI (image) | 95% CI (patient-clustered) |
|---|---|---|---|
| report + CheXagent minus report alone | +0.244 | +0.217 to +0.269 | +0.220 to +0.270 |
| report + CheXagent minus report + MAIRA-2 | +0.144 | +0.109 to +0.177 | +0.109 to +0.179 |
| report + CheXagent minus CheXagent alone | -0.003 | -0.008 to +0.004 | -0.008 to +0.003 |
| report + MAIRA-2 minus MAIRA-2 alone | +0.027 | +0.010 to +0.047 | +0.011 to +0.046 |
| CheXagent alone minus MAIRA-2 alone | +0.173 | +0.133 to +0.212 | +0.132 to +0.212 |

Conditions (of the 9 averaged) where the report score is the same for every image: emphysema.

## Grounding recall: patient-clustered bootstrap (findings of one patient kept together)

| threshold | CheXagent minus MAIRA-2 | 95% CI (patient-clustered) |
|---|---|---|
| IoU 0.5 | +4.3% | +1.5% to +7.3% |
| IoU 0.3 | +22.2% | +18.9% to +25.6% |
