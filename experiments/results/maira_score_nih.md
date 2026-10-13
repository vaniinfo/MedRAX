# MAIRA-2 presence score (no training) vs CheXagent p_present: NIH development split

Design: `maira_score_design.md` (committed before the readings). Score = P(positive sentence) / (P(positive) + P(negative)) as the start of MAIRA-2's findings. AUC 95% CI from 2000 image-level bootstrap resamples (seed 20261013). Decision rule: calibration only for findings with MAIRA-2 AUC >= 0.75.

| finding | positives / n | MAIRA-2 AUC [95% CI] | CheXagent AUC [95% CI] | MAIRA-2 minus CheXagent [95% CI] | goes to calibration |
|---|---|---|---|---|---|
| cardiomegaly | 80 / 3500 | 0.781 [0.732-0.828] | 0.936 [0.917-0.954] | -0.155 [-0.202 to -0.109] | yes |
| pleural effusion | 180 / 3500 | 0.769 [0.724-0.811] | 0.930 [0.906-0.950] | -0.161 [-0.203 to -0.120] | yes |
| pneumothorax | 43 / 3500 | 0.460 [0.382-0.543] | 0.897 [0.825-0.957] | -0.438 [-0.535 to -0.330] | no |
| consolidation | 62 / 3500 | 0.685 [0.606-0.771] | 0.829 [0.772-0.882] | -0.144 [-0.218 to -0.067] | no |
| pulmonary edema | 24 / 3500 | 0.742 [0.619-0.842] | 0.917 [0.826-0.973] | -0.176 [-0.280 to -0.090] | no |
| atelectasis | 218 / 3500 | 0.792 [0.765-0.820] | 0.883 [0.859-0.904] | -0.091 [-0.111 to -0.070] | yes |
