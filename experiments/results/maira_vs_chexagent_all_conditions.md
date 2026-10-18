# CheXagent vs MAIRA-2 presence scores, all 13 conditions

Amendments A1 (six findings) and A2 (seven further conditions) of `maira_score_design.md`, both committed before their readings. AUC: how often a model scores an image with the condition above one without it (0.5 = chance). Paired 95% CIs from 2000 image-level bootstrap resamples. Hypothesis: CheXagent's AUC is higher - supported if the difference's CI lies above 0.

Caveats: CheXagent's training data probably include public NIH and VinDr images; MAIRA-2's do not. VinDr labels nodule and mass together, so both are scored against that one label. NIH labels are NLP-mined.

| dataset | condition | positives / n | CheXagent AUC [95% CI] | MAIRA-2 AUC [95% CI] | CheXagent minus MAIRA-2 [95% CI] | hypothesis |
|---|---|---|---|---|---|---|
| **Six findings (A1)** | | | | | | |
| NIH test | cardiomegaly | 35 / 1500 | 0.936 [0.895-0.965] | 0.707 [0.615-0.792] | +0.229 [+0.146 to +0.320] | supported |
| NIH test | pleural effusion | 77 / 1500 | 0.885 [0.836-0.928] | 0.723 [0.655-0.788] | +0.162 [+0.098 to +0.226] | supported |
| NIH test | pneumothorax | 18 / 1500 | 0.894 [0.770-0.978] | 0.532 [0.412-0.650] | +0.362 [+0.163 to +0.540] | supported |
| NIH test | consolidation | 27 / 1500 | 0.866 [0.781-0.928] | 0.764 [0.645-0.868] | +0.101 [-0.004 to +0.211] | not supported |
| NIH test | pulmonary edema | 10 / 1500 | 0.917 [0.780-0.995] | 0.711 [0.546-0.849] | +0.206 [+0.092 to +0.345] | supported |
| NIH test | atelectasis | 93 / 1500 | 0.894 [0.857-0.925] | 0.780 [0.736-0.820] | +0.115 [+0.079 to +0.149] | supported |
| Indiana | cardiomegaly | 164 / 544 | 0.927 [0.904-0.946] | 0.729 [0.684-0.773] | +0.198 [+0.154 to +0.245] | supported |
| Indiana | pleural effusion | 120 / 544 | 0.953 [0.934-0.970] | 0.850 [0.797-0.895] | +0.103 [+0.062 to +0.153] | supported |
| Indiana | pneumothorax | 28 / 544 | 0.961 [0.914-0.991] | 0.611 [0.493-0.727] | +0.349 [+0.215 to +0.483] | supported |
| Indiana | consolidation | 109 / 544 | 0.797 [0.749-0.842] | 0.683 [0.621-0.743] | +0.115 [+0.066 to +0.169] | supported |
| Indiana | pulmonary edema | 45 / 544 | 0.894 [0.830-0.946] | 0.741 [0.658-0.817] | +0.152 [+0.076 to +0.230] | supported |
| Indiana | atelectasis | 161 / 544 | 0.802 [0.762-0.837] | 0.711 [0.666-0.756] | +0.091 [+0.050 to +0.129] | supported |
| VinDr | cardiomegaly | 358 / 3000 | 0.983 [0.979-0.987] | 0.748 [0.717-0.776] | +0.236 [+0.208 to +0.264] | supported |
| VinDr | pleural effusion | 120 / 3000 | 0.995 [0.992-0.997] | 0.813 [0.763-0.864] | +0.182 [+0.132 to +0.231] | supported |
| VinDr | pneumothorax | 15 / 3000 | 0.999 [0.998-1.000] | 0.505 [0.352-0.663] | +0.494 [+0.335 to +0.648] | supported |
| VinDr | consolidation | 25 / 3000 | 0.991 [0.985-0.995] | 0.954 [0.924-0.976] | +0.037 [+0.016 to +0.064] | supported |
| VinDr | atelectasis | 10 / 3000 | 0.986 [0.975-0.996] | 0.783 [0.610-0.926] | +0.203 [+0.057 to +0.379] | supported |
| **Seven further conditions (A2)** | | | | | | |
| NIH test | pneumonia | 9 / 1500 | 0.734 [0.534-0.913] | 0.622 [0.432-0.807] | +0.113 [-0.019 to +0.278] | not supported (fewer than 10 positives) |
| NIH test | infiltration | 185 / 1500 | 0.600 [0.551-0.646] | 0.586 [0.540-0.630] | +0.013 [-0.016 to +0.044] | not supported |
| NIH test | emphysema | 14 / 1500 | 0.921 [0.865-0.971] | 0.580 [0.416-0.738] | +0.341 [+0.170 to +0.519] | supported |
| NIH test | nodule | 85 / 1500 | 0.802 [0.749-0.848] | 0.687 [0.621-0.750] | +0.114 [+0.057 to +0.174] | supported |
| NIH test | mass | 67 / 1500 | 0.935 [0.906-0.960] | 0.852 [0.799-0.901] | +0.082 [+0.033 to +0.141] | supported |
| NIH test | fibrosis | 24 / 1500 | 0.823 [0.734-0.904] | 0.723 [0.611-0.813] | +0.101 [+0.041 to +0.166] | supported |
| NIH test | pleural thickening | 30 / 1500 | 0.916 [0.873-0.952] | 0.699 [0.607-0.780] | +0.217 [+0.145 to +0.294] | supported |
| VinDr | infiltration | 44 / 3000 | 0.981 [0.973-0.988] | 0.926 [0.897-0.950] | +0.055 [+0.033 to +0.081] | supported |
| VinDr | nodule (VinDr Nodule/Mass label) | 90 / 3000 | 0.967 [0.950-0.979] | 0.805 [0.755-0.850] | +0.162 [+0.120 to +0.210] | supported |
| VinDr | mass (VinDr Nodule/Mass label) | 90 / 3000 | 0.948 [0.931-0.963] | 0.866 [0.822-0.904] | +0.082 [+0.046 to +0.123] | supported |
| VinDr | fibrosis | 182 / 3000 | 0.973 [0.964-0.981] | 0.857 [0.827-0.884] | +0.116 [+0.089 to +0.145] | supported |
| VinDr | pleural thickening | 178 / 3000 | 0.953 [0.941-0.964] | 0.676 [0.627-0.724] | +0.277 [+0.229 to +0.327] | supported |

Six findings (A1): 17 comparisons, supported 16.
Seven further conditions (A2): 12 comparisons, supported 10, not supported 2, reversed 0.
All conditions: 29 comparisons, supported 26.
