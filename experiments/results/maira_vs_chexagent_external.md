# Standalone presence scores: CheXagent vs MAIRA-2 on three held-out labelled sets

Amendment A1 of `maira_score_design.md` (committed before these readings and analyses). AUC: how often a model scores an image with the finding above one without it (0.5 = chance). 95% CIs from 2000 image-level bootstrap resamples (seed 20261017); the difference is paired (same images). Hypothesis: CheXagent's AUC is higher - supported if the difference's CI lies above 0.

Caveat: CheXagent's training data probably include public NIH and VinDr images; MAIRA-2's published training sets do not. Indiana is the least affected set.

| dataset | finding | positives / n | CheXagent AUC [95% CI] | MAIRA-2 AUC [95% CI] | CheXagent minus MAIRA-2 [95% CI] | hypothesis |
|---|---|---|---|---|---|---|
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

Summary over 17 comparisons: supported 16, not supported 1, reversed 0.
