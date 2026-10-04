# EXPLORATORY: CheXagent-DenseNet disagreement as an error signal (development splits)

DenseNet: TorchXRayVision densenet121-res224-chex (CheXpert only). CheXagent call: p_present >= frozen development threshold. Disagreement d = |percentile-rank difference| within dataset and finding. 'Disagree' = top quartile of d. 95% CIs: 2,000-sample image-level bootstrap. These thresholds were fitted on these same development data.

## INDIANA development

| finding | n | positives | DenseNet AUC | CheXagent AUC | Spearman | CheXagent errors | AUROC of d for CheXagent error (95% CI) | error rate: agree / disagree |
|---|---|---|---|---|---|---|---|---|
| cardiomegaly | 381 | 115 | 0.789 | 0.924 | 0.67 | 65 | 0.537 (0.463-0.612) | 16.6% / 18.4% |
| pleural effusion | 381 | 84 | 0.735 | 0.949 | 0.70 | 40 | 0.556 (0.469-0.638) | 10.2% / 11.5% |
| pneumothorax | 381 | 20 | 0.629 | 0.975 | 0.25 | 23 | 0.515 (0.373-0.652) | 5.6% / 7.3% |
| consolidation | 381 | 76 | 0.637 | 0.756 | 0.68 | 106 | 0.501 (0.435-0.564) | 28.8% / 25.0% |
| pulmonary edema | 381 | 31 | 0.757 | 0.880 | 0.74 | 44 | 0.465 (0.378-0.553) | 12.4% / 9.2% |
| atelectasis | 381 | 113 | 0.621 | 0.805 | 0.58 | 109 | 0.508 (0.447-0.570) | 29.8% / 25.0% |

### INDIANA: directional (added after the first table; exploratory)

Among CheXagent's positive calls: AUROC of a LOW DenseNet score for a false positive. Among its negative calls: AUROC of a HIGH DenseNet score for a missed finding.

| finding | positive calls (false) | AUROC for false positive (95% CI) | negative calls (missed) | AUROC for missed finding (95% CI) |
|---|---|---|---|---|
| cardiomegaly | 160 (55) | 0.647 (0.558-0.733) | 221 (10) | 0.673 (0.491-0.836) |
| pleural effusion | 98 (27) | 0.455 (0.323-0.596) | 283 (13) | 0.628 (0.493-0.751) |
| pneumothorax | 39 (21) | 0.511 (0.326-0.701) | 342 (2) | 0.415 (0.259-0.570) |
| consolidation | 126 (78) | 0.509 (0.398-0.617) | 255 (28) | 0.551 (0.441-0.660) |
| pulmonary edema | 63 (38) | 0.497 (0.354-0.647) | 318 (6) | 0.463 (0.202-0.732) |
| atelectasis | 184 (90) | 0.431 (0.351-0.520) | 197 (19) | 0.630 (0.531-0.725) |

## NIH development

| finding | n | positives | DenseNet AUC | CheXagent AUC | Spearman | CheXagent errors | AUROC of d for CheXagent error (95% CI) | error rate: agree / disagree |
|---|---|---|---|---|---|---|---|---|
| cardiomegaly | 3920 | 195 | 0.752 | 0.921 | 0.64 | 592 | 0.470 (0.443-0.496) | 15.2% / 14.8% |
| pleural effusion | 3920 | 285 | 0.817 | 0.905 | 0.62 | 519 | 0.403 (0.377-0.430) | 14.6% / 9.2% |
| pneumothorax | 3920 | 160 | 0.686 | 0.896 | 0.31 | 389 | 0.481 (0.449-0.514) | 9.7% / 10.6% |
| consolidation | 3920 | 179 | 0.763 | 0.831 | 0.64 | 1007 | 0.365 (0.344-0.384) | 29.5% / 14.3% |
| pulmonary edema | 3920 | 139 | 0.849 | 0.934 | 0.76 | 538 | 0.360 (0.335-0.383) | 15.6% / 8.2% |
| atelectasis | 3920 | 281 | 0.719 | 0.884 | 0.64 | 683 | 0.431 (0.406-0.454) | 18.7% / 13.6% |

### NIH: directional (added after the first table; exploratory)

Among CheXagent's positive calls: AUROC of a LOW DenseNet score for a false positive. Among its negative calls: AUROC of a HIGH DenseNet score for a missed finding.

| finding | positive calls (false) | AUROC for false positive (95% CI) | negative calls (missed) | AUROC for missed finding (95% CI) |
|---|---|---|---|---|
| cardiomegaly | 725 (561) | 0.516 (0.468-0.564) | 3195 (31) | 0.661 (0.575-0.745) |
| pleural effusion | 704 (469) | 0.588 (0.544-0.634) | 3216 (50) | 0.654 (0.584-0.727) |
| pneumothorax | 471 (350) | 0.569 (0.511-0.629) | 3449 (39) | 0.528 (0.443-0.613) |
| consolidation | 1114 (971) | 0.603 (0.555-0.651) | 2806 (36) | 0.579 (0.487-0.668) |
| pulmonary edema | 647 (523) | 0.561 (0.499-0.618) | 3273 (15) | 0.770 (0.630-0.883) |
| atelectasis | 844 (623) | 0.461 (0.417-0.504) | 3076 (60) | 0.653 (0.584-0.723) |

## Evidence Engine states, both datasets pooled

How often a state is wrong, when the two models agree (bottom 3 quartiles of d) vs disagree (top quartile).

| state | agree: n, wrong | disagree: n, wrong |
|---|---|---|
| SUPPORTING | 1072, 56.0% | 309, 50.2% |
| REFUTING | 4997, 0.3% | 1815, 0.6% |
| UNCERTAIN (share in group) | 67.9% | 66.6% |
