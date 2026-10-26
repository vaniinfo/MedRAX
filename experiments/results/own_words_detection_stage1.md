# CheXagent vs MAIRA-2 in their own words: caught vs missed (NIH test split, stage 1)

Protocol: `own_words_detection_protocol.md` (committed before any report), amendments A1-A2. Stage 1: a fixed random 500 of the 1,500 NIH test images; C-report = CheXagent's six-area report, C-answer = CheXagent's 'Is there X?' with Yes outweighing No (p_present >= 0.5), MAIRA-2 = its own findings report. Reports read by the quote-checked Qwen lookup. Means over the conditions with at least 10 positives; 2000 image-level bootstrap resamples (seed 20261023); two-sided.

## primary: present only

| system | mean sensitivity | mean specificity | mean balanced accuracy [95% CI] |
|---|---|---|---|
| C-report | 0.328 | 0.959 | 0.643 [0.604-0.680] |
| C-answer | 0.621 | 0.899 | 0.760 [0.715-0.803] |
| MAIRA-2 | 0.290 | 0.956 | 0.623 [0.587-0.661] |

| comparison | measure | difference [95% CI] | result |
|---|---|---|---|
| H1: C-report minus MAIRA-2 | balanced accuracy | +0.020 [-0.014 to +0.053] | no difference shown |
|  | sensitivity | +0.038 [-0.030 to +0.101] | no difference shown |
|  | specificity | +0.003 [-0.005 to +0.011] | no difference shown |
| H2: C-answer minus MAIRA-2 | balanced accuracy | +0.137 [+0.092 to +0.180] | CheXagent better |
|  | sensitivity | +0.331 [+0.246 to +0.415] | CheXagent better |
|  | specificity | -0.057 [-0.069 to -0.046] | MAIRA-2 better |

Per condition: caught / labelled present (sensitivity, Wilson 95% CI) and false alarms / labelled absent; McNemar exact p on discordant pairs vs MAIRA-2 (descriptive, unadjusted).

| condition | positives / n | system | caught | sensitivity [95% CI] | false alarms | specificity [95% CI] | McNemar p (caught / false alarms) |
|---|---|---|---|---|---|---|---|
| cardiomegaly | 19 / 500 | C-report | 8 | 0.421 [0.231-0.637] | 17 | 0.965 [0.944-0.978] | 1 / 0.503 |
| cardiomegaly | 19 / 500 | C-answer | 18 | 0.947 [0.754-0.991] | 41 | 0.915 [0.886-0.937] | 0.00195 / 0.000325 |
| cardiomegaly | 19 / 500 | MAIRA-2 | 8 | 0.421 [0.231-0.637] | 21 | 0.956 [0.934-0.971] |  |
| pleural effusion | 18 / 500 | C-report | 10 | 0.556 [0.337-0.754] | 67 | 0.861 [0.827-0.889] | 0.375 / 2.29e-08 |
| pleural effusion | 18 / 500 | C-answer | 9 | 0.500 [0.290-0.710] | 30 | 0.938 [0.913-0.956] | 0.5 / 0.711 |
| pleural effusion | 18 / 500 | MAIRA-2 | 7 | 0.389 [0.203-0.614] | 27 | 0.944 [0.920-0.961] |  |
| pneumothorax (fewer than 10 positives) | 6 / 500 | C-report | 5 | 0.833 [0.436-0.970] | 20 | 0.960 [0.938-0.974] | 0.5 / 0.00149 |
| pneumothorax | 6 / 500 | C-answer | 5 | 0.833 [0.436-0.970] | 8 | 0.984 [0.968-0.992] | 0.5 / 0.549 |
| pneumothorax | 6 / 500 | MAIRA-2 | 3 | 0.500 [0.188-0.812] | 5 | 0.990 [0.977-0.996] |  |
| consolidation | 10 / 500 | C-report | 4 | 0.400 [0.168-0.687] | 6 | 0.988 [0.974-0.994] | 0.5 / 0.302 |
| consolidation | 10 / 500 | C-answer | 6 | 0.600 [0.313-0.832] | 19 | 0.961 [0.940-0.975] | 0.125 / 0.0963 |
| consolidation | 10 / 500 | MAIRA-2 | 2 | 0.200 [0.057-0.510] | 11 | 0.978 [0.960-0.987] |  |
| pulmonary edema (fewer than 10 positives) | 4 / 500 | C-report | 0 | 0.000 [0.000-0.490] | 4 | 0.992 [0.979-0.997] | 1 / 0.549 |
| pulmonary edema | 4 / 500 | C-answer | 2 | 0.500 [0.150-0.850] | 7 | 0.986 [0.971-0.993] | 0.5 / 1 |
| pulmonary edema | 4 / 500 | MAIRA-2 | 0 | 0.000 [0.000-0.490] | 7 | 0.986 [0.971-0.993] |  |
| atelectasis | 21 / 500 | C-report | 10 | 0.476 [0.283-0.676] | 23 | 0.952 [0.929-0.968] | 0.754 / 0.000535 |
| atelectasis | 21 / 500 | C-answer | 14 | 0.667 [0.454-0.828] | 72 | 0.850 [0.815-0.879] | 0.109 / 0.00954 |
| atelectasis | 21 / 500 | MAIRA-2 | 8 | 0.381 [0.208-0.591] | 49 | 0.898 [0.867-0.922] |  |
| pneumonia (fewer than 10 positives) | 3 / 500 | C-report | 0 | 0.000 [0.000-0.562] | 0 | 1.000 [0.992-1.000] | 1 / 0.25 |
| pneumonia | 3 / 500 | C-answer | 0 | 0.000 [0.000-0.562] | 15 | 0.970 [0.951-0.982] | 1 / 0.00183 |
| pneumonia | 3 / 500 | MAIRA-2 | 0 | 0.000 [0.000-0.562] | 3 | 0.994 [0.982-0.998] |  |
| infiltration | 65 / 500 | C-report | 1 | 0.015 [0.003-0.082] | 3 | 0.993 [0.980-0.998] | 0.125 / 0.0215 |
| infiltration | 65 / 500 | C-answer | 18 | 0.277 [0.183-0.396] | 37 | 0.915 [0.885-0.938] | 0.000977 / 2.56e-06 |
| infiltration | 65 / 500 | MAIRA-2 | 5 | 0.077 [0.033-0.168] | 11 | 0.975 [0.955-0.986] |  |
| emphysema (fewer than 10 positives) | 8 / 500 | C-report | 0 | 0.000 [0.000-0.324] | 5 | 0.990 [0.976-0.996] | 1 / 0.549 |
| emphysema | 8 / 500 | C-answer | 6 | 0.750 [0.409-0.929] | 33 | 0.933 [0.907-0.952] | 0.0312 / 1.09e-05 |
| emphysema | 8 / 500 | MAIRA-2 | 0 | 0.000 [0.000-0.324] | 8 | 0.984 [0.968-0.992] |  |
| nodule | 31 / 500 | C-report | 5 | 0.161 [0.071-0.326] | 13 | 0.972 [0.953-0.984] | 1 / 0.143 |
| nodule | 31 / 500 | C-answer | 16 | 0.516 [0.348-0.680] | 72 | 0.846 [0.811-0.876] | 0.00195 / 3.07e-12 |
| nodule | 31 / 500 | MAIRA-2 | 6 | 0.194 [0.092-0.363] | 20 | 0.957 [0.935-0.972] |  |
| mass | 19 / 500 | C-report | 5 | 0.263 [0.118-0.488] | 8 | 0.983 [0.968-0.992] | 0.5 / 1 |
| mass | 19 / 500 | C-answer | 16 | 0.842 [0.624-0.945] | 65 | 0.865 [0.831-0.893] | 0.00391 / 1.39e-17 |
| mass | 19 / 500 | MAIRA-2 | 7 | 0.368 [0.191-0.590] | 8 | 0.983 [0.968-0.992] |  |
| fibrosis (fewer than 10 positives) | 7 / 500 | C-report | 1 | 0.143 [0.026-0.513] | 2 | 0.996 [0.985-0.999] | 1 / 0.0225 |
| fibrosis | 7 / 500 | C-answer | 5 | 0.714 [0.359-0.918] | 43 | 0.913 [0.885-0.935] | 0.0625 / 3.31e-06 |
| fibrosis | 7 / 500 | MAIRA-2 | 0 | 0.000 [0.000-0.354] | 11 | 0.978 [0.960-0.987] |  |
| pleural thickening (fewer than 10 positives) | 9 / 500 | C-report | 3 | 0.333 [0.121-0.646] | 4 | 0.992 [0.979-0.997] | 1 / 0.508 |
| pleural thickening | 9 / 500 | C-answer | 8 | 0.889 [0.565-0.980] | 63 | 0.872 [0.839-0.898] | 0.0312 / 4.09e-16 |
| pleural thickening | 9 / 500 | MAIRA-2 | 2 | 0.222 [0.063-0.547] | 7 | 0.986 [0.971-0.993] |  |

## possible counts as stated (sensitivity analysis)

| system | mean sensitivity | mean specificity | mean balanced accuracy [95% CI] |
|---|---|---|---|
| C-report | 0.343 | 0.955 | 0.649 [0.611-0.686] |
| C-answer | 0.621 | 0.899 | 0.760 [0.715-0.803] |
| MAIRA-2 | 0.297 | 0.947 | 0.622 [0.585-0.661] |

| comparison | measure | difference [95% CI] | result |
|---|---|---|---|
| H1: C-report minus MAIRA-2 | balanced accuracy | +0.027 [-0.006 to +0.059] | no difference shown |
|  | sensitivity | +0.047 [-0.022 to +0.111] | no difference shown |
|  | specificity | +0.008 [-0.001 to +0.016] | no difference shown |
| H2: C-answer minus MAIRA-2 | balanced accuracy | +0.138 [+0.093 to +0.181] | CheXagent better |
|  | sensitivity | +0.325 [+0.238 to +0.409] | CheXagent better |
|  | specificity | -0.049 [-0.060 to -0.037] | MAIRA-2 better |

Per condition: caught / labelled present (sensitivity, Wilson 95% CI) and false alarms / labelled absent; McNemar exact p on discordant pairs vs MAIRA-2 (descriptive, unadjusted).

| condition | positives / n | system | caught | sensitivity [95% CI] | false alarms | specificity [95% CI] | McNemar p (caught / false alarms) |
|---|---|---|---|---|---|---|---|
| cardiomegaly | 19 / 500 | C-report | 8 | 0.421 [0.231-0.637] | 17 | 0.965 [0.944-0.978] | 1 / 0.503 |
| cardiomegaly | 19 / 500 | C-answer | 18 | 0.947 [0.754-0.991] | 41 | 0.915 [0.886-0.937] | 0.00195 / 0.000325 |
| cardiomegaly | 19 / 500 | MAIRA-2 | 8 | 0.421 [0.231-0.637] | 21 | 0.956 [0.934-0.971] |  |
| pleural effusion | 18 / 500 | C-report | 12 | 0.667 [0.437-0.837] | 74 | 0.846 [0.812-0.876] | 0.125 / 1.67e-07 |
| pleural effusion | 18 / 500 | C-answer | 9 | 0.500 [0.290-0.710] | 30 | 0.938 [0.913-0.956] | 0.5 / 0.72 |
| pleural effusion | 18 / 500 | MAIRA-2 | 7 | 0.389 [0.203-0.614] | 33 | 0.932 [0.905-0.951] |  |
| pneumothorax (fewer than 10 positives) | 6 / 500 | C-report | 5 | 0.833 [0.436-0.970] | 20 | 0.960 [0.938-0.974] | 0.5 / 0.00149 |
| pneumothorax | 6 / 500 | C-answer | 5 | 0.833 [0.436-0.970] | 8 | 0.984 [0.968-0.992] | 0.5 / 0.549 |
| pneumothorax | 6 / 500 | MAIRA-2 | 3 | 0.500 [0.188-0.812] | 5 | 0.990 [0.977-0.996] |  |
| consolidation | 10 / 500 | C-report | 4 | 0.400 [0.168-0.687] | 6 | 0.988 [0.974-0.994] | 0.5 / 0.0636 |
| consolidation | 10 / 500 | C-answer | 6 | 0.600 [0.313-0.832] | 19 | 0.961 [0.940-0.975] | 0.125 / 0.523 |
| consolidation | 10 / 500 | MAIRA-2 | 2 | 0.200 [0.057-0.510] | 15 | 0.969 [0.950-0.981] |  |
| pulmonary edema (fewer than 10 positives) | 4 / 500 | C-report | 0 | 0.000 [0.000-0.490] | 6 | 0.988 [0.974-0.994] | 0.5 / 0.227 |
| pulmonary edema | 4 / 500 | C-answer | 2 | 0.500 [0.150-0.850] | 7 | 0.986 [0.971-0.993] | 1 / 0.344 |
| pulmonary edema | 4 / 500 | MAIRA-2 | 2 | 0.500 [0.150-0.850] | 11 | 0.978 [0.961-0.988] |  |
| atelectasis | 21 / 500 | C-report | 10 | 0.476 [0.283-0.676] | 30 | 0.937 [0.912-0.956] | 1 / 1.57e-05 |
| atelectasis | 21 / 500 | C-answer | 14 | 0.667 [0.454-0.828] | 72 | 0.850 [0.815-0.879] | 0.18 / 0.5 |
| atelectasis | 21 / 500 | MAIRA-2 | 9 | 0.429 [0.245-0.635] | 65 | 0.864 [0.831-0.892] |  |
| pneumonia (fewer than 10 positives) | 3 / 500 | C-report | 0 | 0.000 [0.000-0.562] | 4 | 0.992 [0.979-0.997] | 1 / 1.94e-05 |
| pneumonia | 3 / 500 | C-answer | 0 | 0.000 [0.000-0.562] | 15 | 0.970 [0.951-0.982] | 1 / 0.11 |
| pneumonia | 3 / 500 | MAIRA-2 | 0 | 0.000 [0.000-0.562] | 25 | 0.950 [0.927-0.966] |  |
| infiltration | 65 / 500 | C-report | 1 | 0.015 [0.003-0.082] | 3 | 0.993 [0.980-0.998] | 0.125 / 0.00635 |
| infiltration | 65 / 500 | C-answer | 18 | 0.277 [0.183-0.396] | 37 | 0.915 [0.885-0.938] | 0.000977 / 3.86e-05 |
| infiltration | 65 / 500 | MAIRA-2 | 5 | 0.077 [0.033-0.168] | 13 | 0.970 [0.950-0.982] |  |
| emphysema (fewer than 10 positives) | 8 / 500 | C-report | 0 | 0.000 [0.000-0.324] | 5 | 0.990 [0.976-0.996] | 1 / 0.549 |
| emphysema | 8 / 500 | C-answer | 6 | 0.750 [0.409-0.929] | 33 | 0.933 [0.907-0.952] | 0.0312 / 1.09e-05 |
| emphysema | 8 / 500 | MAIRA-2 | 0 | 0.000 [0.000-0.324] | 8 | 0.984 [0.968-0.992] |  |
| nodule | 31 / 500 | C-report | 5 | 0.161 [0.071-0.326] | 13 | 0.972 [0.953-0.984] | 1 / 0.143 |
| nodule | 31 / 500 | C-answer | 16 | 0.516 [0.348-0.680] | 72 | 0.846 [0.811-0.876] | 0.00195 / 3.07e-12 |
| nodule | 31 / 500 | MAIRA-2 | 6 | 0.194 [0.092-0.363] | 20 | 0.957 [0.935-0.972] |  |
| mass | 19 / 500 | C-report | 5 | 0.263 [0.118-0.488] | 9 | 0.981 [0.965-0.990] | 0.5 / 1 |
| mass | 19 / 500 | C-answer | 16 | 0.842 [0.624-0.945] | 65 | 0.865 [0.831-0.893] | 0.00391 / 2.78e-17 |
| mass | 19 / 500 | MAIRA-2 | 7 | 0.368 [0.191-0.590] | 9 | 0.981 [0.965-0.990] |  |
| fibrosis (fewer than 10 positives) | 7 / 500 | C-report | 1 | 0.143 [0.026-0.513] | 4 | 0.992 [0.979-0.997] | 1 / 0.0118 |
| fibrosis | 7 / 500 | C-answer | 5 | 0.714 [0.359-0.918] | 43 | 0.913 [0.885-0.935] | 0.0625 / 0.000142 |
| fibrosis | 7 / 500 | MAIRA-2 | 0 | 0.000 [0.000-0.354] | 16 | 0.968 [0.948-0.980] |  |
| pleural thickening (fewer than 10 positives) | 9 / 500 | C-report | 3 | 0.333 [0.121-0.646] | 5 | 0.990 [0.976-0.996] | 1 / 0.754 |
| pleural thickening | 9 / 500 | C-answer | 8 | 0.889 [0.565-0.980] | 63 | 0.872 [0.839-0.898] | 0.0312 / 4.09e-16 |
| pleural thickening | 9 / 500 | MAIRA-2 | 2 | 0.222 [0.063-0.547] | 7 | 0.986 [0.971-0.993] |  |

## A1: mean AUC on the 3-level report scores (1 / 0.5 / 0)

| system | mean AUC [95% CI] |
|---|---|
| C-report | 0.649 [0.610-0.686] |
| MAIRA-2 | 0.623 [0.586-0.662] |
| C-answer p_present (reference) | 0.838 [0.803-0.869] |

C-report minus MAIRA-2: +0.026 [-0.008 to +0.059], no difference shown.

| condition | positives / n | C-report AUC | MAIRA-2 AUC | C-answer p_present AUC |
|---|---|---|---|---|
| cardiomegaly | 19 / 500 | 0.693 | 0.689 | 0.959 |
| pleural effusion | 18 / 500 | 0.753 | 0.663 | 0.790 |
| pneumothorax (fewer than 10 positives) | 6 / 500 | 0.896 | 0.745 | 0.983 |
| consolidation | 10 / 500 | 0.694 | 0.586 | 0.925 |
| pulmonary edema (fewer than 10 positives) | 4 / 500 | 0.494 | 0.735 | 0.849 |
| atelectasis | 21 / 500 | 0.710 | 0.650 | 0.849 |
| pneumonia (fewer than 10 positives) | 3 / 500 | 0.496 | 0.475 | 0.484 |
| infiltration | 65 / 500 | 0.504 | 0.524 | 0.594 |
| emphysema (fewer than 10 positives) | 8 / 500 | 0.495 | 0.492 | 0.933 |
| nodule | 31 / 500 | 0.567 | 0.575 | 0.824 |
| mass | 19 / 500 | 0.622 | 0.675 | 0.923 |
| fibrosis (fewer than 10 positives) | 7 / 500 | 0.568 | 0.484 | 0.810 |
| pleural thickening (fewer than 10 positives) | 9 / 500 | 0.662 | 0.604 | 0.956 |
