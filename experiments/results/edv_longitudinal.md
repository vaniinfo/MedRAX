# Longitudinal EDV (prior + current): results

Protocol: `edv_longitudinal_protocol.md` (committed before the readings). 499 NIH test pairs. Mean AUC over the 9 conditions with at least 10 positives; paired 95% CIs from 2000 image-level bootstrap resamples. CheXagent A gave no Yes/No answer for 0 of 6487 image-condition pairs (excluded for all systems).

## Mean AUC

| system | mean AUC [95% CI] |
|---|---|
| MedGemma-L alone | 0.545 [0.531-0.560] |
| MedGemma-L + CheXagent A | 0.835 [0.809-0.861] |
| MedGemma-L + CheXagent B | 0.819 [0.791-0.847] |
| MedGemma-L + MAIRA-2-L | 0.673 [0.631-0.711] |
| MedGemma-S alone | 0.545 [0.531-0.561] |
| MedGemma-S + CheXagent-S | 0.830 [0.801-0.856] |
| MedGemma-S + MAIRA-2-S | 0.670 [0.627-0.713] |
| CheXagent A alone | 0.840 [0.813-0.865] |
| CheXagent B alone | 0.821 [0.793-0.849] |
| CheXagent-S alone | 0.835 [0.807-0.861] |
| MAIRA-2-L alone | 0.667 [0.624-0.705] |
| MAIRA-2-S alone | 0.659 [0.616-0.701] |

## Hypotheses (primary) and exploratory comparisons

| comparison | difference [95% CI] | result |
|---|---|---|
| H1-L: MedGemma-L + CheXagent A minus MedGemma-L alone | +0.291 [+0.264 to +0.316] | supported |
| H2-L: MedGemma-L + CheXagent A minus MedGemma-L + MAIRA-2-L | +0.162 [+0.124 to +0.201] | supported |
| (exploratory) CheXagent A alone minus CheXagent single-image alone | +0.005 [-0.002 to +0.012] | not supported |
| (exploratory) CheXagent B alone minus CheXagent single-image alone | -0.014 [-0.026 to -0.002] | reversed |
| (exploratory) MedGemma-L alone minus MedGemma-S alone | -0.001 [-0.016 to +0.014] | not supported |
| (exploratory) MedGemma-L + CheXagent A minus MedGemma-S + CheXagent-S | +0.006 [-0.002 to +0.014] | not supported |
| (exploratory) MedGemma-L + CheXagent B minus MedGemma-L alone | +0.274 [+0.246 to +0.301] | supported |
| (exploratory) MAIRA-2-L alone minus MAIRA-2 single-image alone | +0.008 [-0.022 to +0.039] | not supported |

## Per condition (AUC)

| condition | positives / n | MedGemma-L alone | MedGemma-L + CheXagent A | MedGemma-L + CheXagent B | MedGemma-L + MAIRA-2-L | CheXagent A alone | CheXagent-S alone | MedGemma-S alone |
|---|---|---|---|---|---|---|---|---|
| cardiomegaly (fewer than 10 positives) | 8 / 499 | 0.620 | 0.932 | 0.910 | 0.609 | 0.936 | 0.941 | 0.620 |
| pleural effusion | 45 / 499 | 0.775 | 0.839 | 0.823 | 0.818 | 0.849 | 0.858 | 0.765 |
| pneumothorax | 16 / 499 | 0.499 | 0.890 | 0.882 | 0.689 | 0.891 | 0.870 | 0.530 |
| consolidation | 17 / 499 | 0.514 | 0.805 | 0.787 | 0.763 | 0.814 | 0.818 | 0.481 |
| pulmonary edema (fewer than 10 positives) | 9 / 499 | 0.698 | 0.837 | 0.839 | 0.886 | 0.847 | 0.863 | 0.759 |
| atelectasis | 41 / 499 | 0.553 | 0.841 | 0.816 | 0.731 | 0.863 | 0.858 | 0.613 |
| pneumonia (fewer than 10 positives) | 4 / 499 | 0.500 | 0.603 | 0.621 | 0.467 | 0.603 | 0.586 | 0.500 |
| infiltration | 61 / 499 | 0.507 | 0.639 | 0.606 | 0.580 | 0.637 | 0.630 | 0.495 |
| emphysema | 11 / 499 | 0.500 | 0.939 | 0.903 | 0.476 | 0.939 | 0.927 | 0.500 |
| nodule | 31 / 499 | 0.513 | 0.794 | 0.788 | 0.660 | 0.794 | 0.771 | 0.511 |
| mass | 36 / 499 | 0.541 | 0.884 | 0.886 | 0.735 | 0.884 | 0.885 | 0.513 |
| fibrosis (fewer than 10 positives) | 8 / 499 | 0.500 | 0.868 | 0.836 | 0.616 | 0.868 | 0.858 | 0.500 |
| pleural thickening | 16 / 499 | 0.500 | 0.885 | 0.881 | 0.608 | 0.885 | 0.899 | 0.500 |

Limitations (recorded in the protocol): 499 pairs; NIH labels NLP-mined; base MedGemma; CheXagent option A beyond its 2,048-token design length; CheXagent probably saw NIH images; percentiles from the evaluation set (label-free).
