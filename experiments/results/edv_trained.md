# Does EDV improve the team's trained MedGemma? NIH longitudinal results

Protocol: `edv_trained_protocol.md` (committed before the test reports). 499 NIH test pairs; trained MedGemma with placeholders for prior report and indication. Mean AUC over the 9 conditions with at least 10 positives; paired 95% CIs from 2000 image-level bootstrap resamples (seed 20261025).

## Mean AUC

| system | mean AUC [95% CI] |
|---|---|
| Trained MedGemma alone | 0.593 [0.571-0.616] |
| Trained + CheXagent A | 0.837 [0.810-0.862] |
| Trained + CheXagent B | 0.823 [0.793-0.850] |
| Trained + CheXagent single-image | 0.833 [0.804-0.859] |
| Trained + MAIRA-2-L | 0.693 [0.655-0.729] |
| Base MedGemma-L alone | 0.545 [0.532-0.558] |
| Base MedGemma-L + CheXagent A | 0.835 [0.809-0.860] |

## Hypotheses and exploratory comparisons

| comparison | difference [95% CI] | result |
|---|---|---|
| H3 (primary): trained + CheXagent A minus trained alone | +0.244 [+0.220 to +0.269] | supported |
| H3b: trained + CheXagent A minus trained + MAIRA-2-L | +0.144 [+0.111 to +0.178] | supported |
| (exploratory) trained alone minus base MedGemma-L alone | +0.049 [+0.025 to +0.073] | supported |
| (exploratory) trained + CheXagent A minus base + CheXagent A | +0.002 [-0.004 to +0.009] | not supported |
| (exploratory) trained + CheXagent B minus trained alone | +0.230 [+0.203 to +0.256] | supported |
| (exploratory) trained + CheXagent single-image minus trained alone | +0.240 [+0.215 to +0.265] | supported |

## Secondary: the [Image check] rule (yes/no)

| | mean sensitivity | mean specificity | mean balanced accuracy |
|---|---|---|---|
| trained MedGemma alone | 0.237 | 0.947 | 0.592 |
| trained + [Image check] rule | 0.237 | 0.946 | 0.592 |
| difference [95% CI] | +0.000 [+0.000, +0.000] | -0.001 [-0.002, -0.000] | -0.000 [-0.001, -0.000] |

## Descriptive: reports that compare with the prior

- Trained MedGemma: 487 of 499
- Base MedGemma (longitudinal run): 3 of 499

## Per condition (AUC)

| condition | positives / n | Trained MedGemma alone | Trained + CheXagent A | Trained + CheXagent B | Trained + CheXagent single-image | Trained + MAIRA-2-L | Base MedGemma-L alone | Base MedGemma-L + CheXagent A |
|---|---|---|---|---|---|---|---|---|
| cardiomegaly (fewer than 10 positives) | 8 / 499 | 0.706 | 0.913 | 0.896 | 0.916 | 0.717 | 0.620 | 0.932 |
| pleural effusion | 45 / 499 | 0.818 | 0.850 | 0.832 | 0.857 | 0.834 | 0.775 | 0.839 |
| pneumothorax | 16 / 499 | 0.586 | 0.884 | 0.878 | 0.864 | 0.783 | 0.499 | 0.890 |
| consolidation | 17 / 499 | 0.669 | 0.832 | 0.817 | 0.838 | 0.783 | 0.514 | 0.805 |
| pulmonary edema (fewer than 10 positives) | 9 / 499 | 0.847 | 0.851 | 0.855 | 0.865 | 0.901 | 0.698 | 0.837 |
| atelectasis | 41 / 499 | 0.665 | 0.845 | 0.836 | 0.840 | 0.749 | 0.553 | 0.841 |
| pneumonia (fewer than 10 positives) | 4 / 499 | 0.719 | 0.667 | 0.672 | 0.654 | 0.605 | 0.500 | 0.603 |
| infiltration | 61 / 499 | 0.528 | 0.631 | 0.600 | 0.625 | 0.574 | 0.507 | 0.639 |
| emphysema | 11 / 499 | 0.500 | 0.939 | 0.903 | 0.927 | 0.476 | 0.500 | 0.939 |
| nodule | 31 / 499 | 0.496 | 0.785 | 0.779 | 0.762 | 0.649 | 0.513 | 0.794 |
| mass | 36 / 499 | 0.578 | 0.881 | 0.883 | 0.884 | 0.786 | 0.541 | 0.884 |
| fibrosis (fewer than 10 positives) | 8 / 499 | 0.493 | 0.863 | 0.831 | 0.852 | 0.614 | 0.500 | 0.868 |
| pleural thickening | 16 / 499 | 0.499 | 0.884 | 0.880 | 0.898 | 0.605 | 0.500 | 0.885 |

Limitations (recorded in the protocol): placeholders for prior report and indication; NIH rather than MIMIC images; NIH labels NLP-mined; CheXagent probably saw NIH images; percentiles from the evaluation set; MedGemma's 3-level score caps its AUC.
