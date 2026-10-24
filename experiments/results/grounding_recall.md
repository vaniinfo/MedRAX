# Grounding recall: CheXagent vs MAIRA-2 on NIH radiologist boxes

Protocol: `grounding_recall_protocol.md` (committed before any box). 947 cases (one radiologist box each) on 847 images; the phrase names the condition without a side. Hit = the best of the model's boxes overlaps the radiologist's box with IoU >= t. 95% CIs: Wilson (recall), 2000 image-level bootstrap resamples (differences, seed 20261024).

## Pooled

| measure | CheXagent | MAIRA-2 |
|---|---|---|
| recall@0.5 (primary) | 23.3% (221/947) [20.8%-26.1%] | 19.0% (180/947) [16.6%-21.6%] |
| recall@0.3 | 52.5% (497/947) [49.3%-55.6%] | 30.3% (287/947) [27.5%-33.3%] |
| pointing game | 68.5% (649/947) [65.5%-71.4%] | 40.0% (379/947) [36.9%-43.2%] |
| union recall@0.5 | 23.2% (220/947) [20.7%-26.0%] | 18.7% (177/947) [16.3%-21.3%] |
| union recall@0.3 | 49.3% (467/947) [46.1%-52.5%] | 29.0% (275/947) [26.2%-32.0%] |
| mean best IoU | 0.31 | 0.22 |
| mean union IoU | 0.30 | 0.22 |
| mean boxes per case | 1.26 | 0.92 |
| median box area / radiologist box area | 0.91 | 1.25 |
| no box returned | 9 | 127 |

## Hypotheses (two-sided)

| comparison | CheXagent minus MAIRA-2 [95% CI] | McNemar b / c, p | result |
|---|---|---|---|
| G1 (primary): recall@0.5 | +4.3% [+1.7% to +7.0%] | 103 / 62, 0.0018 | CheXagent better |
| G2: recall@0.3 | +22.2% [+18.8% to +25.6%] | 267 / 57, 1.2e-33 | CheXagent better |

## Per label (descriptive; McNemar exact p unadjusted)

| label | cases | recall@0.5 CheXagent / MAIRA-2 (p) | recall@0.3 CheXagent / MAIRA-2 (p) | pointing game CheXagent / MAIRA-2 | mean best IoU CheXagent / MAIRA-2 |
|---|---|---|---|---|---|
| Atelectasis | 173 | 12% / 2% (0.00049) | 42% / 16% (2.1e-08) | 61% / 31% | 0.26 / 0.13 |
| Effusion | 150 | 6% / 3% (0.42) | 38% / 28% (0.072) | 67% / 41% | 0.24 / 0.17 |
| Cardiomegaly | 141 | 70% / 100% (4.5e-13) | 99% / 100% (1) | 99% / 100% | 0.56 / 0.75 |
| Infiltrate | 117 | 21% / 11% (0.043) | 58% / 29% (1.2e-06) | 74% / 44% | 0.34 / 0.19 |
| Pneumonia | 116 | 17% / 4% (0.00073) | 53% / 12% (2.4e-10) | 78% / 16% | 0.31 / 0.08 |
| Pneumothorax | 93 | 33% / 13% (6.6e-05) | 55% / 22% (3.7e-08) | 68% / 32% | 0.34 / 0.17 |
| Mass | 81 | 21% / 1% (0.00014) | 47% / 7% (6.7e-08) | 63% / 26% | 0.29 / 0.08 |
| Nodule | 76 | 1% / 0% (1) | 11% / 3% (0.11) | 17% / 3% | 0.08 / 0.03 |
