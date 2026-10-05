# VinDr-CXR Amendment 1 (Part B): claim-blinded vs claim-conditioned grounding

Pre-registered in experiments/vindr_amendment_1_grounding.md. Part A is reported separately (vindr_external_validation.md).

## Cases

| finding | unilateral | bilateral |
|---|---|---|
| pleural effusion | 104 | 15 |
| atelectasis | 10 | 0 |
| consolidation | 24 | 1 |
| pneumothorax | 15 | 0 |

## CheXagent

Unilateral cases with all three readings: 153

- atelectasis: 10 unilateral cases, per-finding endpoints NOT evaluable (< 20)
- pneumothorax: 15 unilateral cases, per-finding endpoints NOT evaluable (< 20)

| cases | n | blinded: agree correct / wrong | LR+_blind (95% CI) | conditioned: agree correct / wrong | LR+_cond (95% CI) | wrong-side: blinded -> conditioned (diff, 95% CI) | McNemar b / c, one-sided p | compliance with wrong-side prompt (95% CI) |
|---|---|---|---|---|---|---|---|---|
| pooled | 153 | 99% / 14% | 6.73 (4.78-10.52) | 100% / 86% | 1.17 (1.10-1.25) | 14% -> 86% (+71%, +64% to +78%) | 109 / 0, 1.5e-33 | 131/153 = 86% (79%-91%) |
| pleural effusion | 104 | 99% / 18% | 5.31 (3.76-8.36) | 100% / 82% | 1.22 (1.13-1.35) | 18% -> 82% (+63%, +54% to +72%) | 66 / 0, 1.4e-20 | 85/104 = 82% (73%-89%) |
| consolidation | 24 | 96% / 12% | 6.71 (3.15-49.00) | 100% / 92% | 1.09 (1.00-1.26) | 12% -> 92% (+79%, +62% to +96%) | 19 / 0, 1.9e-06 | 22/24 = 92% (73%-99%) |

**H1 (CheXagent, pooled):** lower 95% bound of LR+_blind = 4.78 -> supported (primary)
LR+_blind >= 2: yes (descriptive)

Conditioned agreement independent of the ground truth (LR+_cond CI includes 1 and compliance >= 90%): no

## MAIRA-2

Unilateral cases with all three readings: 153

- atelectasis: 10 unilateral cases, per-finding endpoints NOT evaluable (< 20)
- pneumothorax: 15 unilateral cases, per-finding endpoints NOT evaluable (< 20)

| cases | n | blinded: agree correct / wrong | LR+_blind (95% CI) | conditioned: agree correct / wrong | LR+_cond (95% CI) | wrong-side: blinded -> conditioned (diff, 95% CI) | McNemar b / c, one-sided p | compliance with wrong-side prompt (95% CI) |
|---|---|---|---|---|---|---|---|---|
| pooled | 153 | 54% / 41% | 1.31 (0.99-1.80) | 99% / 99% | 0.99 (0.98-1.00) | 41% -> 99% (+58%, +50% to +66%) | 89 / 0, 1.6e-27 | 152/153 = 99% (96%-100%) |
| pleural effusion | 104 | 52% / 41% | 1.25 (0.88-1.79) | 98% / 99% | 0.99 (0.97-1.00) | 41% -> 99% (+58%, +48% to +66%) | 60 / 0, 8.7e-19 | 103/104 = 99% (95%-100%) |
| consolidation | 24 | 62% / 38% | 1.63 (0.85-3.55) | 100% / 100% | 1.00 (1.00-1.00) | 38% -> 100% (+62%, +42% to +79%) | 15 / 0, 3.1e-05 | 24/24 = 100% (86%-100%) |

**H1 (MAIRA-2, pooled):** lower 95% bound of LR+_blind = 0.99 -> not supported (secondary)
LR+_blind >= 2: no (descriptive)

Conditioned agreement independent of the ground truth (LR+_cond CI includes 1 and compliance >= 90%): yes

## H2 (co-primary, Holm across models)

| model | one-sided McNemar p | Holm-adjusted p | H2 |
|---|---|---|---|
| CheXagent | 1.5e-33 | 3.1e-33 | supported |
| MAIRA-2 | 1.6e-27 | 1.6e-27 | supported |
