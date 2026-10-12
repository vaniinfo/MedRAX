# Pilot model comparison (PRELIMINARY)

**PRELIMINARY: scored on unreviewed automatic labels (ground truth and model reports). Not the pilot's final results.**

Protocol: `experiments/model_comparison_pilot_protocol.md` (amendments A1-A4). 19 comparisons, 13 conditions; runs `pilot_runs_20261011-175859`. Counts only. Intervals: 95%, Wilson for proportions, page-level bootstrap (2000, seed 20261011) for differences and AUC. Unconfirmed = stated, but the ground-truth report does not report it (not proof of a false positive). 19 images: read intervals, not point estimates; no ranking claims.

## Primary: all 19 pages, "possible" counted as present

Ground-truth conditions: 32 (of 247 page-conditions).

### Experiment 1: each model's own report

| model | caught (95% CI) | difference vs Team MedGemma (95% CI) | unconfirmed | same side |
|---|---|---|---|---|
| Team MedGemma | 12/32 = 38% [23%-55%] | (reference) | 11 | 8/12 |
| MedGemma 1.5 | 7/32 = 22% [11%-39%] | -16% [-31% to +0%] | 5 | 1/6 |
| CheXagent | 13/32 = 41% [26%-58%] | +3% [+0% to +12%] | 9 | 3/9 |
| MAIRA-2 | 10/32 = 31% [18%-49%] | -6% [-23% to +8%] | 13 | 4/8 |

Per condition (caught / ground-truth count; counts only):

| condition | ground truth | Team MedGemma | MedGemma 1.5 | CheXagent | MAIRA-2 | CheXagent asked |
|---|---|---|---|---|---|---|
| cardiomegaly | 7 | 0/7 (+2 unconf.) | 1/7 (+1 unconf.) | 2/7 (+3 unconf.) | 1/7 (+2 unconf.) | 4/7 |
| pleural effusion | 6 | 6/6 (+4 unconf.) | 5/6 (+0 unconf.) | 6/6 (+4 unconf.) | 4/6 (+1 unconf.) | 5/6 |
| pneumothorax | 1 | 1/1 (+2 unconf.) | 0/1 (+0 unconf.) | 1/1 (+1 unconf.) | 0/1 (+0 unconf.) | 1/1 |
| consolidation | 0 | 0/0 (+0 unconf.) | 0/0 (+1 unconf.) | 0/0 (+0 unconf.) | 0/0 (+1 unconf.) | 0/0 |
| pulmonary edema | 0 | 0/0 (+2 unconf.) | 0/0 (+2 unconf.) | 0/0 (+0 unconf.) | 0/0 (+2 unconf.) | 0/0 |
| atelectasis | 8 | 4/8 (+1 unconf.) | 1/8 (+1 unconf.) | 4/8 (+0 unconf.) | 5/8 (+4 unconf.) | 3/8 |
| pneumonia | 1 | 1/1 (+0 unconf.) | 0/1 (+0 unconf.) | 0/1 (+0 unconf.) | 0/1 (+2 unconf.) | 0/1 |
| emphysema | 4 | 0/4 (+0 unconf.) | 0/4 (+0 unconf.) | 0/4 (+1 unconf.) | 0/4 (+0 unconf.) | 3/4 |
| nodule | 2 | 0/2 (+0 unconf.) | 0/2 (+0 unconf.) | 0/2 (+0 unconf.) | 0/2 (+0 unconf.) | 1/2 |
| fibrosis | 1 | 0/1 (+0 unconf.) | 0/1 (+0 unconf.) | 0/1 (+0 unconf.) | 0/1 (+0 unconf.) | 0/1 |
| pleural thickening | 2 | 0/2 (+0 unconf.) | 0/2 (+0 unconf.) | 0/2 (+0 unconf.) | 0/2 (+1 unconf.) | 1/2 |

### Experiment 2: direct question per condition

| model | found (95% CI) | unconfirmed | AUC (95% CI) | no score |
|---|---|---|---|---|
| CheXagent (cut-offs) | 18/32 = 56% [39%-72%] | 16 | 0.87 [0.79-0.94] | 0 |
| MedGemma 1.5 (answer Yes) | 5/32 = 16% [7%-32%] | 2 | 0.78 [0.70-0.86] | 7 |

### Experiment 3: location without the side (12 one-sided ground-truth findings)

| model | correct side (95% CI) | wrong side | both sides | not located | no answer (cut off) |
|---|---|---|---|---|---|
| CheXagent | 6/12 = 50% [25%-75%] | 2 | 4 | 0 | - |
| MAIRA-2 | 4/12 = 33% [14%-61%] | 5 | 0 | 3 | - |
| MedGemma 1.5 | 2/12 = 17% [5%-45%] | 1 | 1 | 7 | 1 |

## Sensitivity: excluding pages 9-12

Ground-truth conditions: 16 (of 195 page-conditions).

### Experiment 1: each model's own report

| model | caught (95% CI) | difference vs Team MedGemma (95% CI) | unconfirmed | same side |
|---|---|---|---|---|
| Team MedGemma | 8/16 = 50% [28%-72%] | (reference) | 8 | 4/8 |
| MedGemma 1.5 | 3/16 = 19% [7%-43%] | -31% [-50% to -15%] | 3 | 1/3 |
| CheXagent | 9/16 = 56% [33%-77%] | +6% [+0% to +21%] | 8 | 3/5 |
| MAIRA-2 | 6/16 = 38% [18%-61%] | -12% [-27% to +0%] | 11 | 2/4 |

## Sensitivity: "possible" counted as not reported

Ground-truth conditions: 23 (of 247 page-conditions).

### Experiment 1: each model's own report

| model | caught (95% CI) | difference vs Team MedGemma (95% CI) | unconfirmed | same side |
|---|---|---|---|---|
| Team MedGemma | 6/23 = 26% [13%-46%] | (reference) | 17 | 5/6 |
| MedGemma 1.5 | 5/23 = 22% [10%-42%] | -4% [-19% to +12%] | 7 | 1/4 |
| CheXagent | 8/23 = 35% [19%-55%] | +9% [+0% to +28%] | 14 | 0/6 |
| MAIRA-2 | 7/23 = 30% [16%-51%] | +4% [-19% to +20%] | 16 | 3/6 |

## Sensitivity: excluding pages 9-12 and "possible" as not reported

Ground-truth conditions: 9 (of 195 page-conditions).

### Experiment 1: each model's own report

| model | caught (95% CI) | difference vs Team MedGemma (95% CI) | unconfirmed | same side |
|---|---|---|---|---|
| Team MedGemma | 2/9 = 22% [6%-55%] | (reference) | 14 | 1/2 |
| MedGemma 1.5 | 1/9 = 11% [2%-44%] | -11% [-38% to +0%] | 5 | 1/1 |
| CheXagent | 4/9 = 44% [19%-73%] | +22% [+0% to +57%] | 13 | 0/2 |
| MAIRA-2 | 3/9 = 33% [12%-65%] | +11% [+0% to +38%] | 14 | 1/2 |
