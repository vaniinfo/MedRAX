# Generated interpretation revised with CheXagent: original vs revised, against the ground truth

Rule: `chexagent_revision_rule.md` (fixed before applying; no ground truth used). Ground truth: preliminary automatic labels (as in the Excel comparison). 16 comparisons, 13 conditions.

| verdict | original (MedGemma) | revised (MedGemma + CheXagent rule) |
|---|---|---|
| Correct | 7 | 11 |
| Missed | 9 | 6 |
| Wrong side | 2 | 1 |
| Side partly matches | 0 | 0 |
| Not in ground truth | 14 | 14 |
| **Incorrect (missed or wrong side)** | **11** | **7** |

## What the rule changed (page, condition, before -> after, ground truth)

| page | condition | original | verdict | revised | verdict | ground truth |
|---|---|---|---|---|---|---|
| 1 | cardiomegaly | Not stated | Missed | Present (bilateral) | Correct | Present, location not stated |
| 3 | pneumothorax | Present (right) | Wrong side | Present (left) | Correct | Left mediastinum (possible) |
| 4 | cardiomegaly | Not stated | Missed | Present (bilateral) | Correct | Present, location not stated |
| 14 | cardiomegaly | Not stated | Missed | Present (bilateral) | Correct | Present, location not stated |
