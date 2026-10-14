# Generated interpretation revised with CheXagent: original vs revised, against the ground truth

Rule: `chexagent_revision_rule.md` (fixed before applying; no ground truth used). Ground truth: preliminary automatic labels (as in the Excel comparison). 19 comparisons, 13 conditions.

| verdict | original (MedGemma) | revised (MedGemma + CheXagent rule) |
|---|---|---|
| Correct | 8 | 11 |
| Missed | 20 | 16 |
| Wrong side | 3 | 2 |
| Side partly matches | 1 | 3 |
| Not in ground truth | 13 | 14 |
| **Incorrect (missed or wrong side)** | **23** | **18** |

## What the rule changed (page, condition, before -> after, ground truth)

| page | condition | original | verdict | revised | verdict | ground truth |
|---|---|---|---|---|---|---|
| 5 | cardiomegaly | Not stated | Missed | Present (bilateral) | Correct | Present, location not stated |
| 8 | pneumothorax | Present (right) | Wrong side | Present (left) | Correct | Left mediastinum (possible) |
| 9 | cardiomegaly | Not stated | Missed | Present (bilateral) | Correct | Present, location not stated |
| 10 | cardiomegaly | Not stated | Missed | Present (bilateral) | Correct | Present, location not stated |
| 11 | pleural effusion | Present (bilateral) | Correct | Present (right) | Side partly matches | Right; Left |
| 12 | pleural effusion | Present (bilateral) | Correct | Present (right) | Side partly matches | Right; Left |
| 13 | cardiomegaly | Not stated | - | Present (bilateral) | Not in ground truth | NOT IN GROUND TRUTH |
| 14 | cardiomegaly | Not stated | Missed | Present (bilateral) | Correct | Present, location not stated |
