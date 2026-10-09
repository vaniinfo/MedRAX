# Provisional cut-offs for findings beyond the six: NIH random sample

Design: `provisional_findings_design.md`. Fitted on the development split, checked on the test split. Not part of Evidence Engine v1.0. Empirical LR = P(state | finding) / P(state | no finding), 95% CI from 2000 image-level bootstrap resamples (seed 20261009).

| finding | dev pos / n | test pos / n | test AUC | SUPPORTING: from p, coverage, LR [CI] | REFUTING: up to p, coverage, LR [CI], positives in it | used in review |
|---|---|---|---|---|---|---|
| pneumonia | 28 / 3500 | 9 / 1500 | 0.734 | 0.839; 0.4%, 0.00 [0.00-0.00] | 0.0002; not observed; 0 | SUPPORTING no, REFUTING no |
| infiltration | 430 / 3500 | 185 / 1500 | 0.600 | 1.000; not observed | none; not observed; 0 | SUPPORTING no, REFUTING no |
| emphysema | 44 / 3500 | 14 / 1500 | 0.921 | 0.784; 1.7%, 33.52 [13.91-72.88] | 0.0718; not observed; 0 | SUPPORTING yes, REFUTING no |
| nodule | 155 / 3500 | 85 / 1500 | 0.802 | 0.868; 5.3%, 6.43 [3.92-9.98] | 0.0595; 12.1%, 0.09 [0.00-0.31]; 1 | SUPPORTING yes, REFUTING yes |
| mass | 123 / 3500 | 67 / 1500 | 0.935 | 0.871; 4.9%, 19.19 [13.23-28.64] | 0.0752; not observed; 0 | SUPPORTING yes, REFUTING no |
| fibrosis | 56 / 3500 | 24 / 1500 | 0.823 | 0.822; 2.3%, 5.95 [0.00-13.70] | 0.0232; not observed; 0 | SUPPORTING yes, REFUTING no |
| pleural thickening | 80 / 3500 | 30 / 1500 | 0.916 | 0.822; 2.4%, 16.33 [6.95-30.54] | 0.0470; not observed; 0 | SUPPORTING yes, REFUTING no |
