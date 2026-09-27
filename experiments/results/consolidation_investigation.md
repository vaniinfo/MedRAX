# Why Platt calibration worsened consolidation on the Indiana test set

Exploratory and descriptive: uses the existing Indiana readings; the frozen Indiana
parameters (`frozen_dev_params.json`) were neither changed nor used for any decision.
Recorded 2026-09-27, before the NIH external-validation results existed.

## Observation
Consolidation: test AUC 0.89, yet Platt calibration fitted on the development split
worsened the test Brier score (0.103 → 0.110) and ECE (0.035 → 0.074); calibration improved
for the other five findings.

## Findings

| Hypothesis | Evidence | Verdict |
|---|---|---|
| Label mixes two concepts | Positives carrying the MeSH term "Consolidation" (n=25): median p 0.76, 72% ≥ 0.5. Positives with "Airspace Disease" only (n=84): median p 0.33, 39% ≥ 0.5. Both separate well from normal studies (AUC 0.986 / 0.961). | Confirmed |
| Negatives include look-alikes | Consolidation-negative studies with "Infiltrate" (n=19): median p 0.49, 47% ≥ 0.5; "Pleural Effusion" (n=89): 0.33; "Opacity" (n=106): 0.26; normal studies (n=150): 0.01. | Confirmed |
| Wrong calibration shape | 5-fold CV within development only: Brier raw 0.146, Platt 0.136, isotonic 0.135. | Rejected |
| Development and test differ | Consolidation AUC dev 0.756 vs test 0.890; the other five findings differ by ≤ 0.05. | Confirmed |

## Interpretation
cxr-edv's truth mapping counts "Consolidation" and "Airspace Disease" as consolidation
(24 of the 100 studies selected for consolidation carry "Consolidation", 93 carry "Airspace
Disease") but not the near-synonym "Infiltrate". CheXagent's answer to "Is there
consolidation?" tracks the narrow concept, so the middle of the score range is a mixture of
label-ambiguous studies. The development split drew a harder mixture (AUC 0.76) than the
test split (0.89); Platt fitted on it learned a flat slope (0.51, about half of the other
findings'), which compressed the test set's genuinely high scores (0.7–0.9 band: 89% observed
positive vs 51% calibrated). The failure is label ambiguity plus case-mix shift, not the
model or the calibration method.

## Prediction for the NIH external validation (recorded before the results)
NIH's consolidation label is NLP-mined with different wording. If this explanation holds,
consolidation should transfer worst from Indiana to NIH (experiment B) and gain most from
NIH recalibration (experiment C).
