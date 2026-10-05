# Final external validation of Evidence Engine v1.0 on VinDr-CXR

Protocol: `experiments/vindr_external_validation_design.md` (committed before the data was on this machine). Engine: tag `evidence-engine-v1.0`, loaded unchanged with its checksums verified. VinDr-CXR test split: 3000 images (0 excluded as undecodable), 5-radiologist consensus labels. CIs: 2000-sample image-level bootstrap, seed 20260927.

## Findings: positives and evaluability

| Finding | Positives / negatives | Evaluable (≥ 20 positives) | p_present AUC [95% CI] (descriptive) |
|---|---|---|---|
| cardiomegaly | 358 / 2642 | yes | 0.983 [0.979–0.987] |
| pleural effusion | 120 / 2880 | yes | 0.995 [0.992–0.997] |
| pneumothorax | 15 / 2985 | no | 0.999 [0.998–1.000] |
| consolidation | 25 / 2975 | yes | 0.991 [0.986–0.995] |
| pulmonary edema | 0 / 3000 | no | – |
| atelectasis | 10 / 2990 | no | 0.986 [0.974–0.996] |

## Primary endpoints: rule primary 5 / 0.2

Empirical LR = P(state | finding) / P(state | no finding) [95% CI]. SUPPORTING: ✓ point ≥ criterion, ✓✓ lower CI ≥ criterion. REFUTING: ✓ point ≤ criterion, ✓✓ upper CI ≤ criterion. ✗ = not met. Coverage = share of all images in the state.

| Finding | Engine | SUPPORTING: coverage, LR | UNCERTAIN coverage | REFUTING: coverage, LR | Positives assigned to REFUTING | INSUFFICIENT coverage |
|---|---|---|---|---|---|---|
| cardiomegaly | NIH (primary) | 12.8%, 18.3 [15.3–22.5] ✓✓ | 38.1% | 47.8%, 0.00 [0.00–0.00] ✓✓ | 0 / 358 (0%) | 1.4% |
| cardiomegaly | Indiana (comparator) | 8.5%, 29.4 [22.4–40.8] ✓✓ | 38.9% | 50.5%, 0.00 [0.00–0.00] ✓✓ | 0 / 358 (0%) | 2.1% |
| pleural effusion | NIH (primary) | 3.3%, 94.8 [63.2–158.4] ✓✓ | 22.0% | 74.1%, 0.00 [0.00–0.00] ✓✓ | 0 / 120 (0%) | 0.6% |
| pleural effusion | Indiana (comparator) | 2.3%, 101.5 [60.1–200.9] ✓✓ | 22.5% | 53.3%, 0.00 [0.00–0.00] ✓✓ | 0 / 120 (0%) | 21.9% |
| pneumothorax | – | not evaluable (15 positives) | | | | |
| consolidation | NIH (primary) | 1.0%, 36.2 [14.2–74.0] ✓✓ | 98.3% | 0.0%, not observed | 0 / 25 (0%) | 0.7% |
| consolidation | Indiana (comparator) | 0.0%, not observed | 85.1% | 0.0%, not observed | 0 / 25 (0%) | 14.9% |
| pulmonary edema | – | not evaluable (0 positives) | | | | |
| atelectasis | – | not evaluable (10 positives) | | | | |

### UNCERTAIN / INSUFFICIENT DATA by reason (primary rule)

| Finding | Engine | WEAK_EVIDENCE | REFUTING_NOT_EMPIRICALLY_CONFIRMED | OUTSIDE_SUPPORT | FEW_DEV_POSITIVES | FINDING_NOT_VALIDATED |
|---|---|---|---|---|---|---|
| cardiomegaly | NIH (primary) | 38.1% | 0.0% | 1.4% | 0.0% | 0.0% |
| cardiomegaly | Indiana (comparator) | 29.3% | 9.6% | 2.1% | 0.0% | 0.0% |
| pleural effusion | NIH (primary) | 18.5% | 3.5% | 0.6% | 0.0% | 0.0% |
| pleural effusion | Indiana (comparator) | 9.4% | 13.1% | 21.9% | 0.0% | 0.0% |
| consolidation | NIH (primary) | 84.6% | 13.6% | 0.7% | 0.0% | 0.0% |
| consolidation | Indiana (comparator) | 64.7% | 20.4% | 14.9% | 0.0% | 0.0% |

## Descriptive: rule sensitivity 2 / 0.5

Empirical LR = P(state | finding) / P(state | no finding) [95% CI]. SUPPORTING: ✓ point ≥ criterion, ✓✓ lower CI ≥ criterion. REFUTING: ✓ point ≤ criterion, ✓✓ upper CI ≤ criterion. ✗ = not met. Coverage = share of all images in the state.

| Finding | Engine | SUPPORTING: coverage, LR | UNCERTAIN coverage | REFUTING: coverage, LR | Positives assigned to REFUTING | INSUFFICIENT coverage |
|---|---|---|---|---|---|---|
| cardiomegaly | NIH (primary) | 20.0%, 7.89 [7.06–8.95] ✓✓ | 28.1% | 50.5%, 0.00 [0.00–0.00] ✓✓ | 0 / 358 (0%) | 1.4% |
| cardiomegaly | Indiana (comparator) | 13.9%, 13.3 [11.3–15.9] ✓✓ | 33.5% | 50.5%, 0.00 [0.00–0.00] ✓✓ | 0 / 358 (0%) | 2.1% |
| pleural effusion | NIH (primary) | 5.3%, 39.2 [30.9–51.9] ✓✓ | 20.0% | 74.1%, 0.00 [0.00–0.00] ✓✓ | 0 / 120 (0%) | 0.6% |
| pleural effusion | Indiana (comparator) | 3.3%, 67.4 [46.5–106.3] ✓✓ | 21.5% | 53.3%, 0.00 [0.00–0.00] ✓✓ | 0 / 120 (0%) | 21.9% |
| pneumothorax | – | not evaluable (15 positives) | | | | |
| consolidation | NIH (primary) | 9.3%, 6.29 [4.03–8.59] ✓✓ | 42.5% | 47.5%, 0.00 [0.00–0.00] ✓✓ | 0 / 25 (0%) | 0.7% |
| consolidation | Indiana (comparator) | 4.2%, 10.3 [5.20–15.8] ✓✓ | 32.5% | 48.5%, 0.00 [0.00–0.00] ✓✓ | 0 / 25 (0%) | 14.9% |
| pulmonary edema | – | not evaluable (0 positives) | | | | |
| atelectasis | – | not evaluable (10 positives) | | | | |

## Descriptive: rule sensitivity 10 / 0.1

Empirical LR = P(state | finding) / P(state | no finding) [95% CI]. SUPPORTING: ✓ point ≥ criterion, ✓✓ lower CI ≥ criterion. REFUTING: ✓ point ≤ criterion, ✓✓ upper CI ≤ criterion. ✗ = not met. Coverage = share of all images in the state.

| Finding | Engine | SUPPORTING: coverage, LR | UNCERTAIN coverage | REFUTING: coverage, LR | Positives assigned to REFUTING | INSUFFICIENT coverage |
|---|---|---|---|---|---|---|
| cardiomegaly | NIH (primary) | 7.3%, 54.8 [39.0–85.2] ✓✓ | 52.5% | 38.8%, 0.00 [0.00–0.00] ✓✓ | 0 / 358 (0%) | 1.4% |
| cardiomegaly | Indiana (comparator) | 4.4%, 73.8 [45.4–150.4] ✓✓ | 43.0% | 50.5%, 0.00 [0.00–0.00] ✓✓ | 0 / 358 (0%) | 2.1% |
| pleural effusion | NIH (primary) | 2.3%, 212.6 [113.3–540.1] ✓✓ | 97.1% | 0.0%, not observed | 0 / 120 (0%) | 0.6% |
| pleural effusion | Indiana (comparator) | 1.5%, 130.3 [67.9–338.6] ✓✓ | 76.6% | 0.0%, not observed | 0 / 120 (0%) | 21.9% |
| pneumothorax | – | not evaluable (15 positives) | | | | |
| consolidation | NIH (primary) | 0.0%, not observed | 99.3% | 0.0%, not observed | 0 / 25 (0%) | 0.7% |
| consolidation | Indiana (comparator) | 0.0%, not observed | 85.1% | 0.0%, not observed | 0 / 25 (0%) | 14.9% |
| pulmonary edema | – | not evaluable (0 positives) | | | | |
| atelectasis | – | not evaluable (10 positives) | | | | |

## Limitation (stated in advance)

VinDr-CXR played no role in the engine's development, but CheXagent may have been trained on it; untouched by the engine is not the same as unseen by CheXagent.
