# Final external validation of Evidence Engine v1.0 on VinDr-CXR (pre-registered)

Frozen and committed before the VinDr-CXR data was on this machine. This is the single,
final experiment for `evidence-engine-v1.0` (tag at commit `c30c5d7`).

## Question
Does the completely frozen Evidence Engine v1.0 retain its predefined evidence meaning on a
dataset that played no role in its development?

## Hands-off rules
- No threshold, calibration or empirical-band fitting; no state redesign; no
  finding-specific exceptions. The engine is loaded from the tagged, checksum-verified
  bundle (`medrax/evidence`) and used unchanged.
- Run once. Whatever the result, it is reported as the result. No v1.1, no further dataset
  for this version; failures go to Limitations/Discussion.
- Both frozen engines are reported: NIH-derived (primary) and Indiana-derived
  (comparator). Neither is selected on VinDr performance.

## Data
- VinDr-CXR v1.0.0 (PhysioNet), **test split** (3,000 frontal images), image-level labels
  from the consensus of 5 radiologists. All test images are used; no sampling.
- Label mapping: Cardiomegaly → cardiomegaly; Pleural effusion → pleural effusion;
  Pneumothorax → pneumothorax; Consolidation → consolidation; Edema → pulmonary edema;
  Atelectasis → atelectasis. A finding is positive when its consensus label is 1.
- **Evaluability**: a finding is evaluated only if the test split has ≥ 20 positives for it;
  otherwise it is reported as not evaluable, with its counts.
- **Image conversion** (fixed; no per-image adjustment): DICOM pixel data → DICOM VOI LUT /
  windowing (`pydicom` `apply_voi_lut`) → inverted if PhotometricInterpretation is
  MONOCHROME1 → min–max scaled to 8-bit → PNG. Images that cannot be decoded are listed and
  excluded; their number is reported.
- CheXagent query and scoring exactly as in all previous experiments: `Is there {finding}?`,
  p_present = Yes/(Yes+No) over the first answer token, greedy, bfloat16, same snapshot.

## Primary endpoints (rule 5 / 0.2), per evaluable finding and per engine
1. **SUPPORTING**: empirical LR = P(SUPPORTING | finding) / P(SUPPORTING | no finding)
   with 95% CI; criterion met if the point estimate ≥ 5, demonstrated if the lower CI ≥ 5.
2. **REFUTING**: empirical LR with 95% CI; met if ≤ 0.2, demonstrated if the upper CI ≤ 0.2.
3. **Coverage**: share of cases in each state (SUPPORTING, UNCERTAIN, REFUTING,
   INSUFFICIENT DATA), with UNCERTAIN and INSUFFICIENT DATA broken down by reason.
4. **Positive cases assigned to REFUTING**: count and share of the finding's positives.

CIs: 2,000-sample image-level bootstrap of the test split, seed 20260927.

## Descriptive only (not endpoints)
AUC of p_present per finding, and state coverage under the stored sensitivity rules
2 / 0.5 and 10 / 0.1.

## Interpretation, fixed in advance
- A state "retains its meaning" on VinDr for a finding if its criterion is met; the claim is
  strongest where it is demonstrated. Results are reported per finding and engine; there is
  no pooled pass/fail.
- A state with no cases on VinDr is reported as not observed, not as met.

## Known limitation to report regardless of result
VinDr-CXR played no role in the engine's development, but CheXagent may have been trained on
it (as on other public chest X-ray datasets). "Untouched by the engine" is not the same as
"unseen by CheXagent".
