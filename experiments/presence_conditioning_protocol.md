# Does a report's assertion change a checker's presence evidence? Protocol (DRAFT, not yet committed)

Date: 2026-10-25. To be committed before any reading below is generated. Covers plan items A1
(presence conditioning) and A5 (verification test with known truth) of
`paper_strengthening_plan.md`.

## Why
The laterality experiment (VinDr Amendment 1, E1) showed that naming a side in the localisation prompt
largely determines the side checkers return. Report-blinding matters most for presence: whether a
stated condition is there at all. This experiment measures directly whether CheXagent's presence
evidence moves towards what a report asserts, and whether blinded evidence verifies statements better
than assertion-given evidence.

## Data (labels as in all earlier analyses)
- **Primary:** the NIH ChestX-ray14 test split (`split_nih_70_30.json`, 1,500 images, one per
  patient), 13 conditions.
- **Secondary:**
  - Indiana (544 images, 6 conditions);
  - VinDr-CXR Kaggle-derived sample (3,000 images; the conditions it labels, without pulmonary edema).
- No new images, labels or splits.

## Checker and prompts
- **Checker:** CheXagent-2-3b (snapshot 8f19b53a, bf16, greedy, 8 new tokens), the existing
  `XRayVQATool._generate_response` path. The score is p_present = P(Yes) / (P(Yes) + P(No)) of the
  first token, as before.
- **Arms**, for every image and condition, in a fixed order per image:
  - **N (neutral, blinded):** "Is there {c}?" This re-runs the stored readings; agreement with them
    is reported as a determinism check.
  - **C (context control):** "A radiology report has been written for this image. Is there {c}?"
    This adds text without an assertion.
  - **P (positive assertion):** "The radiology report states that there is {c}. Is there {c}?"
  - **A (negative assertion):** "The radiology report states that there is no {c}. Is there {c}?"
- {c} is the same condition string as in all earlier analyses. Answers not starting with Yes or No
  have no score; their share is reported per arm.
- **Secondary checker,** only if it is available locally before the run starts: MedGemma 1.5 4B
  (google/medgemma-1.5-4b-it, bf16), same four arms and the same scoring. If it is added later, this
  is done only by a dated amendment before its readings are generated.

## Statements with known truth (verification test)
Every image-condition pair gives two statements:
- S+ "{c} is present", true if the label is 1;
- S- "{c} is absent", true if the label is 0.

Evidence for a statement:
- **blinded:** p_present from arm N for S+, and 1 - p_present from arm N for S-;
- **assertion-given:** p_present from arm P for S+, and 1 - p_present from arm A for S-.

A statement is accepted if its evidence is at least 0.5.

## Hypotheses (primary dataset NIH test; two-sided; 2,000 image-level bootstrap resamples, seed 20261029)
**H-P1 (conditioning, co-primary).** The share of false statements accepted is higher with the
assertion given than blinded, pooled over the 13 conditions.
- Test: exact McNemar on image-condition pairs, separately for S+ and S-.
- Supported if both are significant after Holm across the two.

**H-P2 (verification, co-primary).** Blinded evidence separates true from false statements better
than assertion-given evidence.
- Measure: AUC of evidence for true vs false statements per condition; the mean over conditions with
  at least 10 positives; paired difference blinded minus assertion-given.
- Supported if the 95% CI lies above 0.

**Holm across H-P1 and H-P2** as a family of two.

**Reported for all arms and datasets:**
- mean shift in logit(p_present) from N to C, P and A;
- presence AUC per arm;
- sensitivity and specificity for accepting true and rejecting false statements;
- likelihood ratios of "accepted" (LR+ = acceptance of true / acceptance of false);
- the same with the frozen Evidence Engine v1.0 decision (SUPPORTING / REFUTING) applied to each arm's
  score, to show what an engine calibrated on blinded scores would conclude from conditioned ones.

**Arm C** separates the effect of extra text from the effect of the assertion: P and A are each
compared with C as well as with N.

## Secondary analysis of existing data (exploratory, labelled as such)
Using E1 outputs (153 one-sided VinDr findings, both checkers), side statements "the finding is on
side s" for the true and the opposite side. Accept if a box lies on s.
- Report sensitivity, specificity and LR for detecting wrong-side statements, side-free vs side-given.
- This re-expresses the pre-registered E1 data in clinical terms; no new readings.

## Limitations recorded now
- Prompt wording is one choice. Other phrasings, and assertions embedded in a full report, may
  differ.
- NIH labels are text-mined.
- CheXagent's development data include NIH and VinDr.
- Indiana was examined in our earlier development work (manuscript Section 3.2).
- Acceptance at 0.5 is a simple rule; the engine decisions are reported alongside.

All results are reported whatever they are; changes only as dated amendments before the affected run.

Implementation: `presence_conditioning.py` (stages `read`, `analyze`), to be written after this
protocol is committed and before any reading.
