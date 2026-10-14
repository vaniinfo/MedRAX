# Rule: revising the generated INTERPRETATION with CheXagent's independent assessment

Date: 2026-10-13. Fixed before it is applied to the 19 comparisons. Uses no ground truth; the
ground truth is used only afterwards, to score the revised interpretation. Based on
`results/chexagent_reliability.md` (Indiana, NIH, VinDr).

## Inputs per comparison
- The generated report's sentences, already sorted under the four INTERPRETATION headings.
- CheXagent's independent screen of the current image ("Is there X?" for 13 conditions,
  `chexagent_independent_screen.py`), its decision per condition (FOUND / AGAINST / uncertain,
  as in `chexagent_reliability.py`) and, for FOUND, the side(s) of its side-free box.

## Rule
1. **Flag (recommend removal)**: a sentence that states a condition as present, where CheXagent's
   decision for that condition is AGAINST. Tag: "[Image check: not supported (score) - recommend
   removal]". Reason: AGAINST was right 96-100% in every dataset.
2. **Side correction**: a sentence that states a condition on one side (right or left), where
   CheXagent's decision is FOUND and its box is only on the opposite side. Tag: "[Image check:
   located on the <side> - confirm side]". Reason: claim-blinded CheXagent side evidence (VinDr
   Part B, LR+ 6.73).
3. **Add**: pleural effusion or cardiomegaly that the report does not state as present, where
   CheXagent's decision is FOUND. Added under LUNGS/PLEURA or HEART/MEDIASTINUM as "[Image check:
   <side> pleural effusion (score)]" / "[Image check: cardiomegaly (score)]". Only these two:
   the most reliable FOUND decisions across datasets.
4. Everything else is unchanged (other CheXagent findings stay "Consider" prompts in the table).

A sentence "states a condition as present" if it contains one of the condition's terms and no
negation (no, without, not, negative for) before it. Statements are tagged, never deleted, so
the reader sees what changed.

## Scoring (afterwards, against the same preliminary ground truth as the Excel comparison)
Per condition, the revised interpretation counts as stating a condition if the report stated it
and it was not flagged, or it was added; its side is CheXagent's where corrected or added.
Verdicts as in `report_vs_ground_truth.py` (Correct, Missed, Wrong side, Side partly matches, Not
in ground truth), for the original and the revised interpretation, all 19 comparisons.
