# Radiologist evaluation of generated findings, change descriptions and CheXagent evidence: analysis plan

Date: 2026-11-03. DRAFT, to be committed before any score from the evaluation forms below is read.

**Disclosure.** One filled form has already been read: doctor 1's form A (`mixed_scoring_form_A_v9.pdf`, 20 cases:
10 older MIMIC-CXR and 10 NIH pairs; 65 rows), returned 2026-11-03. Its scores informed this plan (the categories,
the auditing table and the "cannot assess" problem). Form A is therefore a **pilot**: it is reported separately,
descriptively, and is not part of the primary analysis.

## Evaluation set (primary)
100 prior-current pairs, each read by one of two radiologists (no pair read by both):
- MIMIC-CXR held-out sets 1-3: `mimic_heldout_scoring_form.pdf`, `mimic_heldout_2_scoring_form.pdf`,
  `mimic_heldout_3_scoring_form.pdf` (60 pairs; never used to train the fine-tuned MedGemma);
- CheXpert Plus: `chexpert_plus_scoring_form.pdf` (20 pairs);
- NIH ChestX-ray14: 20 pairs [form to be fixed before sending: the NIH pages of forms A and B, or a separate
  NIH-only form built from them].

Which radiologist reads which form is recorded before the forms are sent. The forms are not changed after they are
sent; if a form must be rebuilt (for example, to fix the reference reader), the rebuilt version is the one sent,
and the change is logged here as a dated amendment before any score from it is read.

## Unit, row types and score categories
- **Unit:** one table row (a finding) within a case. Analyses that pool rows use case-level bootstrap intervals
  (2,000 resamples of cases, seed 20261103), because rows of one case are not independent.
- **Row types** (fixed by the form): *stated* (MedGemma reported the finding); *CheXagent-only* (CheXagent
  supported a condition MedGemma did not report); *missed* (the reference report or labels contain a condition
  neither model reported).
- **Score categories:** 4-5 = **correct**; 1-2 = **incorrect**; 3 = **uncertain**. Blank = **missing**,
  excluded from that dimension (counts reported). A score of 3 or a blank whose comment states the region cannot
  be assessed (e.g. "collimated", "not seen", "cannot assess") is coded **not assessable** and reported as such.
- **Dimensions:** MedGemma finding (all row types), CheXagent evidence (all row types), MedGemma change (stated
  rows), CheXagent grounding (stated rows with a box).
- The two-level CheXagent verdict shown (Agree / Not agree) is analysed as shown; the underlying engine state
  (supported / indeterminate / evidence against) is used only in the secondary analysis.

## Primary analyses (one per research question, descriptive with 95% CIs)
1. **MedGemma finding accuracy (RQ1).** Proportion of stated findings rated correct, overall and by dataset.
2. **MedGemma change accuracy (RQ1).** Proportion of stated findings whose change description is rated correct,
   separately for findings with a change stated (new, unchanged, worse, better, resolved) and "no change stated".
3. **Report auditing (RQ4b; the main comparison).** Among stated findings rated correct or incorrect (uncertain,
   missing and not-assessable excluded): the 2 x 2 table of the radiologist's verdict on MedGemma's finding by
   CheXagent's flag (Not agree = flagged). Reported: sensitivity of the flag for incorrect findings, specificity,
   positive and negative predictive values, and **the difference in flag rate between incorrect and correct
   findings** (risk difference, case-level bootstrap 95% CI). The flag is considered informative if this CI lies
   above 0. This is the only comparison treated as confirmatory.
4. **CheXagent evidence and grounding (RQ3).** Proportion of CheXagent verdicts rated correct on stated rows;
   proportion of boxes rated correct among stated rows with a box.
5. **Additions and misses.** CheXagent-only rows: proportion rated correct by the radiologist (the precision of
   CheXagent's added findings). Missed rows: proportion where the radiologist rated "not detected" correct, i.e.
   disputed the reference, by dataset (NIH labels vs radiologist reports).

## Secondary and exploratory (labelled as such)
- All analyses by dataset (MIMIC-CXR held-out, CheXpert Plus, NIH) and by radiologist.
- Report auditing split by engine state: indeterminate vs evidence against.
- Score distributions (1-5) per dimension.
- Missed rows created by the reference reader in error (for example a condition described as resolved, or only
  seen on CT): counted and listed; such rows are **not** removed after the fact.
- Pilot (form A): the same tables, reported separately.

## Agreement between radiologists
None is estimable in the primary set (each pair has one reader). If time allows, 20 pairs (one form) are read by
both radiologists, chosen at random before reading; weighted kappa per dimension is then reported. Decided and
logged here before any shared form is sent.

## Limitations recorded now
Two readers, no double reading; selected (enriched) cases, not a random sample; readers see the reference report
and the models' outputs (not blinded to them); the five-point scale is used as an accuracy rating, with 3 as an
uncertain middle; CheXagent may have seen MIMIC-CXR, CheXpert and NIH images in training; MedGemma's NIH reports
were generated with placeholder prior reports.

All results are reported whatever they are; changes only as dated amendments before the affected scores are read.

Implementation: `read_radiologist_scores.py` (fields to CSV) and an analysis script written before the first
primary form is returned.

## Amendments
(none)
