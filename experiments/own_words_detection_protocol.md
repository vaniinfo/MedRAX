# CheXagent vs MAIRA-2 in their own words: caught vs missed. Protocol

Date: 2026-10-23. Written and committed before any report below is generated or read.

## Why
The AUC comparison (`maira_score_design.md`, A1 + A2; `results/maira_vs_chexagent_all_conditions.md`)
scored MAIRA-2 with a sentence-likelihood workaround, because it gives no presence probability. An
exploratory check run after those results (`results/maira_score_validity.md`, pleural effusion, 200
NIH development images) found that the workaround understates MAIRA-2. MAIRA-2's own reports stated
effusion in 65 of 100 labelled effusions and 4 of 100 without, close to CheXagent's 68 and 4 at its
cut-off. This experiment compares the models by what each one says, without any score.

## Data (first stage: NIH only)
NIH ChestX-ray14 test split of the AUC comparison: 1,500 images (`split_nih_70_30.json`), 13
conditions, NIH labels (NLP-mined). Twelve conditions have at least 10 positives; pneumonia (9)
is reported but not in any mean. Indiana (544) and VinDr-CXR (3,000) may follow later, each only
through a dated amendment committed before its run.

## What each model says
- **MAIRA-2, own report.** bf16, revision `795a2b1c…` as before; current frontal image only; its
  standard findings prompt (no indication, technique, comparison or prior); greedy; 300 new tokens.
- **CheXagent, own report (C-report).** The six area prompts of `chexagent_report_screen.py`
  (`Please provide a detailed description of "{area}" in the chest X-ray`; View, Airway, Breathing,
  Cardiac, Diaphragm, Everything else); 200 tokens each, greedy; the six answers are joined.
- **CheXagent, plain answer (C-answer).** "Is there {condition}?" on the current image, from the
  existing readings (`ppresent_nih.jsonl` for the six, `provisional_readings.jsonl` for the seven).
  Present if p_present >= 0.5, i.e. "Yes" outweighs "No". This is not the calibrated Evidence Engine
  cut-off, which was tuned on NIH and would favour CheXagent. No new runs are needed.

## Reading the reports (same reader for both models)
Each report is read for the 13 conditions with the quote-checked lookup
(`comparison_edv_pdf.ground_truth_locations`, local Qwen `qwen3.8-edv:latest`; an answer whose
quoted evidence is not in the report counts as not stated), with the amendment A1 rule of
`edv_improvement_protocol.md` (subcutaneous emphysema is not emphysema).

- **Primary:** a condition is stated if the reader answers present.
- **Sensitivity analysis:** "possible" also counts as stated.

The reader sees only the report text, never the image or the labels.

## Measures (per model and condition)
- Sensitivity: caught / labelled present.
- Specificity: correctly not stated / labelled absent.
- Balanced accuracy: the mean of the two.

All are reported with Wilson 95% CIs. Head-to-head on the same images: the discordant pairs (caught
by one model only, among positives; false alarm by one only, among negatives) with an exact McNemar
test, two-sided, reported per pair without multiplicity adjustment (descriptive).

## Hypotheses (two-sided; 2,000 image-level bootstrap resamples, seed 20261023)
Summary measure: mean balanced accuracy over the 12 conditions with at least 10 positives.
- **H1 (primary):** C-report minus MAIRA-2. "CheXagent better" if the 95% CI lies above 0, "MAIRA-2
  better" if it lies below 0, otherwise "no difference shown".
- **H2:** C-answer minus MAIRA-2, same criterion.
- **Also reported:** the same differences for sensitivity and specificity separately, per condition, and the sensitivity analysis.

## Limitations recorded now
- A model that states more conditions gains sensitivity and loses specificity. Balanced accuracy
  weighs both equally, which may not match clinical priorities.
- CheXagent's area prompts are our choice of how to make it write a report; MAIRA-2's prompt is its
  own.
- NIH labels are NLP-mined; one dataset only in this stage.
- CheXagent probably saw NIH images in training; MAIRA-2 did not.
- The reader (Qwen) can make errors; the quote check limits invented findings, not missed ones.
- The motivating effusion check was exploratory and post hoc.

All results are reported whatever they are; changes only as dated amendments before the affected run.

Implementation: `own_words_detection.py` (stages `maira`, `chexagent`, `read`, `analyze`).

## Amendments

**A1 (2026-10-25, before any report was read; 351 MAIRA-2 reports generated, 0 CheXagent reports).**
Added a secondary analysis on the same 3-level report scores used for MedGemma (present 1, possible
0.5, not stated 0):
- mean AUC of CheXagent's own report and of MAIRA-2's own report, over the conditions with at least
  10 positives;
- the difference (C-report minus MAIRA-2), with a paired image-level bootstrap 95% CI, two-sided:
  "CheXagent better" if the CI lies above 0, "MAIRA-2 better" if below, otherwise "no difference
  shown";
- CheXagent's p_present AUC is shown for reference only (a continuous score, not like-for-like).

Both report scores share the same 3-level ceiling, so the comparison is like-for-like.

**A2 (2026-10-25, same time as A1).** The run proceeds in stages. **Stage 1** is a fixed random 500 of
the 1,500 test images (`random.Random(20261025).sample` of the sorted test list). All analyses above
apply to stage 1 alone and are reported as stage 1. In stage 1 the means cover the 7 conditions with at
least 10 positives among the 500, counted from the NIH labels only: cardiomegaly 19, pleural effusion
18, consolidation 10, atelectasis 21, infiltration 65, nodule 31, mass 19. The other 6 conditions are
reported but not averaged. The remaining 1,000 images may follow only by a further dated amendment.
The 351 MAIRA-2 reports already generated (110 of them in stage 1) are kept, since the settings are
unchanged.
