# MAIRA-2 presence score without training (option 1): design

Date: 2026-10-13. Written and committed before any MAIRA-2 reading of the study images exists.
Not part of Evidence Engine v1.0 (frozen) and not part of the pilot protocol.

## Question
Does MAIRA-2's own sentence likelihood carry a usable presence score for the six findings, and
how does it compare with CheXagent's p_present on the same images?

## Score
For each image and finding, MAIRA-2 (microsoft/maira-2, revision 795a2b1c…, bf16, as in the
pilot) gets its standard report prompt (`format_and_preprocess_reporting_input`: current frontal
image only; no lateral, prior, indication, technique, comparison or prior report). Two fixed,
side-free sentences are scored as the start of its findings:

| finding | positive | negative |
|---|---|---|
| cardiomegaly | There is cardiomegaly. | There is no cardiomegaly. |
| pleural effusion | There is a pleural effusion. | There is no pleural effusion. |
| pneumothorax | There is a pneumothorax. | There is no pneumothorax. |
| consolidation | There is consolidation. | There is no consolidation. |
| pulmonary edema | There is pulmonary edema. | There is no pulmonary edema. |
| atelectasis | There is atelectasis. | There is no atelectasis. |

log P(sentence) = sum of the token log-probabilities of the sentence after the prompt (prompt
encoded once, its key/value cache reused per sentence). **Score = P(pos) / (P(pos) + P(neg))**,
the analogue of CheXagent's Yes share. Both log-probabilities are recorded.

Plumbing check (2026-10-13, on two NIH images outside the study samples, 00001398_001/002):
MAIRA-2's own first sentence scored -2.8 / -1.9 against -12 to -25 for the fixed sentences;
tokenisation aligned with generation; cached and batched scoring agreed within 0.33.

## Data
NIH ChestX-ray14, the existing random sample (5,000 images, `nih_sample_5000.json`) and its
frozen 70/30 split (`split_nih_70_30.json`). All 5,000 are scored (the test split is kept for a
later calibration check); this feasibility analysis uses the **development split only**. NIH
labels as in `ppresent_nih.py`. NIH is not among MAIRA-2's published training sets (MIMIC-CXR,
PadChest, USMIE); labels are NLP-mined and noisy.

## Comparison and analysis (development split)
- CheXagent: p_present for the same images and findings (`ppresent_nih.jsonl`, set "random").
- Per finding: AUC of each model's score, 95% CI from 2,000 image-level bootstrap resamples (seed
  20261013); paired AUC difference MAIRA-2 minus CheXagent with its 95% CI; positives and n.
- Reported whatever the result. No change of sentences, prompt or score after seeing readings;
  any change is a dated amendment below and is reported as such.

## Decision rule (fixed now)
A finding goes forward to calibration (Platt + cut-offs, as for the provisional findings, under a
separate design) only if MAIRA-2's development AUC is at least 0.75. Otherwise MAIRA-2's score is
not used for that finding.

## Amendments
(none)
