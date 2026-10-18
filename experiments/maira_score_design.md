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

**A1 (2026-10-17, before any reading or analysis below): standalone comparison with CheXagent on
three held-out labelled sets.**
- Question: does CheXagent's presence score separate present from absent better than MAIRA-2's,
  each on its own (no generated report involved)?
- Data (CheXagent p_present already recorded for all; MAIRA-2 scored with exactly the score,
  sentences, prompt and settings above):
  - NIH test split of the random sample (1,500 images; MAIRA-2 already scored, not analysed);
  - Indiana (544 images, `ppresent_indiana.jsonl` labels and files);
  - VinDr Kaggle-derived (3,000 images, `vindr_test_manifest.json`; pulmonary edema not
    labelled, so five findings).
- Per dataset and finding: AUC of each model, 95% CI from 2,000 image-level bootstrap resamples
  (seed 20261017), paired difference CheXagent minus MAIRA-2 with its 95% CI.
- Hypothesis per dataset and finding: CheXagent's AUC is higher; supported if the difference's
  95% CI lies above 0, not supported if it includes 0, reversed if it lies below 0. All 17
  comparisons are reported; no correction for multiplicity, so single results are read with
  that in mind; the overall pattern across datasets is the main result.
- Findings with fewer than 10 positives in a dataset are reported but marked as too few.
- Caveat recorded now: CheXagent's training data probably include public NIH and VinDr images;
  MAIRA-2's published training sets (MIMIC-CXR, PadChest, USMIE) do not. Indiana is the least
  affected set.

**A2 (2026-10-18, before any reading below): the seven further conditions.**
- Conditions: pneumonia, infiltration, emphysema, nodule, mass, fibrosis, pleural thickening.
- MAIRA-2 sentence pairs, the same mechanical template as the six ("a" for countable nouns only):
  "There is pneumonia." / "There is no pneumonia."; "There is infiltration." / "There is no
  infiltration."; "There is emphysema." / "There is no emphysema."; "There is a nodule." / "There
  is no nodule."; "There is a mass." / "There is no mass."; "There is fibrosis." / "There is no
  fibrosis."; "There is pleural thickening." / "There is no pleural thickening." Score, prompt and
  settings as above.
- Data: NIH test split (1,500; NIH labels Pneumonia, Infiltration, Emphysema, Nodule, Mass,
  Fibrosis, Pleural_Thickening; CheXagent readings already in `provisional_readings.jsonl`), and
  VinDr (3,000) for the classes it labels: Infiltration -> infiltration; Nodule/Mass -> both
  nodule and mass (one combined label, so each is scored against it); Pulmonary fibrosis ->
  fibrosis; Pleural thickening -> pleural thickening. VinDr label = at least 2 of 3 radiologists
  (as in Amendment 2 of the VinDr protocol). CheXagent is read on VinDr for these five with the
  validated query ("Is there {condition}?", 8 tokens). No Indiana labels exist for these.
- Analysis, hypothesis and reporting exactly as in A1 (per dataset and condition; fewer than 10
  positives marked as too few). The six-finding results of A1 are not recomputed; the final
  report lists A1 and A2 together.
