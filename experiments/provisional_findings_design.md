# Provisional cut-offs for findings beyond the six validated ones (design)

Date: 2026-10-09. Written and committed before any CheXagent reading for these findings exists.

## Scope and status
- A **separate, provisional extension** for the radiologist review tool
  (`experiments/comparison_edv_pdf.py`). It is **not** part of Evidence Engine v1.0, which stays
  frozen (tag `evidence-engine-v1.0`), and it is not part of the paper's pre-registered results.
- In the review, statuses based on these cut-offs are labelled **"provisional"**.

## Findings (NIH ChestX-ray14 labels)
| CheXagent question `Is there {q}?` | NIH label | Report terms mapped to it in the review |
|---|---|---|
| pneumonia | Pneumonia | pneumonia |
| infiltration | Infiltration | opacity, opacities, infiltrate, infiltration, airspace disease, airspace opacity |
| emphysema | Emphysema | emphysema, hyperinflation, hyperinflated |
| nodule | Nodule | nodule |
| mass | Mass | mass |
| fibrosis | Fibrosis | fibrosis, fibrotic |
| pleural thickening | Pleural_Thickening | pleural thickening |

Mapped terms are approximations (e.g. hyperinflation is assessed as emphysema; opacity as NIH
"infiltration"); the review names the finding it was assessed as when it differs.

## Data
- NIH ChestX-ray14, the existing **random sample** (5,000 images; `nih_sample_5000.json`) and its
  frozen 70/30 split (`split_nih_70_30.json`): development (fitting) and test (checking) exactly as
  for the six findings. The supplement sample is not used (it was selected on the six findings).
- Labels: NIH "Finding Labels" (NLP-mined from reports; noisy).
- CheXagent readings: same model, prompt form and scoring as all p_present work (bfloat16, greedy,
  first-token Yes/No, fp32 logits).

## Method (as for the six, Evidence Engine v1.0 code)
On the development split only, per finding:
- Platt calibration of p_present; LR(p) = odds(calibrated) / odds(development prevalence), with
  bootstrap CIs (`evidence_engine_d1_d2.Source`, same constants).
- Rule 5 / 0.2, CI-conservative: SUPPORTING if the lower CI of LR >= 5; REFUTING if the upper CI
  <= 0.2, confirmed by empirical development bands (dual-evidence rule,
  `evidence_engine_d3.classify`); scores outside the development support -> no verdict; fewer than
  the engine's minimum development positives -> no verdict.

## Check on the test split, and use in the review (fixed in advance)
Per finding, on the NIH random test split: empirical LR of SUPPORTING and of REFUTING (95% CI,
2,000-sample image-level bootstrap, seed 20261009), coverage, positives assigned to REFUTING,
p_present AUC.
- SUPPORTING is used in the review only if its test empirical LR point estimate is >= 5.
- REFUTING is used only if its test empirical LR point estimate is <= 0.2 and no more than 2% of the
  test positives are assigned to it.
- A state that fails is not shown; scores in it read "Indeterminate (provisional)".
- Results are reported whatever they are; no refitting after looking at the test split.

## Limitations
NIH labels are noisy; the mapping from report wording to NIH labels is approximate; CheXagent may
have been trained on NIH; validation is on one dataset only.
