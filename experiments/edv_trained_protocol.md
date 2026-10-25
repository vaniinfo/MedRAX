# Does EDV improve the team's trained MedGemma? NIH longitudinal protocol

Date: 2026-10-25. Written and committed before any trained-model report on the test pairs below is
generated. Extends `edv_longitudinal_protocol.md` (base MedGemma) to the team's trained model.

## Data
The same 499 NIH test pairs (`results/edv_long_pairs.json`; prior + current image, NIH labels of the
current image, 13 conditions). Mean AUC over the 9 conditions with at least 10 positives (as before).

## Report generator: the team's trained MedGemma
- Model: `vaniinfoorg/medgemma_trained`, revision `e346cb15bd150731d92a91a8a73f7c8da24d350a` (merged
  SFT + RL MedGemma 1.5 4B, trained by the team on MIMIC-CXR). Run locally with transformers 4.57.3,
  bf16.
- Input: the team's training format (`build_prompt_messages`): "Prior chest X-ray:", prior image,
  "Prior report:\n{report_before}", "Current chest X-ray:", current image, "Clinical indication for
  current study: {indication}\n\nCompare the current chest X-ray to the prior chest X-ray. Describe any
  interval change in findings and provide an impression."
- NIH has no reports or indications, so fixed placeholders are used: prior report **"Not
  available."**, indication **"Not provided."**. Chosen on 3 NIH development pairs (format only, no
  labels): this wording gave radiologist-style prose in 2 of 3, against 0 of 3 for "___".
- Images: NIH PNGs as RGB, resized by the model's own processor. Greedy decoding, 512 new tokens.
- Read for the 13 conditions with the same quote-checked lookup and amendment A1 rule as before;
  score present 1 / possible 0.5 / not stated 0.

## EDV scores (already recorded, unchanged)
CheXagent option A (two-image), option B (mean of current and prior readings), single-image
CheXagent, and MAIRA-2 with the prior image, from the longitudinal experiment. Percentile among that
model's scores on the 499 pairs (label-free); combined = 0.5 x MedGemma score + 0.5 x EDV percentile.
Image-condition pairs without a CheXagent A or B score are excluded for every system, as before.

## Hypotheses (paired 95% CIs, 2,000 image-level bootstrap resamples, seed 20261025)
- **H3 (primary):** mean AUC (trained MedGemma + CheXagent A) minus (trained MedGemma alone) > 0.
- **H3b:** mean AUC (trained MedGemma + CheXagent A) minus (trained MedGemma + MAIRA-2-L) > 0.
- Supported if the 95% CI lies above 0.
- **Exploratory, two-sided:**
  - trained MedGemma alone minus base MedGemma-L alone;
  - trained + CheXagent A minus base + CheXagent A;
  - trained + CheXagent B, and + CheXagent single-image, minus trained alone;
  - per-condition AUCs.
- **Secondary (yes/no rule, as in `edv_improvement_protocol.md`):** the report states a condition
  if its score is 1 or 0.5. The [Image check] rule removes a stated condition if CheXagent A's
  decision is AGAINST, and adds pleural effusion or cardiomegaly if FOUND. Reported: mean
  sensitivity, specificity and balanced accuracy over the 9 conditions, trained alone vs with the
  rule, with paired bootstrap CIs.
- **Descriptive:** share of trained-model reports that describe a comparison with the prior (any of
  "compar", "interval", "prior", "previous", "unchanged", "stable", "new", "improv", "worsen",
  "increas", "decreas", "resolv"), next to base MedGemma-L.

## Limitations recorded now
- The trained model runs without the prior report and indication it was trained with
  (placeholders), and on NIH rather than MIMIC images; its output style partly drifts to bullet
  lists. This tests the model out of its training setting.
- NIH labels are NLP-mined, and give presence on the current image only, not change.
- CheXagent probably saw NIH images in training. CheXagent option A runs beyond its 2,048-token
  design length.
- Percentiles come from the evaluation set itself (label-free).
- MedGemma's 3-level score caps its AUC; the secondary yes/no analysis does not have this property.

All results are reported whatever they are; changes only as dated amendments before the affected run.

Implementation: `edv_trained.py`. The `generate` stage runs in the separate trained-model
environment (`.venv-medgemma`); `read` and `analyze` run in the main environment.

## Amendments
(none)
