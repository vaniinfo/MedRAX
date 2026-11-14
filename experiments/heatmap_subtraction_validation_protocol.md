# Validation protocol: BioViL-T heat maps and prior-vs-current subtraction (written 2026-11-09, before any result)

Follows the pilots in `experiments/biovil_t_heatmap_pilot.py` and `experiments/subtraction_pilot.py`, which were run
on 9 findings. Their parameters are frozen and nothing is tuned here. Neither method is shown to radiologists unless
it passes the criterion below.

## A. BioViL-T text-guided heat map: does the map point at radiologist-drawn boxes?

**Model and prompt.**
- Model: microsoft/BiomedVLP-BioViL-T @ 692f09e9, pretrained on MIMIC-CXR. Neither test set below is in its training
  data.
- Prompt: the condition name only.
- Map: as in the pilot.

**Data.**
- VinDr-CXR test, 169 radiologist-boxed images (`experiments/results/vindr_grounding_cases.json`): pleural effusion
  119, consolidation 25, pneumothorax 15, atelectasis 10. Truth = the union of all readers' boxes.
- NIH ChestX-ray14 `BBox_List_2017.csv`: Effusion ("pleural effusion"), Atelectasis ("atelectasis"), Pneumothorax
  ("pneumothorax") and Pneumonia ("pneumonia"); 1024 x 1024 images only.

**Primary metric: pointing game.**
- A hit is when the map's peak (inside the analysed centre crop) falls in a truth box.
- Baseline: a random point inside the lungs, i.e. the share of the PSPNet lung area that lies inside the truth boxes.

**Secondary metrics.**
- Side agreement: the side of the top 10% of the map vs the side of the truth boxes.
- Paired hit-rate difference vs CheXagent and MAIRA-2, using their stored report-blinded boxes and the existing
  box-centre convention (`condition_grounding_models.metrics`). CheXagent is VinDr-trained, so it is optimistic
  on VinDr.

**Criterion, per condition and dataset.** The method is feasible as a "where to look" overlay when the lower bound
of the hit rate's 95% bootstrap CI is above the baseline mean. Conditions with fewer than 15 cases are reported
descriptively only (VinDr atelectasis, n = 10).

**Not tested.** Presence. The pilot showed the map lights up even when the finding is absent, so the map must never
be presented as evidence that a finding exists.

## B. Subtraction: does it separate pairs with label change from stable pairs?

**Data.**
- NIH longitudinal pairs (`experiments/results/edv_long_pairs.json`), same view only (425 of 499).
- Labels: NIH text-mined labels per image.
- Opacity labels: Effusion, Atelectasis, Consolidation, Pneumonia, Infiltration.

**Groups.**
- NEW: at least one opacity label in the current film that is absent on the prior, and none resolved.
- RESOLVED: at least one opacity label on the prior that is absent on the current, and none new.
- STABLE: identical label sets on both films.
- Pneumothorax-new pairs (5) are reported descriptively only.

**Scores.** Frozen pilot pipeline (768 px, ECC registration, per-lung z-score, sigma 6, threshold 0.75 SD):
- denser share = the larger of the two lungs' share of pixels with difference > +0.75 SD;
- lucent share = the same with < -0.75 SD.

**Contrasts.**
- B1: NEW vs STABLE on the denser share.
- B2: RESOLVED vs STABLE on the lucent share.
- Statistic: AUC with 95% bootstrap CI.
- Primary analysis: all pairs, registration failures included.
- Secondary analysis: pairs with lung Dice >= 0.85 only.

**Criterion.** The lower CI bound of the AUC is above 0.6, the same bar as the earlier NIH feasibility checks.

**Known weaknesses, stated in advance.**
- Text-mined labels: a "new" label may reflect report wording rather than real change.
- Per-lung normalization cancels a change that is uniform across a whole lung.
- Basal effusion can lie outside the segmented lung.

## Outputs

- NIH/VinDr numbers only, so they go in `experiments/results/heatmap_subtraction_validation_<stamp>.json` and `.md`.
- No MIMIC content is involved.
