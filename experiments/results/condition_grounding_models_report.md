# Condition-specific grounding models vs CheXagent / MAIRA-2: results (2026-11-08)

**Purpose.** Improve grounding visualizations in the radiologist evaluation reports. MedGemma, CheXagent presence
verification and the Evidence Engine are not touched, and no existing output, score or report was modified.

**Scope.**
- Grad-CAM and attention maps were not used.
- Every model ran on the whole current image. MedGemma's findings, its claimed side and the reference report played
  no part in localization.

**Code and data.**
- Code: `experiments/condition_grounding_models.py` (`validate`, `cases`).
- Environment: `experiments/condition_grounding_models_ENV.md`.
- Validation numbers: `experiments/results/condition_grounding_validation_20261108-200948.json`.
- Gallery and per-case results (MIMIC-CXR, logs only): `logs/condition_grounding_gallery_20261108-201240.pdf` / `.json`.

## 1. Model discovery and suitability matrix

| Model | Pathology | Output | Training data | Reported localization metric | Preprocessing | Licence / deps | Overlap with our evaluation data | Obtained |
|---|---|---|---|---|---|---|---|---|
| vasandesh/pneumothorax-unet-efficientnet-b4 | Pneumothorax | **True segmentation** (mask) + classification | SIIM-ACR (2,379 positives) | Dice 0.51 on positives (card); classification AUC 0.946 | ImageNet norm, 3 ch, 512 | MIT; segmentation_models_pytorch, timm | SIIM-ACR images come from NIH, so NIH is not independent; VinDr is independent | Yes |
| ianpan/pneumothorax-cxr | Pneumothorax | **True segmentation** + classification | **Undocumented** (empty model card) | None reported | 1 ch, stretched to 512, [-1,1] | **Licence not stated**; timm; repo code (reviewed; loaded without remote-code execution) | Unknown, because training data is undocumented | Yes |
| ianpan/pneumonia-cxr | Pneumonia / lung opacity | Segmentation of **ellipses converted from boxes** (not pixel-level truth) + classification | RSNA Pneumonia (NIH images) + SIIM-FISABIO-RSNA COVID-19 | None (classification AUC 0.90 only) | as above | **Licence not stated**; timm | NIH not independent; VinDr independent | Yes |
| YOLOv5x trained on VinDr-CXR (node21-submit, `yolo5x_vindr.pt`) | 14 VinDr classes incl. effusion, atelectasis, consolidation, nodule/mass | Detection (boxes + scores) | VinDr-CXR (Kaggle train) | Challenge mAP (no per-class figures in the repo) | YOLOv5 | Apache-2.0; YOLOv5 code base; pickle checkpoint | Our VinDr annotations are sampled from Kaggle train, so not independent; NIH would be | **No:** GitHub large-file host blocked on this network |
| NODE21 winner (FinnBehrendt/node21-submit: Faster R-CNN / EfficientDet / YOLO ensemble) | Pulmonary nodules | Detection | NODE21 (+ VinDr pretraining) | Challenge detection score | Docker / evalutils | Apache-2.0 | NODE21 includes JSRT/ChestX-ray14 images | **No:** same block; Docker-based |
| JTuong/EDAvaMachineLearningChestXYolo11m | VinDr "4 macro classes" (assumed) | Detection | Not documented (no card) | None | Unknown | Unknown | Unknown | Not used (undocumented) |
| Roboflow "effusion chest-xray segmentation" | Pleural effusion | Instance segmentation | ~200 images | mAP@50 0.63 | Roboflow API | CC BY 4.0; needs account/API | Unknown | Not used (tiny training set; hosted API) |
| ROSALIA (MIMIC-ILS) | 7 lesion types incl. effusion, atelectasis, consolidation | Prompted segmentation (vision-language) | MIMIC-ILS (automatically generated masks from MIMIC-CXR) | Reported in arXiv 2511.15186 | VLM | Weights not located | **MIMIC-CXR**, so it overlaps our MIMIC cases | Not found |

**Conclusions from discovery.**
- **Effusion, atelectasis, nodules:** no downloadable, documented, true-localization model was obtainable on this
  network.
- **Consolidation:** covered only by the ellipse-trained pneumonia model.

## 2. Objective validation (radiologist boxes)

**Method.**
- Metrics were pre-stated. Primary: IoU between the predicted region and the union of the truth boxes. Also reported:
  pointing hit, precision, recall and side agreement.
- The truth is boxes, not masks, so Dice is not available.
- A secondary metric was added after the first run: each mask converted to the bounding boxes of its regions ("as
  boxes"). Tight masks are otherwise penalized against loose truth boxes.

**Caveats.**
- CheXagent's training data (CheXinstruct) includes VinDr-CXR grounding, so the VinDr comparison is optimistic for
  CheXagent.
- On NIH, the SIIM, RSNA and ianpan models may have seen the images.

### VinDr pneumothorax, 15 images (independent of the specific models and MAIRA-2)

| Method | IoU | Pointing hit | Precision | Side correct | Empty | Classification AUC |
|---|---|---|---|---|---|---|
| CheXagent (optimistic) | 0.67 [0.58-0.76] | 0.93 | 0.84 | 1.00 | 0% | - |
| MAIRA-2 | 0.29 [0.18-0.41] | 0.60 | 0.57 | 0.73 | 0% | - |
| vasandesh mask | 0.32 [0.26-0.37] | 0.80 | 0.90 | 1.00 | 0% | 0.99 |
| vasandesh as boxes | **0.65 [0.54-0.75]** | 0.80 | 0.84 | 1.00 | 0% | |
| ianpan mask | 0.22 [0.14-0.30] | 1.00 | 0.98 | 1.00 | 7% | 0.99 |
| ianpan as boxes | 0.53 [0.37-0.69] | 1.00 | 0.95 | 1.00 | 7% | |

**Paired IoU differences** (bootstrap 95% CI):

| Comparison | IoU difference | 95% CI |
|---|---|---|
| vasandesh (as boxes) vs CheXagent | -0.03 | -0.12 to 0.06 |
| vasandesh (as boxes) vs MAIRA-2 | +0.35 | 0.23 to 0.48 |
| ianpan (as boxes) vs MAIRA-2 | +0.24 | 0.05 to 0.44 |

### VinDr consolidation, 25 images (pneumonia model as a proxy)

| Method | IoU | Pointing hit | Side correct | Empty | AUC |
|---|---|---|---|---|---|
| CheXagent (optimistic) | 0.59 | 0.92 | 0.88 | 0% | - |
| MAIRA-2 | 0.12 | 0.32 | 0.57 | 16% | - |
| ianpan pneumonia (mask / as boxes) | 0.14 / 0.16 | 0.76 | 0.92 | **48%** | 0.95 |

### NIH (optimistic for the specific models)

| Condition | n | Result |
|---|---|---|
| Pneumothorax | 93 | vasandesh as boxes IoU 0.40 vs CheXagent 0.33 (+0.06 [0.03-0.10]) and MAIRA-2 0.17 |
| Pneumonia | 116 | ianpan as boxes IoU 0.23 vs CheXagent 0.31 and MAIRA-2 0.07; ianpan empty 45% |

## 3. Feasibility on the scored cases (38 findings from the returned forms)

Of the 38 findings, 11 are pneumothorax and 27 are pneumonia-type (consolidation, pneumonia, infiltration,
opacities).

**vasandesh (pneumothorax, 11 findings).**
- **Regions:** found on 10 of 11, all small (0.3–2.6% of the image), none mostly outside the thorax.
- **Low confidence:** its own probability was below 0.5 on 6 of those 10. The mask is not gated by the classifier,
  so regions appear even when the model says "no pneumothorax".
- **Stability (median Dice):** padding vs stretching 0.86, gamma 0.79–0.84, **horizontal flip 0.64**.
  - Side changed under preprocessing in 4 findings.
  - Dice fell below 0.5 in 3.
  - 2 regions were "diffuse": scattered on both sides along edges, e.g. Comp 8.

**ianpan pneumothorax (11 findings).**
- **No region on any of the 11** (probabilities 0.01–0.39), although it localized well on VinDr.
- These form cases are mostly treated, small or medial pneumothoraces in films with tubes.

**ianpan pneumonia (27 findings).**
- **Regions:** found on 10 of 27; no region on 17, including 9 MedGemma-stated findings.
- **Stability:** very stable (median Dice 0.95–0.97).
- **Shape:** regions are ellipses (2–13% of the image) that can under-represent diffuse or whole-lobe opacity. In an
  NIH case where the opacity fills the left hemithorax, it marked only a small oval.

**Side disagreement between methods.** 2 findings, both pneumothorax:

| Finding | CheXagent | MAIRA-2 | vasandesh | ianpan |
|---|---|---|---|---|
| Comp 8 | left | right | both sides (diffuse) | none |
| Comp 10 | left | right | none | none |

No method agrees with another here.

**Coverage.**
- MAIRA-2 boxes are cached only for the 19-case set: 3 of 38 findings.
- CheXagent has per-finding boxes for 7 findings and an embedded form image for 15. There is no box for 16 findings
  (CheXagent-only or missed rows).

## 4. Failure-case analysis
1. **Ungated masks (vasandesh).** Regions are drawn even at p = 0.05. A display must gate on the model's own
   probability, which would leave 4 of 11 findings with a region.
2. **Left-right instability (vasandesh).** Flip Dice is 0.64, and the side changed in 4 of 11 findings. This is the
   same weakness that made side disagreement a problem for CheXagent.
3. **Domain gap (ianpan pneumothorax).** It is excellent on VinDr but silent on the MIMIC/NIH form cases. Small,
   medial or treated pneumothoraces with chest tubes, as in Comp 8, are the hard cases.
4. **Ellipse supervision (ianpan pneumonia).** Its regions are rounded approximations of boxes, not opacity
   boundaries. It returned no region in 45–48% of positive images in validation.
5. **Box-based truth.** None of the validation sets has pixel masks, so we cannot measure boundary accuracy. Only
   region, side and box overlap are measurable.
6. **Small samples.** There are 15 and 25 independent VinDr images. Every confidence interval is wide, and the form
   cases (11 pneumothorax) are case examples, not statistics.

## 5. Recommendation
- **Not suitable as a replacement for CheXagent's box:** none of the obtainable models. On the independent VinDr
  sets, the best one (vasandesh, as boxes) is statistically indistinguishable from CheXagent on pneumothorax, and
  CheXagent is itself optimistic there. The pneumonia model is clearly worse than CheXagent on consolidation.
- **Possibly suitable, as an extra panel in a future visualization review: vasandesh pneumothorax only**, under three
  conditions:
  - show its mask only when its own probability is at least 0.5;
  - show it beside, not instead of, CheXagent and MAIRA-2;
  - flag cases whose side changes under a horizontal flip.

  Its advantage is clear only against MAIRA-2 (+0.35 IoU on VinDr). A radiologist review would test whether the
  outline helps more than a box, especially for side disagreements (Comps 8 and 10).
- **Not recommended:**
  - `ianpan/pneumothorax-cxr`: undocumented training data and licence, and no regions on our cases.
  - `ianpan/pneumonia-cxr`: no licence, ellipse supervision, and too many missed regions.
- **Effusion, atelectasis, nodules:** no suitable model was obtainable. The most promising documented option, the
  VinDr YOLOv5x detector, needs a manual download (blocked here) and validation on NIH boxes, which are independent of
  VinDr.
- **Claims:** none of these results shows that a model improves clinical localization. They show where the models
  agree with radiologist boxes on public data. Usefulness to radiologists still has to be rated.
