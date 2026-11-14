# Condition-aware grounding display: comparison report (exploratory)

Built 2026-11-08 10:58 by experiments/condition_aware_grounding.py. CheXagent's predictions, the Evidence Engine and all scoring results are unchanged; the boxes are the recorded report-blinded ones (side-free phrase). Segmentation (torchxrayvision PSPNet, ChestX-Det), lung zones and measurements are display aids, not evidence that a condition exists.

## 1. Coordinate checks

| test | result |
|---|---|
| box conversion (normalized 0-100 -> pixels, 3 shapes) | passed |
| scale: full-size vs 1,024-px masks, mean IoU over 5 MIMIC images x 3 organs | 0.993 |
| laterality: horizontal flip swaps right/left lung masks (IoU) | [0.904, 0.872] |
| laterality: patient's left lung lies on the image right (NIH images) | 946/946 |
| radiologist boxes whose centre lies in the segmented lungs or heart | 887/947 |

## 2. Automated comparison on 947 NIH radiologist boxes

Hit = the centre of the radiologist's box falls inside the display (the box, or the shaded region). A larger display hits more easily, so the share of the lungs it covers is reported next to it: a display is only better if it hits more without covering most of the lungs. 95% CIs: bootstrap over findings (2000 resamples).

| condition (NIH label) | display type | n located / n | original box: hit | lung zone: hit | box share of lungs | zone share of lungs | zone side = truth side |
|---|---|---|---|---|---|---|---|
| Atelectasis | regional | 171/173 | 61% [53-68] | 67% [60-74] | 10% | 24% | 87% |
| Cardiomegaly | global-heart | 140/141 | 100% [100-100] | 79% [71-85] | 28% | 37% | - (heart) |
| Effusion | regional | 149/150 | 49% [41-57] | 56% [47-64] | 10% | 24% | 95% |
| Infiltrate | regional | 117/117 | 62% [54-71] | 73% [64-80] | 35% | 54% | 93% |
| Mass | focal | 81/81 | 58% [47-69] | 79% [70-88] | 12% | 34% | 89% |
| Nodule | focal | 75/76 | 25% [16-36] | 64% [53-75] | 6% | 31% | 84% |
| Pneumonia | regional | 112/116 | 61% [52-70] | 74% [66-82] | 19% | 37% | 87% |
| Pneumothorax | focal | 93/93 | 63% [54-73] | 75% [67-84] | 12% | 20% | 96% |

### Cardiomegaly (global, heart): which outline matches the radiologist's box

| display | mean IoU with radiologist box | IoU >= 0.5 | centre hit |
|---|---|---|---|
| CheXagent box (located 140/141) | 0.57 | 71% [64-79] | 100% [100-100] |
| Heart segmentation outline (all 141) | 0.67 | 94% [89-97] | 100% [100-100] |
| MAIRA-2 box (all 141) | 0.75 | 100% [100-100] | - |

## 3. Where the zone display fails (regional findings)

- 183 of 549 located regional findings: the radiologist's box centre is outside the shaded zone(s).
  - box outside the segmented lungs (no zone): 16
  - zone on the wrong side: 51
  - right side, wrong third (zone too coarse or box misplaced vertically): 116
- Radiologist boxes centred outside the segmented lungs and heart (segmentation or lateral/pleural findings at the edge): 60
- Global lung findings (low lung volumes, hyperinflation) have no radiologist boxes in NIH, so they are judged only in the blinded review and the gallery.
- Focal findings: the display is unchanged (original box), so its accuracy is the box's; zones are too coarse to localize nodules and masses.

## 4. Still to do before adoption

- Blinded radiologist review (logs/condition_aware_grounding_review_*.pdf; key kept separately): per display, 'easier to interpret' and 'localization clinically accurate' rated separately, plus preference.
- Adopt a display for a condition only if the review favours it on both questions.
