# Anatomical measurements: feasibility (NIH ChestX-ray14)

Script: `experiments/anatomical_measurements.py` (criteria pre-stated in its docstring). Data: `anatomical_measurements_nih_20261108-164255.json`.
1700 NIH images (PA and AP; per view 150 each of Cardiomegaly, Effusion, Atelectasis, Emphysema and 250 No Finding); 6087 patients used elsewhere in the project excluded. Positive vs No Finding is an optimistic comparison (easy controls).
Feasible for display = computable (QC pass) in >= 80% of images and AUC lower 95% bound > 0.60.

| View | Condition | Measurement | Computable | AUC (95% CI) | Median pos / neg | Verdict |
|---|---|---|---|---|---|---|
| PA | Cardiomegaly | ctr | 95% | 0.93 (0.90-0.95) | 0.538 / 0.443 | **feasible** |
| PA | Emphysema | lung_height | 98% | 0.47 (0.41-0.53) | 0.736 / 0.743 | not supported |
| PA | Emphysema | lung_area | 98% | 0.48 (0.42-0.53) | 0.558 / 0.571 | not supported |
| PA | Emphysema | dome_ratio | 98% | 0.47 (0.41-0.54) | 0.161 / 0.156 | not supported |
| PA | Effusion | cp_angle_max | 99% | 0.77 (0.72-0.81) | 60.478 / 46.628 | **feasible** |
| PA | Effusion | recess_depth_min | 99% | 0.78 (0.73-0.83) | 0.067 / 0.103 | **feasible** |
| PA | Atelectasis | lung_share_dev | 100% | 0.58 (0.52-0.64) | 0.051 / 0.044 | not supported |
| PA | Atelectasis | hilum_asym | 96% | 0.54 (0.48-0.60) | 0.078 / 0.067 | not supported |
| PA | Atelectasis | trachea_shift_abs | 94% | 0.49 (0.43-0.55) | 0.010 / 0.011 | not supported |
| AP | Cardiomegaly | ctr | 76% | 0.85 (0.80-0.89) | 0.586 / 0.497 | not supported |
| AP | Emphysema | lung_height | 92% | 0.53 (0.47-0.59) | 0.841 / 0.840 | not supported |
| AP | Emphysema | lung_area | 92% | 0.54 (0.48-0.61) | 0.651 / 0.627 | not supported |
| AP | Emphysema | dome_ratio | 92% | 0.50 (0.44-0.57) | 0.178 / 0.188 | not supported |
| AP | Effusion | cp_angle_max | 93% | 0.69 (0.64-0.75) | 66.063 / 49.460 | **feasible** |
| AP | Effusion | recess_depth_min | 93% | 0.63 (0.57-0.69) | 0.075 / 0.094 | not supported |
| AP | Atelectasis | lung_share_dev | 93% | 0.46 (0.40-0.52) | 0.040 / 0.050 | not supported |
| AP | Atelectasis | hilum_asym | 80% | 0.50 (0.43-0.56) | 0.070 / 0.074 | not supported |
| AP | Atelectasis | trachea_shift_abs | 74% | 0.51 (0.45-0.58) | 0.015 / 0.012 | not supported |

Low lung volumes: no NIH label, so the measurements (lung height / thoracic width, lung area / width^2, dome levels) are shown as descriptive only, not validated.

## Classifier saliency (candidate opacity location)

Grad-CAM, torchxrayvision DenseNet 'all'. Hit = saliency peak inside the NIH radiologist box; baseline = share of the segmented lung inside the box. (Amendment: the box images are all project patients, so they were included for this check, which tunes nothing.)

| NIH box label | CAM class | n | Hit rate (95% CI) | Baseline | Verdict |
|---|---|---|---|---|---|
| Effusion | Effusion | 150 | 16% (11%-23%) | 13% | not supported |
| Atelectasis | Atelectasis | 174 | 0% (0%-2%) | 8% | not supported |
| Pneumonia | Consolidation | 120 | 13% (8%-21%) | 26% | not supported |
| Infiltrate | Consolidation | 119 | 13% (8%-21%) | 29% | not supported |

Saliency peaks fall mostly on the image's lower edge (below the lungs): it does not localize the lesion. Not shown in the report.

## QC

Rotation limit: |rotation index| <= 0.035 (95th percentile, PA No Finding).

- PA: lungs 99%, heart 98%, midline 94%, base_R 99%, base_L 99%
- AP: lungs 95%, heart 94%, midline 76%, base_R 95%, base_L 95%

## Summary by condition

| Condition | Measurement | Status |
|---|---|---|
| Cardiomegaly | Cardiothoracic ratio with drawn lines | Feasible (PA AUC 0.93; AP 0.85, computable 76%). AP and rotation flagged. Prior-to-current CTR change is NOT reliable (see the MIMIC limitations note in logs/). |
| Low lung volumes | Lung height, projected area, dome levels | Descriptive only (no labels to validate). |
| Hyperinflation | Dome height ratio, lung height, area | Not supported (emphysema AUC about 0.5): outlines plus descriptive lung height only. |
| Pleural effusion | Costophrenic angle, recess depth | Feasible in PA (AUC 0.77 / 0.78); angle feasible in AP (0.69). Describes the lung outline; fluid location is not inferred. |
| Atelectasis | Lung area share, hilum asymmetry, trachea shift | Not supported (AUC 0.46-0.58). |
| Consolidation | Classifier saliency | Not supported (pointing below the random-lung baseline). |