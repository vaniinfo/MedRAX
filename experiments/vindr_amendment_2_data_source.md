# Amendment 2 to the VinDr protocol: data source (Kaggle VinBigData release), pre-registered

Date: 2026-10-04. Written and committed **before any VinDr data, image list or label file was
downloaded or examined.**

## Reason
The pre-registered source, VinDr-CXR v1.0.0 on PhysioNet (test split, 3,000 images, consensus
labels), needs PhysioNet credentialing (CITI training), which will not be done. The same VinDr
images are available without credentialing as the Kaggle competition "VinBigData Chest X-ray
Abnormalities Detection" (2021), whose labelled part is VinDr's training split, read by three
radiologists per image (no consensus) with boxes for 14 local findings. This amendment replaces
the data source and fixes how the Kaggle labels are turned into the labels Part A and Part B
use. Nothing else changes.

## What does not change
- Part A (`vindr_external_validation_design.md`, `vindr_external_validation.py`, commit
  `5280b43`) and Part B (Amendment 1, `vindr_amendment_1_grounding.md`,
  `vindr_grounding_amendment.py`, tag `vindr-amendment-1`): code, prompts, models, endpoints,
  thresholds, seeds and interpretation are unchanged. Both scripts run unmodified on the derived
  files described below (`--data` / `VINDR_DIR` pointing to the derived folder).
- The DICOM to PNG conversion (Part A) is unchanged; original DICOMs are used, not the resized
  PNG copies published by Kaggle users.

## Data
- Source: Kaggle competition `vinbigdata-chest-xray-abnormalities-detection`, files
  `train.csv` and `train/<image_id>.dicom`. The competition's test images are unlabelled and
  are not used.
- **Sample (label-independent, fixed before any label is seen):** the image IDs are taken from
  the competition's file list (file names only). Each ID is ranked by
  SHA-256(`"20261004:" + image_id`); the 3,000 lowest are used. The list is written to
  `experiments/results/vindr_kaggle_sample.json` and committed before `train.csv` is
  downloaded.
- Images with other than exactly three distinct readers (`rad_id`) in `train.csv` are excluded
  and listed.

## Labels derived from the three readers (fixed)
For a finding and image, a reader **marks** it if the reader drew at least one box of that
class.
- **Image-level label (Part A):** present if at least 2 of the 3 readers mark it; otherwise
  absent. The number of images marked by exactly 1 reader is reported per finding.
- Class mapping: "Cardiomegaly" -> cardiomegaly, "Pleural effusion" -> pleural effusion,
  "Pneumothorax" -> pneumothorax, "Consolidation" -> consolidation, "Atelectasis" ->
  atelectasis.
- **Pulmonary edema:** the competition has no edema class. The derived label file sets
  `Edema` = 0 for every image only because Part A's code requires the column; pulmonary edema
  is **not evaluable by construction** and is reported as "not labelled in this data source",
  not as a negative result. (If `train.csv` does contain an "Edema" class, it is used with the
  same rule.)
- **Sides (Part B):** for each finding present by the rule above, a reader marks a side if any
  of the reader's boxes of that class lies on that side (side from the box centre; a box with
  x_min < 40% and x_max > 60% of the width marks both sides; as in Amendment 1). A side is
  **confirmed** if at least 2 readers mark it.
  - One side confirmed: unilateral case; its boxes are the readers' boxes on that side that do
    not cross the midline.
  - Both sides confirmed: bilateral case; all boxes of the readers who marked the finding.
  - No side confirmed (readers disagree on the side): excluded from Part B and counted.
  The derived box file has one set of boxes per image and class, so Part B's single-rater check
  passes by construction.

## Derived files
`experiments/vindr_kaggle_derive.py` writes, under `D:/datasets/vindr-cxr-kaggle-derived/`:
- `test/<image_id>.dicom`: the 3,000 sampled DICOMs (hard links, or copies);
- `annotations/image_labels_test.csv`: one row per image, the five labels above plus `Edema`;
- `annotations/annotations_test.csv`: image_id, class_name, x_min, y_min, x_max, y_max, for
  Part B as above;
- `derivation_report.json`: excluded images, single-reader counts, side-conflict counts.

## Additional limitations, reported regardless of result
- Reference standard: majority of three independent readers, weaker than the five-radiologist
  consensus of the PhysioNet test split.
- Images come from VinDr's training split. Public chest X-ray models may have been trained on
  VinDr's training data, so the risk that CheXagent or MAIRA-2 has seen these images is higher
  than for the test split.
- Image-level labels come from boxes of the 14 local classes; findings not boxed are absent by
  this definition.
- Pulmonary edema cannot be evaluated; Part A reports five of its six findings.

## Procedure
1. Accept the competition rules on Kaggle and check they permit this research use.
2. `python experiments/vindr_kaggle_derive.py sample`, then commit
   `experiments/results/vindr_kaggle_sample.json`.
3. `python experiments/vindr_kaggle_derive.py download` (train.csv and the 3,000 DICOMs only).
4. `python experiments/vindr_kaggle_derive.py derive`.
5. Run Part A and Part B exactly as in `NEXT_STEPS.md`, with
   `VINDR_DIR=D:/datasets/vindr-cxr-kaggle-derived`.
No further amendments once any label or image has been examined.
