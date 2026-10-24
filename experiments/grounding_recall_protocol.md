# Grounding recall: where do CheXagent and MAIRA-2 put the box? Protocol

Date: 2026-10-24. Written and committed before any box below is produced.

## Why
The CheXagent vs MAIRA-2 comparison measured detection (AUC). The professor asked about grounding
recall: of the findings a radiologist outlined, what share does the model's box land on? Earlier
exploratory checks (`results/location_agreement_dev.md`: pointing game on 416 NIH boxes for
effusion, atelectasis and pneumothorax, MAIRA-2 in 8-bit with an unpinned model version) are
known. This experiment is a fresh, complete run with fixed settings. The overlap (IoU) results
below have not been computed for any data before.

## Data
NIH ChestX-ray14 radiologist boxes (`BBox_List_2017.csv`). All rows whose image is present and
not in the NIH test split (`split_nih_70_30.json`, excluded as in the earlier check): **947 cases**
(image, label), one box each, on 847 images. All NIH images are 1024 x 1024, so box coordinates
need no aspect-ratio correction.

| label | cases | phrase given to the models |
|---|---|---|
| Atelectasis | 173 | atelectasis |
| Effusion | 150 | pleural effusion |
| Cardiomegaly | 141 | cardiomegaly |
| Infiltrate | 117 | infiltration |
| Pneumonia | 116 | pneumonia |
| Pneumothorax | 93 | pneumothorax |
| Mass | 81 | mass |
| Nodule | 76 | nodule |

## Models (phrase without a side; the condition is known to be present)
- **CheXagent:** "Please locate the following phrase: {phrase}" through the pipeline's
  `XRayVQATool.locate_findings`. Boxes the model labels as a different finding are not counted, as
  in the pipeline.
- **MAIRA-2:** its phrase-grounding input (`format_and_preprocess_phrase_grounding_input`, phrase
  capitalised), bf16, revision `795a2b1c…` (`MAIRA_REVISION`), greedy, 150 tokens. Boxes are mapped
  to the original image with `adjust_box_for_original_image_size`; if the output is malformed,
  complete boxes are read from the text.

All boxes are stored as percent of image width and height. No box = miss.

## Measures (per case; a model may return several boxes)
- **Hit at IoU t:** the best of the model's boxes has IoU >= t with the radiologist's box.
  Recall@t = hits / cases.
- **Primary: recall@0.5.** Secondary: recall@0.3, mean best IoU, and pointing game (a model box
  centre inside the radiologist's box).
- **Union check:** IoU of the union of all the model's boxes with the radiologist's box (recall@0.3
  and @0.5). This penalises extra boxes. The number of boxes per case and the box area relative to
  the radiologist's box are also reported.

## Analysis
- Recall with Wilson 95% CIs per model and label.
- **G1 (primary, two-sided):** pooled recall@0.5, CheXagent minus MAIRA-2. 95% CI from 2,000
  bootstrap resamples of images (seed 20261024). Result: "CheXagent better" if the CI lies above 0,
  "MAIRA-2 better" if below, otherwise "no difference shown". Exact McNemar test on the discordant
  cases is also reported.
- **G2 (two-sided):** the same for pooled recall@0.3.
- Per label: the differences with McNemar exact p (descriptive, unadjusted). All secondary measures.

## Limitations recorded now
- NIH has one box per case, even when a finding is bilateral; a model boxing both sides may then
  get lower union IoU.
- NIH boxes were drawn for this dataset's labels; their extent conventions (e.g. cardiomegaly
  boxes around the heart) may suit one model's box style more than the other's.
- The model is told the condition is present, so this measures location, not detection.
- CheXagent was probably trained on public NIH images, possibly including these box annotations;
  MAIRA-2 was not.
- One dataset. VinDr (radiologist boxes, non-square images) only by dated amendment after a
  coordinate check.

All results reported whatever they are; changes only as dated amendments before the affected run.

Implementation: `grounding_recall.py` (stages `chexagent`, `maira`, `analyze`).

## Amendments
(none)
