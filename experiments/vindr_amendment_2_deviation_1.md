# Amendment 2, deviation 1: source of the image IDs for the sample

Date: 2026-10-04. Written and committed **before `train.csv` or any image was downloaded**.

## What Amendment 2 said
The 3,000 image IDs are taken from the competition's file list (file names only) and the sample
is committed before `train.csv` is downloaded.

## What happened
With the competition rules accepted, Kaggle's file-list API returns no files for
`vinbigdata-chest-xray-abnormalities-detection` (it works for other competitions, e.g. Titanic),
while single-file downloads work (checked with `sample_submission.csv`, which contains test
image IDs only, no labels). The training image IDs are therefore not available from a file list.

## Deviation
- The training image IDs are read from `train.csv`, **`image_id` column only**: the code reads
  that column (`usecols=["image_id"]`), takes the unique IDs, applies the unchanged sampling rule
  (lowest SHA-256 of `"20261004:" + image_id`, first 3,000) and writes the sample list. It does
  not print, summarise or use any label column.
- The sample list (`experiments/results/vindr_kaggle_sample.json`) is committed before any label
  column of `train.csv` is read by the `derive` step or by anyone.
- Nothing else in Amendment 2, Amendment 1 or Part A changes.

## Why this keeps the sample label-independent
The rule depends only on the IDs and a fixed seed, so the same 3,000 images are selected
whatever the labels are; reading the ID column cannot change the selection.
