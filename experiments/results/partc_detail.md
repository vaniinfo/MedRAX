# EXPLORATORY: Part C detail (per condition, edits, examples)

Computed on 2026-10-25 from committed readings. Combined-score AUCs match the pre-registered results; checker-alone AUCs are the exploratory baselines.

## 1. Base MedGemma, single image (1,500 NIH test images)

| condition | positives / n | report | + CheXagent | + MAIRA-2 | CheXagent alone | MAIRA-2 alone |
|---|---|---|---|---|---|---|
| cardiomegaly | 35 / 1500 | 0.554 | 0.934 | 0.706 | 0.936 | 0.707 |
| pleural effusion | 77 / 1500 | 0.758 | 0.883 | 0.792 | 0.885 | 0.723 |
| pneumothorax | 18 / 1500 | 0.527 | 0.894 | 0.563 | 0.894 | 0.532 |
| consolidation | 27 / 1500 | 0.510 | 0.861 | 0.759 | 0.866 | 0.764 |
| pulmonary edema | 10 / 1500 | 0.743 | 0.915 | 0.806 | 0.917 | 0.711 |
| atelectasis | 93 / 1500 | 0.566 | 0.879 | 0.781 | 0.894 | 0.780 |
| pneumonia | 9 / 1500 | 0.500 | 0.734 | 0.622 | 0.734 | 0.622 |
| infiltration | 185 / 1500 | 0.498 | 0.599 | 0.585 | 0.600 | 0.586 |
| emphysema | 14 / 1500 | 0.500 | 0.921 | 0.580 | 0.921 | 0.580 |
| nodule | 85 / 1500 | 0.514 | 0.801 | 0.686 | 0.802 | 0.687 |
| mass | 67 / 1500 | 0.514 | 0.934 | 0.852 | 0.935 | 0.852 |
| fibrosis | 24 / 1500 | 0.500 | 0.823 | 0.723 | 0.823 | 0.723 |
| pleural thickening | 30 / 1500 | 0.500 | 0.916 | 0.699 | 0.916 | 0.699 |

| condition | added, agrees | added, disagrees | removed, agrees | removed, disagrees |
|---|---|---|---|---|
| cardiomegaly | 14 | 43 | 0 | 0 |
| pleural effusion | 8 | 16 | 0 | 0 |

## 2. Base MedGemma, prior + current (499 NIH pairs)

| condition | positives / n | report | + CheXagent | + MAIRA-2 | CheXagent alone | MAIRA-2 alone |
|---|---|---|---|---|---|---|
| cardiomegaly | 8 / 499 | 0.620 | 0.932 | 0.609 | 0.936 | 0.608 |
| pleural effusion | 45 / 499 | 0.775 | 0.839 | 0.818 | 0.849 | 0.760 |
| pneumothorax | 16 / 499 | 0.499 | 0.890 | 0.689 | 0.891 | 0.689 |
| consolidation | 17 / 499 | 0.514 | 0.805 | 0.763 | 0.814 | 0.776 |
| pulmonary edema | 9 / 499 | 0.698 | 0.837 | 0.886 | 0.847 | 0.849 |
| atelectasis | 41 / 499 | 0.553 | 0.841 | 0.731 | 0.863 | 0.730 |
| pneumonia | 4 / 499 | 0.500 | 0.603 | 0.467 | 0.603 | 0.467 |
| infiltration | 61 / 499 | 0.507 | 0.639 | 0.580 | 0.637 | 0.574 |
| emphysema | 11 / 499 | 0.500 | 0.939 | 0.476 | 0.939 | 0.476 |
| nodule | 31 / 499 | 0.513 | 0.794 | 0.660 | 0.794 | 0.661 |
| mass | 36 / 499 | 0.541 | 0.884 | 0.735 | 0.884 | 0.728 |
| fibrosis | 8 / 499 | 0.500 | 0.868 | 0.616 | 0.868 | 0.616 |
| pleural thickening | 16 / 499 | 0.500 | 0.885 | 0.608 | 0.885 | 0.608 |

| condition | added, agrees | added, disagrees | removed, agrees | removed, disagrees |
|---|---|---|---|---|
| cardiomegaly | 2 | 9 | 0 | 0 |
| pleural effusion | 2 | 3 | 0 | 0 |

## 3. Trained MedGemma, prior + current (499 NIH pairs)

| condition | positives / n | report | + CheXagent | + MAIRA-2 | CheXagent alone | MAIRA-2 alone |
|---|---|---|---|---|---|---|
| cardiomegaly | 8 / 499 | 0.706 | 0.913 | 0.717 | 0.936 | 0.608 |
| pleural effusion | 45 / 499 | 0.818 | 0.850 | 0.834 | 0.849 | 0.760 |
| pneumothorax | 16 / 499 | 0.586 | 0.884 | 0.783 | 0.891 | 0.689 |
| consolidation | 17 / 499 | 0.669 | 0.832 | 0.783 | 0.814 | 0.776 |
| pulmonary edema | 9 / 499 | 0.847 | 0.851 | 0.901 | 0.847 | 0.849 |
| atelectasis | 41 / 499 | 0.665 | 0.845 | 0.749 | 0.863 | 0.730 |
| pneumonia | 4 / 499 | 0.719 | 0.667 | 0.605 | 0.603 | 0.467 |
| infiltration | 61 / 499 | 0.528 | 0.631 | 0.574 | 0.637 | 0.574 |
| emphysema | 11 / 499 | 0.500 | 0.939 | 0.476 | 0.939 | 0.476 |
| nodule | 31 / 499 | 0.496 | 0.785 | 0.649 | 0.794 | 0.661 |
| mass | 36 / 499 | 0.578 | 0.881 | 0.786 | 0.884 | 0.728 |
| fibrosis | 8 / 499 | 0.493 | 0.863 | 0.614 | 0.868 | 0.616 |
| pleural thickening | 16 / 499 | 0.499 | 0.884 | 0.605 | 0.885 | 0.608 |

| condition | added, agrees | added, disagrees | removed, agrees | removed, disagrees |
|---|---|---|---|---|
| cardiomegaly | 0 | 3 | 1 | 0 |
| pleural effusion | 0 | 4 | 0 | 0 |

## Examples (trained model; first by file name in each category)

- **stated, CheXagent supports, label present:** 00000013_033.png, pneumothorax, report score 0.5, CheXagent p_present 0.948 (FOUND), NIH label 1
- **not stated, CheXagent supports, label present (a catch):** 00003574_009.png, pneumothorax, report score 0.0, CheXagent p_present 0.945 (FOUND), NIH label 1
- **not stated, CheXagent supports, label absent (a false alarm):** 00000116_013.png, pneumothorax, report score 0.0, CheXagent p_present 0.787 (FOUND), NIH label 0
- **stated, CheXagent against, label absent (a correct removal):** 00013476_007.png, cardiomegaly, report score 1.0, CheXagent p_present 0.006 (AGAINST), NIH label 0
