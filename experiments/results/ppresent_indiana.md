# CheXagent p_present vs ground truth — Indiana eval set

544 frontal images, 3264 readings. Query: `Is there {finding}?`; p_present = Yes/(Yes+No) over the first answer token. Truth: cxr-edv `truth_for` (MeSH problem terms).

**Enriched corpus** (selected per finding): AUC and sensitivity/specificity transfer; the observed positive rate per band is optimistic for a real clinic.

## Discrimination

| Finding | Pos / Neg | AUC [95% CI] | Sens @0.5 | Spec @0.5 | Best threshold (Youden) | Sens / Spec there |
|---|---|---|---|---|---|---|
| cardiomegaly | 164 / 380 | 0.927 [0.904–0.947] | 0.87 | 0.83 | 0.44 | 0.91 / 0.80 |
| pleural effusion | 120 / 424 | 0.953 [0.934–0.969] | 0.86 | 0.88 | 0.55 | 0.86 / 0.89 |
| pneumothorax | 28 / 516 | 0.961 [0.914–0.991] | 0.64 | 1.00 | 0.09 | 0.93 / 0.90 |
| consolidation | 109 / 435 | 0.797 [0.752–0.839] | 0.47 | 0.91 | 0.17 | 0.79 / 0.67 |
| pulmonary edema | 45 / 499 | 0.894 [0.832–0.944] | 0.69 | 0.93 | 0.36 | 0.80 / 0.88 |
| atelectasis | 161 / 383 | 0.802 [0.766–0.836] | 0.78 | 0.67 | 0.45 | 0.83 / 0.65 |

## What a p_present value is worth (observed positive rate per band)

| Finding | 0.0–0.1 | 0.1–0.3 | 0.3–0.5 | 0.5–0.7 | 0.7–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|
| cardiomegaly | 1% (n=240) | 16% (n=50) | 23% (n=48) | 46% (n=52) | 64% (n=67) | 86% (n=87) |
| pleural effusion | 0% (n=272) | 14% (n=80) | 14% (n=37) | 28% (n=39) | 52% (n=40) | 93% (n=76) |
| pneumothorax | 1% (n=477) | 10% (n=42) | 40% (n=5) | 50% (n=2) | 75% (n=4) | 100% (n=14) |
| consolidation | 6% (n=257) | 21% (n=132) | 22% (n=64) | 44% (n=52) | 63% (n=30) | 100% (n=9) |
| pulmonary edema | 1% (n=343) | 7% (n=90) | 12% (n=43) | 23% (n=31) | 63% (n=27) | 70% (n=10) |
| atelectasis | 3% (n=130) | 12% (n=73) | 24% (n=90) | 41% (n=116) | 51% (n=96) | 74% (n=39) |

Read across a row: if p_present were calibrated, the positive rate in each band would sit inside that band (e.g. ~80% in 0.7–0.9).
