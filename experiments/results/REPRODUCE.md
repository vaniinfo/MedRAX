# Reproducing the p_present validation (tag `ppresent-validation-v1`)

Question: is CheXagent's p_present (Yes/(Yes+No) over the first answer token to
"Is there {finding}?") a usable evidence score, and what does its value mean?

## Chain of experiments

| Step | Script | Output | Frozen before the next step |
|---|---|---|---|
| 1. Indiana discovery (544 images, 6 findings) | `ppresent_indiana.py run` / `analyze` | `ppresent_indiana.jsonl`, `ppresent_indiana.md` | — |
| 2. Indiana held-out (70/30 by image) | `ppresent_split.py` | `ppresent_split.md` | `split_indiana_70_30.json`, `frozen_dev_params.json` (read-only; SHA-256 in `frozen_dev_params.sha256`, written before any NIH data was read) |
| 3. Consolidation investigation (exploratory) | — | `consolidation_investigation.md` (prediction for NIH recorded before NIH results) | — |
| 4. NIH external validation, A/B/C | `ppresent_nih.py sample` → `supplement` → `run` → `analyze` | `ppresent_nih.jsonl`, `ppresent_nih.md`, `ppresent_nih_run.log` | `nih_sample_5000.json`, `nih_supplement.json` (drawn before any NIH reading), `split_nih_70_30.json`, `nih_frozen_dev_params.json` (fitted on NIH dev, frozen before NIH test scoring) |

`ppresent_nih.py analyze` verifies `frozen_dev_params.json` against its SHA-256 before
applying it (experiment B) and refuses to split/freeze on incomplete readings.

## Code
- This repository at tag `ppresent-validation-v1`; the scoring code
  (`medrax/tools/xray_vqa.py`: `_generate_response`, `_yes_no_confidence`, bf16 loading) is
  in commit `960af45`.
- Ground truth for Indiana: `compare_indiana.truth_for` and `edv/findings.FINDINGS` from
  `C:\Users\hari\_workarea_\xray\cxr-edv` at commit `cdb3a773d35cd03b99728dd175e516fff4b5d2a1`
  (imported, not copied).

## Model and environment
- CheXagent: `StanfordAIMI/CheXagent-2-3b`, snapshot `8f19b53a2eceda4c33b0acec6c81fbc293ad80d0`,
  loaded in bfloat16 with `_supports_cache_class = False` (so the image is encoded).
- Python 3.11.8, torch 2.6.0+cu124, transformers 4.46.0.dev0
  (`git+https://github.com/huggingface/transformers.git@88d960937c81a32bfb63356a2e8ecf7999619681`),
  scikit-learn 1.9.1, numpy 1.26.4; NVIDIA GeForce RTX 3090; Windows 11.
- Decoding: greedy, `max_new_tokens=8`; seeds: 20260927 (splits, NIH sample, bootstrap),
  20260928 (NIH supplement).

## Data (not in the repository)
| Dataset | Location | Labels file SHA-256 |
|---|---|---|
| Indiana (Open-i) eval, 544 frontal images | `C:\Users\hari\_workarea_\MedRAX\data\indiana_eval` | `GROUND_TRUTH.csv`: `bf9c823465fd99a6464f703c72d6e236015430587a00e7db0a3bf413ca69d553` |
| NIH ChestX-ray14 (110,634 of 112,120 images on disk) | `D:\datasets\chestxray14` | `Data_Entry_2017_v2020.csv`: `dc1d2df67fdc1c5a7601d48699cda2b13dc2c4841488b4183dcf04884dbaca11` |

## Artifact hashes (SHA-256)
```
5ddccb37affadd778c4e848a53ef74370d656556ecbb4eae440c9837c4d60aa0  ppresent_indiana.jsonl
3c14b9441b81095b68b8f896ee3457ddeb7a995efe4e2882ab77d9255c989d14  frozen_dev_params.json
8148ad9c4670b27075398477e88508bd675f25f44ec8ca86340b99fc37e0e4b7  split_indiana_70_30.json
4894e3add07bfc49eb9cf84e54df791e8257939d16b172ef5d34646a593c9387  nih_sample_5000.json
9d8719c6d5f685a343e202c7fdbee1e4598ed6fd80b22474d25d4e1e57825966  nih_supplement.json
8344d22b3145e0855e3e4be193f368b20f523c500e643062131408acb7d312c8  ppresent_nih.jsonl
79e83a88c7bd447f548d306eea079d64e74fb948e5d2afcc39c8145ba032e41b  split_nih_70_30.json
e256455ec871cbb7d12806ae6a14c8f077942b9fbfe120ac3574869a43cac776  nih_frozen_dev_params.json
```

## Summary of results
- Discrimination generalizes: AUC 0.80–0.96 on the Indiana hold-out, 0.84–0.94 on NIH.
- Raw p_present is not a probability; Platt calibration fitted on Indiana dev improved
  calibration on the Indiana hold-out for 5/6 findings.
- Calibration is population-dependent: frozen Indiana calibration beats raw on NIH but
  over-predicts at NIH's natural prevalence; NIH-fitted calibration is better for all six
  findings (paired Brier difference CI excludes 0 for 5/6).
- A universal 0.50 threshold under-detects pneumothorax, consolidation and pulmonary edema
  on both datasets; lower, pathology-specific thresholds shift the operating point (the
  direction transfers across datasets, the exact values do not). Superiority by Youden's J
  was not established on the Indiana hold-out.
- Consolidation's calibration failure on Indiana traces to label heterogeneity ("Airspace
  Disease" vs "Consolidation", unmapped "Infiltrate") and a dev/test case-mix difference.
- AP films are harder than PA for pneumothorax and atelectasis.

Caveats: Indiana is enriched by selection; NIH labels are NLP-mined; the NIH supplement is
enriched and used only for prevalence-independent metrics.
