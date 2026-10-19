# Longitudinal EDV: do prior + current images change the picture? Protocol

Date: 2026-10-19. Written and committed before any reading below. Extends
`edv_improvement_protocol.md` (single-image) to the setting where every model sees the prior and
the current X-ray, as a radiologist would.

## Data
NIH test split images (1,500) that have an earlier image of the same patient (lower Follow-up #
in `Data_Entry_2017_v2020.csv`): **499 pairs**. Prior = that patient's most recent earlier image in
the same view (PA/AP) if one exists, else the most recent earlier image. Labels: NIH labels of the
current image, 13 conditions. Positives among the 499: effusion 45, atelectasis 41, infiltration 61,
mass 36, nodule 31, consolidation 17, pneumothorax 16, pleural thickening 16, emphysema 11;
fewer than 10: cardiomegaly 8, edema 9, fibrosis 8, pneumonia 4 (reported, not averaged).

## Models (longitudinal versions)
- **MedGemma 1.5** (base, Ollama): prior + current images, the pilot's longitudinal instruction
  ("The first image is the PRIOR study and the second is the CURRENT study; report on the CURRENT
  study and note change from the prior.") followed by the project report prompt; 1024 px, temperature
  0, 450 tokens. Read for the 13 conditions with the same quote-checked lookup and amendment A1
  rule as the single-image experiment. Score present 1 / possible 0.5 / not stated 0.
- **CheXagent, option A (primary): native two-image.** Images [prior, current] and "The first image
  is the prior chest X-ray and the second is the current chest X-ray. Is there {condition} in the
  current chest X-ray?"; p_present = Yes share of the first token (8 tokens, greedy). Recorded now:
  two images plus this question are about 2,125 input tokens, above CheXagent's 2,048 design length.
- **CheXagent, option B (check): two single-image readings.** The validated query "Is there
  {condition}?" on the current image (existing readings) and on the prior image (new readings);
  score = mean of the two p_present values.
- **MAIRA-2:** the same sentence-likelihood score as before (six + seven sentence pairs), with the
  prior image given through its built-in prior-frontal input.

## Systems and combination
EDV score -> percentile among that model's scores for the same condition on the 499 pairs (label-free;
no development set exists for longitudinal readings). Combined = 0.5 x MedGemma score + 0.5 x EDV
percentile. Longitudinal systems: MedGemma-L alone; + CheXagent A; + CheXagent B; + MAIRA-2-L.
Single-image reference on the same 499 (existing readings, same percentile rule): MedGemma-S alone,
+ CheXagent-S, + MAIRA-2-S.

## Analysis
Mean AUC over the 9 conditions with at least 10 positives; paired 95% CIs from 2,000 image-level
bootstrap resamples (seed 20261019).
- **H1-L (primary):** MedGemma-L + CheXagent A minus MedGemma-L alone > 0.
- **H2-L:** MedGemma-L + CheXagent A minus MedGemma-L + MAIRA-2-L > 0.
- Supported if the 95% CI lies above 0.
- Exploratory, reported as such: CheXagent A alone vs CheXagent single-image alone (does two-image
  input degrade or help CheXagent?); CheXagent B alone vs single-image alone; MedGemma-L vs MedGemma-S;
  MedGemma-L + CheXagent A vs MedGemma-S + CheXagent-S; MAIRA-2-L vs MAIRA-2-S. Per-condition AUCs.
- CheXagent A answers that do not start with Yes/No have no score; their share is reported, and
  those image-condition pairs are excluded for every system in that condition.

## Limitations recorded now
499 images (wide intervals; 4 conditions too few to average); NIH labels are NLP-mined; base
MedGemma, not the team's model; CheXagent option A beyond its design length; CheXagent probably saw
NIH images in training; percentiles from the evaluation set itself (label-free). All results
reported; changes only as dated amendments before the affected run.

## Amendments
(none)
