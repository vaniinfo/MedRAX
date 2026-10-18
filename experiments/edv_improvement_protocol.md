# Does EDV improve condition detection in MedGemma reports? Protocol

Date: 2026-10-18. Written and committed before any MedGemma report below is generated.

## Question
Does adding EDV (an independent image check) to a MedGemma-generated report improve how well the
report identifies the conditions present, against labelled ground truth? And is the improvement
larger with CheXagent than with MAIRA-2 as the EDV model?

## Data
NIH ChestX-ray14, the test split of the existing random sample (1,500 images,
`split_nih_70_30.json`), with NIH labels for 13 conditions: cardiomegaly, pleural effusion,
pneumothorax, consolidation, pulmonary edema, atelectasis, pneumonia, infiltration, emphysema,
nodule, mass, fibrosis, pleural thickening. NIH labels are NLP-mined from reports (noisy).

## Report generator
Base MedGemma 1.5 4B (`medgemma1.5:4b`, Ollama), not the team's retrained model: the claim is
about MedGemma-generated reports in general. Current image only (shrunk to fit 1024 px), the
project's report prompt (`medrax/docs/medgemma_report_prompt.txt`), temperature 0, 450 tokens,
one report per image, generated once and kept.

## Reading the reports
Each report is read for the 13 conditions with the quote-checked lookup
(`comparison_edv_pdf.ground_truth_locations`, local Qwen; an answer whose quoted evidence is not in
the report counts as not stated). MedGemma score per image and condition: present 1, possible
0.5, not stated 0.

## EDV scores (already recorded, unchanged)
- CheXagent: p_present ("Is there X?"; `ppresent_nih.jsonl` for the six, `provisional_readings.jsonl`
  for the seven).
- MAIRA-2: sentence-likelihood score (`maira_score_nih.jsonl`, `maira_score7_nih_test.jsonl`).

## Systems (combination fixed now)
1. **MedGemma alone**: the MedGemma score.
2. **MedGemma + EDV (CheXagent)**: 0.5 x MedGemma score + 0.5 x CheXagent percentile.
3. **MedGemma + EDV (MAIRA-2)**: 0.5 x MedGemma score + 0.5 x MAIRA-2 percentile.
An EDV percentile is the model's score converted to its percentile among that model's scores for
the same condition on the NIH **development** split (3,500 images; no test-split information), so
both EDV models enter on the same scale.

## Primary analysis
- Per condition: AUC of each system. Primary measure: the mean AUC over the conditions with at
  least 10 positives in the test split (conditions with fewer are reported, not averaged).
- Paired 95% CIs from 2,000 image-level bootstrap resamples (seed 20261018).
- **H1**: mean AUC (MedGemma + CheXagent) minus mean AUC (MedGemma alone) > 0. Supported if the 95%
  CI lies above 0.
- **H2**: mean AUC (MedGemma + CheXagent) minus mean AUC (MedGemma + MAIRA-2) > 0. Same criterion.
- Also reported: MedGemma + MAIRA-2 vs MedGemma alone, and every per-condition difference.

## Secondary analysis: the [Image check] rule as yes/no decisions
MedGemma alone says present if its score is 1 or 0.5. MedGemma + CheXagent applies
`chexagent_revision_rule.md` without its side check (NIH has no side labels): a condition the
report states is removed if CheXagent's decision is AGAINST; pleural effusion or cardiomegaly is
added if CheXagent's decision is FOUND. Reported per condition and overall: sensitivity,
specificity and balanced accuracy, with paired 95% bootstrap CIs. (MAIRA-2 has no cut-offs, so
it has no rule-based version.)

## Known limitations, recorded now
- MedGemma's score has only three values, which caps its AUC; combining it with any continuous
  score can raise AUC. The secondary yes/no analysis does not have this property.
- The standalone EDV results on these images are already known (CheXagent ahead of MAIRA-2);
  MedGemma's reports on them are not.
- CheXagent was probably trained on public NIH images; MAIRA-2 was not.
- The team's retrained model may detect differently from base MedGemma; a later confirmation with
  it on a subset of these images is planned.
- All results reported whatever they are; any change goes in a dated amendment before it is run.

## Amendments
(none)
