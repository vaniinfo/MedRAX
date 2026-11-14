# Report reader validation: manual review of a random sample. Protocol

Date: 2026-10-25. Written before any manual annotation. The reader's outputs already exist and were
used in E4 (`edv_improvement.md`, `edv_longitudinal.md`, `edv_trained.md`); this is a validation of
those outputs, requested in internal review.

## Why
E4 converts each MedGemma report to 13 condition scores (1 present, 0.5 possible, 0 not stated) with
the quote-checked Qwen reader (`comparison_edv_pdf.ground_truth_locations`, Ollama
`qwen3.8-edv:latest`, digest 67f762aeb910, temperature 0, amendment A1 rule). The quote check stops
invented statements, but a real quote can still be misread (a negated, hedged or prior-only finding
read as currently present), and a stated finding can be missed. Reader errors lower the report-only
AUCs and so affect the interpretation of score fusion.

## Sample (fixed before annotation)
- 30 reports from each E4 setting, drawn with `random.Random(20261027)` from the sorted list of
  report IDs: (a) base MedGemma, single image (`edv_medgemma_reports.jsonl`, 1,500 test images);
  (b) base MedGemma, prior + current (`edv_long_medgemma_reports.jsonl`, 499); (c) trained MedGemma
  (`edv_trained_reports.jsonl`, 499). 90 reports, 1,170 report-condition decisions.
- Sampling is independent of the reader's output, the labels and the checkers' scores.

## Annotation (blind to the reader)
The sheet (`reader_validation.py sample`) shows each report's text and the 13 condition names in a
shuffled report order, without the setting, the image, the NIH labels or the reader's answers.
Annotators code each condition, for the **current** study as stated in the report:
- **P** present; **U** possible / hedged (may, possible, could, suggest, cannot exclude);
- **N** not mentioned or negated;
- **R** mentioned only as prior, resolved, historical or about the prior image.

Rules given to the annotator are the reader's own rules (synonyms such as "layering fluid" = pleural
effusion; vascular congestion is not pulmonary edema; scarring is not atelectasis; subcutaneous
emphysema is not emphysema; side is ignored). At least one annotator with radiology training; a
second annotator if available, coding independently.

## Measures
Annotator code mapped to the reader's scale: P -> 1, U -> 0.5, N and R -> 0.
- **Primary:** exact agreement of the reader's score with the annotator's (all 1,170 decisions),
  with a 95% CI from a bootstrap over reports (2,000 resamples, seed 20261027); linearly weighted
  Cohen's kappa.
- Sensitivity and specificity of "reader states the condition" (score > 0) against the annotator
  (P or U).
- **Error types**, counted: reader present/possible where the annotator coded N (negated or not
  mentioned; the annotator's notes separate the two); where the annotator coded R (temporal error); reader 1 vs annotator U,
  or reader 0.5 vs annotator P (uncertainty error); reader 0 where annotator P or U (missed).
- All measures per setting and pooled. With two annotators: inter-annotator weighted kappa, and
  their consensus (discussion) as the reference.

No threshold for "acceptable" is set; results are reported as they are and discussed as a
limitation of E4. The E4 results are not recomputed or changed on the basis of this validation.

Implementation: `reader_validation.py` (stages `sample`, `analyze`).
