# Pilot protocol: independent model comparison against the ground-truth reports (19 comparisons)

Date: 2026-10-11. Written before the runs it describes. A pilot: results are descriptive, and the
protocol is meant to be reused later on a larger, broader image set.

## Status and honesty note
- Exploratory runs on these same 19 comparisons were made on 2026-10-10/11 (independent CheXagent
  screen, each model's own report, MAIRA-2 variants; `logs/chexagent_screen_20261011-065557*`).
  Their results have been seen. This protocol fixes the method so the runs below are repeatable
  and reported in full, but it is not blind to those exploratory results.
- Any change after this file is committed is recorded as a dated amendment below, before the
  affected run.

## Data
- `D:/datasets/held_out_eval_images/subset_comparisons_clean_filtered.pdf`: 19 comparisons, each
  with a generated report (the team's retrained MedGemma) and a ground-truth report.
- Images: `D:/datasets/held_out_eval_images/held_out_eval_images/row_NN_*/{prior,current}.jpg`
  (full resolution), matched to each page's "Row N" (all 19 verified identical to the PDF images).
- Known properties, recorded now:
  - MIMIC-CXR-derived (subject IDs, "___" redactions). All models below were probably trained
    on MIMIC-CXR, so results may be optimistic for every model.
  - 5 rows' prior image is another row's current image (rows 3, 9, 11, 12, 18).
  - Pages 9/10 and 11/12 have identical ground-truth text but different current images
    (correlation 0.28 and 0.55): either two images of one study or a report duplicated onto the
    wrong row. Unresolved until the source study/image IDs are available. Rows 9-12 are flagged
    in the review sheet, and every result is reported with and without them.
  - The ground truth is free-text reports: a condition that is not mentioned is "not
    reported", not proven absent.

## Conditions
The 13 conditions with an Evidence Engine v1.0 or provisional cut-off: cardiomegaly, pleural
effusion, pneumothorax, consolidation, pulmonary edema, atelectasis, pneumonia, infiltration,
emphysema, nodule, mass, fibrosis, pleural thickening. Side: right, left, bilateral, not stated.

## Step 0: frozen ground-truth labels (before any model is scored)
1. For each page and condition, the ground-truth report is read with the existing quote-checked
   lookup (`comparison_edv_pdf.ground_truth_locations`, local Qwen, the quoted evidence must be
   in the report): present / possible / not reported, plus the side and the quoted sentence.
2. The result is exported as a review sheet (CSV). A radiologist reviews and corrects every row
   (they see only the ground-truth report, not model outputs).
3. The corrected sheet is frozen: a labels-only copy (page, condition, label, side; no report
   text) and the SHA-256 of the full reviewed sheet (kept in `logs/`, as it quotes MIMIC text)
   are committed. All experiments are scored against it only; the lookup is not re-run.
- Primary analysis: "possible" counts as present. Secondary: "possible" counts as not reported.

## Experiment 1: each model's own report
Each model writes a report without seeing the generated report or the ground truth.

| Model | Input | Setting |
|---|---|---|
| Team's MedGemma | (fixed) the generated report in the PDF | used as is; cannot be re-run |
| MedGemma 1.5 4B (Ollama `medgemma1.5:4b`) | prior + current | project report prompt (`medgemma_report_prompt.txt`) preceded by "The first image is the PRIOR study and the second is the CURRENT study; report on the CURRENT study and note change from the prior."; images shrunk to 1024 px; temperature 0; 450 tokens |
| CheXagent-2-3b | current | its six anatomical-area prompts (`chexagent_report_screen.AREAS`), 200 tokens each, greedy |
| MAIRA-2 | prior + current frontal | bf16, greedy, 300 tokens; no prior report, indication, technique or comparison |

- Each report is read for the 13 conditions with the same lookup as Step 0. The extracted
  labels are exported for the same radiologist review (they see the model report only) and
  frozen before scoring.
- Measures per model and condition: ground-truth conditions the report states (count /
  ground-truth total); conditions the report states that the ground truth does not report
  ("unconfirmed", not called false positives); side agreement where both give a side.

## Experiment 2: direct question per condition
- CheXagent: "Is there {condition}?" on the current image (8 tokens, greedy), p_present as in
  all Evidence Engine work. "Found" = at or above the support cut-off (engine v1.0 for the six,
  provisional for the seven; pneumonia and infiltration have none).
- MedGemma 1.5: the same question on the current image; Yes/No answer. A score is used only if
  Ollama returns token log-probabilities (checked before the run and recorded here as an
  amendment); otherwise Yes/No only.
- Measures: per condition, ground-truth conditions found / total and unconfirmed finds;
  for CheXagent, AUC of p_present pooled over all page-condition pairs and per condition where
  there are at least 3 ground-truth positives.

## Experiment 3: location without the side
- Cases: page-conditions the frozen ground truth reports on one side only.
- Each model is asked to locate the condition by name only (no side):
  - CheXagent: "Please locate the following phrase: {condition}";
  - MAIRA-2: phrase grounding (bf16);
  - MedGemma 1.5: its bounding-box output, if a working prompt format is confirmed on page 1
    before the run (recorded as an amendment; otherwise MedGemma 1.5 is not in Experiment 3).
- Side from box centres (`vindr_grounding_amendment.box_sides`). Measures: correct side /
  wrong side / both sides / not located.

## Experiment 4 (exploratory only)
Change since the prior (better / worse / unchanged), only where the frozen ground truth states a
change; MedGemma 1.5 and MAIRA-2 reports from Experiment 1. Reported separately as exploratory.

## Analysis and reporting rules
- Descriptive counts only; no significance tests and no ranking claims (19 images, about 30
  ground-truth findings).
- All results are reported whatever they are, per model and per condition, with the primary
  and secondary handling of "possible".
- No change of prompts, settings, conditions or cut-offs after seeing results; changes go
  through dated amendments before the affected run.
- Outputs quoting report text stay in `logs/` (not committed); the committed summary contains
  counts only.

## Amendments

**A1 (2026-10-11, before any experiment run): MedGemma 1.5 score in Experiment 2.** Ollama returns
token log-probabilities (checked on page 8, "Is there pleural effusion?"). Score = P(yes) /
(P(yes) + P(no)) over the top 20 first-token alternatives, where yes/no are tokens matching
`^\s*yes` / `^\s*no` (case-insensitive); 1.0 or 0.0 if only one of them is among the 20; no
score if neither is. MedGemma often starts "Based on…", so the answer text is recorded too.
"Found" for MedGemma = the answer starts with "Yes" (it has no calibrated cut-off); the score is
used for AUC only. Same request settings as Experiment 1 (current image only for this question,
1024 px, temperature 0).

**A2 (2026-10-11, before any experiment run): MedGemma 1.5 in Experiment 3.** Prompt format from
the MedGemma 1.5 technical report (Table 12 and section 3.2.4): system instruction "You are a
helpful radiology assistant."; user: "Where is the {condition}? Output a JSON list of objects,
each with a "label" and a 2D bounding box "box_2d" as [y0, x0, y1, x1], normalized to the range
[0, 1] with (0, 0) the top-left and (1, 1) the bottom-right corner." The bare "Where is the
{object}?" returned prose on page 1; with the instruction it returned parseable JSON (and [] for
pleural effusion, which page 1 does not have). The page-1 boxes looked anatomically implausible
(heart box in the upper half of the image; right lung box the whole image). MedGemma 1.5 stays
in Experiment 3 as planned; an empty list counts as not located.

**A3 (2026-10-11, before any experiment run): run order.** Experiments 2 and 3 are run for all 13
conditions on all 19 pages; Experiment 3's cases (one-sided conditions in the frozen ground
truth) are selected afterwards, so running does not wait for the radiologist review. Models run
one at a time to fit the GPU: MedGemma 1.5 (Ollama), then CheXagent, then MAIRA-2 (bf16). Raw
outputs go to `logs/pilot_runs_<time>/`. Model reports are read for the 13 conditions only
after all runs, and those labels go to the radiologist review before scoring. Note: MedGemma
1.5 via Ollama on the GPU was not exactly repeatable between runs in exploratory work; each
output is produced once and kept.
