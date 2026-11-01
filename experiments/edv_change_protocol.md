# Change check: does CheXagent's presence check on each image catch errors in MedGemma's change wording? Protocol

Date: 2026-11-01. DRAFT, to be committed before the analysis below is run. Extends
`edv_longitudinal_protocol.md` and `edv_trained_protocol.md` (same 499 NIH test pairs). Not part of
the radiologist scoring forms, which stay as they are.

## Question
The EDV check so far asks whether a condition is present on the current image. A longitudinal report
also says what changed ("new", "unchanged", "resolved"). This protocol tests a change check built only
from the validated presence check, with no new prompts and no new cut-offs: CheXagent reads the prior
and the current image **separately**, and the two verdicts are compared. CheXagent remains a checker;
it is not asked to describe change itself (option A, the native two-image query, is not used here).

## Data already recorded (disclosed before analysis)
- Pairs: `results/edv_long_pairs.json`, 499 NIH test pairs (425 with prior in the same view).
- CheXagent current-image readings: the single-image "Is there {condition}?" readings used throughout
  (CheXagent-S).
- CheXagent prior-image readings: `results/edv_long_chexagent_prior.jsonl`, same query, recorded
  2026-10-19 for option B. **Until now they were only used averaged with the current reading (option
  B AUCs in `edv_longitudinal.md` and `edv_trained.md`). No prior-image verdict, no change category,
  and no comparison with prior-image labels has been computed.**
- Trained MedGemma reports and presence labels: `results/edv_trained_reports.jsonl`,
  `results/edv_trained_labels.jsonl` (already read for presence; change wording not yet read).
- Feasibility counts, from NIH labels only (no model output looked at), over the 13 conditions:

  | reference change | prior label | current label | pair-conditions |
  |---|---|---|---|
  | new | no | yes | 212 |
  | resolved | yes | no | 222 |
  | persistent | yes | yes | 91 |

  Per condition at most 60 in any category (infiltration, resolved); most below 20. Hence the
  analyses pool the 13 conditions.

## Reference change
NIH labels of the prior image and of the current image (`Data_Entry_2017_v2020.csv`), per condition:
new / resolved / persistent / absent as in the table. NIH labels are text-mined per image, so a
change category carries the error of two labels; the high "resolved" count is itself a sign of label
noise (see limitations).

## CheXagent change verdicts (fixed now, no tuning)
Each reading gets its Evidence Engine v1.0 verdict through `chexagent_reliability.decide` (the same
cut-offs, provisional cut-offs and fallback as everywhere else): Supported (FOUND), Not supported
(AGAINST), or Indeterminate.

| prior verdict | current verdict | derived change |
|---|---|---|
| Not supported | Supported | New |
| Supported | Not supported | Resolved |
| Supported | Supported | Persistent |
| Not supported | Not supported | Absent |
| Indeterminate on either image | | Uncertain |

## Analysis 1 (primary): does the prior-image reading tell new from persistent?
Among pair-conditions whose current image is labelled positive (303: 212 new, 91 persistent), the
prior-image score should be higher for persistent than for new.
- Score: the prior-image p_present converted to its percentile among all 499 prior readings for that
  condition (label-free), so that the 13 conditions can be pooled.
- **H5:** pooled AUC (persistent vs new) > 0.5. 95% CI from 2,000 patient-level bootstrap resamples
  (pairs of the same patient resampled together), seed 20261101. Supported if the CI lies above 0.5.
- Reported with it: the 2x3 table of reference (new, persistent) by derived prior verdict (Not
  supported, Supported, Indeterminate).

## Analysis 2 (secondary): checking MedGemma's change wording
- **Reading.** For each condition the trained MedGemma report states (presence score 1 or 0.5), the
  local reader (`qwen3.8-edv:latest`, the same quote-checked lookup as the presence labels) assigns one
  change claim with a verbatim quote: **new**, **unchanged/stable**, **worse**, **better**,
  **resolved**, or **no change stated**. A claim whose quote is not found verbatim in the report
  becomes "no change stated".
- **What each claim implies about the prior image.** new: absent on prior; unchanged, worse, better:
  present on prior. resolved: present on prior, absent on current. Only these presence implications
  are checked; the NIH labels cannot tell worse from better.
- **Flag rule (fixed now).** A claim is flagged when CheXagent contradicts its implication: new with
  prior Supported; unchanged/worse/better with prior Not supported; resolved with current Supported.
  Indeterminate never flags.
- **Reported:** number of change claims of each kind; share flagged; the reference (label) error rate
  of flagged and of unflagged claims, and their difference with a 95% patient-level bootstrap CI (same
  seed). A claim is a label error when its prior-image implication disagrees with the prior label.
- **Feasibility gate.** If the reports contain fewer than 20 checkable change claims in total, Analysis
  2 is reported descriptively only (counts and examples, no interval). The trained model ran without
  the prior report it was trained with (placeholder "Not available."), so few change claims are
  possible.

## Exploratory (labelled as such)
- Analysis 1 restricted to the 425 same-view pairs.
- Resolved vs absent among current-negative pair-conditions (prior-image percentile score), AUC.
- Per-condition counts and AUCs where a category has at least 10 cases.
- Agreement of derived change with reference change, all four categories plus Uncertain, as a table.

## Multiplicity
One primary hypothesis (H5). Analysis 2 and the exploratory results are reported with unadjusted
intervals and are not used as confirmatory evidence.

## Limitations recorded now
- NIH labels are text-mined per image; change is inferred from two labels, and label noise across
  follow-ups is visible (222 "resolved"). No reference says worse or better.
- Prior and current images can differ in view (74 pairs) and in technique, which changes CheXagent's
  readings without any disease change.
- CheXagent probably saw NIH images in training. The cut-offs were set for presence, not change.
- The prior-image readings were generated before this protocol (for option B); see the disclosure
  above.
- The trained MedGemma sees the prior image but not the prior report, so its change wording is out of
  its training setting.
- Pooling 13 conditions with percentiles assumes their scores rank comparably; per-condition results
  are too small to confirm.

All results are reported whatever they are; changes only as dated amendments before the affected run.

Implementation (to be written after this protocol is committed): `edv_change.py` with stages `read`
(change claims, main environment, Ollama) and `analyze`.

## Amendments
(none)
