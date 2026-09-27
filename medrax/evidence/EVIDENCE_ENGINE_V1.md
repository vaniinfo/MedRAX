# Evidence Engine v1.0 — specification

Identifier: `evidence-engine-v1.0`. Parameters: `evidence_engine_v1.0.json`. Checksums:
`evidence_engine_v1.0.sha256`. This specification, the parameter bundle and the engine
implementation are frozen together; any change produces a new version.

## Question answered
> How strongly does CheXagent's independent reading of the image change the odds of this
> finding?

The evidence state is not a confidence score and not a probability that the patient has
the finding.

## Scope
- **Findings**: cardiomegaly, pleural effusion, pneumothorax, consolidation, pulmonary
  edema, atelectasis. Any other finding: INSUFFICIENT DATA, reason `FINDING_NOT_VALIDATED`.
- **Input**: CheXagent p_present, i.e. Yes / (Yes + No) over the first answer token of
  `Is there {finding}?` (StanfordAIMI/CheXagent-2-3b, snapshot
  `8f19b53a2eceda4c33b0acec6c81fbc293ad80d0`, bfloat16, image encoding enabled). p_present is
  computed for every finding assessed, whether or not it was localized.

## Evidence transformation
Per finding, Platt calibration fitted at development prevalence π gives
$\operatorname{logit} q = a \operatorname{logit}(p) + b$ and

$$\mathrm{LR}(p) = \frac{\operatorname{odds}(q(p))}{\operatorname{odds}(\pi)}$$

The 95% CI of LR(p) comes from 1,000 Platt refits on bootstrap resamples of the development
set, stored on a fixed logit(p) grid. The engine loads these; it never refits.

## States (primary rule 5 / 0.2)
| State | Rule |
|---|---|
| SUPPORTING | lower 95% CI of the Platt LR ≥ 5 |
| REFUTING | upper 95% CI of the Platt LR ≤ 0.2 **and** upper 95% CI of the empirical-band LR ≤ 0.2 |
| UNCERTAIN | all remaining supported cases |
| INSUFFICIENT DATA | the validation-support requirements are not met |

| Reason | State |
|---|---|
| `SUPPORTING`, `REFUTING` | the criterion is met |
| `WEAK_EVIDENCE` | UNCERTAIN: neither criterion met by the Platt LR |
| `REFUTING_NOT_EMPIRICALLY_CONFIRMED` | UNCERTAIN: the Platt LR meets REFUTING, the empirical band does not confirm it |
| `OUTSIDE_SUPPORT` | INSUFFICIENT DATA: fewer than 10 development scores at or beyond p in its tail |
| `FEW_DEV_POSITIVES` | INSUFFICIENT DATA: fewer than 20 development positives for the finding |
| `FINDING_NOT_VALIDATED` | INSUFFICIENT DATA: the finding is outside the validated scope |

Empirical-band LR: per fixed band (edges 0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1),
$(k_\text{pos}/N_\text{pos})/(k_\text{neg}/N_\text{neg})$ with an exact Clopper–Pearson
upper bound. It is a decision-rule check only and never replaces the reported LR. No
finding is special-cased: a finding without an empirically confirmed low-score band
(consolidation in v1.0) has no REFUTING region as a consequence of the frozen data.

Sensitivity configurations 2 / 0.5 and 10 / 0.1 are stored for analysis; the production
rule is 5 / 0.2.

## Parameter sources
| Engine | Platt LR | Empirical bands | Role |
|---|---|---|---|
| **NIH (primary)** | NIH ChestX-ray14 development, random part | NIH development, random + supplement | production/default |
| Indiana (comparator) | Indiana (Open-i) development split | NIH development, random + supplement | cross-dataset comparator only |

## Output
State, reason, LR with 95% CI, and p_present. Localization is reported separately and
never modifies the state. A post-test probability is computed only when a deployment
prevalence π is supplied: posterior odds = odds(π) × LR; it is not part of the evidence
state.

## Validation status and data provenance
- Indiana: discovery and held-out validation. NIH ChestX-ray14: external validation of
  p_present (A/B/C), then D1/D2, then D3, which used NIH **development** data to derive the
  empirical REFUTING check. NIH is therefore part of the engine's development and is not an
  external validation of the finished engine.
- A third dataset is reserved for external validation of this version, run unchanged: no
  threshold, calibration or band fitting, no state redesign, no finding-specific
  exceptions. Both the NIH engine and the Indiana comparator are reported.
- Results so far: `experiments/results/ppresent_*.md`, `evidence_engine_d1_d2.md`,
  `evidence_engine_d3.md`.
