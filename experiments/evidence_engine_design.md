# Evidence Engine: design and pre-registered validation (D1/D2)

Frozen before D1/D2 were run. Builds on the p_present validation (tag
`ppresent-validation-v1`). Nothing in the production pipeline changes until D1/D2 are
reported.

## Question the engine answers
MedGemma states that finding *f* is present. CheXagent's independent reading of the image
produces p_present = p. The Evidence Engine answers:

> How strongly does CheXagent's independent reading change the odds of this finding?

It is not a confidence score and not a probability that the patient has *f*. It replaces
both the `p_present ≥ 0.5 → visually supported` rule and the LLM-assigned High/Medium/Low.

## Evidence strength: likelihood ratio
$$\text{posterior odds}(f) = \text{prior odds}(f) \times \mathrm{LR}_f(p), \qquad
\mathrm{LR}_f(p) = \frac{P(p \mid f)}{P(p \mid \neg f)}$$

The LR does not depend on prevalence; the prior does. With a Platt calibration fitted at
development prevalence $\pi_f$, $\operatorname{logit} q_f(p) = a_f \operatorname{logit}(p) + b_f$:
$$\mathrm{LR}_f(p) = \frac{\operatorname{odds}(q_f(p))}{\operatorname{odds}(\pi_f)}
= \frac{e^{a_f \operatorname{logit}(p) + b_f}}{\pi_f / (1 - \pi_f)}$$

- **Engine LR**: Platt-derived (smooth in p). Empirical binned LRs are used only to check it.
- **LR uncertainty**: Platt refitted on 1,000 bootstrap resamples of the source development
  set (resampled prevalence included); LR(p) 95% CI = percentiles over the refits.
- **Support limit**: the engine does not extrapolate. If fewer than 10 development scores
  (of the source set) lie at or beyond p in that tail (≥ p for p above the development
  median, ≤ p below it), the case is outside support.

## States (primary rule)
| State | Rule | Meaning |
|---|---|---|
| SUPPORTING | lower 95% CI of LR(p) ≥ 5 | CheXagent is shown to multiply the prior odds by ≥ 5 |
| REFUTING | upper 95% CI of LR(p) ≤ 0.2 | CheXagent is shown to divide the prior odds by ≥ 5 |
| UNCERTAIN | otherwise | a ≥ 5-fold change in either direction has not been demonstrated; not evidence against the finding |
| INSUFFICIENT DATA | fewer than 20 development positives for *f*, or p outside support | evidence strength cannot be estimated reliably |

SUPPORTING and REFUTING mean the required evidence strength has been **demonstrated**, not
merely estimated. Pre-registered sensitivity analyses: 2 / 0.5 and 10 / 0.1 (reported, not
production configurations). If most cases are UNCERTAIN under 5 / 0.2, that is reported as
a result; the primary definition is not weakened.

Localization (boxes, "not localized") is reported alongside and never changes the state.

## Output
```
Finding: pleural effusion
CheXagent evidence: SUPPORTING
Likelihood ratio: 7.4 [95% CI 5.6–9.8]
Localization: patient's right, lower zone
Population probability: not reported (deployment prevalence not configured)
```
A post-test probability is shown only when a deployment prevalence π is explicitly
configured: posterior odds = odds(π) × LR, e.g. "For a population with an estimated 5%
prevalence, this evidence corresponds to a post-test probability of about 28%."

## D1: how much of the cross-dataset calibration shift is prevalence shift?
On the NIH test set (random part, natural prevalence, n = 1,500), compare probabilities from:

| Model | Evidence relationship | Prevalence |
|---|---|---|
| B | Indiana Platt | Indiana (implicit) |
| B′ | Indiana LR (slope and intercept) | NIH development prevalence (aggregate count only) |
| C | NIH Platt | NIH |

B′: posterior odds = odds(π_NIH-dev) × LR_Indiana(p). This transfers only if the
class-conditional score distributions $P(p \mid f)$ and $P(p \mid \neg f)$ are similar across
datasets; view mix, severity, equipment and label definitions can violate that. D1 therefore
tests *to what extent cross-dataset calibration shift is explained by prevalence shift
alone*. Reported per finding, with 2,000-sample bootstrap CIs: Brier for raw, B, B′, C; the
decomposition B − B′ (explained by prevalence shift) and B′ − C (residual: change in the
evidence relationship); the share explained (B − B′)/(B − C); calibration-in-the-large and
calibration slope.

## D2: does an LR learned on one dataset keep its meaning on another?
State boundaries are derived from each source (Indiana development: transfer; NIH
development, random part: local) and applied to the NIH test set.
- **Primary**: NIH test, random part. **Secondary**: NIH test, random + supplement (LR is
  prevalence-independent; more positives for rare findings).
- For each finding, source and state: empirical LR = P(state | f) / P(state | ¬f) with
  2,000-sample image-level bootstrap CI, and coverage (share of cases per state).
- **Criterion**: SUPPORTING holds if its empirical LR is ≥ 5 (point estimate) and is
  *demonstrated* if its CI lower bound is ≥ 5; REFUTING analogously with ≤ 0.2 and the CI
  upper bound.
- **Platt check**: empirical LR per score band (with CI) vs mean Platt LR in the band, on
  the NIH test set (random + supplement), for Indiana- and NIH-derived LR functions.
- Sensitivity analyses: coverage and empirical state LRs under 2 / 0.5 and 10 / 0.1.

All inputs are committed readings and frozen parameters (SHA-256 verified); no model is
rerun and no frozen artifact is modified.
