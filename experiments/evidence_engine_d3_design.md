# Evidence Engine D3: dual-evidence REFUTING rule (pre-registered)

Addendum to `evidence_engine_design.md` (commit `47c6b64`), frozen before D3 was run.
Motivated by D2 (commit `9871ee8`): REFUTING failed its criterion for some findings, and the
binned check showed the Platt curve overstating evidence in the sparsely observed score
tails. SUPPORTING transferred without a single violation and is left unchanged.

## Rule
Let $\mathrm{UCI}$ / $\mathrm{LCI}$ be upper / lower 95% bounds.

| State | Rule |
|---|---|
| SUPPORTING | $\mathrm{LCI}(\mathrm{LR}_\text{Platt}(p)) \ge 5$ (unchanged) |
| REFUTING | $\mathrm{UCI}(\mathrm{LR}_\text{Platt}(p)) \le 0.2$ **and** $\mathrm{UCI}(\mathrm{LR}_\text{empirical}(\text{band of } p)) \le 0.2$ |
| UNCERTAIN | otherwise |
| INSUFFICIENT DATA | unchanged (fewer than 20 development positives, or p outside support) |

Failure to confirm is not evidence against: when the Platt LR indicates refutation but the
empirical band does not confirm it, the state is UNCERTAIN. The engine records an internal
reason alongside the state:

| Reason | When |
|---|---|
| `SUPPORTING` / `REFUTING` | the state's criterion is met |
| `WEAK_EVIDENCE` | UNCERTAIN because neither criterion is met by the Platt LR |
| `REFUTING_NOT_EMPIRICALLY_CONFIRMED` | UNCERTAIN because the Platt LR meets the REFUTING criterion but the empirical band does not |
| `OUTSIDE_SUPPORT` / `FEW_DEV_POSITIVES` | INSUFFICIENT DATA |

The Platt LR stays the reported, continuous evidence estimate; the empirical band is a
decision-rule check only and never replaces the LR value.

## Empirical band LR (NIH development only)
- Data: NIH development split, random + supplement (the LR does not depend on prevalence;
  the supplement adds positives for rare findings). The NIH test split is not read while
  the bands are built.
- Bands: the edges already used in D2's binned check, fixed in advance:
  0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.
- Per finding and band: $k_\text{pos}$, $k_\text{neg}$ out of $N_\text{pos}$, $N_\text{neg}$;
  $\mathrm{LR}_\text{emp} = (k_\text{pos}/N_\text{pos}) / (k_\text{neg}/N_\text{neg})$.
- Upper bound, exact and conservative (valid with zero positives in a band):
  $\mathrm{UCI}(\mathrm{LR}_\text{emp}) = \mathrm{CP}^{\,97.5\%}_\text{upper}(k_\text{pos}, N_\text{pos}) \,/\, \mathrm{CP}^{\,2.5\%}_\text{lower}(k_\text{neg}, N_\text{neg})$
  (Clopper–Pearson; joint coverage ≥ 95%).
- Frozen to `results/nih_dev_empirical_bands.json` (read-only) before the test set is read.

## Evaluation (once, on the NIH test split already used in D2)
- Platt source: **primary** NIH (frozen `nih_frozen_dev_params.json`); **secondary**
  Indiana (frozen `frozen_dev_params.json`). The empirical bands are NIH development in
  both cases.
- Test sets: **primary** NIH test random part (natural prevalence); **secondary** random +
  supplement.
- For the dual-evidence REFUTING state, per finding: empirical test LR with 2,000-sample
  image-level bootstrap CI. Criterion met (✓) if the point estimate ≤ 0.2; demonstrated
  (✓✓) if the upper CI ≤ 0.2.
- Coverage: share of all test cases classified REFUTING, and positives classified
  REFUTING (count and share of positives: false reassurance).
- Comparison, same test images: the Platt-only REFUTING rule from D2 (coverage, LR,
  positives classified REFUTING).
- Reason breakdown: share of cases per internal reason.
- Pre-registered sensitivity analyses: the same dual rule at 0.5 and at 0.1 (both
  components), with SUPPORTING at 2 and 10 respectively, as in D2.

A rule that meets LR ≤ 0.2 only by classifying almost nothing REFUTING is reported as such;
the criterion is not relaxed after seeing the test results.
