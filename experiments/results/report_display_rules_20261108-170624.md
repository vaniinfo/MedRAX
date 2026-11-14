# Radiologist report: screening criteria vs display restrictions

```
STATISTICAL SCREENING CRITERIA vs CLINICAL DISPLAY RESTRICTIONS

Screening (NIH feasibility check, pre-stated): a measurement passes when it can be computed on >= 80% of images and
the lower 95% bound of its AUC (condition vs No Finding) exceeds 0.60. Passing means only that the number carries
some signal in a selected, easy comparison. It is a necessary condition for display, not a sufficient one.

Display restrictions (clinical judgement, applied after screening):
- CTR on PA films only. AP films magnify the heart, so CTR values are not comparable with the PA reference (0.50);
  showing an AP value invites that comparison. CTR is shown only when the heart outline is not marked incorrect,
  and is labelled PROVISIONAL until a reviewer confirms the outline (Comp 8 showed that a visibly wrong outline can
  pass every automatic check).
- Costophrenic angle / recess depth on PA films only, as exploratory information. AP excluded although the AP angle
  passed screening (AUC 0.69, CI 0.64-0.75; AP depth 0.63, CI 0.57-0.69, did not pass), because:
    * AP films are mostly portable, supine or semi-erect: free fluid layers posteriorly instead of collecting in the
      costophrenic recess, so recess shape is a weaker and less specific sign;
    * AP projection, rotation and limited inspiration change the lung-base shape independently of fluid; AP QC pass
      rates were lower (midline 76% vs 94% PA);
    * the AP signal is modest and was measured against easy No-Finding controls, i.e. it would be weaker in
      practice; one of the two AP measurements failed screening.
- No display of measurements that did not pass screening (hyperinflation dome/height/area, atelectasis displacement,
  classifier saliency for consolidation) or that have no validation data (low lung volumes): outlines or the
  original CheXagent box only.
- Prior-to-current numerical changes are research-only: in the 19 comparisons the same patient's CTR varied by up
  to 0.13 between films, larger than the differences a reader would act on.
```
