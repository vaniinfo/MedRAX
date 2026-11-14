# Results: BioViL-T heat maps and prior-vs-current subtraction (2026-11-09)

Protocol: `experiments/heatmap_subtraction_validation_protocol.md`. Its criteria were written before running.
Code: `experiments/heatmap_subtraction_validation.py`.

Raw results:
- `heatmap_subtraction_validation_biovil_20261109-144817.json`
- `heatmap_subtraction_validation_subtraction_20261109-144817.json`

Parameters were frozen from the 9-finding pilots; nothing was tuned. No existing prediction, score or report was
changed.

## A. BioViL-T pointing game: does the map's peak fall inside radiologist boxes?

| Set | n | BioViL-T hit [95% CI] | Random point in lung | Criterion | CheXagent hit | MAIRA-2 hit | BioViL-T - MAIRA-2 |
|---|---|---|---|---|---|---|---|
| VinDr effusion | 119 | 0.40 [0.32-0.49] | 0.10 | met | 0.87* | 0.28 | +0.13 [0.03, 0.23] |
| VinDr consolidation | 25 | 0.52 [0.32-0.72] | 0.18 | met | 0.92* | 0.32 | +0.20 [-0.04, 0.44] |
| VinDr pneumothorax | 15 | 0.73 [0.47-0.93] | 0.36 | met | 0.93* | 0.60 | +0.13 [-0.13, 0.40] |
| VinDr atelectasis | 10 | 0.10 [0.00-0.30] | 0.12 | descriptive only (n < 15) | 1.00* | 0.00 | +0.10 |
| NIH effusion | 150 | 0.47 [0.39-0.55] | 0.13 | met | 0.55 | 0.39 | +0.08 [-0.02, 0.18] |
| NIH atelectasis | 174 | 0.17 [0.11-0.24] | 0.08 | met | 0.58 | 0.28 | -0.11 [-0.19, -0.03] |
| NIH pneumothorax | 95 | 0.33 [0.23-0.42] | 0.12 | met | 0.67 | 0.32 | +0.01 [-0.10, 0.12] |
| NIH pneumonia | 120 | 0.58 [0.50-0.68] | 0.26 | met | 0.75 | 0.12 | +0.47 [0.37, 0.56] |

\* CheXagent was trained on VinDr, so its VinDr numbers are optimistic.

**Criterion.** Met in 7 of 7 eligible sets: BioViL-T points better than chance.

**Against MAIRA-2.**
- Better on NIH pneumonia (+0.47) and VinDr effusion (+0.13).
- Worse on NIH atelectasis (-0.11).
- Otherwise similar.

**Against CheXagent.** Lower hit rate in every set: by 0.07 on NIH effusion (CI includes 0) and by 0.16-0.41 on the
other NIH sets.

**Atelectasis.** Weak on both datasets.

**Side agreement.** Measured as the side of the top 10% of the map vs the side of the truth boxes: 0.49-0.75, and
0.19 on NIH atelectasis. A map that spreads over both sides counts as a mismatch, so this metric penalises diffuse
maps.

**Presence.** Not tested, by design. The pilot showed the map also lights up when the finding is absent.

## B. Subtraction: does it separate pairs with label change from stable pairs?

Data: NIH longitudinal pairs, same view.
- 351 pairs grouped; 1 lung-segmentation failure.
- Registration OK (lung Dice >= 0.85) in 98%.

| Comparison | New / changed pairs | Stable pairs | AUC [95% CI] | Criterion (lower bound > 0.6) |
|---|---|---|---|---|
| B1: new opacity label, denser share | 62 | 205 | 0.60 [0.52-0.67] | not met |
| B2: resolved opacity label, more-lucent share | 78 | 205 | 0.52 [0.44-0.60] | not met |
| B1, well-registered pairs only | 60 | 202 | 0.60 [0.52-0.67] | not met |
| B2, well-registered pairs only | 77 | 202 | 0.52 [0.45-0.60] | not met |

**New pneumothorax (n = 5).** Lucent share 0.009-0.055 vs a stable-pair median of 0.021, so there is no consistent
signal.

**Interpretation.** There is a weak signal for new opacity and none for resolution. Probable reasons, all stated in
the protocol beforehand:
- noisy text-mined labels;
- per-lung normalization cancelling broad changes;
- basal effusions lying outside the segmented lung.

## Conclusion

- **BioViL-T** passes the pre-stated feasibility bar for effusion, consolidation/pneumonia and pneumothorax.
  - It could be offered as an exploratory "where to look" overlay alongside, not instead of, the CheXagent box. The
    caption must say it shows similarity to the words and not presence.
  - It does not localize better than CheXagent.
  - Atelectasis should be excluded.
- **Subtraction** does not pass and should not be shown to radiologists in its current form.
