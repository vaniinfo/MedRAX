# EXPLORATORY: sensitivity of the E4 editing rule

Run on 2026-10-25 after all pre-registered results were known. Not pre-registered. See the docstring of `review_editing_rule.py`. A change is correct if it removes a condition the label says is absent, or adds one the label says is present.

## (a) base MedGemma, single image (1,500)

### pooled (protocol summary)

| variant | sensitivity | specificity | balanced accuracy | change in balanced accuracy [95% CI] |
|---|---|---|---|---|
| report alone | 0.113 | 0.989 | 0.551 | |
| V0 as reported | 0.145 | 0.986 | 0.566 | +0.015 [+0.008, +0.022] |
| V1 no provisional states | 0.145 | 0.986 | 0.566 | +0.015 [+0.008, +0.022] |
| V2 no fallback | 0.145 | 0.986 | 0.566 | +0.015 [+0.008, +0.022] |
| V3 frozen engine only | 0.145 | 0.986 | 0.566 | +0.015 [+0.008, +0.022] |

### mean over main (other summary, for comparison)

| variant | sensitivity | specificity | balanced accuracy | change in balanced accuracy [95% CI] |
|---|---|---|---|---|
| report alone | 0.126 | 0.988 | 0.557 | |
| V0 as reported | 0.168 | 0.984 | 0.576 | +0.019 [+0.012, +0.027] |
| V1 no provisional states | 0.168 | 0.984 | 0.576 | +0.019 [+0.012, +0.027] |
| V2 no fallback | 0.168 | 0.984 | 0.576 | +0.019 [+0.012, +0.027] |
| V3 frozen engine only | 0.168 | 0.984 | 0.576 | +0.019 [+0.012, +0.027] |

### Changes made by the reported rule (V0), by source

| source | action | correct | wrong |
|---|---|---|---|
| engine | added | 22 | 59 |

## (c) trained MedGemma, prior + current (499)

### mean over main (protocol summary)

| variant | sensitivity | specificity | balanced accuracy | change in balanced accuracy [95% CI] |
|---|---|---|---|---|
| report alone | 0.237 | 0.947 | 0.592 | |
| V0 as reported | 0.237 | 0.946 | 0.592 | -0.000 [-0.001, -0.000] |
| V1 no provisional states | 0.237 | 0.946 | 0.592 | -0.000 [-0.001, -0.000] |
| V2 no fallback | 0.237 | 0.946 | 0.592 | -0.000 [-0.001, -0.000] |
| V3 frozen engine only | 0.237 | 0.946 | 0.592 | -0.000 [-0.001, -0.000] |

### pooled (other summary, for comparison)

| variant | sensitivity | specificity | balanced accuracy | change in balanced accuracy [95% CI] |
|---|---|---|---|---|
| report alone | 0.297 | 0.943 | 0.620 | |
| V0 as reported | 0.297 | 0.942 | 0.620 | -0.000 [-0.001, -0.000] |
| V1 no provisional states | 0.297 | 0.942 | 0.620 | -0.000 [-0.001, -0.000] |
| V2 no fallback | 0.297 | 0.942 | 0.620 | -0.000 [-0.001, -0.000] |
| V3 frozen engine only | 0.297 | 0.942 | 0.620 | -0.000 [-0.001, -0.000] |

### Changes made by the reported rule (V0), by source

| source | action | correct | wrong |
|---|---|---|---|
| engine | removed | 1 | 0 |
| engine | added | 0 | 7 |

