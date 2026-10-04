# EXPLORATORY: does the box follow the side named in the phrase?

## Side named in the phrase (run after location_agreement_dev.md; exploratory)

Same 40 NIH pneumothorax cases (unilateral, radiologist boxes; random sample, seed 20261003),
each located twice with the side in the phrase: once the true side ("left pneumothorax"),
once the wrong side.

| model | true-side phrase: boxes the claimed side | wrong-side phrase: boxes the claimed side |
|---|---|---|
| CheXagent | 40/40 | 40/40 |
| MAIRA-2 | 40/40 | 40/40 |

Both models box whatever side the phrase names, so a box can only be evidence about the side
when the phrase carries no side (as in location_agreement_dev.md, and as in the pipeline, which strips
side words before CheXagent locates). With side-free phrases MAIRA-2 mostly boxes the
patient's right (553 of the 596 answers that name no side and box a single side), which is why its side
LRs in location_agreement_dev.md are near 1; when its own text names a side, its box follows that text.
