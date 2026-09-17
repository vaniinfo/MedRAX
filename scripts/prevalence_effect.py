#!/usr/bin/env python
"""What the enriched eval corpus costs us, and what de-enriching would recover.

build_eval_set.py samples stratified -- every positive of a rare finding is pulled into
the 544 -- which was the right call for the statistics it was built to serve. AUC and
balanced accuracy at the chosen threshold are prevalence-independent, so enrichment buys
precision on rare findings for free.

Then the scoring changed. RELIABILITY moved to ppv/npv at the operating point, because
that is what a claim is actually worth, and ppv is NOT prevalence-independent. So the
positive rows in that table are quoted at a prevalence we chose rather than one we found:
5.1% pneumothorax in the corpus against 0.7% in the collection it was drawn from.

This reports both, holding sensitivity and specificity fixed at the stored threshold --
they genuinely do not move with prevalence -- and re-deriving ppv at each rate:

    ppv = sens*p / (sens*p + (1-spec)*(1-p))

    python scripts/prevalence_effect.py                 # the two tables, side by side
    python scripts/prevalence_effect.py --strength      # what it does to scoring

Nothing here says the table is wrong. These are the right numbers for a population with
this mix. They are optimistic for any population where the finding is rarer, which is
most of them -- and the de-enriched corpus (build_eval_set.py --all) is how to replace
the estimate with a measurement.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from findings import FINDINGS, is_positive  # noqa: E402
from medrax.agent.validator import RELIABILITY  # noqa: E402

COLUMNS = {"chest_xray_expert": "chexagent_p",
           "chest_xray_classifier": "classifier_p",
           "chest_xray_expert_gemma": "chest_xray_expert_gemma_p"}


def ppv_at(sens: float, spec: float, prevalence: float) -> float:
    """Bayes, written out. The only term that changes between the two tables."""
    hit = sens * prevalence
    false_alarm = (1 - spec) * (1 - prevalence)
    return hit / (hit + false_alarm) if hit + false_alarm else float("nan")


def strength_at(rate: float, baseline: float) -> float:
    """How RELIABILITY converts a rate into evidence: credit above the no-skill baseline.

    Answering 'yes' at 40% correct is not 40% worth of evidence when 5% of films have the
    finding, and it is worth even less when 0.7% do. The baseline moves with prevalence,
    so both terms shift -- which is why this cannot be corrected by rescaling afterwards.
    """
    return (rate - baseline) / (1 - baseline) if baseline < 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default="reliability.json")
    ap.add_argument("--strength", action="store_true",
                    help="also show what this does to the scoring weights")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    import build_eval_set as builder
    collection = builder.load_frontal()
    natural = {f: sum(1 for r in collection if is_positive(r.get("problems", ""), f))
               / len(collection) for f in FINDINGS}

    with open(args.input, encoding="utf-8") as handle:
        records = json.load(handle)

    print(f"{len(collection)} frontal films in the collection; "
          f"{len(records)} in the measured corpus\n")
    head = (f"{'finding':17s} {'tool':11s} {'prev here':>9s} {'prev real':>9s} | "
            f"{'sens':>5s} {'spec':>5s} | {'PPV here':>8s} {'PPV real':>8s}")
    if args.strength:
        head += f" | {'str here':>8s} {'str real':>8s}"
    print(head)
    print("-" * (len(head) + 2))

    for finding in FINDINGS:
        for tool, column in COLUMNS.items():
            entry = RELIABILITY.get((tool, finding))
            if not entry:
                continue
            threshold = entry["threshold"]
            readings = [(bool(r["findings"][finding]["truth"]),
                         r["findings"][finding].get(column))
                        for r in records if finding in r["findings"]]
            readings = [(truth, p) for truth, p in readings if p is not None]
            positives = [p for truth, p in readings if truth]
            negatives = [p for truth, p in readings if not truth]
            if not positives or not negatives:
                continue
            sens = sum(p >= threshold for p in positives) / len(positives)
            spec = sum(p < threshold for p in negatives) / len(negatives)

            here, real = entry["prevalence"], natural[finding]
            ppv_here, ppv_real = ppv_at(sens, spec, here), ppv_at(sens, spec, real)
            line = (f"{finding:17s} {tool.replace('chest_xray_', '')[:11]:11s} "
                    f"{here:9.1%} {real:9.1%} | {sens:5.2f} {spec:5.2f} | "
                    f"{ppv_here:8.0%} {ppv_real:8.0%}")
            if args.strength:
                line += (f" | {strength_at(ppv_here, here):8.2f} "
                         f"{strength_at(ppv_real, real):8.2f}")
            print(line)
        print()

    print("sens and spec are prevalence-independent and identical in both columns; only "
          "ppv moves.\nTo measure rather than estimate the right column:\n"
          "  python scripts/build_eval_set.py --all --out data/indiana_full\n"
          "  python scripts/measure_reliability.py --data data/indiana_full")
    return 0


if __name__ == "__main__":
    sys.exit(main())
