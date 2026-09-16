"""Measure how often a synthesised claim is actually correct, by evidence strength.

Every threshold in this project was argued down to a measurement except the last one:
what net evidence justifies calling a claim High. This answers it the same way as the
rest -- by running the whole EDV pipeline over the ground-truth set and asking how
often claims at a given strength turn out to be true.

The question is deliberately not "is 0.45 High". It is:

    When EDV synthesises a claim at this net strength, how often is that claim right?

Boundaries chosen from this table can be defended as observed correctness. Boundaries
chosen from intuition cannot, and this branch has already had to withdraw two of those.

    python scripts/calibrate_claims.py
    python scripts/calibrate_claims.py --bands 6

Note what this can and cannot support. The claims are scored against MeSH labels
derived from the original reports, not an independent read, and the thresholds and
reliabilities being applied were themselves fitted on these same 544 films. The
numbers are therefore optimistic, and the shape of the curve is worth more than any
single cell. Held-out calibration is the next thing this needs.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from findings import FINDINGS  # noqa: E402
from medrax.agent.validator import EvidenceValidator as V  # noqa: E402

# reliability.json column -> the RELIABILITY key it was measured under
COLUMNS = {"chest_xray_expert": "chexagent_p",
           "chest_xray_classifier": "classifier_p",
           "chest_xray_expert_gemma": "chest_xray_expert_gemma_p"}


def claims(records):
    """One synthesised claim per image per finding, with the truth it is scored on."""
    for record in records:
        for finding, entry in record.get("findings", {}).items():
            if finding not in FINDINGS:
                continue
            scored = []
            for tool, column in COLUMNS.items():
                value = entry.get(column)
                if value is None:
                    continue
                item = V._classify(value, tool, finding)
                item["label"] = finding
                scored.append(item)
            if not scored:
                continue
            verdict = V._synthesise(scored, finding)
            yield {"file": record["file"], "finding": finding,
                   "truth": bool(entry["truth"]), **verdict}


def band_table(rows, edges):
    """Observed correctness within each band of |net|."""
    print(f"\n{'net evidence':>16s} {'N':>6s} {'accuracy':>9s} "
          f"{'claims +':>9s} {'PPV':>7s} {'claims -':>9s} {'NPV':>7s}")
    print("-" * 70)
    for low, high in zip(edges, edges[1:]):
        band = [r for r in rows if low <= abs(r["net"]) < high]
        if not band:
            continue
        # The claim is the direction the evidence points; its strength is |net|.
        positive = [r for r in band if r["net"] > 0]
        negative = [r for r in band if r["net"] < 0]
        correct = sum(1 for r in band if (r["net"] > 0) == r["truth"])
        ppv = sum(1 for r in positive if r["truth"]) / len(positive) if positive else float("nan")
        npv = sum(1 for r in negative if not r["truth"]) / len(negative) if negative else float("nan")
        print(f"{low:6.2f} - {high:<6.2f} {len(band):6d} {correct / len(band):8.1%} "
              f"{len(positive):9d} {ppv:6.1%} {len(negative):9d} {npv:6.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default="reliability.json")
    ap.add_argument("--bands", type=int, default=5)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    with open(args.input, encoding="utf-8") as handle:
        records = json.load(handle)

    rows = list(claims(records))
    print(f"{len(rows)} synthesised claims over {len(records)} images "
          f"and {len(FINDINGS)} findings")
    strongest = max(abs(r["net"]) for r in rows)
    edges = [i * strongest / args.bands for i in range(args.bands + 1)]
    edges[-1] += 1e-9
    band_table(rows, edges)

    print("\nby finding, at the strongest quarter of the evidence:")
    cut = strongest * 0.75
    print(f"  {'finding':18s} {'N':>5s} {'accuracy':>9s}")
    for finding in FINDINGS:
        band = [r for r in rows if r["finding"] == finding and abs(r["net"]) >= cut]
        if band:
            hit = sum(1 for r in band if (r["net"] > 0) == r["truth"]) / len(band)
            print(f"  {finding:18s} {len(band):5d} {hit:8.1%}")

    # A claim with no informative evidence at all is not a weak claim, it is an absent
    # one, and it should not be counted as a correct negative just because the finding
    # happened to be absent.
    silent = sum(1 for r in rows if r["n_support"] == 0 and r["n_against"] == 0)
    print(f"\n{silent} claims ({silent / len(rows):.1%}) had no informative evidence "
          "either way")
    return 0


if __name__ == "__main__":
    sys.exit(main())
