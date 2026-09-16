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

WHAT THIS HAS ESTABLISHED, stated no more strongly than the data allows:

    On an unseen 272-film set, EDV scores above 0.80 identified a substantially
    higher-PPV group of positive claims, while the data did not establish separation
    between the two lower evidence ranges.

                   fitted (544)              held out (272)
      < 0.48       37.7% [32.5-43.3]         52.5% [43.6-61.3]
      0.48-0.80    62.9% [57.4-68.1]         67.6% [58.1-75.9]
      > 0.80       85.1% [79.0-89.6]         87.8% [78.5-93.5]

That is one boundary, near 0.80, not three tiers. The two lower bands overlap on
held-out data and must not be reported as distinct. Anyone reading this table and
deriving High/Medium/Low from it is doing the thing this branch spent its history
undoing -- turning a number that looks like a boundary into one.

EXTERNAL VALIDATION IS HARDER THAN IT LOOKS, and one candidate is already ruled out.

The classifier's weights are densenet121-res224-all, and torchxrayvision's "all" means
trained on NIH, PadChest, CheXpert, MIMIC and RSNA. CheXagent was trained on an
aggregate of public CXR corpora, and MedGemma's data is undisclosed. Between them the
three tools have seen most public chest X-ray data, so a dataset being new to this
project does not make it new to the pipeline.

SIIM-ACR Pneumothorax is the obvious choice for testing pneumothorax specifically --
~2700 positives with expert pixel masks -- and it does not qualify. Its metadata
carries NIH ChestX-ray14's fingerprint: the same fields (view position, patient age,
patient sex), DICOM UIDs generated under the DCMTK root 1.2.276.0.7230010.3 (so the
images were converted, not acquired), and NIH's notorious corrupted ages, 148 and 413,
surviving in the same rows. The classifier has trained on those images.

It remains useful as a LABEL-QUALITY test -- expert segmentation against Indiana's
report-derived MeSH terms -- but it cannot answer whether the 0.80 relationship
transfers to an external population. That needs a regionally independent corpus with
independent labels: VinDr-CXR (Vietnam) or CANDID-PTX (New Zealand), both PhysioNet
credentialed. Freeze everything before running it; if the relationship breaks there,
the dataset shift is the finding.

Limits that bound the claim further. Labels come from the original reports rather than
an independent read. The held-out films contain no pneumothorax or edema positives,
because the entire 3818-film collection holds only 28 and 45 of them and the fitting
set took all of them -- so pneumothorax, the finding this code path was built around,
cannot be held-out validated in this corpus at all. Those absent positives also make
the negative side artificially easy, which is why accuracy and NPV read better here
than they should. PPV is the only column that compares fairly between the two sets.
"""
import argparse
import json
import math
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


def wilson(hits, n, z=1.96):
    """95% Wilson score interval for a proportion.

    A percentage with no interval hides how thin the cell behind it is: 85.4% from 171
    claims and 85.4% from 12 are the same number and very different evidence. This is
    the same problem already caught at the individual-tool level, where MedGemma's
    pneumothorax PPV of 54.5% turned out to rest on 11 answers.

    Wilson rather than the normal approximation because it stays inside 0-1 and behaves
    at small n and at proportions near the ends, which is exactly where these land.
    """
    if not n:
        return float("nan"), float("nan")
    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(rows, correct_if):
    hits = sum(1 for r in rows if correct_if(r))
    low, high = wilson(hits, len(rows))
    return hits, (hits / len(rows) if rows else float("nan")), low, high


def band_table(rows, edges):
    """Observed correctness within each band of |net|, with intervals."""
    print(f"\n{'net evidence':>15s} {'N':>5s} {'acc':>7s} | {'pos':>4s} {'PPV':>7s} "
          f"{'95% CI':>15s} | {'neg':>4s} {'NPV':>7s} {'95% CI':>15s}")
    print("-" * 92)
    for low, high in zip(edges, edges[1:]):
        band = [r for r in rows if low <= abs(r["net"]) < high]
        if not band:
            continue
        # The claim is the direction the evidence points; its strength is |net|.
        positive = [r for r in band if r["net"] > 0]
        negative = [r for r in band if r["net"] < 0]
        _, acc, _, _ = rate(band, lambda r: (r["net"] > 0) == r["truth"])
        _, ppv, p_lo, p_hi = rate(positive, lambda r: r["truth"])
        _, npv, n_lo, n_hi = rate(negative, lambda r: not r["truth"])
        p_ci = f"[{p_lo:.1%}-{p_hi:.1%}]" if positive else ""
        n_ci = f"[{n_lo:.1%}-{n_hi:.1%}]" if negative else ""
        print(f"{low:5.2f} - {high:<5.2f} {len(band):5d} {acc:6.1%} | "
              f"{len(positive):4d} {ppv:6.1%} {p_ci:>15s} | "
              f"{len(negative):4d} {npv:6.1%} {n_ci:>15s}")


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
