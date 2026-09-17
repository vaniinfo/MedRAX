#!/usr/bin/env python
"""Does CheXagent's grounding discriminate, or does it box whatever it is asked about?

The validator currently tells the Director that "it grounded 19 of 19 true positives,
so failing to ground weighs against the finding". That statistic was only ever measured
on films where the finding was present. A model that draws a box for every phrase it is
handed scores 19 of 19 by construction, and the claim would rest on nothing.

MAIRA-2, tested the same way, does exactly that: asked to locate pneumothorax on a film
without one, it returned a confident box; asked for atelectasis on a film labelled
normal, it returned "Atelectasis in the right middle lobe" with coordinates. Its task is
localisation, and "where is X" presupposes X. CheXagent may or may not behave the same
way, which is the point of measuring rather than assuming.

The test is the same question asked of both classes, using the validator's own prompt
and region parser so it measures the shipped path:

    known-positive films  -> ask for the finding -> box?
    known-negative films  -> ask for the finding -> box?

    python scripts/grounding_specificity.py --per-class 30

Reading the result: boxing 95% of positives and 10% of negatives would make grounding a
real signal. Boxing 95% and 90% would mean it carries no information about presence, and
the "failing to ground" line should come out of the Director's block.
"""
import argparse
import json
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from findings import FINDINGS  # noqa: E402
from medrax.agent.validator import EvidenceValidator as V  # noqa: E402

DATA = os.getenv("MEDRAX_EVAL_DIR", "data/indiana_eval")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-class", type=int, default=30,
                    help="films per class per finding (default 30)")
    ap.add_argument("--input", default="reliability.json")
    ap.add_argument("--out", default="grounding_specificity.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    with open(args.input, encoding="utf-8") as handle:
        records = json.load(handle)

    from medrax.tools import XRayVQATool
    device = os.getenv("MEDRAX_DEVICE") or "cuda"
    tool = XRayVQATool(cache_dir=os.getenv("MEDRAX_MODEL_DIR",
                                           os.path.expanduser("~/model-weights")),
                       device=device)
    print(f"CheXagent loaded on {device}", flush=True)

    rng = random.Random(args.seed)
    rows, start = [], time.time()
    for finding in FINDINGS:
        pool = {True: [], False: []}
        for record in records:
            entry = record["findings"].get(finding)
            if entry and os.path.isfile(os.path.join(DATA, record["file"])):
                pool[bool(entry["truth"])].append(record["file"])
        for truth in (True, False):
            rng.shuffle(pool[truth])
            for name in pool[truth][:args.per_class]:
                try:
                    out, _ = tool._run(
                        image_paths=[os.path.join(DATA, name)],
                        prompt=V.GROUNDING_PROMPT.format(finding=finding),
                        max_new_tokens=96)
                    regions = out.get("regions", []) or []
                except Exception as exc:
                    print(f"  ! {name} {finding}: {exc}", flush=True)
                    continue
                rows.append({"file": name, "finding": finding, "truth": truth,
                             "boxed": bool(regions), "n_boxes": len(regions),
                             "response": str(out.get("response", ""))[:160]})
        done = [r for r in rows if r["finding"] == finding]
        pos = [r for r in done if r["truth"]]
        neg = [r for r in done if not r["truth"]]
        print(f"  {finding:18s} positives {sum(r['boxed'] for r in pos)}/{len(pos)}  "
              f"negatives {sum(r['boxed'] for r in neg)}/{len(neg)}  "
              f"({time.time() - start:.0f}s)", flush=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=1)

    print(f"\n{'finding':18s} {'TRUE finding':>14s} {'FALSE finding':>15s} {'difference':>12s}")
    print("-" * 64)
    for finding in FINDINGS:
        pos = [r for r in rows if r["finding"] == finding and r["truth"]]
        neg = [r for r in rows if r["finding"] == finding and not r["truth"]]
        if not pos or not neg:
            continue
        p = sum(r["boxed"] for r in pos) / len(pos)
        n = sum(r["boxed"] for r in neg) / len(neg)
        print(f"{finding:18s} {p:13.0%} {n:14.0%} {p - n:+11.0%}")
    allpos = [r for r in rows if r["truth"]]
    allneg = [r for r in rows if not r["truth"]]
    if allpos and allneg:
        p = sum(r["boxed"] for r in allpos) / len(allpos)
        n = sum(r["boxed"] for r in allneg) / len(allneg)
        print("-" * 64)
        print(f"{'ALL':18s} {p:13.0%} {n:14.0%} {p - n:+11.0%}"
              f"   (n={len(allpos)} / {len(allneg)})")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
