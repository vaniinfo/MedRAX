"""Does forced EDV change what the Director concludes, or only how it is described?

Everything measured so far shows EDV computes something defensible. None of it shows
the computation changes an outcome. This is the ablation that would: the same model,
the same question, the same tool outputs, differing only in whether the validated
evidence is supplied.

  arm A  raw tool outputs, as the Director saw them before any of this existed
  arm B  the same outputs as validated evidence records, plus the cross-tool
         synthesis and the computed claim ceiling

Both arms get an identical system prompt and answer in a fixed two-line format, so the
only variable is the evidence framing. Replayed from reliability.json, so no GPU and no
model loading -- which is the return on having committed that file.

    python scripts/ablation_edv.py --images 150            # ~1800 calls
    python scripts/ablation_edv.py --images 5 --dry-run    # print prompts, call nothing

LIMITATION worth stating before reading any result: the radiograph itself is not sent.
The real Director sees it. Excluding it isolates EDV's effect on evidence
interpretation from GPT-4o's own reading of the image, which is the comparison this
answers -- but it is not the full system.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from findings import FINDINGS  # noqa: E402
from medrax.agent.validator import EvidenceValidator as V  # noqa: E402

COLUMNS = {"chest_xray_expert": "chexagent_p",
           "chest_xray_classifier": "classifier_p",
           "chest_xray_expert_gemma": "chest_xray_expert_gemma_p",
           "llava_med_qa": "llava_med_qa_p"}

SYSTEM = (
    "You are analysing a chest X-ray through specialist tools. You cannot see the image; "
    "judge only from what is given.\n"
    "Answer in exactly two lines and nothing else:\n"
    "VERDICT: present|absent\n"
    "CONFIDENCE: High|Medium|Low"
)


def arm_a(finding, entry):
    """What the Director saw before EDV: the tools' own answers, unqualified."""
    lines = [f"Question: is {finding} present in this chest X-ray?", "", "Tool outputs:"]
    for tool, column in COLUMNS.items():
        value = entry.get(column)
        if value is None:
            continue
        said = "Yes" if value >= 0.5 else "No"
        lines.append(f"  {tool}: answered {said!r}, confidence {value:.4f}")
    return "\n".join(lines)


def arm_b(finding, entry):
    """The same answers as validated evidence, plus synthesis and the claim ceiling."""
    scored = []
    for tool, column in COLUMNS.items():
        value = entry.get(column)
        if value is None:
            continue
        item = V._classify(value, tool, finding)
        item.update({"label": finding, "value": value})
        scored.append(item)

    lines = [f"Question: is {finding} present in this chest X-ray?", "",
             "Validated evidence. Each tool's raw answer is shown beside what answering "
             "that way has been measured to be worth, on 544 films with expert labels:"]
    for item in scored:
        side = item.get("polarity") or {}
        rate = side.get("ppv", side.get("npv"))
        lines.append(
            f"  {item['tool']}: raw {item['value']:.4f} -> "
            f"{'supports ' + item['supports'] if item['informative'] else 'NOT DECISIVE'}"
            f", evidence_strength {V._strength(item):.3f}"
            + (f", answering this way is correct {rate:.0%} of the time "
               f"({side.get('n')} such answers)" if rate is not None else
               ", this pair has never been measured"))

    verdict = V._synthesise(scored, finding)
    direction = ("PRESENT" if verdict["net"] > 0 else
                 "ABSENT" if verdict["net"] < 0 else "UNDECIDED")
    lines += ["",
              f"Cross-tool synthesis (strongest evidence anchors; corroboration "
              f"discounted by measured error dependence -- NOT a count of agreeing "
              f"tools): net {verdict['net']:+.2f} -> {direction} "
              f"(support {verdict['support']:.2f}, against {verdict['against']:.2f})",
              f"CLAIM CONFIDENCE CEILING: {V._claim_ceiling(verdict['net'])} "
              f"(you may report lower, never higher)"]
    return "\n".join(lines)


def ask(client, model, prompt):
    reply = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}])
    text = reply.choices[0].message.content or ""
    verdict = re.search(r"VERDICT:\s*(present|absent)", text, re.I)
    confidence = re.search(r"CONFIDENCE:\s*(high|medium|low)", text, re.I)
    return (verdict.group(1).lower() if verdict else None,
            confidence.group(1).capitalize() if confidence else None)


def report(rows, arm):
    """Accuracy, positive-claim precision, and calibration by stated confidence."""
    done = [r for r in rows if r[f"{arm}_verdict"]]
    if not done:
        return
    correct = sum(1 for r in done if (r[f"{arm}_verdict"] == "present") == r["truth"])
    positives = [r for r in done if r[f"{arm}_verdict"] == "present"]
    ppv = (sum(1 for r in positives if r["truth"]) / len(positives)) if positives else float("nan")
    print(f"\n  {arm.upper()}  n={len(done)}  accuracy {correct / len(done):.1%}  "
          f"positive claims {len(positives)}  PPV {ppv:.1%}")
    print(f"    {'stated':>8s} {'n':>5s} {'correct':>9s}")
    for grade in ("High", "Medium", "Low"):
        band = [r for r in done if r[f"{arm}_confidence"] == grade]
        if band:
            hit = sum(1 for r in band if (r[f"{arm}_verdict"] == "present") == r["truth"])
            print(f"    {grade:>8s} {len(band):5d} {hit / len(band):8.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default="reliability.json")
    ap.add_argument("--out", default="ablation.json")
    ap.add_argument("--images", type=int, default=150)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    from dotenv import load_dotenv
    load_dotenv()

    with open(args.input, encoding="utf-8") as handle:
        records = json.load(handle)
    rng = random.Random(args.seed)
    sample = sorted(records, key=lambda r: r["file"])
    rng.shuffle(sample)
    sample = sample[:args.images]

    rows = []
    for record in sample:
        for finding in FINDINGS:
            entry = record["findings"].get(finding)
            if entry:
                rows.append({"file": record["file"], "finding": finding,
                             "truth": bool(entry["truth"]), "entry": entry})
    print(f"{len(rows)} claims x 2 arms = {len(rows) * 2} calls to {args.model}")

    if args.dry_run:
        row = rows[0]
        print(f"\n=== ARM A ===\n{arm_a(row['finding'], row['entry'])}")
        print(f"\n=== ARM B ===\n{arm_b(row['finding'], row['entry'])}")
        return 0

    from openai import OpenAI
    client = OpenAI()
    start = time.time()
    for index, row in enumerate(rows, 1):
        for arm, build in (("a", arm_a), ("b", arm_b)):
            try:
                verdict, confidence = ask(client, args.model,
                                          build(row["finding"], row["entry"]))
            except Exception as exc:
                print(f"  ! {row['file']} {row['finding']} {arm}: {exc}", flush=True)
                verdict = confidence = None
            row[f"{arm}_verdict"], row[f"{arm}_confidence"] = verdict, confidence
        if index % 25 == 0:
            rate = (time.time() - start) / index
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump([{k: v for k, v in r.items() if k != "entry"} for r in rows],
                          handle, indent=1)
            print(f"  {index}/{len(rows)}  (~{rate * (len(rows) - index):.0f}s left)",
                  flush=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump([{k: v for k, v in r.items() if k != "entry"} for r in rows],
                  handle, indent=1)

    print(f"\nDONE in {time.time() - start:.0f}s -> {args.out}")
    report(rows, "a")
    report(rows, "b")

    flipped = [r for r in rows if r.get("a_verdict") and r.get("b_verdict")
               and r["a_verdict"] != r["b_verdict"]]
    print(f"\nverdict changed on {len(flipped)} of {len(rows)} claims")
    if flipped:
        better = sum(1 for r in flipped if (r["b_verdict"] == "present") == r["truth"])
        print(f"  of those, EDV was right on {better} and wrong on {len(flipped) - better}")
    grades = Counter((r.get("a_confidence"), r.get("b_confidence")) for r in rows)
    print("\nstated confidence, A -> B:")
    for (a, b), n in grades.most_common(6):
        print(f"  {str(a):>6s} -> {str(b):<6s} {n:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
