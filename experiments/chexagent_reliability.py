"""Where is CheXagent right? Reliability of its decisions per condition and dataset.

For every labelled CheXagent reading we have, the decision the review tool would take:
- FOUND   = support cut-off reached (engine v1.0 SUPPORTING, or OUTSIDE_SUPPORT with p >= 0.5, for the
            six; provisional SUPPORTING for the seven)
- AGAINST = strong-no (engine REFUTING, or OUTSIDE_SUPPORT with p < 0.5, for the six; provisional
            REFUTING for the seven)
- otherwise UNCERTAIN.
Per dataset and condition: precision of FOUND (label positive), negative predictive value of
AGAINST (label negative), sensitivity of FOUND, how often each decision is made, and likelihood
ratios (prevalence-free). Wilson 95% intervals. Counts only.

Datasets: Indiana (544), NIH random sample (development = engine fitting data, test = held out),
VinDr Kaggle (3,000, external; edema not labelled), the 19 MIMIC pilot comparisons (preliminary
automatic labels).

    python experiments/chexagent_reliability.py
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
RESULTS = HERE / "results"
SIX = ["cardiomegaly", "pleural effusion", "pneumothorax", "consolidation", "pulmonary edema", "atelectasis"]


def wilson(k, n):
    if n == 0:
        return "n/a"
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{p:.0%} ({k}/{n}) [{max(0, c - h):.0%}-{min(1, c + h):.0%}]"


def decide(engine, condition, p):
    import provisional_findings as P
    if condition in SIX:
        ev = engine.assess(condition, p)
        if ev.state == "SUPPORTING" or (ev.reason == "OUTSIDE_SUPPORT" and p >= 0.5):
            return "FOUND"
        if ev.state == "REFUTING" or (ev.reason == "OUTSIDE_SUPPORT" and p < 0.5):
            return "AGAINST"
        return "UNCERTAIN"
    state = (P.assess(condition, p) or ("UNCERTAIN", condition))[0]
    return {"SUPPORTING": "FOUND", "REFUTING": "AGAINST"}.get(state, "UNCERTAIN")


def readings():
    """[(dataset, condition, truth 0/1, p_present)]."""
    out = []
    for r in map(json.loads, (RESULTS / "ppresent_indiana.jsonl").read_text(encoding="utf-8").splitlines()):
        out.append(("Indiana", r["finding"], r["truth"], r["p_present"]))
    split = json.loads((RESULTS / "split_nih_70_30.json").read_text(encoding="utf-8"))["random"]
    for r in map(json.loads, (RESULTS / "ppresent_nih.jsonl").read_text(encoding="utf-8").splitlines()):
        if r["set"] == "random":
            out.append((f"NIH {split[r['file']]}", r["finding"], r["truth"], r["p_present"]))
    import provisional_findings as P
    rows, labels = P.sample()
    for r in map(json.loads, (RESULTS / "provisional_readings.jsonl").read_text(encoding="utf-8").splitlines()):
        if r["p_present"] is not None:
            out.append((f"NIH {split[r['file']]}", r["finding"], int(P.FINDINGS[r["finding"]] in labels[r["file"]]),
                        r["p_present"]))
    for r in map(json.loads, (RESULTS / "vindr_readings.jsonl").read_text(encoding="utf-8").splitlines()):
        if r["finding"] != "pulmonary edema":  # not labelled in the Kaggle-derived data
            out.append(("VinDr", r["finding"], r["truth"], r["p_present"]))
    pilot = ROOT / "logs" / "pilot_runs_20261011-175859" / "chexagent.json"
    gt = RESULTS / "pilot_ground_truth_auto_labels.csv"
    if pilot.exists() and gt.exists():
        labels = {(r["page"], r["condition"]): r["label"] for r in csv.DictReader(gt.open(encoding="utf-8"))}
        for title, page in json.loads(pilot.read_text(encoding="utf-8")).items():
            for c, q in page["questions"].items():
                if q["score"] is not None:
                    out.append(("MIMIC pilot (19)", c, int(labels[(title.split()[1], c)] in ("present", "possible")),
                                q["score"]))
    return out


def main():
    from medrax.evidence import EvidenceEngine

    engine = EvidenceEngine()
    cells = defaultdict(lambda: defaultdict(int))
    for ds, c, y, p in readings():
        cells[(ds, c)][(decide(engine, c, p), y)] += 1
    datasets = ["Indiana", "NIH dev", "NIH test", "VinDr", "MIMIC pilot (19)"]
    conditions = SIX + ["emphysema", "nodule", "mass", "fibrosis", "pleural thickening", "pneumonia", "infiltration"]
    lines = ["# Where is CheXagent right? Reliability of its decisions", "",
             "Decision as in the review tool: FOUND = support cut-off reached; AGAINST = strong-no; otherwise "
             "uncertain. Engine v1.0 cut-offs for the six, provisional NIH cut-offs for the seven. Wilson 95% intervals. "
             "**NIH dev** is the data the cut-offs were fitted on (in-sample); **NIH test** and **VinDr** are held out; "
             "Indiana was discovery data; the MIMIC pilot uses preliminary automatic labels. Precision and NPV depend on "
             "how common the condition is in each dataset; the likelihood ratios do not.", "",
             "| condition | dataset | positives / n | FOUND: right (precision) | FOUND: share of positives caught | "
             "LR+ | AGAINST: right (NPV) | AGAINST: positives wrongly ruled out | AGAINST made on |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in conditions:
        for ds in datasets:
            t = cells.get((ds, c))
            if not t:
                continue
            pos = sum(v for (d, y), v in t.items() if y == 1)
            n = sum(t.values())
            f1, f0 = t[("FOUND", 1)], t[("FOUND", 0)]
            a1, a0 = t[("AGAINST", 1)], t[("AGAINST", 0)]
            lr = (f1 / pos) / (f0 / (n - pos)) if pos and n - pos and f0 else (math.inf if f1 else float("nan"))
            lines.append(
                f"| {c} | {ds} | {pos} / {n} | {wilson(f1, f1 + f0)} | {wilson(f1, pos)} | "
                f"{'inf' if lr == math.inf else 'n/a' if lr != lr else f'{lr:.1f}'} | {wilson(a0, a0 + a1)} | {a1} of {pos} | "
                f"{(a0 + a1) / n:.0%} of images |")
    (RESULTS / "chexagent_reliability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
