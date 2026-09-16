"""Turn reliability.json into the RELIABILITY table in medrax/agent/validator.py.

measure_reliability.py records what each tool said about each image. This decides what
those answers are worth. It was previously a cell in notebooks/medrax_colab.ipynb that
ended with "copy any improved rows into validator.py" -- a manual step, and the reason
the measurement and the table drifted apart.

Three things the notebook cell did not do, each of which changes what the table means:

  * a confidence interval on every number, by bootstrap over images. Thresholds were
    quoted to two decimals off as few as 24 positives, implying a precision they do
    not have.
  * a minimum positive count. A row measured on three positives is noise, and letting
    it into RELIABILITY is worse than leaving the pair out, because the validator
    labels an unmeasured pair ASSUMED and a measured one as established fact.
  * a paired test of CheXagent against the classifier. "CheXagent beats the classifier
    on every finding" is the only evidence behind weighting one tool over the other,
    and two AUCs computed on ~200 images can differ by a lot without meaning anything.

One caveat this cannot fix: thresholds are still chosen and scored on the same images,
which flatters them. The bootstrap interval is what shows by how much -- where it spans
most of 0-1, the point estimate is close to meaningless.

  python scripts/analyze_reliability.py
  python scripts/analyze_reliability.py --bootstrap 2000 --min-positives 25
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from findings import FINDINGS, MIN_POSITIVES

# (RELIABILITY key, display name, reliability.json key) for the in-process tools,
# whose columns predate the naming convention below.
TOOLS = [("chest_xray_expert", "CheXagent", "chexagent_p"),
         ("chest_xray_classifier", "classifier", "classifier_p")]


def discover_tools(records):
    """TOOLS, plus any served model measured by scripts/measure_remote.py.

    A served model writes its column as "<tool name>_p", so a new one is picked up
    without editing this file -- which matters because the whole point of the service
    split is that adding a model should not mean touching the agent or its scripts.
    """
    known = {key for _, _, key in TOOLS}
    extra = set()
    for record in records:
        for entry in record.get("findings", {}).values():
            extra.update(k for k in entry
                         if k.endswith("_p") and k not in known and k != "p")
    tools = list(TOOLS)
    for key in sorted(extra):
        name = key[:-2]
        tools.append((name, name.replace("chest_xray_", ""), key))
    return tools
SWEEP = [t / 100 for t in range(5, 100, 5)]

# A margin only means something if the tool actually produces graded output. Below this
# share of readings in the interior, a "probability" is a yes/no vote wearing a decimal
# point, and the distance from its decision point is an artefact rather than evidence.
#
# The cut point does no delicate work: measured per tool and finding, MedGemma lands
# between 1% and 6% and the other two between 43% and 99%. Nothing falls in the gap.
GRADED_INTERIOR_MIN = 0.10


def output_type(pairs):
    """"probability" if the tool grades its answers, "binary" if it effectively does not."""
    interior = sum(1 for p, _ in pairs if 0.05 <= p <= 0.95) / len(pairs)
    return "probability" if interior >= GRADED_INTERIOR_MIN else "binary"


def auc(pairs):
    """Area under the ROC curve, by the Mann-Whitney rank formulation.

    O(n log n) rather than the O(n+ * n-) pairwise form the notebook used. That only
    matters because the bootstrap runs this a few thousand times per row.
    """
    pos = sum(1 for _, t in pairs if t)
    neg = len(pairs) - pos
    if not pos or not neg:
        return float("nan")
    order = sorted(pairs)
    ranks, i = [0.0] * len(order), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        shared = (i + j) / 2 + 1          # average rank (1-based) across a run of ties
        for k in range(i, j + 1):
            ranks[k] = shared
        i = j + 1
    rank_sum = sum(r for r, (_, t) in zip(ranks, order) if t)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def auc_pairwise(pairs):
    """The notebook's definition, kept only to check the fast one agrees with it."""
    pos = [p for p, t in pairs if t]
    neg = [p for p, t in pairs if not t]
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def balanced(pairs, thr):
    """(sensitivity, specificity, balanced accuracy) at one decision point."""
    tp = fn = fp = tn = 0
    for p, t in pairs:
        if t:
            tp, fn = (tp + 1, fn) if p >= thr else (tp, fn + 1)
        else:
            fp, tn = (fp + 1, tn) if p >= thr else (fp, tn + 1)
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return sens, spec, (sens + spec) / 2


def best_threshold(pairs):
    """The decision point maximising balanced accuracy. NOT 0.5, and not assumed."""
    return max(((t, *balanced(pairs, t)) for t in SWEEP), key=lambda r: r[3])


def percentile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = q / 100 * (len(s) - 1)
    f = int(i)
    return s[f] if f + 1 >= len(s) else s[f] + (s[f + 1] - s[f]) * (i - f)


def ci(xs):
    return percentile(xs, 2.5), percentile(xs, 97.5)


def bootstrap_rows(rows, fields, n, rng):
    """Resample images with replacement, recomputing every statistic on each draw.

    Resampling images rather than predictions keeps one patient's readings from every
    tool together, which is what makes the paired AUC difference a fair test rather
    than independent estimates compared by eye.

    `rows` is [({field: value, ...}, truth)] so any number of tools can be measured --
    a served model added later needs no change here.
    """
    size = len(rows)
    out = {f"{f}_auc": [] for f in fields}
    out.update({f"{f}_thr": [] for f in fields})
    baseline = fields[0]
    out.update({f"delta_{f}": [] for f in fields[1:]})
    for _ in range(n):
        draw = [rows[rng.randrange(size)] for _ in range(size)]
        if not any(t for _, t in draw) or all(t for _, t in draw):
            continue                       # a degenerate draw has no defined AUC
        pairs = {f: [(values[f], t) for values, t in draw if values.get(f) is not None]
                 for f in fields}
        for field, sample in pairs.items():
            if sample:
                out[f"{field}_auc"].append(auc(sample))
                out[f"{field}_thr"].append(best_threshold(sample)[0])
        for field in fields[1:]:
            if pairs[baseline] and pairs[field]:
                out[f"delta_{field}"].append(auc(pairs[baseline]) - auc(pairs[field]))
    return out


def report_rates(records, finding):
    """Precision and recall of the report generator's text for one finding.

    Stance comes from EvidenceValidator._text_stance: ASSERTS, NEGATES, SILENT, or
    CONTRADICTS when a single report both asserts and denies the finding -- its
    FINDINGS and IMPRESSION sections come from two models that never see each other.
    CONTRADICTS is counted separately rather than forced into one of the two.
    """
    asserted = correct = positives = contradicts = 0
    for r in records:
        entry = r["findings"].get(finding)
        if not entry:
            continue
        positives += bool(entry["truth"])
        if entry.get("report_stance") == "CONTRADICTS":
            contradicts += 1
        elif entry.get("report_stance") == "ASSERTS":
            asserted += 1
            correct += bool(entry["truth"])
    return {"recall": correct / positives if positives else float("nan"),
            "precision": correct / asserted if asserted else float("nan"),
            "asserted": asserted, "contradicts": contradicts}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default="reliability.json")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="resamples per row; 0 to skip the intervals entirely")
    ap.add_argument("--min-positives", type=int, default=MIN_POSITIVES,
                    help=f"rows below this are reported but not emitted (default "
                         f"{MIN_POSITIVES})")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed, for reproducibility")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        return f"{args.input} not found -- run scripts/measure_reliability.py first"
    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)
    rng = random.Random(args.seed)

    measured = [f for f in FINDINGS if any(f in r["findings"] for r in records)]
    missing = [f for f in FINDINGS if f not in measured]
    print(f"{len(records)} images, {len(measured)} findings measured")
    if missing:
        print(f"not present in {args.input}: {', '.join(missing)} "
              "(re-run measure_reliability.py to add them)")

    tools = discover_tools(records)
    if len(tools) > len(TOOLS):
        print("served models found in this file: "
              + ", ".join(k for k, _, _ in tools[len(TOOLS):]))
    emit, skipped, notes = {}, [], []
    print(f"\n{'finding':18s} {'n+':>4s} {'tool':11s} {'AUC':>6s} {'95% CI':>14s} "
          f"{'thr':>5s} {'95% CI':>12s} {'bal acc':>8s}")
    for finding in measured:
        rows = [({f: r["findings"][finding].get(f) for _, _, f in tools},
                 bool(r["findings"][finding]["truth"]))
                for r in records if finding in r["findings"]]
        npos = sum(1 for _, t in rows if t)
        fields = [f for _, _, f in tools]
        boot = bootstrap_rows(rows, fields, args.bootstrap, rng) if args.bootstrap else None

        for key, label, field in tools:
            pairs = [(values[field], t) for values, t in rows
                     if values.get(field) is not None]
            if not pairs or npos == 0:
                continue
            a = auc(pairs)
            thr, _sens, _spec, acc = best_threshold(pairs)
            if boot:
                a_lo, a_hi = ci(boot[f"{field}_auc"])
                t_lo, t_hi = ci(boot[f"{field}_thr"])
                a_ci, t_ci = f"{a_lo:.3f}-{a_hi:.3f}", f"{t_lo:.2f}-{t_hi:.2f}"
            else:
                a_lo = a_hi = float("nan")
                t_lo = t_hi = None
                a_ci = t_ci = "--"

            flag = ""
            if npos < args.min_positives:
                flag = f"  UNDERPOWERED (n+={npos})"
                skipped.append((key, finding, f"only {npos} positives"))
            elif boot and a_hi < 0.5:
                flag = "  WORSE THAN CHANCE"
                skipped.append((key, finding,
                                f"AUC CI {a_lo:.3f}-{a_hi:.3f} is entirely below chance"))
            elif boot and a_lo <= 0.5:
                # Cannot rule out chance. Weighting a tool on this would be the same
                # class of mistake as the dead zone: a number treated as a measurement.
                flag = "  CI SPANS CHANCE"
                skipped.append((key, finding,
                                f"AUC CI {a_lo:.3f}-{a_hi:.3f} includes chance"))
            else:
                # thr_ci is what the validator uses to decide whether a reading is a
                # vote at all: inside the interval, a plausible alternative threshold
                # would flip its direction, so the data does not determine which way
                # this tool leans.
                emit[(key, finding)] = {
                    "auc": round(a, 3), "threshold": thr,
                    "thr_ci": None if t_lo is None else (t_lo, t_hi),
                    # Decides whether the validator may multiply by the margin at all.
                    "output_type": output_type(pairs)}

            print(f"{finding:18s} {npos:4d} {label:11s} {a:6.3f} {a_ci:>14s} "
                  f"{thr:5.2f} {t_ci:>12s} {acc:7.1%}{flag}")

        # Every other tool measured against the first, on the same resampled images.
        for _, other_label, other_field in tools[1:]:
            deltas = boot.get(f"delta_{other_field}") if boot else None
            if not deltas:
                continue
            d_lo, d_hi = ci(deltas)
            verdict = (f"{tools[0][1]} better" if d_lo > 0 else
                       f"{other_label} better" if d_hi < 0 else
                       "no separation -- the difference is within noise")
            notes.append(f"  {finding:18s} vs {other_label:12s} "
                         f"delta AUC {percentile(deltas, 50):+.3f} "
                         f"(95% CI {d_lo:+.3f} to {d_hi:+.3f})  {verdict}")

    if notes:
        print(f"\n{tools[0][1]} minus each other tool, paired on the same resampled images:")
        print("\n".join(notes))

    # A one-time agreement check. The fast AUC is worth nothing if it disagrees with
    # the definition the published table was computed from.
    for finding in measured[:1]:
        pairs = [(r["findings"][finding]["chexagent_p"], bool(r["findings"][finding]["truth"]))
                 for r in records
                 if finding in r["findings"] and r["findings"][finding].get("chexagent_p") is not None]
        if pairs:
            fast, slow = auc(pairs), auc_pairwise(pairs)
            ok = "agrees with" if abs(fast - slow) < 1e-9 else "DISAGREES WITH"
            print(f"\nrank AUC {ok} the pairwise definition on {finding} "
                  f"({fast:.6f} vs {slow:.6f})")

    print("\n" + "=" * 78)
    print("Paste into medrax/agent/validator.py, replacing RELIABILITY entirely.")
    print("Pairs absent here are absent on purpose: the validator marks an unmeasured")
    print("pair ASSUMED, which is the honest description of one it could not measure.")
    print("=" * 78)
    if emit:
        if any(v["thr_ci"] is None for v in emit.values()):
            print("# WARNING: run with --bootstrap to get thr_ci. Without it the "
                  "validator\n# falls back to a guessed band for these pairs.")
        print("RELIABILITY: Dict[Tuple[str, str], Dict[str, Any]] = {")
        for (key, finding), v in emit.items():
            ci_text = ("None" if v["thr_ci"] is None
                       else f'({v["thr_ci"][0]:.2f}, {v["thr_ci"][1]:.2f})')
            print(f'    ("{key}", "{finding}"):')
            print(f'        {{"auc": {v["auc"]:.3f}, "threshold": {v["threshold"]:.2f}, '
                  f'"thr_ci": {ci_text}, "output_type": "{v["output_type"]}"}},')
        print("}")
    else:
        print("RELIABILITY = {}   # nothing measured well enough to emit")

    if skipped:
        print("\n# left out, and why:")
        for key, finding, reason in skipped:
            print(f"#   ({key}, {finding}) -- {reason}")

    print("\nREPORT_RELIABILITY: Dict[str, Dict[str, float]] = {")
    for finding in measured:
        rates = report_rates(records, finding)
        if not rates["asserted"]:
            print(f'#   "{finding}": never asserted in any report -- nothing to measure')
            continue
        print(f'    "{finding}":{" " * max(0, 18 - len(finding))} '
              f'{{"recall": {rates["recall"]:.2f}, "precision": {rates["precision"]:.2f}}},'
              f'   # asserted {rates["asserted"]}x'
              + (f", {rates['contradicts']} self-contradictory" if rates["contradicts"] else ""))
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
