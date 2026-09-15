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

# (RELIABILITY key, display name, reliability.json key)
TOOLS = [("chest_xray_expert", "CheXagent", "chexagent_p"),
         ("chest_xray_classifier", "classifier", "classifier_p")]
SWEEP = [t / 100 for t in range(5, 100, 5)]


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


def bootstrap_rows(rows, n, rng):
    """Resample images with replacement, recomputing every statistic on each draw.

    Resampling images rather than predictions keeps a patient's CheXagent and
    classifier readings together, which is what makes the paired AUC difference a
    fair test rather than two independent ones compared by eye.
    """
    size = len(rows)
    out = {"chex_auc": [], "clf_auc": [], "chex_thr": [], "clf_thr": [], "delta": []}
    for _ in range(n):
        draw = [rows[rng.randrange(size)] for _ in range(size)]
        truths = [t for _, _, t in draw]
        if not any(truths) or all(truths):
            continue                       # a degenerate draw has no defined AUC
        chex = [(c, t) for c, _, t in draw if c is not None]
        clf = [(k, t) for _, k, t in draw if k is not None]
        if chex:
            out["chex_auc"].append(auc(chex))
            out["chex_thr"].append(best_threshold(chex)[0])
        if clf:
            out["clf_auc"].append(auc(clf))
            out["clf_thr"].append(best_threshold(clf)[0])
        if chex and clf:
            out["delta"].append(auc(chex) - auc(clf))
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

    emit, skipped, notes = {}, [], []
    print(f"\n{'finding':18s} {'n+':>4s} {'tool':11s} {'AUC':>6s} {'95% CI':>14s} "
          f"{'thr':>5s} {'95% CI':>12s} {'bal acc':>8s}")
    for finding in measured:
        rows = [(r["findings"][finding].get("chexagent_p"),
                 r["findings"][finding].get("classifier_p"),
                 bool(r["findings"][finding]["truth"]))
                for r in records if finding in r["findings"]]
        npos = sum(1 for *_, t in rows if t)
        boot = bootstrap_rows(rows, args.bootstrap, rng) if args.bootstrap else None

        for key, label, field in TOOLS:
            pairs = [(v, t) for c, k, t in rows
                     for v in [c if field == "chexagent_p" else k] if v is not None]
            if not pairs or npos == 0:
                continue
            a = auc(pairs)
            thr, _sens, _spec, acc = best_threshold(pairs)
            if boot:
                stem = "chex" if field == "chexagent_p" else "clf"
                a_lo, a_hi = ci(boot[f"{stem}_auc"])
                t_lo, t_hi = ci(boot[f"{stem}_thr"])
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
                emit[(key, finding)] = {"auc": round(a, 3), "threshold": thr,
                                        "thr_ci": None if t_lo is None else (t_lo, t_hi)}

            print(f"{finding:18s} {npos:4d} {label:11s} {a:6.3f} {a_ci:>14s} "
                  f"{thr:5.2f} {t_ci:>12s} {acc:7.1%}{flag}")

        if boot and boot["delta"]:
            d_lo, d_hi = ci(boot["delta"])
            verdict = ("CheXagent better" if d_lo > 0 else
                       "classifier better" if d_hi < 0 else
                       "no separation -- the difference is within noise")
            notes.append(f"  {finding:18s} delta AUC {percentile(boot['delta'], 50):+.3f} "
                         f"(95% CI {d_lo:+.3f} to {d_hi:+.3f})  {verdict}")

    if notes:
        print("\nCheXagent minus classifier, paired on the same resampled images:")
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
        width = max(len(f'    ("{k}", "{f}"):') for k, f in emit)
        print("RELIABILITY: Dict[Tuple[str, str], Dict[str, Any]] = {")
        for (key, finding), v in emit.items():
            head = f'    ("{key}", "{finding}"):'
            ci_text = ("None" if v["thr_ci"] is None
                       else f'({v["thr_ci"][0]:.2f}, {v["thr_ci"][1]:.2f})')
            print(f'{head:<{width}} {{"auc": {v["auc"]:.3f}, '
                  f'"threshold": {v["threshold"]:.2f}, "thr_ci": {ci_text}}},')
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
