"""EXPLORATORY sensitivity analysis of the E4 editing rule, requested in the third internal review (2026-10-25),
run after all pre-registered results were known.

The reported rule (chexagent_reliability.decide) has two parts that are not the frozen Evidence Engine v1.0:
  - provisional states for seven conditions, whose use was decided by a check on the NIH test split (the E4
    test set);
  - a fallback: when the engine returns INSUFFICIENT DATA because p_present is outside the development
    support, p >= 0.5 counts as FOUND and p < 0.5 as AGAINST.
Variants: V0 as reported; V1 without provisional states (UNCERTAIN for the seven); V2 without the fallback
(OUTSIDE_SUPPORT -> UNCERTAIN); V3 without both (frozen engine only).

Each setting is summarised as in its protocol (a: pooled over all 13 conditions; c: mean over the 9
conditions with at least 10 positives), and also the other way for comparison.

    python experiments/review_editing_rule.py   -> results/review_editing_rule.md
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import edv_improvement as E  # noqa: E402
import edv_longitudinal as L  # noqa: E402
import provisional_findings as P  # noqa: E402
from chexagent_reliability import SIX  # noqa: E402
from medrax.evidence import EvidenceEngine  # noqa: E402

RESULTS = HERE / "results"
OUT = RESULTS / "review_editing_rule.md"
SEED, N_BOOT = 20261028, 2000
C = E.CONDITIONS
ADD = ("pleural effusion", "cardiomegaly")
ENGINE = EvidenceEngine()


def decide(c, p, provisional=True, fallback=True):
    """(decision, source) with source in engine / fallback / provisional / none."""
    if c in SIX:
        ev = ENGINE.assess(c, p)
        if ev.state == "SUPPORTING":
            return "FOUND", "engine"
        if ev.state == "REFUTING":
            return "AGAINST", "engine"
        if ev.reason == "OUTSIDE_SUPPORT" and fallback:
            return ("FOUND" if p >= 0.5 else "AGAINST"), "fallback"
        return "UNCERTAIN", "none"
    if not provisional:
        return "UNCERTAIN", "none"
    state = (P.assess(c, p) or ("UNCERTAIN", c))[0]
    d = {"SUPPORTING": "FOUND", "REFUTING": "AGAINST"}.get(state, "UNCERTAIN")
    return d, ("provisional" if d != "UNCERTAIN" else "none")


def setting_a():
    truth, mg, chex, _, _ = E.load()
    files = E.split_files("test")
    y = {c: np.array([truth[(f, c)] for f in files]) for c in C}
    stated = {c: np.array([mg[(f, c)] for f in files]) >= 0.5 for c in C}
    p = {c: np.array([chex[(f, c)] for f in files]) for c in C}
    return "(a) base MedGemma, single image (1,500)", y, stated, p, "pooled"


def setting_c():
    truth = E.load()[0]
    files = [x["current"] for x in json.loads(L.PAIRS.read_text(encoding="utf-8"))]
    jl = lambda path: {json.loads(x)["current"]: json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()}  # noqa: E731
    a, pr, lab = jl(L.CHEX_A), jl(L.CHEX_PRIOR), jl(RESULTS / "edv_trained_labels.jsonl")
    keep = {c: np.array([a[f]["p_present"][c] is not None and pr[f]["p_present"][c] is not None for f in files])
            for c in C}
    sel = lambda vals, c: np.array([v for v, k in zip(vals, keep[c]) if k], float)  # noqa: E731
    y = {c: sel([truth[(f, c)] for f in files], c) for c in C}
    stated = {c: sel([lab[f]["labels"][c]["score"] for f in files], c) >= 0.5 for c in C}
    p = {c: sel([a[f]["p_present"][c] for f in files], c) for c in C}
    return "(c) trained MedGemma, prior + current (499)", y, stated, p, "mean over main"


def summary(said, y, idx, how, main):
    if how == "pooled":
        tp = sum(int((said[c][idx[c]] & (y[c][idx[c]] == 1)).sum()) for c in C)
        fn = sum(int((~said[c][idx[c]] & (y[c][idx[c]] == 1)).sum()) for c in C)
        tn = sum(int((~said[c][idx[c]] & (y[c][idx[c]] == 0)).sum()) for c in C)
        fp = sum(int((said[c][idx[c]] & (y[c][idx[c]] == 0)).sum()) for c in C)
        se, sp = tp / (tp + fn), tn / (tn + fp)
        return np.array([se, sp, (se + sp) / 2])
    se = np.mean([said[c][idx[c]][y[c][idx[c]] == 1].mean() for c in main])
    sp = np.mean([(~said[c][idx[c]][y[c][idx[c]] == 0]).mean() for c in main])
    return np.array([se, sp, (se + sp) / 2])


def analyse(name, y, stated, p, own):
    main = [c for c in C if y[c].sum() >= 10]
    n = {c: len(y[c]) for c in C}
    rng = np.random.default_rng(SEED)
    # one image-level resample, mapped to each condition's rows (all conditions share the image list in a;
    # in c the kept rows differ per condition, so resample rows per condition with a shared seed per draw)
    boots = []
    for _ in range(N_BOOT):
        seed = int(rng.integers(0, 2**31))
        if len(set(n.values())) == 1:  # setting a: one image index shared by all conditions
            shared = np.random.default_rng(seed).integers(0, n[C[0]], n[C[0]])
            boots.append({c: shared for c in C})
        else:
            boots.append({c: np.random.default_rng(seed).integers(0, n[c], n[c]) for c in C})
    every = {c: np.arange(n[c]) for c in C}
    lines = [f"## {name}", ""]
    variants = {"V0 as reported": (True, True), "V1 no provisional states": (False, True),
                "V2 no fallback": (True, False), "V3 frozen engine only": (False, False)}
    for how in (own, "pooled" if own != "pooled" else "mean over main"):
        tag = "protocol summary" if how == own else "other summary, for comparison"
        lines += [f"### {how} ({tag})", "",
                  "| variant | sensitivity | specificity | balanced accuracy | change in balanced accuracy [95% CI] |",
                  "|---|---|---|---|---|"]
        base = summary(stated, y, every, how, main)
        bb = [summary(stated, y, b, how, main) for b in boots]
        lines.append(f"| report alone | {base[0]:.3f} | {base[1]:.3f} | {base[2]:.3f} | |")
        for v, (prov, fb) in variants.items():
            dec = {c: np.array([decide(c, x, prov, fb)[0] for x in p[c]]) for c in C}
            said = {c: (stated[c] & (dec[c] != "AGAINST")) | ((dec[c] == "FOUND") & (c in ADD)) for c in C}
            r = summary(said, y, every, how, main)
            d = np.array([summary(said, y, b, how, main)[2] - x[2] for b, x in zip(boots, bb)])
            lo, hi = np.percentile(d, [2.5, 97.5])
            lines.append(f"| {v} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | {r[2] - base[2]:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        lines.append("")
    # which part of the rule made each change (V0)
    counts = {}
    for c in C:
        for x, st, t in zip(p[c], stated[c], y[c]):
            d, src = decide(c, x)
            act = "removed" if (st and d == "AGAINST") else "added" if (not st and d == "FOUND" and c in ADD) else None
            if act:
                k = (src, act, "correct" if (act == "removed") == (t == 0) else "wrong")
                counts[k] = counts.get(k, 0) + 1
    lines += ["### Changes made by the reported rule (V0), by source", "", "| source | action | correct | wrong |",
              "|---|---|---|---|"]
    for src in ("engine", "fallback", "provisional"):
        for act in ("removed", "added"):
            ok, bad = counts.get((src, act, "correct"), 0), counts.get((src, act, "wrong"), 0)
            if ok or bad:
                lines.append(f"| {src} | {act} | {ok} | {bad} |")
    return lines + [""]


def main():
    lines = ["# EXPLORATORY: sensitivity of the E4 editing rule", "",
             "Run on 2026-10-25 after all pre-registered results were known. Not pre-registered. See the docstring "
             "of `review_editing_rule.py`. A change is correct if it removes a condition the label says is absent, "
             "or adds one the label says is present.", ""]
    for s in (setting_a(), setting_c()):
        lines += analyse(*s)
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
