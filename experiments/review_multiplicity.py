"""EXPLORATORY multiplicity analyses requested in the second internal review (2026-10-25), run after all
pre-registered results were known.

1. E2 (29 presence comparisons): approximate Holm adjustment. The per-comparison bootstrap draws were not
   stored, so p-values are reconstructed from the reported percentile CIs with a normal approximation:
   SE = (difference - lower bound) / 1.96, i.e. the half-width on the side facing zero; z = difference / SE;
   two-sided p = 2 * (1 - Phi(|z|)). Assumes the bootstrap distribution is roughly normal near zero.
2. E4 (six pre-registered fusion hypotheses: H1, H2 in each of three settings): bootstrap p-values from
   2,000 new image-level resamples (seed below), one-sided p = (1 + #{difference <= 0}) / (B + 1), then Holm
   across the six. The resamples are new draws, so the CIs differ slightly from the pre-registered ones.

    python experiments/review_multiplicity.py   -> results/review_multiplicity.md
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import edv_improvement as E  # noqa: E402
import edv_longitudinal as L  # noqa: E402
from build_detection_grounding_doc import AUC_ROWS  # noqa: E402
from maira_score_nih import auc  # noqa: E402

RESULTS = HERE / "results"
OUT = RESULTS / "review_multiplicity.md"
SEED, N_BOOT, ALPHA = 20261026, 2000, 0.05
C = E.CONDITIONS


def holm(p):
    """Holm-adjusted p-values (step-down, monotone)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def e2_section():
    rows = []
    for r in AUC_ROWS:
        d, lo, hi = (float(x) for x in re.findall(r"[-+]?\d*\.\d+", r[5])[:3])
        se = (d - lo) / 1.96 if d > 0 else (hi - d) / 1.96
        rows.append((r[0], r[1], d, lo, hi, 2 * (1 - norm.cdf(abs(d) / se)), not r[6].startswith("No (fewer")))
    p_all = holm([r[5] for r in rows])
    elig = [i for i, r in enumerate(rows) if r[6]]
    p_elig = dict(zip(elig, holm([rows[i][5] for i in elig])))
    lines = ["## 1. E2: approximate Holm adjustment (p-values reconstructed from CIs)", "",
             "| dataset | condition | difference [95% CI] | approx. p | Holm, all 29 | Holm, comparisons with >= 10 positives |",
             "|---|---|---|---|---|---|"]
    for i, (ds, c, d, lo, hi, p, ok) in enumerate(rows):
        pe = f"{p_elig[i]:.2g}" if ok else "not counted"
        lines.append(f"| {ds} | {c} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.2g} | {p_all[i]:.2g} | {pe} |")
    n_all = int((p_all < ALPHA).sum())
    n_el = sum(p_elig[i] < ALPHA for i in elig)
    lines += ["", f"Significant at 0.05 after Holm: {n_all} of {len(rows)} (all comparisons); {n_el} of {len(elig)} "
              f"(comparisons with at least 10 positives).", ""]
    return lines


def e4_systems(which, truth, mg, chex, maira, devd):
    """(y, report score, fused CheXagent, fused MAIRA-2, keep mask) per condition, as in review_analyses.py."""
    if which == "a":
        files = E.split_files("test")
        pct = lambda m, c, s: np.searchsorted(devd[(m, c)], s, side="right") / len(devd[(m, c)])  # noqa: E731
        y = {c: np.array([truth[(f, c)] for f in files]) for c in C}
        m = {c: np.array([mg[(f, c)] for f in files]) for c in C}
        fc = {c: 0.5 * m[c] + 0.5 * pct("chex", c, np.array([chex[(f, c)] for f in files])) for c in C}
        fm = {c: 0.5 * m[c] + 0.5 * pct("maira", c, np.array([maira[(f, c)] for f in files])) for c in C}
        return len(files), y, m, fc, fm, None
    pairs = json.loads(L.PAIRS.read_text(encoding="utf-8"))
    files = [p["current"] for p in pairs]
    jl = lambda path: {json.loads(x)["current"]: json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()}  # noqa: E731
    a, ml, prior = jl(L.CHEX_A), jl(L.MAIRA_L), jl(L.CHEX_PRIOR)
    lab = jl(L.LABELS) if which == "b" else jl(RESULTS / "edv_trained_labels.jsonl")
    keep = {c: np.array([a[f]["p_present"][c] is not None and prior[f]["p_present"][c] is not None for f in files])
            for c in C}
    sel = lambda vals, c: np.array([v for v, k in zip(vals, keep[c]) if k], float)  # noqa: E731
    pct = lambda v: np.searchsorted(np.sort(v), v, side="right") / len(v)  # noqa: E731
    y = {c: sel([truth[(f, c)] for f in files], c) for c in C}
    m = {c: sel([lab[f]["labels"][c]["score"] for f in files], c) for c in C}
    fc = {c: 0.5 * m[c] + 0.5 * pct(sel([a[f]["p_present"][c] for f in files], c)) for c in C}
    fm = {c: 0.5 * m[c] + 0.5 * pct(sel([ml[f]["findings"][c]["score"] for f in files], c)) for c in C}
    return len(files), y, m, fc, fm, keep


def e4_section():
    data = E.load()
    rng = np.random.default_rng(SEED)
    tests = []
    for which, name in (("a", "(a) base, single image"), ("b", "(b) base, prior + current"),
                        ("c", "(c) trained, prior + current")):
        n, y, m, fc, fm, keep = e4_systems(which, *data)
        main = [c for c in C if y[c].sum() >= 10]
        pos = {c: (np.cumsum(keep[c]) - 1) if keep is not None else np.arange(n) for c in C}
        km = {c: keep[c] if keep is not None else np.ones(n, bool) for c in C}

        def mean_auc(s, idx):
            out = []
            for c in main:
                ii = pos[c][idx][km[c][idx]]
                out.append(auc(s[c][ii][y[c][ii] == 1], s[c][ii][y[c][ii] == 0]))
            return np.nanmean(out)
        every = np.arange(n)
        d1, d2 = [], []
        for _ in range(N_BOOT):
            b = rng.integers(0, n, n)
            fcb = mean_auc(fc, b)
            d1.append(fcb - mean_auc(m, b))
            d2.append(fcb - mean_auc(fm, b))
        for label, s2, ds in (("fused CheXagent minus report alone", m, d1),
                              ("fused CheXagent minus fused MAIRA-2", fm, d2)):
            ds = np.array(ds)
            obs = mean_auc(fc, every) - mean_auc(s2, every)
            tests.append((name, label, obs, *np.percentile(ds, [2.5, 97.5]), int((ds <= 0).sum()),
                          (1 + (ds <= 0).sum()) / (N_BOOT + 1)))
    adj = holm([t[-1] for t in tests])
    lines = ["## 2. E4: bootstrap p-values and Holm across the six pre-registered fusion hypotheses", "",
             f"{N_BOOT} new image-level resamples, seed {SEED}.", "",
             "| setting | comparison | difference [95% CI] | resamples <= 0 | one-sided p | Holm-adjusted p |",
             "|---|---|---|---|---|---|"]
    for t, a in zip(tests, adj):
        lines.append(f"| {t[0]} | {t[1]} | {t[2]:+.3f} [{t[3]:+.3f}, {t[4]:+.3f}] | {t[5]} | {t[6]:.4f} | {a:.4f} |")
    lines += ["", f"The smallest attainable p with {N_BOOT} resamples is {1 / (N_BOOT + 1):.4f}.", ""]
    return lines


def main():
    lines = ["# EXPLORATORY: multiplicity analyses requested in review", "",
             "Run on 2026-10-25 after all pre-registered results were known. Not pre-registered. Methods: see the "
             "docstring of `review_multiplicity.py`.", ""]
    lines += e2_section() + e4_section()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
