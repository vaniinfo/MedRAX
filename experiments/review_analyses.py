"""EXPLORATORY analyses requested in internal review of the manuscript (2026-10-25), run after all
pre-registered results were known.

1. Patient audit: NIH development/test patients, overlap, repeat patients, priors in development.
2. Checker-only baselines for the three report-fusion experiments: checker alone, and fused minus
   checker alone (paired), with image-level and patient-clustered bootstrap CIs.
3. Patient-clustered bootstrap for the pre-registered fusion differences and for grounding recall.

    python experiments/review_analyses.py   -> results/review_analyses.md
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import edv_improvement as E  # noqa: E402
import edv_longitudinal as L  # noqa: E402
import grounding_recall as G  # noqa: E402
from maira_score_nih import auc  # noqa: E402

RESULTS = HERE / "results"
OUT = RESULTS / "review_analyses.md"
SEED, N_BOOT = 20261025, 2000
C = E.CONDITIONS


def patients():
    with open("D:/datasets/chestxray14/Data_Entry_2017_v2020.csv", encoding="utf-8") as f:
        return {r["Image Index"]: r["Patient ID"] for r in csv.DictReader(f)}


def clustered_indices(groups, rng):
    """Resample groups (patients) with replacement; return image indices."""
    uniq = sorted(set(groups))
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([members[uniq[j]] for j in pick])


def main():
    pid = patients()
    lines = ["# EXPLORATORY: analyses requested in internal review", "",
             "Run on 2026-10-25 after all pre-registered results were known. Not pre-registered.", ""]

    # ---------------- 1. patient audit
    dev, test = E.split_files("dev"), E.split_files("test")
    pd_, pt = {pid[f] for f in dev}, {pid[f] for f in test}
    pairs = json.loads(L.PAIRS.read_text(encoding="utf-8"))
    prior_in_dev = sum(p["prior"] in set(dev) for p in pairs)
    prior_in_test = sum(p["prior"] in set(test) for p in pairs)
    per_pat_test = defaultdict(int)
    for f in test:
        per_pat_test[pid[f]] += 1
    gcases = G.cases()
    gpat = {pid[c["file"]] for c in gcases}
    lines += ["## 1. Patient audit (NIH)", "",
              "| quantity | value |", "|---|---|",
              f"| development images / unique patients | {len(dev)} / {len(pd_)} |",
              f"| test images / unique patients | {len(test)} / {len(pt)} |",
              f"| patients in both development and test | {len(pd_ & pt)} |",
              f"| test images whose patient also appears in development | {sum(pid[f] in pd_ for f in test)} |",
              f"| test patients with more than one test image | {sum(v > 1 for v in per_pat_test.values())} "
              f"(max {max(per_pat_test.values())} images) |",
              f"| longitudinal pairs (current in test) / unique patients | {len(pairs)} / "
              f"{len({pid[p['current']] for p in pairs})} |",
              f"| prior images that are development images / test images / neither | {prior_in_dev} / {prior_in_test} / "
              f"{len(pairs) - prior_in_dev - prior_in_test} |",
              f"| grounding cases / images / unique patients | {len(gcases)} / {len({c['file'] for c in gcases})} / "
              f"{len(gpat)} |",
              f"| grounding patients who also have a test-split image | {len(gpat & pt)} |", ""]

    # ---------------- data for experiments
    truth, mg, chex, maira, devd = E.load()
    rng = np.random.default_rng(SEED)

    def exp1():
        files = test
        y = {c: np.array([truth[(f, c)] for f in files]) for c in C}
        pct = lambda m, c, s: np.searchsorted(devd[(m, c)], s, side="right") / len(devd[(m, c)])  # noqa: E731
        m = {c: np.array([mg[(f, c)] for f in files]) for c in C}
        ch = {c: np.array([chex[(f, c)] for f in files]) for c in C}
        ma = {c: np.array([maira[(f, c)] for f in files]) for c in C}
        sy = {"report alone": m, "CheXagent alone": ch, "MAIRA-2 alone": ma,
              "report + CheXagent": {c: 0.5 * m[c] + 0.5 * pct("chex", c, ch[c]) for c in C},
              "report + MAIRA-2": {c: 0.5 * m[c] + 0.5 * pct("maira", c, ma[c]) for c in C}}
        return files, y, sy

    def exp_long(which):
        files = [p["current"] for p in pairs]
        jl = lambda path: {json.loads(l)["current"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()}  # noqa: E731
        a, ml = jl(L.CHEX_A), jl(L.MAIRA_L)
        lab = jl(L.LABELS) if which == "base" else jl(RESULTS / "edv_trained_labels.jsonl")
        prior = jl(L.CHEX_PRIOR)
        keep = {c: np.array([a[f]["p_present"][c] is not None and prior[f]["p_present"][c] is not None for f in files])
                for c in C}
        sel = lambda vals, c: np.array([v for v, k in zip(vals, keep[c]) if k], float)  # noqa: E731
        y = {c: sel([truth[(f, c)] for f in files], c) for c in C}
        m = {c: sel([lab[f]["labels"][c]["score"] for f in files], c) for c in C}
        ch = {c: sel([a[f]["p_present"][c] for f in files], c) for c in C}
        ma = {c: sel([ml[f]["findings"][c]["score"] for f in files], c) for c in C}
        pct = lambda v: np.searchsorted(np.sort(v), v, side="right") / len(v)  # noqa: E731
        sy = {"report alone": m, "CheXagent alone": ch, "MAIRA-2 alone": ma,
              "report + CheXagent": {c: 0.5 * m[c] + 0.5 * pct(ch[c]) for c in C},
              "report + MAIRA-2": {c: 0.5 * m[c] + 0.5 * pct(ma[c]) for c in C}}
        kept_files = {c: [f for f, k in zip(files, keep[c]) if k] for c in C}
        return files, y, sy, keep, kept_files

    def evaluate(name, files, y, sy, keep=None):
        n = len(files)
        groups = [pid[f] for f in files]
        main = [c for c in C if y[c].sum() >= 10]
        # map image-level resample indices to each condition's kept rows
        pos_of = {c: (np.cumsum(keep[c]) - 1) if keep is not None else np.arange(n) for c in C}
        kmask = {c: keep[c] if keep is not None else np.ones(n, bool) for c in C}

        def mean_auc(s, idx=None):
            vals = []
            for c in main:
                ii = np.arange(len(y[c])) if idx is None else pos_of[c][idx][kmask[c][idx]]
                vals.append(auc(sy[s][c][ii][y[c][ii] == 1], sy[s][c][ii][y[c][ii] == 0]))
            return np.nanmean(vals)
        img_boots = [rng.integers(0, n, n) for _ in range(N_BOOT)]
        pat_boots = [clustered_indices(groups, rng) for _ in range(N_BOOT)]
        full = {s: mean_auc(s) for s in sy}
        out = [f"### {name}", "", f"{n} images, {len(set(groups))} patients; mean over {len(main)} conditions with "
               f"at least 10 positives.", "", "| system | mean AUC | 95% CI (image) | 95% CI (patient-clustered) |",
               "|---|---|---|---|"]
        cache = {}
        for s in sy:
            cache[s] = ([mean_auc(s, b) for b in img_boots], [mean_auc(s, b) for b in pat_boots])
            (li, hi_), (lp, hp) = np.percentile(cache[s][0], [2.5, 97.5]), np.percentile(cache[s][1], [2.5, 97.5])
            out.append(f"| {s} | {full[s]:.3f} | {li:.3f}-{hi_:.3f} | {lp:.3f}-{hp:.3f} |")
        out += ["", "| comparison | difference | 95% CI (image) | 95% CI (patient-clustered) |", "|---|---|---|---|"]
        for s1, s2 in (("report + CheXagent", "report alone"), ("report + CheXagent", "report + MAIRA-2"),
                       ("report + CheXagent", "CheXagent alone"), ("report + MAIRA-2", "MAIRA-2 alone"),
                       ("CheXagent alone", "MAIRA-2 alone")):
            di = np.array(cache[s1][0]) - np.array(cache[s2][0])
            dp = np.array(cache[s1][1]) - np.array(cache[s2][1])
            li, hi_ = np.percentile(di[np.isfinite(di)], [2.5, 97.5])
            lp, hp = np.percentile(dp[np.isfinite(dp)], [2.5, 97.5])
            out.append(f"| {s1} minus {s2} | {full[s1] - full[s2]:+.3f} | {li:+.3f} to {hi_:+.3f} | {lp:+.3f} to {hp:+.3f} |")
        # conditions where the report score is constant (fusion ranking then comes from the checker only)
        const = [c for c in main if len(set(sy["report alone"][c])) == 1]
        out += ["", f"Conditions (of the {len(main)} averaged) where the report score is the same for every image: "
                f"{', '.join(const) if const else 'none'}.", ""]
        return out

    lines += ["## 2-3. Checker-only baselines and patient-clustered intervals", ""]
    f1, y1, s1 = exp1()
    lines += evaluate("Experiment 1: base MedGemma, single image (1,500 test images)", f1, y1, s1)
    for which, title in (("base", "Experiment 2: base MedGemma, prior + current (499 pairs)"),
                         ("trained", "Experiment 3: trained MedGemma, prior + current (499 pairs)")):
        f, y, sy, keep, _ = exp_long(which)
        lines += evaluate(title, f, y, sy, keep)

    # ---------------- grounding: patient-clustered intervals
    rec = {m: {(json.loads(l)["file"], json.loads(l)["label"]): json.loads(l)
               for l in p.read_text(encoding="utf-8").splitlines()} for m, p in G.OUT.items()}
    M = {m: [G.measures(rec[m][(c["file"], c["label"])], c) for c in gcases] for m in G.OUT}
    groups = [pid[c["file"]] for c in gcases]
    lines += ["## Grounding recall: patient-clustered bootstrap (findings of one patient kept together)", "",
              "| threshold | CheXagent minus MAIRA-2 | 95% CI (patient-clustered) |", "|---|---|---|"]
    for t in (0.5, 0.3):
        a = np.array([x["best"] >= t for x in M["CheXagent"]])
        b = np.array([x["best"] >= t for x in M["MAIRA-2"]])
        bs = [a[i].mean() - b[i].mean() for i in (clustered_indices(groups, rng) for _ in range(N_BOOT))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lines.append(f"| IoU {t} | {a.mean() - b.mean():+.1%} | {lo:+.1%} to {hi:+.1%} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
