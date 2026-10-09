"""Provisional cut-offs for findings beyond the six (design: provisional_findings_design.md).

    python experiments/provisional_findings.py score     # CheXagent readings, NIH random sample (resumable)
    python experiments/provisional_findings.py analyze   # fit on dev, check on test, export cut-offs

Exports experiments/results/provisional_findings_v0.json, used by the review tool through
`assess(finding, p_present)`. Not part of the frozen Evidence Engine v1.0.
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"
READINGS = RESULTS / "provisional_readings.jsonl"
EXPORT = RESULTS / "provisional_findings_v0.json"
REPORT = RESULTS / "provisional_findings.md"
NIH = Path("D:/datasets/chestxray14")
SEED = 20261009
N_BOOT = 2000

# CheXagent question -> NIH label
FINDINGS = {"pneumonia": "Pneumonia", "infiltration": "Infiltration", "emphysema": "Emphysema",
            "nodule": "Nodule", "mass": "Mass", "fibrosis": "Fibrosis",
            "pleural thickening": "Pleural_Thickening"}
# Report wording -> finding it is assessed as (review tool)
TERMS = {"pneumonia": "pneumonia",
         "opacity": "infiltration", "opacities": "infiltration", "infiltrate": "infiltration",
         "infiltrates": "infiltration", "infiltration": "infiltration", "airspace disease": "infiltration",
         "airspace opacity": "infiltration",
         "emphysema": "emphysema", "hyperinflation": "emphysema", "hyperinflated": "emphysema",
         "nodule": "nodule", "nodules": "nodule", "mass": "mass",
         "fibrosis": "fibrosis", "fibrotic": "fibrosis", "pleural thickening": "pleural thickening"}


def sample():
    """(file, split) for the NIH random sample, and NIH labels per file."""
    images = json.loads((RESULTS / "nih_sample_5000.json").read_text(encoding="utf-8"))["images"]
    split = json.loads((RESULTS / "split_nih_70_30.json").read_text(encoding="utf-8"))["random"]
    wanted = {im["file"] for im in images}
    labels = {}
    with (NIH / "Data_Entry_2017_v2020.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Image Index"] in wanted:
                labels[row["Image Index"]] = set(row["Finding Labels"].split("|"))
    return [(im["file"], split[im["file"]]) for im in images], labels


def score():
    rows, _ = sample()
    done = set()
    if READINGS.exists():
        done = {(r["file"], r["finding"]) for r in map(json.loads, READINGS.read_text(encoding="utf-8").splitlines())}
    todo = [(f, q) for f, _ in rows for q in FINDINGS if (f, q) not in done]
    print(f"{len(done)} readings done, {len(todo)} to go", flush=True)
    if not todo:
        return
    from medrax.tools.xray_vqa import XRayVQATool

    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    import time
    start = time.time()
    with READINGS.open("a", encoding="utf-8") as out:
        for n, (f, q) in enumerate(todo, 1):
            text, yes_no = tool._generate_response([str(NIH / "images" / f)], f"Is there {q}?", 8)
            out.write(json.dumps({"file": f, "finding": q, "answer": text,
                                  "p_present": yes_no["p_yes"] if yes_no else None}) + "\n")
            out.flush()
            if n % 300 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"{n}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


def rank_auc(score, y):
    """Mann-Whitney AUC (ties count half)."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def analyze():
    import evidence_engine_d1_d2 as E
    import evidence_engine_d3 as D

    rows, labels = sample()
    split = dict(rows)
    p = {}
    for r in map(json.loads, READINGS.read_text(encoding="utf-8").splitlines()):
        if r["p_present"] is not None:
            p[(r["file"], r["finding"])] = r["p_present"]
    rng = np.random.default_rng(SEED)
    export = {"version": "provisional-v0", "design": "experiments/provisional_findings_design.md",
              "rule": "5 / 0.2, CI-conservative, dual-evidence REFUTING", "findings": {}}
    lines = ["# Provisional cut-offs for findings beyond the six: NIH random sample", "",
             "Design: `provisional_findings_design.md`. Fitted on the development split, checked on the "
             "test split. Not part of Evidence Engine v1.0. Empirical LR = P(state | finding) / P(state | "
             f"no finding), 95% CI from {N_BOOT} image-level bootstrap resamples (seed {SEED}).", "",
             "| finding | dev pos / n | test pos / n | test AUC | SUPPORTING: from p, coverage, LR [CI] | "
             "REFUTING: up to p, coverage, LR [CI], positives in it | used in review |",
             "|---|---|---|---|---|---|---|"]
    for q, label in FINDINGS.items():
        def arrays(which):
            files = [f for f, s in rows if s == which and (f, q) in p]
            return (np.array([p[(f, q)] for f in files]),
                    np.array([int(label in labels[f]) for f in files]))
        dev_p, dev_y = arrays("dev")
        test_p, test_y = arrays("test")
        platt = dict(zip(("coef", "intercept"), E.fit_platt(dev_p, dev_y)))
        source = E.Source("NIH", q, platt, dev_p, dev_y, rng)
        # Empirical development bands, as D3 builds them (here: random development only)
        idx = D.band_of(dev_p)
        n_pos, n_neg = int(dev_y.sum()), int((1 - dev_y).sum())
        bands = []
        for b in range(len(D.BANDS) - 1):
            k_pos, k_neg = int(((idx == b) & (dev_y == 1)).sum()), int(((idx == b) & (dev_y == 0)).sum())
            lower_neg = D.cp_lower(k_neg, n_neg)
            bands.append({"lo": D.BANDS[b], "hi": D.BANDS[b + 1], "k_pos": k_pos, "k_neg": k_neg,
                          "uci": D.cp_upper(k_pos, n_pos) / lower_neg if lower_neg > 0 else None})
        _, states, _ = D.classify(source, test_p, 5.0, 0.2, bands)
        p_support, p_refute = source.cutpoints(5.0, 0.2)

        def lr_of(state, idx_):
            s, y = states[idx_], test_y[idx_]
            pos, neg = (s[y == 1] == state).mean(), (s[y == 0] == state).mean()
            return pos / neg if neg > 0 else (math.inf if pos > 0 else float("nan"))

        all_idx = np.arange(len(test_p))
        boots = [rng.integers(0, len(test_p), len(test_p)) for _ in range(N_BOOT)]
        cell, use = {}, {}
        for state in ("SUPPORTING", "REFUTING"):
            est = lr_of(state, all_idx)
            vals = [v for v in (lr_of(state, b) for b in boots) if np.isfinite(v)]
            lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan")))
            cov = (states == state).mean()
            cell[state] = (est, lo, hi, cov)
        pos_in_refute = int(((states == "REFUTING") & (test_y == 1)).sum())
        use["supporting"] = bool(np.isfinite(cell["SUPPORTING"][0]) and cell["SUPPORTING"][0] >= 5)
        use["refuting"] = bool(np.isfinite(cell["REFUTING"][0]) and cell["REFUTING"][0] <= 0.2
                               and pos_in_refute <= 0.02 * max(test_y.sum(), 1))
        auc = rank_auc(test_p, test_y)
        export["findings"][q] = {
            "nih_label": label, "platt": platt, "dev_prevalence": float(dev_y.mean()), "dev_positives": n_pos,
            "p_support": p_support, "p_refute": p_refute, "bands": bands,
            "support_range": [float(np.sort(dev_p)[E.SUPPORT_MIN - 1]), float(np.sort(dev_p)[-E.SUPPORT_MIN])],
            "use_supporting": use["supporting"], "use_refuting": use["refuting"],
            "test": {"supporting_lr": cell["SUPPORTING"][:3], "supporting_coverage": cell["SUPPORTING"][3],
                     "refuting_lr": cell["REFUTING"][:3], "refuting_coverage": cell["REFUTING"][3],
                     "positives_in_refuting": pos_in_refute, "auc": auc},
        }

        def fmt(c):
            est, lo, hi, cov = c
            return "not observed" if cov == 0 else f"{cov:.1%}, {est:.2f} [{lo:.2f}-{hi:.2f}]"
        lines.append(f"| {q} | {n_pos} / {len(dev_y)} | {int(test_y.sum())} / {len(test_y)} | {auc:.3f} | "
                     f"{'none' if p_support is None else f'{p_support:.3f}'}; {fmt(cell['SUPPORTING'])} | "
                     f"{'none' if p_refute is None else f'{p_refute:.4f}'}; {fmt(cell['REFUTING'])}; {pos_in_refute} | "
                     f"SUPPORTING {'yes' if use['supporting'] else 'no'}, REFUTING {'yes' if use['refuting'] else 'no'} |")
    EXPORT.write_text(json.dumps(export, indent=1), encoding="utf-8")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))


_EXPORT_CACHE = None


def assess(finding, p):
    """(state, provisional finding name) for a report term and its p_present, or None if the term
    has no provisional cut-offs. State: SUPPORTING, REFUTING, UNCERTAIN (incl. states not used in
    the review), or INSUFFICIENT DATA (outside the development score range)."""
    global _EXPORT_CACHE
    q = next((v for k, v in sorted(TERMS.items(), key=lambda kv: -len(kv[0])) if k in finding.lower()), None)
    if q is None or not EXPORT.exists():
        return None
    if _EXPORT_CACHE is None:
        _EXPORT_CACHE = json.loads(EXPORT.read_text(encoding="utf-8"))["findings"]
    f = _EXPORT_CACHE.get(q)
    if f is None:
        return None
    lo, hi = f["support_range"]
    if p < lo or p > hi:
        return "INSUFFICIENT DATA", q
    if f["use_supporting"] and f["p_support"] is not None and p >= f["p_support"]:
        return "SUPPORTING", q
    if f["use_refuting"] and f["p_refute"] is not None and p <= f["p_refute"]:
        band = next(b for b in f["bands"] if b["lo"] <= p < b["hi"] or b["hi"] >= 1 >= p >= b["lo"])
        if band["uci"] is not None and band["uci"] <= 0.2:
            return "REFUTING", q
    return "UNCERTAIN", q


if __name__ == "__main__":
    {"score": score, "analyze": analyze}[sys.argv[1]]()
