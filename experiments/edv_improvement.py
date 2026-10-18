"""edv_improvement_protocol.md: does EDV improve condition detection in MedGemma reports?

    python experiments/edv_improvement.py maira_dev   # MAIRA-2 seven-condition scores, NIH development split (percentile reference)
    python experiments/edv_improvement.py generate    # base MedGemma 1.5 reports, NIH test split (resumable)
    python experiments/edv_improvement.py read        # reports read for the 13 conditions (quote-checked lookup, resumable)
    python experiments/edv_improvement.py analyze     # -> results/edv_improvement.md
"""

import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import provisional_findings as P  # noqa: E402
from maira_score_nih import NIH, SENTENCES, auc, sample  # noqa: E402

RESULTS = HERE / "results"
REPORTS = RESULTS / "edv_medgemma_reports.jsonl"
LABELS = RESULTS / "edv_medgemma_labels.jsonl"
MAIRA7_DEV = RESULTS / "maira_score7_nih_dev.jsonl"
REPORT = RESULTS / "edv_improvement.md"
SIX = list(SENTENCES)
SEVEN = list(P.FINDINGS)
CONDITIONS = SIX + SEVEN
SEED, N_BOOT = 20261018, 2000
# Amendment A1 (before any reading): subcutaneous emphysema is not (pulmonary) emphysema
EXTRA_RULES = ["Subcutaneous emphysema (air in the soft tissues) is not emphysema."]


def split_files(which):
    return [f for f, s in sample() if s == which]


def done(path, key):
    return {json.loads(l)[key] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()


# ------------------------------------------------------------------ stages

def maira_dev():
    from PIL import Image

    from maira_score_all_conditions import SENTENCES7
    from maira_score_nih import make_scorer
    todo = [f for f in split_files("dev") if f not in done(MAIRA7_DEV, "id")]
    print(f"MAIRA-2 dev: {len(todo)} to go", flush=True)
    if not todo:
        return
    scorer, start = make_scorer(SENTENCES7), time.time()
    with MAIRA7_DEV.open("a", encoding="utf-8") as out:
        for n, f in enumerate(todo, 1):
            out.write(json.dumps({"id": f, "findings": scorer(Image.open(NIH / f).convert("RGB"))}) + "\n")
            out.flush()
            if n % 200 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"MAIRA-2 dev {n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


def generate():
    from PIL import Image
    prompt = (ROOT / "medrax" / "docs" / "medgemma_report_prompt.txt").read_text(encoding="utf-8").strip()
    todo = [f for f in split_files("test") if f not in done(REPORTS, "file")]
    print(f"MedGemma: {len(todo)} to go", flush=True)
    start = time.time()
    with REPORTS.open("a", encoding="utf-8") as out:
        for n, f in enumerate(todo, 1):
            image = Image.open(NIH / f).convert("RGB")
            image.thumbnail((1024, 1024))
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            body = {"model": "medgemma1.5:4b", "temperature": 0, "max_tokens": 450, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
                {"type": "text", "text": prompt}]}]}
            req = urllib.request.Request("http://localhost:11434/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
            out.write(json.dumps({"file": f, "report": text}) + "\n")
            out.flush()
            if n % 50 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"MedGemma {n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


def read():
    from comparison_edv_pdf import NOT_IN_GT, ground_truth_locations, make_extractor
    extractor = make_extractor("qwen3.8-edv:latest")
    reports = [json.loads(l) for l in REPORTS.read_text(encoding="utf-8").splitlines()]
    todo = [r for r in reports if r["file"] not in done(LABELS, "file")]
    print(f"read: {len(todo)} to go", flush=True)
    start = time.time()
    with LABELS.open("a", encoding="utf-8") as out:
        for n, r in enumerate(todo, 1):
            ans = ground_truth_locations(extractor, r["report"], CONDITIONS, with_evidence=True, extra_rules=EXTRA_RULES)
            out.write(json.dumps({"file": r["file"], "labels": {
                c: {"answer": a[0], "evidence": a[1],
                    "score": 0.0 if a[0] == NOT_IN_GT else 0.5 if "(possible)" in a[0] else 1.0}
                for c, a in ans.items()}}) + "\n")
            out.flush()
            if n % 25 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"read {n}/{len(todo)}  {rate:.3f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


# ------------------------------------------------------------------ analysis

def load():
    """truth, medgemma, chexagent, maira as {(file, condition): value} over the test split, plus
    development-split score lists per (model, condition) for the percentiles."""
    _, nih = P.sample()
    split = dict(sample())
    truth, chex, maira = {}, {}, {}
    dev = {("chex", c): [] for c in CONDITIONS} | {("maira", c): [] for c in CONDITIONS}
    for r in map(json.loads, (RESULTS / "ppresent_nih.jsonl").read_text(encoding="utf-8").splitlines()):
        if r["set"] != "random":
            continue
        if split[r["file"]] == "test":
            truth[(r["file"], r["finding"])] = r["truth"]
            chex[(r["file"], r["finding"])] = r["p_present"]
        else:
            dev[("chex", r["finding"])].append(r["p_present"])
    for r in map(json.loads, (RESULTS / "provisional_readings.jsonl").read_text(encoding="utf-8").splitlines()):
        if r["p_present"] is None:
            continue
        if split[r["file"]] == "test":
            truth[(r["file"], r["finding"])] = int(P.FINDINGS[r["finding"]] in nih[r["file"]])
            chex[(r["file"], r["finding"])] = r["p_present"]
        else:
            dev[("chex", r["finding"])].append(r["p_present"])
    for r in map(json.loads, (RESULTS / "maira_score_nih.jsonl").read_text(encoding="utf-8").splitlines()):
        for c, v in r["findings"].items():
            (maira.__setitem__((r["file"], c), v["score"]) if r["split"] == "test" else dev[("maira", c)].append(v["score"]))
    for path, is_test in ((RESULTS / "maira_score7_nih_test.jsonl", True), (MAIRA7_DEV, False)):
        for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            for c, v in r["findings"].items():
                (maira.__setitem__((r["id"], c), v["score"]) if is_test else dev[("maira", c)].append(v["score"]))
    medgemma = {(r["file"], c): v["score"] for r in map(json.loads, LABELS.read_text(encoding="utf-8").splitlines())
                for c, v in r["labels"].items()}
    return truth, medgemma, chex, maira, {k: np.sort(v) for k, v in dev.items()}


def analyze():
    from chexagent_reliability import decide
    from medrax.evidence import EvidenceEngine

    truth, mg, chex, maira, dev = load()
    files = split_files("test")
    missing = [f for f in files if (f, CONDITIONS[0]) not in mg]
    if missing:
        raise SystemExit(f"{len(missing)} test images without MedGemma labels")
    pct = lambda model, c, s: np.searchsorted(dev[(model, c)], s, side="right") / len(dev[(model, c)])  # noqa: E731
    F = len(files)
    arr = lambda d: {c: np.array([d[(f, c)] for f in files]) for c in CONDITIONS}  # noqa: E731
    y, m, ch, ma = arr(truth), arr(mg), arr(chex), arr(maira)
    systems = {"MedGemma alone": m,
               "MedGemma + CheXagent": {c: 0.5 * m[c] + 0.5 * pct("chex", c, ch[c]) for c in CONDITIONS},
               "MedGemma + MAIRA-2": {c: 0.5 * m[c] + 0.5 * pct("maira", c, ma[c]) for c in CONDITIONS}}
    main = [c for c in CONDITIONS if y[c].sum() >= 10]

    def aucs(idx):
        return {s: {c: auc(v[c][idx][y[c][idx] == 1], v[c][idx][y[c][idx] == 0]) for c in CONDITIONS}
                for s, v in systems.items()}
    full = aucs(np.arange(F))
    rng = np.random.default_rng(SEED)
    boots = [aucs(rng.integers(0, F, F)) for _ in range(N_BOOT)]
    mean = lambda a, s: np.nanmean([a[s][c] for c in main])  # noqa: E731
    ci = lambda vals: np.percentile([v for v in vals if np.isfinite(v)], [2.5, 97.5])  # noqa: E731

    def diff(s1, s2, c=None):
        point = (mean(full, s1) - mean(full, s2)) if c is None else full[s1][c] - full[s2][c]
        vals = [(mean(b, s1) - mean(b, s2)) if c is None else b[s1][c] - b[s2][c] for b in boots]
        lo, hi = ci(vals)
        return point, lo, hi

    def verdict(lo, hi):
        return "supported" if lo > 0 else "reversed" if hi < 0 else "not supported"
    lines = ["# Does EDV improve condition detection in MedGemma reports?", "",
             f"Protocol: `edv_improvement_protocol.md` (committed before the reports). Base MedGemma 1.5 reports on the {F} "
             f"NIH test images, read for 13 conditions. Paired 95% CIs from {N_BOOT} image-level bootstrap resamples. "
             f"Mean AUC over the {len(main)} conditions with at least 10 positives.", "",
             "## Primary: mean AUC", "", "| system | mean AUC [95% CI] |", "|---|---|"]
    for s in systems:
        lo, hi = ci([mean(b, s) for b in boots])
        lines.append(f"| {s} | {mean(full, s):.3f} [{lo:.3f}-{hi:.3f}] |")
    lines += ["", "| hypothesis | difference [95% CI] | result |", "|---|---|---|"]
    for name, s1, s2 in (("H1: MedGemma + CheXagent minus MedGemma alone", "MedGemma + CheXagent", "MedGemma alone"),
                         ("H2: MedGemma + CheXagent minus MedGemma + MAIRA-2", "MedGemma + CheXagent", "MedGemma + MAIRA-2"),
                         ("(also) MedGemma + MAIRA-2 minus MedGemma alone", "MedGemma + MAIRA-2", "MedGemma alone")):
        d, lo, hi = diff(s1, s2)
        lines.append(f"| {name} | {d:+.3f} [{lo:+.3f} to {hi:+.3f}] | {verdict(lo, hi)} |")
    lines += ["", "## Per condition (AUC)", "",
              "| condition | positives / n | MedGemma alone | + CheXagent | + MAIRA-2 | CheXagent minus alone [95% CI] | "
              "CheXagent minus MAIRA-2 [95% CI] |", "|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        d1 = diff("MedGemma + CheXagent", "MedGemma alone", c)
        d2 = diff("MedGemma + CheXagent", "MedGemma + MAIRA-2", c)
        flag = "" if c in main else " (fewer than 10 positives; not in the mean)"
        lines.append(f"| {c}{flag} | {int(y[c].sum())} / {F} | {full['MedGemma alone'][c]:.3f} | "
                     f"{full['MedGemma + CheXagent'][c]:.3f} | {full['MedGemma + MAIRA-2'][c]:.3f} | "
                     f"{d1[0]:+.3f} [{d1[1]:+.3f} to {d1[2]:+.3f}] | {d2[0]:+.3f} [{d2[1]:+.3f} to {d2[2]:+.3f}] |")

    # secondary: the [Image check] rule as yes/no decisions
    engine = EvidenceEngine()
    dec = {c: np.array([decide(engine, c, v) for v in ch[c]]) for c in CONDITIONS}
    alone = {c: m[c] >= 0.5 for c in CONDITIONS}
    rule = {c: np.where(dec[c] == "AGAINST", False, alone[c] | ((dec[c] == "FOUND") & (c in ("pleural effusion", "cardiomegaly"))))
            for c in CONDITIONS}

    def rates(pred, idx):
        tp = sum(int((pred[c][idx] & (y[c][idx] == 1)).sum()) for c in CONDITIONS)
        fn = sum(int((~pred[c][idx] & (y[c][idx] == 1)).sum()) for c in CONDITIONS)
        tn = sum(int((~pred[c][idx] & (y[c][idx] == 0)).sum()) for c in CONDITIONS)
        fp = sum(int((pred[c][idx] & (y[c][idx] == 0)).sum()) for c in CONDITIONS)
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        return sens, spec, (sens + spec) / 2
    full_r = {"MedGemma alone": rates(alone, np.arange(F)), "MedGemma + [Image check] rule": rates(rule, np.arange(F))}
    rng = np.random.default_rng(SEED + 1)
    bdiff = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, F, F)
        bdiff.append(np.array(rates(rule, idx)) - np.array(rates(alone, idx)))
    bdiff = np.array(bdiff)
    lines += ["", "## Secondary: the [Image check] rule as yes/no decisions (all 13 conditions pooled)", "",
              "| system | sensitivity | specificity | balanced accuracy |", "|---|---|---|---|"]
    for s, (se, sp, ba) in full_r.items():
        lines.append(f"| {s} | {se:.3f} | {sp:.3f} | {ba:.3f} |")
    d = np.array(full_r["MedGemma + [Image check] rule"]) - np.array(full_r["MedGemma alone"])
    lo, hi = np.percentile(bdiff, [2.5, 97.5], axis=0)
    lines.append(f"| difference [95% CI] | {d[0]:+.3f} [{lo[0]:+.3f} to {hi[0]:+.3f}] | {d[1]:+.3f} [{lo[1]:+.3f} to "
                 f"{hi[1]:+.3f}] | {d[2]:+.3f} [{lo[2]:+.3f} to {hi[2]:+.3f}] |")
    lines += ["", "Limitations (recorded in the protocol): MedGemma's score has three values, which caps its AUC; "
              "NIH labels are noisy; CheXagent was probably trained on NIH images; base MedGemma, not the team's model."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"maira_dev": maira_dev, "generate": generate, "read": read, "analyze": analyze}[sys.argv[1]]()
