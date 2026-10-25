"""edv_trained_protocol.md: does EDV improve the team's trained MedGemma? NIH longitudinal pairs.

    .venv-medgemma/Scripts/python.exe experiments/edv_trained.py generate   # trained-model reports (resumable)
    .venv/Scripts/python.exe experiments/edv_trained.py read                # 13 conditions (resumable)
    .venv/Scripts/python.exe experiments/edv_trained.py analyze             # -> results/edv_trained.md
"""

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PAIRS = RESULTS / "edv_long_pairs.json"
REPORTS = RESULTS / "edv_trained_reports.jsonl"
LABELS = RESULTS / "edv_trained_labels.jsonl"
REPORT = RESULTS / "edv_trained.md"
NIH = Path("D:/datasets/chestxray14/images")
MODEL = "D:/model-weights/medgemma_trained_e346cb1"  # vaniinfoorg/medgemma_trained @ e346cb15
PRIOR_REPORT, INDICATION = "Not available.", "Not provided."
SEED, N_BOOT = 20261025, 2000
CHANGE = re.compile(r"compar|interval|prior|previous|unchanged|stable|\bnew\b|improv|worsen|increas|decreas|resolv", re.I)


def done(path, key="current"):
    return {json.loads(l)[key] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()


def generate():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    todo = [p for p in pairs if p["current"] not in done(REPORTS)]
    print(f"trained MedGemma: {len(todo)} to go", flush=True)
    if not todo:
        return
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    start = time.time()
    with REPORTS.open("a", encoding="utf-8") as out:
        for n, p in enumerate(todo, 1):
            msgs = [{"role": "user", "content": [
                {"type": "text", "text": "Prior chest X-ray:"},
                {"type": "image", "image": Image.open(NIH / p["prior"]).convert("RGB")},
                {"type": "text", "text": f"Prior report:\n{PRIOR_REPORT}"},
                {"type": "text", "text": "Current chest X-ray:"},
                {"type": "image", "image": Image.open(NIH / p["current"]).convert("RGB")},
                {"type": "text", "text": f"Clinical indication for current study: {INDICATION}\n\n"
                                         "Compare the current chest X-ray to the prior chest X-ray. "
                                         "Describe any interval change in findings and provide an impression."}]}]
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt").to("cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                ids = model.generate(**inp, max_new_tokens=512, do_sample=False)
            text = proc.decode(ids[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            out.write(json.dumps({"current": p["current"], "prior": p["prior"], "report": text}) + "\n")
            out.flush()
            if n % 25 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"generate {n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


def read():
    sys.path[:0] = [str(HERE.parent), str(HERE)]
    import edv_improvement as E
    from comparison_edv_pdf import NOT_IN_GT, ground_truth_locations, make_extractor
    extractor = make_extractor("qwen3.8-edv:latest")
    reports = [json.loads(l) for l in REPORTS.read_text(encoding="utf-8").splitlines()]
    todo = [r for r in reports if r["current"] not in done(LABELS)]
    print(f"read: {len(todo)} to go", flush=True)
    start = time.time()
    with LABELS.open("a", encoding="utf-8") as out:
        for n, r in enumerate(todo, 1):
            ans = ground_truth_locations(extractor, r["report"], E.CONDITIONS, with_evidence=True,
                                         extra_rules=E.EXTRA_RULES)
            out.write(json.dumps({"current": r["current"], "labels": {
                c: {"answer": a[0], "evidence": a[1],
                    "score": 0.0 if a[0] == NOT_IN_GT else 0.5 if "(possible)" in a[0] else 1.0}
                for c, a in ans.items()}}) + "\n")
            out.flush()
            if n % 25 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"read {n}/{len(todo)}  {rate:.3f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


def analyze():
    import numpy as np
    sys.path[:0] = [str(HERE.parent), str(HERE)]
    import edv_improvement as E
    import edv_longitudinal as L
    from chexagent_reliability import decide
    from maira_score_nih import auc
    from medrax.evidence import EvidenceEngine
    C = E.CONDITIONS
    truth, _, chex_s, _, _ = E.load()
    files = [p["current"] for p in json.loads(PAIRS.read_text(encoding="utf-8"))]
    jl = lambda path: {json.loads(l)["current"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()}  # noqa: E731
    a, pr, ml, base, lab, rep = jl(L.CHEX_A), jl(L.CHEX_PRIOR), jl(L.MAIRA_L), jl(L.LABELS), jl(LABELS), jl(REPORTS)
    base_rep = jl(L.REPORTS)
    if len(lab) < len(files):
        raise SystemExit(f"{len(files) - len(lab)} pairs without trained-model labels")
    raw = {
        "T": {c: [lab[f]["labels"][c]["score"] for f in files] for c in C},
        "L": {c: [base[f]["labels"][c]["score"] for f in files] for c in C},
        "A": {c: [a[f]["p_present"][c] for f in files] for c in C},
        "B": {c: [None if pr[f]["p_present"][c] is None else (chex_s[(f, c)] + pr[f]["p_present"][c]) / 2 for f in files]
              for c in C},
        "S": {c: [chex_s[(f, c)] for f in files] for c in C},
        "M": {c: [ml[f]["findings"][c]["score"] for f in files] for c in C},
    }
    keep = {c: np.array([v is not None for v in raw["A"][c]]) & np.array([v is not None for v in raw["B"][c]]) for c in C}
    y = {c: np.array([truth[(f, c)] for f in files])[keep[c]] for c in C}
    vals = {s: {c: np.array([v for v, k in zip(raw[s][c], keep[c]) if k], float) for c in C} for s in raw}
    pct = lambda v: np.searchsorted(np.sort(v), v, side="right") / len(v)  # noqa: E731
    systems = {
        "Trained MedGemma alone": vals["T"],
        "Trained + CheXagent A": {c: 0.5 * vals["T"][c] + 0.5 * pct(vals["A"][c]) for c in C},
        "Trained + CheXagent B": {c: 0.5 * vals["T"][c] + 0.5 * pct(vals["B"][c]) for c in C},
        "Trained + CheXagent single-image": {c: 0.5 * vals["T"][c] + 0.5 * pct(vals["S"][c]) for c in C},
        "Trained + MAIRA-2-L": {c: 0.5 * vals["T"][c] + 0.5 * pct(vals["M"][c]) for c in C},
        "Base MedGemma-L alone": vals["L"],
        "Base MedGemma-L + CheXagent A": {c: 0.5 * vals["L"][c] + 0.5 * pct(vals["A"][c]) for c in C},
    }
    main = [c for c in C if y[c].sum() >= 10]
    n_c = {c: len(y[c]) for c in C}

    def aucs(idx):
        return {s: {c: auc(v[c][idx[c]][y[c][idx[c]] == 1], v[c][idx[c]][y[c][idx[c]] == 0]) for c in C}
                for s, v in systems.items()}
    full = aucs({c: np.arange(n_c[c]) for c in C})
    rng = np.random.default_rng(SEED)
    pos_of = {c: np.cumsum(keep[c]) - 1 for c in C}
    idx_boots = []
    for _ in range(N_BOOT):
        b = rng.integers(0, len(files), len(files))
        idx_boots.append({c: pos_of[c][b][keep[c][b]] for c in C})
    boots = [aucs(ib) for ib in idx_boots]
    mean = lambda d, s: np.nanmean([d[s][c] for c in main])  # noqa: E731
    verdict = lambda lo, hi: "supported" if lo > 0 else "reversed" if hi < 0 else "not supported"  # noqa: E731

    def diff(s1, s2):
        v = [mean(b, s1) - mean(b, s2) for b in boots]
        lo, hi = np.percentile([x for x in v if np.isfinite(x)], [2.5, 97.5])
        return mean(full, s1) - mean(full, s2), lo, hi

    lines = ["# Does EDV improve the team's trained MedGemma? NIH longitudinal results", "",
             f"Protocol: `edv_trained_protocol.md` (committed before the test reports). {len(files)} NIH test pairs; "
             f"trained MedGemma with placeholders for prior report and indication. Mean AUC over the {len(main)} "
             f"conditions with at least 10 positives; paired 95% CIs from {N_BOOT} image-level bootstrap resamples "
             f"(seed {SEED}).", "", "## Mean AUC", "", "| system | mean AUC [95% CI] |", "|---|---|"]
    for s in systems:
        lo, hi = np.percentile([mean(b, s) for b in boots], [2.5, 97.5])
        lines.append(f"| {s} | {mean(full, s):.3f} [{lo:.3f}-{hi:.3f}] |")
    lines += ["", "## Hypotheses and exploratory comparisons", "", "| comparison | difference [95% CI] | result |",
              "|---|---|---|"]
    for name, s1, s2 in (
            ("H3 (primary): trained + CheXagent A minus trained alone", "Trained + CheXagent A", "Trained MedGemma alone"),
            ("H3b: trained + CheXagent A minus trained + MAIRA-2-L", "Trained + CheXagent A", "Trained + MAIRA-2-L"),
            ("(exploratory) trained alone minus base MedGemma-L alone", "Trained MedGemma alone", "Base MedGemma-L alone"),
            ("(exploratory) trained + CheXagent A minus base + CheXagent A", "Trained + CheXagent A",
             "Base MedGemma-L + CheXagent A"),
            ("(exploratory) trained + CheXagent B minus trained alone", "Trained + CheXagent B", "Trained MedGemma alone"),
            ("(exploratory) trained + CheXagent single-image minus trained alone", "Trained + CheXagent single-image",
             "Trained MedGemma alone")):
        d, lo, hi = diff(s1, s2)
        lines.append(f"| {name} | {d:+.3f} [{lo:+.3f} to {hi:+.3f}] | {verdict(lo, hi)} |")

    # Secondary: the [Image check] rule as yes/no decisions on the trained model's report
    engine = EvidenceEngine()
    stated = {c: vals["T"][c] >= 0.5 for c in C}
    dec = {c: np.array([decide(engine, c, p) for p in vals["A"][c]]) for c in C}
    revised = {c: (stated[c] & (dec[c] != "AGAINST")) | ((dec[c] == "FOUND") & (c in ("pleural effusion", "cardiomegaly")))
               for c in C}

    def rates(said, idx):
        se = [said[c][idx[c]][y[c][idx[c]] == 1].mean() for c in main]
        sp = [(~said[c][idx[c]][y[c][idx[c]] == 0]).mean() for c in main]
        return np.mean(se), np.mean(sp), np.mean([(s1 + s2) / 2 for s1, s2 in zip(se, sp)])
    f0 = rates(stated, {c: np.arange(n_c[c]) for c in C})
    f1 = rates(revised, {c: np.arange(n_c[c]) for c in C})
    bd = np.array([np.subtract(rates(revised, ib), rates(stated, ib)) for ib in idx_boots])
    lo, hi = np.percentile(bd, [2.5, 97.5], axis=0)
    lines += ["", "## Secondary: the [Image check] rule (yes/no)", "",
              "| | mean sensitivity | mean specificity | mean balanced accuracy |", "|---|---|---|---|",
              f"| trained MedGemma alone | {f0[0]:.3f} | {f0[1]:.3f} | {f0[2]:.3f} |",
              f"| trained + [Image check] rule | {f1[0]:.3f} | {f1[1]:.3f} | {f1[2]:.3f} |",
              f"| difference [95% CI] | {f1[0] - f0[0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}] | "
              f"{f1[1] - f0[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}] | {f1[2] - f0[2]:+.3f} [{lo[2]:+.3f}, {hi[2]:+.3f}] |"]
    t_change = sum(bool(CHANGE.search(rep[f]["report"])) for f in files)
    b_change = sum(bool(CHANGE.search(base_rep[f]["report"])) for f in files)
    lines += ["", "## Descriptive: reports that compare with the prior", "",
              f"- Trained MedGemma: {t_change} of {len(files)}",
              f"- Base MedGemma (longitudinal run): {b_change} of {len(files)}"]
    show = list(systems)
    lines += ["", "## Per condition (AUC)", "", "| condition | positives / n | " + " | ".join(show) + " |",
              "|---|---|" + "---|" * len(show)]
    for c in C:
        flag = "" if c in main else " (fewer than 10 positives)"
        lines.append(f"| {c}{flag} | {int(y[c].sum())} / {n_c[c]} | " + " | ".join(f"{full[s][c]:.3f}" for s in show) + " |")
    lines += ["", "Limitations (recorded in the protocol): placeholders for prior report and indication; NIH rather "
              "than MIMIC images; NIH labels NLP-mined; CheXagent probably saw NIH images; percentiles from the "
              "evaluation set; MedGemma's 3-level score caps its AUC."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"generate": generate, "read": read, "analyze": analyze}[sys.argv[1]]()
