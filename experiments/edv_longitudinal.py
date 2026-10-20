"""edv_longitudinal_protocol.md: longitudinal EDV (prior + current) on the 499 NIH test pairs.

    python experiments/edv_longitudinal.py pairs       # the 499 pairs (deterministic) -> results/edv_long_pairs.json
    python experiments/edv_longitudinal.py chexagent   # option A (two-image) + prior-image readings for option B
    python experiments/edv_longitudinal.py maira       # MAIRA-2 with the prior image
    python experiments/edv_longitudinal.py generate    # MedGemma 1.5 longitudinal reports
    python experiments/edv_longitudinal.py read        # reports read for the 13 conditions
    python experiments/edv_longitudinal.py analyze     # -> results/edv_longitudinal.md
"""

import base64
import csv
import io
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import edv_improvement as E  # noqa: E402
from maira_score_nih import NIH, SENTENCES, auc  # noqa: E402

RESULTS = HERE / "results"
PAIRS = RESULTS / "edv_long_pairs.json"
CHEX_A = RESULTS / "edv_long_chexagent_A.jsonl"
CHEX_PRIOR = RESULTS / "edv_long_chexagent_prior.jsonl"
MAIRA_L = RESULTS / "edv_long_maira.jsonl"
REPORTS = RESULTS / "edv_long_medgemma_reports.jsonl"
LABELS = RESULTS / "edv_long_medgemma_labels.jsonl"
REPORT = RESULTS / "edv_longitudinal.md"
CONDITIONS = E.CONDITIONS
SEED, N_BOOT = 20261019, 2000
A_PROMPT = ("The first image is the prior chest X-ray and the second is the current chest X-ray. "
            "Is there {c} in the current chest X-ray?")
LONGITUDINAL = ("The first image is the PRIOR study and the second is the CURRENT study; report on the CURRENT "
                "study and note change from the prior.\n")


def done(path, key):
    return {json.loads(l)[key] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()


def pairs():
    if PAIRS.exists():
        return json.loads(PAIRS.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open("D:/datasets/chestxray14/Data_Entry_2017_v2020.csv", encoding="utf-8")))
    info = {r["Image Index"]: r for r in rows}
    by = defaultdict(list)
    for r in rows:
        by[r["Patient ID"]].append(r)
    out = []
    for f in E.split_files("test"):
        r = info[f]
        earlier = [x for x in by[r["Patient ID"]] if int(x["Follow-up #"]) < int(r["Follow-up #"])]
        if not earlier:
            continue
        same = [x for x in earlier if x["View Position"] == r["View Position"]]
        prior = max(same or earlier, key=lambda x: int(x["Follow-up #"]))
        out.append({"current": f, "prior": prior["Image Index"], "same_view": bool(same)})
    PAIRS.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(out)} pairs -> {PAIRS}")
    return out


def chexagent():
    from medrax.tools.xray_vqa import XRayVQATool
    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    for path, kind in ((CHEX_A, "A"), (CHEX_PRIOR, "prior")):
        todo = [p for p in pairs() if p["current"] not in done(path, "current")]
        print(f"CheXagent {kind}: {len(todo)} to go", flush=True)
        start = time.time()
        with path.open("a", encoding="utf-8") as out:
            for n, p in enumerate(todo, 1):
                rec = {"current": p["current"], "answers": {}, "p_present": {}}
                for c in CONDITIONS:
                    if kind == "A":
                        text, yn = tool._generate_response([str(NIH / p["prior"]), str(NIH / p["current"])],
                                                           A_PROMPT.format(c=c), 8)
                    else:
                        text, yn = tool._generate_response([str(NIH / p["prior"])], f"Is there {c}?", 8)
                    rec["answers"][c] = text
                    rec["p_present"][c] = yn["p_yes"] if yn else None
                out.write(json.dumps(rec) + "\n")
                out.flush()
                if n % 50 == 0 or n == len(todo):
                    rate = n / (time.time() - start)
                    print(f"CheXagent {kind} {n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:.0f} min",
                          flush=True)


def maira():
    from PIL import Image

    from maira_score_all_conditions import SENTENCES7
    from maira_score_nih import make_scorer
    todo = [p for p in pairs() if p["current"] not in done(MAIRA_L, "current")]
    print(f"MAIRA-2 longitudinal: {len(todo)} to go", flush=True)
    if not todo:
        return
    scorer, start = make_scorer({**SENTENCES, **SENTENCES7}), time.time()
    with MAIRA_L.open("a", encoding="utf-8") as out:
        for n, p in enumerate(todo, 1):
            rec = {"current": p["current"], "findings": scorer(Image.open(NIH / p["current"]).convert("RGB"),
                                                               Image.open(NIH / p["prior"]).convert("RGB"))}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if n % 50 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"MAIRA-2 {n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


def generate():
    from PIL import Image

    def url(f):
        im = Image.open(NIH / f).convert("RGB")
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}
    prompt = LONGITUDINAL + (ROOT / "medrax" / "docs" / "medgemma_report_prompt.txt").read_text(encoding="utf-8").strip()
    todo = [p for p in pairs() if p["current"] not in done(REPORTS, "current")]
    print(f"MedGemma longitudinal: {len(todo)} to go", flush=True)
    start = time.time()
    with REPORTS.open("a", encoding="utf-8") as out:
        for n, p in enumerate(todo, 1):
            body = {"model": "medgemma1.5:4b", "temperature": 0, "max_tokens": 450, "messages": [{"role": "user", "content": [
                url(p["prior"]), url(p["current"]), {"type": "text", "text": prompt}]}]}
            req = urllib.request.Request("http://localhost:11434/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
            out.write(json.dumps({"current": p["current"], "prior": p["prior"], "report": text}) + "\n")
            out.flush()
            if n % 50 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"MedGemma {n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


def read():
    from comparison_edv_pdf import NOT_IN_GT, ground_truth_locations, make_extractor
    extractor = make_extractor("qwen3.8-edv:latest")
    reports = [json.loads(l) for l in REPORTS.read_text(encoding="utf-8").splitlines()]
    todo = [r for r in reports if r["current"] not in done(LABELS, "current")]
    print(f"read: {len(todo)} to go", flush=True)
    start = time.time()
    with LABELS.open("a", encoding="utf-8") as out:
        for n, r in enumerate(todo, 1):
            ans = ground_truth_locations(extractor, r["report"], CONDITIONS, with_evidence=True, extra_rules=E.EXTRA_RULES)
            out.write(json.dumps({"current": r["current"], "labels": {
                c: {"answer": a[0], "evidence": a[1],
                    "score": 0.0 if a[0] == NOT_IN_GT else 0.5 if "(possible)" in a[0] else 1.0}
                for c, a in ans.items()}}) + "\n")
            out.flush()
            if n % 25 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"read {n}/{len(todo)}  {rate:.3f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


def analyze():
    truth, mg_s, chex_s, maira_s, _ = E.load()
    P = pairs()
    files = [p["current"] for p in P]
    jl = lambda path, key="current": {json.loads(l)[key]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()}  # noqa: E731
    a, pr, ml, lab = jl(CHEX_A), jl(CHEX_PRIOR), jl(MAIRA_L), jl(LABELS)
    if len(lab) < len(files):
        raise SystemExit(f"{len(files) - len(lab)} pairs without longitudinal MedGemma labels")
    raw = {
        "MedGemma-L": {c: [lab[f]["labels"][c]["score"] for f in files] for c in CONDITIONS},
        "MedGemma-S": {c: [mg_s[(f, c)] for f in files] for c in CONDITIONS},
        "CheXagent A": {c: [a[f]["p_present"][c] for f in files] for c in CONDITIONS},
        "CheXagent B": {c: [None if pr[f]["p_present"][c] is None else (chex_s[(f, c)] + pr[f]["p_present"][c]) / 2
                            for f in files] for c in CONDITIONS},
        "CheXagent-S": {c: [chex_s[(f, c)] for f in files] for c in CONDITIONS},
        "MAIRA-2-L": {c: [ml[f]["findings"][c]["score"] for f in files] for c in CONDITIONS},
        "MAIRA-2-S": {c: [maira_s[(f, c)] for f in files] for c in CONDITIONS},
    }
    y_all = {c: np.array([truth[(f, c)] for f in files]) for c in CONDITIONS}
    # pairs where CheXagent A gave no Yes/No answer are excluded for every system in that condition
    keep = {c: np.array([v is not None for v in raw["CheXagent A"][c]]) & np.array([v is not None for v in raw["CheXagent B"][c]])
            for c in CONDITIONS}
    y = {c: y_all[c][keep[c]] for c in CONDITIONS}
    vals = {s: {c: np.array([v for v, k in zip(raw[s][c], keep[c]) if k], float) for c in CONDITIONS} for s in raw}

    def pct(v):
        srt = np.sort(v)
        return np.searchsorted(srt, v, side="right") / len(v)
    systems = {
        "MedGemma-L alone": vals["MedGemma-L"],
        "MedGemma-L + CheXagent A": {c: 0.5 * vals["MedGemma-L"][c] + 0.5 * pct(vals["CheXagent A"][c]) for c in CONDITIONS},
        "MedGemma-L + CheXagent B": {c: 0.5 * vals["MedGemma-L"][c] + 0.5 * pct(vals["CheXagent B"][c]) for c in CONDITIONS},
        "MedGemma-L + MAIRA-2-L": {c: 0.5 * vals["MedGemma-L"][c] + 0.5 * pct(vals["MAIRA-2-L"][c]) for c in CONDITIONS},
        "MedGemma-S alone": vals["MedGemma-S"],
        "MedGemma-S + CheXagent-S": {c: 0.5 * vals["MedGemma-S"][c] + 0.5 * pct(vals["CheXagent-S"][c]) for c in CONDITIONS},
        "MedGemma-S + MAIRA-2-S": {c: 0.5 * vals["MedGemma-S"][c] + 0.5 * pct(vals["MAIRA-2-S"][c]) for c in CONDITIONS},
        "CheXagent A alone": vals["CheXagent A"], "CheXagent B alone": vals["CheXagent B"],
        "CheXagent-S alone": vals["CheXagent-S"],
        "MAIRA-2-L alone": vals["MAIRA-2-L"], "MAIRA-2-S alone": vals["MAIRA-2-S"],
    }
    main = [c for c in CONDITIONS if y[c].sum() >= 10]
    n_c = {c: len(y[c]) for c in CONDITIONS}

    def aucs(idx_by_c):
        return {s: {c: auc(v[c][idx_by_c[c]][y[c][idx_by_c[c]] == 1], v[c][idx_by_c[c]][y[c][idx_by_c[c]] == 0])
                    for c in CONDITIONS} for s, v in systems.items()}
    full = aucs({c: np.arange(n_c[c]) for c in CONDITIONS})
    # image-level bootstrap: one resample of the 499 pairs, mapped to each condition's kept rows
    rng = np.random.default_rng(SEED)
    pos_of = {c: np.cumsum(keep[c]) - 1 for c in CONDITIONS}
    boots = []
    for _ in range(N_BOOT):
        b = rng.integers(0, len(files), len(files))
        boots.append(aucs({c: pos_of[c][b][keep[c][b]] for c in CONDITIONS}))
    mean = lambda d, s: np.nanmean([d[s][c] for c in main])  # noqa: E731

    def diff(s1, s2):
        vals_ = [mean(b, s1) - mean(b, s2) for b in boots]
        lo, hi = np.percentile([v for v in vals_ if np.isfinite(v)], [2.5, 97.5])
        return mean(full, s1) - mean(full, s2), lo, hi
    verdict = lambda lo, hi: "supported" if lo > 0 else "reversed" if hi < 0 else "not supported"  # noqa: E731
    no_answer = {c: int((~keep[c]).sum()) for c in CONDITIONS}
    lines = ["# Longitudinal EDV (prior + current): results", "",
             f"Protocol: `edv_longitudinal_protocol.md` (committed before the readings). {len(files)} NIH test pairs. "
             f"Mean AUC over the {len(main)} conditions with at least 10 positives; paired 95% CIs from {N_BOOT} "
             "image-level bootstrap resamples. CheXagent A gave no Yes/No answer for "
             f"{sum(no_answer.values())} of {len(files) * len(CONDITIONS)} image-condition pairs (excluded for all systems).", "",
             "## Mean AUC", "", "| system | mean AUC [95% CI] |", "|---|---|"]
    for s in systems:
        lo, hi = np.percentile([mean(b, s) for b in boots], [2.5, 97.5])
        lines.append(f"| {s} | {mean(full, s):.3f} [{lo:.3f}-{hi:.3f}] |")
    lines += ["", "## Hypotheses (primary) and exploratory comparisons", "", "| comparison | difference [95% CI] | result |",
              "|---|---|---|"]
    for name, s1, s2 in (
            ("H1-L: MedGemma-L + CheXagent A minus MedGemma-L alone", "MedGemma-L + CheXagent A", "MedGemma-L alone"),
            ("H2-L: MedGemma-L + CheXagent A minus MedGemma-L + MAIRA-2-L", "MedGemma-L + CheXagent A", "MedGemma-L + MAIRA-2-L"),
            ("(exploratory) CheXagent A alone minus CheXagent single-image alone", "CheXagent A alone", "CheXagent-S alone"),
            ("(exploratory) CheXagent B alone minus CheXagent single-image alone", "CheXagent B alone", "CheXagent-S alone"),
            ("(exploratory) MedGemma-L alone minus MedGemma-S alone", "MedGemma-L alone", "MedGemma-S alone"),
            ("(exploratory) MedGemma-L + CheXagent A minus MedGemma-S + CheXagent-S", "MedGemma-L + CheXagent A",
             "MedGemma-S + CheXagent-S"),
            ("(exploratory) MedGemma-L + CheXagent B minus MedGemma-L alone", "MedGemma-L + CheXagent B", "MedGemma-L alone"),
            ("(exploratory) MAIRA-2-L alone minus MAIRA-2 single-image alone", "MAIRA-2-L alone", "MAIRA-2-S alone")):
        d, lo, hi = diff(s1, s2)
        lines.append(f"| {name} | {d:+.3f} [{lo:+.3f} to {hi:+.3f}] | {verdict(lo, hi)} |")
    show = ["MedGemma-L alone", "MedGemma-L + CheXagent A", "MedGemma-L + CheXagent B", "MedGemma-L + MAIRA-2-L",
            "CheXagent A alone", "CheXagent-S alone", "MedGemma-S alone"]
    lines += ["", "## Per condition (AUC)", "", "| condition | positives / n | " + " | ".join(show) + " |",
              "|---|---|" + "---|" * len(show)]
    for c in CONDITIONS:
        flag = "" if c in main else " (fewer than 10 positives)"
        lines.append(f"| {c}{flag} | {int(y[c].sum())} / {n_c[c]} | " + " | ".join(f"{full[s][c]:.3f}" for s in show) + " |")
    lines += ["", "Limitations (recorded in the protocol): 499 pairs; NIH labels NLP-mined; base MedGemma; CheXagent option "
              "A beyond its 2,048-token design length; CheXagent probably saw NIH images; percentiles from the evaluation "
              "set (label-free)."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"pairs": pairs, "chexagent": chexagent, "maira": maira, "generate": generate, "read": read,
     "analyze": analyze}[sys.argv[1]]()
