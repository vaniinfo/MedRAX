"""own_words_detection_protocol.md: CheXagent vs MAIRA-2 in their own words, NIH test split.

    python experiments/own_words_detection.py maira       # MAIRA-2 own reports (resumable)
    python experiments/own_words_detection.py chexagent   # CheXagent six-area reports (resumable)
    python experiments/own_words_detection.py read        # both read for 13 conditions (resumable)
    python experiments/own_words_detection.py analyze     # -> results/own_words_detection.md
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import edv_improvement as E  # noqa: E402
from maira_score_nih import NIH  # noqa: E402

RESULTS = HERE / "results"
REPORTS = {"MAIRA-2": RESULTS / "own_words_maira_reports.jsonl",
           "CheXagent": RESULTS / "own_words_chexagent_reports.jsonl"}
LABELS = RESULTS / "own_words_labels.jsonl"
REPORT = RESULTS / "own_words_detection.md"
AREAS = ["View", "Airway", "Breathing", "Cardiac", "Diaphragm",
         "Everything else (e.g., mediastinal contours, bones, soft tissues, tubes, valves, and pacemakers)"]
SEED, N_BOOT = 20261023, 2000


def done(path, key="file"):
    if not path.exists():
        return set()
    return {tuple(json.loads(l)[k] for k in key) if isinstance(key, tuple) else json.loads(l)[key]
            for l in path.read_text(encoding="utf-8").splitlines()}


def progress(n, total, start, label):
    if n % 50 == 0 or n == total:
        rate = n / (time.time() - start)
        print(f"{label} {n}/{total}  {rate:.2f}/s  eta {(total - n) / rate / 60:.0f} min", flush=True)


def maira():
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    from vindr_grounding_amendment import MAIRA_REVISION
    todo = [f for f in E.split_files("test") if f not in done(REPORTS["MAIRA-2"])]
    print(f"MAIRA-2: {len(todo)} to go", flush=True)
    if not todo:
        return
    kw = dict(cache_dir="C:/Users/hari/model-weights", trust_remote_code=True, revision=MAIRA_REVISION)
    model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", device_map="cuda", torch_dtype=torch.bfloat16,
                                                 **kw).eval()
    proc = AutoProcessor.from_pretrained("microsoft/maira-2", **kw)
    start = time.time()
    with REPORTS["MAIRA-2"].open("a", encoding="utf-8") as out:
        for n, f in enumerate(todo, 1):
            inputs = proc.format_and_preprocess_reporting_input(
                current_frontal=Image.open(NIH / f).convert("RGB"), current_lateral=None, prior_frontal=None,
                indication=None, technique=None, comparison=None, prior_report=None, get_grounding=False,
                return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=300, use_cache=True, do_sample=False)
            report = proc.decode(ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            out.write(json.dumps({"file": f, "report": report}) + "\n")
            out.flush()
            progress(n, len(todo), start, "MAIRA-2")


def chexagent():
    from medrax.tools.xray_vqa import XRayVQATool
    todo = [f for f in E.split_files("test") if f not in done(REPORTS["CheXagent"])]
    print(f"CheXagent: {len(todo)} to go", flush=True)
    if not todo:
        return
    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    start = time.time()
    with REPORTS["CheXagent"].open("a", encoding="utf-8") as out:
        for n, f in enumerate(todo, 1):
            parts = [tool._generate_response([str(NIH / f)], f'Please provide a detailed description of "{a}" in the '
                                                               f'chest X-ray', 200)[0].strip() for a in AREAS]
            out.write(json.dumps({"file": f, "areas": parts, "report": " ".join(parts)}) + "\n")
            out.flush()
            progress(n, len(todo), start, "CheXagent")


def read():
    from comparison_edv_pdf import NOT_IN_GT, ground_truth_locations, make_extractor
    extractor = make_extractor("qwen3.8-edv:latest")
    have = done(LABELS, ("model", "file"))
    todo = [(m, json.loads(l)) for m, path in REPORTS.items() for l in path.read_text(encoding="utf-8").splitlines()]
    todo = [(m, r) for m, r in todo if (m, r["file"]) not in have]
    print(f"read: {len(todo)} to go", flush=True)
    start = time.time()
    with LABELS.open("a", encoding="utf-8") as out:
        for n, (m, r) in enumerate(todo, 1):
            ans = ground_truth_locations(extractor, r["report"], E.CONDITIONS, with_evidence=True,
                                         extra_rules=E.EXTRA_RULES)
            out.write(json.dumps({"model": m, "file": r["file"], "labels": {
                c: {"answer": a[0], "evidence": a[1],
                    "score": 0.0 if a[0] == NOT_IN_GT else 0.5 if "(possible)" in a[0] else 1.0}
                for c, a in ans.items()}}) + "\n")
            out.flush()
            progress(n, len(todo), start, "read")


# ------------------------------------------------------------------ analysis

def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar(b, c):
    from scipy.stats import binomtest
    return binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


def analyze(possible_counts=False):
    truth, _, chex, _, _ = E.load()
    files = E.split_files("test")
    labels = {(r["model"], r["file"]): r["labels"]
              for r in map(json.loads, LABELS.read_text(encoding="utf-8").splitlines())}
    missing = [(m, f) for m in REPORTS for f in files if (m, f) not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} reports not read yet")
    cut = 0.5 if possible_counts else 1.0
    said = {
        "C-report": {c: np.array([labels[("CheXagent", f)][c]["score"] >= cut for f in files]) for c in E.CONDITIONS},
        "C-answer": {c: np.array([chex[(f, c)] >= 0.5 for f in files]) for c in E.CONDITIONS},
        "MAIRA-2": {c: np.array([labels[("MAIRA-2", f)][c]["score"] >= cut for f in files]) for c in E.CONDITIONS},
    }
    y = {c: np.array([truth[(f, c)] for f in files]).astype(bool) for c in E.CONDITIONS}
    main = [c for c in E.CONDITIONS if y[c].sum() >= 10]

    def ba(s, yy):
        return 0.5 * (s[yy].mean() + (~s[~yy]).mean())

    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(files), len(files)) for _ in range(N_BOOT)]

    def mean_ci(fn):
        point = np.mean([fn(c, np.arange(len(files))) for c in main])
        bs = [np.mean([fn(c, b) for c in main]) for b in boots]
        return point, *np.percentile(bs, [2.5, 97.5])

    title = "possible counts as stated (sensitivity analysis)" if possible_counts else "primary: present only"
    lines = [f"## {title}", "", "| system | mean sensitivity | mean specificity | mean balanced accuracy [95% CI] |",
             "|---|---|---|---|"]
    for s in said:
        sens = np.mean([said[s][c][y[c]].mean() for c in main])
        spec = np.mean([(~said[s][c][~y[c]]).mean() for c in main])
        p, lo, hi = mean_ci(lambda c, b, s=s: ba(said[s][c][b], y[c][b]))
        lines.append(f"| {s} | {sens:.3f} | {spec:.3f} | {p:.3f} [{lo:.3f}-{hi:.3f}] |")
    lines += ["", "| comparison | measure | difference [95% CI] | result |", "|---|---|---|---|"]
    for name, a in (("H1: C-report minus MAIRA-2", "C-report"), ("H2: C-answer minus MAIRA-2", "C-answer")):
        for measure, fn in (
                ("balanced accuracy", lambda s, c, b: ba(said[s][c][b], y[c][b])),
                ("sensitivity", lambda s, c, b: said[s][c][b][y[c][b]].mean()),
                ("specificity", lambda s, c, b: (~said[s][c][b][~y[c][b]]).mean())):
            p, lo, hi = mean_ci(lambda c, b, fn=fn, a=a: fn(a, c, b) - fn("MAIRA-2", c, b))
            res = "CheXagent better" if lo > 0 else "MAIRA-2 better" if hi < 0 else "no difference shown"
            lines.append(f"| {name if measure == 'balanced accuracy' else ''} | {measure} | {p:+.3f} [{lo:+.3f} to "
                         f"{hi:+.3f}] | {res} |")
    lines += ["", "Per condition: caught / labelled present (sensitivity, Wilson 95% CI) and false alarms / labelled "
              "absent; McNemar exact p on discordant pairs vs MAIRA-2 (descriptive, unadjusted).", "",
              "| condition | positives / n | system | caught | sensitivity [95% CI] | false alarms | specificity [95% CI] "
              "| McNemar p (caught / false alarms) |", "|---|---|---|---|---|---|---|---|"]
    for c in E.CONDITIONS:
        pos, n = int(y[c].sum()), len(files)
        tag = "" if c in main else " (fewer than 10 positives)"
        for s in said:
            k, fa = int(said[s][c][y[c]].sum()), int(said[s][c][~y[c]].sum())
            sl, sh = wilson(k, pos)
            pl, ph = wilson(n - pos - fa, n - pos)
            mp = ""
            if s != "MAIRA-2":
                m = said["MAIRA-2"][c]
                p1 = mcnemar(int((said[s][c] & ~m & y[c]).sum()), int((~said[s][c] & m & y[c]).sum()))
                p2 = mcnemar(int((said[s][c] & ~m & ~y[c]).sum()), int((~said[s][c] & m & ~y[c]).sum()))
                mp = f"{p1:.3g} / {p2:.3g}"
            lines.append(f"| {c}{tag if s == 'C-report' else ''} | {pos} / {n} | {s} | {k} | "
                         f"{k / pos if pos else float('nan'):.3f} [{sl:.3f}-{sh:.3f}] | {fa} | "
                         f"{(n - pos - fa) / (n - pos):.3f} [{pl:.3f}-{ph:.3f}] | {mp} |")
    return lines


def report():
    head = ["# CheXagent vs MAIRA-2 in their own words: caught vs missed (NIH test split)", "",
            "Protocol: `own_words_detection_protocol.md` (committed before any report). 1,500 NIH test images; "
            "C-report = CheXagent's six-area report, C-answer = CheXagent's 'Is there X?' with Yes outweighing No "
            "(p_present >= 0.5), MAIRA-2 = its own findings report. Reports read by the quote-checked Qwen lookup. "
            f"Means over the conditions with at least 10 positives; {N_BOOT} image-level bootstrap resamples "
            f"(seed {SEED}); two-sided.", ""]
    lines = head + analyze() + [""] + analyze(possible_counts=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"maira": maira, "chexagent": chexagent, "read": read, "analyze": report}[sys.argv[1]]()
