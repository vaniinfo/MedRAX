"""EXPLORATORY: own_words_detection_protocol.md's comparison on the 20 NIH examples of the longitudinal
example PDF (logs/edv_longitudinal_examples_20261020-065035.pdf). Same models, prompts and reader as
the protocol; current image only.

Caution: these 20 were selected *because* CheXagent found a condition MedGemma's report missed and
NIH labels it, so the selection favours CheXagent. Examples, not a test.

    python experiments/own_words_examples20.py run       # reports + reading (resumable)
    python experiments/own_words_examples20.py analyze   # -> results/own_words_examples20.md
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import edv_improvement as E  # noqa: E402
import own_words_detection as O  # noqa: E402

RESULTS = HERE / "results"
EXAMPLES_JSON = HERE.parent / "logs" / "edv_longitudinal_examples_20261020-065035.json"
REPORTS = {"MAIRA-2": RESULTS / "own_words_ex20_maira_reports.jsonl",
           "CheXagent": RESULTS / "own_words_ex20_chexagent_reports.jsonl"}
LABELS = RESULTS / "own_words_ex20_labels.jsonl"
REPORT = RESULTS / "own_words_examples20.md"


def files():
    return [r["current"] for r in json.loads(EXAMPLES_JSON.read_text(encoding="utf-8"))]


def run():
    fs = files()
    # MAIRA-2 reports already written for the NIH run (identical settings) are reused
    if not REPORTS["MAIRA-2"].exists() and O.REPORTS["MAIRA-2"].exists():
        reuse = [l for l in O.REPORTS["MAIRA-2"].read_text(encoding="utf-8").splitlines()
                 if json.loads(l)["file"] in fs]
        REPORTS["MAIRA-2"].write_text("".join(l + "\n" for l in reuse), encoding="utf-8")
        print(f"reused {len(reuse)} MAIRA-2 reports", flush=True)
    O.maira(fs, REPORTS["MAIRA-2"])
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()
    O.chexagent(fs, REPORTS["CheXagent"])
    O.read(REPORTS, LABELS)


def analyze():
    truth, _, chex, _, _ = E.load()
    fs = files()
    lab = {(r["model"], r["file"]): r["labels"] for r in map(json.loads, LABELS.read_text(encoding="utf-8").splitlines())}
    rep = {(m, json.loads(l)["file"]): json.loads(l)["report"] for m, p in REPORTS.items()
           for l in p.read_text(encoding="utf-8").splitlines()}
    systems = {
        "CheXagent own report": lambda f, c: lab[("CheXagent", f)][c]["score"] >= 1.0,
        "CheXagent Yes/No": lambda f, c: chex[(f, c)] >= 0.5,
        "MAIRA-2 own report": lambda f, c: lab[("MAIRA-2", f)][c]["score"] >= 1.0,
    }
    pos = sum(truth[(f, c)] for f in fs for c in E.CONDITIONS)
    neg = len(fs) * len(E.CONDITIONS) - pos
    lines = ["# EXPLORATORY: CheXagent vs MAIRA-2 in their own words, 20 NIH examples", "",
             "Images: the 20 examples of `logs/edv_longitudinal_examples_20261020-065035.pdf`. **Selected because "
             "CheXagent (two-image score) found a condition MedGemma missed and NIH labels it, so the selection "
             "favours CheXagent; examples, not a test.** Same models, prompts and quote-checked reader as "
             "`own_words_detection_protocol.md`; current image only; 'stated' = reader answers present (not "
             "'possible'). CheXagent Yes/No = 'Is there X?' with Yes outweighing No (p_present >= 0.5). Ground "
             "truth: NIH labels (text-mined).", "",
             f"## Totals over 20 images x 13 conditions ({pos} labelled present, {neg} labelled absent)", "",
             "| system | caught (of labelled present) | false alarms (of labelled absent) |", "|---|---|---|"]
    for s, said in systems.items():
        caught = sum(said(f, c) for f in fs for c in E.CONDITIONS if truth[(f, c)])
        fa = sum(said(f, c) for f in fs for c in E.CONDITIONS if not truth[(f, c)])
        lines.append(f"| {s} | {caught} / {pos} | {fa} / {neg} |")
    lines += ["", "## Per condition (caught / labelled present; false alarms)", "",
              "| condition | labelled present | " + " | ".join(systems) + " |", "|---|---|" + "---|" * len(systems)]
    for c in E.CONDITIONS:
        p = sum(truth[(f, c)] for f in fs)
        cells = []
        for said in systems.values():
            k = sum(said(f, c) for f in fs if truth[(f, c)])
            fa = sum(said(f, c) for f in fs if not truth[(f, c)])
            cells.append(f"{k} / {p}; {fa} false" if p else f"-; {fa} false")
        lines.append(f"| {c} | {p} | " + " | ".join(cells) + " |")
    lines += ["", "## Per image", "",
              "Conditions each system states; **bold** = NIH labels it present. NIH labels not caught by any system "
              "are listed under 'missed by all'.", "",
              "| example | image | NIH labels | " + " | ".join(systems) + " | missed by all |",
              "|---|---|---|" + "---|" * len(systems) + "---|"]
    for k, f in enumerate(fs, 1):
        gt = [c for c in E.CONDITIONS if truth[(f, c)]]
        cells = []
        for said in systems.values():
            st = [c for c in E.CONDITIONS if said(f, c)]
            cells.append(", ".join(f"**{c}**" if c in gt else c for c in st) or "none")
        missed = [c for c in gt if not any(said(f, c) for said in systems.values())]
        lines.append(f"| {k} | {f} | {', '.join(gt) or 'none'} | " + " | ".join(cells) + f" | {', '.join(missed) or '-'} |")
    lines += ["", "## The reports", ""]
    for k, f in enumerate(fs, 1):
        lines += [f"**Example {k} ({f})**", "", f"- CheXagent: {rep[('CheXagent', f)]}",
                  f"- MAIRA-2: {rep[('MAIRA-2', f)]}", ""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
