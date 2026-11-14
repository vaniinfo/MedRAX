"""EXPLORATORY (development split only): better ways to score MAIRA-2 than one fixed sentence pair.

Two disjoint random sets of NIH development images (seed 20261024): 300 to learn MAIRA-2's own
phrasings, 300 to compare the methods. The test split is not touched.

Methods, per image and condition:
  old  P("There is X.") vs P("There is no X.") as the first sentence (the current score)
  A    the same with MAIRA-2's own phrasings: up to 5 most frequent sentences stating X and 5 negating
       X in its greedy reports on the phrasing set (plus the old pair); log-sum-exp per side
  B    10 reports sampled at temperature 1 (MAIRA-2's own distribution); score = share stating X
The rule reader below decides "states X" for the greedy and sampled reports (same rule for all).
Compared on the evaluation set: faithfulness (AUC for MAIRA-2's own greedy statement; no labels) and
detection (AUC for the NIH label), next to CheXagent p_present on the same images.

    python experiments/maira_score_methods_dev.py mine       # greedy reports, phrasing set -> phrasings
    python experiments/maira_score_methods_dev.py evaluate   # greedy + 10 samples + old/A scores (resumable)
    python experiments/maira_score_methods_dev.py analyze    # -> results/maira_score_methods_dev.md
"""

import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import provisional_findings as P  # noqa: E402
from edv_improvement import CONDITIONS  # noqa: E402
from maira_score_all_conditions import SENTENCES7  # noqa: E402
from maira_score_nih import NIH, SENTENCES, auc  # noqa: E402

RESULTS = HERE / "results"
MINE = RESULTS / "maira_methods_mine_reports.jsonl"
PHRASES = RESULTS / "maira_methods_phrasings.json"
EVAL = RESULTS / "maira_methods_eval.jsonl"
REPORT = RESULTS / "maira_score_methods_dev.md"
SEED, N_SET, N_SAMPLES = 20261024, 300, 10
OLD = {**SENTENCES, **SENTENCES7}
TERMS = {
    "cardiomegaly": r"cardiomegaly|(heart|cardiac)( size| silhouette| shadow)?( is| are)?( mildly| moderately| markedly| severely)? enlarg|enlarged (heart|cardiac)",
    "pleural effusion": r"effusion",
    "pneumothorax": r"pneumothora",
    "consolidation": r"consolidat",
    "pulmonary edema": r"edema|oedema",
    "atelectasis": r"atelecta",
    "pneumonia": r"pneumonia",
    "infiltration": r"infiltrat",
    "emphysema": r"(?<!subcutaneous )emphysema|hyperinflat|hyperexpan|copd",
    "nodule": r"nodul",
    "mass": r"\bmass",
    "fibrosis": r"fibros|fibrotic",
    "pleural thickening": r"pleural thickening|pleural[\w\s-]{0,20}thicken|thickening of the pleura",
}
NEG = re.compile(r"\b(no|not|without|negative for|free of|resolved|resolution of|clear of)\b")


def sentences(report):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", report) if s.strip()]


def stance(sentence, c):
    """'pos' if the sentence states condition c, 'neg' if it negates it, None if it does not mention it."""
    low = sentence.lower()
    m = re.search(TERMS[c], low)
    if not m:
        return None
    return "neg" if NEG.search(low[:m.end()]) else "pos"


def states(report, c):
    return any(stance(s, c) == "pos" for s in sentences(report))


def sets():
    rows, _ = P.sample()
    dev = sorted(f for f, s in rows if s == "dev")
    rng = random.Random(SEED)
    picked = rng.sample(dev, 2 * N_SET)
    return picked[:N_SET], picked[N_SET:]


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from vindr_grounding_amendment import MAIRA_REVISION
    kw = dict(cache_dir="C:/Users/hari/model-weights", trust_remote_code=True, revision=MAIRA_REVISION)
    model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", device_map="cuda", torch_dtype=torch.bfloat16,
                                                 **kw).eval()
    return model, AutoProcessor.from_pretrained("microsoft/maira-2", **kw)


def inputs_for(proc, f):
    from PIL import Image
    inp = proc.format_and_preprocess_reporting_input(
        current_frontal=Image.open(NIH / f).convert("RGB"), current_lateral=None, prior_frontal=None,
        indication=None, technique=None, comparison=None, prior_report=None, get_grounding=False, return_tensors="pt")
    return {k: v.to("cuda") for k, v in inp.items()}


def generate(model, proc, inp, sample=False):
    import torch
    kw = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
    with torch.no_grad():
        ids = model.generate(**inp, max_new_tokens=300, use_cache=True, **kw)
    return proc.decode(ids[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def done(path):
    return {json.loads(l)["file"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()


def mine():
    phrasing_set, _ = sets()
    todo = [f for f in phrasing_set if f not in done(MINE)]
    if todo:
        model, proc = load_model()
        with MINE.open("a", encoding="utf-8") as out:
            for n, f in enumerate(todo, 1):
                out.write(json.dumps({"file": f, "report": generate(model, proc, inputs_for(proc, f))}) + "\n")
                out.flush()
                if n % 50 == 0:
                    print(f"mine {n}/{len(todo)}", flush=True)
    reports = [json.loads(l)["report"] for l in MINE.read_text(encoding="utf-8").splitlines()]
    phr = {}
    for c in CONDITIONS:
        count = {"pos": Counter(), "neg": Counter()}
        for r in reports:
            for s in sentences(r):
                st = stance(s, c)
                if st:
                    count[st][re.sub(r"\s+", " ", s)] += 1
        phr[c] = {side: list(dict.fromkeys([OLD[c][0 if side == "pos" else 1]] +
                                           [s for s, _ in count[side].most_common(5)]))
                  for side in ("pos", "neg")}
        phr[c]["counts"] = {side: dict(count[side].most_common(5)) for side in ("pos", "neg")}
    PHRASES.write_text(json.dumps(phr, indent=1), encoding="utf-8")
    for c in CONDITIONS:
        print(c, phr[c]["counts"])


def evaluate():
    import torch
    _, eval_set = sets()
    phr = json.loads(PHRASES.read_text(encoding="utf-8"))
    all_sents = sorted({s for c in CONDITIONS for side in ("pos", "neg") for s in phr[c][side]})
    todo = [f for f in eval_set if f not in done(EVAL)]
    print(f"evaluate: {len(todo)} to go", flush=True)
    if not todo:
        return
    model, proc = load_model()
    enc = {s: proc.tokenizer(s, add_special_tokens=False)["input_ids"] for s in all_sents}
    start = time.time()
    with EVAL.open("a", encoding="utf-8") as out:
        for n, f in enumerate(todo, 1):
            inp = inputs_for(proc, f)
            with torch.no_grad():
                o = model(**inp, use_cache=True)
                cache, first = o.past_key_values, torch.log_softmax(o.logits[0, -1].float(), -1)
                n_prefix = cache[0][0].shape[-2]
                lp = {}
                for s, c in enc.items():
                    so = model(input_ids=torch.tensor([c]).cuda(), past_key_values=cache, use_cache=True,
                               attention_mask=torch.ones(1, n_prefix + len(c), dtype=torch.long).cuda())
                    lps = torch.log_softmax(so.logits[0].float(), -1)
                    lp[s] = first[c[0]].item() + sum(lps[t, c[t + 1]].item() for t in range(len(c) - 1))
                del o, cache
            greedy = generate(model, proc, inp)
            samples = [generate(model, proc, inp, sample=True) for _ in range(N_SAMPLES)]
            out.write(json.dumps({"file": f, "logp": lp, "greedy": greedy, "samples": samples}) + "\n")
            out.flush()
            if n % 10 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"evaluate {n}/{len(todo)}  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


DEVICES = re.compile(r"\b(tube|catheter|line|pacemaker|pacer|wire|port|stent|valve|picc|drain|lead|clip|sternotomy)s?\b")


def strict(phr, c, side):
    """A-strict (fixed 2026-10-24 after seeing the phrasings, before any evaluation result): the old
    sentence plus mined phrasings written at least twice that mention no other condition and no device."""
    keep = [s for s, k in phr[c]["counts"][side].items()
            if k >= 2 and not DEVICES.search(s.lower()) and all(stance(s, o) is None for o in CONDITIONS if o != c)]
    return [OLD[c][0 if side == "pos" else 1]] + keep


def lse(v):
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


def analyze():
    phr = json.loads(PHRASES.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines()]
    _, nih = P.sample()
    chex = {}
    for r in map(json.loads, (RESULTS / "ppresent_nih.jsonl").read_text(encoding="utf-8").splitlines()):
        if r["set"] == "random":
            chex[(r["file"], r["finding"])] = r["p_present"]
    for r in map(json.loads, (RESULTS / "provisional_readings.jsonl").read_text(encoding="utf-8").splitlines()):
        if r["p_present"] is not None:
            chex[(r["file"], r["finding"])] = r["p_present"]
    nih_name = {**{c: n for c, n in zip(SENTENCES, ["Cardiomegaly", "Effusion", "Pneumothorax", "Consolidation",
                                                    "Edema", "Atelectasis"])}, **P.FINDINGS}
    sig = lambda d: 1 / (1 + math.exp(-d))  # noqa: E731
    lines = ["# EXPLORATORY: scoring MAIRA-2 better (NIH development split only)", "",
             f"Evaluation set: {len(rows)} random development images (seed {SEED}); MAIRA-2's phrasings learned "
             f"from {N_SET} other development images. old = fixed sentence pair; A = MAIRA-2's own phrasings; "
             "A-strict = A keeping only phrasings written at least twice that mention no other condition and no "
             "device (fixed before any evaluation result); "
             f"B = share of {N_SAMPLES} reports sampled at temperature 1 that state the condition; greedy = its "
             "single normal report (stated / not; rule reader). Faithfulness = AUC for MAIRA-2's own greedy "
             "statement (no labels). Detection = AUC for the NIH label. Not pre-registered.", "",
             "| condition | NIH positives | greedy states | faithfulness: old / A / A-strict / B | detection: old / A / A-strict / B / "
             "greedy (yes-no) | CheXagent detection |", "|---|---|---|---|---|---|"]
    keys = ("fo", "fa", "fs", "fb", "do", "da", "ds", "db", "dg", "dc")
    summary = {k: [] for k in keys}
    for c in CONDITIONS:
        y = np.array([int(nih_name[c] in nih[r["file"]]) for r in rows])
        g = np.array([states(r["greedy"], c) for r in rows])
        old = np.array([sig(r["logp"][OLD[c][0]] - r["logp"][OLD[c][1]]) for r in rows])
        a = np.array([sig(lse([r["logp"][s] for s in phr[c]["pos"]]) - lse([r["logp"][s] for s in phr[c]["neg"]]))
                      for r in rows])
        a_s = np.array([sig(lse([r["logp"][s] for s in strict(phr, c, "pos")]) -
                            lse([r["logp"][s] for s in strict(phr, c, "neg")])) for r in rows])
        b = np.array([np.mean([states(s, c) for s in r["samples"]]) for r in rows])
        ch = np.array([chex.get((r["file"], c), np.nan) for r in rows])
        A = lambda v, t: auc(v[t == 1], v[t == 0]) if 0 < t.sum() < len(t) else float("nan")  # noqa: E731
        f = [A(v, g.astype(int)) for v in (old, a, a_s, b)]
        d = [A(v, y) for v in (old, a, a_s, b, g.astype(float))]
        ok = ~np.isnan(ch)
        dc = auc(ch[ok][y[ok] == 1], ch[ok][y[ok] == 0]) if 0 < y[ok].sum() < ok.sum() else float("nan")
        enough = y.sum() >= 10
        lines.append(f"| {c}{'' if enough else ' (fewer than 10 positives)'} | {y.sum()} | {g.sum()} | "
                     + " / ".join(f"{x:.3f}" for x in f) + " | " + " / ".join(f"{x:.3f}" for x in d)
                     + f" | {dc:.3f} |")
        if enough:
            for k, v in zip(keys, f + d + [dc]):
                summary[k].append(v)
    m = {k: np.nanmean(v) for k, v in summary.items()}
    lines += ["", f"**Mean over the {len(summary['do'])} conditions with at least 10 positives:** "
              f"faithfulness old {m['fo']:.3f}, A {m['fa']:.3f}, A-strict {m['fs']:.3f}, B {m['fb']:.3f}; detection "
              f"old {m['do']:.3f}, A {m['da']:.3f}, A-strict {m['ds']:.3f}, B {m['db']:.3f}, greedy yes/no {m['dg']:.3f}; CheXagent {m['dc']:.3f}.", "",
              "## MAIRA-2's own phrasings used by A (count in the phrasing set)", ""]
    for c in CONDITIONS:
        lines.append(f"- **{c}**: states {phr[c]['counts']['pos']}; negates {phr[c]['counts']['neg']}; "
                     f"A-strict keeps {strict(phr, c, 'pos')[1:]} / {strict(phr, c, 'neg')[1:]}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"mine": mine, "evaluate": evaluate, "analyze": analyze}[sys.argv[1]]()
