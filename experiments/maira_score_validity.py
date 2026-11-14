"""EXPLORATORY: does the MAIRA-2 sentence-likelihood score reflect what MAIRA-2 itself writes?

On NIH development images (100 labelled pleural effusion, 100 not; seed 20261023) plus the five
worked-example images, MAIRA-2 writes its own findings (same model, bf16, prompt and image as the
score; greedy, 300 tokens). A report "states effusion" if a sentence mentions effusion with no
negation before it. Then:
  1. Faithfulness (no labels): AUC of the score for MAIRA-2's own statement.
  2. Detection (NIH labels): how often MAIRA-2's own report states effusion when NIH labels it,
     and when NIH does not, next to CheXagent at its Evidence Engine cut-off.

    python experiments/maira_score_validity.py generate   # resumable -> results/maira_score_validity.jsonl
    python experiments/maira_score_validity.py analyze    # -> results/maira_score_validity.md
"""

import json
import random
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from maira_score_nih import NIH, READINGS, auc  # noqa: E402

RESULTS = HERE / "results"
OUT = RESULTS / "maira_score_validity.jsonl"
REPORT = RESULTS / "maira_score_validity.md"
EXAMPLES = ["00000096_007.png", "00000231_004.png", "00000090_008.png", "00000006_000.png", "00000029_000.png"]
NEG = re.compile(r"\b(no|not|without|negative for|free of|resolved|resolution of)\b")


def cases():
    maira = {json.loads(l)["file"]: json.loads(l) for l in READINGS.read_text(encoding="utf-8").splitlines()}
    chex = {}
    for line in (RESULTS / "ppresent_nih.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["set"] == "random" and r["finding"] == "pleural effusion":
            chex[r["file"]] = (r["p_present"], r["truth"])
    dev = sorted(f for f in maira if maira[f]["split"] == "dev" and f in chex)
    rng = random.Random(20261023)
    pos = rng.sample([f for f in dev if chex[f][1] == 1 and f not in EXAMPLES], 100)
    neg = rng.sample([f for f in dev if chex[f][1] == 0 and f not in EXAMPLES], 100)
    return [{"file": f, "truth": chex[f][1], "chexagent": chex[f][0],
             "maira_score": maira[f]["findings"]["pleural effusion"]["score"], "example": f in EXAMPLES}
            for f in EXAMPLES + pos + neg]


def states_effusion(report):
    for sentence in re.split(r"(?<=[.;])\s+", report.lower()):
        i = sentence.find("effusion")
        if i >= 0 and not NEG.search(sentence[:i]):
            return True
    return False


def generate():
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    from vindr_grounding_amendment import MAIRA_REVISION
    done = {json.loads(l)["file"] for l in OUT.read_text(encoding="utf-8").splitlines()} if OUT.exists() else set()
    todo = [c for c in cases() if c["file"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    kw = dict(cache_dir="C:/Users/hari/model-weights", trust_remote_code=True, revision=MAIRA_REVISION)
    model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", device_map="cuda", torch_dtype=torch.bfloat16,
                                                 **kw).eval()
    proc = AutoProcessor.from_pretrained("microsoft/maira-2", **kw)
    with OUT.open("a", encoding="utf-8") as out:
        for n, c in enumerate(todo, 1):
            inputs = proc.format_and_preprocess_reporting_input(
                current_frontal=Image.open(NIH / c["file"]).convert("RGB"), current_lateral=None, prior_frontal=None,
                indication=None, technique=None, comparison=None, prior_report=None, get_grounding=False,
                return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=300, use_cache=True, do_sample=False)
            report = proc.decode(ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            out.write(json.dumps({**c, "report": report}) + "\n")
            out.flush()
            if n % 20 == 0 or c["example"]:
                print(f"{n}/{len(todo)} {c['file']}: {report[:160]}", flush=True)


def analyze():
    from medrax.evidence import EvidenceEngine

    from chexagent_reliability import decide
    engine = EvidenceEngine()
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["stated"] = states_effusion(r["report"])
        r["chex_found"] = decide(engine, "pleural effusion", r["chexagent"]) == "FOUND"
    sample = [r for r in rows if not r["example"]]
    s = np.array([r["maira_score"] for r in sample])
    st = np.array([r["stated"] for r in sample])
    y = np.array([r["truth"] for r in sample])
    lines = ["# EXPLORATORY: does the MAIRA-2 score reflect what MAIRA-2 itself writes? (pleural effusion)", "",
             "NIH development images, 100 labelled effusion and 100 not (seed 20261023), plus the five worked "
             "examples. MAIRA-2 wrote its own findings (bf16, same prompt and image as the score, greedy, 300 "
             "tokens). 'States effusion' = a sentence mentions effusion with no negation before it. Not "
             "pre-registered; run after the score results were known.", "",
             "## 1. Faithfulness (no labels used)", "",
             f"- MAIRA-2's own report states effusion in {st.sum()} of {len(st)} images.",
             f"- AUC of the score for its own statement: {auc(s[st], s[~st]):.3f}" if 0 < st.sum() < len(st) else
             "- AUC not computable (all or none stated).",
             f"- Median score when it states effusion: {np.median(s[st]) if st.any() else float('nan'):.3f}; "
             f"when it does not: {np.median(s[~st]):.3f}", "",
             "## 2. Detection against the NIH labels", "",
             "| | NIH effusion (n=100) | NIH no effusion (n=100) |", "|---|---|---|",
             f"| MAIRA-2's own report states effusion | {st[y == 1].sum()} | {st[y == 0].sum()} |",
             f"| CheXagent FOUND (Evidence Engine cut-off) | {sum(r['chex_found'] for r in sample if r['truth'])} | "
             f"{sum(r['chex_found'] for r in sample if not r['truth'])} |",
             f"| MAIRA-2 score AUC (labels) | {auc(s[y == 1], s[y == 0]):.3f} | |",
             f"| CheXagent score AUC (labels) | {auc(np.array([r['chexagent'] for r in sample if r['truth']]), np.array([r['chexagent'] for r in sample if not r['truth']])):.3f} | |",
             "", "## 3. The five worked examples", "",
             "| image | NIH | CheXagent | MAIRA-2 score | MAIRA-2 report states effusion | MAIRA-2's report |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        if r["example"]:
            lines.append(f"| {r['file']} | {'effusion' if r['truth'] else 'no effusion'} | {r['chexagent']:.3f} | "
                         f"{r['maira_score']:.3f} | {'yes' if r['stated'] else 'no'} | {r['report'].replace('|', '/')} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"generate": generate, "analyze": analyze}[sys.argv[1]]()
