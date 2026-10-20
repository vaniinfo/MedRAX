"""MAIRA-2 presence score without training (design: maira_score_design.md).

    python experiments/maira_score_nih.py score     # MAIRA-2 readings, NIH random sample (resumable)
    python experiments/maira_score_nih.py analyze   # development split: AUC vs CheXagent p_present
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
RESULTS = HERE / "results"
READINGS = RESULTS / "maira_score_nih.jsonl"
REPORT = RESULTS / "maira_score_nih.md"
NIH = Path("D:/datasets/chestxray14/images")
SEED, N_BOOT = 20261013, 2000
SENTENCES = {
    "cardiomegaly": ("There is cardiomegaly.", "There is no cardiomegaly."),
    "pleural effusion": ("There is a pleural effusion.", "There is no pleural effusion."),
    "pneumothorax": ("There is a pneumothorax.", "There is no pneumothorax."),
    "consolidation": ("There is consolidation.", "There is no consolidation."),
    "pulmonary edema": ("There is pulmonary edema.", "There is no pulmonary edema."),
    "atelectasis": ("There is atelectasis.", "There is no atelectasis."),
}


def sample():
    images = json.loads((RESULTS / "nih_sample_5000.json").read_text(encoding="utf-8"))["images"]
    split = json.loads((RESULTS / "split_nih_70_30.json").read_text(encoding="utf-8"))["random"]
    return [(im["file"], split[im["file"]]) for im in images]


def make_scorer(sentences=None):
    """Loads MAIRA-2 (bf16) and returns score(image) -> {finding: {logp_pos, logp_neg, score}} for
    `sentences` (default: the six findings' fixed pairs)."""
    sentences = sentences or SENTENCES
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from vindr_grounding_amendment import MAIRA_REVISION

    kw = dict(cache_dir="C:/Users/hari/model-weights", trust_remote_code=True, revision=MAIRA_REVISION)
    model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", device_map="cuda", torch_dtype=torch.bfloat16,
                                                 **kw).eval()
    proc = AutoProcessor.from_pretrained("microsoft/maira-2", **kw)
    tok = proc.tokenizer
    encoded = {s: tok(s, add_special_tokens=False)["input_ids"] for pair in sentences.values() for s in pair}

    def logprobs(image, prior=None):
        inp = {k: v.cuda() for k, v in proc.format_and_preprocess_reporting_input(
            current_frontal=image, current_lateral=None, prior_frontal=prior, indication=None, technique=None,
            comparison=None, prior_report=None, get_grounding=False, return_tensors="pt").items()}
        with torch.no_grad():
            out = model(**inp, use_cache=True)
            cache = out.past_key_values  # legacy tuple: reused per sentence, never modified
            n_prefix = cache[0][0].shape[-2]
            first = torch.log_softmax(out.logits[0, -1].float(), -1)
            result = {}
            for s, c in encoded.items():
                o = model(input_ids=torch.tensor([c]).cuda(), past_key_values=cache, use_cache=True,
                          attention_mask=torch.ones(1, n_prefix + len(c), dtype=torch.long).cuda())
                lp = torch.log_softmax(o.logits[0].float(), -1)
                result[s] = first[c[0]].item() + sum(lp[t, c[t + 1]].item() for t in range(len(c) - 1))
        return result

    def score_image(image, prior=None):
        lp = logprobs(image, prior)
        return {finding: {"logp_pos": lp[pos], "logp_neg": lp[neg], "score": 1 / (1 + math.exp(-(lp[pos] - lp[neg])))}
                for finding, (pos, neg) in sentences.items()}
    return score_image


def score():
    import time

    from PIL import Image

    rows = sample()
    done = set()
    if READINGS.exists():
        done = {json.loads(l)["file"] for l in READINGS.read_text(encoding="utf-8").splitlines()}
    todo = [(f, s) for f, s in rows if f not in done]
    print(f"{len(done)} images done, {len(todo)} to go", flush=True)
    if not todo:
        return
    score_image = make_scorer()
    start = time.time()
    with READINGS.open("a", encoding="utf-8") as out:
        for n, (f, split) in enumerate(todo, 1):
            rec = {"file": f, "split": split, "findings": score_image(Image.open(NIH / f).convert("RGB"))}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if n % 200 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"{n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def analyze():
    maira = {json.loads(l)["file"]: json.loads(l) for l in READINGS.read_text(encoding="utf-8").splitlines()}
    dev = [f for f, s in sample() if s == "dev"]
    missing = [f for f in dev if f not in maira]
    if missing:
        raise SystemExit(f"{len(missing)} development images not scored yet")
    chex, truth = {}, {}
    for line in (RESULTS / "ppresent_nih.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["set"] == "random":
            chex[(r["file"], r["finding"])] = r["p_present"]
            truth[(r["file"], r["finding"])] = r["truth"]
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(dev), len(dev)) for _ in range(N_BOOT)]
    lines = ["# MAIRA-2 presence score (no training) vs CheXagent p_present: NIH development split", "",
             "Design: `maira_score_design.md` (committed before the readings). Score = P(positive sentence) / "
             "(P(positive) + P(negative)) as the start of MAIRA-2's findings. AUC 95% CI from "
             f"{N_BOOT} image-level bootstrap resamples (seed {SEED}). Decision rule: calibration only for "
             "findings with MAIRA-2 AUC >= 0.75.", "",
             "| finding | positives / n | MAIRA-2 AUC [95% CI] | CheXagent AUC [95% CI] | MAIRA-2 minus CheXagent [95% CI] "
             "| goes to calibration |", "|---|---|---|---|---|---|"]
    for finding in SENTENCES:
        y = np.array([truth[(f, finding)] for f in dev])
        m = np.array([maira[f]["findings"][finding]["score"] for f in dev])
        c = np.array([chex[(f, finding)] for f in dev])
        a_m, a_c = auc(m[y == 1], m[y == 0]), auc(c[y == 1], c[y == 0])
        bm, bc = [], []
        for b in boots:
            yb = y[b]
            bm.append(auc(m[b][yb == 1], m[b][yb == 0]))
            bc.append(auc(c[b][yb == 1], c[b][yb == 0]))
        bm, bc = np.array(bm), np.array(bc)
        ok = np.isfinite(bm) & np.isfinite(bc)
        ci = lambda v: np.percentile(v[ok], [2.5, 97.5])  # noqa: E731
        (ml, mh), (cl, ch), (dl, dh) = ci(bm), ci(bc), ci(bm - bc)
        lines.append(f"| {finding} | {int(y.sum())} / {len(y)} | {a_m:.3f} [{ml:.3f}-{mh:.3f}] | "
                     f"{a_c:.3f} [{cl:.3f}-{ch:.3f}] | {a_m - a_c:+.3f} [{dl:+.3f} to {dh:+.3f}] | "
                     f"{'yes' if a_m >= 0.75 else 'no'} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"score": score, "analyze": analyze}[sys.argv[1]]()
