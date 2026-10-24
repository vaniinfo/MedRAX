"""grounding_recall_protocol.md: CheXagent vs MAIRA-2 grounding recall on NIH radiologist boxes.

    python experiments/grounding_recall.py chexagent   # boxes (resumable) -> results/grounding_chexagent.jsonl
    python experiments/grounding_recall.py maira       # boxes (resumable) -> results/grounding_maira2.jsonl
    python experiments/grounding_recall.py analyze     # -> results/grounding_recall.md
"""

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
RESULTS = HERE / "results"
NIH_DIR = Path("D:/datasets/chestxray14/images")
BOXES = Path("D:/datasets/chestxray14/BBox_List_2017.csv")
OUT = {"CheXagent": RESULTS / "grounding_chexagent.jsonl", "MAIRA-2": RESULTS / "grounding_maira2.jsonl"}
REPORT = RESULTS / "grounding_recall.md"
PHRASE = {"Atelectasis": "atelectasis", "Effusion": "pleural effusion", "Cardiomegaly": "cardiomegaly",
          "Infiltrate": "infiltration", "Pneumonia": "pneumonia", "Pneumothorax": "pneumothorax", "Mass": "mass",
          "Nodule": "nodule"}
SEED, N_BOOT = 20261024, 2000


def cases():
    split = json.loads((RESULTS / "split_nih_70_30.json").read_text(encoding="utf-8"))
    test = {f for s in split.values() for f, v in s.items() if v == "test"}
    out = []
    with BOXES.open(encoding="utf-8") as fh:
        for row in list(csv.reader(fh))[1:]:
            f, label, x, y, w, h = row[0], row[1], *map(float, row[2:6])
            if f in test or not (NIH_DIR / f).is_file():
                continue
            out.append({"file": f, "label": label, "phrase": PHRASE[label],
                        "truth": [v / 1024 * 100 for v in (x, y, x + w, y + h)]})
    return out


def done(path):
    return {(json.loads(l)["file"], json.loads(l)["label"]) for l in path.read_text(encoding="utf-8").splitlines()} \
        if path.exists() else set()


def run(model, locate):
    todo = [c for c in cases() if (c["file"], c["label"]) not in done(OUT[model])]
    print(f"{model}: {len(todo)} to go", flush=True)
    with OUT[model].open("a", encoding="utf-8") as out:
        for n, c in enumerate(todo, 1):
            out.write(json.dumps({"file": c["file"], "label": c["label"], "phrase": c["phrase"],
                                  **locate(str(NIH_DIR / c["file"]), c["phrase"])}) + "\n")
            out.flush()
            if n % 50 == 0 or n == len(todo):
                print(f"{model} {n}/{len(todo)}", flush=True)


def chexagent():
    from medrax.tools.xray_vqa import XRayVQATool
    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")

    def locate(path, phrase):
        r = tool.locate_findings(path, [phrase])[0][0]
        return {"boxes": [b["box_normalized"] for b in r["boxes"]], "other_labels": r.get("substituted_labels", [])}
    run("CheXagent", locate)


def maira():
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    from vindr_grounding_amendment import MAIRA_REVISION
    kw = dict(cache_dir="C:/Users/hari/model-weights", trust_remote_code=True, revision=MAIRA_REVISION)
    model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", device_map="cuda", torch_dtype=torch.bfloat16,
                                                 **kw).eval()
    proc = AutoProcessor.from_pretrained("microsoft/maira-2", **kw)

    def locate(path, phrase):
        image = Image.open(path).convert("RGB")
        inp = proc.format_and_preprocess_phrase_grounding_input(frontal_image=image, phrase=phrase.capitalize(),
                                                                 return_tensors="pt")
        inp = {k: v.to("cuda") for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=150, use_cache=True, do_sample=False)
        text = proc.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        parse_error = False
        try:
            raw = [b for _, bxs in proc.convert_output_to_plaintext_or_grounded_sequence(text) or [] for b in bxs or []]
        except AssertionError:
            parse_error = True
            raw = [tuple((int(v) + 0.5) / 100 for v in m)
                   for m in re.findall(r"<box><x(\d+)><y(\d+)><x(\d+)><y(\d+)></box>", text)]
        boxes = [[v * 100 for v in proc.adjust_box_for_original_image_size(b, width=image.size[0],
                                                                            height=image.size[1])] for b in raw]
        return {"boxes": boxes, "raw": text, "parse_error": parse_error}
    run("MAIRA-2", locate)


# ------------------------------------------------------------------ analysis

def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    inter = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


def union_iou(boxes, t, grid=200):
    """IoU of the union of `boxes` with box t, on a grid over the image (percent coordinates)."""
    if not boxes:
        return 0.0
    xs = (np.arange(grid) + 0.5) * 100 / grid
    X, Y = np.meshgrid(xs, xs)
    inside = lambda b: (X >= b[0]) & (X <= b[2]) & (Y >= b[1]) & (Y <= b[3])  # noqa: E731
    m = np.zeros_like(X, bool)
    for b in boxes:
        m |= inside(b)
    tm = inside(t)
    u = (m | tm).sum()
    return float((m & tm).sum() / u) if u else 0.0


def wilson(k, n):
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def measures(rec, c):
    bx, t = rec["boxes"], c["truth"]
    best = max((iou(b, t) for b in bx), default=0.0)
    un = union_iou(bx, t)
    point = any(t[0] <= (b[0] + b[2]) / 2 <= t[2] and t[1] <= (b[1] + b[3]) / 2 <= t[3] for b in bx)
    return {"best": best, "union": un, "point": point, "n_boxes": len(bx),
            "area_ratio": (sum(area(b) for b in bx) / area(t)) if bx and area(t) else float("nan")}


def analyze():
    from scipy.stats import binomtest
    cs = cases()
    recs = {m: {(json.loads(l)["file"], json.loads(l)["label"]): json.loads(l)
                for l in p.read_text(encoding="utf-8").splitlines()} for m, p in OUT.items()}
    for m in OUT:
        missing = [c for c in cs if (c["file"], c["label"]) not in recs[m]]
        if missing:
            raise SystemExit(f"{m}: {len(missing)} cases without boxes")
    M = {m: [measures(recs[m][(c["file"], c["label"])], c) for c in cs] for m in OUT}
    labels = list(PHRASE)
    images = sorted({c["file"] for c in cs})
    idx = {f: [i for i, c in enumerate(cs) if c["file"] == f] for f in images}
    rng = np.random.default_rng(SEED)
    boots = [np.concatenate([idx[images[j]] for j in rng.integers(0, len(images), len(images))]) for _ in range(N_BOOT)]
    hit = lambda m, key, t: np.array([x[key] >= t for x in M[m]])  # noqa: E731
    point = {m: np.array([x["point"] for x in M[m]]) for m in OUT}

    lines = ["# Grounding recall: CheXagent vs MAIRA-2 on NIH radiologist boxes", "",
             f"Protocol: `grounding_recall_protocol.md` (committed before any box). {len(cs)} cases (one radiologist "
             f"box each) on {len(images)} images; the phrase names the condition without a side. Hit = the best of "
             "the model's boxes overlaps the radiologist's box with IoU >= t. 95% CIs: Wilson (recall), "
             f"{N_BOOT} image-level bootstrap resamples (differences, seed {SEED}).", "",
             "## Pooled", "", "| measure | CheXagent | MAIRA-2 |", "|---|---|---|"]
    rows = [("recall@0.5 (primary)", lambda m: hit(m, "best", 0.5)), ("recall@0.3", lambda m: hit(m, "best", 0.3)),
            ("pointing game", lambda m: point[m]), ("union recall@0.5", lambda m: hit(m, "union", 0.5)),
            ("union recall@0.3", lambda m: hit(m, "union", 0.3))]
    for name, fn in rows:
        cells = []
        for m in OUT:
            h = fn(m)
            lo, hi = wilson(int(h.sum()), len(h))
            cells.append(f"{h.mean():.1%} ({h.sum()}/{len(h)}) [{lo:.1%}-{hi:.1%}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for name, key in (("mean best IoU", "best"), ("mean union IoU", "union"), ("mean boxes per case", "n_boxes"),
                      ("median box area / radiologist box area", "area_ratio")):
        f = np.nanmedian if "median" in name else np.mean
        lines.append(f"| {name} | " + " | ".join(f"{f([x[key] for x in M[m]]):.2f}" for m in OUT) + " |")
    lines.append("| no box returned | " + " | ".join(f"{sum(x['n_boxes'] == 0 for x in M[m])}" for m in OUT) + " |")

    lines += ["", "## Hypotheses (two-sided)", "", "| comparison | CheXagent minus MAIRA-2 [95% CI] | McNemar b / c, p | result |",
              "|---|---|---|---|"]
    for name, t in (("G1 (primary): recall@0.5", 0.5), ("G2: recall@0.3", 0.3)):
        a, b = hit("CheXagent", "best", t), hit("MAIRA-2", "best", t)
        d = a.mean() - b.mean()
        bs = [a[i].mean() - b[i].mean() for i in boots]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        res = "CheXagent better" if lo > 0 else "MAIRA-2 better" if hi < 0 else "no difference shown"
        lines.append(f"| {name} | {d:+.1%} [{lo:+.1%} to {hi:+.1%}] | {n10} / {n01}, {p:.2g} | {res} |")

    lines += ["", "## Per label (descriptive; McNemar exact p unadjusted)", "",
              "| label | cases | recall@0.5 CheXagent / MAIRA-2 (p) | recall@0.3 CheXagent / MAIRA-2 (p) | pointing game "
              "CheXagent / MAIRA-2 | mean best IoU CheXagent / MAIRA-2 |", "|---|---|---|---|---|---|"]
    for lab in labels:
        ii = np.array([c["label"] == lab for c in cs])
        cells = []
        for t in (0.5, 0.3):
            a, b = hit("CheXagent", "best", t)[ii], hit("MAIRA-2", "best", t)[ii]
            n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
            p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            cells.append(f"{a.mean():.0%} / {b.mean():.0%} ({p:.2g})")
        cells.append(f"{point['CheXagent'][ii].mean():.0%} / {point['MAIRA-2'][ii].mean():.0%}")
        cells.append(" / ".join(f"{np.mean([x['best'] for x, k in zip(M[m], ii) if k]):.2f}" for m in OUT))
        lines.append(f"| {lab} | {ii.sum()} | " + " | ".join(cells) + " |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"chexagent": chexagent, "maira": maira, "analyze": analyze}[sys.argv[1]]()
