"""VinDr-CXR protocol, Amendment 1 (Part B): claim-blinded vs claim-conditioned grounding.

Pre-registered in experiments/vindr_amendment_1_grounding.md, before the VinDr data was
available. Part A (experiments/vindr_external_validation.py) is unchanged and must be prepared
first: this script reuses its PNG conversion and manifest.

    python experiments/vindr_external_validation.py prepare      # Part A, as pre-registered
    python experiments/vindr_grounding_amendment.py prepare [--data DIR]
    python experiments/vindr_grounding_amendment.py run chexagent
    python experiments/vindr_grounding_amendment.py run maira
    python experiments/vindr_grounding_amendment.py analyze      # experiments/results/vindr_grounding_amendment.md
"""

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import vindr_external_validation as part_a  # noqa: E402  (paths only; Part A is not changed)

RESULTS = HERE / "results"
CASES = RESULTS / "vindr_grounding_cases.json"
READINGS = {"chexagent": RESULTS / "vindr_grounding_chexagent.jsonl",
            "maira": RESULTS / "vindr_grounding_maira2.jsonl"}
REPORT = RESULTS / "vindr_grounding_amendment.md"

CLASSES = {"Pleural effusion": "pleural effusion", "Atelectasis": "atelectasis",
           "Consolidation": "consolidation", "Pneumothorax": "pneumothorax"}
SEED = 20261004
BOOT = 2000
MIN_PER_FINDING = 20
MIN_POOLED = 40
CHEXAGENT_SNAPSHOT = "8f19b53a2eceda4c33b0acec6c81fbc293ad80d0"
MAIRA_REVISION = "795a2b1cd4a310624b4e3d14b5a23e41fd273deb"


# ------------------------------------------------------------------ sides

def box_sides(boxes):
    """Sides boxed, from box centres (patient's right = image left); boxes are
    [x1, y1, x2, y2] in percent of image width / height; a box with x1 < 40 and x2 > 60
    counts as both sides."""
    sides = set()
    for x1, _, x2, _ in boxes:
        if x1 < 40 and x2 > 60:
            sides |= {"right", "left"}
        else:
            sides.add("right" if (x1 + x2) / 2 < 50 else "left")
    return sides


# ------------------------------------------------------------------ prepare

def prepare(data):
    import pandas as pd
    from PIL import Image

    if CASES.exists():
        print(f"{CASES.name} already exists; not rebuilding")
        return
    if not part_a.MANIFEST.exists():
        raise SystemExit("Part A's manifest is missing: run vindr_external_validation.py prepare first")
    manifest = json.loads(part_a.MANIFEST.read_text(encoding="utf-8"))
    pngs = {r["image_id"]: r["png"] for r in manifest["images"]}

    files = list(data.rglob("annotations_test.csv"))
    assert len(files) == 1, f"expected one annotations_test.csv under {data}, found {files}"
    ann = pd.read_csv(files[0])
    for col in ("image_id", "class_name", "x_min", "y_min", "x_max", "y_max"):
        assert col in ann.columns, f"column {col} missing; columns are {list(ann.columns)}"
    rater = next((c for c in ann.columns if c.lower() in ("rad_id", "rater", "radiologist")), None)
    if rater and ann.groupby("image_id")[rater].nunique().max() > 1:
        raise SystemExit("annotations_test.csv has more than one rater per image; the amendment "
                         "assumes one set of boxes per image, so stop and decide before continuing")

    cases = []
    ann = ann[ann["class_name"].isin(CLASSES)]
    for (image_id, cls), group in ann.groupby(["image_id", "class_name"]):
        if image_id not in pngs:
            continue
        width, height = Image.open(pngs[image_id]).size
        boxes = [[r.x_min / width * 100, r.y_min / height * 100, r.x_max / width * 100, r.y_max / height * 100]
                 for r in group.itertuples()]
        sides = box_sides(boxes)
        cases.append({"image_id": image_id, "png": pngs[image_id], "finding": CLASSES[cls],
                      "truth_boxes": boxes, "unilateral": len(sides) == 1,
                      "side": next(iter(sides)) if len(sides) == 1 else "bilateral"})
    CASES.write_text(json.dumps({"annotations_file": str(files[0]), "cases": cases}, indent=1), encoding="utf-8")
    os.chmod(CASES, stat.S_IREAD)
    for f in CLASSES.values():
        uni = sum(c["finding"] == f and c["unilateral"] for c in cases)
        bi = sum(c["finding"] == f and not c["unilateral"] for c in cases)
        print(f"  {f:17} {uni} unilateral, {bi} bilateral cases"
              f"{'' if uni >= MIN_PER_FINDING else '  (per-finding: NOT evaluable, < 20)'}")


# ------------------------------------------------------------------ models

def chexagent_locator():
    from medrax.tools.xray_vqa import XRayVQATool

    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")

    def locate(png, finding, side=None):
        phrase = f"{side} {finding}" if side else finding
        prompt = f"Please locate the following phrase: {phrase}"
        response, _ = tool._generate_response([png], prompt, 128)
        _, boxes = tool._parse_grounding(response)
        return {"prompt": prompt, "raw": response, "boxes": [b["box_normalized"] for b in boxes]}
    return locate


def maira_locator():
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    kwargs = dict(cache_dir="C:/Users/hari/model-weights", trust_remote_code=True, revision=MAIRA_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/maira-2", device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        **kwargs).eval()
    processor = AutoProcessor.from_pretrained("microsoft/maira-2", **kwargs)

    def locate(png, finding, side=None):
        phrase = f"{side} {finding}".capitalize() if side else finding.capitalize()
        image = Image.open(png).convert("RGB")
        inputs = processor.format_and_preprocess_phrase_grounding_input(
            frontal_image=image, phrase=phrase, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, use_cache=True, do_sample=False)
        text = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        parse_error = False
        try:
            raw = [b for _, bxs in processor.convert_output_to_plaintext_or_grounded_sequence(text) or []
                   for b in bxs or []]
        except AssertionError:
            parse_error = True
            raw = [tuple((int(v) + 0.5) / 100 for v in m)
                   for m in re.findall(r"<box><x(\d+)><y(\d+)><x(\d+)><y(\d+)></box>", text)]
        boxes = [[v * 100 for v in processor.adjust_box_for_original_image_size(
            b, width=image.size[0], height=image.size[1])] for b in raw]
        return {"prompt": phrase, "raw": text, "boxes": boxes, "parse_error": parse_error}
    return locate


def run(model):
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    path = READINGS[model]
    done = set()
    if path.exists():
        done = {(r["image_id"], r["finding"], r["condition"])
                for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    todo = []
    for c in cases:
        conditions = ["blind"] + (["true", "wrong"] if c["unilateral"] else [])
        todo += [(c, k) for k in conditions if (c["image_id"], c["finding"], k) not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    locate = maira_locator() if model == "maira" else chexagent_locator()
    with path.open("a", encoding="utf-8") as out:
        for n, (c, condition) in enumerate(todo, 1):
            side = None
            if condition == "true":
                side = c["side"]
            elif condition == "wrong":
                side = "left" if c["side"] == "right" else "right"
            result = locate(c["png"], c["finding"], side)
            out.write(json.dumps({"image_id": c["image_id"], "finding": c["finding"], "condition": condition,
                                  "side_in_prompt": side, **result}) + "\n")
            out.flush()
            if n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)}", flush=True)


# ------------------------------------------------------------------ analyze

def lr(agree_correct, agree_wrong):
    a, b = np.asarray(agree_correct, float), np.asarray(agree_wrong, float)
    return ((a.sum() + 0.5) / (len(a) + 1)) / ((b.sum() + 0.5) / (len(b) + 1))


def cluster_ci(images, stat, *arrays):
    """95% CI of stat(*arrays) from a bootstrap that resamples images (all their cases together)."""
    rng = np.random.default_rng(SEED)
    ids = np.unique(images)
    rows = {i: np.flatnonzero(images == i) for i in ids}
    vals = []
    for _ in range(BOOT):
        idx = np.concatenate([rows[i] for i in rng.choice(ids, len(ids))])
        vals.append(stat(*(a[idx] for a in arrays)))
    return np.percentile(vals, [2.5, 97.5])


def mcnemar_one_sided(b, c):
    """Exact one-sided McNemar p for P(C=1) > P(B=1): b = cases with B=0, C=1; c = B=1, C=0."""
    from scipy.stats import binom
    n = b + c
    return 1.0 if n == 0 else float(binom.sf(b - 1, n, 0.5))


def clopper_pearson(k, n):
    from scipy.stats import beta
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def analyze():
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    lines = ["# VinDr-CXR Amendment 1 (Part B): claim-blinded vs claim-conditioned grounding", "",
             "Pre-registered in experiments/vindr_amendment_1_grounding.md. Part A is reported separately "
             "(vindr_external_validation.md).", ""]
    findings = list(CLASSES.values())
    lines += ["## Cases", "", "| finding | unilateral | bilateral |", "|---|---|---|"]
    for f in findings:
        lines.append(f"| {f} | {sum(c['finding'] == f and c['unilateral'] for c in cases)} | "
                     f"{sum(c['finding'] == f and not c['unilateral'] for c in cases)} |")
    lines.append("")

    h2_p = {}
    per_model = {}
    for model, path in READINGS.items():
        if not path.exists():
            lines += [f"## {model}: no readings", ""]
            continue
        rd = {(r["image_id"], r["finding"], r["condition"]): r
              for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
        rows = []
        for c in cases:
            if not c["unilateral"]:
                continue
            keys = [(c["image_id"], c["finding"], k) for k in ("blind", "true", "wrong")]
            if not all(k in rd for k in keys):
                continue
            wrong = "left" if c["side"] == "right" else "right"
            blind, true, wr = (box_sides(rd[k]["boxes"]) for k in keys)
            rows.append({"image": c["image_id"], "finding": c["finding"],
                         "blind_correct": int(c["side"] in blind), "blind_wrong": int(wrong in blind),
                         "cond_correct": int(c["side"] in true), "cond_wrong": int(wrong in wr)})
        per_model[model] = rows
        name = "CheXagent" if model == "chexagent" else "MAIRA-2"
        lines += [f"## {name}", "", f"Unilateral cases with all three readings: {len(rows)}", ""]
        if len(rows) < MIN_POOLED:
            lines += [f"Pooled endpoints NOT evaluable (< {MIN_POOLED} cases).", ""]
            continue

        def block(sel, label):
            img = np.array([r["image"] for r in sel])
            get = lambda k: np.array([r[k] for r in sel])
            bc, bw, cc, cw = get("blind_correct"), get("blind_wrong"), get("cond_correct"), get("cond_wrong")
            lr_b = lr(bc, bw); lo_b, hi_b = cluster_ci(img, lr, bc, bw)
            lr_c = lr(cc, cw); lo_c, hi_c = cluster_ci(img, lr, cc, cw)
            b = int(((bw == 0) & (cw == 1)).sum()); c_ = int(((bw == 1) & (cw == 0)).sum())
            p = mcnemar_one_sided(b, c_)
            diff = cw.mean() - bw.mean()
            lo_d, hi_d = cluster_ci(img, lambda x, y: y.mean() - x.mean(), bw, cw)
            k, n = int(cw.sum()), len(cw)
            cp_lo, cp_hi = clopper_pearson(k, n)
            return {"label": label, "n": len(sel), "lr_b": (lr_b, lo_b, hi_b), "lr_c": (lr_c, lo_c, hi_c),
                    "rates": (bc.mean(), bw.mean(), cc.mean(), cw.mean()), "mcnemar": (b, c_, p),
                    "diff": (diff, lo_d, hi_d), "compliance": (k, n, cp_lo, cp_hi)}

        results = [block(rows, "pooled")]
        for f in findings:
            sel = [r for r in rows if r["finding"] == f]
            if len(sel) >= MIN_PER_FINDING:
                results.append(block(sel, f))
            else:
                lines.append(f"- {f}: {len(sel)} unilateral cases, per-finding endpoints NOT evaluable (< {MIN_PER_FINDING})")
        lines += ["", "| cases | n | blinded: agree correct / wrong | LR+_blind (95% CI) | conditioned: agree correct / wrong | "
                  "LR+_cond (95% CI) | wrong-side: blinded -> conditioned (diff, 95% CI) | McNemar b / c, one-sided p | "
                  "compliance with wrong-side prompt (95% CI) |", "|---|---|---|---|---|---|---|---|---|"]
        for r in results:
            bc, bw, cc, cw = r["rates"]
            b, c_, p = r["mcnemar"]
            k, n, lo, hi = r["compliance"]
            lines.append(f"| {r['label']} | {r['n']} | {bc:.0%} / {bw:.0%} | {r['lr_b'][0]:.2f} ({r['lr_b'][1]:.2f}-{r['lr_b'][2]:.2f}) | "
                         f"{cc:.0%} / {cw:.0%} | {r['lr_c'][0]:.2f} ({r['lr_c'][1]:.2f}-{r['lr_c'][2]:.2f}) | "
                         f"{bw:.0%} -> {cw:.0%} ({r['diff'][0]:+.0%}, {r['diff'][1]:+.0%} to {r['diff'][2]:+.0%}) | "
                         f"{b} / {c_}, {p:.2g} | {k}/{n} = {k / n:.0%} ({lo:.0%}-{hi:.0%}) |")
        pooled = results[0]
        h2_p[name] = pooled["mcnemar"][2]
        lo_b = pooled["lr_b"][1]
        lines += ["", f"**H1 ({name}, pooled):** lower 95% bound of LR+_blind = {lo_b:.2f} -> "
                  f"{'supported' if lo_b > 1 else 'not supported'}"
                  + (" (primary)" if model == "chexagent" else " (secondary)"),
                  f"LR+_blind >= 2: {'yes' if pooled['lr_b'][0] >= 2 else 'no'} (descriptive)", ""]
        k, n, lo, hi = pooled["compliance"]
        lr_c = pooled["lr_c"]
        indep = lr_c[1] <= 1 <= lr_c[2] and k / n >= 0.9
        lines += [f"Conditioned agreement independent of the ground truth (LR+_cond CI includes 1 and "
                  f"compliance >= 90%): {'yes' if indep else 'no'}", ""]

    if h2_p:
        # Holm across the two models
        ordered = sorted(h2_p.items(), key=lambda kv: kv[1])
        m, adjusted, running = len(ordered), {}, 0.0
        for i, (name, p) in enumerate(ordered):
            running = max(running, min(1.0, (m - i) * p))
            adjusted[name] = running
        lines += ["## H2 (co-primary, Holm across models)", "", "| model | one-sided McNemar p | Holm-adjusted p | H2 |",
                  "|---|---|---|---|"]
        for name, p in h2_p.items():
            lines.append(f"| {name} | {p:.2g} | {adjusted[name]:.2g} | {'supported' if adjusted[name] < 0.05 else 'not supported'} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "run", "analyze"])
    ap.add_argument("model", nargs="?", choices=["chexagent", "maira"])
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    if args.step == "prepare":
        prepare(part_a.data_dir(args.data))
    elif args.step == "run":
        run(args.model)
    else:
        analyze()
