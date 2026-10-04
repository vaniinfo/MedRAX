"""EXPLORATORY: is box location agreement evidence that a claimed side is right?

Ground-truth location:
  - Indiana (development split only): side (right / left / bilateral) and zone from the MeSH
    qualifiers of positive findings, e.g. "Pleural Effusion/bilateral",
    "Pulmonary Atelectasis/base/right";
  - NIH: radiologist boxes (BBox_List_2017.csv); side from each box's centre. Images in the
    frozen NIH test split are excluded.
Findings: pleural effusion, atelectasis, consolidation (Indiana: Consolidation / Airspace
Disease), pneumothorax.

Models, each asked to locate the finding:
  - CheXagent: "Please locate the following phrase: {finding}" (as in the pipeline);
  - MAIRA-2: phrase grounding (microsoft/maira-2, 8-bit).
A box's side comes from its centre, as in describe_location (patient's right = image left;
a box spanning the midline counts as both sides).

Location as evidence: for every unilateral true finding, two claims are made, one with the
true side and one with the opposite side; a claim "agrees" if the model boxes the claimed
side. P(agree | correct side) / P(agree | wrong side) says how much agreement is worth as
evidence about the side. Absent findings (Indiana development negatives, claimed side drawn
at random) show whether agreement says anything about presence.

    python experiments/location_agreement.py cases       # builds the case list
    python experiments/location_agreement.py chexagent   # CheXagent boxes (cached, resumable)
    python experiments/location_agreement.py maira       # MAIRA-2 boxes (cached, resumable)
    python experiments/location_agreement.py analyze     # writes experiments/results/location_agreement_dev.md
"""

import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "experiments" / "results"
CASES = RESULTS / "location_cases.json"
CHEX = RESULTS / "location_chexagent.jsonl"
MAIRA = RESULTS / "location_maira2.jsonl"
OUT = RESULTS / "location_agreement_dev.md"
INDIANA_DIR = Path("C:/Users/hari/_workarea_/MedRAX/data/indiana_eval")
NIH_DIR = Path("D:/datasets/chestxray14/images")
NIH_BOXES = Path("D:/datasets/chestxray14/BBox_List_2017.csv")
SEED = 20261003

MESH = {"pleural effusion": r"Pleural Effusion", "atelectasis": r"Pulmonary Atelectasis",
        "consolidation": r"Consolidation|Airspace Disease", "pneumothorax": r"Pneumothorax"}
NIH_LABELS = {"Effusion": "pleural effusion", "Atelectasis": "atelectasis", "Pneumothorax": "pneumothorax"}
ZONES = [("lower", r"/(base|lower lobe)"), ("upper", r"/(apex|upper lobe)"), ("middle", r"/(middle lobe|lingula)")]
NEGATIVES_PER_FINDING = 40


def path_of(case):
    return str((INDIANA_DIR if case["dataset"] == "indiana" else NIH_DIR) / case["file"])


def cases():
    rng = random.Random(SEED)
    gt = {r["file"]: r["mesh"] for r in csv.DictReader(open(INDIANA_DIR / "GROUND_TRUTH.csv", encoding="utf-8"))}
    split = json.loads((RESULTS / "split_indiana_70_30.json").read_text(encoding="utf-8"))
    truth = {}
    for line in (RESULTS / "ppresent_indiana.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        truth[(r["file"], r["finding"])] = r["truth"]
    out = []
    for finding, pattern in MESH.items():
        negatives = []
        for file, mesh in sorted(gt.items()):
            if split.get(file) != "dev" or (file, finding) not in truth:
                continue
            if truth[(file, finding)] == 0:
                negatives.append(file)
                continue
            terms = [m for m in mesh.split(";") if re.match(rf"^({pattern})(/|$)", m)]
            sides = {s for m in terms for s in ("right", "left", "bilateral") if f"/{s}" in m}
            if not sides:
                continue
            side = "bilateral" if "bilateral" in sides or sides == {"right", "left"} else sides.pop()
            zones = sorted({z for m in terms for z, rx in ZONES if re.search(rx, m)})
            out.append({"dataset": "indiana", "file": file, "finding": finding, "present": 1,
                        "side": side, "zones": zones, "terms": terms})
        for file in rng.sample(negatives, min(NEGATIVES_PER_FINDING, len(negatives))):
            out.append({"dataset": "indiana", "file": file, "finding": finding, "present": 0,
                        "side": rng.choice(["right", "left"]), "zones": [], "terms": []})

    nih_split = json.loads((RESULTS / "split_nih_70_30.json").read_text(encoding="utf-8"))
    test = {f for s in nih_split.values() for f, v in s.items() if v == "test"}
    boxes = {}
    with NIH_BOXES.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]
    for row in rows:
        file, label, x, y, w, h = row[0], row[1], *map(float, row[2:6])
        if label in NIH_LABELS and file not in test and (NIH_DIR / file).is_file():
            boxes.setdefault((file, NIH_LABELS[label]), []).append([x, y, x + w, y + h])
    for (file, finding), bxs in sorted(boxes.items()):
        # NIH images are 1024 x 1024; patient's right is the image's left half
        sides = {"right" if (b[0] + b[2]) / 2 < 512 else "left" for b in bxs}
        side = "bilateral" if sides == {"right", "left"} else sides.pop()
        out.append({"dataset": "nih", "file": file, "finding": finding, "present": 1, "side": side,
                    "zones": [], "truth_boxes": [[v / 1024 * 100 for v in b] for b in bxs]})
    CASES.write_text(json.dumps(out, indent=1), encoding="utf-8")
    from collections import Counter
    print(Counter((c["dataset"], c["finding"], c["present"]) for c in out))


def run(model):
    cases_ = json.loads(CASES.read_text(encoding="utf-8"))
    cache = MAIRA if model == "maira" else CHEX
    done = set()
    if cache.exists():
        done = {(r["dataset"], r["file"], r["finding"]) for r in map(json.loads, cache.read_text(encoding="utf-8").splitlines())}
    todo = [c for c in cases_ if (c["dataset"], c["file"], c["finding"]) not in done]
    todo = list({(c["dataset"], c["file"], c["finding"]): c for c in todo}.values())
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    if model == "maira":
        locate = maira_locator()
    else:
        from medrax.tools.xray_vqa import XRayVQATool

        tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")

        def locate(path, finding):
            result, _ = tool.locate_findings(path, [finding])
            r = result[0]
            return {"boxes": [b["box_normalized"] for b in r["boxes"]],
                    "other_labels": r.get("substituted_labels", []),
                    "p_present": r["boxes"][0]["p_present"] if r["boxes"] else None}
    with cache.open("a", encoding="utf-8") as out:
        for n, c in enumerate(todo, 1):
            res = locate(path_of(c), c["finding"])
            out.write(json.dumps({"dataset": c["dataset"], "file": c["file"], "finding": c["finding"], **res}) + "\n")
            out.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"{n}/{len(todo)}", flush=True)


def maira_locator():
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/maira-2", cache_dir="C:/Users/hari/model-weights", trust_remote_code=True,
        device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()
    processor = AutoProcessor.from_pretrained("microsoft/maira-2", cache_dir="C:/Users/hari/model-weights",
                                              trust_remote_code=True)

    def locate(path, finding):
        image = Image.open(path).convert("RGB")
        inputs = processor.format_and_preprocess_phrase_grounding_input(
            frontal_image=image, phrase=finding.capitalize(), return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, use_cache=True, do_sample=False)
        text = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        parse_error = False
        try:
            raw_boxes = [b for _, bxs in processor.convert_output_to_plaintext_or_grounded_sequence(text) or []
                         for b in bxs or []]
        except AssertionError:
            # Malformed output (e.g. an unclosed tag): read complete boxes from the text. Box
            # values are bins of a 100-cell grid; the processor's parser maps bin n to (n + 0.5) / 100
            parse_error = True
            raw_boxes = [tuple((int(v) + 0.5) / 100 for v in m) for m in
                         re.findall(r"<box><x(\d+)><y(\d+)><x(\d+)><y(\d+)></box>", text)]
        boxes = []
        for b in raw_boxes:
            x1, y1, x2, y2 = processor.adjust_box_for_original_image_size(b, width=image.size[0], height=image.size[1])
            boxes.append([round(v * 100, 1) for v in (x1, y1, x2, y2)])
        return {"boxes": boxes, "raw": text, "parse_error": parse_error}
    return locate


def box_sides(boxes):
    sides = set()
    for x1, _, x2, _ in boxes:
        if x1 < 40 and x2 > 60:
            sides |= {"right", "left"}
        else:
            sides.add("right" if (x1 + x2) / 2 < 50 else "left")
    return sides


def box_zones(boxes):
    thirds = [("upper", 0, 100 / 3), ("middle", 100 / 3, 200 / 3), ("lower", 200 / 3, 100)]
    return {z for _, y1, _, y2 in boxes for z, top, bottom in thirds
            if (min(y2, bottom) - max(y1, top)) / max(y2 - y1, 1e-6) >= 0.25}


def analyze():
    cases_ = json.loads(CASES.read_text(encoding="utf-8"))
    models = {}
    for name, path in (("CheXagent", CHEX), ("MAIRA-2", MAIRA)):
        if path.exists():
            models[name] = {(r["dataset"], r["file"], r["finding"]): r
                            for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    rng = np.random.default_rng(SEED)

    def rate(xs):
        return f"{np.mean(xs):.0%} ({sum(xs)}/{len(xs)})" if xs else "-"

    def lr_ci(a, b, n=2000):
        """P(agree | correct) / P(agree | wrong), +0.5 (Haldane) correction so a zero count
        of wrong-side agreements gives a finite estimate."""
        a, b = np.array(a, float), np.array(b, float)
        ratio = lambda x, y: ((x.sum() + 0.5) / (len(x) + 1)) / ((y.sum() + 0.5) / (len(y) + 1))
        boots = []
        for _ in range(n):
            ia, ib = rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b))
            boots.append(ratio(a[ia], b[ib]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return f"{ratio(a, b):.1f} ({lo:.1f}-{hi:.1f})"

    lines = ["# EXPLORATORY: location agreement as evidence (development data)", "",
             "Indiana: development split, side/zone from MeSH qualifiers. NIH: radiologist boxes "
             "(BBox_List_2017.csv), frozen test-split images excluded. 'Agrees with a claim' = the model "
             "boxes the claimed side. Wrong-side claims are the true side flipped (unilateral findings "
             "only). LR = P(agree | correct side) / P(agree | wrong side) with a +0.5 correction, 95% "
             "bootstrap CI. Caution: NIH usually has one radiologist box per finding even when the finding "
             "is bilateral, so for NIH effusion and atelectasis a model that boxes both sides is counted "
             "as agreeing with the 'wrong' side; NIH pneumothorax is not affected in this way.", ""]
    for name, res in models.items():
        lines += [f"## {name}", "",
                  "| data | finding | true cases | boxed | side exactly right | unilateral: true side boxed | "
                  "unilateral: wrong side only | agree, correct claim | agree, wrong claim | LR for side | "
                  "absent findings: agree with random claim |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for dataset in ("indiana", "nih"):
            for finding in MESH:
                pos = [c for c in cases_ if c["dataset"] == dataset and c["finding"] == finding and c["present"]]
                neg = [c for c in cases_ if c["dataset"] == dataset and c["finding"] == finding and not c["present"]]
                pos = [c for c in pos if (dataset, c["file"], finding) in res]
                if not pos:
                    continue
                boxed, exact, uni_true, uni_wrong_only, agree_ok, agree_bad = [], [], [], [], [], []
                for c in pos:
                    sides = box_sides(res[(dataset, c["file"], finding)]["boxes"])
                    truth = {"right", "left"} if c["side"] == "bilateral" else {c["side"]}
                    boxed.append(int(bool(sides)))
                    exact.append(int(sides == truth))
                    if c["side"] != "bilateral":
                        other = "left" if c["side"] == "right" else "right"
                        uni_true.append(int(c["side"] in sides))
                        uni_wrong_only.append(int(bool(sides) and c["side"] not in sides))
                        agree_ok.append(int(c["side"] in sides))
                        agree_bad.append(int(other in sides))
                absent = [int(c["side"] in box_sides(res[(dataset, c["file"], finding)]["boxes"]))
                          for c in neg if (dataset, c["file"], finding) in res]
                lr = lr_ci(agree_ok, agree_bad) if agree_ok else "-"
                lines.append(f"| {dataset} | {finding} | {len(pos)} | {rate(boxed)} | {rate(exact)} | {rate(uni_true)} | "
                             f"{rate(uni_wrong_only)} | {rate(agree_ok)} | {rate(agree_bad)} | {lr} | {rate(absent)} |")
        # Zones (Indiana, where the MeSH term gives one) and NIH pointing game
        zone_hits, point_hits = [], []
        for c in cases_:
            r = res.get((c["dataset"], c["file"], c["finding"]))
            if not r or not c["present"]:
                continue
            if c["zones"]:
                zone_hits.append(int(bool(set(c["zones"]) & box_zones(r["boxes"]))))
            if c.get("truth_boxes"):
                centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in r["boxes"]]
                point_hits.append(int(any(t[0] <= x <= t[2] and t[1] <= y <= t[3]
                                          for x, y in centres for t in c["truth_boxes"])))
        lines += ["", f"- Indiana zone (a box in the true zone, where MeSH gives one): {rate(zone_hits)}",
                  f"- NIH pointing game (a box centre inside a radiologist box): {rate(point_hits)}", ""]

    if len(models) == 2:
        a, b = models["CheXagent"], models["MAIRA-2"]
        lines += ["## Both models", "", "Unilateral true findings: claim agreement when both models box the claimed side.", "",
                  "| data | agree (both), correct claim | agree (both), wrong claim | the two models' sides match |",
                  "|---|---|---|---|"]
        for dataset in ("indiana", "nih"):
            ok, bad, same = [], [], []
            for c in cases_:
                k = (c["dataset"], c["file"], c["finding"])
                if c["dataset"] != dataset or not c["present"] or k not in a or k not in b:
                    continue
                sa, sb = box_sides(a[k]["boxes"]), box_sides(b[k]["boxes"])
                same.append(int(sa == sb))
                if c["side"] != "bilateral":
                    other = "left" if c["side"] == "right" else "right"
                    ok.append(int(c["side"] in sa and c["side"] in sb))
                    bad.append(int(other in sa and other in sb))
            lines.append(f"| {dataset} | {rate(ok)} | {rate(bad)} | {rate(same)} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    step = sys.argv[1]
    {"cases": cases, "chexagent": lambda: run("chexagent"), "maira": lambda: run("maira"),
     "analyze": analyze}[step]()
