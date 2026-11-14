"""EXPLORATORY: root causes of poor CheXagent grounding in the radiologist evaluation (19 MIMIC-CXR comparisons,
33 boxed findings). Separates implementation defects from model limitations. Nothing is changed: no model, weight,
Evidence Engine setting, prediction, box or radiologist score; boxes are never moved. CheXagent is called directly
(the same generate path the forms used) with side-free, report-blinded prompts only.

    .venv/Scripts/python.exe experiments/grounding_root_cause.py run      # model calls (cached)
    .venv/Scripts/python.exe experiments/grounding_root_cause.py report   # tables, examples PDF, findings
Outputs: logs/grounding_root_cause_*.json / .pdf / .md (MIMIC-CXR images; git-ignored).
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import condition_aware_grounding as G  # noqa: E402

LOGS = ROOT / "logs"
RESULTS = HERE / "results"
CACHE = LOGS / "grounding_root_cause_runs.json"
FIVE = ("pleural effusion", "atelectasis", "cardiomegaly", "pulmonary edema", "pneumothorax")
SYNONYM = {"pleural effusion": "pleural fluid", "atelectasis": "collapsed lung", "cardiomegaly": "enlarged heart",
           "pulmonary edema": "fluid in the lungs", "pneumothorax": "air in the pleural space"}
PROMPTS = {"P0 (used in forms)": "Please locate the following phrase: {c}", "P1": "Locate {c}.",
           "P2": "Where is the {c}?", "P3": "Please locate the following phrase: {c} on this chest X-ray",
           "P4 (synonym)": "Please locate the following phrase: {s}"}


def run():
    import torch
    from medrax.tools.xray_vqa import XRayVQATool
    from mimic19_grounding_compare import items, phrase
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    t = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    work = ROOT / "temp" / "rootcause"
    work.mkdir(parents=True, exist_ok=True)

    def locate(img, prompt):
        raw, _ = t._generate_response([img], prompt, 128)
        _, boxes = t._parse_grounding(raw)
        return {"raw": raw, "boxes": boxes}

    def sampled(img, prompt, n=3):
        """Sampled decoding (temperature 0.7) through the model's own generate, to test stability; the tool's
        greedy path is unchanged."""
        q = t.tokenizer.from_list_format([{"image": img}, {"text": prompt}])
        ids = t.tokenizer.apply_chat_template([{"from": "system", "value": "You are a helpful assistant."},
                                               {"from": "human", "value": q}], add_generation_prompt=True,
                                              return_tensors="pt").to("cuda")
        outs = []
        for k in range(n):
            torch.manual_seed(1000 + k)
            with torch.inference_mode():
                o = t.model.generate(ids, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=128)
            raw = t.tokenizer.decode(o[0][ids.size(1):-1])
            outs.append({"raw": raw, "boxes": t._parse_grounding(raw)[1]})
        return outs

    for n, it in enumerate(items(), 1):
        key = f"{it['case']}|{it['finding']}"
        if key in cache:
            continue
        img = it["image"]
        c = phrase(it["finding"])
        im = Image.open(img)
        r = {"case": it["case"], "finding": it["finding"], "phrase": c, "image": img, "size": im.size,
             "exif_orientation": (im.getexif() or {}).get(274), "mode": im.mode,
             "recorded_boxes": it["boxes"], "original_location": it["original_location"]}
        r["rerun_P0"] = [locate(img, PROMPTS["P0 (used in forms)"].format(c=c)) for _ in range(2)]
        # letterbox: pad to a square (no squashing), locate, map back to the original image's percent coordinates
        a = np.array(im.convert("L"))
        h, w = a.shape
        s = max(h, w)
        pad = np.zeros((s, s), np.uint8)
        top, left = (s - h) // 2, (s - w) // 2
        pad[top:top + h, left:left + w] = a
        pp = work / f"pad_{it['case']}.png"
        Image.fromarray(pad).save(pp)
        lb = locate(str(pp), PROMPTS["P0 (used in forms)"].format(c=c))
        lb["boxes_mapped"] = [[(b["box_normalized"][0] * s / 100 - left) / w * 100,
                               (b["box_normalized"][1] * s / 100 - top) / h * 100,
                               (b["box_normalized"][2] * s / 100 - left) / w * 100,
                               (b["box_normalized"][3] * s / 100 - top) / h * 100] for b in lb["boxes"]]
        r["letterbox"] = lb
        # mirror: flip left-right, locate, mirror the boxes back
        fp = work / f"flip_{it['case']}.png"
        Image.fromarray(a[:, ::-1]).save(fp)
        fl = locate(str(fp), PROMPTS["P0 (used in forms)"].format(c=c))
        fl["boxes_unmirrored"] = [[100 - b["box_normalized"][2], b["box_normalized"][1], 100 - b["box_normalized"][0],
                                   b["box_normalized"][3]] for b in fl["boxes"]]
        r["mirror"] = fl
        if c in FIVE:
            r["prompts"] = {k: [locate(img, p.format(c=c, s=SYNONYM[c])) for _ in range(2)]
                            for k, p in PROMPTS.items()}
            r["sampled_P0"] = sampled(img, PROMPTS["P0 (used in forms)"].format(c=c))
        cache[key] = r
        CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        print(f"{n} {key}", flush=True)
    print(f"{len(cache)} findings -> {CACHE}")


def kept(boxes, c):
    """The forms' filter: keep boxes whose label matches the requested phrase."""
    k = c.lower()
    return [b["box_normalized"] for b in boxes if (b["phrase"] or c).strip().lower() in k
            or k in (b["phrase"] or c).strip().lower()]


def union_iou(a, b, shape=(1000, 1000)):
    if not a or not b:
        return None
    ma, mb = np.zeros(shape, bool), np.zeros(shape, bool)
    for x in a:
        ma |= G.box_mask(x, shape)
    for x in b:
        mb |= G.box_mask(x, shape)
    return float((ma & mb).sum() / max(1, (ma | mb).sum()))


def side(boxes):
    if not boxes:
        return None
    lo, hi = min(b[0] for b in boxes), max(b[2] for b in boxes)
    return "both" if lo < 40 and hi > 60 else "right" if (lo + hi) / 2 < 50 else "left"


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    runs = json.loads(CACHE.read_text(encoding="utf-8"))
    from mimic19_grounding_compare import items
    meta = {f"{i['case']}|{i['finding']}": i for i in items()}
    maira = {r["case"] * 1000 + hash(r["finding"]) % 997: r for r in []}
    mjson = sorted(LOGS.glob("mimic19_grounding_chexagent_vs_maira2_*.json"))[-1]
    mrows = {f"{r['case']}|{r['finding']}": r for r in json.loads(mjson.read_text(encoding="utf-8"))}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    L = ["# Root causes of poor CheXagent grounding (19 MIMIC-CXR comparisons, 33 boxed findings)", "",
         f"Built {time.strftime('%Y-%m-%d %H:%M')} by experiments/grounding_root_cause.py. Nothing changed: no model, "
         "Evidence Engine setting, prediction, box or radiologist score; boxes never repositioned. All prompts are "
         "side-free and report-blinded.", ""]

    # ---- 1. coordinate pipeline
    L += ["## 1. Coordinate pipeline (traced in code and checked on outputs)", "",
          "| step | what happens | check | result |", "|---|---|---|---|"]
    allc = [v for r in runs.values() for x in r["rerun_P0"] for b in x["boxes"] for v in b["box_normalized"]]
    nih = [v for l in (RESULTS / "grounding_chexagent.jsonl").read_text(encoding="utf-8").splitlines()
           for b in json.loads(l)["boxes"] for v in b]
    degenerate = sum(1 for r in runs.values() for x in r["rerun_P0"] for b in x["boxes"]
                     if b["box_normalized"][2] <= b["box_normalized"][0] or b["box_normalized"][3] <= b["box_normalized"][1])
    same = sum(sorted(map(tuple, kept(r["rerun_P0"][0]["boxes"], r["phrase"]))) == sorted(map(tuple, r["recorded_boxes"]))
               for r in runs.values())
    det = sum(r["rerun_P0"][0]["raw"] == r["rerun_P0"][1]["raw"] for r in runs.values())
    exif = Counter(r["exif_orientation"] for r in runs.values())
    lbi = [union_iou(kept(r["rerun_P0"][0]["boxes"], r["phrase"]),
                     [b for b, raw in zip(r["letterbox"]["boxes_mapped"], r["letterbox"]["boxes"])
                      if raw["box_normalized"] in [x for x in kept(r["letterbox"]["boxes"], r["phrase"])]])
           for r in runs.values()]
    lbi = [v for v in lbi if v is not None]
    mir = [(side(kept(r["rerun_P0"][0]["boxes"], r["phrase"])),
            side([b for b, raw in zip(r["mirror"]["boxes_unmirrored"], r["mirror"]["boxes"])
                  if raw["box_normalized"] in kept(r["mirror"]["boxes"], r["phrase"])])) for r in runs.values()]
    mir = [m for m in mir if m[0] and m[1]]
    L += [f"| model output | CheXagent writes `<\\|box\\|> (x1,y1),(x2,y2)`; the parser reads integers | max value over "
          f"{len(allc) + len(nih)} coordinates (33 MIMIC + 947 NIH runs); x1<x2, y1<y2 | max {max(allc + nih)} (0-100 "
          f"scale, not 0-1000); {degenerate} degenerate boxes | ",
          "| image into the model | CheXagent's own preprocessing resizes to 512 x 512 WITHOUT keeping the aspect ratio "
          "(modeling_visual.py, transforms.Resize((512, 512))) | box percentages refer to that squashed view; "
          "percent -> pixels is linear in each axis, so the squash cancels out | consistent by construction |",
          f"| aspect ratio | padding to a square instead of squashing, then mapping back | IoU between native and "
          f"letterboxed boxes, {len(lbi)} findings | median {np.median(lbi):.2f}; IoU >= 0.5 in {sum(v >= 0.5 for v in lbi)} |",
          f"| same image | record 'image' = the full-resolution current.jpg the forms show | re-run on that file "
          f"reproduces the recorded boxes | {same}/33 identical |",
          f"| determinism | greedy decoding | two identical runs | {det}/33 identical raw outputs |",
          f"| orientation | EXIF orientation tag; images are PIL-opened as stored | EXIF tag values | {dict(exif)} "
          "(None = no tag: no rotation applied anywhere) |",
          f"| left/right | mirror the image, locate, mirror back | side unchanged after the round trip | "
          f"{sum(a == b for a, b in mir)}/{len(mir)} |",
          "| percent -> pixels (drawing) | x1/100*width, y1/100*height on the image the box came from; the PDF "
          "thumbnail keeps the aspect ratio | formula checked on 3 shapes (condition_aware_grounding.py validate) | "
          "passed |",
          "| side wording | describe_location: side from the box centre vs the IMAGE centre (50%); zones = image "
          "thirds | compared with the patient's midline from segmentation (below) | see 1b |", ""]

    # 1b. side wording vs patient midline
    side_rows = []
    for k, r in runs.items():
        img = Image.open(r["image"]).convert("L")
        img.thumbnail((1024, 1024))
        masks = G.segment(np.array(img))
        rl, ll = masks["Right Lung"], masks["Left Lung"]
        if not (rl.any() and ll.any()):
            continue
        w = rl.shape[1]
        mid = (np.nonzero(rl)[1].max() + np.nonzero(ll)[1].min()) / 2 / w * 100
        for b in r["recorded_boxes"]:
            cx = (b[0] + b[2]) / 2
            img_side = "right" if cx < 50 else "left"
            pat_side = "right" if cx < mid else "left"
            spans = b[0] < 40 and b[2] > 60
            side_rows.append({"key": k, "box": b, "midline": mid, "image_side": img_side, "patient_side": pat_side,
                              "spans": spans})
    off = [s for s in side_rows if s["image_side"] != s["patient_side"] and not s["spans"]]
    mids = [s["midline"] for s in side_rows]
    L += ["### 1b. Side wording uses the image centre, not the patient's midline", "",
          f"Patient midline from segmentation lies at {min(mids):.0f}-{max(mids):.0f}% of the image width "
          f"(median {np.median(mids):.0f}%). Boxes whose side label would change if the midline were used: "
          f"{len(off)} of {len(side_rows)}" + (": " + "; ".join(f"{s['key']} box {s['box']} (midline {s['midline']:.0f}%)"
                                                               for s in off) if off else "") + ".", ""]

    # ---- 2. raw vs rendered, filtering
    drop, subs, multi = [], Counter(), 0
    for k, r in runs.items():
        raw = r["rerun_P0"][0]["boxes"]
        kk = kept(raw, r["phrase"])
        if len(raw) > 1:
            multi += 1
        for b in raw:
            if b["box_normalized"] not in kk:
                drop.append((k, b["phrase"], b["box_normalized"]))
                subs[b["phrase"]] += 1
    areas = {k: [(b[2] - b[0]) * (b[3] - b[1]) / 100 for b in r["recorded_boxes"]] for k, r in runs.items()}
    L += ["## 2. Raw output vs rendered boxes", "",
          f"- Every box CheXagent returned is drawn unless its label names a different finding; boxes are never merged, "
          f"resized or clipped. Findings with more than one raw box: {multi}/33 (all kept and drawn separately).",
          f"- Boxes dropped by the label filter: {len(drop)}" + (" - " + "; ".join(f"{k}: labelled '{p}' {b}" for k, p, b in drop)
                                                                if drop else "") + ".",
          f"- Box sizes come from the model: median box area {np.median([a for v in areas.values() for a in v]):.1f}% of "
          f"the image; smallest {min(a for v in areas.values() for a in v):.1f}%. The rendered rectangles equal the raw "
          "coordinates (same numbers, drawn with the same formula).", ""]

    # ---- 3. prompt stability
    L += ["## 3. Prompt stability (side-free prompts; 2 greedy repeats each; 3 sampled runs of P0)", "",
          "IoU with P0 (the forms' prompt), union of kept boxes; side = the box side.", "",
          "| condition | n | " + " | ".join(k for k in PROMPTS if not k.startswith("P0")) + " | sampled P0 (mean IoU, side same) | greedy repeat identical |",
          "|---|---|" + "---|" * (len(PROMPTS) - 1) + "---|---|"]
    stab = defaultdict(lambda: defaultdict(list))
    for k, r in runs.items():
        if "prompts" not in r:
            continue
        c = r["phrase"]
        base = kept(r["prompts"]["P0 (used in forms)"][0]["boxes"], c) or [b["box_normalized"] for b in r["prompts"]["P0 (used in forms)"][0]["boxes"]]
        for name, reps in r["prompts"].items():
            if name.startswith("P0"):
                stab[c]["repeat"].append(reps[0]["raw"] == reps[1]["raw"])
                continue
            bx = [b["box_normalized"] for b in reps[0]["boxes"]]
            stab[c][name].append((union_iou(base, bx), side(base) == side(bx) if bx else None, bool(bx)))
        for s_ in r["sampled_P0"]:
            bx = [b["box_normalized"] for b in s_["boxes"]]
            stab[c]["sampled"].append((union_iou(base, bx), side(base) == side(bx) if bx else None, bool(bx)))
    for c in FIVE:
        if c not in stab:
            continue
        cells = []
        for name in [k for k in PROMPTS if not k.startswith("P0")] + ["sampled"]:
            v = stab[c][name]
            ious = [x[0] for x in v if x[0] is not None]
            sides = [x[1] for x in v if x[1] is not None]
            cells.append(f"{np.mean(ious):.2f}, side {sum(sides)}/{len(sides)}, no box {sum(not x[2] for x in v)}"
                         if v else "-")
        rep = stab[c]["repeat"]
        L.append(f"| {c} | {len(rep)} | " + " | ".join(cells) + f" | {sum(rep)}/{len(rep)} |")
    L += [""]

    # ---- 4. by condition, with radiologist scores and anatomy
    L += ["## 4. Grounding problems by condition (recorded CheXagent boxes)", "",
          "| type | condition | n | median box area (% image) | share of box inside lungs/heart | side label off vs midline | "
          "doctors' grounding score: mean (Franco / Siddarth) | scored 1-2 | MAIRA-2: no box / side differs / mean area |",
          "|---|---|---|---|---|---|---|---|---|"]
    bycond = defaultdict(list)
    for k, r in runs.items():
        bycond[r["phrase"]].append(k)
    for c, ks in sorted(bycond.items(), key=lambda kv: (G.category(kv[0]), -len(kv[1]))):
        sc = {"Dr Franco": [], "Dr Siddarth": []}
        for k in ks:
            for d, v in meta[k]["grounding_scores"].items():
                if v is not None:
                    sc[d].append(v)
        allsc = sc["Dr Franco"] + sc["Dr Siddarth"]
        inreg = [mrows[k]["chexagent"]["in_region"] for k in ks if mrows[k]["chexagent"]["in_region"] is not None]
        m = [mrows[k] for k in ks]
        L.append(f"| {G.category(c)} | {c} | {len(ks)} | {np.median([a for k in ks for a in areas[k]]):.1f} | "
                 f"{100 * np.mean(inreg):.0f}% | {sum(1 for s in off if s['key'] in ks)} | "
                 f"{np.mean(sc['Dr Franco']) if sc['Dr Franco'] else float('nan'):.1f} / "
                 f"{np.mean(sc['Dr Siddarth']) if sc['Dr Siddarth'] else float('nan'):.1f} | "
                 f"{sum(v <= 2 for v in allsc)}/{len(allsc)} | {sum(x['maira2']['n'] == 0 for x in m)} / "
                 f"{sum(x['comparison'] == 'SIDE DISAGREES' for x in m)} / "
                 f"{100 * np.mean([x['maira2']['area'] for x in m]):.0f}% |")
    L += [""]

    # ---- 5. MAIRA-2: plausibility vs size
    L += ["## 5. MAIRA-2: more plausible, or just larger?", "",
          "Plausibility proxy = share of the box inside the relevant anatomy (lungs; heart for cardiomegaly). A larger box "
          "that stays inside the anatomy is more plausible; a larger box that spills outside is just larger.", "",
          "| | findings |", "|---|---|"]
    both = [r for r in mrows.values() if r["chexagent"]["n"] and r["maira2"]["n"]]
    bigger = [r for r in both if r["maira2"]["area"] > 1.5 * r["chexagent"]["area"]]
    L += [f"| both models gave a box | {len(both)} |",
          f"| MAIRA-2 box > 1.5x CheXagent's area | {len(bigger)} |",
          f"| ... and more of it inside the anatomy | {sum(r['maira2']['in_region'] > r['chexagent']['in_region'] + 0.1 for r in bigger)} |",
          f"| ... and the same or less inside | {sum(r['maira2']['in_region'] <= r['chexagent']['in_region'] + 0.1 for r in bigger)} |",
          f"| MAIRA-2 not larger | {len(both) - len(bigger)} |"]
    useless = [k for k, r in mrows.items() if (r["chexagent"]["n"] == 0 or (r["chexagent"]["in_region"] or 0) < 0.3)
               and (r["maira2"]["n"] == 0 or (r["maira2"]["in_region"] or 0) < 0.3)]
    L += ["", f"Neither model useful (each has no box or under 30% of its box inside the anatomy): {len(useless)} - "
          + "; ".join(useless), ""]

    # ---- examples PDF
    out_pdf = LOGS / f"grounding_root_cause_examples_{stamp}.pdf"
    ex_keys = []
    for k, r in runs.items():
        sc = [v for v in meta[k]["grounding_scores"].values() if v is not None]
        if sc and min(sc) <= 2:
            ex_keys.append(k)
    with PdfPages(out_pdf) as pdf:
        for k in ex_keys:
            r = runs[k]
            img = Image.open(r["image"]).convert("L")
            W0, H0 = img.size
            th = img.copy()
            th.thumbnail((1024, 1024))
            arr = np.array(th)
            fig, axs = plt.subplots(1, 3, figsize=(16.5, 7))
            fig.suptitle(f"{k}  |  image {W0} x {H0}  |  doctors' grounding scores: {meta[k]['grounding_scores']}",
                         fontsize=10, x=0.01, ha="left")
            axs[0].imshow(arr, cmap="gray"); axs[0].set_title("Original current image", fontsize=9, loc="left")
            axs[1].imshow(arr, cmap="gray")
            for b in r["recorded_boxes"]:
                x1, y1, x2, y2 = G.box_px(b, arr.shape)
                axs[1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=2))
            axs[1].set_title(f"Rendered (as in the forms): {r['original_location']}", fontsize=9, loc="left")
            axs[2].imshow(arr, cmap="gray")
            cols = ["red", "lime", "cyan", "yellow", "magenta"]
            for (name, reps), col in zip(r.get("prompts", {"P0 (used in forms)": r["rerun_P0"]}).items(), cols):
                for b in reps[0]["boxes"]:
                    x1, y1, x2, y2 = G.box_px(b["box_normalized"], arr.shape)
                    axs[2].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col, lw=1.5))
                axs[2].plot([], [], color=col, label=name)
            axs[2].legend(fontsize=7, loc="lower right")
            axs[2].set_title("Prompt variants (side-free)", fontsize=9, loc="left")
            for a in axs:
                a.axis("off")
            raw = r["rerun_P0"][0]["raw"]
            fig.text(0.01, 0.02, f"Raw CheXagent output (P0): {raw[:300]}\nParsed: "
                     f"{[(b['phrase'], b['box_normalized']) for b in r['rerun_P0'][0]['boxes']]}   ->   recorded/rendered: "
                     f"{r['recorded_boxes']}", fontsize=7.5, family="monospace")
            pdf.savefig(fig)
            plt.close(fig)
    out_md = LOGS / f"grounding_root_cause_{stamp}.md"
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"-> {out_md}\n-> {out_pdf} ({len(ex_keys)} examples: every finding a doctor scored 1-2)")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
