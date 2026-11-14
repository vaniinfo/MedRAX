"""EXPLORATORY: CheXagent vs MAIRA-2 grounding on the 19 MIMIC-CXR comparisons of the radiologist evaluation, and a
proposed report layout. MedGemma, CheXagent, MAIRA-2 and the Evidence Engine are unchanged; no prediction, threshold
or radiologist score is changed; the original evaluation reports are not overwritten (new time-stamped files in
logs/, git-ignored: MIMIC-CXR content).

Report-blinded grounding: both models get the current image and the condition name only (side words removed); never
MedGemma's report, its side, or the reference report. CheXagent's boxes are the recorded ones (same side-free phrase).
All boxes are model-generated candidate locations, not confirmed abnormalities.

Choice of grounding image per condition - FIXED BEFORE THE 19 CASES WERE COMPARED, from the independent NIH grounding
experiment (results/grounding_recall.md, 947 radiologist boxes); never from these cases' reference reports or the
doctors' scores:
  - MAIRA-2 for cardiomegaly (NIH recall@0.5 100% vs CheXagent 70%);
  - CheXagent for atelectasis, effusion, pneumonia, infiltration, pneumothorax, mass, nodule (CheXagent better or
    equal on NIH);
  - no NIH evidence (low lung volumes, hyperinflation, pulmonary edema, consolidation, other findings): the existing
    CheXagent box; for diffuse lung findings a lung-segmentation outline is shown as well, labelled as anatomy.

    .venv/Scripts/python.exe experiments/mimic19_grounding_compare.py maira     # MAIRA-2 boxes (cached)
    .venv/Scripts/python.exe experiments/mimic19_grounding_compare.py gallery   # side-by-side gallery + summary
    .venv/Scripts/python.exe experiments/mimic19_grounding_compare.py sample    # redesigned report, 4 cases
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import condition_aware_grounding as G  # noqa: E402

LOGS = ROOT / "logs"
MAIRA_CACHE = LOGS / "mimic19_maira2_grounding.json"
CHOICE = {"cardiomegaly": "MAIRA-2"}  # everything else: CheXagent (see docstring)
NIH_EVIDENCE = {"cardiomegaly", "pleural effusion", "atelectasis", "pneumonia", "infiltration", "pneumothorax",
                "mass", "nodule"}
DIFFUSE_LUNG = ("low lung volumes", "hyperinflation", "pulmonary edema", "emphysema")


def phrase(finding):
    """Condition name only: side words removed."""
    return re.sub(r"\b(right|left|bilateral|both|lower|upper)\b", "", finding, flags=re.I).strip()


def items():
    out = []
    for it in G.gallery_items():
        it["phrase"] = phrase(it["finding"])
        out.append(it)
    return out


def maira():
    import torch
    from vindr_grounding_amendment import maira_locator
    cache = json.loads(MAIRA_CACHE.read_text(encoding="utf-8")) if MAIRA_CACHE.exists() else {}
    loc = None
    for it in items():
        key = f"{it['image']}|{it['phrase']}"
        if key in cache:
            continue
        loc = loc or maira_locator()
        cache[key] = loc(it["image"], it["phrase"])
        MAIRA_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        print(f"{it['title']} {it['phrase']}: {len(cache[key]['boxes'])} boxes", flush=True)
    torch.cuda.empty_cache()
    print(f"{len(cache)} MAIRA-2 results -> {MAIRA_CACHE}")


def describe(boxes, shape):
    from medrax.tools.medgemma import describe_location
    if not boxes:
        return "no box"
    return "; ".join(dict.fromkeys(describe_location(b).replace("patient's ", "") for b in boxes))


def union_side(boxes, shape):
    if not boxes:
        return None
    m = np.zeros(shape, bool)
    for b in boxes:
        m |= G.box_mask(b, shape)
    return G.side_of(m, shape)


def compare(it, arr, masks, mboxes):
    shape = arr.shape
    cboxes = it["boxes"]
    lungs = masks["Left Lung"] | masks["Right Lung"]
    region = masks["Heart"] if G.category(it["finding"]) == "global-heart" else lungs

    def stats(boxes):
        if not boxes:
            return {"n": 0, "side": None, "area": 0.0, "in_region": None, "where": "no box"}
        m = np.zeros(shape, bool)
        for b in boxes:
            m |= G.box_mask(b, shape)
        return {"n": len(boxes), "side": G.side_of(m, shape), "area": float(m.mean()),
                "in_region": float((m & region).sum() / max(1, m.sum())), "where": describe(boxes, shape)}
    c, m = stats(cboxes), stats(mboxes)
    if not mboxes:
        agree = "MAIRA-2: no box"
    elif not cboxes:
        agree = "CheXagent: no box"
    else:
        best = max(G.iou(a, b) for a in cboxes for b in mboxes)
        same = c["side"] == m["side"] or "both" in (c["side"], m["side"])
        agree = ("agree (side and overlap)" if same and best >= 0.3 else
                 "same side, little overlap" if same else "SIDE DISAGREES")
        c["iou_between_models"] = m["iou_between_models"] = best
    return c, m, agree


def load(it, size=1024):
    img = Image.open(it["image"]).convert("L")
    full_shape = np.array(img).shape
    img.thumbnail((size, size))
    arr = np.array(img)
    return arr, G.segment(arr), full_shape


def chosen(it):
    return CHOICE.get(G.validated_or(it["finding"]) if hasattr(G, "validated_or") else it["phrase"], "CheXagent")


def model_for(it):
    for k, v in CHOICE.items():
        if k in it["phrase"].lower():
            return v
    return "CheXagent"


def draw(ax, arr, boxes=None, color="red", mask=None, title=""):
    from matplotlib.patches import Rectangle
    h, w = arr.shape
    ax.imshow(arr, cmap="gray")
    ax.axis("off")
    if mask is not None and mask.any():
        ax.imshow(np.ma.masked_where(~mask, np.ones_like(arr)), cmap="winter", alpha=0.22)
        ax.imshow(np.ma.masked_where(~G.outline(mask), np.ones_like(arr)), cmap="winter", alpha=0.95)
    for b in boxes or []:
        x1, y1, x2, y2 = G.box_px(b, (h, w))
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=2))
    ax.set_title(title, fontsize=8, loc="left")


def gallery():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    cache = json.loads(MAIRA_CACHE.read_text(encoding="utf-8"))
    rows = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = LOGS / f"mimic19_grounding_chexagent_vs_maira2_{stamp}.pdf"
    with PdfPages(out) as pdf:
        pages = []
        for it in items():
            arr, masks, full = load(it)
            mres = cache[f"{it['image']}|{it['phrase']}"]
            mboxes = mres["boxes"]
            c, m, agree = compare(it, arr, masks, mboxes)
            cat = G.category(it["finding"])
            gs = it["grounding_scores"]
            rows.append({"case": it["case"], "finding": it["finding"], "phrase": it["phrase"], "category": cat,
                         "chexagent": c, "maira2": m, "comparison": agree, "maira_parse_error": mres["parse_error"],
                         "selected_model": model_for(it), "doctor_scores_chexagent_box": gs})
            pages.append((it, arr, masks, cboxes := it["boxes"], mboxes, c, m, agree, cat, gs))
        # summary pages first
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("CheXagent vs MAIRA-2 grounding, 19 MIMIC-CXR comparisons (33 findings) - summary",
                     fontsize=12, x=0.02, ha="left")
        cnt = Counter(r["comparison"] for r in rows)
        bycat = Counter((r["category"], r["comparison"]) for r in rows)
        txt = ["Report-blinded: both models got the current image and the condition name only (no side, no report).",
               "Boxes are model-generated candidate locations, not confirmed abnormalities. Doctors' scores are for the",
               "ORIGINAL CheXagent box only; they were not used to choose or compare anything.", "",
               "Overall: " + ", ".join(f"{k}: {v}" for k, v in cnt.most_common()), "",
               f"{'category':14s} {'agree':>7s} {'same side, little overlap':>27s} {'side disagrees':>16s} {'MAIRA-2 no box':>16s}"]
        for cat in ("global-heart", "global-lung", "regional", "focal"):
            txt.append(f"{cat:14s} {bycat[(cat, 'agree (side and overlap)')]:>7d} "
                       f"{bycat[(cat, 'same side, little overlap')]:>27d} {bycat[(cat, 'SIDE DISAGREES')]:>16d} "
                       f"{bycat[(cat, 'MAIRA-2: no box')]:>16d}")
        txt += ["", f"{'case':>4s} {'finding':22s} {'CheXagent box(es)':38s} {'MAIRA-2 box(es)':38s} comparison"]
        for r in rows:
            txt.append(f"{r['case']:>4d} {r['phrase'][:22]:22s} {r['chexagent']['where'][:38]:38s} "
                       f"{r['maira2']['where'][:38]:38s} {r['comparison']}")
        fig.text(0.02, 0.93, "\n".join(txt[:60]), fontsize=6.6, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        for it, arr, masks, cboxes, mboxes, c, m, agree, cat, gs in pages:
            fig, axs = plt.subplots(1, 4, figsize=(16.5, 6.2))
            lungs = masks["Left Lung"] | masks["Right Lung"]
            anat = masks["Heart"] if cat == "global-heart" else lungs if cat == "global-lung" else None
            fig.suptitle(f"{it['title']} - MedGemma: {it['name']}   |   grounding phrase given to both models: "
                         f"'{it['phrase']}' (no side)   |   comparison: {agree}", fontsize=10, x=0.01, ha="left")
            draw(axs[0], arr, title="Current X-ray (no annotation)")
            draw(axs[1], arr, cboxes, "red", title=f"CheXagent: {c['where']}\nside {c['side']}; {c['n']} box(es); "
                 f"{100 * c['area']:.0f}% of image; "
                 + (f"{100 * c['in_region']:.0f}% inside {'heart' if cat == 'global-heart' else 'lungs'}"
                    if c['in_region'] is not None else ""))
            draw(axs[2], arr, mboxes, "orange", title=f"MAIRA-2: {m['where']}\nside {m['side']}; {m['n']} box(es); "
                 f"{100 * m['area']:.0f}% of image; "
                 + (f"{100 * m['in_region']:.0f}% inside {'heart' if cat == 'global-heart' else 'lungs'}"
                    if m['in_region'] is not None else "NO VALID BOX"))
            if anat is not None:
                draw(axs[3], arr, mask=anat, title=f"Anatomy ({'heart' if cat == 'global-heart' else 'both lungs'}; "
                     "segmentation)\nnot a model localization; not evidence of disease")
            else:
                axs[3].axis("off")
                axs[3].set_title("(no anatomy view: regional / focal finding)", fontsize=8, loc="left")
            fig.text(0.01, 0.03, f"Doctors' grounding scores for the ORIGINAL CheXagent box: "
                     + ", ".join(f"{k} {v if v is not None else 'N/A'}" for k, v in gs.items())
                     + f"   |   pre-specified grounding image for this condition: {model_for(it)}", fontsize=8)
            fig.tight_layout(rect=(0, 0.05, 1, 0.92))
            pdf.savefig(fig)
            plt.close(fig)
    js = out.with_suffix(".json")
    js.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} findings -> {out}\n{dict(Counter(r['comparison'] for r in rows))}\n-> {js}")


def sample():
    """Redesigned report for 4 representative cases (chosen by finding type only, in record order): fillable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import comparison_edv_pdf as C
    cache = json.loads(MAIRA_CACHE.read_text(encoding="utf-8"))
    its = items()
    by_case = {}
    for it in its:
        by_case.setdefault(it["case"], []).append(it)
    # representative: first case (record order) containing each finding type; no scores or reference used
    want, picked = ["global-heart", "global-lung", "regional", "focal"], []
    for cat in want:
        for case, group in sorted(by_case.items()):
            if case not in picked and any(G.category(i["finding"]) == cat for i in group):
                picked.append(case)
                break
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = LOGS / f"mimic19_redesigned_report_sample_{stamp}.pdf"
    W_IN, H_IN = 8.3, 11.7
    mm = lambda fx, fy, fw, fh: (fx * W_IN * 25.4, (1 - fy - fh) * H_IN * 25.4, fw * W_IN * 25.4, fh * H_IN * 25.4)  # noqa: E731
    fields, titles = [], []
    verdict = {"Supported": "Agree", "Indeterminate": "Not agree", "Requires confirmation": "Not agree"}
    with PdfPages(out) as pdf:
        for pno, case in enumerate(picked, 1):
            group = by_case[case]
            first = group[0]
            arr, masks, _ = load(first)
            prior = Image.open(Path(first["image"]).with_name("prior.jpg")).convert("L")
            prior.thumbnail((1024, 1024))
            fig = plt.figure(figsize=(W_IN, H_IN))
            fig.text(0.04, 0.975, f"{first['title']} - proposed layout (sample)", fontsize=11, weight="bold", va="top")
            for k, (im, lab) in enumerate(((np.array(prior), "Prior"), (arr, "Current"))):
                ax = fig.add_axes([0.04 + k * 0.47, 0.70, 0.44, 0.25])
                ax.imshow(im, cmap="gray")
                ax.axis("off")
                ax.set_title(lab, fontsize=9, loc="left")
            heads = ["MedGemma finding", "CheXagent verdict\n(presence, unchanged)", "Grounding visualization",
                     "Grounding model", "Scores (1-5)"]
            xs = [0.04, 0.25, 0.43, 0.70, 0.84]
            for x, hd in zip(xs, heads):
                fig.text(x, 0.675, hd, fontsize=7.5, weight="bold", va="top")
            row_h = min(0.6 / max(1, len(group)), 0.2)
            for r, it in enumerate(group):
                y = 0.64 - (r + 1) * row_h
                model = model_for(it)
                boxes = cache[f"{it['image']}|{it['phrase']}"]["boxes"] if model == "MAIRA-2" else it["boxes"]
                cat = G.category(it["finding"])
                anat = (masks["Left Lung"] | masks["Right Lung"]) if cat == "global-lung" else None
                fig.text(xs[0], y + row_h - 0.01, it["name"], fontsize=7.5, va="top", wrap=True)
                fig.text(xs[1], y + row_h - 0.01, verdict.get(it["status"], it["status"]), fontsize=7.5, va="top",
                         color="#1a7f37" if it["status"] == "Supported" else "#b42318", weight="bold")
                ax = fig.add_axes([xs[2], y + 0.005, 0.25, row_h - 0.015])
                draw(ax, arr, boxes if boxes else None, "orange" if model == "MAIRA-2" else "red", anat)
                label = (f"{model} box" + (" (no box returned)" if not boxes else "")
                         + ("\n+ lung outline\n(anatomy, segmentation;\nnot a localization)" if anat is not None else "")
                         + "\nCandidate location,\nnot a confirmed\nabnormality")
                fig.text(xs[3], y + row_h - 0.01, label, fontsize=6.5, va="top")
                for j, lab in enumerate(("Finding", "Verdict", "Grounding")):
                    fig.text(xs[4], y + row_h - 0.012 - j * 0.03, lab, fontsize=6.5, va="top")
                fields.append({"page": pno, "row": r + 1, "finding": it["name"], "evidence": verdict.get(it["status"], ""),
                               "ground_truth": "", "tooltip": None, "tooltip_rect": None, "gt_comment": "",
                               "comment": "", "medgemma_change": "",
                               "scores": [(k, mm(xs[4] + 0.08, y + row_h - 0.022 - j * 0.03, 0.06, 0.022))
                                          for j, k in enumerate(("medgemma", "chexagent", "grounding"))],
                               "comment_rect": None})
            fig.text(0.04, 0.02, "Grounding image per condition is fixed in advance from an independent NIH experiment "
                     "(MAIRA-2 for cardiomegaly; CheXagent otherwise). Both models receive the condition name only, "
                     "never the report or its side.", fontsize=6.5, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
            titles.append(first["title"])
    C.add_form_fields(out, fields, titles)
    print(f"cases {picked} -> {out}")


if __name__ == "__main__":
    {"maira": maira, "gallery": gallery, "sample": sample}[sys.argv[1]]()
