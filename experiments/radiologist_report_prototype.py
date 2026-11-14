"""Prototype of a revised radiologist evaluation report (presentation only) for 5 of the 19 MIMIC-CXR comparisons.

Unchanged: MedGemma's report, CheXagent's presence verdicts, the reference report, every model-generated box (never
moved, resized or corrected), the Evidence Engine, all predictions and every radiologist score. Existing grounding
scores are NOT reused for the new visualizations; the prototype has new, empty rating fields. New files only (logs/,
git-ignored: MIMIC-CXR content).

Presentation rules:
  low lung volumes, hyperinflation -> the X-ray with lung outlines (segmentation), no box;
  cardiomegaly -> heart outline, plus the CheXagent and MAIRA-2 candidate boxes for comparison;
  pleural effusion, atelectasis, consolidation, pneumothorax (and any other boxed finding) -> the original CheXagent
    box, with MAIRA-2 side by side where available;
  side disagreement between the two models is flagged, no winner chosen;
  every box is labelled "Model-generated candidate location - not confirmed disease extent";
  lung-zone words only from lung segmentation, otherwise the side only;
  presence verification (CheXagent, report-blinded) is a separate column from the grounding visualization.

Cases: chosen by finding type from model outputs only, in record order (first comparison with cardiomegaly; with
low lung volumes or hyperinflation; with effusion or atelectasis; with pneumothorax; with a model side disagreement;
each not already chosen) - never by radiologist scores or the reference report.

    .venv/Scripts/python.exe experiments/radiologist_report_prototype.py
"""

import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import condition_aware_grounding as G  # noqa: E402

LOGS = ROOT / "logs"
LABEL = "Model-generated candidate location - not confirmed disease extent"
OUTLINE_ONLY = ("low lung volumes", "hyperinflation")
VERDICT = {"Supported": ("Agree", "#1a7f37"), "Indeterminate": ("Not agree (inconclusive)", "#b42318"),
           "Requires confirmation": ("Not agree", "#b42318")}


def kind(finding):
    f = finding.lower()
    if any(k in f for k in OUTLINE_ONLY):
        return "outline"
    if "cardiomegaly" in f:
        return "heart"
    return "boxes"


def pick_cases(items, mrows):
    by_case = {}
    for it in items:
        by_case.setdefault(it["case"], []).append(it)
    tests = [lambda g: any("cardiomegaly" in i["finding"] for i in g),
             lambda g: any(kind(i["finding"]) == "outline" for i in g),
             lambda g: any(i["finding"] in ("pleural effusion", "atelectasis") for i in g),
             lambda g: any("pneumothorax" in i["finding"] for i in g),
             lambda g: any(mrows[f"{i['case']}|{i['finding']}"]["comparison"] == "SIDE DISAGREES" for i in g)]
    picked = []
    for t in tests:
        for case in sorted(by_case):
            if case not in picked and t(by_case[case]):
                picked.append(case)
                break
    return picked, by_case


def where(boxes, masks, shape):
    """Location words: lung zones if the boxes fall in the segmented lungs, else the side only."""
    if not boxes:
        return "no box"
    _, zones = G.zones_for_boxes(boxes, masks)
    if zones:
        return "; ".join(z.replace(" zone", " lung") for z in zones) + " (zones from lung segmentation)"
    m = np.zeros(shape, bool)
    for b in boxes:
        m |= G.box_mask(b, shape)
    s = G.side_of(m, shape)
    return {"both": "both sides", "right": "patient's right", "left": "patient's left"}.get(s, "-") + \
        " (side only; box outside the segmented lungs)"


def draw(ax, arr, boxes=None, color="red", mask=None, caption=""):
    from matplotlib.patches import Rectangle
    ax.imshow(arr, cmap="gray")
    ax.axis("off")
    if mask is not None and mask.any():
        ax.imshow(np.ma.masked_where(~G.outline(mask), np.ones_like(arr)), cmap="winter", alpha=0.95)
    for b in boxes or []:
        x1, y1, x2, y2 = G.box_px(b, arr.shape)
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=1.6))
    ax.set_title(caption, fontsize=5.6, loc="left")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pypdf import PdfReader
    import comparison_edv_pdf as C
    from comparison_edv_pdf import parse_page
    from mimic19_grounding_compare import items as get_items

    items = get_items()
    mrows = {f"{r['case']}|{r['finding']}": r for r in json.loads(
        sorted(LOGS.glob("mimic19_grounding_chexagent_vs_maira2_*.json"))[-1].read_text(encoding="utf-8"))}
    maira = json.loads((LOGS / "mimic19_maira2_grounding.json").read_text(encoding="utf-8"))
    pages = {p["title"]: p for p in (parse_page(x.extract_text()) for x in PdfReader(
        "D:/datasets/held_out_eval_images/subset_comparisons_clean_filtered.pdf").pages)}
    picked, by_case = pick_cases(items, mrows)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = LOGS / f"radiologist_report_prototype_{stamp}.pdf"
    W_IN, H_IN = 11.7, 8.3
    mm = lambda fx, fy, fw, fh: (fx * W_IN * 25.4, (1 - fy - fh) * H_IN * 25.4, fw * W_IN * 25.4, fh * H_IN * 25.4)  # noqa: E731
    fields, titles, pno = [], [], 0
    with PdfPages(out) as pdf:
        for case in picked:
            group = by_case[case]
            first = group[0]
            page = pages[first["title"]]
            img = Image.open(first["image"]).convert("L")
            img.thumbnail((1024, 1024))
            arr = np.array(img)
            masks = G.segment(arr)
            lungs = masks["Left Lung"] | masks["Right Lung"]
            prior = Image.open(Path(first["image"]).with_name("prior.jpg")).convert("L")
            prior.thumbnail((1024, 1024))

            # ---- page A: the revised report
            fig = plt.figure(figsize=(W_IN, H_IN))
            pno += 1
            titles.append(f"{first['title']} (revised report)")
            fig.text(0.015, 0.985, f"{first['title']} - revised report layout (prototype)", fontsize=11,
                     weight="bold", va="top")
            for k, (im, lab) in enumerate(((np.array(prior), "Prior"), (arr, "Current"))):
                ax = fig.add_axes([0.015 + k * 0.125, 0.70, 0.12, 0.26])
                ax.imshow(im, cmap="gray")
                ax.axis("off")
                ax.set_title(lab, fontsize=8, loc="left")
            wrap = lambda s, n: "\n".join(textwrap.wrap(" ".join(s.split()), n))  # noqa: E731
            fig.text(0.27, 0.965, "MedGemma report (unchanged)", fontsize=7.5, weight="bold", va="top")
            fig.text(0.27, 0.945, wrap(page["generated_text"], 78), fontsize=5.6, va="top", family="monospace")
            fig.text(0.635, 0.965, "Reference report (unchanged)", fontsize=7.5, weight="bold", va="top")
            fig.text(0.635, 0.945, wrap(page["truth_text"], 78), fontsize=5.6, va="top", family="monospace")
            heads = [(0.015, "MedGemma finding"), (0.135, "Presence verification\n(CheXagent, report-blinded;\nunchanged)"),
                     (0.275, "Grounding visualization (separate from presence verification)"),
                     (0.735, "Grounding source and location"), (0.905, "Ratings (1-5)")]
            for x, h in heads:
                fig.text(x, 0.675, h, fontsize=6.8, weight="bold", va="top")
            n = len(group)
            row_h = min(0.6 / n, 0.2)
            for r, it in enumerate(group):
                y0 = 0.62 - (r + 1) * row_h
                key = f"{it['case']}|{it['finding']}"
                mb = maira[f"{it['image']}|{it['phrase']}"]["boxes"]
                comp = mrows[key]["comparison"]
                v, col = VERDICT.get(it["status"], (it["status"], "black"))
                fig.text(0.015, y0 + row_h - 0.01, "\n".join(textwrap.wrap(it["name"], 22)), fontsize=7, va="top")
                fig.text(0.135, y0 + row_h - 0.01, v + (f"\np_present {it['p_present']:.2f}" if it["p_present"] else "")
                         + f"\n{it['calibration']}", fontsize=6.3, va="top", color=col)
                k = kind(it["finding"])
                panels = []
                if k == "outline":
                    panels = [("Lung outline (segmentation; anatomy, not a model localization)", None, None, lungs)]
                elif k == "heart":
                    panels = [("Heart outline (segmentation)", None, None, masks["Heart"]),
                              (f"CheXagent box - {LABEL}", it["boxes"], "red", None),
                              (f"MAIRA-2 box - {LABEL}", mb, "orange", None)]
                else:
                    panels = [(f"CheXagent box - {LABEL}", it["boxes"], "red", None)]
                    panels.append((f"MAIRA-2 box - {LABEL}" if mb else "MAIRA-2: no box returned", mb, "orange", None))
                pw = min(0.15, (row_h - 0.02) * H_IN / W_IN * 0.95)
                for j, (cap, bx, colr, msk) in enumerate(panels):
                    ax = fig.add_axes([0.275 + j * (pw + 0.005), y0 + 0.008, pw, row_h - 0.025])
                    draw(ax, arr, bx, colr, msk, "\n".join(textwrap.wrap(cap, 34)))
                src = []
                if k == "outline":
                    src.append("No box shown: the box tool cannot represent a whole-lung finding.")
                    src.append(f"(CheXagent's original box kept in the before/after page.)")
                elif k == "heart":
                    for nm, bx in (("CheXagent", it["boxes"]), ("MAIRA-2", mb)):
                        if bx:
                            bm = np.zeros(arr.shape, bool)
                            for bb in bx:
                                bm |= G.box_mask(bb, arr.shape)
                            src.append(f"{nm}: box covers {100 * (bm & masks['Heart']).sum() / max(1, masks['Heart'].sum()):.0f}% "
                                       "of the segmented heart")
                        else:
                            src.append(f"{nm}: no box")
                else:
                    src.append(f"CheXagent: {where(it['boxes'], masks, arr.shape)}")
                    src.append(f"MAIRA-2: {where(mb, masks, arr.shape)}")
                if comp == "SIDE DISAGREES":
                    src.append("MODELS DISAGREE ON SIDE - neither is selected; please judge.")
                fig.text(0.735, y0 + row_h - 0.01, "\n".join(sum((textwrap.wrap(s, 40) for s in src), [])),
                         fontsize=5.8, va="top", color="#b42318" if comp == "SIDE DISAGREES" else "black")
                for j, lab in enumerate(("Finding", "Verdict", "Grounding")):
                    fig.text(0.905, y0 + row_h - 0.012 - j * 0.028, lab, fontsize=5.8, va="top")
                fields.append({"page": pno, "row": r + 1, "finding": it["name"], "evidence": v, "ground_truth": "",
                               "tooltip": None, "tooltip_rect": None, "gt_comment": "", "comment": "",
                               "medgemma_change": "", "comment_rect": None,
                               "scores": [(kk, mm(0.958, y0 + row_h - 0.022 - j * 0.028, 0.035, 0.02))
                                          for j, kk in enumerate(("medgemma", "chexagent", "grounding"))]})
            fig.text(0.015, 0.012, "Boxes are model-generated candidate locations, not confirmed disease extent; they "
                     "are shown exactly as produced (never moved or resized). Presence verification and grounding are "
                     "separate: a box does not confirm a finding. Ratings here are new and empty.", fontsize=6)
            pdf.savefig(fig)
            plt.close(fig)

            # ---- page B: before and after
            fig = plt.figure(figsize=(W_IN, H_IN))
            pno += 1
            titles.append(f"{first['title']} (before and after)")
            fig.text(0.015, 0.985, f"{first['title']} - before and after", fontsize=11, weight="bold", va="top")
            ax = fig.add_axes([0.015, 0.06, 0.3, 0.86])
            allb = [b for it in group for b in it["boxes"]]
            draw(ax, arr, allb, "red", None, "BEFORE: CheXagent visual proof (all boxes on one image, as in the forms)")
            for r, it in enumerate(group):
                mb = maira[f"{it['image']}|{it['phrase']}"]["boxes"]
                k = kind(it["finding"])
                y = 0.93 - (r + 1) * (0.88 / len(group))
                h = 0.88 / len(group) - 0.02
                fig.text(0.33, y + h + 0.004, f"AFTER - {it['name']}", fontsize=7.5, weight="bold")
                specs = ([("Lung outline (anatomy)", None, None, lungs),
                          (f"Original CheXagent box (kept for reference) - {LABEL}", it["boxes"], "red", None)]
                         if k == "outline" else
                         [("Heart outline", None, None, masks["Heart"]), (f"CheXagent - {LABEL}", it["boxes"], "red", None),
                          (f"MAIRA-2 - {LABEL}", mb, "orange", None)] if k == "heart" else
                         [(f"CheXagent - {LABEL}", it["boxes"], "red", None),
                          (f"MAIRA-2 - {LABEL}" if mb else "MAIRA-2: no box", mb, "orange", None)])
                pw = min(0.2, h * H_IN / W_IN)
                for j, (cap, bx, colr, msk) in enumerate(specs):
                    a = fig.add_axes([0.33 + j * (pw + 0.01), y, pw, h - 0.03])
                    draw(a, arr, bx, colr, msk, "\n".join(textwrap.wrap(cap, 40)))
            pdf.savefig(fig)
            plt.close(fig)

        # ---- explanation and recommended layout
        fig = plt.figure(figsize=(W_IN, H_IN))
        pno += 1
        titles.append("Explanation and recommendation")
        txt = """WHAT CHANGED (presentation only)
- One grounding panel per finding instead of all boxes on one image: each box is next to the finding it belongs to.
- Low lung volumes and hyperinflation: the X-ray with lung outlines, no box (the box tool returns small basal boxes;
  all 10 radiologist grounding scores for these were 1-2). The original box stays on the before/after page.
- Cardiomegaly: heart outline plus the CheXagent and MAIRA-2 candidate boxes, all labelled.
- Effusion, atelectasis, consolidation, pneumothorax and other boxed findings: the original CheXagent box with
  MAIRA-2 alongside; MAIRA-2 'no box' is stated.
- Side disagreement between the two models is flagged in red; neither model is chosen.
- Every box carries 'Model-generated candidate location - not confirmed disease extent'.
- Lung-zone words only from lung segmentation; otherwise the side only.
- Presence verification (CheXagent's report-blinded verdict) and grounding are separate columns, each with its own
  empty rating; existing grounding scores are not reused.

WHAT DID NOT CHANGE
MedGemma's report, CheXagent's presence verdicts and p_present, the reference report, every model box (never
moved, resized or corrected), the Evidence Engine, all predictions and all existing radiologist scores.

EXPECTED READABILITY GAINS (to be confirmed by the radiologists)
- The reader no longer has to work out which box belongs to which finding.
- Global findings are no longer shown as misleading small boxes.
- Model disagreement and uncertainty are visible instead of hidden behind one box.

LIMITATIONS THAT REMAIN
- The boxes themselves are unchanged: still small (median 4.4% of the image), unstable under re-sampling and image
  padding, and in 2 cases their side followed image position rather than anatomy (mirror test).
- MAIRA-2 boxes are larger but only about half are more anatomically plausible; MAIRA-2 returns no box in 4/33.
- Lung outlines and zone words come from a segmentation model that can fail on rotated or poorly positioned films;
  effusions often lie partly outside the segmented lung.
- More panels per page: rows with several findings become small; a one-finding-per-row page may be needed.
- No claim of better clinical localization until radiologists rate the new layout (empty rating fields provided).

RECOMMENDED FINAL LAYOUT (consistent for every case)
1. Header: prior and current images; MedGemma report and reference report, unchanged.
2. One row per finding: MedGemma finding | CheXagent presence verification (verdict, p_present, calibration
   status) | grounding visualization chosen by condition type (rules above; model named on every panel) |
   location words (segmentation zones or side only) and any side-disagreement flag | ratings: finding, presence
   verdict, grounding visualization (1-5, new fields).
3. The single-image 'visual proof' is dropped from the main page and kept as a reference page."""
        fig.text(0.03, 0.97, txt, fontsize=7.4, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
    C.add_form_fields(out, fields, titles)
    print(f"cases {picked}: {len(fields)} findings -> {out}")


if __name__ == "__main__":
    main()
