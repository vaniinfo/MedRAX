"""Radiologist visualization evaluation report: cardiomegaly, low lung volumes, hyperinflation, pulmonary edema.

Revision of focused_conditions_report.py (same rows, same scores, same A and B images). Purpose: to evaluate whether
an alternative visualization is more useful to radiologists than the original CheXagent box. It does NOT claim that
segmentation improves diagnostic accuracy.

Changes from the focused report
  - Question (only where a CheXagent box for the finding was shown on the original form): "Which visualization better
    represents the anatomy or image features relevant to the stated finding?" (A better / B better / Equally useful /
    Neither useful) + optional comment "Does visualization B show the abnormality itself, or only the relevant
    anatomy?".
  - Previous radiologist scores are placed BELOW the question (still visible) so they do not anchor the judgement.
  - Pulmonary edema: B labelled "Lung anatomy only - does not localize edema".
  - Cardiomegaly: CTR values are provisional and unvalidated; every CTR page is flagged for manual review of the heart
    and thoracic-width landmarks (a review dropdown is provided); the thoracic width is identified as non-standard
    (outer lung edges at or above the diaphragm domes, not the inner rib margins).
  - Reference status: "Present in reference" / "Absent - explicitly negated in the reference report" (rule-based
    negation check of the report text, quote shown) / "Not mentioned in the reference report (absence not
    established)"; NIH: "Absent per NIH labels" (label not assigned; labels are text-mined).
  - Doctor disagreement flagged when the two radiologists' scores for the same finding differ by >= 2 points on the
    same dimension (original ratings unchanged, not reconciled).
  - One-page summary first.

Unchanged: all original radiologist scores and comments, model predictions, boxes, source images, Evidence Engine
outputs, earlier reports. New time-stamped files in logs/ (MIMIC-CXR content).

    .venv/Scripts/python.exe experiments/visualization_evaluation_report.py
"""

import csv
import json
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import anatomical_measurements as A  # noqa: E402
import condition_aware_grounding as G  # noqa: E402
import focused_conditions_report as FC  # noqa: E402
import radiologist_report_final as F  # noqa: E402
import visualization_comparison_report as V  # noqa: E402

LOGS = ROOT / "logs"
HELD = Path("D:/datasets/held_out_eval_images")
CONDS = FC.CONDS
DIMS = V.DIMS
DOCTOR = V.DOCTOR
FORM_NAME = FC.FORM_NAME
QUESTION = "Which visualization better represents the anatomy or image features relevant to the stated finding?"
ANSWERS = ["", "A better", "B better", "Equally useful", "Neither useful"]
COMMENT_Q = "Optional: Does visualization B show the abnormality itself, or only the relevant anatomy?"
LANDMARK = ["", "Landmarks acceptable", "Heart border incorrect", "Thoracic width incorrect", "Both incorrect",
            "Cannot judge"]
CTR_NOTE = ("PROVISIONAL, NOT VALIDATED. Manual review of the heart borders and thoracic-width landmarks required. "
            "Non-standard thoracic width: widest distance between the OUTER LUNG EDGES at or above the diaphragm "
            "domes (segmentation), not the conventional inner rib margins.")
EDEMA_B = "Lung anatomy only - does not localize edema"
PURPOSE = ("This report evaluates whether alternative visualizations are more useful to radiologists than the original "
           "CheXagent box. It makes no claim that segmentation improves diagnostic accuracy.")

NEG = {  # sentence-level explicit negation cues, per condition (rule-based; the quote is always shown)
    "cardiomegaly": [r"heart size (is |was |remains )?(normal|within normal|top[- ]normal)",
                     r"normal (heart|cardiac) size", r"cardiac (silhouette|size) (is |remains )?(normal|within normal)",
                     r"no cardiomegaly", r"heart (is )?not enlarged",
                     r"cardiomediastinal (silhouette|contours?) (is |are |remains? )?(normal|within normal|unremarkable)"],
    "pulmonary edema": [r"\bno (\w+ ){0,4}(pulmonary )?edema", r"without (\w+ ){0,3}(pulmonary )?edema",
                        r"negative for (pulmonary )?edema", r"(pulmonary )?edema (has )?resolved"],
    "low lung volumes": [r"lung volumes (are |remain )?normal", r"normal lung volumes", r"lungs are well[- ]expanded"],
    "hyperinflation": [r"no (\w+ ){0,3}hyperinflation", r"not hyperinflated", r"normal lung volumes",
                       r"lung volumes (are |remain )?normal"],
}


def reference_texts():
    """{current image key: reference report text} for the MIMIC sets (19-case, 16-case matched, Form C)."""
    import pickle
    import pandas as pd
    from pypdf import PdfReader
    from comparison_edv_pdf import parse_page
    out = {}
    pdfs = {}
    for name in ("subset_comparisons_clean_filtered.pdf", "subset_comparisons_16_matched.pdf"):
        pdfs[name] = {p["title"]: p for p in (parse_page(x.extract_text()) for x in PdfReader(str(HELD / name)).pages)}
    for r in json.loads((LOGS / "radiologist_requested_plus_edv_scoring_form_v26.json").read_text(encoding="utf-8")):
        out[r["image"]] = pdfs["subset_comparisons_clean_filtered.pdf"][r["title"]]["truth_text"]
    for f in pickle.load(open(LOGS / "mixed_scoring_phase1_v8.pkl", "rb")).values():
        for pg in f:
            if "rec" in pg and pg["rec"]["image"] not in out:
                out[pg["rec"]["image"]] = pdfs[pg["source"]][pg["rec"]["title"]]["truth_text"]
    subset = pd.read_csv(HELD / "subset.csv")
    for p in json.loads((LOGS / "mimic_heldout_inputs_2.json").read_text(encoding="utf-8")):
        out[p["current"]] = str(subset.report_after[p["row"]])
    return out


def reference_status(e, texts):
    ref = e["reference"] or ""
    if ref and ref != "NOT IN GROUND TRUTH" and not ref.startswith("No"):
        q = f' - "{e["reference_quote"]}"' if e["reference_quote"] and "Not mentioned" not in e["reference_quote"] else ""
        return "Present in reference" + (" (NIH label)" if e["dataset"].startswith("NIH") else f" ({ref})") + q
    if e["dataset"].startswith("NIH"):
        return "Absent per NIH labels (label not assigned; labels are text-mined)"
    text = texts.get(e["key"], "")
    for sent in re.split(r"(?<=[.;])\s+", " ".join(text.split())):
        for pat in NEG[e["condition"]]:
            if re.search(pat, sent, flags=re.I):
                return f'Absent - explicitly negated in the reference report: "{sent.strip()}"'
    return "Not mentioned in the reference report (absence not established)"


def disagreements(es):
    """Pairs (Franco entry, Siddharth entry) differing by >= 2 points on a dimension both scored."""
    out = []
    for a in (e for e in es if e["doctor"] == "Dr Franco"):
        for b in (e for e in es if e["doctor"] == "Dr Siddarth"):
            d = [(lab, a["scores"][k], b["scores"][k]) for k, lab in DIMS
                 if a["scores"][k] is not None and b["scores"][k] is not None and abs(a["scores"][k] - b["scores"][k]) >= 2]
            if d:
                out.append((a, b, d))
    return out


def add_fields(path, fields):
    """As visualization_comparison_report.add_fields, with per-field options."""
    saved = V.OPTIONS
    try:
        groups = {}
        for f in fields:
            groups.setdefault(tuple(f.get("options", ANSWERS)), []).append(f)
        # pypdf writer per call; add all fields in one pass by temporarily patching the option list per field
        from pypdf import PdfWriter
        from pypdf.generic import (ArrayObject, BooleanObject, DictionaryObject, FloatObject, NameObject,
                                   NumberObject, TextStringObject)
        w = PdfWriter(clone_from=str(path))
        mm = 72 / 25.4
        font = w._add_object(DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                               NameObject("/Subtype"): NameObject("/Type1"),
                                               NameObject("/BaseFont"): NameObject("/Helvetica"),
                                               NameObject("/Encoding"): NameObject("/WinAnsiEncoding")}))
        allf = ArrayObject()
        for f in fields:
            page = w.pages[f["page"] - 1]
            ph = float(page.mediabox.height)
            x, y, wd, ht = f["rect"]
            d = DictionaryObject({
                NameObject("/Type"): NameObject("/Annot"), NameObject("/Subtype"): NameObject("/Widget"),
                NameObject("/T"): TextStringObject(f["name"]), NameObject("/TU"): TextStringObject(f["tooltip"]),
                NameObject("/Rect"): ArrayObject([FloatObject(v) for v in (x * mm, ph - (y + ht) * mm, (x + wd) * mm,
                                                                            ph - y * mm)]),
                NameObject("/F"): NumberObject(4), NameObject("/P"): page.indirect_reference,
                NameObject("/V"): TextStringObject(""), NameObject("/DA"): TextStringObject("/Helv 9 Tf 0 g"),
                NameObject("/MK"): DictionaryObject({NameObject("/BG"): ArrayObject([FloatObject(v) for v in
                                                                                       (0.93, 0.96, 1.0)]),
                                                     NameObject("/BC"): ArrayObject([FloatObject(v) for v in
                                                                                       (0.47, 0.55, 0.69)])})})
            if f["kind"] == "choice":
                d[NameObject("/FT")] = NameObject("/Ch")
                d[NameObject("/Ff")] = NumberObject(1 << 17)
                d[NameObject("/Opt")] = ArrayObject([TextStringObject(o) for o in f.get("options", ANSWERS)])
            else:
                d[NameObject("/FT")] = NameObject("/Tx")
                d[NameObject("/Ff")] = NumberObject(1 << 12)
                d[NameObject("/DA")] = TextStringObject("/Helv 8 Tf 0 g")
            ref = w._add_object(d)
            if "/Annots" not in page:
                page[NameObject("/Annots")] = ArrayObject()
            page["/Annots"].append(ref)
            allf.append(ref)
        w._root_object[NameObject("/AcroForm")] = DictionaryObject({
            NameObject("/Fields"): allf, NameObject("/NeedAppearances"): BooleanObject(True),
            NameObject("/DA"): TextStringObject("/Helv 9 Tf 0 g"),
            NameObject("/DR"): DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/Helv"): font})})})
        with open(path, "wb") as fh:
            w.write(fh)
    finally:
        V.OPTIONS = saved


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    entries, page_comments, v26, book, n_wb = FC.restore()
    texts = reference_texts()
    N = json.loads(sorted((HERE / "results").glob("anatomical_measurements_nih_*.json"))[-1].read_text(encoding="utf-8"))
    QC = json.loads(F.QC_FILE.read_text(encoding="utf-8")) if F.QC_FILE.exists() else {}
    items = {}
    for e in entries:
        items.setdefault((e["key"], e["condition"], e["row_type"]), []).append(e)
    order = sorted(items, key=lambda k: (CONDS.index(k[1]), items[k][0]["dataset"], k[2], k[0]))
    readers = {}
    bis = {k: FC.box_image(items[k][0], v26, readers) for k in order}
    eligible = {k for k in order if bis[k] is not None and bis[k][0] != "embedded_other"}
    both = {k for k in order if {e["doctor"] for e in items[k]} == {"Dr Franco", "Dr Siddarth"}}
    disag = {k: disagreements(items[k]) for k in order}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = LOGS / f"visualization_evaluation_report_{stamp}.pdf"
    W_IN, H_IN = 11.7, 8.3
    mm = lambda fx, fy, fw, fh: (fx * W_IN * 25.4, (1 - fy - fh) * H_IN * 25.4, fw * W_IN * 25.4, fh * H_IN * 25.4)  # noqa: E731
    wrap = lambda s, n: "\n".join(textwrap.wrap(" ".join(str(s).split()), n))  # noqa: E731
    fields, rows_out, ctr_pages, pno = [], [], [], 1   # page 1 = summary (written after the loop is planned)
    plan_pages = {k: i + 2 for i, k in enumerate(order)}
    with PdfPages(out) as pdf:
        # ---------------- one-page summary
        fig = plt.figure(figsize=(W_IN, H_IN))
        cases_both = {k[0] for k in both}
        n_ctr_planned = None  # filled in the text below after computing
        lines = ["RADIOLOGIST VISUALIZATION EVALUATION - SUMMARY", "", wrap(PURPOSE, 110), "",
                 f"Unique findings:                                  {len(order)}",
                 f"Eligible for A-versus-B comparison:               {len(eligible)}  (a CheXagent box for the finding "
                 "was shown on the original form)",
                 f"Findings evaluated by both radiologists:          {len(both)}  ({len(cases_both)} distinct cases / "
                 "image pairs)",
                 f"Findings with substantial doctor disagreement:    {sum(1 for k in order if disag[k])}  (>= 2 points "
                 "on the same dimension; ratings not changed)",
                 f"Scored rows restored from the returned forms:     {len(entries)}  (workbook: {n_wb}; all values "
                 "read from the PDFs)", "",
                 f"{'Condition':18s} {'Findings':>8s} {'Eligible A vs B':>16s} {'Both doctors':>13s} "
                 f"{'Disagreement':>13s}"]
        for c in CONDS:
            ks = [k for k in order if k[1] == c]
            lines.append(f"{c:18s} {len(ks):8d} {sum(k in eligible for k in ks):16d} {sum(k in both for k in ks):13d} "
                         f"{sum(1 for k in ks if disag[k]):13d}")
        lines += ["", "Row types: " + ", ".join(f"{t} {sum(1 for k in order if k[2] == t)}" for t in
                                              ("MedGemma stated", "CheXagent only", "Missed by both")),
                  "Datasets: " + ", ".join(f"{d} {n}" for d, n in sorted(
                      {d: sum(1 for k in order if items[k][0]['dataset'] == d) for d in
                       {items[k][0]['dataset'] for k in order}}.items())),
                  "", "How to complete each page: look at A and B, answer the question and (optionally) the comment, "
                  "then read the previous", "radiologist evaluation below it. CTR pages also ask for a landmark review. "
                  "Previous scores rate the original report", "and box A - they are not ratings of visualization B.",
                  "", "MIMIC-CXR content (PhysioNet DUA): do not distribute."]
        txt_summary = fig.text(0.04, 0.95, "\n".join(lines), fontsize=8.6, va="top", family="monospace")
        # (CTR count appended after the loop via a second text object)
        summary_fig = fig
        pages_figs = []
        for key in order:
            es = sorted(items[key], key=lambda e: (e["doctor"], e["form"] == "MIMIC-19", e["form"]))
            e0 = es[0]
            cond, rtype = key[1], key[2]
            pno = plan_pages[key]
            fig = plt.figure(figsize=(W_IN, H_IN))
            name = e0["name"].split(" (")[0] if rtype == "MedGemma stated" else cond.capitalize()
            fig.text(0.015, 0.985, f"{name}  -  {rtype}", fontsize=13, weight="bold", va="top")
            fig.text(0.015, 0.955, f"{e0['dataset']}; " + "; ".join(sorted({f"{FORM_NAME[e['form']]} "
                     f"{e['form_case']}" for e in es})) + f"; projection: {e0['projection'] or 'not recorded'}",
                     fontsize=7.5, va="top")
            ims = {}
            for w in ("prior", "current"):
                im = Image.open(e0[w]).convert("L")
                im.thumbnail((1024, 1024))
                ims[w] = np.array(im)
            R = A.measure(A.segment_all(ims["current"]), N["rot_limit"])
            for j, (w, lab) in enumerate((("prior", "Prior"), ("current", "Current"))):
                ax = fig.add_axes([0.015 + j * 0.135, 0.57, 0.13, 0.33])
                ax.imshow(ims[w], cmap="gray")
                ax.axis("off")
                ax.set_title(lab + " (original)", fontsize=7.5, loc="left")
            bi = bis[key]
            ax = fig.add_axes([0.29, 0.53, 0.2, 0.37])
            if bi is None:
                ax.imshow(ims["current"], cmap="gray")
                ax.axis("off")
                capa = "A. No CheXagent box was shown for this row on the original form"
            elif bi[0] == "boxes":
                ax.imshow(ims["current"], cmap="gray")
                ax.axis("off")
                for b in bi[1]:
                    x1, y1, x2, y2 = G.box_px(b, ims["current"].shape)
                    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
                capa = "A. Original CheXagent box for this finding (unchanged; model-generated, not disease extent)"
            else:
                ax.imshow(bi[1])
                ax.axis("off")
                capa = (f"A. Exact box image from the returned {FORM_NAME[e0['form']]} (all CheXagent boxes of that "
                        "page; model-generated)" if bi[0] == "embedded" else
                        f"A. Box image from the returned {FORM_NAME[e0['form']]}: CheXagent placed NO box for this "
                        "finding (boxes shown belong to other findings)")
            ax.set_title(wrap(capa, 46), fontsize=6.6, loc="left")
            ax = fig.add_axes([0.5, 0.53, 0.2, 0.37])
            k = F.kind(cond)
            note, ctr = "", False
            if k == "heart":
                q = R["qc"]
                hq = QC.get(e0.get("title19"), {}).get("heart_outline", "not reviewed")
                if q["lungs"] and q["heart"] and hq != "incorrect":
                    if e0["projection"] == "PA" and q["midline"]:
                        F.draw_vis(ax, ims["current"], R, ("ctr", []))
                        ctr = True
                        labb = "B. Heart outline + CTR lines (PSPNet segmentation) - PROVISIONAL"
                        note = f"CTR {R['val']['ctr']:.2f} - {CTR_NOTE}"
                    else:
                        F.draw_vis(ax, ims["current"], R, ("heart_outline", []))
                        labb = "B. Heart outline (PSPNet segmentation; anatomy; outline not manually reviewed)"
                        note = ("AP film: no CTR shown." if e0["projection"] == "AP" else
                                "Projection not recorded: no CTR shown." if e0["projection"] is None else
                                "Rotation/midline check failed: no CTR shown.")
                else:
                    ax.imshow(ims["current"], cmap="gray")
                    ax.axis("off")
                    labb = "B. Heart outline not shown (segmentation failed QC or marked incorrect)"
            else:
                if R["qc"]["lungs"]:
                    F.draw_vis(ax, ims["current"], R, ("outline", []))
                    labb = (f"B. {EDEMA_B} (PSPNet lung segmentation)" if cond == "pulmonary edema" else
                            "B. Lung outline (PSPNet segmentation; lung anatomy, no measurement)")
                else:
                    ax.imshow(ims["current"], cmap="gray")
                    ax.axis("off")
                    labb = "B. Lung outline not shown (segmentation failed QC)"
            ax.set_title(wrap(labb, 46), fontsize=6.6, loc="left", color="#b42318" if ctr else "black")
            if note:
                fig.text(0.5, 0.52, wrap(note, 62), fontsize=6.2, va="top", color="#b42318" if ctr else "#1f4e9c")
            if ctr:
                ctr_pages.append((pno, name, e0["dataset"], R["val"]["ctr"]))
                fig.text(0.29, 0.955, "MANUAL LANDMARK REVIEW REQUIRED (provisional CTR)", fontsize=8.5,
                         weight="bold", color="#b42318", va="top")
            # finding facts
            v = F.VERD.get(e0["status"], (e0["chexagent_shown"], "black"))[0]
            facts = [f"MedGemma finding: {e0['name'] if rtype == 'MedGemma stated' else 'Not detected'}",
                     f"MedGemma interval change: {e0['change'] or ('-' if rtype != 'MedGemma stated' else 'not stated')}",
                     f"CheXagent shown on the form: {e0['chexagent_shown']}" + (f" (p_present {e0['p_present']:.2f})"
                                                                               if e0.get("p_present") else ""),
                     "Reference: " + reference_status(e0, texts)]
            fig.text(0.715, 0.9, "Finding", fontsize=8.5, weight="bold", va="top")
            fig.text(0.715, 0.88, "\n".join(sum((textwrap.wrap(s, 50) for s in facts), [])), fontsize=6.5, va="top")
            # ---- new evaluation (first)
            fig.text(0.015, 0.43, "Visualization evaluation", fontsize=10, weight="bold", va="top")
            if key in eligible:
                fig.text(0.015, 0.405, wrap(QUESTION, 58), fontsize=7.8, va="top")
                fields.append({"page": pno, "kind": "choice", "name": f"p{pno:03d}_which_better",
                               "tooltip": f"{name}: {QUESTION}", "rect": mm(0.015, 0.325, 0.3, 0.03),
                               "options": ANSWERS})
                fig.text(0.33, 0.405, wrap(COMMENT_Q, 70), fontsize=7, va="top")
                fields.append({"page": pno, "kind": "text", "name": f"p{pno:03d}_b_shows",
                               "tooltip": f"{name}: {COMMENT_Q}", "rect": mm(0.33, 0.3, 0.37, 0.075)})
            else:
                fig.text(0.015, 0.405, wrap("Not eligible for the A-versus-B question: no CheXagent box for this finding "
                                            "was shown on the original form.", 95), fontsize=7.5, va="top",
                         color="#555555")
            if ctr:
                fig.text(0.715, 0.43, "Landmark review (CTR)", fontsize=8.5, weight="bold", va="top",
                         color="#b42318")
                fig.text(0.715, 0.408, "Are the heart borders and thoracic-width landmarks acceptable?", fontsize=6.8,
                         va="top")
                fields.append({"page": pno, "kind": "choice", "name": f"p{pno:03d}_ctr_landmarks",
                               "tooltip": f"{name}: CTR landmark review", "rect": mm(0.715, 0.355, 0.27, 0.03),
                               "options": LANDMARK})
            # ---- previous evaluation (below)
            fig.plot = None
            fig.add_artist(plt.Line2D([0.015, 0.985], [0.285, 0.285], color="#999999", lw=0.6))
            fig.text(0.015, 0.275, "Previous radiologist evaluation (please answer the question above first)",
                     fontsize=9, weight="bold", va="top")
            fig.text(0.015, 0.255, "Original scores (1-5) and comments from the returned forms. They rate the original "
                     "report and box A - not visualization B. '-' not asked; 'blank (N/A)' left blank.", fontsize=6.2,
                     va="top", style="italic")
            dup_f = {e["form"] for e in es if e["doctor"] == "Dr Franco"} >= {"A", "MIMIC-19"}
            for i, (kk, lab) in enumerate(DIMS):
                fig.text(0.015, 0.19 - i * 0.022, lab, fontsize=6.8, va="top")
            for j, e in enumerate(es[:5]):
                x = 0.17 + j * 0.095
                pref = " (pref.)" if dup_f and e["doctor"] == "Dr Franco" and e["form"] == "A" else ""
                fig.text(x, 0.232, f"{DOCTOR[e['doctor']]}\n{FORM_NAME[e['form']]}\n{e['form_case']}{pref}",
                         fontsize=5.6, weight="bold", va="top")
                allblank = all(e["scores"][kk] is None for kk in e["asked"] if kk in e["scores"])
                for i, (kk, lab) in enumerate(DIMS):
                    val = "-" if kk not in e["asked"] else "Not evaluated" if allblank else \
                        str(e["scores"][kk]) if e["scores"][kk] is not None else "blank (N/A)"
                    fig.text(x, 0.19 - i * 0.022, val, fontsize=6.8, va="top", weight="bold" if val.isdigit() else
                             "normal")
            cl = [f'{DOCTOR[e["doctor"]]} ({FORM_NAME[e["form"]]}): "{e["comment"]}"' for e in es if e["comment"]]
            ctx = list(dict.fromkeys(f'{DOCTOR[e["doctor"]]} ({FORM_NAME[e["form"]]}, row "{lab}"): "{com}"'
                                     for e in es for fp, lab, com in page_comments.get((e["pdf"].name, e["page"]), [])
                                     if fp != e["field_prefix"]))
            fig.text(0.66, 0.245, "Comments on this finding", fontsize=7, weight="bold", va="top")
            fig.text(0.66, 0.228, "\n".join(sum((textwrap.wrap(c, 62) for c in cl), [])) or "none", fontsize=6.1,
                     va="top")
            fig.text(0.66, 0.14, "Other comments on the same case (context)", fontsize=7, weight="bold", va="top")
            fig.text(0.66, 0.123, "\n".join(sum((textwrap.wrap(c, 62) for c in ctx[:4]), [])) or "none",
                     fontsize=5.8, va="top", color="#444444")
            if disag[key]:
                msg = "; ".join(f"{', '.join(f'{lab} {x} vs {y}' for lab, x, y in d)} ({FORM_NAME[a['form']]} vs "
                                f"{FORM_NAME[b['form']]})" for a, b, d in disag[key])
                fig.text(0.015, 0.085, wrap("DOCTORS DIFFER SUBSTANTIALLY (>= 2 points; Dr. Franco vs Dr. Siddharth): "
                                            + msg + ". Original ratings shown unchanged; not reconciled.", 105),
                         fontsize=6.4, va="top", color="#7a4b00", weight="bold")
            fig.text(0.015, 0.008, "Red = CheXagent box (model-generated candidate location, not disease extent); "
                     "blue/magenta outlines = anatomy from segmentation.", fontsize=5.8)
            fig.text(0.015, 0.024, PURPOSE, fontsize=5.8)
            pages_figs.append(fig)
            rows_out.append({"page": pno, "condition": cond, "row_type": rtype, "dataset": e0["dataset"],
                             "forms": "; ".join(f"{DOCTOR[e['doctor']]} {FORM_NAME[e['form']]} {e['form_case']}"
                                                for e in es),
                             "eligible_A_vs_B": key in eligible, "both_doctors": key in both,
                             "disagreement": "; ".join(f"{lab} {x} vs {y}" for a, b, d in disag[key] for lab, x, y in d),
                             "reference_status": reference_status(e0, texts), "ctr_review": ctr})
        summary_fig.text(0.04, 0.2, f"CTR pages flagged for manual landmark review: {len(ctr_pages)} (pages "
                         + ", ".join(str(p[0]) for p in ctr_pages) + "). CTR values are provisional and unvalidated;"
                         "\nthe thoracic width is non-standard (outer lung edges, not inner rib margins).",
                         fontsize=8.6, va="top", family="monospace", color="#b42318")
        pdf.savefig(summary_fig)
        plt.close(summary_fig)
        for fig in pages_figs:
            pdf.savefig(fig)
            plt.close(fig)
        # comments index
        fig = plt.figure(figsize=(W_IN, H_IN))
        L = ["ALL DOCTOR COMMENTS ON THESE FOUR CONDITIONS (verbatim, from the returned PDFs)", ""]
        for r, key in zip(rows_out, order):
            for e in items[key]:
                if e["comment"]:
                    L.append(f"p{r['page']:02d}  {DOCTOR[e['doctor']]:14s} {FORM_NAME[e['form']]:13s} "
                             f"{e['form_case']:18s} {key[1]:16s}: \"{e['comment']}\"")
        fig.text(0.03, 0.97, "\n".join(L), fontsize=7.2, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
    add_fields(out, fields)
    csvp = out.with_name(out.stem + "_findings.csv")
    with csvp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    from collections import Counter
    rs = Counter(r["reference_status"].split(" (")[0].split(":")[0].split(" - ")[0] for r in rows_out)
    print(f"-> {out} ({len(order) + 2} pages, {len(fields)} fields)\n-> {csvp}")
    print(f"findings {len(order)}, eligible {len(eligible)}, both doctors {len(both)} ({len({k[0] for k in both})} "
          f"cases), disagreement {sum(1 for k in order if disag[k])}, CTR pages {len(ctr_pages)}")
    print("reference status:", dict(rs))
    for r in rows_out:
        if r["disagreement"]:
            print(f"  p{r['page']} {r['condition']} ({r['dataset']}): {r['disagreement']}")


if __name__ == "__main__":
    main()
