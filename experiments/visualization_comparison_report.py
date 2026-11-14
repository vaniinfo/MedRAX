"""Focused visualization comparison report for the six comparisons of the final prototype (Comps 8, 15, 7, 13, 3, 5).

Shows each finding's PREVIOUS radiologist evaluation (Dr Franco, Dr Siddharth; original scores and comments, read
directly from the returned PDF forms), the original CheXagent box (A) and the anatomy-aware visualization (B), and
asks one question only where B differs meaningfully from A. The original four ratings are not asked again.

Nothing is modified: model outputs, boxes, Evidence Engine results, scores, comments, source reports and the earlier
prototype PDF stay as they are. New time-stamped files in logs/ (MIMIC-CXR content).

Score restoration
  Sources: the returned PDFs listed in doctor_scores_to_excel.FORMS (field values read from the PDF, mapped through
  each form's field sidecar). Match = the exact source image (v26 record path) + condition + row type 'MedGemma
  stated'. Each value is cross-checked against the doctor-response workbook.
  Duplicates: Dr Franco - Form A preferred over the 19-case form (established rule); the 19-case scores are kept in
  the audit. Dr Siddharth - no rule has been established for Form B vs the 19-case form, so both are shown and the
  case is listed for manual reconciliation. The boxes and verdicts shown in Forms A/B were checked to be identical to
  the 19-case form's for the duplicated comparisons.
  Blank cells are shown as 'blank (N/A)' (the form's N/A convention); a finding with no score at all from a doctor
  is 'Not evaluated'.

Visualization B
  low lung volumes, hyperinflation -> lung segmentation outline; cardiomegaly -> heart segmentation outline (both
  always differ from a box: question asked); regional/focal findings (effusion, atelectasis, pneumothorax,
  pneumonia, edema, ...) -> MAIRA-2 box only if MAIRA-2 returned a box that differs meaningfully from CheXagent's
  (different side or best IoU < 0.5); otherwise A is retained alone and no question is asked. Measurements only
  under the existing quality and projection rules (radiologist_report_final.decide).

    .venv/Scripts/python.exe experiments/visualization_comparison_report.py
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
import radiologist_report_final as F  # noqa: E402

LOGS = ROOT / "logs"
CASES = F.CASES
DIMS = (("medgemma", "MedGemma finding"), ("change", "MedGemma interval change"),
        ("chexagent", "CheXagent presence evidence"), ("grounding", "CheXagent grounding"))
WB_COL = {"medgemma": "Score: MedGemma finding", "chexagent": "Score: CheXagent evidence",
          "change": "Score: MedGemma change", "grounding": "Score: CheXagent grounding"}
DOCTOR = {"Dr Franco": "Dr. Franco", "Dr Siddarth": "Dr. Siddharth"}
QUESTION = "Compared with the original CheXagent box, does the new visualization help you assess this finding?"
OPTIONS = ["", "New visualization is more useful", "Both are equally useful", "Original CheXagent box is more useful",
           "Neither is useful"]
FORM_NAME = {"A": "Form A", "B": "Form B", "MIMIC-19": "19-case form"}


# ------------------------------------------------------------------ step 1: scores from the returned PDFs
def restore_scores(v26_by_image):
    import doctor_scores_to_excel as D
    from comparison_edv_pdf import validated_name
    from openpyxl import load_workbook
    from pypdf import PdfReader
    wanted = {r["image"]: int(r["title"].split()[1]) for r in v26_by_image.values()
              if int(r["title"].split()[1]) in CASES}
    out = []
    for doctor, form, pdf, side_path, rec_path, source in D.FORMS:
        if form not in ("A", "B", "MIMIC-19"):
            continue
        side = json.loads(side_path.read_text(encoding="utf-8"))
        loaded = json.loads(rec_path.read_text(encoding="utf-8"))
        rows = D.rows_mixed(source[-1]) if source.startswith("mixed_") else D.rows_nineteen()
        assert len(rows) == len(side)
        values = {k: str(v.get("/V") or "").strip() for k, v in (PdfReader(str(pdf)).get_fields() or {}).items()}
        if source == "nineteen":
            img_of = {r["title"]: r["image"] for r in loaded}
        else:
            img_of = {f"Case {r['case']} of": r.get("image") for r in loaded}
        for (title, item), s in zip(rows, side):
            if item is None or item["status"] in ("Consider", "Missed"):
                continue
            img = img_of.get(title) if source == "nineteen" else img_of.get(" ".join(title.split()[:2]) + " of")
            if img not in wanted:
                continue
            get = lambda k: values.get(s["field_prefix"] + "_" + k, "")  # noqa: E731
            sc = {k: (int(get(k)) if get(k).isdigit() else None) for k, _ in DIMS}
            out.append({"doctor": doctor, "form": form, "pdf": pdf.name, "form_case": title.split("  (")[0],
                        "field_prefix": s["field_prefix"], "comparison": wanted[img], "finding": item["finding"],
                        "condition": validated_name(item["finding"]) or item["finding"], "name": item["name"],
                        "scores": sc, "comment": get("comment")})
    # cross-check against the workbook
    book = sorted(Path("D:/datasets/doctor_response").glob("doctor_responses_*.xlsx"))[-1]
    ws = load_workbook(book, read_only=True)["All rows"]
    rows = list(ws.iter_rows(values_only=True))
    H = {h: i for i, h in enumerate(rows[0])}
    wb = {}
    for r in rows[1:]:
        if r[H["Row type"]] == "MedGemma stated":
            wb[(r[H["Doctor"]], r[H["Form"]], r[H["Case"]], r[H["Condition"]])] = r
    for o in out:
        case = int(o["form_case"].split()[1])
        r = wb.get((o["doctor"], o["form"], case, o["condition"]))
        if r is None:
            o["workbook_check"] = "NOT FOUND in workbook"
        else:
            same = all(r[H[WB_COL[k]]] == o["scores"][k] for k, _ in DIMS) and \
                (r[H["Doctor comment"]] or "") == (o["comment"] or "")
            o["workbook_check"] = "match" if same else "MISMATCH with workbook"
    return out, book.name


def select(restored, comp, finding):
    """{doctor: [(entry, role)]}, notes. role: 'shown' or 'audit only'."""
    sel, notes = {}, []
    for doc in ("Dr Franco", "Dr Siddarth"):
        ents = [e for e in restored if e["doctor"] == doc and e["comparison"] == comp and e["finding"] == finding]
        if len(ents) <= 1:
            sel[doc] = [(e, "shown") for e in ents]
            continue
        forms = {e["form"] for e in ents}
        if doc == "Dr Franco" and forms == {"A", "MIMIC-19"}:
            sel[doc] = [(e, "shown" if e["form"] == "A" else "audit only (Form A preferred)") for e in ents]
            notes.append(f"{DOCTOR[doc]}: scored in Form A and the 19-case form; Form A shown (established rule).")
        elif len({json.dumps([e["scores"], e["comment"] or ""]) for e in ents}) == 1:
            first = sorted(ents, key=lambda e: e["form"])[0]
            sel[doc] = [(e, "shown" if e is first else "audit only (identical duplicate)") for e in ents]
            notes.append(f"{DOCTOR[doc]}: identical scores and comment in "
                         f"{' and '.join(FORM_NAME[f] for f in sorted(forms))}.")
        else:
            sel[doc] = [(e, "shown (ambiguous: both forms shown)") for e in sorted(ents, key=lambda e: e["form"])]
            notes.append(f"{DOCTOR[doc]}: scored in {' and '.join(FORM_NAME[f] for f in sorted(forms))}; no "
                         "established rule - both shown, manual reconciliation needed.")
    return sel, notes


# ------------------------------------------------------------------ step 2: visualization B
def vis_b(it, x, qc, maira):
    """(spec, label, differs, reason)."""
    from mimic19_grounding_compare import phrase
    k = F.kind(it["finding"])
    R = x["res"]["current"]
    vp = next((p for p in x["r"]["visual_proof"] if p["finding"] == it["finding"] and p.get("boxes")), None)
    cboxes = [b["box_normalized"] for b in vp["boxes"]] if vp else []
    spec, notes, reason = F.decide(it, x, qc)
    if k in ("outline", "heart"):
        if spec[0] in ("outline", "heart_outline", "ctr"):
            lab = {"outline": "Lung outline - torchxrayvision PSPNet segmentation (anatomy, not a model localization)",
                   "heart_outline": "Heart outline - torchxrayvision PSPNet segmentation (anatomy; outline not yet "
                                    "manually reviewed)",
                   "ctr": "Heart outline and CTR lines - PSPNet segmentation"}[spec[0]]
            return spec, lab, True, notes, reason
        return ("box", cboxes), "Segmentation failed quality check: original box retained", False, [], reason
    mb = (maira.get(f"{x['r']['image']}|{phrase(it['finding'])}") or {}).get("boxes") or []
    if not mb:
        return ("box", cboxes), "", False, [], "MAIRA-2 returned no box: original CheXagent box retained, no comparison."
    if not cboxes:
        return ("maira", mb), "MAIRA-2 phrase-grounding box (model-generated, not confirmed disease extent)", True, [], \
            "CheXagent recorded no box; MAIRA-2 offers one."
    shape = x["imgs"]["current"].shape

    def side(bs):
        m = np.zeros(shape, bool)
        for b in bs:
            m |= G.box_mask(b, shape)
        return G.side_of(m, shape)
    best = max(G.iou(a, b) for a in cboxes for b in mb)
    sc, sm = side(cboxes), side(mb)
    if sc != sm and "both" not in (sc, sm):
        why = f"MAIRA-2 places it on the other side ({sm} vs CheXagent {sc})."
    elif best < 0.5:
        why = f"MAIRA-2 box differs in position/extent (best overlap IoU {best:.2f})."
    else:
        return ("box", cboxes), "", False, [], f"MAIRA-2 box essentially the same as CheXagent's (IoU {best:.2f}): " \
                                               "no comparison requested."
    return ("maira", mb), "MAIRA-2 phrase-grounding box (model-generated, not confirmed disease extent)", True, [], why


def draw_img(ax, arr, R, spec, color="red"):
    from matplotlib.patches import Rectangle
    if spec[0] in ("box", "maira"):
        ax.imshow(arr, cmap="gray")
        ax.axis("off")
        for b in spec[1]:
            x1, y1, x2, y2 = G.box_px(b, arr.shape)
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=1.6))
        if not spec[1]:
            ax.text(0.5, 0.04, "no box recorded", color="yellow", ha="center", transform=ax.transAxes, fontsize=8)
    else:
        F.draw_vis(ax, arr, R, spec)
        ax.set_title("")


# ------------------------------------------------------------------ PDF fields (custom options)
def add_fields(path, fields):
    from pypdf import PdfWriter
    from pypdf.generic import (ArrayObject, BooleanObject, DictionaryObject, FloatObject, NameObject, NumberObject,
                               TextStringObject)
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
            d[NameObject("/Opt")] = ArrayObject([TextStringObject(o) for o in OPTIONS])
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


# ------------------------------------------------------------------ main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pypdf import PdfReader
    from comparison_edv_pdf import parse_page

    v26 = json.loads((LOGS / "radiologist_requested_plus_edv_scoring_form_v26.json").read_text(encoding="utf-8"))
    v26_by_image = {r["image"]: r for r in v26}
    rec = {int(r["title"].split()[1]): r for r in v26}
    restored, book = restore_scores(v26_by_image)
    N = json.loads(sorted((HERE / "results").glob("anatomical_measurements_nih_*.json"))[-1].read_text(encoding="utf-8"))
    proj_csv = A.projections()
    QC = json.loads(F.QC_FILE.read_text(encoding="utf-8"))
    maira = json.loads((LOGS / "mimic19_maira2_grounding.json").read_text(encoding="utf-8"))
    pages = {p["title"]: p for p in (parse_page(x.extract_text()) for x in PdfReader(
        "D:/datasets/held_out_eval_images/subset_comparisons_clean_filtered.pdf").pages)}
    data = [F.load_case(rec[c], N["rot_limit"], proj_csv) for c in CASES]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = LOGS / f"visualization_comparison_report_{stamp}.pdf"
    W_IN, H_IN = 11.7, 8.3
    mm = lambda fx, fy, fw, fh: (fx * W_IN * 25.4, (1 - fy - fh) * H_IN * 25.4, fw * W_IN * 25.4, fh * H_IN * 25.4)  # noqa: E731
    wrap = lambda s, n: "\n".join(textwrap.wrap(" ".join(str(s).split()), n))  # noqa: E731
    fields, summary, audit, ambiguous, pno = [], [], [], [], 0
    with PdfPages(out) as pdf:
        for x in data:
            r = x["r"]
            comp = int(r["title"].split()[1])
            page = pages[r["title"]]
            qc = QC.get(r["title"], {})
            proj = qc.get("projection_confirmed") or x["proj"]
            # overview (unchanged context)
            fig = plt.figure(figsize=(W_IN, H_IN))
            pno += 1
            fig.text(0.015, 0.985, r["title"], fontsize=13, weight="bold", va="top")
            for k, w in enumerate(("prior", "current")):
                ax = fig.add_axes([0.01 + k * 0.29, 0.05, 0.285, 0.87])
                ax.imshow(x["imgs"][w], cmap="gray")
                ax.axis("off")
                ax.set_title(w.capitalize() + " (original, no overlays)", fontsize=9, loc="left")
            fig.text(0.6, 0.94, "MedGemma report", fontsize=9, weight="bold", va="top")
            fig.text(0.6, 0.915, wrap(page["generated_text"], 72), fontsize=6.6, va="top", family="monospace")
            fig.text(0.6, 0.55, "Reference report", fontsize=9, weight="bold", va="top")
            fig.text(0.6, 0.525, wrap(page["truth_text"], 72), fontsize=6.6, va="top", family="monospace")
            fig.text(0.6, 0.1, f"Projection: {proj or 'not recorded'}\nFindings: {len(r['review'])} (following "
                     "pages)", fontsize=7.5, va="top")
            pdf.savefig(fig)
            plt.close(fig)
            for it in r["review"]:
                fig = plt.figure(figsize=(W_IN, H_IN))
                pno += 1
                spec, lab_b, differs, notes, reason = vis_b(it, x, qc, maira)
                vpb = next((p for p in r["visual_proof"] if p["finding"] == it["finding"] and p.get("boxes")), None)
                cboxes = [b["box_normalized"] for b in vpb["boxes"]] if vpb else []
                ch = r.get("change_claims", {}).get(it["finding"]) or it.get("change") or ["not stated", ""]
                v, col = F.VERD.get(it["status"], (it["status"], "black"))
                fig.text(0.015, 0.985, f"{r['title']} - {it['name']}", fontsize=13, weight="bold", va="top")
                fig.text(0.015, 0.945, f"MedGemma finding: {it['name']}", fontsize=8.5, va="top")
                fig.text(0.015, 0.92, wrap("MedGemma interval change: " + ch[0] + (f' - "{ch[1]}"' if ch[1] else ""),
                                           110), fontsize=8.5, va="top")
                fig.text(0.015, 0.875, "CheXagent presence verdict (report-blinded, unchanged): " + v +
                         (f" (p_present {it['p_present']:.2f})" if it.get("p_present") else ""), fontsize=8.5,
                         va="top", color=col)
                # A and B
                ax = fig.add_axes([0.015, 0.2, 0.27, 0.6])
                draw_img(ax, x["imgs"]["current"], x["res"]["current"], ("box", cboxes))
                ax.set_title(wrap("A. Original CheXagent grounding box (unchanged) - model-generated candidate "
                                  "location, not confirmed disease extent", 52), fontsize=7, loc="left")
                if differs:
                    ax = fig.add_axes([0.295, 0.2, 0.27, 0.6])
                    draw_img(ax, x["imgs"]["current"], x["res"]["current"], spec, color="orange")
                    ax.set_title(wrap("B. New visualization: " + lab_b, 52), fontsize=7, loc="left")
                else:
                    fig.text(0.3, 0.6, wrap("B. No genuinely new visualization for this finding.\n" + reason, 50),
                             fontsize=8, va="top", color="#555555")
                note = " ".join(notes) if notes else ""
                fig.text(0.015, 0.185, wrap(("Measurement: " + note + " " if note else "") + "Note: " + reason, 120),
                         fontsize=7, va="top", color="#1f4e9c")
                # previous radiologist evaluation
                sel, snotes = select(restored, comp, it["finding"])
                cols = [(DOCTOR[d], e) for d in ("Dr Franco", "Dr Siddarth") for e, role in sel[d]
                        if role.startswith("shown")] or []
                docs_missing = [DOCTOR[d] for d in ("Dr Franco", "Dr Siddarth") if not any(
                    role.startswith("shown") for _, role in sel[d])]
                tx = 0.585
                fig.text(tx, 0.81, "Previous radiologist evaluation", fontsize=10, weight="bold", va="top")
                fig.text(tx, 0.788, wrap("Original scores and comments from the completed forms (1-5). They rate the "
                                         "original report and the original CheXagent box (A) - not the new "
                                         "visualization.", 75), fontsize=6.6, va="top", style="italic")
                heads = [(n, e) for n, e in cols] + [(n, None) for n in docs_missing]
                cw = 0.39 / max(1, len(heads))
                y = 0.735
                for j, (n, e) in enumerate(heads):
                    src = f"{FORM_NAME[e['form']]}, {e['form_case']}" if e else "no form row"
                    fig.text(tx + 0.15 + j * 0.08, y, n + "\n" + src.replace(", ", "\n"), fontsize=6.3,
                             weight="bold", va="top")
                for i, (k, lab) in enumerate(DIMS):
                    yy = y - 0.065 - i * 0.032
                    fig.text(tx, yy, lab, fontsize=7.2, va="top")
                    for j, (n, e) in enumerate(heads):
                        if e is None or all(s is None for s in e["scores"].values()):
                            val = "Not evaluated"
                        else:
                            val = str(e["scores"][k]) if e["scores"][k] is not None else "blank (N/A)"
                        fig.text(tx + 0.15 + j * 0.08, yy, val, fontsize=7.5, va="top",
                                 weight="bold" if val.isdigit() else "normal")
                yy = y - 0.065 - 4 * 0.032 - 0.01
                fig.text(tx, yy, "Original comments", fontsize=7.2, weight="bold", va="top")
                cl = []
                for d in ("Dr Franco", "Dr Siddarth"):
                    for e, role in sel[d]:
                        if e["comment"]:
                            tag = "" if role.startswith("shown") else "; its scores are not shown"
                            cl.append(f'{DOCTOR[d]} ({FORM_NAME[e["form"]]}{tag}): "{e["comment"]}"')
                fig.text(tx, yy - 0.022, "\n".join(sum((textwrap.wrap(c, 80) for c in cl), [])) or "none",
                         fontsize=6.6, va="top")
                if snotes:
                    fig.text(tx, 0.33, "Score source: " + " ".join(snotes), fontsize=6.2, va="top", color="#7a4b00",
                             wrap=True)
                # question
                if differs:
                    fig.text(tx, 0.255, "Follow-up question", fontsize=9, weight="bold", va="top")
                    fig.text(tx, 0.232, wrap(QUESTION, 78), fontsize=7.6, va="top")
                    fig.text(tx, 0.145, "Optional comment", fontsize=7, va="top")
                    fields.append({"page": pno, "kind": "choice", "name": f"c{comp:02d}_{it['finding'][:20]}_answer",
                                   "tooltip": f"{r['title']}: {it['name']} - {QUESTION}",
                                   "rect": mm(tx, 0.165, 0.38, 0.032)})
                    fields.append({"page": pno, "kind": "text", "name": f"c{comp:02d}_{it['finding'][:20]}_comment",
                                   "tooltip": f"{r['title']}: {it['name']} - optional comment",
                                   "rect": mm(tx, 0.035, 0.38, 0.1)})
                else:
                    fig.text(tx, 0.255, "No follow-up question for this finding (no genuinely new visualization).",
                             fontsize=8, va="top", color="#555555")
                fig.text(0.015, 0.012, "Red = CheXagent box, orange = MAIRA-2 box (model-generated candidate "
                         "locations, not disease extent); blue/magenta outlines = anatomy from segmentation.",
                         fontsize=6.5)
                pdf.savefig(fig)
                plt.close(fig)
                summary.append({"comparison": r["title"], "finding": it["name"], "visualization_B": spec[0] if differs
                                else "none (A retained)", "follow_up": differs, "reason": reason, "label": lab_b})
                for d in ("Dr Franco", "Dr Siddarth"):
                    if not sel[d]:
                        audit.append({"comparison": r["title"], "finding": it["name"], "doctor": DOCTOR[d],
                                      "status": "Not evaluated (no form row)"})
                        ambiguous.append(f"{r['title']} / {it['name']} / {DOCTOR[d]}: no matching form row")
                    for e, role in sel[d]:
                        audit.append({"comparison": r["title"], "finding": it["name"], "condition": e["condition"],
                                      "doctor": DOCTOR[d], "source_form": FORM_NAME[e["form"]],
                                      "source_pdf": e["pdf"], "form_case": e["form_case"],
                                      "field_prefix": e["field_prefix"],
                                      **{f"score_{k}": e["scores"][k] for k, _ in DIMS}, "comment": e["comment"],
                                      "use": role, "workbook_check": e["workbook_check"]})
                        if "ambiguous" in role:
                            ambiguous.append(f"{r['title']} / {it['name']} / {DOCTOR[d]}: {FORM_NAME[e['form']]} "
                                             f"({e['form_case']}) scores {[e['scores'][k] for k, _ in DIMS]}")
                        if e["workbook_check"] != "match":
                            ambiguous.append(f"{r['title']} / {it['name']} / {DOCTOR[d]} {FORM_NAME[e['form']]}: "
                                             f"{e['workbook_check']}")
                        if role.startswith("audit only (Form A"):
                            shown = next(s for s, rl in sel[d] if rl == "shown")
                            if shown["scores"] != e["scores"] or (shown["comment"] or "") != (e["comment"] or ""):
                                ambiguous.append(f"{r['title']} / {it['name']} / {DOCTOR[d]}: Form A shown "
                                                 f"{[shown['scores'][k] for k, _ in DIMS]}; 19-case form differs "
                                                 f"{[e['scores'][k] for k, _ in DIMS]} (rule applied; listed for "
                                                 "information)")
    add_fields(out, fields)
    acsv = out.with_name(out.stem + "_score_audit.csv")
    keys = sorted({k for a in audit for k in a}, key=lambda k: list(audit[0]).index(k) if k in audit[0] else 99)
    with acsv.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(audit)
    md = out.with_name(out.stem + "_summary.md")
    L = [f"# Visualization comparison report - summary ({out.name})", "",
         f"Scores read from the returned PDFs and cross-checked against `{book}`. MIMIC-CXR content: do not "
         "distribute.", "", "## Findings and follow-up", "",
         "| Comparison | Finding | Visualization B | Follow-up question | Reason |", "|---|---|---|---|---|"]
    for s in summary:
        L.append(f"| {s['comparison']} | {s['finding']} | {s['visualization_B']} | {'yes' if s['follow_up'] else 'no'} "
                 f"| {s['reason']} |")
    L += ["", f"Follow-up questions: {sum(s['follow_up'] for s in summary)} of {len(summary)} findings.", "",
          "## Unmatched or ambiguous scores (manual reconciliation)", "",
          "Score order: [MedGemma finding, MedGemma interval change, CheXagent presence evidence, CheXagent grounding]; "
          "None = blank on the form.", ""] + [f"- {a}" for a in ambiguous or ["none"]]
    L += ["", f"Audit rows: {len(audit)} (`{acsv.name}`); workbook check: "
          f"{sum(a.get('workbook_check') == 'match' for a in audit)} match, "
          f"{sum(a.get('workbook_check', 'match') != 'match' for a in audit if 'workbook_check' in a)} not."]
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {out} ({pno} pages, {len(fields)} fields)\n-> {acsv}\n-> {md}")
    print("\n".join(L[6:]))


if __name__ == "__main__":
    main()
