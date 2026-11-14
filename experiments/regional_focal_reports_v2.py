"""QC revision (v2) of the Regional and Focal findings visualization evaluation PDFs.

No model is rerun and nothing existing is modified: CheXagent boxes come from the original records / form images,
MAIRA-2 from logs/maira2_grounding_v2_*.json, the exploratory vasandesh panel is reproduced from the previous focal
PDF rendering (cropped page region; probability and flags from the stored feasibility results). Scores are restored
from the returned PDFs as before. New versioned files only.

Changes from v1
  1. Reference status from reference_sentence_audit (explicitly present / explicitly absent / possible-uncertain /
     not mentioned; laterality-specific negation; exact sentences quoted). NIH: labels only.
  2. CheXagent panels from embedded form images: the number of findings with a CheXagent box on that form page is
     counted from the form records; if > 1 the panel is labelled 'all original boxes' and the paired comparison is
     flagged as potentially ambiguous.
  3. Paired question gains 'Cannot determine / abnormality not identifiable'; response schema v2.
  4. Patient-side markers under every image panel; orientation / scaling checks (same source array for every
     overlay; embedded image aspect ratio; no mirroring in any rendering path).
  5. Automated layout QC on every page (text-image, text-text, out-of-page and form-field overlaps), larger minimum
     font, long reports continued in an appendix instead of truncated; form-field check (names, options, rects,
     fill test on a temporary copy).

    .venv/Scripts/python.exe experiments/regional_focal_reports_v2.py
"""

import csv
import json
import shutil
import sys
import tempfile
import textwrap
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import condition_grounding_models as CG  # noqa: E402
import focused_conditions_report as FC  # noqa: E402
import reference_sentence_audit as RA  # noqa: E402
import regional_focal_reports as R  # noqa: E402
import visualization_evaluation_report as VE  # noqa: E402

LOGS = ROOT / "logs"
VERSION = "v2"   # report version (v3 = 2026-11-09 pneumothorax reference correction); set from argv
SCHEMA = "v2"    # response-file schema (unchanged in v3)
QA_OPT = ["", "CheXagent box better", "MAIRA-2 box better", "Equally useful", "Neither useful",
          "Cannot determine / abnormality not identifiable"]
QB_OPT = ["", "Useful", "Partly useful", "Not useful", "Cannot judge"]
MIN_FONT = 6.3
W_IN, H_IN = 11.7, 8.3
V1 = {"regional": sorted(LOGS.glob("regional_findings_evaluation_2*.pdf"))[-1],
      "focal": sorted(LOGS.glob("focal_findings_evaluation_2*.pdf"))[-1]}


def form_page_boxes():
    """{(pdf name, page): number of rows with a CheXagent grounding score (= a CheXagent box) on that form page}."""
    import doctor_scores_to_excel as D
    out = Counter()
    for doctor, form, pdf, side_path, rec_path, source in D.FORMS:
        for s in json.loads(side_path.read_text(encoding="utf-8")):
            if "grounding" in s["scored"]:
                out[(pdf.name, int(s["field_prefix"][1:3]))] += 1
    return out


def v1_page_numbers(F):
    """Page of each finding in the v1 PDFs (same ordering rule as regional_focal_reports.render)."""
    pages = {}
    for conds, slug in ((R.REGIONAL, "regional"), (R.FOCAL, "focal")):
        Fs = sorted([f for f in F if f["key"][1] in conds], key=lambda f: ("ABCD".index(f["group"]),
                                                                          conds.index(f["key"][1]),
                                                                          f["e0"]["dataset"], f["key"][0]))
        for i, f in enumerate(Fs):
            pages[id(f)] = (slug, i + 2)
    return pages


def vasandesh_from_v1(F, pages):
    """Crop the exploratory panel from the v1 focal PDF; probability/flags from the stored feasibility results."""
    import subprocess
    gal = json.loads(sorted(LOGS.glob("condition_grounding_gallery_*.json"))[-1].read_text(encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp(prefix="v1crop_"))
    out = {}
    for f in F:
        if f["key"][1] != "pneumothorax":
            continue
        slug, pg = pages[id(f)]
        forms = {f"{e['doctor']} {e['form']} {e['form_case']}" for e in f["es"]}
        g = next((x for x in gal if x["condition"] == "pneumothorax" and set(x["forms"].split("; ")) == forms), None)
        subprocess.run(["pdftoppm", "-r", "200", "-png", "-f", str(pg), "-l", str(pg), str(V1["focal"]),
                        str(tmp / f"p{pg}")], check=True)
        page_img = Image.open(next(tmp.glob(f"p{pg}*.png")))
        Wp, Hp = page_img.size
        pw = 0.976 / 5
        x0, y0, w, h = 0.012 + 4 * pw, 0.56, pw - 0.006, 0.36      # v1 axes rectangle of the 5th panel
        crop = page_img.crop((int(x0 * Wp), int((1 - y0 - h) * Hp), int((x0 + w) * Wp), int((1 - y0) * Hp)))
        # keep only the X-ray itself: rows/columns mostly non-white (drops any title text inside the rectangle)
        dark = np.array(crop.convert("L")) < 245
        rows, cols = np.nonzero(dark.mean(1) > 0.6)[0], np.nonzero(dark.mean(0) > 0.6)[0]
        crop = crop.crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))
        m = g["models"]["ptx_vasandesh"] if g else None
        warn = []
        if m and m["side"]:
            fl = m["stab"]["horizontal flip (mapped back)"]
            if fl["dice"] < 0.5 or (fl["side"] and fl["side"] != m["side"]):
                warn.append(f"MIRROR INSTABILITY (flip Dice {fl['dice']:.2f}, side {m['side']} -> {fl['side']})")
            if m["area"] < 0.005:
                warn.append("very small region")
            if m["side"] == "both":
                warn.append("scattered on both sides")
        out[id(f)] = dict(img=np.array(crop.convert("RGB")), p=m["p"] if m else None,
                          shown=bool(m and m["side"] and m["p"] >= 0.5), warn=warn, region=bool(m and m["side"]))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def layout_issues(fig, field_rects):
    """Overlaps between texts, image axes, page edges and form fields (figure-fraction boxes)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = []
    for t in fig.texts:
        if t.get_text().strip():
            boxes.append(("text", t.get_text()[:40], inv.transform_bbox(t.get_window_extent(r))))
    axes = []
    for ax in fig.axes:
        axes.append(inv.transform_bbox(ax.get_window_extent(r)))
        for t in (ax.title, ax._left_title, ax._right_title):   # set_title(loc='left') uses _left_title
            if t.get_text().strip():
                boxes.append(("title", t.get_text()[:40], inv.transform_bbox(t.get_window_extent(r))))
    issues = []

    def ov(a, b, tol=0.0015):
        return (min(a.x1, b.x1) - max(a.x0, b.x0) > tol) and (min(a.y1, b.y1) - max(a.y0, b.y0) > tol)
    for kind, txt, b in boxes:
        if b.x0 < -0.001 or b.x1 > 1.001 or b.y0 < -0.001 or b.y1 > 1.001:
            issues.append(f"{kind} outside page: '{txt}'")
        for a in axes:
            if ov(a, b):
                issues.append(f"{kind} overlaps an image: '{txt}'")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if ov(boxes[i][2], boxes[j][2]):
                issues.append(f"text overlap: '{boxes[i][1]}' / '{boxes[j][1]}'")
    from matplotlib.transforms import Bbox
    for name, (x, y, w, h) in field_rects:
        fb = Bbox.from_bounds(x, y, w, h)
        for kind, txt, b in boxes:
            if ov(fb, b):
                issues.append(f"field {name} overlaps text '{txt}'")
        for a in axes:
            if ov(fb, a):
                issues.append(f"field {name} overlaps an image")
    return sorted(set(issues))


def render(F, conds, slug, title, out, page_comments, texts, ambiguity, vas, refaudit, focal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    Fs = sorted([f for f in F if f["key"][1] in conds], key=lambda f: ("ABCD".index(f["group"]),
                                                                      conds.index(f["key"][1]),
                                                                      f["e0"]["dataset"], f["key"][0]))
    mm = lambda fx, fy, fw, fh: (fx * W_IN * 25.4, (1 - fy - fh) * H_IN * 25.4, fw * W_IN * 25.4, fh * H_IN * 25.4)  # noqa: E731
    wrap = lambda s, n: textwrap.wrap(" ".join(str(s).split()), n)  # noqa: E731
    fields, resp, qc, appendix, pno = [], [], {}, [], 1
    figs = []
    # ---- cover
    fig = plt.figure(figsize=(W_IN, H_IN))
    cnt = Counter((f["key"][1], f["group"]) for f in Fs)
    L = [f"{title.upper()}  ({VERSION}, quality-controlled)", "", *wrap(VE.PURPOSE, 110), "",
         f"{len(Fs)} unique findings. Groups (CheXagent / MAIRA-2):", "",
         f"{'Condition':18s}" + "".join(f"{g:>6s}" for g in "ABCD") + "   Total"]
    for c in conds:
        L.append(f"{c:18s}" + "".join(f"{cnt[(c, g)]:6d}" for g in "ABCD") + f"   {sum(cnt[(c, g)] for g in 'ABCD'):5d}")
    L += ["", "A = both boxes: paired question - " + R.QA,
          "    answers include 'Cannot determine / abnormality not identifiable' (distinct from 'Neither useful').",
          "B = MAIRA-2 only: usefulness question - " + R.QB + " (not a model preference).",
          "C = CheXagent box, MAIRA-2 no box; D = neither: documented, no question.", "",
          "AMBIGUOUS CheXagent panel: the original form image shows boxes for several findings and the box for this",
          "finding cannot be isolated from the records; the paired comparison is flagged as potentially ambiguous.", "",
          "Reference status: rule-based sentence audit - EXPLICITLY PRESENT / EXPLICITLY ABSENT / POSSIBLE-UNCERTAIN /",
          "NOT MENTIONED (absence never inferred from silence); the exact reference sentence is quoted.", "",
          "Display convention: patient's right on the image left (markers under every image). All boxes and masks are",
          "model-generated candidate locations, not confirmed disease boundaries. Previous radiologist scores (below",
          "each question) rate the original report and CheXagent box, not this comparison.",
          "MIMIC-CXR content (PhysioNet DUA): do not distribute."]
    fig.text(0.04, 0.95, "\n".join(L), fontsize=8.6, va="top", family="monospace")
    figs.append((fig, []))
    for f in Fs:
        pno += 1
        es, e0, cond = f["es"], f["e0"], f["key"][1]
        ims = {}
        for w_ in ("prior", "current"):
            im = Image.open(e0[w_]).convert("L")
            im.thumbnail((1024, 1024))
            ims[w_] = np.array(im)
        cur = ims["current"]
        checks = []
        fig = plt.figure(figsize=(W_IN, H_IN))
        name = e0["name"].split(" (")[0] if e0["row_type"] == "MedGemma stated" else cond.capitalize()
        fig.text(0.012, 0.988, f"{name} - {e0['row_type']}   [{R.GROUP_TEXT[f['group']]}]", fontsize=11.5,
                 weight="bold", va="top")
        fig.text(0.012, 0.962, f"{e0['dataset']}; " + "; ".join(sorted({f"{FC.FORM_NAME[e['form']]} "
                 f"{e['form_case']}" for e in es})), fontsize=7, va="top")
        amb = ambiguity.get(id(f))
        panels = [("Prior (original)", ims["prior"], None), ("Current (original, no annotations)", cur, None)]
        bi = f["bi"]
        if bi and bi[0] == "boxes":
            panels.append(("CheXagent - original box for this finding (model-generated)", cur, ("box", bi[1], "red")))
        elif bi and bi[0] == "embedded":
            emb = np.array(bi[1])
            ar_e, ar_c = emb.shape[1] / emb.shape[0], cur.shape[1] / cur.shape[0]
            checks.append(f"embedded aspect {ar_e:.3f} vs current {ar_c:.3f}: "
                          + ("OK" if abs(ar_e / ar_c - 1) < 0.02 else "MISMATCH"))
            panels.append((("CheXagent - ALL ORIGINAL BOXES on the form image (" + str(amb) + " findings with boxes; "
                            "this finding's box cannot be isolated)") if amb and amb > 1 else
                           "CheXagent - original form image (only this finding had a box on that page)", emb, "rgb"))
        else:
            panels.append(("CheXagent - NO BOX for this finding" + (" (form image shows other findings' boxes only)"
                                                                  if bi else ""), cur, ("box", [], "red")))
        m = f["maira"] or {}
        panels.append((("MAIRA-2 - box (model-generated)" if f["mx"] else
                        "MAIRA-2 - NO BOX returned" + (" (unparsable output)" if m.get("parse_error") else "")
                        if m else "MAIRA-2 - not run"), cur, ("box", f["mx"], "yellow")))
        if focal:
            v = vas.get(id(f))
            cap = ("EXPLORATORY vasandesh segmentation (reproduced from the previous rendering; no re-inference); "
                   f"p = {v['p']:.2f}" + ("" if v["shown"] else " (< 0.5: mask not shown)" if v["region"] else
                                          " (no region)") + ("; " + "; ".join(v["warn"]) if v["warn"] else "")) \
                if v and v["p"] is not None else "EXPLORATORY vasandesh: stored result not found"
            panels.append((cap, v["img"] if v and v["shown"] else cur, "rgb" if v and v["shown"] else None))
        n = len(panels)
        pw = 0.976 / n
        tw = int((pw - 0.01) * W_IN * 72 / (MIN_FONT * 0.6))   # conservative characters per caption line
        for j, (cap, arr, spec) in enumerate(panels):
            ax = fig.add_axes([0.012 + j * pw, 0.585, pw - 0.008, 0.28])
            ax.imshow(arr, cmap=None if spec == "rgb" else "gray")
            ax.axis("off")
            if spec and spec != "rgb" and spec[0] == "box":
                assert arr is cur          # every box is drawn on the identical current-image array
                for b in spec[1]:
                    x1, y1 = b[0] / 100 * cur.shape[1], b[1] / 100 * cur.shape[0]
                    ax.add_patch(Rectangle((x1, y1), (b[2] - b[0]) / 100 * cur.shape[1],
                                           (b[3] - b[1]) / 100 * cur.shape[0], fill=False, ec=spec[2], lw=1.4))
            red = any(k in cap for k in ("NO BOX", "not run", "INSTAB", "ALL ORIGINAL BOXES"))
            # caption anchored to the panel's own column (not to the aspect-shrunk image), above the image area
            fig.text(0.012 + j * pw, 0.872, "\n".join(wrap(cap, tw)), fontsize=MIN_FONT, va="bottom",
                     color="#b42318" if red else "black")
            fig.text(0.012 + j * pw, 0.578, "R (patient right)", fontsize=6, va="top", color="#555555")
            fig.text(0.012 + (j + 1) * pw - 0.008, 0.578, "patient left (L)", fontsize=6, va="top", ha="right",
                     color="#555555")
        # facts + reference audit
        ra = refaudit[id(f)]
        v_ = VE.F.VERD.get(e0["status"], (e0["chexagent_shown"], "black"))[0]
        facts = [f"MedGemma: {e0['name'] if e0['row_type'] == 'MedGemma stated' else 'Not detected'}; interval "
                 f"change: {e0['change'] or '-'}",
                 f"CheXagent presence verdict (unchanged): {e0['chexagent_shown']}"
                 + (f" (p_present {e0['p_present']:.2f})" if e0.get("p_present") else ""),
                 f"Localization: CheXagent {('box, ' + R.side_txt(f['cx_side'])) if f['cx_avail'] else 'no box'}"
                 + (" (coordinates not recorded)" if f["cx_avail"] and not f["cx_coords"] else "")
                 + f"; MAIRA-2 {('box, ' + R.side_txt(f['mx_side'])) if f['mx'] else 'no box' if m else 'not run'}"
                 + (f"; sides: {f['side_agree']}" if f["side_agree"] else "")
                 + (f"; box overlap IoU {f['iou']:.2f}" if f["iou"] is not None else ""),
                 f"Reference report (sentence audit): {ra['label']}"]
        lines = sum((wrap(s, 118) for s in facts), [])
        ev = ra["sentences"]
        shown_ev = ev[:2]
        for lab, s in shown_ev:
            lines += wrap(f'  [{lab}] "{s}"', 118)
        if len(ev) > 2:
            lines.append(f"  (+{len(ev) - 2} more reference sentences: appendix)")
        if ra.get("related"):
            lines += wrap(f'  related description (not classified): "{ra["related"][0]}"', 118)
        fig.text(0.012, 0.548, "\n".join(lines[:11]), fontsize=6.5, va="top",
                 color="#b42318" if f["side_agree"] == "OPPOSITE sides" else "black")
        # question
        frects = []
        fig.text(0.012, 0.415, "Visualization question", fontsize=9, weight="bold", va="top")
        fid = f"{'F' if focal else 'R'}{pno:03d}"
        if f["group"] in ("A", "B"):
            q, opts = (R.QA, QA_OPT) if f["group"] == "A" else (R.QB, QB_OPT)
            fig.text(0.012, 0.393, "\n".join(wrap(q, 62)), fontsize=7.6, va="top")
            frects.append((f"{fid}_answer", (0.012, 0.33, 0.3, 0.03)))
            if f["group"] == "A" and amb and amb > 1:
                fig.text(0.012, 0.322, "\n".join(wrap("Potentially ambiguous comparison: the CheXagent panel shows "
                                                      "all original boxes of that form page.", 70)),
                         fontsize=6.4, va="top", color="#b42318")
            fig.text(0.33, 0.393, "Optional comment", fontsize=7, va="top")
            frects.append((f"{fid}_comment", (0.33, 0.295, 0.2, 0.085)))
            resp.append({"schema_version": SCHEMA, "report_version": VERSION, "field_id": f"{fid}_answer", "comment_field": f"{fid}_comment",
                         "page": pno, "group": f["group"], "condition": cond, "row_type": e0["row_type"],
                         "dataset": e0["dataset"],
                         "forms": "; ".join(f"{R.DOC[e['doctor']]} {FC.FORM_NAME[e['form']]} {e['form_case']}"
                                            for e in es),
                         "chexagent_panel": ("all original boxes (ambiguous)" if amb and amb > 1 else
                                             "form image, single finding" if bi and bi[0] == "embedded" else
                                             "recorded box" if bi and bi[0] == "boxes" else "no box"),
                         "reference_category": ra["category"], "question": q, "options": " | ".join(opts[1:]),
                         "answer_dr_franco": "", "comment_dr_franco": "", "answer_dr_siddharth": "",
                         "comment_dr_siddharth": ""})
        else:
            fig.text(0.012, 0.393, "\n".join(wrap("No question: " + ("MAIRA-2 returned no box, so there is no pair to "
                                                                     "compare." if f["group"] == "C" else
                                                                     "neither model provided a localization."), 70)),
                     fontsize=7.6, va="top", color="#555555")
        for nm, (x, y, w, h) in frects:
            fd = {"page": pno, "kind": "choice" if nm.endswith("answer") else "text", "name": nm,
                  "tooltip": f"{name}: {q if nm.endswith('answer') else 'optional comment'}", "rect": mm(x, y, w, h)}
            if nm.endswith("answer"):
                fd["options"] = opts
            fields.append(fd)
        # reports (continued in the appendix when long)
        mg, ref = texts["rep"].get(e0["key"], texts["rep"].get(Path(e0["current"]).name, ("-", "-")))
        for k, (lab, txt, ytop) in enumerate((("MedGemma report", mg, 0.548), ("Reference report", ref, 0.415))):
            ls = wrap(txt, 92)
            more = len(ls) > 7
            fig.text(0.56, ytop, lab, fontsize=7.2, weight="bold", va="top")
            fig.text(0.56, ytop - 0.017, "\n".join(ls[:7]) + ("\n(continued in the appendix)" if more else ""),
                     fontsize=MIN_FONT, va="top")
            if more:
                appendix.append((pno, name, lab, txt))
        # previous evaluation
        fig.add_artist(plt.Line2D([0.012, 0.988], [0.27, 0.27], color="#999999", lw=0.6))
        fig.text(0.012, 0.262, "Previous radiologist evaluation (answer the question above first). Original scores 1-5; "
                 "they rate the original report and CheXagent box, not this comparison. '-' not asked.",
                 fontsize=7.4, weight="bold", va="top")
        dup_f = {e["form"] for e in es if e["doctor"] == "Dr Franco"} >= {"A", "MIMIC-19"}
        for i, (kk, lab) in enumerate(VE.DIMS):
            fig.text(0.012, 0.19 - i * 0.022, lab, fontsize=6.8, va="top")
        for j, e in enumerate(es[:5]):
            x = 0.17 + j * 0.095
            pref = " (pref.)" if dup_f and e["doctor"] == "Dr Franco" and e["form"] == "A" else ""
            fig.text(x, 0.238, f"{R.DOC[e['doctor']]}\n{FC.FORM_NAME[e['form']]}\n{e['form_case']}{pref}",
                     fontsize=MIN_FONT, weight="bold", va="top")
            allblank = all(e["scores"][kk] is None for kk in e["asked"] if kk in e["scores"])
            for i, (kk, lab) in enumerate(VE.DIMS):
                val = "-" if kk not in e["asked"] else "Not evaluated" if allblank else \
                    str(e["scores"][kk]) if e["scores"][kk] is not None else "blank (N/A)"
                fig.text(x, 0.19 - i * 0.022, val, fontsize=6.8, va="top", weight="bold" if val.isdigit() else "normal")
        cl = [f'{R.DOC[e["doctor"]]} ({FC.FORM_NAME[e["form"]]}): "{e["comment"]}"' for e in es if e["comment"]]
        ctx = list(dict.fromkeys(f'{R.DOC[e["doctor"]]} ({FC.FORM_NAME[e["form"]]}, row "{lab}"): "{com}"'
                                 for e in es for fp, lab, com in page_comments.get((e["pdf"].name, e["page"]), [])
                                 if fp != e["field_prefix"]))
        fig.text(0.665, 0.238, "Comments on this finding", fontsize=7, weight="bold", va="top")
        fig.text(0.665, 0.221, "\n".join(sum((wrap(c, 58) for c in cl), [])[:5]) or "none", fontsize=MIN_FONT,
                 va="top")
        fig.text(0.665, 0.135, "Other comments on the same case", fontsize=7, weight="bold", va="top")
        fig.text(0.665, 0.118, "\n".join(sum((wrap(c, 58) for c in ctx[:3]), [])[:5]) or "none", fontsize=MIN_FONT,
                 va="top", color="#444444")
        dis = VE.disagreements(es)
        if dis:
            fig.text(0.012, 0.088, "\n".join(wrap("Doctors differ substantially (>= 2 points): " + "; ".join(
                ", ".join(f"{lab} {a_} vs {b_}" for lab, a_, b_ in d) for a, b, d in dis)
                + ". Original ratings unchanged.", 110)[:2]), fontsize=MIN_FONT, va="top", color="#7a4b00")
        fig.text(0.012, 0.034, VE.PURPOSE, fontsize=6)
        fig.text(0.012, 0.018, "Model-generated candidate locations (red CheXagent, yellow MAIRA-2"
                 + (", orange exploratory segmentation" if focal else "") + "), not confirmed disease boundaries. "
                 "Patient's right on the image left.", fontsize=6)
        issues = layout_issues(fig, [(nm, r_) for nm, r_ in frects])
        qc[pno] = {"finding": name, "group": f["group"], "layout": issues, "orientation": checks}
        figs.append((fig, issues))
    # appendix
    for k in range(0, len(appendix), 2):
        fig = plt.figure(figsize=(W_IN, H_IN))
        fig.text(0.03, 0.97, "APPENDIX - full report text (continued from the finding pages)", fontsize=10,
                 weight="bold", va="top")
        for j, (p, nm, lab, txt) in enumerate(appendix[k:k + 2]):
            fig.text(0.03, 0.92 - j * 0.45, f"Page {p} - {nm} - {lab}", fontsize=8, weight="bold", va="top")
            fig.text(0.03, 0.9 - j * 0.45, "\n".join(wrap(txt, 150)[:30]), fontsize=7, va="top")
        figs.append((fig, layout_issues(fig, [])))
    with PdfPages(out) as pdf:
        for fig, _ in figs:
            pdf.savefig(fig)
            plt.close(fig)
    VE.add_fields(out, fields)
    return Fs, fields, resp, qc


def check_fields(path, fields):
    from pypdf import PdfReader, PdfWriter
    r = PdfReader(str(path))
    got = r.get_fields() or {}
    problems = []
    names = [f["name"] for f in fields]
    if len(set(names)) != len(names):
        problems.append("duplicate field names")
    for f in fields:
        g = got.get(f["name"])
        if g is None:
            problems.append(f"missing field {f['name']}")
            continue
        if f["kind"] == "choice":
            opts = [str(o) for o in g.get("/Opt", [])]
            if opts != f["options"]:
                problems.append(f"options differ for {f['name']}")
    for i, pg in enumerate(r.pages):
        mb = pg.mediabox
        for a in pg.get("/Annots") or []:
            o = a.get_object()
            if o.get("/Subtype") == "/Widget":
                x0, y0, x1, y1 = [float(v) for v in o["/Rect"]]
                if x0 < 0 or y0 < 0 or x1 > float(mb.width) or y1 > float(mb.height):
                    problems.append(f"field outside page {i + 1}")
    tmp = Path(tempfile.mkdtemp()) / "filltest.pdf"
    w = PdfWriter(clone_from=str(path))
    test = {f["name"]: (f["options"][-1] if f["kind"] == "choice" else "test comment") for f in fields[:4]}
    try:
        for pg in w.pages:
            w.update_page_form_field_values(pg, test, auto_regenerate=False)
    except Exception as ex:
        problems.append(f"fill test error: {ex!r}")
    w.write(str(tmp))
    back = {k: str(v.get("/V")) for k, v in (PdfReader(str(tmp)).get_fields() or {}).items()}
    fill_ok = all(back.get(k) == v for k, v in test.items())
    shutil.rmtree(tmp.parent, ignore_errors=True)
    return {"fields": len(fields), "found": sum(1 for f in fields if f["name"] in got), "problems": problems,
            "fill_test": "passed" if fill_ok else "FAILED"}


def main(only=None):
    F, page_comments, n_wb, n_entries, mc = R.build()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    texts = {"rep": CG.report_texts()}
    refs = VE.reference_texts()
    fpb = form_page_boxes()
    pages_v1 = v1_page_numbers(F)
    ambiguity, refaudit = {}, {}
    for f in F:
        e0 = f["e0"]
        if f["bi"] and f["bi"][0] == "embedded":
            ambiguity[id(f)] = fpb.get((e0["pdf"].name, e0["page"]), 0)
        if e0["dataset"].startswith("NIH"):
            present = e0["reference"] == "Yes"
            refaudit[id(f)] = {"category": "NIH label: present" if present else "NIH label: not assigned",
                               "label": "NIH label present (text-mined labels; no report text)" if present else
                               "NIH label not assigned (text-mined labels; absence per labels only)",
                               "sentences": [], "related": []}
        else:
            res = RA.classify(f["key"][1], refs.get(e0["key"], ""))
            refaudit[id(f)] = {"category": res["category"], "label": RA.describe(res), "sentences": res["evidence"],
                               "related": res["related"]}
    vas = vasandesh_from_v1(F, pages_v1)
    results = {}
    for conds, slug, title, focal in ((R.REGIONAL, "regional", "Regional findings - visualization evaluation", False),
                                      (R.FOCAL, "focal", "Focal findings (pneumothorax) - visualization evaluation",
                                       True)):
        if only and slug not in only:
            continue
        out = LOGS / f"{slug}_findings_evaluation_{VERSION}_{stamp}.pdf"
        Fs, fields, resp, qc = render(F, conds, slug, title, out, page_comments, {"rep": texts["rep"]}, ambiguity,
                                      vas, refaudit, focal)
        fc = check_fields(out, fields)
        for ext, writer in (("csv", None), ("json", None)):
            pass
        rp = out.with_name(out.stem + "_responses.csv")
        with rp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(resp[0]))
            w.writeheader()
            w.writerows(resp)
        out.with_name(out.stem + "_responses.json").write_text(json.dumps(
            {"schema_version": SCHEMA, "report_version": VERSION, "reference_rules": RA.RULES_VERSION if hasattr(RA, "RULES_VERSION") else "v2", "paired_options": QA_OPT[1:], "usefulness_options": QB_OPT[1:],
             "questions": resp}, indent=1), encoding="utf-8")
        results[slug] = dict(out=out, Fs=Fs, fields=fc, qc=qc, n_q=len(resp))
        print(slug, out.name, "fields", fc, "pages with layout issues:",
              {p: v["layout"] for p, v in qc.items() if v["layout"]}, flush=True)
    # QC data for the report
    rows = []
    for f in F:
        slug, p1 = pages_v1[id(f)]
        if slug not in results:
            continue
        e0 = f["e0"]
        old = VE.reference_status(e0, refs) if not e0["dataset"].startswith("NIH") else \
            ("Present in reference (NIH label)" if e0["reference"] == "Yes" else "Absent per NIH labels")
        new = refaudit[id(f)]
        rows.append({"report": slug, "v1_page": p1, "condition": f["key"][1], "row_type": f["key"][2],
                     "dataset": e0["dataset"], "group": f["group"],
                     "forms": "; ".join(f"{R.DOC[e['doctor']]} {FC.FORM_NAME[e['form']]} {e['form_case']}"
                                        for e in f["es"]),
                     "v1_reference_status": old, "v2_reference_category": new["category"],
                     "v2_reference_label": new["label"],
                     "v2_reference_sentences": " || ".join(f"[{lab}] {s}" for lab, s in new["sentences"]),
                     "chexagent_panel": ("ALL ORIGINAL BOXES - ambiguous" if ambiguity.get(id(f), 0) > 1 else
                                         "form image - single finding" if id(f) in ambiguity else
                                         "recorded box" if f["cx_coords"] else "no box"),
                     "boxed_findings_on_form_page": ambiguity.get(id(f), "")})
    qp = LOGS / f"regional_focal_qc_findings_{VERSION}_{stamp}.csv"
    with qp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({k: {"pdf": v["out"].name, "fields": v["fields"], "questions": v["n_q"],
                   "qc": {str(p): q for p, q in v["qc"].items()}} for k, v in results.items()},
              open(LOGS / f"regional_focal_qc_pages_{VERSION}_{stamp}.json", "w", encoding="utf-8"), indent=1)
    print("findings QC ->", qp)


if __name__ == "__main__":
    # usage: regional_focal_reports_v2.py [regional|focal|all] [version]
    if len(sys.argv) > 2:
        VERSION = sys.argv[2]
    main(None if len(sys.argv) < 2 or sys.argv[1] == "all" else {sys.argv[1]})
