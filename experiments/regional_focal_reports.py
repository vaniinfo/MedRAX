"""Regional (pleural effusion, atelectasis, consolidation) and Focal (pneumothorax) visualization evaluation reports.

Visualization report only: no model output, score, report or Evidence Engine file is modified; new files only
(MIMIC-CXR content in logs/).

Per finding (unique current image + condition + row type, all returned forms):
  CheXagent: the original box as shown on the form (recorded per-finding coordinates, or - NIH and Form C - only the
             box image embedded in the returned PDF; Form C did not record coordinates or locations).
  MAIRA-2:   the versioned cache logs/maira2_grounding_v2_*.json (report-blinded, condition-name prompt).
  Group A both boxes -> paired preference question; B MAIRA-2 only -> usefulness question (not a model preference);
  C CheXagent only and D neither -> documented, no question. Missing outputs are shown, never treated as negatives.
  Focal: vasandesh pneumothorax segmentation as an exploratory third panel (mask shown only when its own probability
  >= 0.5; probability and mirror-stability warnings always shown); not ground truth, not a validated improvement.
Historical scores (Dr Franco, Dr Siddharth) restored from the returned PDFs (focused_conditions_report.restore,
cross-checked with the workbook); precedence: Dr Franco Form A over the 19-case form (marked 'pref.'), Dr Siddharth
both forms shown (no established rule).

    .venv/Scripts/python.exe experiments/regional_focal_reports.py
"""

import csv
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
import condition_grounding_models as CG  # noqa: E402
import focused_conditions_report as FC  # noqa: E402
import visualization_evaluation_report as VE  # noqa: E402

LOGS = ROOT / "logs"
REGIONAL = ("pleural effusion", "atelectasis", "consolidation")
FOCAL = ("pneumothorax",)
DIMS = VE.DIMS
VE.NEG.update({  # rule-based explicit negation cues for these conditions (the matched sentence is always quoted)
    "pleural effusion": [r"\bno (\w+ ){0,4}(pleural )?effusions?", r"without (\w+ ){0,3}(pleural )?effusions?",
                         r"(pleural )?effusions? (has |have )?resolved"],
    "atelectasis": [r"\bno (\w+ ){0,4}atelectasis", r"without (\w+ ){0,3}atelectasis", r"atelectasis (has )?resolved"],
    "consolidation": [r"\bno (\w+ ){0,4}consolidations?", r"without (\w+ ){0,3}consolidation"],
    "pneumothorax": [r"\bno (\w+ ){0,4}pneumothora(x|ces)", r"without (\w+ ){0,3}pneumothorax",
                     r"pneumothorax (has )?resolved"]})
DOC = {"Dr Franco": "Dr. Franco", "Dr Siddarth": "Dr. Siddharth"}
QA = "Which visualization better represents the suspected abnormality's location?"
QA_OPT = ["", "CheXagent box better", "MAIRA-2 box better", "Equally useful", "Neither useful"]
QB = "Is the MAIRA-2 localization useful for assessing this finding?"
QB_OPT = ["", "Useful", "Partly useful", "Not useful", "Cannot judge"]
GROUP_TEXT = {"A": "Group A - CheXagent box and MAIRA-2 box", "B": "Group B - no CheXagent box; MAIRA-2 box",
              "C": "Group C - CheXagent box; MAIRA-2 returned no box", "D": "Group D - neither model localized"}


def side_txt(s):
    return {"right": "patient's right", "left": "patient's left", "both": "both sides"}.get(s, "-")


def build():
    import doctor_scores_analysis_tabs as T
    from mimic19_grounding_compare import phrase
    FC.CONDS = REGIONAL + FOCAL
    entries, page_comments, v26, book, n_wb = FC.restore()
    X = T.extras()
    mc = json.loads(sorted(LOGS.glob("maira2_grounding_v2_*.json"))[-1].read_text(encoding="utf-8"))
    items = {}
    for e in entries:
        items.setdefault((e["key"], e["condition"], e["row_type"]), []).append(e)
    readers, F = {}, []
    for k, es in items.items():
        es = sorted(es, key=lambda e: (e["doctor"], e["form"] == "MIMIC-19", e["form"]))
        e0 = es[0]
        bi = FC.box_image(e0, v26, readers)
        cx_avail = bool(bi) and bi[0] in ("boxes", "embedded")
        cx_coords = bi[1] if bi and bi[0] == "boxes" else None
        loc = None
        for e in es:
            x = X.get((e["doctor"], e["form"], int(e["form_case"].split()[1]), e["condition"], e["row_type"]), {})
            loc = loc or x.get("box_location")
        cx_side = (CG.side(CG.box_mask(cx_coords, (1000, 1000))) if cx_coords else
                   T.box_side(loc) if loc else None)
        m = mc["results"].get(f"{e0['current']}|{phrase(e0['finding'])}")
        mx = (m or {}).get("boxes") or []
        mx_side = CG.side(CG.box_mask(mx, (1000, 1000))) if mx else None
        iou = None
        if cx_coords and mx:
            a, b = CG.box_mask(cx_coords, (1000, 1000)), CG.box_mask(mx, (1000, 1000))
            iou = float((a & b).sum() / (a | b).sum())
        grp = "A" if cx_avail and mx else "B" if mx else "C" if cx_avail else "D"
        agree = None if not (cx_side and mx_side) else ("same side" if cx_side == mx_side else
                                                         "partly different" if "both" in (cx_side, mx_side) else
                                                         "OPPOSITE sides")
        F.append(dict(key=k, es=es, e0=e0, bi=bi, cx_avail=cx_avail, cx_coords=cx_coords, cx_loc=loc,
                      cx_side=cx_side, maira=m, mx=mx, mx_side=mx_side, iou=iou, group=grp, side_agree=agree))
    return F, page_comments, n_wb, len(entries), mc


def coverage(F, path):
    rows = []
    for f in F:
        e0, m = f["e0"], f["maira"] or {}
        rows.append({
            "category": "Focal" if f["key"][1] in FOCAL else "Regional", "condition": f["key"][1],
            "row_type": f["key"][2], "dataset": e0["dataset"],
            "forms": "; ".join(f"{DOC[e['doctor']]} {FC.FORM_NAME[e['form']]} {e['form_case']}" for e in f["es"]),
            "chexagent_box": "yes" if f["cx_avail"] else "no",
            "chexagent_coordinates": json.dumps(f["cx_coords"]) if f["cx_coords"] else (
                "only in the form image (not recorded)" if f["cx_avail"] else "-"),
            "chexagent_location_text": f["cx_loc"] or "", "chexagent_side": f["cx_side"] or "",
            "maira_box": "yes" if f["mx"] else "no" if m else "not run",
            "maira_coordinates": json.dumps(f["mx"]) if f["mx"] else "", "maira_side": f["mx_side"] or "",
            "maira_parse_error": m.get("parse_error", ""), "maira_source": m.get("source", ""),
            "maira_raw": (m.get("raw") or "")[:300], "side_agreement": f["side_agree"] or "not assessable",
            "box_iou": "" if f["iou"] is None else round(f["iou"], 3),
            "failure": "; ".join(x for x in ("CheXagent: no box" if not f["cx_avail"] else "",
                                             "MAIRA-2: no box" if not f["mx"] else "",
                                             "MAIRA-2: unparsable output" if m.get("parse_error") else "") if x),
            "group": f["group"]})
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def render(F, conds, title, out, page_comments, texts, focal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    import anatomical_measurements as A
    Fs = sorted([f for f in F if f["key"][1] in conds], key=lambda f: ("ABCD".index(f["group"]),
                                                                      conds.index(f["key"][1]), f["e0"]["dataset"],
                                                                      f["key"][0]))
    W, H = 11.7, 8.3
    mm = lambda fx, fy, fw, fh: (fx * W * 25.4, (1 - fy - fh) * H * 25.4, fw * W * 25.4, fh * H * 25.4)  # noqa: E731
    wrap = lambda s, n: "\n".join(textwrap.wrap(" ".join(str(s).split()), n))  # noqa: E731
    fields, resp, pno = [], [], 1
    from collections import Counter
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(W, H))
        cnt = Counter((f["key"][1], f["group"]) for f in Fs)
        L = [title.upper(), "", wrap(VE.PURPOSE, 110), "",
             f"{len(Fs)} unique findings. Groups (CheXagent / MAIRA-2):", ""]
        L.append(f"{'Condition':18s}" + "".join(f"{g:>6s}" for g in "ABCD") + "   Total")
        for c in conds:
            L.append(f"{c:18s}" + "".join(f"{cnt[(c, g)]:6d}" for g in "ABCD") + f"   {sum(cnt[(c, g)] for g in 'ABCD'):5d}")
        L += ["", "A = both boxes: paired question - " + QA, "B = MAIRA-2 only: usefulness question - " + QB,
              "    (B is not a model preference: CheXagent provided no box.)",
              "C = CheXagent box, MAIRA-2 no box; D = neither model localized: documented, no question.", "",
              "All boxes and masks are model-generated candidate locations, not confirmed disease boundaries.",
              "Previous radiologist scores appear below the question; they rate the original report and CheXagent box,",
              "not the new comparison. Dr. Franco duplicates: Form A preferred over the 19-case form (marked pref.);",
              "Dr. Siddharth duplicates: both forms shown (no established rule).",
              "MAIRA-2: report-blinded, prompt = condition name only (microsoft/maira-2, pinned revision, 8-bit, greedy)."]
        if focal:
            L += ["", "Exploratory third panel: vasandesh SIIM-ACR pneumothorax segmentation. Mask shown only when its own",
                  "probability >= 0.5; probability and mirror-stability warnings always shown. Not ground truth and not",
                  "a validated improvement (feasibility study: tied with CheXagent on 15 VinDr images)."]
        L += ["", "MIMIC-CXR content (PhysioNet DUA): do not distribute."]
        fig.text(0.04, 0.95, "\n".join(L), fontsize=8.6, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        for f in Fs:
            pno += 1
            es, e0, cond = f["es"], f["e0"], f["key"][1]
            ims = {}
            for w_ in ("prior", "current"):
                im = Image.open(e0[w_]).convert("L")
                im.thumbnail((1024, 1024))
                ims[w_] = np.array(im)
            cur = ims["current"]
            fig = plt.figure(figsize=(W, H))
            name = e0["name"].split(" (")[0] if e0["row_type"] == "MedGemma stated" else cond.capitalize()
            fig.text(0.012, 0.988, f"{name} - {e0['row_type']}   [{GROUP_TEXT[f['group']]}]", fontsize=12,
                     weight="bold", va="top")
            fig.text(0.012, 0.962, f"{e0['dataset']}; " + "; ".join(sorted({f"{FC.FORM_NAME[e['form']]} "
                     f"{e['form_case']}" for e in es})), fontsize=7, va="top")
            panels = [("Prior (original)", ims["prior"], None), ("Current (original, no annotations)", cur, None)]
            bi = f["bi"]
            if bi and bi[0] == "boxes":
                panels.append(("CheXagent - original box (model-generated candidate location)", cur,
                               ("box", bi[1], "red")))
            elif bi and bi[0] == "embedded":
                panels.append((f"CheXagent - exact box image from the returned {FC.FORM_NAME[e0['form']]} (all "
                               "CheXagent boxes of that page; model-generated)", np.array(bi[1]), "rgb"))
            else:
                panels.append(("CheXagent - NO BOX for this finding" + (" (the form image shows other findings' "
                               "boxes only)" if bi else ""), cur, ("box", [], "red")))
            m = f["maira"] or {}
            panels.append((("MAIRA-2 - box (model-generated candidate location)" if f["mx"] else
                            "MAIRA-2 - NO BOX returned" + (" (unparsable output)" if m.get("parse_error") else "")
                            if m else "MAIRA-2 - not run"), cur, ("box", f["mx"], "yellow")))
            if focal:
                p, pc = CG.predict("ptx_vasandesh", cur)
                reg = CG.region_from_prob(p)
                pf, _ = CG.predict("ptx_vasandesh", cur, hflip=True)
                rf = CG.region_from_prob(pf)
                dice = float(2 * (reg & rf).sum() / (reg.sum() + rf.sum())) if reg.sum() + rf.sum() else 1.0
                warn = []
                if reg.any() and (dice < 0.5 or (CG.side(rf) and CG.side(rf) != CG.side(reg))):
                    warn.append(f"MIRROR INSTABILITY (flip Dice {dice:.2f}, side {CG.side(reg)} -> {CG.side(rf)})")
                if reg.any() and reg.mean() < 0.005:
                    warn.append("very small region")
                if reg.any() and CG.side(reg) == "both":
                    warn.append("scattered on both sides")
                show = pc >= 0.5 and reg.any()
                f["vas"] = dict(p=pc, shown=show, side=CG.side(reg) if show else None, warn=warn)
                panels.append((f"EXPLORATORY - vasandesh pneumothorax segmentation; p = {pc:.2f}"
                               + ("" if show else " (< 0.5: mask not shown)" if reg.any() else " (no region)")
                               + ("; " + "; ".join(warn) if warn else ""), cur, ("mask", reg if show else None)))
            pw = 0.976 / len(panels)
            for j, (cap, arr, spec) in enumerate(panels):
                ax = fig.add_axes([0.012 + j * pw, 0.56, pw - 0.006, 0.36])
                ax.imshow(arr, cmap=None if spec == "rgb" else "gray")
                ax.axis("off")
                if spec and spec != "rgb" and spec[0] == "box":
                    for b in spec[1]:
                        x1, y1 = b[0] / 100 * cur.shape[1], b[1] / 100 * cur.shape[0]
                        ax.add_patch(Rectangle((x1, y1), (b[2] - b[0]) / 100 * cur.shape[1],
                                               (b[3] - b[1]) / 100 * cur.shape[0], fill=False, ec=spec[2], lw=1.4))
                elif spec and spec != "rgb" and spec[0] == "mask" and spec[1] is not None:
                    ax.contour(spec[1].astype(float), levels=[0.5], colors="orange", linewidths=1.2)
                ax.set_title(wrap(cap, int(pw * 230)), fontsize=5.8, loc="left",
                             color="#b42318" if ("NO BOX" in cap or "not run" in cap or "INSTAB" in cap) else "black")
            v = VE.F.VERD.get(e0["status"], (e0["chexagent_shown"], "black"))[0]
            facts = [f"MedGemma: {e0['name'] if e0['row_type'] == 'MedGemma stated' else 'Not detected'}; interval "
                     f"change: {e0['change'] or '-'}",
                     f"CheXagent presence verdict (unchanged): {e0['chexagent_shown']}"
                     + (f" (p_present {e0['p_present']:.2f})" if e0.get("p_present") else ""),
                     "Reference: " + VE.reference_status(e0, texts["ref"]),
                     f"Localization: CheXagent {('box, ' + side_txt(f['cx_side'])) if f['cx_avail'] else 'no box'}"
                     + (" (side from its recorded location)" if f["cx_avail"] and not f["cx_coords"] and f["cx_side"]
                        else " (coordinates not recorded)" if f["cx_avail"] and not f["cx_coords"] else "")
                     + f"; MAIRA-2 {('box, ' + side_txt(f['mx_side'])) if f['mx'] else 'no box' if m else 'not run'}"
                     + (f"; sides: {f['side_agree']}" if f["side_agree"] else "")
                     + (f"; box overlap IoU {f['iou']:.2f}" if f["iou"] is not None else "")]
            fig.text(0.012, 0.535, "\n".join(sum((textwrap.wrap(s, 112) for s in facts), [])), fontsize=6.6,
                     va="top", color="#b42318" if f["side_agree"] == "OPPOSITE sides" else "black")
            # question
            fig.text(0.012, 0.44, "Visualization question", fontsize=9, weight="bold", va="top")
            fid = f"{'F' if focal else 'R'}{pno:03d}"
            if f["group"] in ("A", "B"):
                q, opts = (QA, QA_OPT) if f["group"] == "A" else (QB, QB_OPT)
                fig.text(0.012, 0.418, wrap(q, 55), fontsize=7.6, va="top")
                fields.append({"page": pno, "kind": "choice", "name": f"{fid}_answer", "options": opts,
                               "tooltip": f"{name}: {q}", "rect": mm(0.012, 0.33, 0.3, 0.03)})
                fig.text(0.33, 0.418, "Optional comment", fontsize=7, va="top")
                fields.append({"page": pno, "kind": "text", "name": f"{fid}_comment",
                               "tooltip": f"{name}: optional comment", "rect": mm(0.33, 0.31, 0.2, 0.095)})
                resp.append({"field_id": f"{fid}_answer", "comment_field": f"{fid}_comment", "page": pno,
                             "group": f["group"], "condition": cond, "row_type": e0["row_type"],
                             "dataset": e0["dataset"],
                             "forms": "; ".join(f"{DOC[e['doctor']]} {FC.FORM_NAME[e['form']]} {e['form_case']}"
                                                for e in es),
                             "question": q, "options": " | ".join(opts[1:]), "answer_dr_franco": "",
                             "comment_dr_franco": "", "answer_dr_siddharth": "", "comment_dr_siddharth": ""})
            else:
                fig.text(0.012, 0.418, wrap("No question: " + ("MAIRA-2 returned no box, so there is no pair to "
                                                                "compare." if f["group"] == "C" else
                                                                "neither CheXagent nor MAIRA-2 provided a "
                                                                "localization."), 80), fontsize=7.6, va="top",
                         color="#555555")
            mg, ref = texts["rep"].get(e0["key"], texts["rep"].get(Path(e0["current"]).name, ("-", "-")))
            fig.text(0.56, 0.535, "MedGemma report", fontsize=7, weight="bold", va="top")
            fig.text(0.56, 0.518, wrap(mg, 95)[:900], fontsize=5.2, va="top", family="monospace")
            fig.text(0.56, 0.4, "Reference report", fontsize=7, weight="bold", va="top")
            fig.text(0.56, 0.383, wrap(ref, 95)[:800], fontsize=5.2, va="top", family="monospace")
            # previous evaluation
            fig.add_artist(plt.Line2D([0.012, 0.988], [0.285, 0.285], color="#999999", lw=0.6))
            fig.text(0.012, 0.275, "Previous radiologist evaluation (answer the question above first). Original scores "
                     "1-5; they rate the original report and CheXagent box, not this comparison. '-' not asked.",
                     fontsize=7.5, weight="bold", va="top")
            dup_f = {e["form"] for e in es if e["doctor"] == "Dr Franco"} >= {"A", "MIMIC-19"}
            for i, (kk, lab) in enumerate(DIMS):
                fig.text(0.012, 0.2 - i * 0.022, lab, fontsize=6.8, va="top")
            for j, e in enumerate(es[:5]):
                x = 0.17 + j * 0.095
                pref = " (pref.)" if dup_f and e["doctor"] == "Dr Franco" and e["form"] == "A" else ""
                fig.text(x, 0.245, f"{DOC[e['doctor']]}\n{FC.FORM_NAME[e['form']]}\n{e['form_case']}{pref}",
                         fontsize=5.6, weight="bold", va="top")
                allblank = all(e["scores"][kk] is None for kk in e["asked"] if kk in e["scores"])
                for i, (kk, lab) in enumerate(DIMS):
                    val = "-" if kk not in e["asked"] else "Not evaluated" if allblank else \
                        str(e["scores"][kk]) if e["scores"][kk] is not None else "blank (N/A)"
                    fig.text(x, 0.2 - i * 0.022, val, fontsize=6.8, va="top",
                             weight="bold" if val.isdigit() else "normal")
            cl = [f'{DOC[e["doctor"]]} ({FC.FORM_NAME[e["form"]]}): "{e["comment"]}"' for e in es if e["comment"]]
            ctx = list(dict.fromkeys(f'{DOC[e["doctor"]]} ({FC.FORM_NAME[e["form"]]}, row "{lab}"): "{com}"'
                                     for e in es for fp, lab, com in page_comments.get((e["pdf"].name, e["page"]), [])
                                     if fp != e["field_prefix"]))
            fig.text(0.66, 0.245, "Comments on this finding", fontsize=7, weight="bold", va="top")
            fig.text(0.66, 0.228, "\n".join(sum((textwrap.wrap(c, 62) for c in cl), [])) or "none", fontsize=6,
                     va="top")
            fig.text(0.66, 0.14, "Other comments on the same case", fontsize=7, weight="bold", va="top")
            fig.text(0.66, 0.123, "\n".join(sum((textwrap.wrap(c, 62) for c in ctx[:4]), [])) or "none",
                     fontsize=5.7, va="top", color="#444444")
            dis = VE.disagreements(es)
            if dis:
                fig.text(0.012, 0.09, wrap("Doctors differ substantially (>= 2 points): " + "; ".join(
                    ", ".join(f"{lab} {a_} vs {b_}" for lab, a_, b_ in d) for a, b, d in dis)
                    + ". Original ratings unchanged.", 105), fontsize=6.2, va="top", color="#7a4b00")
            fig.text(0.012, 0.006, "Model-generated candidate locations (red CheXagent, yellow MAIRA-2"
                     + (", orange exploratory segmentation" if focal else "") + "), not confirmed disease boundaries. "
                     , fontsize=5.6)
            fig.text(0.012, 0.022, VE.PURPOSE, fontsize=5.6)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  p{pno} {cond} group {f['group']}", flush=True)
    VE.add_fields(out, fields)
    rp = out.with_name(out.stem + "_responses.csv")
    if resp:
        with rp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(resp[0]))
            w.writeheader()
            w.writerows(resp)
        out.with_name(out.stem + "_responses.json").write_text(json.dumps(resp, indent=1), encoding="utf-8")
    return len(Fs), len(fields) // 2


def main():
    F, page_comments, n_wb, n_entries, mc = build()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows = coverage(F, LOGS / f"regional_focal_coverage_{stamp}.csv")
    from collections import Counter
    texts = {"ref": VE.reference_texts(), "rep": CG.report_texts()}
    outs = {}
    for conds, title, slug, focal in ((REGIONAL, "Regional findings - visualization evaluation", "regional", False),
                                      (FOCAL, "Focal findings (pneumothorax) - visualization evaluation", "focal",
                                       True)):
        out = LOGS / f"{slug}_findings_evaluation_{stamp}.pdf"
        n, q = render(F, conds, title, out, page_comments, texts, focal)
        outs[slug] = (out, n, q)
    # audit
    audit = []
    for f in F:
        dup_f = {e["form"] for e in f["es"] if e["doctor"] == "Dr Franco"} >= {"A", "MIMIC-19"}
        for e in f["es"]:
            audit.append({"category": "Focal" if f["key"][1] in FOCAL else "Regional", "condition": f["key"][1],
                          "row_type": f["key"][2], "dataset": e["dataset"], "doctor": DOC[e["doctor"]],
                          "source_form": FC.FORM_NAME[e["form"]], "source_pdf": e["pdf"].name,
                          "form_case": e["form_case"], "field_prefix": e["field_prefix"],
                          **{f"score_{k}": e["scores"][k] for k, _ in DIMS}, "comment": e["comment"],
                          "precedence": ("preferred (Form A over 19-case form)" if dup_f and e["doctor"] == "Dr Franco"
                                         and e["form"] == "A" else "not preferred (Form A exists)" if dup_f and
                                         e["doctor"] == "Dr Franco" else "shown (no precedence rule)"
                                         if e["doctor"] == "Dr Siddarth" and len([x for x in f["es"] if x["doctor"] ==
                                                                                   "Dr Siddarth"]) > 1 else "only source"),
                          "workbook_check": e["workbook_check"]})
    ap = LOGS / f"regional_focal_score_audit_{stamp}.csv"
    with ap.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(audit[0]))
        w.writeheader()
        w.writerows(audit)
    cnt = Counter((r["condition"], r["group"]) for r in rows)
    S = [f"# Regional and focal findings: MAIRA-2 coverage and eligible comparisons ({stamp})", "",
         f"MAIRA-2 cache: `{sorted(LOGS.glob('maira2_grounding_v2_*.json'))[-1].name}` "
         f"({sum(1 for v in mc['results'].values() if v['source'] == 'reused')} reused, "
         f"{sum(1 for v in mc['results'].values() if v['source'] == 'new run')} new runs; revision "
         f"{mc['meta']['revision'][:8]}). {len(F)} unique findings from {n_entries} scored rows (workbook check: "
         f"{sum(a['workbook_check'] == 'match' for a in audit)}/{len(audit)} match).", "",
         "| Condition | A both | B MAIRA-2 only | C CheXagent only | D neither | Total |", "|---|---|---|---|---|---|"]
    for c in REGIONAL + FOCAL:
        S.append(f"| {c} | " + " | ".join(str(cnt[(c, g)]) for g in "ABCD") + f" | {sum(cnt[(c, g)] for g in 'ABCD')} |")
    S.append("| **All** | " + " | ".join(str(sum(1 for r in rows if r["group"] == g)) for g in "ABCD") +
             f" | {len(rows)} |")
    sa = Counter(r["side_agreement"] for r in rows if r["group"] == "A")
    S += ["", f"Group A side agreement: {dict(sa)}. Box IoU available (both coordinates recorded) for "
          f"{sum(1 for r in rows if r['box_iou'] != '')} findings; median "
          f"{np.median([r['box_iou'] for r in rows if r['box_iou'] != '']) if any(r['box_iou'] != '' for r in rows) else '-'}.",
          f"MAIRA-2 unparsable outputs: {sum(1 for r in rows if r['maira_parse_error'] is True)}.", ""]
    for slug, (out, n, q) in outs.items():
        S.append(f"- {slug}: `{out.name}` - {n} findings, {q} questions (responses: `{out.stem}_responses.csv/.json`)")
    S += [f"- Coverage (all findings, coordinates, sides, failures): `regional_focal_coverage_{stamp}.csv`",
          f"- Score audit: `{ap.name}`"]
    md = LOGS / f"regional_focal_summary_{stamp}.md"
    md.write_text("\n".join(S), encoding="utf-8")
    print("\n".join(S))


if __name__ == "__main__":
    main()
