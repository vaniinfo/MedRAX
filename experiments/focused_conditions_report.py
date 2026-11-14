"""Focused visualization report: cardiomegaly, low lung volumes, hyperinflation and pulmonary edema, across ALL
returned radiologist forms (Dr Franco: Form A, Form C, 19-case form; Dr Siddharth: Form B, 19-case form).

For every scored row of these four conditions (any row type: MedGemma stated, CheXagent only, missed by both) the
report shows the previous radiologist evaluation (scores and comments read directly from the returned PDFs; the
doctor's other comments on the same case page as context), the original CheXagent box (A) and the anatomy-aware
visualization (B), and asks one question where a box was shown on the original form.

Nothing is modified: scores, comments, model outputs, boxes, Evidence Engine results, reports and earlier PDFs.
New time-stamped files in logs/ (MIMIC-CXR content).

Visualization A
  19-case MIMIC set (also in Forms A/B, boxes verified identical): the recorded per-finding CheXagent boxes.
  NIH (Forms A/B) and Form C: the boxes were never saved, only drawn into the form, so A is the exact box image
  embedded in the returned PDF the doctor scored (all boxes of that page, not only this finding's).
  CheXagent-only and missed rows: no box was shown on the form.
Visualization B (PSPNet segmentation, never placed by a finding)
  cardiomegaly -> heart outline; CTR lines and value only on PA films passing automatic QC, labelled PROVISIONAL
  unless the heart outline was manually reviewed (logs/report_manual_qc_mimic19.json; only 19-case titles).
  low lung volumes, hyperinflation, pulmonary edema (diffuse, bilateral) -> lung outline; no lung measurement.
Question (only where a box was shown on the original form): "Compared with the original CheXagent box, does the
new visualization help you assess this finding?" + one optional comment.

    .venv/Scripts/python.exe experiments/focused_conditions_report.py
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
import anatomical_measurements as A  # noqa: E402
import condition_aware_grounding as G  # noqa: E402
import radiologist_report_final as F  # noqa: E402
import visualization_comparison_report as V  # noqa: E402

LOGS = ROOT / "logs"
NIH = Path("D:/datasets/chestxray14")
HELD = Path("D:/datasets/held_out_eval_images")
CONDS = ("cardiomegaly", "low lung volumes", "hyperinflation", "pulmonary edema")
ROWTYPE = {"Consider": "CheXagent only", "Missed": "Missed by both"}
FORM_NAME = {"A": "Form A", "B": "Form B", "C": "Form C", "MIMIC-19": "19-case form"}
DOCTOR = V.DOCTOR
DIMS = V.DIMS


def restore():
    import doctor_scores_to_excel as D
    from comparison_edv_pdf import validated_name
    from openpyxl import load_workbook
    from pypdf import PdfReader
    import pandas as pd
    v26 = {r["title"]: r for r in json.loads((LOGS / "radiologist_requested_plus_edv_scoring_form_v26.json")
                                             .read_text(encoding="utf-8"))}
    inputs2 = {p["current"]: p for p in json.loads((LOGS / "mimic_heldout_inputs_2.json").read_text(encoding="utf-8"))}
    subset = pd.read_csv(HELD / "subset.csv")
    nih = pd.read_csv(NIH / "Data_Entry_2017_v2020.csv").set_index("Image Index")["View Position"].to_dict()
    proj_csv = A.projections()
    QC = json.loads(F.QC_FILE.read_text(encoding="utf-8")) if F.QC_FILE.exists() else {}
    import pickle
    ph1 = pickle.load(open(LOGS / "mixed_scoring_phase1_v8.pkl", "rb"))
    vp_by_image = {pg["rec"]["image"]: pg["rec"]["visual_proof"] for f in ph1.values() for pg in f if "rec" in pg}
    entries, page_comments = [], {}
    for doctor, form, pdf, side_path, rec_path, source in D.FORMS:
        side = json.loads(side_path.read_text(encoding="utf-8"))
        loaded = json.loads(rec_path.read_text(encoding="utf-8"))
        rows = D.rows_mixed(source[-1]) if source.startswith("mixed_") else D.rows_nineteen() \
            if source == "nineteen" else D.rows_heldout(2)
        assert len(rows) == len(side)
        values = {k: str(v.get("/V") or "").strip() for k, v in (PdfReader(str(pdf)).get_fields() or {}).items()}
        by_case = {r["case"]: r for r in loaded} if source != "nineteen" else None
        for (title, item), s in zip(rows, side):
            get = lambda k: values.get(s["field_prefix"] + "_" + k, "")  # noqa: E731
            page = int(s["field_prefix"][1:3])
            if get("comment"):
                page_comments.setdefault((pdf.name, page), []).append(
                    (s["field_prefix"], s["finding"], get("comment")))
            if item is None:
                continue
            cond = validated_name(item["finding"]) or item["finding"]
            if cond not in CONDS:
                continue
            if source == "nineteen":
                rec = v26[title]
                cur = Path(rec["image"])
                img = dict(dataset="MIMIC-CXR (19-case set)", current=str(cur), prior=str(cur.with_name("prior.jpg")),
                           key=str(cur), title19=title, vp=rec["visual_proof"],
                           projection=QC.get(title, {}).get("projection_confirmed") or
                           proj_csv.get(cur.parent.name.split("_subject_")[1]))
            else:
                rec = by_case[int(title.split()[1])]
                if source == "heldout_2":
                    p = inputs2[rec["current"]]
                    img = dict(dataset="MIMIC-CXR held-out (Form C)", current=str(HELD / "images/images" /
                                                                                  p["current_png"]),
                               prior=str(HELD / "images/images" / p["prior_png"]), key=rec["current"],
                               projection=subset.view_position_after[p["row"]])
                elif "NIH" in rec["dataset"]:
                    img = dict(dataset="NIH ChestX-ray14", current=str(NIH / "images" / rec["current"]),
                               prior=str(NIH / "images" / rec["prior"]), key=rec["current"],
                               projection=nih.get(rec["current"]))
                else:
                    cur = Path(rec["image"])
                    t19 = rec.get("source_title")
                    is19 = str(t19).endswith("of 19")
                    img = dict(dataset="MIMIC-CXR (19-case set)" if is19 else "MIMIC-CXR (16-case matched set)",
                               current=str(cur), prior=str(cur.with_name("prior.jpg")), key=str(cur),
                               title19=t19 if is19 else None, vp=vp_by_image[str(cur)],
                               projection=QC.get(t19, {}).get("projection_confirmed") or
                               proj_csv.get(cur.parent.name.split("_subject_")[1]))
            sc = {k: (int(get(k)) if get(k).isdigit() else None) for k, _ in DIMS}
            ch = item.get("change")
            entries.append({
                "doctor": doctor, "form": form, "pdf": pdf, "page": page, "field_prefix": s["field_prefix"],
                "form_case": title.split("  (")[0], "condition": cond, "row_type": ROWTYPE.get(item["status"],
                                                                                              "MedGemma stated"),
                "finding": item["finding"], "name": item["name"], "status": item["status"],
                "p_present": item.get("p_present"), "asked": s["scored"], "scores": sc, "comment": get("comment"),
                "chexagent_shown": s["chexagent_evidence"],
                "change": "" if not ch else ("No change stated" if ch[0] in ("no change stated", "not mentioned")
                                             else f'{ch[0]}: "{ch[1]}"'),
                "reference": item.get("reference", item.get("ground_truth", "")),
                "reference_quote": item.get("reference_quote", item.get("gt_quote", "")), **img})
    # cross-check with the workbook
    book = sorted(Path("D:/datasets/doctor_response").glob("doctor_responses_*.xlsx"))[-1]
    ws = load_workbook(book, read_only=True)["All rows"]
    wr = list(ws.iter_rows(values_only=True))
    H = {h: i for i, h in enumerate(wr[0])}
    wb = {(r[H["Doctor"]], r[H["Form"]], r[H["Case"]], r[H["Condition"]], r[H["Row type"]]): r for r in wr[1:]}
    for e in entries:
        r = wb.get((e["doctor"], e["form"], int(e["form_case"].split()[1]), e["condition"], e["row_type"]))
        e["workbook_check"] = "NOT FOUND" if r is None else "match" if all(
            r[H[V.WB_COL[k]]] == e["scores"][k] for k, _ in DIMS) and (r[H["Doctor comment"]] or "") == \
            (e["comment"] or "") else "MISMATCH"
    n_wb = sum(1 for r in wr[1:] if r[H["Condition"]] in CONDS)
    return entries, page_comments, v26, book.name, n_wb


def box_image(e, v26, readers):
    """('boxes', [...]) | ('embedded', PIL image) | None: the original CheXagent box shown for this row."""
    if e["row_type"] != "MedGemma stated":
        return None
    if e.get("vp") is not None:
        vp = next((p for p in e["vp"] if p["finding"] == e["finding"] and p.get("boxes")), None)
        return ("boxes", [b["box_normalized"] for b in vp["boxes"]]) if vp else None
    from pypdf import PdfReader
    if e["pdf"] not in readers:
        readers[e["pdf"]] = PdfReader(str(e["pdf"]))
    ims = readers[e["pdf"]].pages[e["page"] - 1].images
    if len(ims) < 3:
        return None
    # the form asked for a grounding score only when CheXagent placed a box for this finding
    return ("embedded", ims[2].image) if "grounding" in e["asked"] else ("embedded_other", ims[2].image)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    entries, page_comments, v26, book, n_wb = restore()
    N = json.loads(sorted((HERE / "results").glob("anatomical_measurements_nih_*.json"))[-1].read_text(encoding="utf-8"))
    QC = json.loads(F.QC_FILE.read_text(encoding="utf-8")) if F.QC_FILE.exists() else {}
    items = {}
    for e in entries:
        items.setdefault((e["key"], e["condition"], e["row_type"]), []).append(e)
    order = sorted(items, key=lambda k: (CONDS.index(k[1]), items[k][0]["dataset"], k[2], k[0]))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = LOGS / f"focused_conditions_report_{stamp}.pdf"
    W_IN, H_IN = 11.7, 8.3
    mm = lambda fx, fy, fw, fh: (fx * W_IN * 25.4, (1 - fy - fh) * H_IN * 25.4, fw * W_IN * 25.4, fh * H_IN * 25.4)  # noqa: E731
    wrap = lambda s, n: "\n".join(textwrap.wrap(" ".join(str(s).split()), n))  # noqa: E731
    readers, fields, summary, audit, flags, pno = {}, [], [], [], [], 0
    seg_cache = {}
    with PdfPages(out) as pdf:
        # cover
        fig = plt.figure(figsize=(W_IN, H_IN))
        pno += 1
        cnt = {c: sum(1 for k in order if k[1] == c) for c in CONDS}
        cover = [f"FOCUSED VISUALIZATION REPORT - {', '.join(CONDS)}", "",
                 f"{len(entries)} scored rows from all returned forms ({n_wb} rows of these conditions in the "
                 f"workbook), grouped into {len(order)} findings: " + ", ".join(f"{c} {n}" for c, n in cnt.items()) + ".",
                 "Forms: Dr. Franco - Form A, Form C, 19-case form; Dr. Siddharth - Form B, 19-case form.", "",
                 "Each page: prior and current X-rays; A = the original CheXagent box (as shown on the form);",
                 "B = anatomy-aware visualization (segmentation outline; CTR only on PA films passing QC, provisional);",
                 "Previous radiologist evaluation = the original scores and comments, read directly from the returned "
                 "PDFs.", "They rate the original report and box A, not the new visualization. '-' = not asked for "
                 "that row type;", "'blank (N/A)' = left blank on the form; 'Not evaluated' = nothing scored.", "",
                 "Follow-up question only where a box was shown on the original form. Dr. Franco duplicates: Form A "
                 "preferred over the", "19-case form (established rule; both shown, preferred marked). Dr. Siddharth "
                 "duplicates: both shown (no rule).", "", "Pulmonary edema uses the lung outline (diffuse, bilateral "
                 "finding), not MAIRA-2.", "",
                 "MIMIC-CXR content (PhysioNet DUA): do not distribute."]
        fig.text(0.04, 0.95, "\n".join(cover), fontsize=9, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        for key in order:
            es = sorted(items[key], key=lambda e: (e["doctor"], e["form"] == "MIMIC-19", e["form"]))
            e0 = es[0]
            cond, rtype = key[1], key[2]
            fig = plt.figure(figsize=(W_IN, H_IN))
            pno += 1
            name = e0["name"].split(" (")[0] if rtype == "MedGemma stated" else cond.capitalize()
            fig.text(0.015, 0.985, f"{name}  -  {rtype}", fontsize=13, weight="bold", va="top")
            fig.text(0.015, 0.953, f"{e0['dataset']}; " + "; ".join(sorted({f"{FORM_NAME[e['form']]} "
                     f"{e['form_case']}" for e in es})) + f"; projection: {e0['projection'] or 'not recorded'}",
                     fontsize=7.5, va="top")
            # images
            if e0["current"] not in seg_cache:
                ims = {}
                for w in ("prior", "current"):
                    im = Image.open(e0[w]).convert("L")
                    im.thumbnail((1024, 1024))
                    ims[w] = np.array(im)
                seg_cache.clear()
                seg_cache[e0["current"]] = (ims, A.measure(A.segment_all(ims["current"]), N["rot_limit"]))
            ims, R = seg_cache[e0["current"]]
            for j, (w, lab) in enumerate((("prior", "Prior"), ("current", "Current"))):
                ax = fig.add_axes([0.015 + j * 0.16, 0.5, 0.155, 0.41])
                ax.imshow(ims[w], cmap="gray")
                ax.axis("off")
                ax.set_title(lab + " (original)", fontsize=7.5, loc="left")
            bi = box_image(e0, v26, readers)
            ax = fig.add_axes([0.345, 0.5, 0.2, 0.41])
            if bi is None:
                ax.imshow(ims["current"], cmap="gray")
                ax.axis("off")
                ax.set_title(wrap("A. No CheXagent box was shown for this row on the original form", 46),
                             fontsize=6.6, loc="left")
            elif bi[0] == "boxes":
                ax.imshow(ims["current"], cmap="gray")
                ax.axis("off")
                for b in bi[1]:
                    x1, y1, x2, y2 = G.box_px(b, ims["current"].shape)
                    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
                ax.set_title(wrap("A. Original CheXagent box for this finding (unchanged; model-generated, not "
                                  "disease extent)", 46), fontsize=6.6, loc="left")
            else:
                ax.imshow(bi[1])
                ax.axis("off")
                ax.set_title(wrap(f"A. Exact box image from the returned {FORM_NAME[e0['form']]} (all CheXagent "
                                  "boxes of that page; model-generated)" if bi[0] == "embedded" else
                                  f"A. Box image from the returned {FORM_NAME[e0['form']]}: CheXagent placed NO box "
                                  "for this finding (the boxes shown belong to other findings)", 46),
                             fontsize=6.6, loc="left")
            # B
            ax = fig.add_axes([0.555, 0.5, 0.2, 0.41])
            k = F.kind(cond)
            note = ""
            if k == "heart":
                q = R["qc"]
                hq = QC.get(e0.get("title19"), {}).get("heart_outline", "not reviewed")
                if q["lungs"] and q["heart"] and hq != "incorrect":
                    if e0["projection"] == "PA" and q["midline"]:
                        F.draw_vis(ax, ims["current"], R, ("ctr", []))
                        note = (f"CTR {R['val']['ctr']:.2f}" + (" - PROVISIONAL (heart outline not manually "
                                                               "reviewed)" if hq != "correct" else "")
                                + "; conventional PA reference 0.50.")
                        labb = "B. Heart outline + CTR lines (PSPNet segmentation)"
                    else:
                        F.draw_vis(ax, ims["current"], R, ("heart_outline", []))
                        note = ("AP film: no CTR (heart magnified)." if e0["projection"] == "AP" else
                                "Projection not recorded: no CTR." if e0["projection"] is None else
                                "Rotated or midline not identified: no CTR.")
                        labb = "B. Heart outline (PSPNet segmentation; anatomy, not reviewed)"
                else:
                    ax.imshow(ims["current"], cmap="gray")
                    ax.axis("off")
                    labb = "B. Heart outline not shown (segmentation failed QC or marked incorrect)"
            else:
                if R["qc"]["lungs"]:
                    F.draw_vis(ax, ims["current"], R, ("outline", []))
                    labb = "B. Lung outline (PSPNet segmentation; anatomy, orientation only)"
                    note = "No lung measurement shown (none validated)."
                else:
                    ax.imshow(ims["current"], cmap="gray")
                    ax.axis("off")
                    labb = "B. Lung outline not shown (segmentation failed QC: " + R["why"].get("lungs", "") + ")"
            ax.set_title(wrap(labb, 46), fontsize=6.6, loc="left")
            if note:
                fig.text(0.555, 0.485, wrap(note, 60), fontsize=6.4, va="top", color="#1f4e9c")
            # finding facts (from the first entry; identical across forms)
            v, col = F.VERD.get(e0["status"], (e0["chexagent_shown"], "black"))
            facts = [f"MedGemma finding: {e0['name'] if rtype == 'MedGemma stated' else 'Not detected'}",
                     f"MedGemma interval change: {e0['change'] or ('-' if rtype != 'MedGemma stated' else 'not stated')}",
                     f"CheXagent shown on the form: {e0['chexagent_shown']}" + (f" (p_present {e0['p_present']:.2f})"
                                                                               if e0.get("p_present") else ""),
                     f"Reference: {e0['reference'] or '-'}" + (f' - "{e0["reference_quote"]}"'
                                                              if e0["reference_quote"] else "")]
            fig.text(0.77, 0.91, "Finding", fontsize=8.5, weight="bold", va="top")
            fig.text(0.77, 0.89, "\n".join(sum((textwrap.wrap(s, 44) for s in facts), [])), fontsize=6.6, va="top")
            # previous evaluation table
            fig.text(0.015, 0.455, "Previous radiologist evaluation", fontsize=10, weight="bold", va="top")
            fig.text(0.015, 0.432, "Original scores (1-5) and comments from the returned forms; they rate the original "
                     "report and box A, not the new visualization.", fontsize=6.5, va="top", style="italic")
            dup_f = {e["form"] for e in es if e["doctor"] == "Dr Franco"} >= {"A", "MIMIC-19"}
            for j, e in enumerate(es[:5]):
                x = 0.2 + j * 0.105
                pref = " (preferred)" if dup_f and e["doctor"] == "Dr Franco" and e["form"] == "A" else \
                    " (not preferred)" if dup_f and e["doctor"] == "Dr Franco" and e["form"] == "MIMIC-19" else ""
                fig.text(x, 0.405, f"{DOCTOR[e['doctor']]}\n{FORM_NAME[e['form']]}\n{e['form_case']}{pref}",
                         fontsize=6, weight="bold", va="top")
                allblank = all(e["scores"][kk] is None for kk in e["asked"] if kk in e["scores"])
                for i, (kk, lab) in enumerate(DIMS):
                    val = "-" if kk not in e["asked"] else "Not evaluated" if allblank else \
                        str(e["scores"][kk]) if e["scores"][kk] is not None else "blank (N/A)"
                    fig.text(x, 0.345 - i * 0.026, val, fontsize=7.2, va="top",
                             weight="bold" if val.isdigit() else "normal")
            for i, (kk, lab) in enumerate(DIMS):
                fig.text(0.015, 0.345 - i * 0.026, lab, fontsize=7, va="top")
            cl = []
            for e in es:
                if e["comment"]:
                    cl.append(f'{DOCTOR[e["doctor"]]} ({FORM_NAME[e["form"]]}, this finding): "{e["comment"]}"')
            ctx = []
            for e in es:
                for fp, lab, com in page_comments.get((e["pdf"].name, e["page"]), []):
                    if fp != e["field_prefix"]:
                        ctx.append(f'{DOCTOR[e["doctor"]]} ({FORM_NAME[e["form"]]}, row "{lab}"): "{com}"')
            ctx = list(dict.fromkeys(ctx))
            fig.text(0.015, 0.23, "Doctor comments on this finding", fontsize=7.5, weight="bold", va="top")
            fig.text(0.015, 0.21, "\n".join(sum((textwrap.wrap(c, 95) for c in cl), [])) or "none", fontsize=6.5,
                     va="top")
            fig.text(0.015, 0.125, "Other comments by the doctor on the same case (context)", fontsize=7.5,
                     weight="bold", va="top")
            fig.text(0.015, 0.105, "\n".join(sum((textwrap.wrap(c, 95) for c in ctx[:5]), [])) or "none",
                     fontsize=6.2, va="top", color="#444444")
            # question
            ask = bi is not None and bi[0] != "embedded_other"
            if ask:
                fig.text(0.58, 0.3, "Follow-up question", fontsize=9, weight="bold", va="top")
                fig.text(0.58, 0.278, wrap(V.QUESTION, 72), fontsize=7.6, va="top")
                fig.text(0.58, 0.18, "Optional comment", fontsize=7, va="top")
                fid = f"q{pno:03d}"
                fields.append({"page": pno, "kind": "choice", "name": f"{fid}_answer",
                               "tooltip": f"{name} ({e0['dataset']}) - {V.QUESTION}", "rect": mm(0.58, 0.2, 0.4, 0.032)})
                fields.append({"page": pno, "kind": "text", "name": f"{fid}_comment",
                               "tooltip": f"{name} ({e0['dataset']}) - optional comment",
                               "rect": mm(0.58, 0.04, 0.4, 0.13)})
            else:
                fig.text(0.58, 0.3, wrap("No follow-up question: no CheXagent box was shown for this finding on the "
                                         "original form, so there is nothing to compare against.", 70),
                         fontsize=7.5, va="top", color="#555555")
            fig.text(0.015, 0.008, "Red = CheXagent box (model-generated candidate location, not disease extent); "
                     "blue/magenta outlines = anatomy from segmentation.", fontsize=6.3)
            pdf.savefig(fig)
            plt.close(fig)
            summary.append({"page": pno, "condition": cond, "row_type": rtype, "dataset": e0["dataset"],
                            "forms": "; ".join(f"{DOCTOR[e['doctor']]} {FORM_NAME[e['form']]} {e['form_case']}"
                                               for e in es),
                            "visualization_A": "none" if bi is None else bi[0], "B": labb, "question": ask,
                            "comments": len(cl), "context_comments": len(ctx)})
            for e in es:
                audit.append({"page": pno, "condition": cond, "row_type": rtype, "dataset": e["dataset"],
                              "doctor": DOCTOR[e["doctor"]], "source_form": FORM_NAME[e["form"]],
                              "source_pdf": e["pdf"].name, "form_case": e["form_case"],
                              "field_prefix": e["field_prefix"], "asked": "/".join(e["asked"]),
                              **{f"score_{kk}": e["scores"][kk] for kk, _ in DIMS}, "comment": e["comment"],
                              "workbook_check": e["workbook_check"]})
                if e["workbook_check"] != "match":
                    flags.append(f"{name} ({e['dataset']}) {DOCTOR[e['doctor']]} {FORM_NAME[e['form']]} "
                                 f"{e['form_case']}: workbook {e['workbook_check']}")
            sids = [e for e in es if e["doctor"] == "Dr Siddarth"]
            if len(sids) > 1 and len({json.dumps(e["scores"]) for e in sids}) > 1:
                flags.append(f"{name} ({e0.get('title19') or e0['dataset']}) Dr. Siddharth: Form B vs 19-case form differ "
                             + " vs ".join(str([e["scores"][kk] for kk, _ in DIMS]) for e in sids))
            frs = [e for e in es if e["doctor"] == "Dr Franco"]
            if dup_f and len({json.dumps(e["scores"]) for e in frs}) > 1:
                flags.append(f"{name} ({e0.get('title19') or e0['dataset']}) Dr. Franco: Form A (preferred) vs 19-case form differ "
                             + " vs ".join(str([e["scores"][kk] for kk, _ in DIMS]) for e in frs))
        # comments index
        fig = plt.figure(figsize=(W_IN, H_IN))
        pno += 1
        L = ["ALL DOCTOR COMMENTS ON THESE FOUR CONDITIONS (verbatim, from the returned PDFs)", ""]
        for s, key in zip(summary, order):
            for e in items[key]:
                if e["comment"]:
                    L.append(f"p{s['page']:02d}  {DOCTOR[e['doctor']]:14s} {FORM_NAME[e['form']]:13s} "
                             f"{e['form_case']:18s} {key[1]:16s}: \"{e['comment']}\"")
        fig.text(0.03, 0.97, "\n".join(L), fontsize=7.2, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
    V.add_fields(out, fields)
    acsv = out.with_name(out.stem + "_score_audit.csv")
    with acsv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(audit[0]))
        w.writeheader()
        w.writerows(audit)
    md = out.with_name(out.stem + "_summary.md")
    M = [f"# Focused conditions report - summary ({out.name})", "",
         f"{len(entries)} scored rows (workbook has {n_wb}); {len(order)} findings; follow-up questions: "
         f"{sum(s['question'] for s in summary)}. Workbook check: {sum(a['workbook_check'] == 'match' for a in audit)}"
         f"/{len(audit)} match.", "", "| Page | Condition | Row type | Dataset | Forms | A | Question | Comments |",
         "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        M.append(f"| {s['page']} | {s['condition']} | {s['row_type']} | {s['dataset']} | {s['forms']} | "
                 f"{s['visualization_A']} | {'yes' if s['question'] else 'no'} | {s['comments']} "
                 f"(+{s['context_comments']} context) |")
    M += ["", "## Needs manual reconciliation", "", "Score order: [MedGemma finding, interval change, CheXagent "
          "presence evidence, CheXagent grounding].", ""] + [f"- {f}" for f in flags or ["none"]]
    md.write_text("\n".join(M), encoding="utf-8")
    print(f"-> {out} ({pno} pages, {len(fields)} fields)\n-> {acsv}\n-> {md}")
    print("\n".join(M[2:3] + M[-len(flags or [0]) - 4:]))


if __name__ == "__main__":
    main()
