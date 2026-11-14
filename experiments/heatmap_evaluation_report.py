"""Radiologist visualization evaluation report: BioViL-T heat map vs the original CheXagent box.

Conditions: pleural effusion, consolidation, pneumonia, pneumothorax. These are the conditions where the BioViL-T
heat map passed the pre-stated pointing-game check (experiments/results/heatmap_subtraction_validation_20261109.md).
Atelectasis did not and is excluded.

Same layout, scores, comments, disagreement flags, summary page and form fields as visualization_evaluation_report.py
(the 20261108-182229 report). Panel B is the BioViL-T text-guided similarity map instead of the segmentation outline.

- Prompt: the condition name only (report-blinded). MedGemma's text or side and the reference report are never used
  to steer the map.
- Panel B is exploratory. It shows where image features resemble the words, not whether the finding is present (the
  map also lights up when the finding is absent). In the validation it pointed inside radiologist boxes more often
  than chance, but less often than CheXagent.
- Reference status: rule-based sentence audit (reference_sentence_audit, rules v3, exact sentences quoted).
  Pneumonia uses its own term; airspace opacity and consolidation count as indirect signs (set in memory here, the
  shared rules file is not changed).
- Nothing existing is changed. New time-stamped files go in logs/ (MIMIC-CXR content).

    .venv/Scripts/python.exe experiments/heatmap_evaluation_report.py
"""

import csv
import json
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import condition_aware_grounding as G  # noqa: E402
import focused_conditions_report as FC  # noqa: E402
import reference_sentence_audit as RA  # noqa: E402
import regional_focal_reports_v2 as R2  # noqa: E402
import visualization_evaluation_report as VE  # noqa: E402
from biovil_t_heatmap_pilot import REVISION, load_engine  # noqa: E402

LOGS = ROOT / "logs"
CONDS = ("pleural effusion", "consolidation", "pneumonia", "pneumothorax")
DIMS, DOCTOR, FORM_NAME = VE.DIMS, VE.DOCTOR, FC.FORM_NAME
QUESTION = "Which visualization better shows where to look for the stated finding?"
ANSWERS = ["", "A better", "B better", "Equally useful", "Neither useful"]
COMMENT_Q = "Optional: Does heat map B highlight the abnormality itself, or other structures?"
PURPOSE = ("This report evaluates whether an alternative visualization (a text-guided heat map) is more useful to "
           "radiologists than the original CheXagent box. It makes no claim of improved diagnostic accuracy.")
# pointing-game hit rate (peak inside radiologist boxes) from the validation, for the caption
VALID = {"pleural effusion": "VinDr 0.40, NIH 0.47 (CheXagent 0.87*, 0.55)",
         "consolidation": "VinDr 0.52 (CheXagent 0.92*)",
         "pneumonia": "NIH 0.58 (CheXagent 0.75)",
         "pneumothorax": "VinDr 0.73, NIH 0.33 (CheXagent 0.93*, 0.67)"}
RA.TERMS.setdefault("pneumonia", r"\bpneumoni\w*")
RA.INDIRECT.setdefault("pneumonia", r"airspace (?:opacit\w*|disease)|\bconsolidat\w*")


def reference(e0, texts):
    if e0["dataset"].startswith("NIH"):
        ref = e0["reference"] or ""
        present = ref and ref != "NOT IN GROUND TRUTH" and not ref.startswith("No")
        return ("Present per NIH labels (text-mined; no report text)" if present else
                "Absent per NIH labels (label not assigned; labels are text-mined)"), []
    res = RA.classify(e0["condition"], texts.get(e0["key"], ""))
    return RA.describe(res), list(dict.fromkeys(s for _, s in res["evidence"]))


def heat(eng, e0, shape):
    sim = eng.get_similarity_map_from_raw_data(Path(e0["current"]), e0["condition"], interpolation="bilinear")
    sim = cv2.resize(sim.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    fin = np.isfinite(sim)
    v = np.where(fin, sim, -np.inf)
    py, px = np.unravel_index(np.argmax(v), v.shape)
    top = fin & (sim >= np.quantile(sim[fin], 0.9))
    w = shape[1]
    st = {"peak": float(sim[py, px]), "peak_xy": (int(px), int(py)),
          "peak_side": "patient right" if px < 0.45 * w else "patient left" if px > 0.55 * w else "midline",
          "top10_right": float(top[:, : w // 2].sum() / top.sum()), "top10_left": float(top[:, w // 2:].sum() / top.sum())}
    return sim, top, st


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    FC.CONDS = CONDS
    entries, page_comments, v26, book, n_wb = FC.restore()
    texts = VE.reference_texts()
    eng = load_engine("cuda" if torch.cuda.is_available() else "cpu")
    items = {}
    for e in entries:
        items.setdefault((e["key"], e["condition"], e["row_type"]), []).append(e)
    order = sorted(items, key=lambda k: (CONDS.index(k[1]), items[k][0]["dataset"], k[2], k[0]))
    readers = {}
    bis = {k: FC.box_image(items[k][0], v26, readers) for k in order}
    eligible = {k for k in order if bis[k] is not None and bis[k][0] != "embedded_other"}
    both = {k for k in order if {e["doctor"] for e in items[k]} == {"Dr Franco", "Dr Siddarth"}}
    disag = {k: VE.disagreements(items[k]) for k in order}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = LOGS / f"heatmap_evaluation_report_{stamp}.pdf"
    W_IN, H_IN = 11.7, 8.3
    mm = lambda fx, fy, fw, fh: (fx * W_IN * 25.4, (1 - fy - fh) * H_IN * 25.4, fw * W_IN * 25.4, fh * H_IN * 25.4)  # noqa: E731
    wrap = lambda s, n: "\n".join(textwrap.wrap(" ".join(str(s).split()), n))  # noqa: E731
    fields, rows_out, layout = [], [], {}
    plan_pages = {k: i + 2 for i, k in enumerate(order)}
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(W_IN, H_IN))
        lines = ["RADIOLOGIST VISUALIZATION EVALUATION - TEXT-GUIDED HEAT MAP (BioViL-T) - SUMMARY", "",
                 wrap(PURPOSE, 110), "",
                 f"Unique findings:                                  {len(order)}",
                 f"Eligible for A-versus-B comparison:               {len(eligible)}  (a CheXagent box for the finding "
                 "was shown on the original form)",
                 f"Findings evaluated by both radiologists:          {len(both)}  ({len({k[0] for k in both})} "
                 "distinct cases / image pairs)",
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
                       {items[k][0]['dataset'] for k in order}}.items())), "",
                  "Visualization B: BioViL-T (microsoft/BiomedVLP-BioViL-T) similarity between each image region and",
                  "the condition name (e.g. 'pleural effusion'); report-blinded. Red = region resembles the words, blue =",
                  "does not; dashed outline = the 10% most similar area; + = peak. It does NOT indicate whether the",
                  "finding is present (it also highlights regions when the finding is absent). Checked on radiologist-",
                  "drawn boxes (VinDr, NIH): the peak fell inside the box more often than chance, but less often than",
                  "CheXagent. Atelectasis is excluded (failed that check).", "",
                  "How to complete each page: look at A and B, answer the question and (optionally) the comment, then",
                  "read the previous radiologist evaluation below it. Previous scores rate the original report and",
                  "box A - they are not ratings of visualization B.", "",
                  "MIMIC-CXR content (PhysioNet DUA): do not distribute."]
        fig.text(0.04, 0.95, "\n".join(lines), fontsize=8.4, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        for key in order:
            es = sorted(items[key], key=lambda e: (e["doctor"], e["form"] == "MIMIC-19", e["form"]))
            e0 = es[0]
            cond, rtype = key[1], key[2]
            pno = plan_pages[key]
            frects = []
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
            for j, (w, lab) in enumerate((("prior", "Prior"), ("current", "Current"))):
                ax = fig.add_axes([0.015 + j * 0.135, 0.57, 0.13, 0.33])
                ax.imshow(ims[w], cmap="gray")
                ax.axis("off")
                ax.set_title(lab + " (original)", fontsize=7.5, loc="left")
            bi = bis[key]
            ax = fig.add_axes([0.29, 0.53, 0.2, 0.37])
            if bi is None:
                ax.imshow(ims["current"], cmap="gray")
                capa = "A. No CheXagent box was shown for this row on the original form"
            elif bi[0] == "boxes":
                ax.imshow(ims["current"], cmap="gray")
                for b in bi[1]:
                    x1, y1, x2, y2 = G.box_px(b, ims["current"].shape)
                    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
                capa = "A. Original CheXagent box for this finding (unchanged; model-generated, not disease extent)"
            else:
                ax.imshow(bi[1])
                capa = (f"A. Exact box image from the returned {FORM_NAME[e0['form']]} (all CheXagent boxes of that "
                        "page; model-generated)" if bi[0] == "embedded" else
                        f"A. Box image from the returned {FORM_NAME[e0['form']]}: CheXagent placed NO box for this "
                        "finding (boxes shown belong to other findings)")
            ax.axis("off")
            ax.set_title(wrap(capa, 46), fontsize=6.6, loc="left")
            # ---- B: heat map
            cur = ims["current"]
            sim, top, hs = heat(eng, e0, cur.shape)
            ax = fig.add_axes([0.5, 0.53, 0.2, 0.37])
            ax.imshow(cur, cmap="gray")
            hm = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.5)
            ax.contour(top.astype(float), levels=[0.5], colors="black", linewidths=0.8, linestyles="dashed")
            ax.plot(*hs["peak_xy"], marker="+", color="black", ms=9, mew=1.4)
            ax.axis("off")
            ax.set_title(wrap(f"B. Text-guided heat map: similarity to '{cond}' (BioViL-T; EXPLORATORY - "
                              "where to look, not whether present)", 46), fontsize=6.6, loc="left")
            cb = fig.add_axes([0.5, 0.515, 0.2, 0.008])
            fig.colorbar(hm, cax=cb, orientation="horizontal").ax.tick_params(labelsize=5, pad=1)
            fig.text(0.5, 0.48, wrap(f"Peak similarity {hs['peak']:.2f} at the {hs['peak_side']}; top 10%: "
                                     f"{100 * hs['top10_right']:.0f}% patient right / {100 * hs['top10_left']:.0f}% left. "
                                     f"Validation hit rate for '{cond}': {VALID[cond]}. *VinDr-trained, optimistic.",
                                     66), fontsize=6.0, va="top", color="#1f4e9c")
            for axx in (fig.axes[2], fig.axes[3]):
                axx.text(0.015, 0.015, "R", transform=axx.transAxes, fontsize=6.5, va="bottom", color="yellow",
                         weight="bold")
                axx.text(0.985, 0.015, "L", transform=axx.transAxes, fontsize=6.5, va="bottom", ha="right",
                         color="yellow", weight="bold")
            # ---- finding facts
            ref_label, ref_sents = reference(e0, texts)
            facts = [f"MedGemma finding: {e0['name'] if rtype == 'MedGemma stated' else 'Not detected'}",
                     f"MedGemma interval change: {e0['change'] or ('-' if rtype != 'MedGemma stated' else 'not stated')}",
                     f"CheXagent shown on the form: {e0['chexagent_shown']}" + (f" (p_present {e0['p_present']:.2f})"
                                                                               if e0.get("p_present") else ""),
                     "Reference: " + ref_label] + [f'  "{s}"' for s in ref_sents[:2]]
            ftxt = "\n".join(sum((textwrap.wrap(s, 50) for s in facts), []))
            if ftxt.count("\n") > 17:
                ftxt = "\n".join(ftxt.split("\n")[:17]) + "\n  [...]"
            fig.text(0.715, 0.9, "Finding", fontsize=8.5, weight="bold", va="top")
            fig.text(0.715, 0.88, ftxt, fontsize=6.5, va="top")
            # ---- new evaluation (first)
            fig.text(0.015, 0.43, "Visualization evaluation", fontsize=10, weight="bold", va="top")
            if key in eligible:
                fig.text(0.015, 0.405, wrap(QUESTION, 58), fontsize=7.8, va="top")
                fields.append({"page": pno, "kind": "choice", "name": f"p{pno:03d}_which_better",
                               "tooltip": f"{name}: {QUESTION}", "rect": mm(0.015, 0.325, 0.3, 0.03),
                               "options": ANSWERS})
                frects.append(("which_better", (0.015, 0.325, 0.3, 0.03)))
                fig.text(0.33, 0.405, wrap(COMMENT_Q, 70), fontsize=7, va="top")
                fields.append({"page": pno, "kind": "text", "name": f"p{pno:03d}_b_shows",
                               "tooltip": f"{name}: {COMMENT_Q}", "rect": mm(0.33, 0.3, 0.37, 0.075)})
                frects.append(("b_shows", (0.33, 0.3, 0.37, 0.075)))
            else:
                fig.text(0.015, 0.405, wrap("Not eligible for the A-versus-B question: no CheXagent box for this "
                                            "finding was shown on the original form.", 95), fontsize=7.5, va="top",
                         color="#555555")
            # ---- previous evaluation (below)
            fig.add_artist(plt.Line2D([0.015, 0.985], [0.285, 0.285], color="#999999", lw=0.6))
            fig.text(0.015, 0.275, "Previous radiologist evaluation (please answer the question above first)",
                     fontsize=9, weight="bold", va="top")
            fig.text(0.015, 0.255, "Original scores (1-5) and comments from the returned forms; they rate the original report "
                     "and box A, not B. '-' not asked; 'blank (N/A)' left blank.", fontsize=6.2,
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
            fig.text(0.66, 0.228, "\n".join(sum((textwrap.wrap(c, 62) for c in cl), [])[:6]) or "none", fontsize=6.1,
                     va="top")
            fig.text(0.66, 0.14, "Other comments on the same case (context)", fontsize=7, weight="bold", va="top")
            fig.text(0.66, 0.123, "\n".join(sum((textwrap.wrap(c, 62) for c in ctx[:4]), [])[:7]) or "none",
                     fontsize=5.8, va="top", color="#444444")
            if disag[key]:
                msg = "; ".join(f"{', '.join(f'{lab} {x} vs {y}' for lab, x, y in d)} ({FORM_NAME[a['form']]} vs "
                                f"{FORM_NAME[b['form']]})" for a, b, d in disag[key])
                dtxt = wrap("DOCTORS DIFFER SUBSTANTIALLY (>= 2 points; Dr. Franco vs Dr. Siddharth): " + msg +
                            ". Original ratings shown unchanged; not reconciled.", 105).split(chr(10))
                fig.text(0.015, 0.085, chr(10).join(dtxt[:3]) + (" [...]" if len(dtxt) > 3 else ""),
                         fontsize=6.4, va="top", color="#7a4b00", weight="bold")
            fig.text(0.015, 0.008, "Red box = CheXagent (model-generated candidate location, not disease extent). "
                     "Heat map: red = resembles the condition words, blue = does not; dashed = top 10%; + = peak. "
                     "R/L = patient right/left.", fontsize=5.8)
            fig.text(0.015, 0.024, PURPOSE, fontsize=5.8)
            layout[pno] = R2.layout_issues(fig, frects)
            pdf.savefig(fig)
            plt.close(fig)
            rows_out.append({"page": pno, "condition": cond, "row_type": rtype, "dataset": e0["dataset"],
                             "forms": "; ".join(f"{DOCTOR[e['doctor']]} {FORM_NAME[e['form']]} {e['form_case']}"
                                                for e in es),
                             "eligible_A_vs_B": key in eligible, "both_doctors": key in both,
                             "disagreement": "; ".join(f"{lab} {x} vs {y}" for a, b, d in disag[key] for lab, x, y in d),
                             "reference_status": ref_label, "reference_sentences": " | ".join(ref_sents),
                             "heatmap_prompt": cond, "heatmap_peak": round(hs["peak"], 3),
                             "heatmap_peak_side": hs["peak_side"], "heatmap_top10_right": round(hs["top10_right"], 3),
                             "heatmap_top10_left": round(hs["top10_left"], 3)})
            print(f"p{pno} {cond} {rtype} {e0['dataset'][:12]} peak {hs['peak']:.2f} {hs['peak_side']}", flush=True)
        fig = plt.figure(figsize=(W_IN, H_IN))
        L = ["ALL DOCTOR COMMENTS ON THESE CONDITIONS (verbatim, from the returned PDFs)", ""]
        for r, key in zip(rows_out, order):
            for e in items[key]:
                if e["comment"]:
                    L.append(f"p{r['page']:02d}  {DOCTOR[e['doctor']]:14s} {FORM_NAME[e['form']]:13s} "
                             f"{e['form_case']:18s} {key[1]:16s}: \"{e['comment'][:90]}\"")
        fig.text(0.03, 0.97, "\n".join(L[:70]), fontsize=7.0, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
    VE.add_fields(out, fields)
    fq = R2.check_fields(out, fields)
    csvp = out.with_name(out.stem + "_findings.csv")
    with csvp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    qc = out.with_name(out.stem + "_qc.json")
    qc.write_text(json.dumps({"model": f"microsoft/BiomedVLP-BioViL-T@{REVISION}", "fields": fq,
                              "layout_issues": {k: v for k, v in layout.items() if v}}, indent=1), encoding="utf-8")
    print(f"-> {out} ({len(order) + 2} pages, {len(fields)} fields)\n-> {csvp}\n-> {qc}")
    print(f"findings {len(order)}, eligible {len(eligible)}, both doctors {len(both)}, disagreement "
          f"{sum(1 for k in order if disag[k])}; fields {fq}; pages with layout issues "
          f"{sum(1 for v in layout.values() if v)}")


if __name__ == "__main__":
    main()
