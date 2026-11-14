"""Revised radiologist report prototype (presentation only) for the six comparisons of the anatomical-measurement
feasibility experiment (Comps 8, 15, 7, 13, 3, 5). No new measurements are developed.

Unchanged: model predictions, MedGemma's reports and change statements, CheXagent's verdicts and boxes, the Evidence
Engine, every radiologist score and every source image. New time-stamped files only; MIMIC-CXR content in logs/.

Display rules
  1. CTR: current film only, eligible = projection PA AND automatic QC (lungs, heart, midline/rotation) passed AND
     manual heart-outline review not 'incorrect'. Heart and thoracic lines visible. 'not reviewed' -> CTR labelled
     PROVISIONAL; 'incorrect' -> no CTR, no lines. The 0.50 reference is quoted only on eligible PA films; AP and
     unconfirmed projections never get a CTR value.
  2. Costophrenic angle and recess depth: exploratory, eligible PA films only (QC passed); never described as proof of
     pleural effusion. AP films are excluded (see the screening-vs-display note).
  3. Low lung volumes, hyperinflation: lung outline only, no numbers.
  4. Atelectasis, consolidation/pneumonia/opacities, pulmonary edema and all other findings: no measurement; the
     original CheXagent candidate box (unchanged) where one exists.
  5. Prior-to-current numerical changes: research appendix only (not on the reader pages).
  6. Manual QC: logs/report_manual_qc_mimic19.json (created with 'not reviewed' if missing; never overwritten).
     Per comparison: heart_outline in {not reviewed, correct, incorrect}; projection_confirmed in {null, PA, AP}
     (a confirmed value overrides the held-out CSV; null + no CSV value = projection not recorded = not eligible).
  7. The original prior and current X-rays are shown large and clean: no outline, line or box is drawn on them.
     The one visualization per finding is a separate image. The reference report stays on the overview page, as in
     the existing review protocol.

    .venv/Scripts/python.exe experiments/radiologist_report_final.py
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
import anatomical_measurements as A  # noqa: E402
import condition_aware_grounding as G  # noqa: E402

LOGS = ROOT / "logs"
RESULTS = HERE / "results"
CASES = [8, 15, 7, 13, 3, 5]
QC_FILE = LOGS / "report_manual_qc_mimic19.json"
VERD = {"Supported": ("Agree", "#1a7f37"), "Indeterminate": ("Not agree (inconclusive)", "#b42318"),
        "Requires confirmation": ("Not agree", "#b42318")}
BOX_LABEL = "Original CheXagent candidate box (unchanged) - model-generated, not confirmed disease extent"

SCREEN_VS_DISPLAY = """STATISTICAL SCREENING CRITERIA vs CLINICAL DISPLAY RESTRICTIONS

Screening (NIH feasibility check, pre-stated): a measurement passes when it can be computed on >= 80% of images and
the lower 95% bound of its AUC (condition vs No Finding) exceeds 0.60. Passing means only that the number carries
some signal in a selected, easy comparison. It is a necessary condition for display, not a sufficient one.

Display restrictions (clinical judgement, applied after screening):
- CTR on PA films only. AP films magnify the heart, so CTR values are not comparable with the PA reference (0.50);
  showing an AP value invites that comparison. CTR is shown only when the heart outline is not marked incorrect,
  and is labelled PROVISIONAL until a reviewer confirms the outline (Comp 8 showed that a visibly wrong outline can
  pass every automatic check).
- Costophrenic angle / recess depth on PA films only, as exploratory information. AP excluded although the AP angle
  passed screening (AUC 0.69, CI 0.64-0.75; AP depth 0.63, CI 0.57-0.69, did not pass), because:
    * AP films are mostly portable, supine or semi-erect: free fluid layers posteriorly instead of collecting in the
      costophrenic recess, so recess shape is a weaker and less specific sign;
    * AP projection, rotation and limited inspiration change the lung-base shape independently of fluid; AP QC pass
      rates were lower (midline 76% vs 94% PA);
    * the AP signal is modest and was measured against easy No-Finding controls, i.e. it would be weaker in
      practice; one of the two AP measurements failed screening.
- No display of measurements that did not pass screening (hyperinflation dome/height/area, atelectasis displacement,
  classifier saliency for consolidation) or that have no validation data (low lung volumes): outlines or the
  original CheXagent box only.
- Prior-to-current numerical changes are research-only: in the 19 comparisons the same patient's CTR varied by up
  to 0.13 between films, larger than the differences a reader would act on."""


def manual_qc(data):
    if QC_FILE.exists():
        return json.loads(QC_FILE.read_text(encoding="utf-8"))
    qc = {x["r"]["title"]: {"heart_outline": "not reviewed", "projection_confirmed": None,
                            "projection_from_csv": x["proj"], "reviewer": "", "date": "", "note": ""} for x in data}
    QC_FILE.write_text(json.dumps(qc, indent=1), encoding="utf-8")
    print(f"manual QC template -> {QC_FILE}")
    return qc


def load_case(r, rot_limit, proj_csv):
    cur = Path(r["image"])
    sid = cur.parent.name.split("_subject_")[1]
    imgs, res = {}, {}
    for which, p in (("prior", cur.with_name("prior.jpg")), ("current", cur)):
        im = Image.open(p).convert("L")
        im.thumbnail((1024, 1024))
        imgs[which] = np.array(im)
        res[which] = A.measure(A.segment_all(imgs[which]), rot_limit)
    return dict(r=r, sid=sid, imgs=imgs, res=res, proj=proj_csv.get(sid))


def kind(finding):
    f = finding.lower()
    if "cardiomegaly" in f:
        return "heart"
    if "effusion" in f:
        return "cp"
    if "low lung volumes" in f or "hyperinflation" in f:
        return "outline"
    return "box"


def decide(it, x, qc):
    """(visualization spec, measurement note lines, reason) for one finding."""
    k = kind(it["finding"])
    R = x["res"]["current"]
    q, v = R["qc"], R["val"]
    proj = qc.get("projection_confirmed") or x["proj"]
    pa = proj == "PA"
    vp = next((p for p in x["r"]["visual_proof"] if p["finding"] == it["finding"] and p.get("boxes")), None)
    boxes = [b["box_normalized"] for b in vp["boxes"]] if vp else []
    box_vis = ("box", boxes) if boxes else ("none", [])
    if k == "heart":
        auto_ok = q["lungs"] and q["heart"] and q["midline"]
        hq = qc.get("heart_outline", "not reviewed")
        if not auto_ok:
            why = "; ".join(s for s in (R["why"].get("lungs"), R["why"].get("heart"), R["why"].get("midline")) if s)
            return ("heart_outline" if q["lungs"] and q["heart"] else box_vis[0], boxes), [], \
                f"CTR not shown: automatic quality check failed ({why})."
        if hq == "incorrect":
            return box_vis, [], "CTR suppressed: heart outline marked incorrect at manual review."
        if not pa:
            reason = ("AP film: CTR not shown (heart magnified; the PA reference does not apply)." if proj == "AP"
                      else "Projection not recorded: CTR shown only on confirmed PA films.")
            return ("heart_outline", []), [], reason
        prov = hq != "correct"
        return ("ctr", []), [f"Cardiothoracic ratio {v['ctr']:.2f}" + (" - PROVISIONAL (heart outline not yet "
                                                                       "reviewed)" if prov else "")
                             + "; conventional PA reference 0.50."], \
            "Measurement from heart and lung segmentation; check the outline before using the value."
    if k == "cp":
        if pa and q["lungs"] and q["base_R"] and q["base_L"]:
            return ("cp", []), [f"EXPLORATORY - costophrenic angle R {v['cp_angle_R']:.0f} deg, L "
                                f"{v['cp_angle_L']:.0f} deg; recess depth R {v['recess_depth_R']:.2f}, L "
                                f"{v['recess_depth_L']:.2f} (x thoracic width)."], \
                "Describes the lung outline only; not proof of pleural effusion (fluid is not segmented)."
        reason = ("AP film" if proj == "AP" else "projection not recorded" if proj is None else "QC failed")
        return box_vis, [], f"No costophrenic measurement ({reason}: shown on eligible PA films only)."
    if k == "outline":
        if q["lungs"]:
            return ("outline", []), [], ("Lung outline for orientation only. No lung measurement is validated for "
                                         "this finding, so none is shown.")
        return box_vis, [], "Lung segmentation failed quality check; no outline shown."
    return box_vis, [], "No validated anatomical measurement for this finding."


def draw_vis(ax, arr, R, spec):
    from matplotlib.patches import Rectangle
    typ, boxes = spec
    if typ == "ctr":
        A.draw_anatomy(ax, arr, R, ("ctr",))
        cap = "Heart outline (magenta) and CTR lines: yellow = heart width either side of the midline, cyan = " \
              "thoracic width (segmentation)"
    elif typ == "cp":
        A.draw_anatomy(ax, arr, R, ("cp",))
        cap = "Lung outline and costophrenic lines (segmentation; exploratory)"
    elif typ == "heart_outline":
        ax.imshow(arr, cmap="gray")
        ax.axis("off")
        ax.imshow(np.ma.masked_where(~G.outline(R["geo"]["heart_mask"]), np.ones_like(arr)), cmap="spring",
                  alpha=0.9)
        cap = "Heart outline (segmentation; no measurement)"
    elif typ == "outline":
        A.draw_anatomy(ax, arr, R, ())
        cap = "Lung outline (segmentation; orientation only)"
    else:
        ax.imshow(arr, cmap="gray")
        ax.axis("off")
        for b in boxes:
            x1, y1, x2, y2 = G.box_px(b, arr.shape)
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
        cap = BOX_LABEL if boxes else "No CheXagent box recorded for this finding"
    ax.set_title("\n".join(textwrap.wrap(cap, 46)), fontsize=6.5, loc="left")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pypdf import PdfReader
    import comparison_edv_pdf as C
    from comparison_edv_pdf import parse_page

    N = json.loads(sorted(RESULTS.glob("anatomical_measurements_nih_*.json"))[-1].read_text(encoding="utf-8"))
    proj_csv = A.projections()
    rec = {int(r["title"].split()[1]): r for r in json.loads(
        (LOGS / "radiologist_requested_plus_edv_scoring_form_v26.json").read_text(encoding="utf-8"))}
    pages = {p["title"]: p for p in (parse_page(x.extract_text()) for x in PdfReader(
        "D:/datasets/held_out_eval_images/subset_comparisons_clean_filtered.pdf").pages)}
    data = [load_case(rec[c], N["rot_limit"], proj_csv) for c in CASES]
    QC = manual_qc(data)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = LOGS / f"radiologist_report_final_prototype_{stamp}.pdf"
    W_IN, H_IN = 11.7, 8.3
    mm = lambda fx, fy, fw, fh: (fx * W_IN * 25.4, (1 - fy - fh) * H_IN * 25.4, fw * W_IN * 25.4, fh * H_IN * 25.4)  # noqa: E731
    wrap = lambda s, n: "\n".join(textwrap.wrap(" ".join(str(s).split()), n))  # noqa: E731
    C.DROPDOWNS = list(C.DROPDOWNS) + ["visualization"]
    RATINGS = (("medgemma", "Finding accuracy"), ("change", "Interval-change accuracy"),
               ("chexagent", "CheXagent presence evidence"), ("visualization", "Visualization usefulness"))
    fields, titles, pno, log = [], [], 0, []
    with PdfPages(out) as pdf:
        for x in data:
            r = x["r"]
            page = pages[r["title"]]
            qc = QC.get(r["title"], {})
            proj = qc.get("projection_confirmed") or x["proj"]
            # ---- overview: large clean X-rays + reports
            fig = plt.figure(figsize=(W_IN, H_IN))
            pno += 1
            titles.append(f"{r['title']} (overview)")
            fig.text(0.015, 0.985, f"{r['title']}", fontsize=13, weight="bold", va="top")
            for k, w in enumerate(("prior", "current")):
                ax = fig.add_axes([0.01 + k * 0.29, 0.05, 0.285, 0.87])
                ax.imshow(x["imgs"][w], cmap="gray")
                ax.axis("off")
                ax.set_title(w.capitalize() + " (original, no overlays)", fontsize=9, loc="left")
            fig.text(0.6, 0.94, "MedGemma report", fontsize=9, weight="bold", va="top")
            fig.text(0.6, 0.915, wrap(page["generated_text"], 72), fontsize=6.6, va="top", family="monospace")
            fig.text(0.6, 0.55, "Reference report", fontsize=9, weight="bold", va="top")
            fig.text(0.6, 0.525, wrap(page["truth_text"], 72), fontsize=6.6, va="top", family="monospace")
            fig.text(0.6, 0.1, f"Projection: {proj or 'not recorded'}"
                     + (" (confirmed at review)" if qc.get("projection_confirmed") else
                        " (held-out metadata)" if x["proj"] else "")
                     + f"\nHeart outline review: {qc.get('heart_outline', 'not reviewed')}"
                     + f"\nFindings for rating: {len(r['review'])} (following pages)", fontsize=7.5, va="top")
            pdf.savefig(fig)
            plt.close(fig)
            # ---- findings: two per page
            rows = r["review"]
            for s in range(0, len(rows), 2):
                fig = plt.figure(figsize=(W_IN, H_IN))
                pno += 1
                titles.append(f"{r['title']} (findings)")
                fig.text(0.015, 0.985, f"{r['title']} - findings for rating", fontsize=11, weight="bold", va="top")
                for ri, it in enumerate(rows[s:s + 2]):
                    y0 = 0.5 - ri * 0.47
                    spec, notes, reason = decide(it, x, qc)
                    ax = fig.add_axes([0.015, y0 + 0.005, 0.3, 0.41])
                    draw_vis(ax, x["imgs"]["current"], x["res"]["current"], spec)
                    ch = r.get("change_claims", {}).get(it["finding"]) or it.get("change") or ["not stated", ""]
                    v, col = VERD.get(it["status"], (it["status"], "black"))
                    tx = 0.34
                    fig.text(tx, y0 + 0.43, it["name"], fontsize=12, weight="bold", va="top")
                    fig.text(tx, y0 + 0.39, "MedGemma interval change", fontsize=7.5, weight="bold", va="top")
                    fig.text(tx, y0 + 0.37, wrap(ch[0] + (f' - "{ch[1]}"' if ch[1] else ""), 70), fontsize=7.5,
                             va="top")
                    fig.text(tx, y0 + 0.31, "CheXagent presence verdict (report-blinded, unchanged)", fontsize=7.5,
                             weight="bold", va="top")
                    fig.text(tx, y0 + 0.29, v + (f"  (p_present {it['p_present']:.2f}; "
                                                 f"{G.calibration_status(it['finding'])})" if it.get("p_present")
                                                 else ""), fontsize=7.5, va="top", color=col)
                    fig.text(tx, y0 + 0.23, "Measurement note", fontsize=7.5, weight="bold", va="top",
                             color="#1f4e9c")
                    fig.text(tx, y0 + 0.21, wrap(" ".join(notes + [reason]), 70), fontsize=7.2, va="top")
                    for j, (_, lab) in enumerate(RATINGS):
                        fig.text(0.79, y0 + 0.4 - j * 0.06, lab, fontsize=7.5, va="top")
                    fields.append({"page": pno, "row": ri + 1, "finding": it["name"], "comment_rect": None,
                                   "evidence": v, "ground_truth": "", "tooltip": None, "tooltip_rect": None,
                                   "gt_comment": "", "comment": "", "medgemma_change": ch[0],
                                   "scores": [(kk, mm(0.94, y0 + 0.377 - j * 0.06, 0.045, 0.028))
                                              for j, (kk, _) in enumerate(RATINGS)]})
                    log.append({"comparison": r["title"], "finding": it["finding"], "visualization": spec[0],
                                "measurement_shown": bool(notes), "reason": reason})
                fig.text(0.015, 0.008, "Outlines and lines: anatomy from segmentation (not a diagnosis). Red box: "
                         "model-generated candidate location, not disease extent. Rating fields are new and empty.",
                         fontsize=6.5)
                pdf.savefig(fig)
                plt.close(fig)
        # ---- research appendix: prior-to-current numbers (not for reading)
        fig = plt.figure(figsize=(W_IN, H_IN))
        pno += 1
        titles.append("Research appendix")
        L = ["RESEARCH APPENDIX - prior-to-current measurement changes (NOT part of the reader report; not validated)",
             "Values divided by thoracic width; CP angle in degrees. 'Comparable' = both films pass QC, no rotation, "
             "same known projection.", ""]
        for x in data:
            comp, why = A.comparable(x["res"]["prior"], x["res"]["current"],
                                     QC.get(x["r"]["title"], {}).get("projection_confirmed") or x["proj"])
            L.append(f"{x['r']['title']}  comparable: {'yes' if comp else 'no'} ({why})")
            vp, vc = x["res"]["prior"]["val"], x["res"]["current"]["val"]
            parts = []
            for k, lab, d in (("ctr", "CTR", 2), ("lung_height", "lung height", 2), ("lung_area", "lung area", 2),
                              ("cp_angle_R", "CP R", 0), ("cp_angle_L", "CP L", 0), ("dome_ratio", "dome", 2)):
                if k in vp and k in vc:
                    parts.append(f"{lab} {vp[k]:.{d}f}->{vc[k]:.{d}f} ({vc[k] - vp[k]:+.{d}f})")
                else:
                    parts.append(f"{lab} n/a")
            L.append("    " + "; ".join(parts))
        L += ["", "Same-patient CTR varied by up to 0.13 across the 19 comparisons; these changes are not used in "
              "the reader pages."]
        fig.text(0.03, 0.97, "\n".join(L), fontsize=6.8, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        # ---- display rules
        fig = plt.figure(figsize=(W_IN, H_IN))
        pno += 1
        titles.append("Display rules")
        fig.text(0.03, 0.97, SCREEN_VS_DISPLAY + "\n\nMANUAL QUALITY CONTROL\nFile: " + QC_FILE.name +
                 " - heart_outline: not reviewed (CTR provisional) / correct / incorrect (CTR suppressed); "
                 "projection_confirmed:\nPA / AP (overrides the metadata). Rebuild the report after editing.",
                 fontsize=7.3, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
    C.add_form_fields(out, fields, titles)
    side = out.with_suffix(".display_log.json")
    side.write_text(json.dumps(log, indent=1), encoding="utf-8")
    note = RESULTS / f"report_display_rules_{stamp}.md"
    note.write_text("# Radiologist report: screening criteria vs display restrictions\n\n```\n" + SCREEN_VS_DISPLAY
                    + "\n```\n", encoding="utf-8")
    print(f"-> {out}\n-> {side}\n-> {note}")
    for l in log:
        print(l)


if __name__ == "__main__":
    main()
