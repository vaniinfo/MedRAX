"""Radiologist scoring form for the 20 NIH prior + current examples (edv_longitudinal_examples_*.json), in the
layout of the MIMIC scoring form (comparison_edv_pdf.py --gt-under-images --radiologist-form ...): large
full-resolution X-rays, ground truth under them, the base MedGemma longitudinal report, and a Finding Comparison
table with a 1-5 dropdown in each of the first three cells, an editable comment, and mouse-over CheXagent scores.

NIH has no ground-truth report text: the ground truth is the NIH labels of both images, the ground-truth column
says Yes / No (labels give no location), and the comment box starts empty.

    python experiments/nih20_scoring_pdf.py --examples logs/edv_longitudinal_examples_20261020-065035.json
"""

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import comparison_edv_pdf as C  # noqa: E402
import edv_longitudinal as L  # noqa: E402
from maira_score_nih import NIH  # noqa: E402
from medrax.evidence import EvidenceEngine  # noqa: E402
from medrax.tools.medgemma import describe_location  # noqa: E402
from preview_onepage_report import OnePage  # noqa: E402

LEGEND = [
    ("Agree", "the automated image check supports the finding: CheXagent's score (prior + current images) is at or "
              "above the condition's support cut-off (a likelihood ratio of at least 5). NIH labels give no side, so "
              "laterality is not checked."),
    ("Not agree", "the automated image check does not support the finding: the score is below the support cut-off "
                  "or at or below the condition's strong-no cut-off."),
    ("Ground truth", "NIH ChestX-ray14 labels, text-mined from the original reports (which are not released); "
                     "Yes = the current image is labelled with the condition."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True, help="edv_longitudinal_examples_*.json (the 20 chosen pairs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "logs" / f"nih20_scoring_form_{time.strftime('%Y%m%d-%H%M%S')}.pdf"
    if out.exists():
        raise SystemExit(f"{out} already exists; not overwriting")

    examples = json.loads(Path(args.examples).read_text(encoding="utf-8"))
    reports = {json.loads(x)["current"]: json.loads(x) for x in L.REPORTS.read_text(encoding="utf-8").splitlines()}
    with open("D:/datasets/chestxray14/Data_Entry_2017_v2020.csv", encoding="utf-8") as f:
        meta = {r["Image Index"]: r for r in csv.DictReader(f)}
    lab = lambda im: meta[im]["Finding Labels"].replace("_", " ").replace("|", ", ")  # noqa: E731

    from medrax.tools.xray_vqa import XRayVQATool
    chexagent = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    engine = EvidenceEngine()
    work = Path(tempfile.mkdtemp(prefix="nih20_pdf_", dir=ROOT / "temp"))

    pdf = OnePage()
    pdf.radiologist_form, pdf.form_fields = True, []
    pdf.hide_scores, pdf.two_level, pdf.gt_comment = True, True, False
    pdf.image_max, pdf.image_gap = 1800, 1.5
    pdf.set_margins(6, 6, 6)
    pdf.zoom_targets = []  # click an X-ray to open it full page
    pdf.collapsible, pdf.toggle_buttons = True, []  # structured report behind a show/hide button
    sorter = C.make_extractor("qwen3.8-edv:latest")
    titles = []
    for ex in examples:
        f, prior = ex["current"], ex["prior"]
        items = []
        for r in ex["rows"]:
            item = dict(r)
            item["ground_truth"] = "Yes" if r["ground_truth"].startswith("Present") else C.NOT_IN_GT
            item["tooltip"] = C.evidence_tooltip(engine, item)
            items.append(item)
        conds = [r["finding"] for r in items]
        annotated = None
        if conds:  # CheXagent's boxes on the current image (location only; labels without scores)
            proof, annotated = chexagent.locate_findings(str(NIH / f), conds)
            boxes = [{"phrase": q["finding"], "box_normalized": b["box_normalized"], "p_present": None}
                     for q in proof if q["localized"] for b in q["boxes"]]
            annotated = C.shrink(C.XRayVQATool._visualize_boxes(chexagent, str(NIH / f), boxes), work, size=1800) \
                if boxes else None
            where = {q["finding"]: "; ".join(dict.fromkeys(describe_location(b["box_normalized"]).replace("patient's ", "")
                                                           for b in q["boxes"])) for q in proof if q["localized"]}
            for i in items:
                if where.get(i["finding"]):
                    i["image_location"] = where[i["finding"]]
                    i["tooltip"] = C.evidence_tooltip(engine, i)
        page = {
            "title": f"NIH example {ex['example']} of {len(examples)}",
            "row": "",
            # some reports put "IMPRESSION:" at the end of the findings line; give it its own line
            "generated": C.sections(re.sub(r"\s*IMPRESSION:\s*", "\nIMPRESSION:\n",
                                           reports[f]["report"]).strip().splitlines()),
            "truth": [("", f"Current image ({f}): {lab(f)}."), ("", f"Prior image ({prior}): {lab(prior)}.")],
        }
        # the radiologist's template filled from MedGemma's report (sentences unchanged; cached in the .json)
        rec = {"title": page["title"], "structured": ex.get("structured")}
        C.add_structured(rec, page, sorter)
        ex["structured"] = rec["structured"]
        C.draw_page(pdf, page, str(NIH / prior), str(NIH / f), annotated, items, rec["structured"],
                    truth_title="Ground Truth (NIH labels)", legend=LEGEND, gt_under_images=True)
        titles.append(f"{page['title']} ({f})")
        print(f"{page['title']}: {f}  {[(i['finding'], i['status'], i['ground_truth']) for i in items]}", flush=True)
    C.add_zoom_pages(pdf)
    pdf.output(str(out))
    C.add_structured_toggle(out, pdf.toggle_buttons)
    n = C.add_form_fields(out, pdf.form_fields, titles)
    C.add_zoom_overlays(out, pdf.zoom_targets)
    out.with_suffix(".json").write_text(json.dumps(examples, indent=1), encoding="utf-8")
    print(f"{len(examples)} pages, {n} scored rows -> {out}")


if __name__ == "__main__":
    main()
