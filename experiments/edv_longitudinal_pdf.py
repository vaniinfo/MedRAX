"""Example pages from the longitudinal EDV run (edv_longitudinal_protocol.md), in the layout of the
19-comparison review PDF: prior + current image, CheXagent's boxes, MedGemma's longitudinal report,
the NIH labels as ground truth, and the Finding Comparison table.

Illustrative examples, selected with the labels (not a representative sample): pairs where
CheXagent found at least one condition the report does not state that NIH labels present, and
none that NIH does not label. A summary page gives the counts over all 499 pairs.

    python experiments/edv_longitudinal_pdf.py [--n 20]
"""

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import comparison_edv_pdf as C  # noqa: E402
import edv_improvement as E  # noqa: E402
import edv_longitudinal as L  # noqa: E402
from chexagent_reliability import decide  # noqa: E402
from maira_score_nih import NIH  # noqa: E402
from medrax.evidence import EvidenceEngine  # noqa: E402
from medrax.tools.medgemma import describe_location  # noqa: E402
from preview_onepage_report import GREY, OnePage  # noqa: E402

STATUS = {"FOUND": "Supported", "AGAINST": "Requires confirmation", "UNCERTAIN": "Indeterminate"}
IN_GT = "Present (NIH label; no location given)"
LEGEND = [
    ("Supported", "CheXagent's score (prior + current images, \"Is there X in the current chest X-ray?\") is at "
                  "or above the condition's support cut-off (Evidence Engine v1.0; provisional NIH cut-offs for "
                  "the seven further conditions)."),
    ("Indeterminate", "the score is between the cut-offs: not enough evidence either way."),
    ("Requires confirmation", "the score is at or below the condition's strong-no cut-off."),
    ("Consider", "CheXagent found this condition on its own (score at its support cut-off), but the report does "
                 "not mention it; a prompt for radiologist review, not part of the report."),
    ("Ground truth", "NIH ChestX-ray14 labels of the current image, text-mined from the original reports; they "
                     "give no side or location, so laterality is not checked. Boxes: CheXagent's location of each "
                     "condition on the current image."),
]


def nih_labels():
    with open("D:/datasets/chestxray14/Data_Entry_2017_v2020.csv", encoding="utf-8") as f:
        return {r["Image Index"]: r for r in csv.DictReader(f)}


def table(engine, report, labels, chex, truth, f):
    """Rows of the Finding Comparison table and the event counts used for selection."""
    rows, events = [], Counter()
    for c in L.CONDITIONS:
        p = chex["p_present"][c]
        if p is None:
            continue
        d, y = decide(engine, c, p), truth[(f, c)]
        lab = labels[c]
        gt = IN_GT if y else C.NOT_IN_GT
        if lab["score"] >= 0.5:  # stated by MedGemma (present 1, possible 0.5)
            side = C.report_side(report, c)
            side_word = {"both": "bilateral", "right": "right", "left": "left"}.get(side)
            name = f"{side_word} {c}" if side_word else c
            name = name[0].upper() + name[1:] + (" (possible)" if lab["score"] < 1 else "")
            rows.append({"finding": c, "name": name, "status": STATUS[d], "p_present": p, "ground_truth": gt})
            events[("stated", d, y)] += 1
        elif d == "FOUND":
            rows.append({"finding": c, "name": f"{c.capitalize()} (not in report)", "status": "Consider",
                         "p_present": p, "ground_truth": gt})
            events[("added", y)] += 1
    return rows, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "logs" / f"edv_longitudinal_examples_{time.strftime('%Y%m%d-%H%M%S')}.pdf"
    if out.exists():
        raise SystemExit(f"{out} already exists; not overwriting")

    truth, *_ = E.load()
    engine = EvidenceEngine()
    jl = lambda path: {json.loads(l)["current"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()}  # noqa: E731
    chex, reports, labels = jl(L.CHEX_A), jl(L.REPORTS), jl(L.LABELS)
    pairs = L.pairs()

    pages, total = [], Counter()
    for pr in pairs:
        f = pr["current"]
        rows, ev = table(engine, reports[f]["report"], labels[f]["labels"], chex[f], truth, f)
        total.update(ev)
        pages.append({"pair": pr, "rows": rows, "events": ev})
    # Selection: >= 1 correct addition, no addition NIH does not label; most correct additions first,
    # then most stated conditions CheXagent supports that NIH labels, then file name
    eligible = [p for p in pages if p["events"][("added", 1)] >= 1 and p["events"][("added", 0)] == 0]
    eligible.sort(key=lambda p: (-p["events"][("added", 1)], -p["events"][("stated", "FOUND", 1)], p["pair"]["current"]))
    chosen = eligible[:args.n]
    print(f"{len(eligible)} eligible pairs; {len(chosen)} chosen")

    # CheXagent's boxes on the current image for every condition in the table (location only)
    from medrax.tools.xray_vqa import XRayVQATool
    chexagent = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    work = Path(tempfile.mkdtemp(prefix="edv_long_pdf_", dir=ROOT / "temp"))
    meta = nih_labels()
    pdf = OnePage()
    summary_page(pdf, total, len(pairs), len(eligible), len(chosen))
    record = []
    for k, p in enumerate(chosen, 1):
        f, prior = p["pair"]["current"], p["pair"]["prior"]
        conds = [r["finding"] for r in p["rows"]]
        annotated = None
        if conds:
            proof, annotated = chexagent.locate_findings(str(NIH / f), conds)
            annotated = C.shrink(annotated, work)
            where = {q["finding"]: "; ".join(dict.fromkeys(describe_location(b["box_normalized"]).replace("patient's ", "")
                                                           for b in q["boxes"])) for q in proof if q["localized"]}
            for r in p["rows"]:
                if r["status"] == "Consider" and where.get(r["finding"]):
                    r["name"] = f"{r['finding'].capitalize()} (not in report; CheXagent: {where[r['finding']]})"
        lab = lambda im: meta[im]["Finding Labels"].replace("_", " ").replace("|", ", ")  # noqa: E731
        page = {
            "title": f"Example {k}",
            "row": f"NIH patient {meta[f]['Patient ID']}  ·  prior {prior} (follow-up {meta[prior]['Follow-up #']}, "
                   f"{meta[prior]['View Position']})  ·  current {f} (follow-up {meta[f]['Follow-up #']}, "
                   f"{meta[f]['View Position']})",
            "generated": C.sections(reports[f]["report"].splitlines()),
            "truth": [("", f"Current image: {lab(f)}."), ("", f"Prior image: {lab(prior)}.")],
        }
        C.draw_page(pdf, page, str(NIH / prior), str(NIH / f), annotated, p["rows"],
                    truth_title="Ground Truth (NIH labels)", legend=LEGEND)
        record.append({"example": k, **p["pair"], "rows": p["rows"]})
        print(f"Example {k}: {f}  {[(r['name'], r['status'], round(r['p_present'], 2), r['ground_truth'][:7]) for r in p['rows']]}",
              flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    out.with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"{len(chosen)} examples -> {out}")


def summary_page(pdf, t, n_pairs, n_eligible, n_chosen):
    stated = sum(v for k, v in t.items() if k[0] == "stated")
    s = lambda d, y: t[("stated", d, y)]  # noqa: E731
    added = t[("added", 1)] + t[("added", 0)]
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_xy(pdf.l_margin, 14)
    pdf.font(15, True)
    pdf.multi_cell(W, 7, "MedGemma longitudinal reports with a CheXagent image check: examples", align="C")
    pdf.font(8.5, color=GREY)
    pdf.cell(W, 5, "NIH ChestX-ray14 test split, prior + current image pairs (edv_longitudinal_protocol.md)",
             align="C", new_x="LMARGIN", new_y="NEXT")
    paras = [
        ("What each page shows", "The prior and current X-ray; CheXagent's boxes on the current image (Visual Proof); "
         "the report base MedGemma 1.5 wrote from both images; the NIH labels of both images (ground truth); and a "
         "table of every condition the report states plus every condition CheXagent found that the report does not "
         "state, with CheXagent's status and score (two-image question, as in the experiment) and the NIH label."),
        ("How these examples were chosen", f"Selected to show how CheXagent can improve the report, using the labels; "
         f"they are not a representative sample. Of the {n_pairs} pairs, {n_eligible} have at least one condition "
         f"CheXagent found that the report does not state and NIH labels present, and none that NIH does not label. "
         f"The {n_chosen} with the most such conditions are shown (ties: most report conditions CheXagent supports "
         f"that NIH labels, then file name)."),
        (f"Over all {n_pairs} pairs (the full picture)",
         f"Conditions the report states: {stated}. CheXagent supported {s('FOUND', 1) + s('FOUND', 0)} "
         f"({s('FOUND', 1)} labelled present by NIH, {s('FOUND', 0)} not labelled), was indeterminate on "
         f"{s('UNCERTAIN', 1) + s('UNCERTAIN', 0)} and required confirmation on {s('AGAINST', 1) + s('AGAINST', 0)}. "
         f"Conditions CheXagent found that the report does not state ('Consider'): {added}; NIH labels "
         f"{t[('added', 1)]} of them present and {t[('added', 0)]} not ({t[('added', 1)] / added:.0%} correct). "
         f"So 'Consider' rows are prompts for review, not corrections to apply automatically."),
        ("Main result (all pairs, pre-registered)", "Mean AUC over the 9 conditions with at least 10 positives: "
         "MedGemma alone 0.545; MedGemma + CheXagent 0.835 (difference +0.291, 95% CI +0.264 to +0.316); "
         "MedGemma + MAIRA-2 0.673. CheXagent's benefit comes from ranking how likely each condition is, which "
         "the yes/no 'Consider' prompts on these pages show only in part."),
        ("Limitations", "NIH labels are text-mined from reports and can be wrong; they give no location. Base "
         "MedGemma 1.5, not the team's retrained model. CheXagent probably saw NIH images in training. Two images "
         "exceed CheXagent's 2,048-token design length, though its scores track its single-image scores "
         "closely. These pages are an AI research output, not reviewed by a radiologist."),
    ]
    y = pdf.get_y() + 6
    for head, text in paras:
        y = pdf.title_bar(pdf.l_margin, y, W, head)
        pdf.set_xy(pdf.l_margin, y)
        pdf.font(9)
        pdf.multi_cell(W, 4.6, text, align="L")
        y = pdf.get_y() + 5
    pdf.set_xy(pdf.l_margin, pdf.h - 12)
    pdf.font(6.8, color=GREY)
    pdf.cell(W, 3.2, f"Page {pdf.page_no()} of {{nb}}", align="R")


if __name__ == "__main__":
    main()
