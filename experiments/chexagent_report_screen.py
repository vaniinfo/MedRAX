"""CheXagent's own report (no "Is there X?" questions) for each comparison's current image, read
for the 13 screened conditions, and the final table:
MedGemma (generated report) | CheXagent (own report) | MAIRA-2 (own report) | ground truth.

CheXagent writes a structured report the way its demo does: one request per anatomical area
("Please provide a detailed description of "<area>" in the chest X-ray"), on the full-resolution
current image; the answers are joined. It never sees the generated report or the ground truth.
The conditions are looked up in its report with the same quote-checked lookup as the ground truth.
MedGemma and ground-truth columns come from the three-way run; the MAIRA-2 column from a
maira_findings_screen.py run.

    python experiments/chexagent_report_screen.py THREE_WAY.json MAIRA_FOUR_WAY.json PDF --images-dir DIR
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from pypdf import PdfReader  # noqa: E402

from chexagent_independent_screen import CONDITIONS  # noqa: E402
from comparison_edv_pdf import NOT_IN_GT, ground_truth_locations, make_extractor, parse_page, row_image  # noqa: E402

AREAS = ["View", "Airway", "Breathing", "Cardiac", "Diaphragm",
         "Everything else (e.g., mediastinal contours, bones, soft tissues, tubes, valves, and pacemakers)"]


def chexagent_reports(pdf, images_dir):
    import torch

    from medrax.tools.xray_vqa import XRayVQATool

    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    reports = {}
    for p in PdfReader(pdf).pages:
        page = parse_page(p.extract_text())
        image = row_image(images_dir, page["row"])
        parts = []
        for area in AREAS:
            text, _ = tool._generate_response(
                [image], f'Please provide a detailed description of "{area}" in the chest X-ray', 200)
            parts.append(text.strip())
        reports[page["title"]] = " ".join(parts)
        print(f"{page['title']}: {reports[page['title']]}", flush=True)
    del tool
    torch.cuda.empty_cache()
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("three_way_json")
    ap.add_argument("maira_json", help="four-way .json from maira_findings_screen.py (its 'maira' column is used)")
    ap.add_argument("pdf")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--ollama-model", default="qwen3.8-edv:latest")
    args = ap.parse_args()

    base = Path(args.three_way_json)
    cache = base.with_name(base.stem.replace("_three_way", "") + "_chexagent_reports.json")
    if cache.exists():
        reports = json.loads(cache.read_text(encoding="utf-8"))
    else:
        reports = chexagent_reports(args.pdf, args.images_dir)
        cache.write_text(json.dumps(reports, indent=1), encoding="utf-8")

    extractor = make_extractor(args.ollama_model)
    found = {t: ground_truth_locations(extractor, text, CONDITIONS) for t, text in reports.items()}
    maira = {(r["page"], r["condition"]): r["maira"]
             for r in json.loads(Path(args.maira_json).read_text(encoding="utf-8"))}
    record = json.loads(base.read_text(encoding="utf-8"))
    for r in record:
        where = found[r["page"]][r["condition"]]
        r["chexagent_report"] = "Not stated" if where == NOT_IN_GT else where
        r["maira"] = maira[(r["page"], r["condition"])]

    out = base.with_name(base.stem.replace("_three_way", "_reports_table"))
    out.with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    stated = lambda v: v not in ("Not stated", NOT_IN_GT)  # noqa: E731
    rows = [r for r in record if any(stated(r[k]) for k in ("medgemma", "chexagent_report", "maira", "ground_truth"))]
    number = lambda title: title.split()[1]  # noqa: E731
    lines = ["# Each model's own report vs ground truth", "",
             f"MedGemma: generated report. CheXagent: its own structured report (current image). MAIRA-2: its own "
             f"findings report (`{args.maira_json}`). Rows where any column has the condition.", "",
             "| page | condition | MedGemma | CheXagent | MAIRA-2 | ground truth |", "|---|---|---|---|---|---|"]
    lines += [f"| {number(r['page'])} | {r['condition']} | {r['medgemma']} | {r['chexagent_report']} | {r['maira']} | "
              f"{r['ground_truth']} |" for r in rows]
    positives = [r for r in record if stated(r["ground_truth"])]
    lines += ["", f"Ground-truth conditions caught ({len(positives)}): "
              + ", ".join(f"{name} {sum(stated(r[k]) for r in positives)}"
                          for name, k in (("MedGemma", "medgemma"), ("CheXagent", "chexagent_report"),
                                          ("MAIRA-2", "maira"))),
              "Stated but not in ground truth: "
              + ", ".join(f"{name} {sum(stated(r[k]) and not stated(r['ground_truth']) for r in record)}"
                          for name, k in (("MedGemma", "medgemma"), ("CheXagent", "chexagent_report"),
                                          ("MAIRA-2", "maira"))),
              "", "## CheXagent reports", ""] + [f"- **{t}**: {text}" for t, text in reports.items()]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}.md / .json")


if __name__ == "__main__":
    main()
