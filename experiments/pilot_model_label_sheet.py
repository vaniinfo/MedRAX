"""Experiment 1 of model_comparison_pilot_protocol.md: read each model's own report for the 13
conditions and export a review sheet for the radiologist (who sees the model report only).

Reports: the team's MedGemma (generated report in the PDF) and the MedGemma 1.5, CheXagent and
MAIRA-2 reports from a pilot_runs.py output directory. Same quote-checked lookup as the
ground-truth sheet. Run once; the sheet (in logs/) is then reviewed, not regenerated.

    python experiments/pilot_model_label_sheet.py PDF RUNS_DIR
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from pypdf import PdfReader  # noqa: E402

from chexagent_independent_screen import CONDITIONS  # noqa: E402
from comparison_edv_pdf import NOT_IN_GT, ground_truth_locations, make_extractor, parse_page  # noqa: E402
from pilot_ground_truth_sheet import side_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("runs_dir")
    ap.add_argument("--ollama-model", default="qwen3.8-edv:latest")
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    reports = {"team_medgemma": {}, "medgemma15": {}, "chexagent": {}, "maira2": {}}
    for p in PdfReader(args.pdf).pages:
        page = parse_page(p.extract_text())
        reports["team_medgemma"][page["title"]] = re.sub(r"\s+", " ", page["generated_text"]).strip()
    for model in ("medgemma15", "chexagent", "maira2"):
        for title, v in json.loads((runs / f"{model}.json").read_text(encoding="utf-8")).items():
            reports[model][title] = v["report"]

    extractor = make_extractor(args.ollama_model)
    out = runs / "model_labels_review.csv"
    columns = ["model", "page", "condition", "auto_label", "auto_side", "auto_location", "auto_evidence",
               "auto_rejected_answer", "radiologist_label", "radiologist_side", "radiologist_comment", "model_report"]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        for model, by_page in reports.items():
            for title, text in by_page.items():
                answers = ground_truth_locations(extractor, text, CONDITIONS, with_evidence=True)
                for i, c in enumerate(CONDITIONS):
                    where, evidence, rejected = answers[c]
                    label = "not stated" if where == NOT_IN_GT else "possible" if "(possible)" in where else "present"
                    location = "" if where == NOT_IN_GT else where.replace("(possible)", "").strip()
                    writer.writerow({
                        "model": model, "page": title.split()[1], "condition": c, "auto_label": label,
                        "auto_side": side_of(location) if location else "", "auto_location": location,
                        "auto_evidence": evidence if where != NOT_IN_GT else "", "auto_rejected_answer": rejected or "",
                        "model_report": text if i == 0 else ""})
                print(f"{model} {title}: {[c for c in CONDITIONS if answers[c][0] != NOT_IN_GT]}", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
