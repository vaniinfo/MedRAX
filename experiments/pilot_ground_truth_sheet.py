"""Step 0 of model_comparison_pilot_protocol.md: ground-truth review sheet for the radiologist.

One row per page and condition (19 x 13): the quote-checked lookup's label (present / possible /
not reported), side, location and quoted evidence, and empty columns for the radiologist's
corrections. The full ground-truth report is on each page's first row. Rows 9-12 are flagged.
Run once; the sheet (in logs/, it quotes MIMIC text) is then reviewed, not regenerated.

    python experiments/pilot_ground_truth_sheet.py PDF
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from pypdf import PdfReader  # noqa: E402

from chexagent_independent_screen import CONDITIONS  # noqa: E402
from comparison_edv_pdf import NOT_IN_GT, ground_truth_locations, make_extractor, parse_page  # noqa: E402

FLAGGED = {9: "ground truth identical to row 10, different image", 10: "ground truth identical to row 9, different image",
           11: "ground truth identical to row 12, different image", 12: "ground truth identical to row 11, different image"}


def side_of(location):
    loc = location.lower()
    if re.search(r"bilateral|both|bibasilar|bibasal|biapical|right.{0,6}(and|;|,).{0,6}left|left.{0,6}(and|;|,).{0,6}right", loc):
        return "bilateral"
    if "right" in loc:
        return "right"
    if "left" in loc:
        return "left"
    return "not stated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--ollama-model", default="qwen3.8-edv:latest")
    args = ap.parse_args()

    extractor = make_extractor(args.ollama_model)
    out = ROOT / "logs" / f"pilot_ground_truth_review_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    columns = ["page", "row_folder_subject", "flag", "condition", "auto_label", "auto_side", "auto_location",
               "auto_evidence", "auto_rejected_answer", "radiologist_label", "radiologist_side", "radiologist_comment",
               "ground_truth_report"]
    with out.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel opens it cleanly
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        for k, p in enumerate(PdfReader(args.pdf).pages, 1):
            page = parse_page(p.extract_text())
            n = int(re.search(r"\d+", page["row"]).group(0))
            answers = ground_truth_locations(extractor, page["truth_text"], CONDITIONS, with_evidence=True)
            for i, c in enumerate(CONDITIONS):
                where, evidence, rejected = answers[c]
                label = ("not reported" if where == NOT_IN_GT else "possible" if "(possible)" in where else "present")
                location = "" if where == NOT_IN_GT else where.replace("(possible)", "").strip()
                writer.writerow({
                    "page": k, "row_folder_subject": page["row"], "flag": FLAGGED.get(n, ""), "condition": c,
                    "auto_label": label, "auto_side": side_of(location) if location else "",
                    "auto_location": location, "auto_evidence": evidence if where != NOT_IN_GT else "",
                    "auto_rejected_answer": rejected or "",
                    "ground_truth_report": re.sub(r"\s+", " ", page["truth_text"]).strip() if i == 0 else ""})
            print(f"page {k}: {[(c, answers[c][0]) for c in CONDITIONS if answers[c][0] != NOT_IN_GT]}", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
