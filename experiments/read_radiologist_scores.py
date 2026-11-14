"""Reads the radiologist's 1-5 scores and comments from a filled-in review PDF into a CSV.

    python experiments/read_radiologist_scores.py FILLED.pdf [--fields ORIGINAL.fields.json] [--out scores.csv]

The .fields.json written with the blank PDF maps each field to its page, finding, CheXagent evidence and
ground truth. Unfilled rows are kept with an empty score.
"""

import argparse
import csv
import json
from pathlib import Path

from pypdf import PdfReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--fields", default=None, help="the .fields.json of the blank form (default: next to the PDF)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pdf = Path(args.pdf)
    fields = Path(args.fields) if args.fields else pdf.with_suffix(".fields.json")
    rows = json.loads(fields.read_text(encoding="utf-8"))
    values = {k: (v.get("/V") or "") for k, v in (PdfReader(str(pdf)).get_fields() or {}).items()}
    out = Path(args.out) if args.out else pdf.with_name(pdf.stem + "_scores.csv")
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["page", "finding", "chexagent_evidence", "ground_truth", "score_medgemma_finding",
                    "score_chexagent_evidence", "score_ground_truth_location", "medgemma_change", "score_medgemma_change", "score_chexagent_grounding",
                    "score_bounding_box", "gt_comment", "comments"])
        get = lambda r, k: str(values.get(r["field_prefix"] + "_" + k, "") or "").strip()  # noqa: E731
        for r in rows:
            w.writerow([r["page_title"], r["finding"], r["chexagent_evidence"], r["ground_truth"],
                        get(r, "medgemma"), get(r, "chexagent"), get(r, "groundtruth"),
                        r.get("medgemma_change", ""), get(r, "change"), get(r, "grounding"), get(r, "boundingbox"),
                        r.get("gt_comment", r.get("comment", "")), get(r, "comment")])
    boxes = sum(len(r.get("scored", [])) for r in rows)
    filled = sum(1 for r in rows for k in r.get("scored", []) if get(r, k))
    print(f"{filled} of {boxes} score boxes filled ({len(rows)} rows) -> {out}")


if __name__ == "__main__":
    main()
