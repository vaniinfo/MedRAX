"""Three-way table from an independent CheXagent screen: MedGemma (generated report) | CheXagent
(its own screen, chexagent_independent_screen.py) | ground truth, for the 13 screened conditions.

A row is listed when any of the three says the condition is present. The ground truth is looked
up for all 13 conditions on every page (quote-checked lookup, as in the review tool).

    python experiments/screen_three_way.py SCREEN.json PDF --review-json RUN.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from pypdf import PdfReader  # noqa: E402

from chexagent_independent_screen import CONDITIONS, canonical  # noqa: E402
from comparison_edv_pdf import NOT_IN_GT, ground_truth_locations, make_extractor, parse_page, report_side  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("screen_json")
    ap.add_argument("pdf")
    ap.add_argument("--review-json", required=True)
    ap.add_argument("--ollama-model", default="qwen3.8-edv:latest")
    args = ap.parse_args()

    screen = {p["title"]: p for p in json.loads(Path(args.screen_json).read_text(encoding="utf-8"))}
    review = {r["title"]: r for r in json.loads(Path(args.review_json).read_text(encoding="utf-8"))}
    extractor = make_extractor(args.ollama_model)
    rows, record = [], []
    for p in PdfReader(args.pdf).pages:
        page = parse_page(p.extract_text())
        title = page["title"]
        gt = ground_truth_locations(extractor, page["truth_text"], CONDITIONS)
        stated = {}
        for phrase in review[title]["findings"]:
            c = canonical(phrase)
            if c:
                side = report_side(page["generated_text"], phrase)
                stated[c] = "Present" + (f" ({'bilateral' if side == 'both' else side})" if side else "")
        for s in screen[title]["screen"]:
            c = s["condition"]
            chex = (f"Found ({s['p_present']:.2f}), {s['chexagent_location']}" if s["chexagent_found"]
                    else f"Not found ({s['p_present']:.2f})")
            row = {"page": title, "condition": c, "medgemma": stated.get(c, "Not stated"),
                   "chexagent": chex, "chexagent_found": s["chexagent_found"], "ground_truth": gt[c]}
            record.append(row)
            if c in stated or s["chexagent_found"] or gt[c] != NOT_IN_GT:
                rows.append(row)
        print(f"{title}: GT {({c: v for c, v in gt.items() if v != NOT_IN_GT})}", flush=True)

    out = Path(args.screen_json).with_name(Path(args.screen_json).stem + "_three_way")
    out.with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    lines = ["# MedGemma vs independent CheXagent vs ground truth", "",
             f"Screen: `{args.screen_json}`; generated-report findings from `{args.review_json}`. "
             "Rows where any of the three has the condition.", "",
             "| page | condition | MedGemma | CheXagent (independent) | ground truth |", "|---|---|---|---|---|"]
    number = lambda title: re.search(r"\d+", title).group(0)  # noqa: E731  "Comparison 8 of 19" -> "8"
    lines += [f"| {number(r['page'])} | {r['condition']} | {r['medgemma']} | {r['chexagent']} | "
              f"{r['ground_truth']} |" for r in rows]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}.md / .json")


if __name__ == "__main__":
    main()
