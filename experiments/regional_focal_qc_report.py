"""Writes the QC report for the v2 Regional/Focal visualization PDFs from the recorded QC files.

    .venv/Scripts/python.exe experiments/regional_focal_qc_report.py <stamp>
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pypdf import PdfReader

LOGS = Path(__file__).resolve().parent.parent / "logs"


def main(st):
    r = list(csv.DictReader(open(LOGS / f"regional_focal_qc_findings_v2_{st}.csv", encoding="utf-8")))
    q = json.loads((LOGS / f"regional_focal_qc_pages_v2_{st}.json").read_text(encoding="utf-8"))
    npg = {k: len(PdfReader(str(LOGS / v["pdf"])).pages) for k, v in q.items()}

    def pages(rows):
        d = defaultdict(list)
        for x in rows:
            d[x["report"]].append(int(x["v1_page"]))
        return "; ".join(f"{k} p. {', '.join(map(str, sorted(v)))}" for k, v in sorted(d.items()))

    def same(x):
        o, n = x["v1_reference_status"], x["v2_reference_category"]
        return (x["dataset"].startswith("NIH") or (o.startswith("Present") and n == "explicitly present") or
                (o.startswith("Not mentioned") and n == "not mentioned") or
                (o.startswith("Absent - explicitly") and n == "explicitly absent"))
    chg = [x for x in r if not same(x)]
    groups = defaultdict(list)
    for x in chg:
        groups[(x["v1_reference_status"].split(" (")[0].split(" -")[0], x["v2_reference_category"])].append(x)
    amb = [x for x in r if x["chexagent_panel"].startswith("ALL")]
    grp_a = [x for x in r if x["group"] == "A"]
    cats = Counter(x["v2_reference_category"] for x in r)
    n_orient = sum(len(p["orientation"]) for v in q.values() for p in v["qc"].values())
    n_ok = sum(1 for v in q.values() for p in v["qc"].values() for c in p["orientation"] if c.endswith("OK"))
    n_layout = sum(len(p["layout"]) for v in q.values() for p in v["qc"].values())
    L = [f"# QC report: Regional and Focal findings visualization PDFs ({st})", "",
         "Scope: final quality control of `regional_findings_evaluation_20261108-204046.pdf` and "
         "`focal_findings_evaluation_20261108-204046.pdf` (v1). No model was rerun; no prediction, cache, score, "
         "report or Evidence Engine file was changed. Purpose unchanged: radiologist assessment of visualization "
         "usefulness, not proof of diagnostic accuracy.", "",
         "## New versioned outputs (logs/)",
         f"- `regional_findings_evaluation_v2_{st}.pdf` ({npg['regional']} pages: cover, 74 findings, appendix) + "
         f"`_responses.csv/.json` (schema v2, {q['regional']['questions']} questions)",
         f"- `focal_findings_evaluation_v2_{st}.pdf` ({npg['focal']} pages: cover, 11 findings, appendix) + "
         f"`_responses.csv/.json` (schema v2, {q['focal']['questions']} questions)",
         f"- `regional_focal_qc_findings_v2_{st}.csv`: per finding - v1 page, v1 vs v2 reference status, exact "
         "reference sentences, CheXagent panel type, number of boxed findings on the form page",
         f"- `regional_focal_qc_pages_v2_{st}.json`: per page - layout and orientation checks; form-field checks",
         "- Code: `experiments/reference_sentence_audit.py`, `experiments/regional_focal_reports_v2.py`, this report: "
         "`experiments/regional_focal_qc_report.py`", "",
         "## 1. Reference-report classification (85 findings)",
         "v1 used a negation regex with two states (mentioned / negated) and copied the earlier reader's label, which "
         "turned 'possible' into 'present'. v2 uses a sentence-level rule audit (23 unit sentences, all passing): "
         "explicitly present / explicitly absent / possible-uncertain / not mentioned; clause-scoped and list-carried "
         "negation; laterality-specific negation; each mention judged separately; interval-only negation ('no new X') "
         "= uncertain; trailing hedges ('cannot be excluded'); indirect signs (costophrenic blunting, volume loss, "
         "airspace opacity / pneumonia) = uncertain; 'pericardial effusion' excluded. Exact sentences are quoted on "
         "each page; absence is never inferred from silence.", "",
         "v2 distribution: " + ", ".join(f"{k} {v}" for k, v in cats.most_common()) +
         ". NIH findings have labels only (no report text).", "", f"Changed from v1 ({len(chg)} findings):"]
    for (o, n), xs in sorted(groups.items()):
        L.append(f"- v1 '{o}' -> v2 '{n}': {len(xs)} ({pages(xs)})")
    L += ["", "## 2. CheXagent panels in Group A",
          f"- {len(grp_a)} Group A findings: {sum(1 for x in grp_a if x['chexagent_panel'] == 'recorded box')} with "
          "per-finding recorded boxes (unambiguous); "
          f"{sum(1 for x in grp_a if x['chexagent_panel'].startswith('form image - single'))} form images where this "
          f"was the only finding with a CheXagent box on that page (unambiguous); **{len(amb)} form images showing "
          "boxes of two or more findings** (NIH Forms A/B and Form C: per-finding boxes were never saved).",
          f"- These are now labelled 'ALL ORIGINAL BOXES ... this finding's box cannot be isolated' and the question "
          f"is flagged 'potentially ambiguous comparison': {pages(amb)}.",
          "- The boxes on those images carry small text labels drawn when the forms were built; a reader may identify "
          "the box visually, but this cannot be verified from the records.", "",
          "## 3. Response options",
          f"- The paired question (Group A, {len(grp_a)} findings: every Group A page of v1) gains 'Cannot determine / "
          "abnormality not identifiable', distinct from 'Neither useful'. Response files use schema v2 (options in the "
          "JSON header; each question records its CheXagent panel type and reference category). The Group B question "
          "is unchanged ('Cannot judge' already present).", "",
          "## 4. Orientation and scaling",
          "- v1 had no patient-side labels on any page. v2 places 'R (patient right)' / 'patient left (L)' under every "
          "image panel (standard radiographic display: patient's right on the image left; on the films "
          "spot-checked (regional p. 16, focal p. 2 and earlier gallery pages) the burned-in 'L' marker sits at the "
          "image right, consistent).",
          "- All recorded CheXagent and MAIRA-2 boxes are drawn on the identical current-image array (asserted in code "
          "for every panel) from coordinates normalized to the original image; no rendering path mirrors an image "
          "(the only flip in the project, the vasandesh stability test, is mapped back).",
          f"- Embedded form images: aspect ratio compared with the current image for all {n_orient} panels: "
          f"{n_ok} within 2%.",
          "- vasandesh panel: reproduced from the v1 rendering of the same current image (cropped to the X-ray "
          "area), so orientation and scale are those of the original overlay; no re-inference.", "",
          "## 5. Layout, form fields and readability",
          "- v1 issues: report text truncated at 800-900 characters (content lost); smallest fonts 5.2 pt (reports) "
          "and 5.6-5.8 pt (captions, footers); no automated overlap check.",
          "- v2: full report text (long reports continued in an appendix, page referenced); minimum 6.3 pt for "
          "content, 6 pt for side markers and footers; automated per-page check of text-image, text-text, out-of-page "
          "and field-text overlaps. The check found and v2 fixed: report text beyond the right page edge, report text "
          "running into the 'Reference report' heading, panel captions overlapping neighbouring captions and leaving "
          "the page, and a fragment of the old caption inside the reproduced vasandesh crop. The checker itself had a "
          "bug (left-aligned axes titles were not measured); a deliberate test exposed it and it was fixed before the "
          f"final run. **Final run: {n_layout} layout issues across both PDFs.**",
          f"- Form fields: regional {q['regional']['fields']['found']}/{q['regional']['fields']['fields']}, focal "
          f"{q['focal']['fields']['found']}/{q['focal']['fields']['fields']} present; names unique; dropdown options "
          "match the schema; all inside the page; fill test on a temporary copy "
          f"{q['regional']['fields']['fill_test']} / {q['focal']['fields']['fill_test']}; problems: "
          f"{q['regional']['fields']['problems'] + q['focal']['fields']['problems'] or 'none'}.",
          "- Visual spot checks at 72 dpi (normal zoom) of regional p. 16 and focal p. 2: legible, no overlaps.", "",
          "## 6. Unresolved limitations",
          "- The reference audit is rule-based, not a radiologist reading; it is transparent (sentences quoted) but can "
          "misread unusual phrasing. NIH findings have labels only.",
          f"- {len(amb)} Group A comparisons remain potentially ambiguous (CheXagent box not isolable); analyse their "
          "answers separately or exclude them.",
          "- CheXagent coordinates/locations were not recorded for Form C (only location text for NIH), so CheXagent "
          "side and box overlap cannot be computed for those findings.",
          "- Patient-side labels assume the standard display convention; they were not verified image by image against "
          "burned-in markers (no OCR).",
          "- 6.3 pt text is legible on screen at 100% and on A4 landscape print, but small; A3 printing or 125% zoom "
          "is advisable.",
          "- The vasandesh panel is a raster reproduction of the previous rendering (exploratory only)."]
    out = LOGS / f"regional_focal_QC_report_{st}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(out)
    print("\n".join(L[17:40]))


if __name__ == "__main__":
    main(sys.argv[1])
