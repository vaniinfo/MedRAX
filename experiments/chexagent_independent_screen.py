"""Independent CheXagent screen of the comparison PDF's current images, against the generated
(MedGemma) report and the ground truth.

Rules, fixed before running:
- Conditions: the six Evidence Engine v1.0 findings and the seven provisional NIH findings
  (provisional_findings_v0.json). CheXagent is asked "Is there {condition}?" (the validated
  query, 8 new tokens) on each page's full-resolution current image, without the report.
- CheXagent FOUND a condition if its p_present is at or above the support cut-off: SUPPORTING
  from the engine for the six; SUPPORTING from the provisional cut-offs for the seven (findings
  whose SUPPORTING failed its test-split check, pneumonia and infiltration, can never be found).
  Engine OUTSIDE_SUPPORT above 0.5 counts as found, as in the review tool.
- For each found condition, CheXagent locates it with the side-free phrase; the side of its
  box(es) is reported.
- MedGemma found a condition if the generated report's extracted findings (saved in the review
  run's .json, devices excluded) map to it (validated_name for the six, provisional TERMS for
  the seven).
- Ground truth: present / location, or NOT IN GROUND TRUTH, from the same quote-checked lookup
  as the Finding Comparison table.
- Reported: every condition CheXagent found that MedGemma did not, with the ground-truth
  answer; counts confirmed vs not in ground truth. Also the conditions both found, for context.

    python experiments/chexagent_independent_screen.py PDF --images-dir DIR --review-json RUN.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from pypdf import PdfReader  # noqa: E402

import provisional_findings as P  # noqa: E402
from comparison_edv_pdf import (NOT_IN_GT, SIX, ground_truth_locations, make_extractor,  # noqa: E402
                                parse_page, row_image, validated_name)
from medrax.evidence import EvidenceEngine  # noqa: E402
from medrax.tools.medgemma import describe_location  # noqa: E402

CONDITIONS = SIX + list(P.FINDINGS)


def canonical(finding):
    """Screen condition a stated finding maps to, or None."""
    v = validated_name(finding)
    if v in SIX:
        return v
    return next((q for k, q in sorted(P.TERMS.items(), key=lambda kv: -len(kv[0])) if k in finding.lower()), None)


def found(engine, condition, p):
    if condition in SIX:
        ev = engine.assess(condition, p)
        return ev.state == "SUPPORTING" or (ev.reason == "OUTSIDE_SUPPORT" and p >= 0.5), ev.state
    state = (P.assess(condition, p) or ("UNCERTAIN", condition))[0]
    return state == "SUPPORTING", f"provisional {state}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--review-json", required=True, help="review run whose extracted MedGemma findings are used")
    ap.add_argument("--ollama-model", default="qwen3.8-edv:latest")
    args = ap.parse_args()

    from medrax.tools.xray_vqa import XRayVQATool

    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    engine = EvidenceEngine()
    extractor = make_extractor(args.ollama_model)
    review = {r["title"]: r for r in json.loads(Path(args.review_json).read_text(encoding="utf-8"))}
    pages = []
    for p in PdfReader(args.pdf).pages:
        page = parse_page(p.extract_text())
        image = row_image(args.images_dir, page["row"])
        medgemma = {c for c in map(canonical, review[page["title"]]["findings"]) if c}
        screen = []
        for c in CONDITIONS:
            _, yes_no = tool._generate_response([image], f"Is there {c}?", 8)
            p_present = yes_no["p_yes"] if yes_no else None
            hit, state = found(engine, c, p_present) if p_present is not None else (False, "no yes/no answer")
            screen.append({"condition": c, "p_present": p_present, "state": state, "chexagent_found": hit,
                           "medgemma_found": c in medgemma})
        hits = [s for s in screen if s["chexagent_found"]]
        for s in hits:
            proof, _ = tool.locate_findings(image, [s["condition"]])
            s["chexagent_location"] = "; ".join(
                describe_location(b["box_normalized"]).replace("patient's ", "") for b in proof[0]["boxes"]) \
                if proof[0]["localized"] else "not located"
        gt = ground_truth_locations(extractor, page["truth_text"], [s["condition"] for s in hits])
        for s in hits:
            s["ground_truth"] = gt[s["condition"]]
        pages.append({"title": page["title"], "row": page["row"], "image": image,
                      "medgemma_findings": review[page["title"]]["findings"], "screen": screen})
        print(f"{page['title']}: CheXagent {[(s['condition'], round(s['p_present'], 3)) for s in hits]}; "
              f"MedGemma {sorted(medgemma)}", flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "logs" / f"chexagent_screen_{stamp}"
    out.with_suffix(".json").write_text(json.dumps(pages, indent=1), encoding="utf-8")
    only = [(pg, s) for pg in pages for s in pg["screen"] if s["chexagent_found"] and not s["medgemma_found"]]
    both = [(pg, s) for pg in pages for s in pg["screen"] if s["chexagent_found"] and s["medgemma_found"]]
    confirmed = [x for x in only if x[1]["ground_truth"] != NOT_IN_GT]
    lines = [f"# Independent CheXagent screen ({len(pages)} comparisons)", "",
             f"Rules: see `experiments/chexagent_independent_screen.py`. Review run: `{args.review_json}`.", "",
             f"CheXagent found {len(only) + len(both)} conditions: {len(both)} also in the generated report, "
             f"{len(only)} not. Of the {len(only)} CheXagent-only conditions, {len(confirmed)} are in the "
             "ground truth.", "",
             "## Found by CheXagent, not in the generated report", "",
             "| page | condition | p_present | CheXagent location | ground truth |", "|---|---|---|---|---|"]
    lines += [f"| {pg['title']} | {s['condition']} | {s['p_present']:.3f} | {s['chexagent_location']} | "
              f"{s['ground_truth']} |" for pg, s in only]
    lines += ["", "## Found by both", "", "| page | condition | p_present | CheXagent location | ground truth |",
              "|---|---|---|---|---|"]
    lines += [f"| {pg['title']} | {s['condition']} | {s['p_present']:.3f} | {s['chexagent_location']} | "
              f"{s['ground_truth']} |" for pg, s in both]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}.md / .json")


if __name__ == "__main__":
    main()
