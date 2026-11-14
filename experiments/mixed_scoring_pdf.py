"""Two radiologist scoring forms (one per doctor), each with 10 MIMIC-CXR and 10 NIH ChestX-ray14 prior + current
cases in one shuffled PDF, in the layout of the MIMIC scoring form (large X-rays, ground truth under them, MedGemma's
report, structured report behind a show/hide button, Finding Comparison table with 1-5 dropdowns, comments,
mouse-over CheXagent scores, click-to-enlarge X-rays). No case is in both forms.

Selection (labels only, never model outputs), weighted towards the six conditions MedGemma reports reliably
(cardiomegaly, pleural effusion, pneumothorax, consolidation, pulmonary edema, atelectasis):
  - MIMIC: of the 28 distinct cases with images on disk (the 19 of the first form and the 9 new ones of the second
    batch), the 20 whose radiologist report mentions the most of the six conditions (ties broken at random), dealt
    alternately to the two forms.
  - NIH: for each form, at least 2 cases labelled with each of the six conditions where possible (rarest first, at
    random within a condition), filled to 10 with random pairs labelled with any of the six.

    python experiments/mixed_scoring_pdf.py   -> logs/mixed_scoring_form_A.pdf, logs/mixed_scoring_form_B.pdf
"""

import hashlib
import json
import pickle
import random
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import comparison_edv_pdf as C  # noqa: E402
import edv_improvement as E  # noqa: E402
import edv_longitudinal as L  # noqa: E402
import edv_change as X  # noqa: E402
import edv_trained as T  # noqa: E402
from edv_longitudinal_pdf import table  # noqa: E402
from maira_score_nih import NIH  # noqa: E402
from medrax.evidence import EvidenceEngine  # noqa: E402
from medrax.tools.medgemma import describe_location  # noqa: E402
from nih20_scoring_pdf import LEGEND as _NIH_LEGEND  # noqa: E402
from preview_onepage_report import OnePage  # noqa: E402
from pypdf import PdfReader  # noqa: E402

NIH_LEGEND = [x for x in _NIH_LEGEND if x[0] != "Ground truth"]  # no ground-truth note under the table

SEED = 20261102
VISUAL_PROOF = True  # prior, current, and the current X-ray with CheXagent's boxes
# Finding Comparison columns (key, share of the width): no ground-truth or bounding-box column for now
COLUMNS = [("medgemma", 0.23), ("chexagent", 0.18), ("change", 0.25), ("grounding", 0.13), ("comment", 0.21)]
SIX = C.SIX
LOGS = ROOT / "logs"
HELD = Path("D:/datasets/held_out_eval_images")
MIMIC_SOURCES = [  # (source PDF, saved run, CheXagent screen, saved ground-truth lookup)
    (HELD / "subset_comparisons_clean_filtered.pdf", LOGS / "radiologist_requested_plus_edv_scoring_form_v9.json",
     LOGS / "chexagent_screen_20261011-065557.json", LOGS / "chexagent_screen_20261011-065557_three_way.json"),
    (HELD / "subset_comparisons_16_matched.pdf", LOGS / "new16_radiologist_format.json",
     LOGS / "chexagent_screen_20261017-193302.json", LOGS / "chexagent_screen_20261017-193302_three_way.json"),
]


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_mimic():
    """Distinct MIMIC cases (first source wins for duplicates) with everything needed to draw them."""
    cases, seen = [], set()
    for pdf_path, rec_path, screen_path, truth_path in MIMIC_SOURCES:
        reader = PdfReader(str(pdf_path))
        recs = json.loads(rec_path.read_text(encoding="utf-8"))
        screen = {p["title"]: p["screen"] for p in json.loads(screen_path.read_text(encoding="utf-8"))}
        truth = {(r["page"], r["condition"]): r["ground_truth"]
                 for r in json.loads(truth_path.read_text(encoding="utf-8"))}
        for k, rec in enumerate(recs):
            h = md5(rec["image"])
            if h in seen:
                continue
            seen.add(h)
            page = C.parse_page(reader.pages[k].extract_text())
            assert page["title"] == rec["title"], (page["title"], rec["title"])
            n_six = sum(truth.get((rec["title"], c), C.NOT_IN_GT) != C.NOT_IN_GT for c in SIX)
            cases.append(dict(dataset="MIMIC-CXR", rec=rec, page=page, screen=screen, truth=truth, n_six=n_six,
                              source=str(pdf_path.name)))
    return cases


def select_nih(n_per, exclude=()):
    truth, *_ = E.load()
    pairs = sorted((p for p in L.pairs() if p["current"] not in exclude), key=lambda p: p["current"])
    six = {p["current"]: {c for c in SIX if truth[(p["current"], c)]} for p in pairs}
    rng = random.Random(SEED)
    sets, used = {"A": [], "B": []}, set()

    def take(pool, s):
        pool = [p for p in pool if p["current"] not in used]
        if pool:
            p = rng.choice(pool)
            used.add(p["current"])
            sets[s].append(p)

    for c in sorted(SIX, key=lambda c: sum(c in v for v in six.values())):
        for s in ("A", "B") * 2:
            have = sum(c in six[p["current"]] for p in sets[s])
            if have < 2 and len(sets[s]) < n_per:
                take([p for p in pairs if c in six[p["current"]]], s)
    for s in ("A", "B"):
        while len(sets[s]) < n_per:
            take([p for p in pairs if six[p["current"]]], s)
    return sets, truth


def structured_or_none(rec, page, extractor):
    """The template filled from the report; None (no show/hide section on that page) if the sort fails."""
    try:
        C.add_structured(rec, page, extractor)
        return rec["structured"]
    except RuntimeError as e:
        print(f"  structured section skipped: {e}", flush=True)
        return None


def unload_ollama(model):
    """Free the GPU memory held by the local text model before CheXagent is loaded."""
    import urllib.request
    req = urllib.request.Request("http://localhost:11434/api/generate", method="POST",
                                 data=json.dumps({"model": model, "keep_alive": 0}).encode())
    urllib.request.urlopen(req, timeout=60).read()


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""  # e.g. "_v2": new files, earlier ones kept
    outs = {s: LOGS / f"mixed_scoring_form_{s}{suffix}.pdf" for s in ("A", "B")}
    for p in outs.values():
        if p.exists():
            raise SystemExit(f"{p} already exists; not overwriting")
    rng = random.Random(SEED)
    mimic = load_mimic()
    rng.shuffle(mimic)  # random tie-break
    mimic.sort(key=lambda c: -c["n_six"])
    chosen_mimic = {"A": mimic[0:20:2], "B": mimic[1:20:2]}
    nih_sets, truth = select_nih(10)

    engine = EvidenceEngine()
    extractor = C.make_extractor("qwen3.8-edv:latest")
    chexagent = None
    jl = lambda path: {json.loads(x)["current"]: json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()}  # noqa: E731
    chex, reports, labs = jl(L.CHEX_A), jl(T.REPORTS), jl(T.LABELS)  # the team's trained MedGemma
    import csv
    with open("D:/datasets/chestxray14/Data_Entry_2017_v2020.csv", encoding="utf-8") as f:
        meta = {r["Image Index"]: r for r in csv.DictReader(f)}
    lab = lambda im: meta[im]["Finding Labels"].replace("_", " ").replace("|", ", ")  # noqa: E731

    # ---- phase 1: everything that needs the local text model (Qwen), for both forms
    # an earlier run's phase-1 results (argv[2], a .pkl) skip this phase: layout changes rebuild in minutes
    reuse = len(sys.argv) > 2
    forms = pickle.loads(Path(sys.argv[2]).read_bytes()) if reuse else {}
    for s in (() if reuse else ("A", "B")):
        cases = [dict(c) for c in chosen_mimic[s]] + [dict(dataset="NIH ChestX-ray14", pair=p) for p in nih_sets[s]]
        random.Random(f"{SEED}-{s}").shuffle(cases)
        for k, case in enumerate(cases, 1):
            case["title"] = f"Case {k} of {len(cases)}  ({case['dataset']})"
            if case["dataset"] == "MIMIC-CXR":
                rec, page = case["rec"], dict(case["page"])
                C.add_ground_truth(extractor, page, rec)
                C.add_suggestions(extractor, page, rec, case["screen"], case["truth"])
                C.add_ground_truth_quotes(extractor, page, rec)
                case["structured"] = structured_or_none(rec, page, extractor)
                case["items"] = rec["review"] + rec.get("suggestions", [])
                # conditions the radiologist's report contains that neither model reported
                listed = {C.validated_name(i["finding"]) for i in case["items"]}
                miss = [c for c in L.CONDITIONS if c not in listed
                        and case["truth"].get((rec["title"], c), C.NOT_IN_GT) != C.NOT_IN_GT]
                if miss:
                    ans = C.ground_truth_locations(extractor, page["truth_text"], miss, with_evidence=True)
                    p_of = {s_["condition"]: s_["p_present"] for s_ in case["screen"].get(rec["title"], [])}
                    for c in miss:
                        where, evidence, _ = ans[c]
                        case["items"].append({"finding": c, "name": c.capitalize(), "status": "Missed",
                                              "p_present": p_of.get(c),
                                              "ground_truth": case["truth"][(rec["title"], c)],
                                              "gt_quote": evidence.strip() if where != C.NOT_IN_GT else ""})
                page["title"] = case["title"]
                case["page_draw"] = page
            else:
                f, prior = case["pair"]["current"], case["pair"]["prior"]
                rows, _ = table(engine, reports[f]["report"], labs[f]["labels"], chex[f], truth, f)
                listed = {r["finding"] for r in rows}
                for c in L.CONDITIONS:  # labelled by NIH but reported by neither model
                    if truth[(f, c)] and c not in listed:
                        rows.append({"finding": c, "name": c.capitalize(), "status": "Missed",
                                     "p_present": chex[f]["p_present"][c], "ground_truth": "Present (NIH label)"})
                case["items"] = [dict(r, ground_truth="Yes" if r["ground_truth"].startswith("Present") else C.NOT_IN_GT)
                                 for r in rows]
                page = {"title": case["title"], "row": "",
                        "generated": C.sections(re.sub(r"\s*IMPRESSION:\s*", "\nIMPRESSION:\n",
                                                       reports[f]["report"]).strip().splitlines()),
                        "truth": [("", f"Current image ({f}): {lab(f)}."), ("", f"Prior image ({prior}): {lab(prior)}.")]}
                case["structured"] = structured_or_none({"title": case["title"]}, page, extractor)
                case["page_draw"] = page
            # the change MedGemma reports for each finding it stated (quote-checked reader, edv_change.read_claims)
            text = case["page_draw"]["generated_text"] if case["dataset"] == "MIMIC-CXR" \
                else reports[case["pair"]["current"]]["report"]
            stated = [i for i in case["items"] if i.get("status") not in ("Consider", "Missed")]
            if stated:
                try:
                    claims = X.read_claims(extractor, text, sorted({i["finding"] for i in stated}))
                except RuntimeError as e:
                    print(f"  change claims not read: {e}", flush=True)
                    claims = {}
                for i in stated:
                    i["change"] = claims.get(i["finding"])
            print(f"prepared form {s} {case['title']}", flush=True)
        forms[s] = cases

    if not reuse:
        (LOGS / f"mixed_scoring_phase1{suffix}.pkl").write_bytes(pickle.dumps(forms))

    # ---- phase 2: free the GPU, then CheXagent boxes and the PDFs
    unload_ollama("qwen3.8-edv:latest")
    if VISUAL_PROOF:
        from medrax.tools.xray_vqa import XRayVQATool
        chexagent = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    for s, cases in forms.items():
        work = Path(tempfile.mkdtemp(prefix=f"mixed_{s}_", dir=ROOT / "temp"))
        drawer = SimpleNamespace(temp_dir=work)
        pdf = OnePage()
        pdf.radiologist_form, pdf.form_fields = True, []
        pdf.hide_scores, pdf.two_level = True, True
        pdf.form_columns = COLUMNS  # includes MedGemma's reported change, graded 1-5
        pdf.image_max, pdf.image_gap = 1800, 1.5
        pdf.set_margins(6, 6, 6)
        pdf.collapsible, pdf.toggle_buttons = True, []
        pdf.zoom_targets = []
        titles, record = [], []
        for k, case in enumerate(cases, 1):
            items, page = case["items"], case["page_draw"]
            if case["dataset"] == "MIMIC-CXR":
                rec = case["rec"]
                folder = Path(rec["image"]).parent
                prior, current = str(folder / "prior.jpg"), str(folder / "current.jpg")
                boxes = [{"phrase": v["finding"], "box_normalized": b["box_normalized"], "p_present": b["p_present"],
                          "hide_score": True}
                         for v in rec["visual_proof"] if not C.is_device(v["finding"]) for b in v.get("boxes", [])]
                annotated = C.shrink(C.XRayVQATool._visualize_boxes(drawer, current, boxes), work, size=1800) \
                    if boxes and VISUAL_PROOF else None
                for i in items:
                    i["tooltip"] = C.evidence_tooltip(engine, i)
                pdf.gt_comment = True
                C.draw_page(pdf, page, prior, current, annotated, items, case["structured"],
                            legend=C.TWO_LEVEL_LEGEND, gt_under_images=True)
                record.append({"case": k, "dataset": "MIMIC-CXR", "source": case["source"],
                               "source_title": rec["title"], "image": rec["image"]})
            else:
                f, prior = case["pair"]["current"], case["pair"]["prior"]
                annotated = None
                # boxes for the findings MedGemma reported (as on the MIMIC pages)
                conds = [i["finding"] for i in items if i.get("status") not in ("Consider", "Missed")]
                if conds and VISUAL_PROOF:  # CheXagent's boxes on the current image (location only)
                    proof, _ = chexagent.locate_findings(str(NIH / f), conds)
                    boxes = [{"phrase": q["finding"], "box_normalized": b["box_normalized"], "p_present": None}
                             for q in proof if q["localized"] for b in q["boxes"]]
                    annotated = C.shrink(C.XRayVQATool._visualize_boxes(chexagent, str(NIH / f), boxes), work,
                                         size=1800) if boxes else None
                    where = {q["finding"]: "; ".join(dict.fromkeys(describe_location(b["box_normalized"])
                                                                   .replace("patient's ", "") for b in q["boxes"]))
                             for q in proof if q["localized"]}
                    for i in items:
                        if where.get(i["finding"]):
                            i["image_location"] = where[i["finding"]]
                for i in items:
                    i["tooltip"] = C.evidence_tooltip(engine, i)
                pdf.gt_comment = False
                C.draw_page(pdf, page, str(NIH / prior), str(NIH / f), annotated, items, case["structured"],
                            truth_title="Ground Truth (NIH labels)", legend=NIH_LEGEND, gt_under_images=True)
                record.append({"case": k, "dataset": "NIH ChestX-ray14", **case["pair"], "nih_labels": lab(f)})
            titles.append(case["title"])
            print(f"form {s} {case['title']}: {[(i['finding'], i['status']) for i in items]}", flush=True)
        C.add_zoom_pages(pdf)
        pdf.output(str(outs[s]))
        C.add_structured_toggle(outs[s], pdf.toggle_buttons)
        n = C.add_form_fields(outs[s], pdf.form_fields, titles)
        C.add_zoom_overlays(outs[s], pdf.zoom_targets)
        outs[s].with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"form {s}: {len(cases)} cases ({sum(c['dataset'] == 'MIMIC-CXR' for c in cases)} MIMIC), "
              f"{n} scored rows -> {outs[s]}", flush=True)

if __name__ == "__main__":
    main()
