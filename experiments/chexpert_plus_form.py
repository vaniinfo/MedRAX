"""Radiologist scoring form of 20 CheXpert Plus prior + current pairs (Stanford AIMI research use agreement: research
only, not to be redistributed) in the layout of the mixed forms: prior, current and CheXagent-box images; the
radiologist's report of the current study under them; the team's trained MedGemma report; Finding Comparison with
MedGemma finding / CheXagent evidence / MedGemma change / CheXagent grounding (1-5 each) and comments.

Fixed before any report is generated:
- Candidates: logs/chexpert_plus_candidates.json (chexpert_plus_pairs.py, report text only). The form takes the
  first 20 in that order whose trained-MedGemma report states at least one of the 13 conditions and does not
  refuse to compare; the others are spares.
- Trained MedGemma (vaniinfoorg/medgemma_trained @ e346cb15) in its training format with the REAL prior report
  (findings + impression of the prior study) and the current study's clinical history as indication.
- CheXagent: the validated single-image query "Is there {condition}?" on the current image, Evidence Engine v1.0
  verdicts (calibrated six, provisional seven, fallback) - two levels on the form: Supported = Agree, else Not agree.
  Conditions MedGemma does not state but CheXagent finds: "Consider" rows; conditions in the radiologist's report
  that neither model reports: "missed by both" rows. Boxes: CheXagent's location on the current image for each
  finding MedGemma stated.

    .venv/Scripts/python.exe -I experiments/chexpert_plus_form.py prepare
    .venv-medgemma/Scripts/python.exe experiments/chexpert_plus_form.py generate
    .venv/Scripts/python.exe experiments/chexpert_plus_form.py build [suffix]
"""

import json
import re
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = ROOT / "logs"
CANDIDATES = LOGS / "chexpert_plus_candidates.json"
INPUTS = LOGS / "chexpert_plus_inputs.json"
REPORTS = LOGS / "chexpert_plus_trained_reports.jsonl"
CSV = Path("D:/datasets/chexpert_plus/df_chexpert_plus_240401.csv")
IMAGES = Path("D:/datasets/chexpert_plus/images")
MODEL = "D:/model-weights/medgemma_trained_e346cb1"  # vaniinfoorg/medgemma_trained @ e346cb15
N = 20
# dataset name on the pages and file-name stem; mimic_heldout_form.py sets these (and the paths above) for MIMIC
NAME, SLUG = "CheXpert Plus", "chexpert_plus"
SKIP_EMPTY = True  # skip a pair whose report states none of the 13 conditions (False: keep it)
REFUSAL = re.compile(r"no prior chest x-ray|cannot provide|I would need|not available for comparison", re.I)


def section(r, *cols):
    return "\n".join(f"{name}:\n{str(r[c]).strip()}" for c, name in cols if isinstance(r[c], str) and r[c].strip())


def prepare():
    import pandas as pd
    if INPUTS.exists():
        raise SystemExit(f"{INPUTS} exists; not overwriting")
    cands = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    keep = {p for q in cands for p in (q["prior"], q["current"])}
    df = pd.read_csv(CSV, usecols=["path_to_image", "section_clinical_history", "section_history", "section_comparison",
                                   "section_findings", "section_impression"])
    df = df[df["path_to_image"].isin(keep)].set_index("path_to_image")
    out = []
    for q in cands:
        pr, cu = df.loc[q["prior"]], df.loc[q["current"]]
        indication = " ".join(str(cu[c]).strip() for c in ("section_clinical_history", "section_history")
                              if isinstance(cu[c], str) and cu[c].strip())
        out.append({**q, "prior_report": section(pr, ("section_findings", "FINDINGS"), ("section_impression", "IMPRESSION")),
                    "indication": indication,
                    "truth_text": section(cu, ("section_comparison", "COMPARISON"), ("section_findings", "FINDINGS"),
                                          ("section_impression", "IMPRESSION"))})
    INPUTS.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(out)} pairs -> {INPUTS}")


def generate():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    pairs = json.loads(INPUTS.read_text(encoding="utf-8"))
    done = {json.loads(x)["current"] for x in REPORTS.read_text(encoding="utf-8").splitlines()} if REPORTS.exists() else set()
    todo = [p for p in pairs if p["current"] not in done]
    print(f"trained MedGemma: {len(todo)} to go", flush=True)
    if not todo:
        return
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    with REPORTS.open("a", encoding="utf-8") as out:
        for p in todo:
            msgs = [{"role": "user", "content": [
                {"type": "text", "text": "Prior chest X-ray:"},
                {"type": "image", "image": Image.open(IMAGES / p["prior_png"]).convert("RGB")},
                {"type": "text", "text": f"Prior report:\n{p['prior_report']}"},
                {"type": "text", "text": "Current chest X-ray:"},
                {"type": "image", "image": Image.open(IMAGES / p["current_png"]).convert("RGB")},
                {"type": "text", "text": f"Clinical indication for current study: {p['indication']}\n\n"
                                         "Compare the current chest X-ray to the prior chest X-ray. "
                                         "Describe any interval change in findings and provide an impression."}]}]
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt").to("cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                ids = model.generate(**inp, max_new_tokens=512, do_sample=False)
            text = proc.decode(ids[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            out.write(json.dumps({"current": p["current"], "report": text}) + "\n")
            out.flush()
            print(f"{p['current']}: {text[:100]!r}", flush=True)


def build():
    sys.path[:0] = [str(ROOT), str(HERE)]
    import comparison_edv_pdf as C
    import edv_change as X
    import edv_improvement as E
    from chexagent_reliability import decide
    from edv_longitudinal_pdf import STATUS
    from medrax.evidence import EvidenceEngine
    from medrax.tools.medgemma import describe_location
    from mixed_scoring_pdf import COLUMNS, structured_or_none, unload_ollama
    from preview_onepage_report import OnePage
    from types import SimpleNamespace

    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    out = LOGS / f"{SLUG}_scoring_form{suffix}.pdf"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    cache_path = LOGS / f"{SLUG}_build_cache.json"  # reader + CheXagent outputs, reused by later rebuilds
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    save = lambda: cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")  # noqa: E731
    pairs = json.loads(INPUTS.read_text(encoding="utf-8"))
    reports = {json.loads(x)["current"]: json.loads(x)["report"] for x in REPORTS.read_text(encoding="utf-8").splitlines()}
    engine = EvidenceEngine()
    conditions = E.CONDITIONS

    # ---- phase 1: the local reader (Qwen): MedGemma's findings, the radiologist's findings, change wording, template
    extractor = C.make_extractor("qwen3.8-edv:latest")
    chosen = []
    for p in pairs:
        if len(chosen) == N:
            break
        f, report = p["current"], reports[p["current"]]
        r = cache.setdefault(f, {})
        if REFUSAL.search(report):
            print(f"skip {f}: refuses to compare", flush=True)
            continue
        if "medgemma" not in r:
            r["medgemma"] = {c: list(a) for c, a in C.ground_truth_locations(
                extractor, report, conditions, with_evidence=True, extra_rules=E.EXTRA_RULES).items()}
            save()
        stated = [c for c in conditions if r["medgemma"][c][0] != C.NOT_IN_GT]
        if not stated and SKIP_EMPTY:
            print(f"skip {f}: no condition stated", flush=True)
            continue
        if "truth" not in r:
            r["truth"] = {c: list(a) for c, a in C.ground_truth_locations(
                extractor, p["truth_text"], conditions, with_evidence=True, extra_rules=E.EXTRA_RULES).items()}
            save()
        if "change" not in r:
            r["change"] = {c: list(a) for c, a in X.read_claims(extractor, report, stated).items()} if stated else {}
            save()
        page = {"title": f"Case {len(chosen) + 1} of {N}  ({NAME})", "row": "",
                "generated": C.sections(re.sub(r"\s*IMPRESSION:\s*", "\nIMPRESSION:\n", report).strip().splitlines()),
                "truth": C.sections(p["truth_text"].splitlines())}
        if "structured" not in r:
            r["structured"] = structured_or_none({"title": page["title"]}, page, extractor)
            save()
        chosen.append((p, page, stated))
        print(f"prepared {page['title']}: {p['current']}", flush=True)
    unload_ollama("qwen3.8-edv:latest")

    # ---- phase 2: CheXagent (single-image presence query, boxes) and the PDF
    from medrax.tools.xray_vqa import XRayVQATool
    chexagent = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    work = Path(tempfile.mkdtemp(prefix=f"{SLUG}_", dir=ROOT / "temp"))
    pdf = OnePage()
    pdf.radiologist_form, pdf.form_fields = True, []
    pdf.hide_scores, pdf.two_level, pdf.gt_comment = True, True, False
    pdf.form_columns = COLUMNS
    pdf.image_max, pdf.image_gap = 1800, 1.5
    pdf.set_margins(6, 6, 6)
    pdf.collapsible, pdf.toggle_buttons = True, []
    pdf.zoom_targets = []
    titles, record = [], []
    for p, page, stated in chosen:
        f, r = p["current"], cache[p["current"]]
        current, prior = str(IMAGES / p["current_png"]), str(IMAGES / p["prior_png"])
        if "p_present" not in r:
            r["p_present"] = {}
            for c in conditions:
                _, yn = chexagent._generate_response([current], f"Is there {c}?", 8)
                r["p_present"][c] = yn["p_yes"] if yn else None
            save()
        items = []
        for c in conditions:
            pp = r["p_present"][c]
            d = decide(engine, c, pp) if pp is not None else "UNCERTAIN"
            in_truth = r["truth"][c][0] != C.NOT_IN_GT
            if c in stated:
                where = r["medgemma"][c][0]
                side = C.report_side(reports[f], c)
                word = {"both": "bilateral", "right": "right", "left": "left"}.get(side)
                name = (f"{word} {c}" if word else c)
                name = name[0].upper() + name[1:] + (" (possible)" if "(possible)" in where else "")
                items.append({"finding": c, "name": name, "status": STATUS[d], "p_present": pp,
                              "change": r["change"].get(c)})
            elif d == "FOUND":
                items.append({"finding": c, "name": f"{c.capitalize()} (not in report)", "status": "Consider",
                              "p_present": pp})
            elif in_truth:
                items.append({"finding": c, "name": c.capitalize(), "status": "Missed", "p_present": pp})
        proof, _ = chexagent.locate_findings(current, stated) if stated else ([], None)
        boxes = [{"phrase": q["finding"], "box_normalized": b["box_normalized"], "p_present": None}
                 for q in proof if q["localized"] for b in q["boxes"]]
        annotated = C.shrink(C.XRayVQATool._visualize_boxes(chexagent, current, boxes), work, size=1800) \
            if boxes else None
        where = {q["finding"]: "; ".join(dict.fromkeys(describe_location(b["box_normalized"]).replace("patient's ", "")
                                                       for b in q["boxes"])) for q in proof if q["localized"]}
        for i in items:
            if i["status"] not in ("Consider", "Missed") and where.get(i["finding"]):
                i["image_location"] = where[i["finding"]]
            i["tooltip"] = C.evidence_tooltip(engine, i)
        C.draw_page(pdf, page, prior, current, annotated, items, r["structured"], truth_title="Ground Truth Report",
                    legend=C.TWO_LEVEL_LEGEND, gt_under_images=True)
        titles.append(page["title"])
        record.append({"case": len(record) + 1, "dataset": NAME, "prior": p["prior"], "current": f})
        print(f"{page['title']}: {[(i['finding'], i['status']) for i in items]}", flush=True)
    C.add_zoom_pages(pdf)
    pdf.output(str(out))
    C.add_structured_toggle(out, pdf.toggle_buttons)
    n = C.add_form_fields(out, pdf.form_fields, titles)
    C.add_zoom_overlays(out, pdf.zoom_targets)
    out.with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"{len(chosen)} cases, {n} scored rows -> {out}")


if __name__ == "__main__":
    {"prepare": prepare, "generate": generate, "build": build}[sys.argv[1]]()
