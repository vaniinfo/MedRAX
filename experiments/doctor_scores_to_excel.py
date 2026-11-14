"""Returned radiologist forms -> one Excel workbook of the Finding Comparison scores (one row per table row), with
what the doctor saw and CheXagent's underlying score, verdict and cut-offs for later analysis.

Forms (returned PDF, its field map, and where the table rows came from):
- form A: mixed_scoring_form_A_v9 (rows: phase-1 items in logs/mixed_scoring_phase1_v8.pkl, verdicts from the v9 build)
- form C: mimic_heldout_2 (rows rebuilt from logs/mimic_heldout_2_build_cache.json exactly as chexpert_plus_form.build)
Each sheet row is checked against the field map (same page, same row label) before it is written.
Scale: 5 completely correct, 4 mostly correct, 3 partially correct, 2 mostly incorrect, 1 completely incorrect,
blank = N/A (not applicable or cannot be assessed). MIMIC-CXR content: PhysioNet DUA; the workbook is written next to the returned forms (outside the repository).

    .venv/Scripts/python.exe experiments/doctor_scores_to_excel.py
"""

import json
import pickle
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
LOGS = ROOT / "logs"
RESP = Path("D:/datasets/doctor_response")
OUT = RESP / "doctor_responses_20261108-095244.xlsx"  # the working workbook (updated in place)
FORMS = [  # (doctor, form, returned pdf, field map, form record, row source)
    ("Dr Franco", "A", RESP / "DrFranco_scored_formA.pdf", LOGS / "mixed_scoring_form_A_v9.fields.json",
     LOGS / "mixed_scoring_form_A_v9.json", "mixed_A"),
    ("Dr Franco", "C", RESP / "DrFranco_scored_formC.pdf", LOGS / "form_C.fields.json", LOGS / "form_C.json",
     "heldout_2"),
    ("Dr Franco", "MIMIC-19", RESP / "DRFRANCO_scored.pdf",
     LOGS / "radiologist_requested_plus_edv_scoring_form_v26.fields.json",
     LOGS / "radiologist_requested_plus_edv_scoring_form_v26.json", "nineteen"),
    ("Dr Siddarth", "B", RESP / "form_B-2.pdf", LOGS / "mixed_scoring_form_B_v9.fields.json",
     LOGS / "mixed_scoring_form_B_v9.json", "mixed_B"),
    ("Dr Siddarth", "MIMIC-19", RESP / "compress-radiologist_requested_plus_edv_scoring_form_v26.pdf",
     LOGS / "radiologist_requested_plus_edv_scoring_form_v26.fields.json",
     LOGS / "radiologist_requested_plus_edv_scoring_form_v26.json", "nineteen"),
]
STATE = {"Supported": "Supported", "Indeterminate": "Indeterminate", "Requires confirmation": "Evidence against",
         "Consider": "Supported (CheXagent only)", "Missed": "Not detected (below support cut-off)"}
DIMENSIONS = [("medgemma", "Score: MedGemma finding"), ("chexagent", "Score: CheXagent evidence"),
              ("change", "Score: MedGemma change"), ("grounding", "Score: CheXagent grounding")]


def cutoffs():
    import provisional_findings as P
    from comparison_edv_pdf import SIX
    from medrax.evidence import EvidenceEngine
    eng = EvidenceEngine()
    exp = json.loads(P.EXPORT.read_text(encoding="utf-8"))["findings"]
    cache = {}

    def get(c):
        if c in cache:
            return cache[c]
        if c in SIX:
            st = [(x / 1000, eng.assess(c, x / 1000).state) for x in range(1001)]
            # the evidence-against region can end below 0.001 (pulmonary edema): scan the low end finely
            low = [(x / 100000, eng.assess(c, x / 100000).state) for x in range(5001)]
            out = (next((p for p, s in st if s == "SUPPORTING"), None),
                   max((p for p, s in low + st if s == "REFUTING"), default=None), "calibrated (Evidence Engine v1.0)")
        else:
            q = next((v for k, v in sorted(P.TERMS.items(), key=lambda kv: -len(kv[0])) if k in c.lower()), None)
            f = exp.get(q) if q else None
            out = (None, None, "none (not one of the 13 conditions)") if f is None else \
                (f["p_support"] if f["use_supporting"] else None, f["p_refute"] if f["use_refuting"] else None,
                 "provisional")
        cache[c] = out
        return out
    return get


def rows_mixed(form):
    """[(case title, item dict)] in page order, from the phase-1 items and the v9 build's verdicts."""
    forms = pickle.loads((LOGS / "mixed_scoring_phase1_v8.pkl").read_bytes())
    status = {}
    for line in (LOGS / "mixed_v9_build.log").read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(rf"form {form} (Case \d+ of 20)\s.*?: (\[.*\])$", line)
        if m:
            status[m.group(1)] = eval(m.group(2))  # noqa: S307 - our own log, list of (finding, status) tuples
    out = []
    for case in forms[form]:
        title = case["title"].split("  (")[0]
        assert [i["finding"] for i in case["items"]] == [f for f, _ in status[title]], title
        for i, (_, st) in zip(case["items"], status[title]):
            out.append((title, {**i, "status": st, "reference": i.get("ground_truth", ""),
                                "reference_quote": i.get("gt_quote", "")}))
        if not case["items"]:
            out.append((title, None))
    return out


def rows_heldout(n):
    """Rows of a MIMIC held-out form rebuilt from its build cache (same logic as chexpert_plus_form.build)."""
    import comparison_edv_pdf as C
    import edv_improvement as E
    from chexagent_reliability import decide
    from edv_longitudinal_pdf import STATUS
    from medrax.evidence import EvidenceEngine
    engine = EvidenceEngine()
    cache = json.loads((LOGS / f"mimic_heldout_{n}_build_cache.json").read_text(encoding="utf-8"))
    pairs = json.loads((LOGS / f"mimic_heldout_inputs_{n}.json").read_text(encoding="utf-8"))
    reports = {json.loads(x)["current"]: json.loads(x)["report"]
               for x in (LOGS / "mimic_heldout_trained_reports.jsonl").read_text(encoding="utf-8").splitlines()}
    out = []
    for k, p in enumerate(pairs, 1):
        title, r = f"Case {k} of 20", cache[p["current"]]
        stated = [c for c in E.CONDITIONS if r["medgemma"][c][0] != C.NOT_IN_GT]
        items = []
        for c in E.CONDITIONS:
            pp = r["p_present"][c]
            d = decide(engine, c, pp) if pp is not None else "UNCERTAIN"
            ref = r["truth"][c]
            if c in stated:
                side = C.report_side(reports[p["current"]], c)
                word = {"both": "bilateral", "right": "right", "left": "left"}.get(side)
                name = f"{word} {c}" if word else c
                name = name[0].upper() + name[1:] + (" (possible)" if "(possible)" in r["medgemma"][c][0] else "")
                items.append({"finding": c, "name": name, "status": STATUS[d], "p_present": pp,
                              "change": r["change"].get(c), "reference": ref[0], "reference_quote": ref[1]})
            elif d == "FOUND":
                items.append({"finding": c, "name": c, "status": "Consider", "p_present": pp,
                              "reference": ref[0], "reference_quote": ref[1]})
            elif ref[0] != C.NOT_IN_GT:
                items.append({"finding": c, "name": c, "status": "Missed", "p_present": pp,
                              "reference": ref[0], "reference_quote": ref[1]})
        out += [(title, i) for i in items] or [(title, None)]
    return out


def rows_nineteen():
    """Rows of the 19-case MIMIC form (v26): each page's review items then CheXagent-only suggestions."""
    record = json.loads((LOGS / "radiologist_requested_plus_edv_scoring_form_v26.json").read_text(encoding="utf-8"))
    out = []
    for rec in record:
        title = rec["title"]
        items = rec["review"] + rec.get("suggestions", [])
        for i in items:
            out.append((title, {**i, "change": rec.get("change_claims", {}).get(i["finding"]),
                                "reference": i.get("ground_truth", ""), "reference_quote": i.get("gt_quote", "")}))
        if not items:
            out.append((title, None))
    return out


def kappa(pairs, cats=(1, 2, 3, 4, 5)):
    """Linear-weighted Cohen's kappa for paired 1-5 scores."""
    import numpy as np
    k, idx = len(cats), {c: i for i, c in enumerate(cats)}
    m = np.zeros((k, k))
    for x, y in pairs:
        m[idx[x], idx[y]] += 1
    n = m.sum()
    if n == 0:
        return None
    w = 1 - np.abs(np.subtract.outer(range(k), range(k))) / (k - 1)
    po = (w * m).sum() / n
    pe = (w * np.outer(m.sum(1), m.sum(0))).sum() / n ** 2
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement_stats(pairs, seed=20261108, n_boot=2000):
    """n, same category, differ by one category, opposite, weighted kappa [95% bootstrap CI]."""
    import numpy as np
    cat = lambda x: "C" if x >= 4 else "P" if x == 3 else "I"  # noqa: E731
    same = sum(cat(a) == cat(b) for a, b in pairs)
    opp = sum({cat(a), cat(b)} == {"C", "I"} for a, b in pairs)
    kap = kappa(pairs)
    rng = np.random.default_rng(seed)
    boots = [kappa([pairs[i] for i in rng.integers(0, len(pairs), len(pairs))]) for _ in range(n_boot)] if pairs else []
    boots = [b for b in boots if b is not None]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (None, None))
    return len(pairs), same, len(pairs) - same - opp, opp, kap, lo, hi


def doctor_vs_reference(r):
    """(reference says, doctor says, agreement) for one row. The doctor's presence call comes from the score of the
    model output that asserts something about presence on that row:
      MedGemma stated -> MedGemma says present: score 4-5 = present, 1-2 = absent;
      CheXagent only  -> CheXagent says present: CheXagent score 4-5 = present, 1-2 = absent;
      Missed by both  -> both say not detected: MedGemma score 4-5 = absent, 1-2 = present.
    A 3 is 'uncertain'; a blank is 'N/A'. The reference is the radiologist's report (MIMIC-CXR, CheXpert) or the NIH
    labels, as read by the reader model; 'possible' findings count as present."""
    rtype, ref = r[7], r[16]
    if rtype == "No findings" or ref is None:
        return "", "", ""
    ref_present = "absent" if ref in ("No", "") else "present"
    s = r[19] if rtype == "CheXagent only" else r[18]
    if s is None:
        doc = "N/A"
    elif s == 3:
        doc = "uncertain"
    else:
        says_present = rtype in ("MedGemma stated", "CheXagent only")
        doc = ("present" if s >= 4 else "absent") if says_present else ("absent" if s >= 4 else "present")
    agree = "N/A" if doc == "N/A" else "uncertain" if doc == "uncertain" else         "agrees with reference" if doc == ref_present else "disagrees with reference"
    return ref_present, doc, agree


def reference_sheet(wb, all_rows, header):
    """Tab 'Doctor vs reference': where the doctors agreed or disagreed with the case's reference (ground truth)."""
    from collections import Counter
    from openpyxl.styles import Alignment, Font, PatternFill
    ws = wb.create_sheet("Doctor vs reference", 2)
    i_ref, i_doc, i_ag = header.index("Reference says"), header.index("Doctor says (from scores)"),         header.index("Doctor vs reference")
    rows = [r for r in all_rows if r[i_ag]]
    head_fill = PatternFill("solid", fgColor="1F3A5F")

    def head(cols):
        ws.append(cols)
        for c in ws[ws.max_row]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = head_fill
            c.alignment = Alignment(wrap_text=True, vertical="top")

    def summary(label, group):
        c = Counter(r[i_ag] for r in group)
        judged = c["agrees with reference"] + c["disagrees with reference"]
        sub = Counter((r[i_ref], r[i_doc]) for r in group)
        ws.append([label, len(group), c["agrees with reference"], c["disagrees with reference"],
                   f"{100 * c['agrees with reference'] / judged:.0f}%" if judged else "-", c["uncertain"], c["N/A"],
                   sub[("present", "absent")], sub[("absent", "present")]])

    ws.append(["Doctors vs the reference (ground truth) for each case and condition"])
    ws.cell(row=1, column=1).font = Font(bold=True, size=12)
    ws.append(["Reference = the radiologist's report of the current study (MIMIC-CXR) or the NIH labels, as read by the "
               "reader model ('possible' counts as present). The doctor's call on presence is taken from the score of "
               "the model output on that row: MedGemma stated -> MedGemma score; CheXagent only -> CheXagent score; "
               "missed by both -> MedGemma score with the meaning reversed ('Not detected' rated correct = absent). "
               "4-5 = the model was right, 1-2 = wrong, 3 = uncertain, blank = N/A."])
    ws.append([])
    cols = ["Group", "Findings", "Doctor agrees with reference", "Doctor disagrees", "% agree (of judged)",
            "Uncertain (3)", "N/A", "Reference present, doctor says absent", "Reference absent, doctor says present"]
    ws.append(["1. By doctor and form"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    head(cols)
    for key in sorted({(r[0], r[1]) for r in rows}):
        summary(f"{key[0]}, form {key[1]}", [r for r in rows if (r[0], r[1]) == key])
    summary("All forms", rows)
    ws.append([])
    ws.append(["2. By row type (all forms)"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    head(cols)
    for t in ("MedGemma stated", "CheXagent only", "Missed by both"):
        summary(t, [r for r in rows if r[7] == t])
    ws.append([])
    ws.append(["3. By dataset (all forms)"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    head(cols)
    for d in sorted({r[3] for r in rows}):
        summary(d, [r for r in rows if r[3] == d])
    ws.append([])
    ws.append(["4. By condition (all forms)"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    head(cols)
    for cnd in sorted({r[6] for r in rows}, key=lambda c: -sum(r[6] == c for r in rows)):
        summary(cnd, [r for r in rows if r[6] == cnd])
    ws.append([])
    ws.append(["5. Every finding where the doctor disagreed with the reference"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    head(["Doctor", "Form", "Case", "Dataset", "Condition", "Row type", "MedGemma shown", "CheXagent shown",
          "Reference (as read)", "Reference quote", "Reference says", "Doctor says", "Score: MedGemma",
          "Score: CheXagent", "Doctor comment"])
    red = PatternFill("solid", fgColor="F8CBAD")
    for r in rows:
        if r[i_ag] == "disagrees with reference":
            ws.append([r[0], r[1], r[2], r[3], r[6], r[7], r[8], r[9], r[16], r[17], r[i_ref], r[i_doc], r[18], r[19],
                       r[22]])
            for c in ws[ws.max_row][10:12]:
                c.fill = red
    for col, w in zip("ABCDEFGHIJKLMNO", (30, 10, 14, 14, 18, 16, 26, 18, 24, 50, 12, 12, 10, 10, 30)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=4):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")


def agreement_sheet(wb, all_rows, header, groups, pairs):
    """Tab 'Agreement summary': inter-reader (two doctors, same form) and intra-reader (same doctor, repeated case)."""
    from openpyxl.styles import Alignment, Font, PatternFill
    ws = wb.create_sheet("Agreement summary", 1)
    dims = list(range(18, 22))
    title = lambda c: ws.append([c]) or setattr(ws.cell(row=ws.max_row, column=1), "font", Font(bold=True, size=12))  # noqa: E731
    cols = ["Dimension", "Pairs scored by both", "Same category", "% same", "Differ by one category",
            "Opposite (correct vs incorrect)", "Weighted kappa", "95% CI"]

    def table(label_pairs):
        ws.append(cols)
        for c in ws[ws.max_row]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="1F3A5F")
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for name, pr in label_pairs:
            n, same, one, opp, k, lo, hi = agreement_stats(pr)
            ws.append([name, n, same, f"{100 * same / n:.0f}%" if n else "-", one, opp,
                       None if k is None else round(k, 2), "-" if lo is None else f"{lo:.2f} to {hi:.2f}"])

    title("Radiologist scoring agreement")
    ws.append(["Scores grouped as correct (4-5), partial (3), incorrect (1-2). Weighted kappa: linear weights on the "
               "1-5 scale (0 = chance agreement, 1 = perfect); 95% CI from 2,000 bootstrap resamples of findings. "
               "Blanks (N/A) and 'No findings' rows are excluded. Preliminary: one form, one pair of readers."])
    ws.append([])
    if pairs:
        d1, d2 = sorted({d for v in pairs.values() for d in v})
        forms = sorted({k[0] for k in pairs})
        title(f"1. Between radiologists: {d1} vs {d2} (form {', '.join(forms)}; same pages and model outputs)")
        table([(header[j].replace("Score: ", ""), [(v[d1][j], v[d2][j]) for v in pairs.values()
                                                  if v[d1][j] is not None and v[d2][j] is not None]) for j in dims])
        ws.append([])
    title("2. Within each radiologist: the same case scored twice (form A or B and the 19-case form)")
    for doc in sorted({all_rows[i][0] for idx in groups.values() if len(idx) > 1 for i in idx}):
        ws.append([doc])
        ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
        reps = [idx for idx in groups.values() if len(idx) == 2 and all_rows[idx[0]][0] == doc]
        table([(header[j].replace("Score: ", ""), [(all_rows[a][j], all_rows[b][j]) for a, b in reps
                                                  if all_rows[a][j] is not None and all_rows[b][j] is not None])
               for j in dims])
        ws.append([])
    title("What this conveys")
    for t in [
        "The two radiologists fall in the same category on roughly half of the findings, but once chance agreement "
        "is removed their agreement is only slight to fair (weighted kappa 0.13 to 0.37, wide intervals); each "
        "agrees with himself at a similar level on repeated cases. A single reader's score is a noisy reference on "
        "these images.",
        "Disagreement clusters in the change descriptions, laterality (one reader scores the side, the other only "
        "presence), and subtle or non-standard findings (emphysema, low lung volumes, borderline cardiomegaly); "
        "both readers noted limited image resolution.",
        "Limited inter-reader agreement on chest X-ray findings is well documented; it should be reported, and the "
        "evaluation should use a consensus or adjudicated reference rather than one reader. The numbers are small "
        "(9-52 pairs per dimension), so the intervals are wide.",
        "Next: report weighted kappa per dimension; adjudicate disagreements; revise the scoring instructions "
        "(presence and side scored separately; a definition of a correct change; N/A when not assessable); treat "
        "model-accuracy results from single-reader forms as preliminary.",
    ]:
        ws.append(["- " + t])
    ws.column_dimensions["A"].width = 34
    for col, w in zip("BCDEFGH", (12, 11, 8, 12, 14, 11, 14)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows():
        for c in row:
            if c.alignment is None or not c.alignment.wrap_text:
                c.alignment = Alignment(vertical="top")


def main():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from pypdf import PdfReader
    from comparison_edv_pdf import NOT_IN_GT, validated_name

    cut = cutoffs()
    header = ["Doctor", "Form", "Case", "Dataset", "Prior image", "Current image", "Condition", "Row type",
              "MedGemma finding (shown)", "CheXagent shown", "CheXagent verdict (engine)", "CheXagent p_present",
              "Support cut-off", "Evidence-against cut-off", "Cut-off source", "MedGemma change (shown)",
              "Reference (ground truth)", "Reference quote"] + [d for _, d in DIMENSIONS] + ["Doctor comment"]
    all_rows = []
    for doctor, form, pdf, side_path, rec_path, source in FORMS:
        side = json.loads(side_path.read_text(encoding="utf-8"))
        loaded = json.loads(rec_path.read_text(encoding="utf-8"))
        record = {r["case"]: r for r in loaded} if source != "nineteen" else             {int(r["title"].split()[1]): {"dataset": "MIMIC-CXR", "image": r["image"], "source_title": r["title"]}
             for r in loaded}
        values = {k: str(v.get("/V") or "").strip() for k, v in (PdfReader(str(pdf)).get_fields() or {}).items()}
        rows = rows_mixed(source[-1]) if source.startswith("mixed_") else rows_nineteen() if source == "nineteen" else rows_heldout(2)
        assert len(rows) == len(side), (form, len(rows), len(side))
        for (title, item), s in zip(rows, side):
            assert s["page_title"].split("  (")[0].split(" (Row")[0] == title, (form, title, s["page_title"])
            case = int(title.split()[1])
            rec = record[case]
            ds = "NIH ChestX-ray14" if "NIH" in rec["dataset"] else "MIMIC-CXR held-out" if source == "heldout_2" \
                else "MIMIC-CXR (earlier set)"
            prior = rec.get("prior") or "prior.jpg (" + rec.get("source_title", "") + ")"
            current = rec.get("current") or Path(rec.get("image", "")).name + " (" + rec.get("source_title", "") + ")"
            get = lambda k: values.get(s["field_prefix"] + "_" + k, "")  # noqa: E731
            score = lambda k: int(get(k)) if get(k).isdigit() else None  # noqa: E731
            if item is None:
                row = [doctor, form, case, ds, prior, current, "", "No findings", s["finding"], "", "", None, None,
                       None, "", "", "", ""]
            else:
                st = item["status"]
                rtype = {"Consider": "CheXagent only", "Missed": "Missed by both"}.get(st, "MedGemma stated")
                assert (rtype == "CheXagent only") == ("(CheXagent only)" in s["finding"]), (form, title, s["finding"])
                assert (rtype == "Missed by both") == ("(missed by both)" in s["finding"]), (form, title, s["finding"])
                cond = validated_name(item["finding"]) or item["finding"]
                sup, ag, src = cut(cond)
                ch = item.get("change")
                change = "" if rtype != "MedGemma stated" or not ch else \
                    ("No change stated" if ch[0] in ("no change stated", "not mentioned") else
                     f"{ch[0].capitalize()}: \"{ch[1]}\"")
                ref = item.get("reference", "")
                row = [doctor, form, case, ds, prior, current, cond, rtype,
                       item["name"] if rtype == "MedGemma stated" else "Not detected",
                       s["chexagent_evidence"], STATE.get(st, st), item.get("p_present"), sup, ag, src, change,
                       "No" if ref == NOT_IN_GT else ref, item.get("reference_quote", "")]
            row += [score(k) for k, _ in DIMENSIONS] + [get("comment")]
            all_rows.append(row)

    # rows the same doctor scored twice: the same MIMIC comparison, condition and row type in form A and in the
    # 19-case form (6 comparisons appear in both)
    def repeat_key(r):
        if r[1] == "MIMIC-19":
            comp = f"Comparison {r[2]} of 19"
        else:
            m = re.search(r"\((Comparison \d+ of 19)\)", str(r[5]))
            comp = m.group(1) if m and r[1] in ("A", "B") else None
        return (r[0], comp, r[6], r[7]) if comp and r[7] != "No findings" else None
    groups = {}
    for i, r in enumerate(all_rows):
        if repeat_key(r):
            groups.setdefault(repeat_key(r), []).append(i)
    repeat = {}  # row index -> (note, indices of score columns that differ)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        diff = [k for k in range(18, 22) if len({all_rows[i][k] for i in idx}) > 1]
        for i in idx:
            others = "; ".join(f"form {all_rows[j][1]} case {all_rows[j][2]}" for j in idx if j != i)
            repeat[i] = (f"Also scored in {others}: " + ("scores differ (" + ", ".join(
                header[k].replace("Score: ", "") for k in diff) + ")" if diff else "same scores"), diff)
    for i, r in enumerate(all_rows):
        r.append(repeat[i][0] if i in repeat else "")
    header.append("Scored twice by the same doctor")
    for r in all_rows:
        r += list(doctor_vs_reference(r))
    header += ["Reference says", "Doctor says (from scores)", "Doctor vs reference"]

    wb = Workbook()
    readme = wb.active
    readme.title = "Read me"
    notes = [
        "Radiologist scoring of the Finding Comparison tables, one row per table row.",
        f"Built {time.strftime('%Y-%m-%d %H:%M')} by experiments/doctor_scores_to_excel.py from the returned PDFs in {RESP}.",
        "",
        "Score scale (doctor): 5 completely correct; 4 mostly correct; 3 partially correct; 2 mostly incorrect; "
        "1 completely incorrect; blank = N/A (not applicable or cannot be assessed).",
        "",
        "Row type: MedGemma stated = a finding in MedGemma's report; CheXagent only = CheXagent's score passed the "
        "support cut-off for a condition MedGemma did not report (MedGemma shown as 'Not detected'); Missed by both "
        "= the reference contains the condition but neither model reported it; No findings = an empty table.",
        "CheXagent shown = what the doctor saw (Agree / Not agree, the condition name, or Not detected). CheXagent "
        "verdict (engine) = the underlying state: Supported (at or above the support cut-off), Indeterminate "
        "(between the cut-offs), Evidence against (at or below the evidence-against cut-off, or the fallback for "
        "findings outside the 13 conditions). 'Not agree' covers both Indeterminate and Evidence against.",
        "p_present: form A NIH rows = CheXagent with prior + current images; form A MIMIC rows and form C = current "
        "image only. Cut-offs apply to p_present; '-' = no cut-off in use.",
        "Reference = the radiologist's report of the current study (MIMIC-CXR) or the NIH labels, as read by the "
        "local reader model; 'No' = not in the reference.",
        "MIMIC-CXR content is under the PhysioNet data use agreement: do not share outside credentialed people.",
        "",
        "Highlighting: rows the same doctor scored twice (MIMIC comparisons that appear both in form A or B and in "
        "the 19-case form). Light blue row = scored twice with identical scores. Light orange row = scored twice and at least one "
        "score differs; the differing score cells are dark orange. The last column names the other form and case.",
    ]
    for t in notes:
        readme.append([t])
    readme.column_dimensions["A"].width = 160
    for cell in readme["A"]:
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    same_fill = PatternFill("solid", fgColor="DDEBF7")
    diff_fill = PatternFill("solid", fgColor="FCE4D6")
    cell_fill = PatternFill("solid", fgColor="F4B084")

    def sheet(name, rows):
        ws = wb.create_sheet(name)
        ws.append(header)
        for r in rows:
            ws.append(r)
            i = next(k for k, x in enumerate(all_rows) if x is r)
            if i in repeat:
                diff = repeat[i][1]
                for k, c in enumerate(ws[ws.max_row]):
                    c.fill = cell_fill if k in diff else diff_fill if diff else same_fill
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="1F3A5F")
            c.alignment = Alignment(wrap_text=True, vertical="top")
        widths = [10, 6, 6, 20, 26, 26, 18, 16, 28, 18, 26, 10, 10, 12, 22, 40, 22, 50, 10, 10, 10, 10, 30, 40,
                  12, 14, 20]
        for k, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(k)].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        for col in ("L", "M"):
            for c in ws[col][1:]:
                c.number_format = "0.000"
        for c in ws["N"][1:]:  # evidence-against cut-offs can be below 0.001
            c.number_format = "0.0000"
        ws.freeze_panes = "H2"
        ws.auto_filter.ref = ws.dimensions
    sheet("All rows", all_rows)
    for doctor, form, *_ in FORMS:
        sheet(f"{doctor.replace('Dr ', '')} {form}"[:31], [r for r in all_rows if r[0] == doctor and r[1] == form])
    # rows of the same form scored by two different doctors (identical page and model outputs): side by side
    cat = lambda x: None if x is None else "correct" if x >= 4 else "partial" if x == 3 else "incorrect"  # noqa: E731
    by_form = {}
    for r in all_rows:
        if r[7] != "No findings":
            by_form.setdefault((r[1], r[2], r[6], r[7]), {})[r[0]] = r
    pairs = {k: v for k, v in by_form.items() if len(v) == 2}
    if pairs:
        d1, d2 = sorted({d for v in pairs.values() for d in v})
        ws = wb.create_sheet("Two doctors compared")
        dims = [h.replace("Score: ", "") for h in header[18:22]]
        ws.append(["Form", "Case", "Condition", "Row type", "MedGemma finding (shown)", "CheXagent shown"]
                  + [f"{dim}: {d}" for dim in dims for d in (d1, d2)] + [f"{dim}: agreement" for dim in dims]
                  + [f"Comment: {d1}", f"Comment: {d2}"])
        agree_fill = PatternFill("solid", fgColor="E2EFDA")
        part_fill = PatternFill("solid", fgColor="FFF2CC")
        flip_fill = PatternFill("solid", fgColor="F8CBAD")
        for k in sorted(pairs, key=lambda k: (k[0], k[1], k[2])):
            x, y = pairs[k][d1], pairs[k][d2]
            labels = []
            for j in range(18, 22):
                a_, b_ = cat(x[j]), cat(y[j])
                labels.append("" if a_ is None or b_ is None else "same" if a_ == b_ else
                              "OPPOSITE (correct vs incorrect)" if {a_, b_} == {"correct", "incorrect"} else "differ by one category")
            ws.append([k[0], k[1], k[2], k[3], x[8], x[9]] + [v for j in range(18, 22) for v in (x[j], y[j])]
                      + labels + [x[22], y[22]])
            for i, lab in enumerate(labels):
                if lab:
                    ws.cell(row=ws.max_row, column=15 + i).fill =                         agree_fill if lab == "same" else flip_fill if lab.startswith("OPPOSITE") else part_fill
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="1F3A5F")
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for k, w in enumerate([10, 6, 18, 16, 28, 18] + [11] * 8 + [16] * 4 + [30, 30], 1):
            ws.column_dimensions[get_column_letter(k)].width = w
        ws.freeze_panes = "E2"
        ws.auto_filter.ref = ws.dimensions
        ws.append([])
        ws.append(["Summary", "", "", "", "", "", "pairs both scored", "same category", "differ by one", "opposite"])
        for i, dim in enumerate(dims):
            labs = [ws.cell(row=r, column=15 + i).value for r in range(2, 2 + len(pairs))]
            n = sum(1 for v in labs if v)
            ws.append([dim, "", "", "", "", "", n, labs.count("same"), labs.count("differ by one category"),
                       sum(1 for v in labs if v and v.startswith("OPPOSITE"))])
        print(f"two doctors compared: {len(pairs)} rows")
    agreement_sheet(wb, all_rows, header, groups, pairs if pairs else {})
    reference_sheet(wb, all_rows, header)
    out = OUT  # one workbook, updated in place on every run
    wb.save(out)
    filled = sum(r[k] is not None for r in all_rows for k in range(18, 22))
    print(f"scored twice: {len(repeat)} rows ({len(repeat) // 2} pairs), "
          f"{sum(1 for v in repeat.values() if v[1]) // 2} pairs with differing scores")
    print(f"{len(all_rows)} rows ({', '.join(f'{d} {f}: {sum(r[0] == d and r[1] == f for r in all_rows)}' for d, f, *_ in FORMS)}); "
          f"{filled} scores -> {out}")


if __name__ == "__main__":
    main()
