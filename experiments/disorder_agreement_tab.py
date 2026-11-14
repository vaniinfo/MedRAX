"""Adds the tab 'Disorder agreement' (professor's format) to the doctor-response workbook, in place.

Columns: Disorders | Events per GT | GT vs Dr. F | GT vs Dr. S | Dr. F vs Dr. S | MedGemma vs Dr. F |
CheXagent vs Dr. S | MedGemma vs CheXagent, plus two extra columns (MedGemma vs Dr. S, CheXagent vs Dr. F).

Unit: one finding = one current image x one condition. Scored rows are read from the workbook's own 'All rows' tab
(columns 'Reference says' and 'Doctor says (from scores)'); dataset + current image is the key, so a case seen on
several forms counts once. findings_pdf() re-reads the returned PDFs instead, for cross-checking.

Presence call per source (present / absent):
  GT          reference report (MIMIC-CXR) or NIH labels, as in 'Reference says' ('possible' counts as present)
  MedGemma    present if MedGemma stated the finding, otherwise absent
  CheXagent   present if CheXagent was shown as Agree (or found the condition: 'CheXagent only'), otherwise absent
  Doctor      the 'Doctor says (from scores)' rule: MedGemma stated -> MedGemma score; CheXagent only -> CheXagent
              score; Missed by both -> MedGemma score reversed. 4-5 = model right, 1-2 = wrong; 3 = uncertain and
              blank = N/A (both excluded from agreement)
Same doctor, same finding on two forms: Dr Franco Form A preferred over the 19-case form (established rule);
Dr Siddharth Form B preferred over the 19-case form (same rule applied by analogy - not yet confirmed).
Consolidation / opacity / infiltrate: merged per image (present if any member row is present).
Cells: % agreement (agreeing / pairs where both calls exist); Cohen's kappa in the second table.

    .venv/Scripts/python.exe experiments/disorder_agreement_tab.py [workbook.xlsx]   (default doctor_responses.xlsx)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import focused_conditions_report as FC  # noqa: E402

BOOK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/datasets/doctor_response/doctor_responses.xlsx")
TAB = "Summary"
OLD_TABS = ("Disorder agreement",)  # earlier name of this tab
GROUPS = [("Pleural effusion", {"pleural effusion"}), ("Pneumothorax", {"pneumothorax"}),
          ("Pneumonia", {"pneumonia"}),
          ("Consolidation / opacity / infiltrate", {"consolidation", "opacities", "lung opacities", "patchy opacity",
                                                    "infiltration"}),
          ("Atelectasis", {"atelectasis"}), ("Cardiomegaly", {"cardiomegaly"}),
          ("Pulmonary edema", {"pulmonary edema"}), ("Low lung volumes", {"low lung volumes"}),
          ("Hyperinflation", {"hyperinflation"}), ("Mass", {"mass"}), ("Nodule", {"nodule"}),
          ("Fibrosis", {"fibrosis"}),
          ("Other (gastric distension, blastic metastases, dextroscoliosis, pulmonary vascular congestion)",
           {"gastric distension", "blastic metastases", "dextroscoliosis", "pulmonary vascular congestion"})]
MERGE = {"Consolidation / opacity / infiltrate"}
EXCLUDED = {"emphysema", "pleural thickening"}  # chest X-ray not reliable for these (team decision)
PAIRS = [("GT vs Dr. F", "GT", "F"), ("GT vs Dr. S", "GT", "S"), ("Dr. F vs Dr. S", "F", "S"),
         ("MedGemma vs Dr. F", "MG", "F"), ("CheXagent vs Dr. S", "CX", "S"), ("MedGemma vs CheXagent", "MG", "CX"),
         ("MedGemma vs Dr. S (extra)", "MG", "S"), ("CheXagent vs Dr. F (extra)", "CX", "F")]


def doctor_call(e):
    s = e["scores"].get("chexagent" if e["row_type"] == "CheXagent only" else "medgemma")
    if s is None:
        return None
    if s == 3:
        return "uncertain"
    says_present = e["row_type"] in ("MedGemma stated", "CheXagent only")
    return ("present" if s >= 4 else "absent") if says_present else ("absent" if s >= 4 else "present")


def score_keys():
    import visualization_evaluation_report as VE
    return {k for k, _ in VE.DIMS}


def findings_pdf():
    """Original source: rows restored from the returned PDFs (kept for cross-checking)."""
    class All:
        def __contains__(self, x):
            return True
    FC.CONDS = All()
    entries, *_ = FC.restore()
    rows = [{"image": e["key"], "cond": e["condition"], "dataset": e["dataset"], "doctor": e["doctor"],
             "form": e["form"], "row_type": e["row_type"], "chexagent_shown": e["chexagent_shown"],
             "gt": "absent" if (e["reference"] or "") in ("No", "", "NOT IN GROUND TRUTH") else "present",
             "call": doctor_call(e), "cx_score": e["scores"].get("chexagent"),
             "mg_score": e["scores"].get("medgemma")} for e in entries]
    return build(rows)


def findings_book(book):
    """Rows from the workbook's 'All rows' tab: 'Reference says' (GT) and 'Doctor says (from scores)'."""
    from openpyxl import load_workbook
    ws = load_workbook(book, read_only=True, data_only=True)["All rows"]
    it = ws.iter_rows(values_only=True)
    H = next(it)
    rows = []
    for v in it:
        r = dict(zip(H, tuple(v) + (None,) * (len(H) - len(v))))
        if r["Row type"] == "No findings" or not r["Condition"]:
            continue
        ds = r["Dataset"]
        # MIMIC earlier set: the 19 comparisons are named 'current.jpg (Comparison k of 19)' on every form
        img = f"{ds}|{r['Current image']}"
        call = r["Doctor says (from scores)"]
        rows.append({"image": img, "cond": r["Condition"], "dataset": ds, "doctor": r["Doctor"], "form": r["Form"],
                     "row_type": r["Row type"], "chexagent_shown": r["CheXagent shown"], "gt": r["Reference says"],
                     "call": call if call in ("present", "absent", "uncertain") else None,
                     "cx_score": r["Score: CheXagent evidence"], "mg_score": r["Score: MedGemma finding"]})
    return build(rows)


def build(rows_in):
    F, info = {}, {"siddharth_dup": 0, "siddharth_dup_conflict": 0, "franco_dup": 0, "franco_dup_conflict": 0}
    for e in rows_in:
        if e["cond"] in EXCLUDED:
            continue
        F.setdefault((e["image"], e["cond"]), []).append(e)
    out = {}
    for k, rows in F.items():
        e0 = rows[0]
        mg = "present" if e0["row_type"] == "MedGemma stated" else "absent"
        cx = "present" if (e0["row_type"] == "CheXagent only" or e0["chexagent_shown"] == "Agree") else "absent"
        calls = {"GT": e0["gt"], "MG": mg, "CX": cx}
        for doc, tag, main in (("Dr Franco", "F", "A"), ("Dr Siddarth", "S", "B")):
            dr = [e for e in rows if e["doctor"] == doc]
            if len(dr) > 1:
                info[f"{'franco' if tag == 'F' else 'siddharth'}_dup"] += 1
                if len({e["call"] for e in dr}) > 1:
                    info[f"{'franco' if tag == 'F' else 'siddharth'}_dup_conflict"] += 1
                dr = sorted(dr, key=lambda e: e["form"] != main)
            calls[tag] = dr[0]["call"] if dr else None
            # model vs doctor: the doctor's own score of that model's output, 3-5 = agrees, 1-2 = disagrees
            for m, fld in (("CXA_", "cx_score"), ("MGA_", "mg_score")):
                sc = dr[0][fld] if dr else None
                calls[m + tag] = None if sc is None else "agree" if sc >= 3 else "disagree"
            calls[tag + "_form"] = dr[0]["form"] if dr else None
        out[k] = {"cond": e0["cond"], "image": k[0], "dataset": e0["dataset"], **calls}
    return out, info


def merge(fs):
    """Per image: present if any member is present; else absent if any absent; else uncertain / missing."""
    by = {}
    for f in fs:
        by.setdefault(f["image"], []).append(f)
    out = []
    for img, g in by.items():
        m = {"image": img, "cond": "+".join(sorted({x["cond"] for x in g}))}
        for t in ("GT", "MG", "CX", "F", "S"):
            v = [x[t] for x in g]
            m[t] = ("present" if "present" in v else "absent" if "absent" in v else
                    "uncertain" if "uncertain" in v else None)
        for t in ("CXA_F", "CXA_S", "MGA_F", "MGA_S"):  # agrees only if the doctor agreed on every scored member row
            v = [x.get(t) for x in g if x.get(t)]
            m[t] = None if not v else "disagree" if "disagree" in v else "agree"
        out.append(m)
    return out


def agree(fs, a, b):
    if a in ("CX", "MG") and b in ("F", "S"):  # direct score: share of findings the doctor rated the model 3-5
        t = ("CXA_" if a == "CX" else "MGA_") + b
        v = [f.get(t) for f in fs if f.get(t)]
        return (sum(x == "agree" for x in v) / len(v), None, len(v)) if v else (None, None, 0)
    pr = [(f[a], f[b]) for f in fs if f[a] in ("present", "absent") and f[b] in ("present", "absent")]
    if not pr:
        return None, None, 0
    n = len(pr)
    po = sum(x == y for x, y in pr) / n
    pa = sum(x == "present" for x, _ in pr) / n
    pb = sum(y == "present" for _, y in pr) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = None if pe == 1 else (po - pe) / (1 - pe)
    return po, kappa, n


def verifier_split(book):
    """Table 2: doctor rating of MedGemma-stated findings, split by the CheXagent verdict shown (Agree / Not agree)."""
    from openpyxl import load_workbook
    ws = load_workbook(book, read_only=True, data_only=True)["All rows"]
    it = ws.iter_rows(values_only=True)
    H = next(it)
    R = [dict(zip(H, tuple(v) + (None,) * (len(H) - len(v)))) for v in it]
    R = [r for r in R if r["Row type"] == "MedGemma stated" and r["Condition"] not in EXCLUDED]
    pct = lambda k, n: f"{100 * k / n:.0f}% ({k}/{n})" if n else "-"  # noqa: E731
    out = []
    for doc, label in (("Dr Franco", "Dr. Franco"), ("Dr Siddarth", "Dr. Siddharth"), (None, "Both doctors (pooled)")):
        F = {}
        for r in R:
            if doc is None or r["Doctor"] == doc:
                F.setdefault((r["Doctor"], r["Dataset"], r["Current image"], r["Condition"]), []).append(r)
        rows = [sorted(v, key=lambda x: x["Form"] not in ("A", "B"))[0] for v in F.values()]
        for cx in ("Agree", "Not agree"):
            g = [x for x in rows if x["CheXagent shown"] == cx]
            sc = [x["Score: MedGemma finding"] for x in g if x["Score: MedGemma finding"] is not None]
            out.append([label, cx, len(sc), pct(sum(v >= 4 for v in sc), len(sc)), pct(sum(v >= 3 for v in sc), len(sc)),
                        pct(sum(x["Reference says"] == "present" for x in g), len(g))])
    return out


def main():
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    F, info = findings_book(BOOK)
    rows = []
    allf = []
    for name, conds in GROUPS:
        fs = [f for f in F.values() if f["cond"] in conds]
        if name in MERGE:
            fs = merge(fs)
        allf += fs
        rows.append((name, fs))
    rows.append(("All disorders", allf))
    wb = load_workbook(BOOK)
    for t in (TAB, *OLD_TABS):
        if t in wb.sheetnames:
            del wb[t]
    ws = wb.create_sheet(TAB, 0)
    for w in wb.worksheets:
        w.sheet_view.tabSelected = w.title == TAB
    wb.active = 0
    head_fill = PatternFill("solid", fgColor="1F3A5F")
    bold = Font(bold=True)

    def header(cols):
        ws.append(cols)
        for c in ws[ws.max_row]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = head_fill
            c.alignment = Alignment(wrap_text=True, vertical="top")

    ws.append(["Agreement on presence, per disorder: reference (GT), Dr. Franco (F), Dr. Siddharth (S), MedGemma, "
               "CheXagent"])
    ws.cell(row=1, column=1).font = Font(bold=True, size=12)
    notes = [
        "Source: this workbook's 'All rows' tab ('Reference says', 'Doctor says (from scores)'). "
        "Unit: one finding = one current image x one condition, counted once even if it appears on several forms. "
        "GT = reference report (MIMIC-CXR) or NIH labels ('possible' = present). MedGemma: present if it stated the "
        "finding. CheXagent: present if shown as Agree or if it found the condition (CheXagent only).",
        "Doctor call from the doctor's score of the model output on that row (rule of the 'Doctor vs reference' tab): "
        "4-5 = model right, 1-2 = wrong; 3 = uncertain and blank = N/A are excluded from agreement. So the doctor "
        "call is not independent of the model shown: a doctor rating a model output as correct agrees with it.",
        "Model vs doctor columns (MedGemma vs Dr. F / Dr. S, CheXagent vs Dr. S / Dr. F): the doctor's own score of "
        "that model's output on that finding is used directly ('Score: MedGemma finding' / 'Score: CheXagent "
        "evidence', which rate what the model showed, including 'Not detected'): 3, 4 or 5 = doctor agrees with the "
        "model, 1-2 = disagrees, blank = excluded (team decision 2026-11-13). Cell = share of scored findings rated "
        "3-5. Consolidation / opacity / infiltrate: agrees only if "
        "the doctor gave 3-5 on every scored member row of that image. GT vs doctor and Dr. F vs Dr. S keep the rule "
        "above (3 = uncertain, excluded).",
        "Same doctor, same finding on two forms: Dr. Franco Form A preferred over the 19-case form (established rule); "
        f"Dr. Siddharth Form B preferred over the 19-case form (applied by analogy, NOT yet confirmed). Duplicates: "
        f"Dr. Franco {info['franco_dup']} ({info['franco_dup_conflict']} with a different call), Dr. Siddharth "
        f"{info['siddharth_dup']} ({info['siddharth_dup_conflict']} with a different call).",
        f"Excluded: {', '.join(sorted(EXCLUDED))} (not reliably assessable on chest X-ray; removed from every row "
        "including 'All disorders'). "
        "Consolidation / opacity / infiltrate merged per image (present if any member row says present). "
        "Cells: % agreement (n = findings where both calls exist). Small n: read as descriptive.",
        "Dr. F and Dr. S scored different forms (A, C / B), overlapping only on the 19-case form and shared cases, so "
        "the Dr. F vs Dr. S column covers only the findings both scored."]
    for t in notes:
        ws.append([t])
        ws.cell(row=ws.max_row, column=1).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=ws.max_row, start_column=1, end_row=ws.max_row, end_column=10)
        ws.row_dimensions[ws.max_row].height = 42
    ws.append([])
    cols = ["Disorders", "Events per GT (present / findings)"] + [p[0] for p in PAIRS]
    ws.append(["1. Percent agreement"])
    ws.cell(row=ws.max_row, column=1).font = bold
    header(cols)
    for name, fs in rows:
        cells = [name, f"{sum(f['GT'] == 'present' for f in fs)} / {len(fs)}"]
        for _, a, b in PAIRS:
            po, _, n = agree(fs, a, b)
            cells.append("-" if po is None else f"{100 * po:.0f}% (n={n})")
        ws.append(cells)
        if name == "All disorders":
            for c in ws[ws.max_row]:
                c.font = bold
    ws.append([])
    ws.append(["2. Does CheXagent's verdict separate right from wrong MedGemma findings?"])
    ws.cell(row=ws.max_row, column=1).font = bold
    for t in [
        "What this answers: the paper's question is not whether CheXagent reads X-rays better than MedGemma (Table 1's "
        "model-vs-doctor columns compare each model on different findings and scoring questions). It is whether an "
        "independent, claim-blinded check tells which MedGemma findings to trust. So: among findings MedGemma stated, "
        "how often did the doctor rate MedGemma's finding correct when CheXagent agreed vs did not agree?",
        "Rows: 'MedGemma stated' only (emphysema, pleural thickening excluded), one row per doctor x finding; same-doctor "
        "duplicates resolved as in Table 1 (Form A / Form B preferred). 'Both doctors' pools the two doctors' rows, so "
        "a finding scored by both counts twice. MedGemma rated correct = doctor's 'Score: MedGemma finding' 4-5; "
        "acceptable = 3-5; blank scores excluded. Present in reference = 'Reference says' present (all rows, so n can "
        "be larger). Exploratory analysis, done after the scores were seen; not pre-registered. Even when CheXagent "
        "agreed, some MedGemma findings were rated wrong: the check flags risk, it does not certify a finding. No "
        "claim of improved diagnostic accuracy."]:
        ws.append([t])
        ws.cell(row=ws.max_row, column=1).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=ws.max_row, start_column=1, end_row=ws.max_row, end_column=10)
        ws.row_dimensions[ws.max_row].height = 58
    header(["Doctor", "CheXagent shown", "MedGemma findings scored (n)", "MedGemma rated correct (4-5)",
            "MedGemma rated acceptable (3-5)", "Finding present in reference"])
    for r in verifier_split(BOOK):
        ws.append(r)
    ws.column_dimensions["A"].width = 44
    ws.column_dimensions["B"].width = 18
    for col in "CDEFGHIJ":
        ws.column_dimensions[col].width = 17
    wb.save(BOOK)
    print(f"tab '{TAB}' written to {BOOK}")
    print(info)
    for name, fs in rows:
        print(f"{name[:36]:36s} GT {sum(f['GT'] == 'present' for f in fs):3d}/{len(fs):3d} | " +
              " | ".join(f"{p[0].split(' (')[0]}: " + ("-" if agree(fs, p[1], p[2])[0] is None else
                         f"{100 * agree(fs, p[1], p[2])[0]:.0f}%/{agree(fs, p[1], p[2])[2]}") for p in PAIRS))


if __name__ == "__main__":
    main()
