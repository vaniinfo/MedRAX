"""Adds five analysis tabs to the doctor-response workbook (updated in place, as requested; no new Excel file):

  Cond - model vs doctors     per condition, how the radiologists rated each model output (correct 4-5 / partial 3 /
                              incorrect 1-2), per doctor and both together (every rating event)
  Cond - model vs reference   per condition, MedGemma and CheXagent against the reference report (unique findings)
                              and against the doctor's own call (rating events)
  CheXagent boxes             grounding scores per condition and doctor; box side vs reference side and vs MedGemma
  Side errors                 every MedGemma finding with a stated side compared with the reference side
  Side corrections            MedGemma's side contradicted by the reference or by CheXagent (box side / Evidence
                              Engine side statement), with each doctor's verdict

Nothing in the existing tabs, scores or comments is changed. Sources: the workbook's 'All rows' tab, plus the form
records (doctor_scores_to_excel.FORMS) for MedGemma's stated side, CheXagent's box location and the Evidence Engine
statement (Form C did not record box locations).

    .venv/Scripts/python.exe experiments/doctor_scores_analysis_tabs.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import doctor_scores_to_excel as D  # noqa: E402

TABS = ["Cond - model vs doctors", "Cond - model vs reference", "CheXagent boxes", "Side errors", "Side corrections"]
DOCS = ("Dr Franco", "Dr Siddarth")
SC = {"medgemma": "Score: MedGemma finding", "chexagent": "Score: CheXagent evidence",
      "change": "Score: MedGemma change", "grounding": "Score: CheXagent grounding"}


def cat(s):
    return None if s is None else "correct" if s >= 4 else "partial" if s == 3 else "incorrect"


def side_of_text(t):
    """'right' / 'left' / 'both' / None from free text."""
    t = (t or "").lower()
    if re.search(r"\b(bilateral|both)\b", t):
        return "both"
    r, l = bool(re.search(r"\bright\b|\brt\b", t)), bool(re.search(r"\bleft\b|\blt\b", t))
    return "both" if r and l else "right" if r else "left" if l else None


def box_side(loc):
    if not loc:
        return None
    sides = set()
    for seg in loc.split(";"):
        s = seg.strip().lower()
        sides |= {"right", "left"} if s.startswith("both") else {"right"} if s.startswith("right") else \
            {"left"} if s.startswith("left") else set()
    return "both" if sides == {"right", "left"} else sides.pop() if sides else None


def compare(a, b):
    if a is None or b is None:
        return "not assessable"
    if a == b:
        return "same side"
    if "both" in (a, b):
        return "partly different (bilateral vs one side)"
    return "OPPOSITE side"


def extras():
    """(doctor, form, case, condition, row type) -> {report_side, box_location, engine_text}."""
    from comparison_edv_pdf import validated_name
    out = {}
    for doctor, form, pdf, side_path, rec_path, source in D.FORMS:
        rows = D.rows_mixed(source[-1]) if source.startswith("mixed_") else D.rows_nineteen() \
            if source == "nineteen" else D.rows_heldout(2)
        for title, item in rows:
            if item is None:
                continue
            rtype = {"Consider": "CheXagent only", "Missed": "Missed by both"}.get(item["status"], "MedGemma stated")
            cond = validated_name(item["finding"]) or item["finding"]
            out[(doctor, form, int(title.split()[1]), cond, rtype)] = {
                "report_side": item.get("report_side"), "box_location": item.get("image_location"),
                "engine_text": item.get("text") or "", "box_recorded": source != "heldout_2"}
    return out


def style(ws, widths=None):
    from openpyxl.styles import Alignment, Font, PatternFill
    for row in ws.iter_rows():
        for c in row:
            if isinstance(c.value, str) and c.value.startswith("#H "):
                c.value = c.value[3:]
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="1F3A5F")
                c.alignment = Alignment(wrap_text=True, vertical="top")
            elif isinstance(c.value, str) and c.value.startswith("#T "):
                c.value = c.value[3:]
                c.font = Font(bold=True, size=12)
            elif isinstance(c.value, str) and c.value.startswith("#N "):
                c.value = c.value[3:]
                c.font = Font(italic=True, color="555555")
    for i, w in enumerate(widths or [], 1):
        from openpyxl.utils import get_column_letter
        ws.column_dimensions[get_column_letter(i)].width = w


def main():
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill
    wb = load_workbook(D.OUT)
    ws = wb["All rows"]
    rows = list(ws.iter_rows(values_only=True))
    H = {h: i for i, h in enumerate(rows[0])}
    X = extras()
    R = []
    for r in rows[1:]:
        if r[H["Row type"]] == "No findings" or not r[H["Condition"]]:
            continue
        d = {h: r[i] for h, i in H.items()}
        d.update(X.get((d["Doctor"], d["Form"], d["Case"], d["Condition"], d["Row type"]), {}))
        ref = str(d["Reference (ground truth)"] or "")
        d["ref_present"] = not (ref in ("", "No") or ref.startswith("NOT IN"))
        d["ref_side"] = side_of_text(ref) if d["ref_present"] else None
        d["mg_side"] = d.get("report_side") or (side_of_text(d["MedGemma finding (shown)"])
                                                if d["Row type"] == "MedGemma stated" else None)
        d["bx_side"] = box_side(d.get("box_location"))
        m = re.search(r"supports the finding on the (\w+) rather than the reported (\w+)", d.get("engine_text", ""))
        d["engine_side"] = m.group(1) if m else None
        d["fkey"] = (d["Dataset"], d["Current image"], d["Condition"], d["Row type"])
        R.append(d)
    conds = sorted({d["Condition"] for d in R}, key=lambda c: (-sum(d["Condition"] == c for d in R), c))
    for t in TABS:
        if t in wb.sheetnames:
            del wb[t]
    note_rep = ("#N Rating events: every scored row of every form. The 19-case comparisons that also appear in Form A "
                "(Dr Franco) or Form B (Dr Siddarth) are therefore rated twice by the same doctor and counted twice.")

    # ---------------- 1. per condition: model vs doctors
    s = wb.create_sheet(TABS[0])
    s.append(["#T Per condition: how the radiologists rated each model output"])
    s.append(["#N Correct = 4-5, partial = 3, incorrect = 1-2; blank (N/A) excluded. 'MedGemma stated' rates a "
              "positive MedGemma finding; 'MedGemma not detected' rates MedGemma's silence on a condition CheXagent "
              "found or the reference contains; 'CheXagent output' rates CheXagent's verdict on every row type."])
    s.append([note_rep])
    dims = [("MedGemma stated finding", "medgemma", lambda d: d["Row type"] == "MedGemma stated"),
            ("MedGemma 'not detected'", "medgemma", lambda d: d["Row type"] != "MedGemma stated"),
            ("CheXagent output", "chexagent", lambda d: True),
            ("MedGemma interval change", "change", lambda d: True),
            ("CheXagent grounding (box)", "grounding", lambda d: True)]
    head = ["#H Condition", "#H Doctor"]
    for lab, _, _ in dims:
        head += [f"#H {lab}: n", f"#H {lab}: % correct", f"#H {lab}: % partial", f"#H {lab}: % incorrect"]
    s.append([])
    s.append(head)
    for c in conds + ["ALL CONDITIONS"]:
        for doc in DOCS + ("Both doctors",):
            sub = [d for d in R if (c == "ALL CONDITIONS" or d["Condition"] == c) and
                   (doc == "Both doctors" or d["Doctor"] == doc)]
            if not sub:
                continue
            line = [c, doc.replace("Dr Siddarth", "Dr Siddharth")]
            for lab, k, f in dims:
                v = [cat(d[SC[k]]) for d in sub if f(d) and d[SC[k]] is not None]
                n = len(v)
                line += [n] + ([round(100 * v.count(x) / n) for x in ("correct", "partial", "incorrect")] if n
                               else ["", "", ""])
            s.append(line)
    style(s, [20, 14] + [8, 9, 9, 9] * len(dims))
    s.freeze_panes = "C6"

    # ---------------- 2. per condition: model vs reference / doctor call
    s = wb.create_sheet(TABS[1])
    s.append(["#T Per condition: model output vs the reference report, and vs the doctor's own call"])
    s.append(["#N A. Unique findings (each case + condition counted once). Only conditions that appear on the forms "
              "are counted (a MedGemma finding, a CheXagent-only finding or a reference finding both missed), so true "
              "negatives are not available: sensitivity and precision only. 'Not in reference' includes 'not "
              "mentioned' (absence not established), so precision is a lower bound."])
    s.append(["#N MedGemma positive = stated in its report. CheXagent positive = verdict Supported (incl. CheXagent "
              "only); Indeterminate / Evidence against / Not detected = negative."])
    s.append([])
    s.append(["#H Condition", "#H Unique findings", "#H In reference", "#H MedGemma TP", "#H MedGemma FP (not in ref)",
              "#H MedGemma FN", "#H MedGemma sensitivity", "#H MedGemma precision", "#H CheXagent TP",
              "#H CheXagent FP (not in ref)", "#H CheXagent FN", "#H CheXagent sensitivity", "#H CheXagent precision"])
    uniq = {}
    for d in R:
        uniq.setdefault(d["fkey"], d)

    def counts(items, pos):
        tp = sum(1 for d in items if pos(d) and d["ref_present"])
        fp = sum(1 for d in items if pos(d) and not d["ref_present"])
        fn = sum(1 for d in items if not pos(d) and d["ref_present"])
        return tp, fp, fn, (f"{100 * tp / (tp + fn):.0f}%" if tp + fn else ""), \
            (f"{100 * tp / (tp + fp):.0f}%" if tp + fp else "")
    mg_pos = lambda d: d["Row type"] == "MedGemma stated"  # noqa: E731
    cx_pos = lambda d: str(d["CheXagent verdict (engine)"]).startswith("Supported")  # noqa: E731
    for c in conds + ["ALL CONDITIONS"]:
        it = [d for d in uniq.values() if c == "ALL CONDITIONS" or d["Condition"] == c]
        s.append([c, len(it), sum(d["ref_present"] for d in it), *counts(it, mg_pos), *counts(it, cx_pos)])
    s.append([])
    s.append(["#N B. Against the doctor's call (rating events; 'Doctor says (from scores)': present / absent; "
              "uncertain and N/A excluded). Agreement = the model's positive/negative output matches the doctor's "
              "call."])
    s.append([note_rep])
    s.append(["#H Condition", "#H Doctor", "#H Calls (present/absent)", "#H Doctor says present",
              "#H MedGemma agrees with doctor", "#H MedGemma % agree", "#H MedGemma missed (doctor present, not "
              "stated)", "#H MedGemma over-called (doctor absent, stated)", "#H CheXagent agrees with doctor",
              "#H CheXagent % agree", "#H CheXagent missed", "#H CheXagent over-called"])
    for c in conds + ["ALL CONDITIONS"]:
        for doc in DOCS + ("Both doctors",):
            it = [d for d in R if (c == "ALL CONDITIONS" or d["Condition"] == c) and
                  (doc == "Both doctors" or d["Doctor"] == doc) and d["Doctor says (from scores)"] in ("present",
                                                                                                    "absent")]
            if not it:
                continue
            pres = [d["Doctor says (from scores)"] == "present" for d in it]
            line = [c, doc.replace("Dr Siddarth", "Dr Siddharth"), len(it), sum(pres)]
            for pos in (mg_pos, cx_pos):
                ag = sum(1 for d, p in zip(it, pres) if pos(d) == p)
                line += [ag, f"{100 * ag / len(it):.0f}%", sum(1 for d, p in zip(it, pres) if p and not pos(d)),
                         sum(1 for d, p in zip(it, pres) if not p and pos(d))]
            s.append(line)
    style(s, [20, 14, 12, 12, 14, 12, 14, 14, 12, 14, 12, 14, 14])

    # ---------------- 3. CheXagent boxes
    s = wb.create_sheet(TABS[2])
    s.append(["#T CheXagent bounding boxes"])
    s.append(["#N A. Radiologist grounding scores per condition (rating events; correct 4-5 / partial 3 / incorrect "
              "1-2). A grounding score was asked only where CheXagent placed a box for a MedGemma finding."])
    s.append([note_rep])
    s.append(["#H Condition", "#H Doctor", "#H Boxes rated", "#H Mean score", "#H % correct", "#H % partial",
              "#H % incorrect"])
    for c in conds + ["ALL CONDITIONS"]:
        for doc in DOCS + ("Both doctors",):
            v = [d[SC["grounding"]] for d in R if (c == "ALL CONDITIONS" or d["Condition"] == c) and
                 (doc == "Both doctors" or d["Doctor"] == doc) and d[SC["grounding"]] is not None]
            if v:
                cs = [cat(x) for x in v]
                s.append([c, doc.replace("Dr Siddarth", "Dr Siddharth"), len(v), round(sum(v) / len(v), 2),
                          *[round(100 * cs.count(x) / len(v)) for x in ("correct", "partial", "incorrect")]])
    s.append([])
    s.append(["#N B. Box side (unique MedGemma findings with a recorded box; Form C boxes were not recorded). Box side "
              "from CheXagent's box location; reference side from the reference report (often not stated)."])
    s.append(["#H Condition", "#H Findings with a recorded box", "#H Box vs reference: same side",
              "#H Box vs reference: partly different", "#H Box vs reference: OPPOSITE",
              "#H Box vs reference: not assessable", "#H Box vs MedGemma: same", "#H Box vs MedGemma: partly different",
              "#H Box vs MedGemma: OPPOSITE", "#H Box vs MedGemma: not assessable"])
    boxed = [d for d in uniq.values() if d["Row type"] == "MedGemma stated" and d["bx_side"]]
    for c in sorted({d["Condition"] for d in boxed}) + ["ALL CONDITIONS"]:
        it = [d for d in boxed if c == "ALL CONDITIONS" or d["Condition"] == c]
        a = [compare(d["bx_side"], d["ref_side"]) for d in it]
        b = [compare(d["bx_side"], d["mg_side"]) for d in it]
        keys = ("same side", "partly different (bilateral vs one side)", "OPPOSITE side", "not assessable")
        s.append([c, len(it)] + [a.count(k) for k in keys] + [b.count(k) for k in keys])
    style(s, [20, 14, 12, 12, 12, 12, 12, 12, 12, 12])

    # ---------------- 4. side errors (unique findings)
    s = wb.create_sheet(TABS[3])
    s.append(["#T Side (laterality) errors: MedGemma's stated side vs the reference side"])
    s.append(["#N Unique MedGemma findings that state a side (left / right / bilateral). Reference side from the "
              "reference report as read by the reader model; 'not assessable' when the reference does not state a "
              "side or does not contain the finding. Doctor columns: the doctor's MedGemma-finding score and comment "
              "(all forms in which that doctor saw the finding)."])
    s.append([])
    hd = ["#H Dataset", "#H Case", "#H Condition", "#H MedGemma finding", "#H MedGemma side", "#H Reference",
          "#H Reference quote", "#H Reference side", "#H MedGemma vs reference"]
    for doc in DOCS:
        dn = doc.replace("Dr Siddarth", "Dr Siddharth")
        hd += [f"#H {dn}: MedGemma score", f"#H {dn}: comment"]
    s.append(hd)
    by_f = defaultdict(list)
    for d in R:
        by_f[d["fkey"]].append(d)
    side_rows = sorted((k for k, v in by_f.items() if v[0]["Row type"] == "MedGemma stated" and v[0]["mg_side"]),
                       key=lambda k: (compare(by_f[k][0]["mg_side"], by_f[k][0]["ref_side"]) != "OPPOSITE side", k))
    opp_fill = PatternFill("solid", fgColor="F8CBAD")
    part_fill = PatternFill("solid", fgColor="FFF2CC")
    for k in side_rows:
        v = by_f[k]
        d0 = v[0]
        comp = compare(d0["mg_side"], d0["ref_side"])
        line = [d0["Dataset"], d0["Current image"], d0["Condition"], d0["MedGemma finding (shown)"], d0["mg_side"],
                d0["Reference (ground truth)"], d0["Reference quote"], d0["ref_side"] or "-", comp]
        for doc in DOCS:
            dv = [x for x in v if x["Doctor"] == doc]
            line += [", ".join(f"{x[SC['medgemma']]} ({x['Form']})" for x in dv if x[SC["medgemma"]] is not None) or "-",
                     " | ".join(f"{x['Doctor comment']} ({x['Form']})" for x in dv if x["Doctor comment"]) or ""]
        s.append(line)
        if comp == "OPPOSITE side":
            for cc in s[s.max_row]:
                cc.fill = opp_fill
        elif comp.startswith("partly"):
            for cc in s[s.max_row]:
                cc.fill = part_fill
    s.append([])
    comps = [compare(by_f[k][0]["mg_side"], by_f[k][0]["ref_side"]) for k in side_rows]
    s.append(["Summary", f"{len(side_rows)} MedGemma findings with a side", "", "", "", "", "", "",
              "; ".join(f"{c}: {comps.count(c)}" for c in ("same side", "partly different (bilateral vs one side)",
                                                           "OPPOSITE side", "not assessable"))])
    style(s, [16, 30, 16, 26, 10, 22, 40, 11, 22, 14, 30, 14, 30])
    s.freeze_panes = "D5"

    # ---------------- 5. side corrections
    s = wb.create_sheet(TABS[4])
    s.append(["#T Side corrections: MedGemma's side contradicted by the reference or by CheXagent, and what the "
              "doctors said"])
    s.append(["#N Included: unique MedGemma findings with a stated side where the Evidence Engine stated 'supports the "
              "finding on the X rather than the reported Y side', CheXagent's box lies only on the other side, or the "
              "reference side differs from MedGemma's. CheXagent's side = the Engine statement, else the box side; a "
              "box covering both sides is not a correction of a one-sided finding."])
    s.append(["#N Doctor's side verdict (per form): agrees with CheXagent = a comment naming CheXagent's side (and "
              "not MedGemma's), or MedGemma-finding score 1-2 with CheXagent-evidence or grounding score 4-5; agrees "
              "with MedGemma = MedGemma 4-5 with CheXagent 1-2; comments are NOT interpreted automatically "
              "(free text such as 'on left is included but not right' is easily misread): they are shown verbatim and flagged 'read manually' when they name a side; otherwise mixed. The doctor decides the outcome; the "
              "reference side is shown alongside. Original scores and comments unchanged."])
    s.append([])
    hd = ["#H Dataset", "#H Case", "#H Condition", "#H MedGemma finding", "#H MedGemma side", "#H CheXagent box",
          "#H Box side", "#H Evidence Engine side statement", "#H CheXagent verdict", "#H CheXagent's side",
          "#H Correction type", "#H Reference side", "#H Reference quote"]
    for doc in DOCS:
        dn = doc.replace("Dr Siddarth", "Dr Siddharth")
        hd += [f"#H {dn}: scores MedGemma / CheXagent / grounding", f"#H {dn}: side verdict", f"#H {dn}: comment"]
    hd += ["#H Outcome (doctor-based)", "#H Reference agrees with"]
    s.append(hd)
    out_rows = []
    for k, v in by_f.items():
        d0 = v[0]
        if d0["Row type"] != "MedGemma stated" or not d0["mg_side"]:
            continue
        mg, ref, bx, en = d0["mg_side"], d0["ref_side"], d0["bx_side"], d0["engine_side"]
        if en:
            cx, ctype = en, "Evidence Engine side statement"
        elif bx and compare(bx, mg) == "OPPOSITE side":
            cx, ctype = bx, "box on the opposite side"
        elif bx and mg == "both" and bx != "both":
            cx, ctype = bx, "box on one side only (MedGemma: bilateral)"
        else:
            cx, ctype = None, ("no correction (box same side or both sides)" if bx else
                               "no box side recorded" + (" (Form C)" if d0.get("box_recorded") is False else ""))
        mg_vs_ref = compare(mg, ref)
        if not cx and mg_vs_ref in ("same side", "not assessable"):
            continue
        line = [d0["Dataset"], d0["Current image"], d0["Condition"], d0["MedGemma finding (shown)"], mg,
                d0.get("box_location") or ("not recorded (Form C)" if d0.get("box_recorded") is False else "no box"),
                bx or "-", (f"supports the finding on the {en} rather than the reported side" if en else "-"),
                d0["CheXagent verdict (engine)"], cx or "-", ctype, ref or "not stated", d0["Reference quote"]]
        verdicts = []
        for doc in DOCS:
            dv = [x for x in v if x["Doctor"] == doc]
            sc_txt, vd = [], []
            for x in dv:
                m, c, g = x[SC["medgemma"]], x[SC["chexagent"]], x[SC["grounding"]]
                com = (x["Doctor comment"] or "").lower()
                cs = None   # comments are not interpreted automatically (shown verbatim, flagged if a side is named)
                sc_txt.append(f"{m}/{c}/{g} ({x['Form']})")
                if cx and cs and cs == cx and cs != mg:
                    vd.append("agrees with CheXagent (comment)")
                elif cs and cs == mg and cx and cs != cx:
                    vd.append("agrees with MedGemma (comment)")
                elif cs and cx and cs not in (mg, cx):
                    vd.append(f"neither - comment names the {cs} side")
                elif m is not None and m <= 2 and ((c or 0) >= 4 or (g or 0) >= 4):
                    vd.append("agrees with CheXagent (scores)")
                elif m is not None and m >= 4 and c is not None and c <= 2:
                    vd.append("agrees with MedGemma (scores)")
                else:
                    vd.append("mixed / uncertain" if m is not None or c is not None else "not scored")
            verdicts += vd
            coms = [f"{x['Doctor comment']} ({x['Form']})" for x in dv if x["Doctor comment"]]
            flag = "READ MANUALLY (names a side): " if any(side_of_text(c) for c in coms) else ""
            line += [", ".join(sc_txt).replace("None", "-") or "-", "; ".join(vd) or "not seen",
                     (flag + " | ".join(coms)) if coms else ""]
        yes = any(x.startswith("agrees with CheXagent") for x in verdicts)
        no = any(x.startswith(("agrees with MedGemma", "neither")) for x in verdicts)
        if cx:
            outcome = ("CheXagent corrected MedGemma's side - doctor agreed with CheXagent" if yes and not no else
                       "CheXagent corrected MedGemma's side - doctors split" if yes and no else
                       "CheXagent corrected MedGemma's side - doctor agreed with MedGemma" if no else
                       "CheXagent corrected MedGemma's side - doctor mixed / uncertain")
        else:
            outcome = "MedGemma side differs from reference - CheXagent did not correct"
        refagree = ("not stated" if ref is None else "CheXagent" if cx and compare(cx, ref) == "same side" else
                    "MedGemma" if compare(mg, ref) == "same side" else "neither exactly")
        out_rows.append((outcome, line + [outcome, refagree]))
    order = ["CheXagent corrected MedGemma's side - doctor agreed with CheXagent",
             "CheXagent corrected MedGemma's side - doctors split",
             "CheXagent corrected MedGemma's side - doctor agreed with MedGemma",
             "CheXagent corrected MedGemma's side - doctor mixed / uncertain",
             "MedGemma side differs from reference - CheXagent did not correct"]
    good = PatternFill("solid", fgColor="E2EFDA")
    for o, line in sorted(out_rows, key=lambda x: (order.index(x[0]), x[1][1])):
        s.append(line)
        if o == order[0]:
            for cc in s[s.max_row]:
                cc.fill = good
    s.append([])
    s.append(["Summary"] + [f"{o}: {sum(1 for x in out_rows if x[0] == o)}" for o in order])
    style(s, [16, 30, 16, 24, 9, 28, 8, 24, 18, 9, 22, 10, 34, 22, 26, 30, 22, 26, 30, 36, 14])
    s.freeze_panes = "D6"

    # read me
    rm = wb["Read me"]
    marker = "Analysis tabs (added by experiments/doctor_scores_analysis_tabs.py)"
    for row in list(rm.iter_rows()):
        if row[0].value and str(row[0].value).startswith(marker):
            rm.delete_rows(row[0].row, rm.max_row)
            break
    rm.append([marker + ": 'Cond - model vs doctors' (per-condition radiologist ratings of each model output), "
               "'Cond - model vs reference' (MedGemma and CheXagent vs the reference report and vs the doctor's call), "
               "'CheXagent boxes' (grounding scores and box side), 'Side errors' (MedGemma side vs reference side), "
               "'Side corrections' (MedGemma side contradicted by the reference or CheXagent, with the doctors' "
               "verdicts). Existing tabs, scores and comments are unchanged."])
    try:
        wb.save(D.OUT)
    except PermissionError:
        raise SystemExit(f"Could not save {D.OUT}: close it in Excel and run again (nothing was changed).")
    print(f"updated {D.OUT}: {', '.join(TABS)}")
    print("side errors:", {c: comps.count(c) for c in set(comps)})
    from collections import Counter
    print("side corrections:", Counter(o for o, _ in out_rows))
    for o, line in out_rows:
        print("  ", line[1], "|", line[3], "| CheXagent", line[9], "|", line[10], "| ref", line[11], "|", o)


if __name__ == "__main__":
    main()
