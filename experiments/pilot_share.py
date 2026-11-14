"""Shareable outputs of the pilot (no report text, no subject IDs; pages by number only):
a one-page PDF summary and an Excel workbook (summary, Experiment 1-3 detail, notes).

Summary numbers are read from the scored results file, so they match it exactly; the detail
tabs come from the labels-only files and the raw run outputs.

    python experiments/pilot_share.py RUNS_DIR [--preliminary]
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
RESULTS = ROOT / "experiments" / "results"
MODELS = {"team_medgemma": "Team MedGemma", "medgemma15": "MedGemma 1.5", "chexagent": "CheXagent",
          "maira2": "MAIRA-2"}


def md_tables(text):
    """{heading: [rows]} for the markdown tables under each '### ' heading of the primary section."""
    primary = text.split("## Sensitivity")[0]
    tables, head = {}, None
    for line in primary.splitlines():
        if line.startswith("### "):
            head = line[4:].strip()
        elif line.startswith("Per condition"):
            head = "Per condition"
        elif line.startswith("|") and not line.startswith("|---") and head:
            tables.setdefault(head, []).append([c.strip() for c in line.strip("|").split("|")])
    return tables


def sensitivity_rows(text):
    """[(analysis, model, caught)] from the sensitivity sections' Experiment 1 tables."""
    out = []
    for block in text.split("## Sensitivity")[1:]:
        name = block.splitlines()[0].strip(": ")
        for line in block.splitlines():
            if line.startswith("| ") and "%" in line:
                cells = [c.strip() for c in line.strip("|").split("|")]
                out.append((name, cells[0], cells[1]))
    return out


def answer_kind(text):
    t = text.strip().lower()
    return "Yes" if t.startswith("yes") else "No" if t.startswith("no") else \
        "no answer (thinking cut off)" if "<unused" in t else "hedged / other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--preliminary", action="store_true")
    args = ap.parse_args()

    from chexagent_independent_screen import CONDITIONS, found
    from medrax.evidence import EvidenceEngine
    from vindr_grounding_amendment import box_sides

    tag = "_PRELIMINARY" if args.preliminary else ""
    results_md = (RESULTS / f"pilot_model_comparison{tag}.md").read_text(encoding="utf-8")
    tables, sens = md_tables(results_md), sensitivity_rows(results_md)
    runs = Path(args.runs_dir)
    raw = {m: json.loads((runs / f"{m}.json").read_text(encoding="utf-8")) for m in ("medgemma15", "chexagent", "maira2")}
    titles = {t.split()[1]: t for t in raw["chexagent"]}
    names = ("ground_truth_auto", "model_auto") if args.preliminary else ("ground_truth", "model")
    gt = {(r["page"], r["condition"]): r for r in csv.DictReader(open(RESULTS / f"pilot_{names[0]}_labels.csv", encoding="utf-8"))}
    ml = {(r["model"], r["page"], r["condition"]): r for r in csv.DictReader(open(RESULTS / f"pilot_{names[1]}_labels.csv", encoding="utf-8"))}
    engine = EvidenceEngine()
    pages = sorted(titles, key=int)
    status = "PRELIMINARY (unreviewed automatic labels)" if args.preliminary else "Final (radiologist-reviewed labels)"
    stamp = time.strftime("%Y-%m-%d")
    base = ROOT / "logs" / f"pilot_summary{tag}_{time.strftime('%Y%m%d-%H%M%S')}"

    # ------------------------------------------------------------------ Excel
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    bold, head_fill = Font(bold=True), PatternFill("solid", fgColor="E8ECF2")

    def sheet(title, header, rows, widths=None):
        ws = wb.create_sheet(title)
        ws.append(header)
        for c in ws[1]:
            c.font, c.fill = bold, head_fill
        for r in rows:
            ws.append(r)
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for i, w in enumerate(widths or [], 1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = w
        return ws

    ws = wb.active
    ws.title = "Summary"
    ws.append([f"Pilot model comparison: {status}"])
    ws["A1"].font = Font(bold=True, size=13)
    ws.append([f"19 prior/current chest X-ray comparisons, 13 conditions. Generated {stamp}. 95% intervals in brackets."])
    for name in ("Experiment 1: each model's own report", "Per condition",
                 "Experiment 2: direct question per condition"):
        key = next((k for k in tables if k.startswith(name)), None)
        if not key:
            continue
        ws.append([])
        ws.append([key if name != "Per condition" else "Experiment 1 per condition (caught / ground-truth count)"])
        ws.cell(ws.max_row, 1).font = bold
        for i, row in enumerate(tables[key]):
            ws.append(row)
            if i == 0:
                for c in ws[ws.max_row]:
                    c.font, c.fill = bold, head_fill
    key = next(k for k in tables if k.startswith("Experiment 3"))
    ws.append([])
    ws.append([key])
    ws.cell(ws.max_row, 1).font = bold
    for i, row in enumerate(tables[key]):
        ws.append(row)
        if i == 0:
            for c in ws[ws.max_row]:
                c.font, c.fill = bold, head_fill
    ws.append([])
    ws.append(["Sensitivity analyses, Experiment 1 (caught)"])
    ws.cell(ws.max_row, 1).font = bold
    ws.append(["analysis", "model", "caught (95% CI)"])
    for c in ws[ws.max_row]:
        c.font, c.fill = bold, head_fill
    for row in sens:
        ws.append(list(row))
    for col, w in zip("ABCDEFG", (34, 26, 30, 22, 22, 22, 22)):
        ws.column_dimensions[col].width = w

    sheet("Exp1 labels", ["page", "condition", "ground truth", "ground-truth side"]
          + [f"{n} {k}" for n in MODELS.values() for k in ("label", "side")],
          [[int(p), c, gt[(p, c)]["label"], gt[(p, c)]["side"]]
           + [v for m in MODELS for v in (ml[(m, p, c)]["label"], ml[(m, p, c)]["side"])]
           for p in pages for c in CONDITIONS], [6, 18, 14, 14] + [14] * 8)

    def cx(p, c):
        q = raw["chexagent"][titles[p]]["questions"][c]
        hit, state = found(engine, c, q["score"]) if q["score"] is not None else (False, "no score")
        return [round(q["score"], 4) if q["score"] is not None else None, "found" if hit else "not found", state]

    sheet("Exp2 questions", ["page", "condition", "ground truth", "CheXagent p_present", "CheXagent at cut-off",
                             "CheXagent evidence state", "MedGemma 1.5 answer", "MedGemma 1.5 Yes share"],
          [[int(p), c, gt[(p, c)]["label"]] + cx(p, c)
           + [answer_kind(raw["medgemma15"][titles[p]]["questions"][c]["answer"]),
              (lambda s: round(s, 4) if s is not None else None)(raw["medgemma15"][titles[p]]["questions"][c]["score"])]
           for p in pages for c in CONDITIONS], [6, 18, 14, 16, 16, 22, 24, 18])

    def where(m, p, c, side):
        boxes = raw[m][titles[p]]["locations"][c]["boxes"]
        if boxes is None:
            return "no answer (cut off)", "no answer"
        sides = box_sides([b["box"] for b in boxes])
        shown = " + ".join(sorted(sides)) or "none"
        return shown, ("not located" if not sides else "both sides" if len(sides) > 1 else
                       "correct" if side in sides else "wrong side")

    cases = [(p, c) for p in pages for c in CONDITIONS
             if gt[(p, c)]["label"] in ("present", "possible") and gt[(p, c)]["side"] in ("right", "left")]
    sheet("Exp3 location", ["page", "condition", "ground-truth side"]
          + [f"{n} {k}" for n in ("CheXagent", "MAIRA-2", "MedGemma 1.5") for k in ("boxed side", "result")],
          [[int(p), c, gt[(p, c)]["side"]] + [v for m in ("chexagent", "maira2", "medgemma15")
                                              for v in where(m, p, c, gt[(p, c)]["side"])] for p, c in cases],
          [6, 18, 16] + [16] * 6)

    notes = [
        ("Status", status + ". Final results follow the radiologist's review of the label sheets."),
        ("Data", "19 prior/current comparisons from MIMIC-CXR (pages by number only; no report text or IDs here)."),
        ("Ground truth", "The radiologist report per comparison, read for 13 conditions: present / possible / not reported, "
                         "and side. 'Not reported' is not proof of absence."),
        ("Models", "Team MedGemma: retrained MedGemma, reports as provided. MedGemma 1.5 4B (base, Ollama, 4-bit). "
                   "CheXagent-2-3b. MAIRA-2 (bf16). Each run independently, without the generated report or ground truth."),
        ("Experiment 1", "Each model's own report, read for the 13 conditions."),
        ("Experiment 2", "'Is there X?' per condition. CheXagent: found = score at or above its validated or provisional "
                         "cut-off. MedGemma 1.5: found = answer starts with Yes; score = Yes share of its first token."),
        ("Experiment 3", "Locate the condition by name only (no side); side from the box position; cases = one-sided "
                         "ground-truth findings."),
        ("Statistics", "95% intervals: Wilson for proportions; page-level bootstrap (2,000) for differences and AUC. "
                       "No p-values; 19 images support no ranking claims."),
        ("Unconfirmed", "Stated by the model but not reported in the ground truth; not the same as a false positive."),
        ("Caveat", "All models were probably trained on MIMIC-CXR; results may be optimistic and do not generalise."),
        ("Protocol", "experiments/model_comparison_pilot_protocol.md (amendments A1-A4)."),
    ]
    sheet("Notes", ["item", "note"], notes, [16, 120])
    for row in wb["Notes"].iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(base.with_suffix(".xlsx"))

    # ------------------------------------------------------------------ one-page PDF
    from preview_onepage_report import BLACK, GREY, OnePage
    from fpdf.fonts import FontFace

    pdf = OnePage()
    pdf.add_page()
    W, x = pdf.w - 24, 12
    pdf.set_xy(x, 10)
    pdf.font(14, True)
    pdf.cell(W, 7, "Independent model comparison: 19 chest X-ray comparisons")
    pdf.set_xy(x, 17)
    pdf.font(8.5, True, (185, 60, 20) if args.preliminary else BLACK)
    pdf.cell(W, 4.5, f"{status}. {stamp}.")
    y = 23

    def para(text, size=7.6, gap=0.6, color=BLACK):
        nonlocal y
        pdf.set_xy(x, y)
        pdf.font(size, color=color)
        pdf.multi_cell(W, size * 0.43, text, markdown=True)
        y = pdf.get_y() + gap

    def table(rows, widths, size=7.2):
        nonlocal y
        pdf.set_xy(x, y)
        pdf.font(size)
        with pdf.table(col_widths=widths, width=W, line_height=size * 0.48, text_align="LEFT",
                       headings_style=FontFace(emphasis="BOLD", fill_color=(232, 236, 242)),
                       borders_layout="SINGLE_TOP_LINE") as t:
            for r in rows:
                t.row(r)
        y = pdf.get_y() + 1.8

    y = pdf.title_bar(x, y, W, "Setup", 9.5)
    para("**Data:** 19 prior/current comparisons (MIMIC-CXR-derived); ground truth = the radiologist report, read for "
         "13 conditions (present / possible / not reported, and side). **Models**, each run independently without the "
         "generated report or ground truth: Team MedGemma (retrained; reports as provided), MedGemma 1.5 4B (base), "
         "CheXagent-2-3b, MAIRA-2. **Three tests:** (1) each model's own report; (2) a direct question per condition; "
         "(3) locating the condition without being told the side. 95% intervals in brackets; 'unconfirmed' = stated "
         "but not in the ground-truth report (not proof of a false positive).")
    y = pdf.title_bar(x, y, W, "1. Own report: ground-truth findings caught (32 in total)", 9.5)
    e1 = tables[next(k for k in tables if k.startswith("Experiment 1"))]
    table([[r[0], r[1], r[2], r[3], r[4]] for r in e1], (24, 30, 34, 14, 14))
    pc = tables.get("Per condition", [])
    keep = [r for r in pc[1:] if r[0] in ("pleural effusion", "atelectasis", "cardiomegaly", "emphysema")]
    short = lambda s: s.split(" (")[0]  # noqa: E731
    table([["condition", "in ground truth", "Team MedGemma", "MedGemma 1.5", "CheXagent", "MAIRA-2",
            "CheXagent asked directly"]] + [[r[0], r[1]] + [short(v) for v in r[2:6]] + [r[6]] for r in keep],
          (22, 14, 16, 16, 14, 14, 20))
    y = pdf.title_bar(x, y, W, "2. Direct question and 3. Location without the side", 9.5)
    e2 = tables[next(k for k in tables if k.startswith("Experiment 2"))]
    para("**Direct question per condition** (ground-truth findings found, of 32):", 7.4, 0.4)
    table([[r[0], r[1], r[2], r[3]] for r in e2], (34, 32, 14, 24))
    e3k = next(k for k in tables if k.startswith("Experiment 3"))
    n_cases = re.search(r"\((\d+)", e3k).group(1)
    para(f"**Location without the side** ({n_cases} one-sided ground-truth findings):", 7.4, 0.4)
    table([[r[0], r[1], r[2], r[3], r[4]] for r in tables[e3k]], (20, 32, 14, 14, 16))
    y = pdf.title_bar(x, y, W, "What the data supports", 9.5)
    for b in ("No clear winner among the report writers: all catch roughly 20-40% of ground-truth findings, with "
              "overlapping intervals.",
              "The team's retrained MedGemma is ahead of the base MedGemma 1.5 (clearly so when the duplicated pages "
              "9-12 are excluded); settings also differ, so not attributable to retraining alone.",
              "Asking directly finds more than writing a report: CheXagent finds more when asked per condition, "
              "mostly findings reports tend to omit (cardiomegaly, emphysema). Supports a direct-question check "
              "alongside the report.",
              "Pleural effusion is the strong point (team model and CheXagent caught all); laterality is mixed."):
        para("- " + b, 7.4, 0.3)
    y += 1
    y = pdf.title_bar(x, y, W, "Limitations and next steps", 9.5)
    para("19 images (about 30 ground-truth findings) support no ranking or significance claims. Ground truth is "
         "free-text reports, so absence is not confirmed. All models were probably trained on MIMIC-CXR, so results "
         "may be optimistic. Rows 9-12 have duplicated ground truth (being checked). Results change with how "
         "'possible' findings are counted. **Next:** radiologist review of the labels, then final results; then a larger "
         "study on images the models have not seen.", 7.4)
    pdf.set_xy(x, pdf.h - 9)
    pdf.font(6.5, color=GREY)
    pdf.cell(W, 3, "Protocol: experiments/model_comparison_pilot_protocol.md (amendments A1-A4). No report text or "
                   "patient identifiers in this summary.")
    if pdf.get_y() > pdf.h - 8 or y > pdf.h - 11:
        print(f"warning: content may overflow the page (y={y:.0f} mm)")
    pdf.output(str(base.with_suffix(".pdf")))
    print(f"-> {base}.pdf / .xlsx")


if __name__ == "__main__":
    main()
