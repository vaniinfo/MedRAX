"""Workbook: CheXagent vs MAIRA-2 against the ground truth, one sheet per filter.

Source: a report_vs_ground_truth.py workbook (columns by position, as its verdict headers repeat)
and maira_score_nih.md (NIH AUCs). PRELIMINARY ground truth for the 19 comparisons.

    python experiments/chexagent_vs_maira_excel.py logs/report_vs_ground_truth_<time>.xlsx
"""

import re
import sys
import time
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parents[1]
NOT = ("Not stated", "NOT IN GROUND TRUTH")
FILL = {"Missed": "F8D3D3", "Wrong side": "FBDCC7", "Side partly matches": "E3DDF4", "Not in ground truth": "FDF0C2",
        "found": "D6EAD8"}
HEAD = PatternFill("solid", fgColor="E8ECF2")
# source columns (report_vs_ground_truth.py layout)
SRC = {"page": "A", "condition": "B", "medgemma": "C", "medgemma_v": "D", "chex_cur": "E", "chex_cur_v": "F",
       "chex_long": "G", "chex_long_v": "H", "maira": "I", "maira_v": "J", "gt": "K", "chex_score": "L",
       "chex_found": "M", "maira_score": "N"}
COLS = [("page", "page"), ("condition", "condition"), ("Ground truth", "gt"),
        ("CheXagent score (Is there X?)", "chex_score"), ("CheXagent at cut-off", "chex_found"),
        ("MAIRA-2 (own report)", "maira"), ("MAIRA-2 vs ground truth", "maira_v"),
        ("CheXagent (own report, current image)", "chex_cur"), ("CheXagent (current) vs ground truth", "chex_cur_v"),
        ("CheXagent (own report, prior + current)", "chex_long"), ("CheXagent (prior + current) vs ground truth", "chex_long_v"),
        ("MAIRA-2 score (sentence likelihood)", "maira_score")]
WIDTHS = [6, 18, 30, 14, 12, 26, 20, 30, 20, 30, 20, 14]


def load(path):
    ws = load_workbook(path)["Comparison"]
    out = []
    for row in ws.iter_rows(min_row=2):
        cells = {c.column_letter: c.value for c in row}
        out.append({k: cells.get(col) for k, col in SRC.items()})
    return out


def sheet(wb, title, note, rows):
    ws = wb.create_sheet(title)
    ws.append([note])
    ws["A1"].font = Font(italic=True, color="52514E")
    ws.append([h for h, _ in COLS])
    for c in ws[2]:
        c.font, c.fill = Font(bold=True), HEAD
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for r in rows:
        ws.append([r[k] for _, k in COLS])
        for i, (_, k) in enumerate(COLS, 1):
            v = r[k]
            if isinstance(v, str) and v in FILL and (k.endswith("_v") or k == "chex_found"):
                ws.cell(ws.max_row, i).fill = PatternFill("solid", fgColor=FILL[v])
    for i, w in enumerate(WIDTHS, 1):
        ws.column_dimensions[ws.cell(2, i).column_letter].width = w
    ws.freeze_panes = "C3"
    ws.auto_filter.ref = f"A2:{ws.cell(2, len(COLS)).column_letter}{ws.max_row}"
    return len(rows)


def main(src):
    rows = load(src)
    gt_pos = [r for r in rows if r["gt"] not in NOT]
    found = lambda r: r["chex_found"] == "found"  # noqa: E731
    maira_caught = lambda r: r["maira_v"] in ("Correct", "Side partly matches", "Wrong side")  # noqa: E731

    wb = Workbook()
    s = wb.active
    s.title = "Summary"
    filters = [
        ("1 CheXagent yes, MAIRA-2 no", [r for r in gt_pos if r["maira_v"] == "Missed" and found(r)],
         "Ground-truth findings that MAIRA-2's report missed and CheXagent's direct question found at its cut-off."),
        ("1b MAIRA-2 yes, CheXagent no", [r for r in gt_pos if maira_caught(r) and not found(r)],
         "The reverse: ground-truth findings MAIRA-2's report stated and CheXagent's direct question did not find."),
        ("2 Ground-truth findings", gt_pos,
         "All ground-truth findings: compare 'CheXagent at cut-off' = found with MAIRA-2's verdict."),
        ("3 Not in ground truth", [r for r in rows if r["gt"] in NOT and (r["maira_v"] == "Not in ground truth" or found(r)
                                    or "Not in ground truth" in (r["chex_cur_v"], r["chex_long_v"]))],
         "Statements or findings the ground-truth report does not mention (possible false findings, or not written "
         "down by the radiologist)."),
        ("4 Side", [r for r in gt_pos if "Wrong side" in (r["maira_v"], r["chex_cur_v"], r["chex_long_v"])
                     or "Side partly matches" in (r["maira_v"], r["chex_cur_v"], r["chex_long_v"])],
         "Findings where a report's side disagrees fully (wrong side) or partly with the ground truth."),
    ]
    counts = {name: sheet(wb, name[:31], note, rs) for name, rs, note in filters}
    sheet(wb, "All rows", "All rows of the comparison (use the column filters).", rows)

    nih = (ROOT / "experiments" / "results" / "maira_score_nih.md").read_text(encoding="utf-8")
    nih_rows = [[c.strip() for c in l.strip("|").split("|")] for l in nih.splitlines()
                if l.startswith("| ") and re.match(r"\| [a-z]", l) and "finding" not in l]

    def count(rs, pred):
        return sum(pred(r) for r in rs)
    lines = [
        ("CheXagent vs MAIRA-2 against the ground truth", None),
        ("PRELIMINARY for the 19 comparisons: ground truth read automatically, not yet reviewed by a radiologist.", None),
        ("", None),
        ("Main evidence: 3,500 labelled NIH images (AUC: how well the score separates present from absent; 0.5 = chance)", "h"),
    ]
    for text, kind in lines:
        s.append([text])
        if kind == "h" or text.startswith("CheXagent vs"):
            s.cell(s.max_row, 1).font = Font(bold=True, size=12 if text.startswith("CheXagent vs") else 11)
    s.append(["finding", "CheXagent AUC [95% CI]", "MAIRA-2 AUC [95% CI]", "MAIRA-2 minus CheXagent [95% CI]"])
    for c in s[s.max_row]:
        c.font, c.fill = Font(bold=True), HEAD
    for r in nih_rows:
        s.append([r[0], r[3], r[2], r[4].replace("+", "").replace("--", "-")])
    s.append([])
    s.append(["Our 19 comparisons (sheet per filter)"])
    s.cell(s.max_row, 1).font = Font(bold=True, size=11)
    s.append(["filter", "CheXagent", "MAIRA-2", "sheet"])
    for c in s[s.max_row]:
        c.font, c.fill = Font(bold=True), HEAD
    rows_summary = [
        ("Ground-truth findings caught (of %d)" % len(gt_pos), count(gt_pos, found), count(gt_pos, maira_caught),
         "2 Ground-truth findings"),
        ("Caught by one, missed by the other", counts["1 CheXagent yes, MAIRA-2 no"],
         counts["1b MAIRA-2 yes, CheXagent no"], "1 and 1b"),
        ("Stated / found but not in the ground truth: CheXagent direct question vs MAIRA-2 report",
         count(rows, lambda r: r["gt"] in NOT and found(r)), count(rows, lambda r: r["maira_v"] == "Not in ground truth"),
         "3 Not in ground truth"),
        ("Not in the ground truth: CheXagent own report (current image | prior + current)",
         f"{count(rows, lambda r: r['chex_cur_v'] == 'Not in ground truth')} | "
         f"{count(rows, lambda r: r['chex_long_v'] == 'Not in ground truth')}", "", "3 Not in ground truth"),
        ("Wrong side (reports): CheXagent current | prior + current vs MAIRA-2",
         f"{count(rows, lambda r: r['chex_cur_v'] == 'Wrong side')} | {count(rows, lambda r: r['chex_long_v'] == 'Wrong side')}",
         count(rows, lambda r: r["maira_v"] == "Wrong side"), "4 Side"),
    ]
    for r in rows_summary:
        s.append(list(r))
    s.append([])
    for note in ("Reading: on the NIH images CheXagent is better for every finding. In our 19 cases CheXagent's direct "
                 "question catches more ground-truth findings and catches 11 that MAIRA-2 missed (the reverse: 3), but "
                 "it also flags more findings the ground truth does not mention, and MAIRA-2 made no wrong-side statements.",
                 "Caveats: 19 cases; preliminary ground truth; all models were probably trained on MIMIC-CXR. "
                 "'Not in ground truth' is not proof of a false finding."):
        s.append([note])
        s.cell(s.max_row, 1).alignment = Alignment(wrap_text=True, vertical="top")
        s.merge_cells(start_row=s.max_row, start_column=1, end_row=s.max_row, end_column=4)
        s.row_dimensions[s.max_row].height = 45
    for col, w in zip("ABCD", (70, 26, 26, 26)):
        s.column_dimensions[col].width = w
    # Guide: one row per sheet, first in the workbook
    g = wb.create_sheet("Guide", 0)
    g.append(["CheXagent vs MAIRA-2 against the ground truth: guide to the sheets"])
    g["A1"].font = Font(bold=True, size=12)
    g.append(["PRELIMINARY for the 19 comparisons (ground truth not yet reviewed by a radiologist)."])
    g.append([])
    g.append(["Sheet", "What it shows", "Rows"])
    for c in g[4]:
        c.font, c.fill = Font(bold=True), HEAD
    one = filters[0][1]
    ex = "; ".join(f"page {r['page']} {r['condition']} ({r['chex_score']:.2f})" for r in one[:3])
    nf = lambda key: count(rows, lambda r: r[key] == "Not in ground truth")  # noqa: E731
    ws_ = lambda key: count(rows, lambda r: r[key] == "Wrong side")  # noqa: E731
    guide = [
        ("Summary", "NIH results (3,500 images, AUC per finding for both models), then the 19-case numbers, with notes "
                    "and caveats", ""),
        ("1 CheXagent yes, MAIRA-2 no", f"Ground-truth findings MAIRA-2's report missed and CheXagent's check found, "
                                        f"e.g. {ex}", len(one)),
        ("1b MAIRA-2 yes, CheXagent no", "The reverse: ground-truth findings MAIRA-2's report stated and CheXagent's "
                                         "check did not find", len(filters[1][1])),
        ("2 Ground-truth findings", f"All {len(gt_pos)} ground-truth findings side by side: CheXagent found "
                                    f"{count(gt_pos, found)}, MAIRA-2 caught {count(gt_pos, maira_caught)}", len(gt_pos)),
        ("3 Not in ground truth", f"Findings the ground truth doesn't mention: CheXagent direct question "
                                  f"{count(rows, lambda r: r['gt'] in NOT and found(r))}, MAIRA-2 report {nf('maira_v')}, "
                                  f"CheXagent's own reports {nf('chex_cur_v')} (current image) and {nf('chex_long_v')} "
                                  f"(prior + current)", len(filters[3][1])),
        ("4 Side", f"Side disagreements: MAIRA-2 {ws_('maira_v')} wrong sides; CheXagent's reports {ws_('chex_cur_v')} "
                   f"(current image) and {ws_('chex_long_v')} (prior + current)", len(filters[4][1])),
        ("All rows", "Everything, with column filters on, for anyone who wants to explore", len(rows)),
    ]
    for r in guide:
        g.append(list(r))
        g.cell(g.max_row, 2).alignment = Alignment(wrap_text=True, vertical="top")
        g.cell(g.max_row, 1).alignment = Alignment(vertical="top")
    g.append([])
    g.append(["Colours: red = missed; orange = wrong side; purple = side partly matches; yellow = not in ground truth; "
              "green = CheXagent found at its cut-off."])
    for col, w in zip("ABC", (30, 110, 8)):
        g.column_dimensions[col].width = w
    out = ROOT / "logs" / f"chexagent_vs_maira_{time.strftime('%Y%m%d-%H%M%S')}.xlsx"
    wb.save(out)
    print(out)
    for r in rows_summary:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1])
