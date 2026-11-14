"""Colour-coded Excel: each report vs the ground truth, per page and condition (19 comparisons,
13 conditions). PRELIMINARY ground truth (automatic lookup, unreviewed).

Reports: MedGemma (team), CheXagent own report (current image, area by area), CheXagent
longitudinal report (prior + current), MAIRA-2 own report (prior + current). Verdicts: Correct,
Missed, Wrong side, Side partly matches, Not in ground truth.

    python experiments/report_vs_ground_truth.py BASE   (BASE = logs/chexagent_screen_<time>)
"""

import json
import sys
import time
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from pilot_ground_truth_sheet import side_of  # noqa: E402

NOT = ("Not stated", "NOT IN GROUND TRUTH")
ORDER = ["Correct", "Missed", "Wrong side", "Side partly matches", "Not in ground truth"]
FILL = {"Missed": "F8D3D3", "Wrong side": "FBDCC7", "Side partly matches": "E3DDF4", "Not in ground truth": "FDF0C2"}
MEANING = {"Correct": "Both mention the finding, sides agree (or a side is not given)",
           "Missed": "Ground truth has it; the report does not mention it",
           "Wrong side": "Both mention it, on opposite sides (right vs left)",
           "Side partly matches": "Both mention it; one says both sides, the other one side",
           "Not in ground truth": "The report mentions it; the ground-truth report does not (possible false finding, "
                                  "or not written down by the radiologist)"}
MODELS = [("MedGemma (team report)", "medgemma"), ("CheXagent (own report, current image)", "chexagent_report"),
          ("CheXagent (own report, prior + current)", "chexagent_longitudinal"), ("MAIRA-2 (own report)", "maira")]


def side(v):
    v2 = v.replace("Present", "").replace("(possible)", "")
    return side_of(v2) if v2.strip(" ,()") else "not stated"


def verdict(mv, gv):
    m, g = mv not in NOT, gv not in NOT
    if not m and not g:
        return ""
    if g and not m:
        return "Missed"
    if m and not g:
        return "Not in ground truth"
    ms, gs = side(mv), side(gv)
    if ms in ("right", "left", "bilateral") and gs in ("right", "left", "bilateral") and ms != gs:
        return "Side partly matches" if "bilateral" in (ms, gs) else "Wrong side"
    return "Correct"


def main(base):
    base = Path(base)
    rec = {(r["page"], r["condition"]): r for r in json.loads(Path(f"{base}_reports_table.json").read_text(encoding="utf-8"))}
    screen = {(p["title"], s["condition"]): s for p in json.loads(Path(f"{base}.json").read_text(encoding="utf-8"))
              for s in p["screen"]}
    maira_score = json.loads(Path(f"{base}_maira_scores.json").read_text(encoding="utf-8"))
    longi = json.loads(Path(f"{base}_chexagent_longitudinal.json").read_text(encoding="utf-8"))["labels"]
    for (t, c), r in rec.items():
        r["chexagent_longitudinal"] = longi[t][c]

    wb = Workbook()
    ws = wb.active
    ws.title = "Comparison"
    head = ["page", "condition"]
    for name, _ in MODELS:
        head += [name, "vs ground truth"]
    head += ["Ground truth", "CheXagent score (Is there X?)", "CheXagent at cut-off", "MAIRA-2 score (sentence likelihood)"]
    ws.append(head)
    for c in ws[1]:
        c.font, c.fill = Font(bold=True), PatternFill("solid", fgColor="E8ECF2")
        c.alignment = Alignment(wrap_text=True, vertical="top")
    counts = {k: dict.fromkeys(ORDER, 0) for _, k in MODELS}
    conditions = list(dict.fromkeys(c for _, c in rec))
    for n in range(1, 20):
        title = f"Comparison {n} of 19"
        for cond in conditions:
            r, s = rec[(title, cond)], screen[(title, cond)]
            if not (any(r[k] not in NOT for _, k in MODELS) or r["ground_truth"] not in NOT or s["chexagent_found"]):
                continue
            row, colour = [n, cond], []
            for _, k in MODELS:
                v = verdict(r[k], r["ground_truth"])
                if v:
                    counts[k][v] += 1
                colour.append((len(row) + 1, v))
                row += [r[k], v]
            ms = maira_score[title].get(cond)
            row += [r["ground_truth"], round(s["p_present"], 3), "found" if s["chexagent_found"] else "not found",
                    round(ms, 3) if ms is not None else "not scored"]
            ws.append(row)
            for col, v in colour:
                if v in FILL:
                    ws.cell(ws.max_row, col).fill = PatternFill("solid", fgColor=FILL[v])
    for i, w in enumerate([6, 18] + [26, 18] * len(MODELS) + [32, 14, 12, 16], 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions

    s = wb.create_sheet("Summary")
    s.append(["How often each report disagrees with the ground truth (19 comparisons, 13 conditions)"])
    s["A1"].font = Font(bold=True, size=12)
    s.append(["PRELIMINARY: ground truth read automatically from the ground-truth reports, not yet reviewed by a "
              "radiologist. \"Possible\" ground-truth findings count as present."])
    s.append([])
    s.append(["Category"] + [name for name, _ in MODELS] + ["Colour", "Meaning"])
    for c in s[4]:
        c.font, c.fill = Font(bold=True), PatternFill("solid", fgColor="E8ECF2")
    for k in ORDER:
        s.append([k] + [counts[m][k] for _, m in MODELS] + ["", MEANING[k]])
        if k in FILL:
            s.cell(s.max_row, len(MODELS) + 2).fill = PatternFill("solid", fgColor=FILL[k])
    s.append([])
    for label, keys in (("Ground-truth findings (total)", ("Correct", "Missed", "Wrong side", "Side partly matches")),
                        ("Missed or wrong side", ("Missed", "Wrong side")),
                        ("Missed, wrong side or side partly matches", ("Missed", "Wrong side", "Side partly matches"))):
        s.append([label] + [sum(counts[m][k] for k in keys) for _, m in MODELS])
        s.cell(s.max_row, 1).font = Font(bold=True)
    s.append([])
    for note in ("Inputs: MedGemma = the team's generated report as provided. CheXagent (current image) = its area-by-area "
                 "description of the current image. CheXagent (prior + current) = its findings for the current image with "
                 "change from the prior (two images: about 2,061 input tokens, slightly over its 2,048 design length). "
                 "MAIRA-2 = its findings from prior + current images (full precision, no prior report).",
                 "Page 5 atelectasis is a judgement call (ground truth: retrocardiac region, i.e. left, and right lower lobe; "
                 "read by text as right only).",
                 "On pages 9 and 10 MAIRA-2 wrote only \"no substantial change since the recent radiograph\" (it expects the "
                 "prior report, which it was not given), so its findings there count as missed."):
        s.append(["Note: " + note])
    for col, w in zip("ABCDEFG", (44, 22, 22, 22, 22, 10, 100)):
        s.column_dimensions[col].width = w
    wb.move_sheet("Summary", -1)
    out = ROOT / "logs" / f"report_vs_ground_truth_{time.strftime('%Y%m%d-%H%M%S')}.xlsx"
    wb.save(out)
    print(out)
    for name, k in MODELS:
        print(f"{name}: {counts[k]}")


if __name__ == "__main__":
    main(sys.argv[1])
