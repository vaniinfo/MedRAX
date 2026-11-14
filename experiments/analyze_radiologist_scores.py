"""Confusion-matrix statistics from a radiologist's filled scoring form (scores CSV from read_radiologist_scores.py).

The radiologist's scale: 5 completely correct, 4 mostly correct, 3 partially correct, 2 mostly incorrect,
1 completely incorrect, blank = N/A (not applicable or cannot be assessed). The truth is the radiologist's judgement
of each model output, not the dataset's reference report or labels.

Each row is turned into one decision per model:
- MedGemma: a stated finding is a "present" call; a "Not detected" row (CheXagent-only or missed by both) is an
  "absent" call. Rated correct -> TP / TN; rated incorrect -> FP / FN.
- CheXagent: "Agree" on a stated finding and every CheXagent-only row are "present" calls; "Not agree" on a stated
  finding and "Not detected" on a missed row are "absent" calls; scored with the CheXagent-evidence rating.
Correct = 4-5, incorrect = 1-2. A 3 (partially correct) is reported separately and handled two ways: strict
(excluded) and lenient (counted as correct). N/A is excluded.

    .venv/Scripts/python.exe experiments/analyze_radiologist_scores.py logs/returned_form_A_doctor1_scores.csv
"""

import csv
import sys
from collections import Counter
from pathlib import Path


def num(v):
    return int(v) if str(v).strip().isdigit() else None


def kind(r):
    f = r["finding"]
    return "chexagent_only" if "(CheXagent only)" in f else "missed" if "(missed by both)" in f else "stated"


def dataset(r):
    p = r["page"]
    return "NIH" if "NIH" in p else "CheXpert Plus" if "CheXpert" in p else "MIMIC-CXR"


def decisions(rows):
    """[(model, call, rating, dataset)] with call 'present' / 'absent'."""
    out = []
    for r in rows:
        k, d = kind(r), dataset(r)
        out.append(("MedGemma", "present" if k == "stated" else "absent", num(r["score_medgemma_finding"]), d))
        if k == "stated":
            call = "present" if r["chexagent_evidence"].startswith("Agree") else "absent"
        else:
            call = "present" if k == "chexagent_only" else "absent"
        out.append(("CheXagent", call, num(r["score_chexagent_evidence"]), d))
    return out


def confusion(decs, lenient=False):
    c = Counter()
    for _, call, s, _ in decs:
        if s is None:
            c["N/A"] += 1
        elif s == 3 and not lenient:
            c["partial"] += 1
        else:
            ok = s >= 3
            c[("TP" if ok else "FP") if call == "present" else ("TN" if ok else "FN")] += 1
    return c


def rates(c):
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    f = lambda a, b: f"{a / (a + b):.0%} ({a}/{a + b})" if a + b else "-"  # noqa: E731
    return {"precision (PPV)": f(tp, fp), "sensitivity": f(tp, fn), "specificity": f(tn, fp), "NPV": f(tn, fn),
            "accuracy": f(tp + tn, fp + fn)}


def dist(values):
    v = Counter(values)
    n = sum(v[k] for k in (1, 2, 3, 4, 5))
    return (f"5:{v[5]} 4:{v[4]} 3:{v[3]} 2:{v[2]} 1:{v[1]} N/A:{v[None]} | correct (4-5) "
            f"{(v[4] + v[5]) / n:.0%}, partial {v[3] / n:.0%}, incorrect (1-2) {(v[1] + v[2]) / n:.0%} (n={n})"
            if n else "no ratings")


def main():
    path = Path(sys.argv[1])
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    L = [f"# Radiologist scores: {path.name}", "",
         f"{len(rows)} rows from {len({r['page'] for r in rows})} cases; row types "
         f"{dict(Counter(kind(r) for r in rows))}; datasets {dict(Counter(dataset(r) for r in rows))}. "
         "Truth = the radiologist's rating. Correct = 4-5, incorrect = 1-2, partial = 3, N/A = blank.", ""]
    for scope, sel in [("All cases", lambda d: True)] + [(f"{x} only", (lambda x: lambda d: d == x)(x))
                                                         for x in sorted({dataset(r) for r in rows})]:
        L += [f"## Detection: {scope}", "",
              "| model | TP | FP | TN | FN | partial (3) | N/A | precision | sensitivity | specificity | NPV | accuracy |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for model in ("MedGemma", "CheXagent"):
            decs = [x for x in decisions(rows) if x[0] == model and sel(x[3])]
            c, r = confusion(decs), rates(confusion(decs))
            L.append(f"| {model} (strict) | {c['TP']} | {c['FP']} | {c['TN']} | {c['FN']} | {c['partial']} | {c['N/A']} | "
                     + " | ".join(r.values()) + " |")
            c2, r2 = confusion(decs, lenient=True), rates(confusion(decs, lenient=True))
            L.append(f"| {model} (3 = correct) | {c2['TP']} | {c2['FP']} | {c2['TN']} | {c2['FN']} | - | {c2['N/A']} | "
                     + " | ".join(r2.values()) + " |")
        L.append("")

    stated = [r for r in rows if kind(r) == "stated"]
    L += ["## Report auditing: does CheXagent's 'Not agree' flag MedGemma's incorrect findings?", "",
          "MedGemma-stated findings, by the radiologist's MedGemma rating and CheXagent's verdict.", "",
          "| radiologist on MedGemma finding | CheXagent Agree | CheXagent Not agree (flag) |", "|---|---|---|"]
    t = Counter()
    for r in stated:
        s = num(r["score_medgemma_finding"])
        g = "N/A" if s is None else "correct (4-5)" if s >= 4 else "partial (3)" if s == 3 else "incorrect (1-2)"
        t[(g, r["chexagent_evidence"].startswith("Agree"))] += 1
    for g in ("correct (4-5)", "partial (3)", "incorrect (1-2)", "N/A"):
        L.append(f"| {g} | {t[(g, True)]} | {t[(g, False)]} |")
    inc_flag, inc = t[("incorrect (1-2)", False)], t[("incorrect (1-2)", False)] + t[("incorrect (1-2)", True)]
    cor_flag, cor = t[("correct (4-5)", False)], t[("correct (4-5)", False)] + t[("correct (4-5)", True)]
    flags = inc_flag + cor_flag
    L += ["", f"- Incorrect findings flagged (sensitivity of the flag): {inc_flag}/{inc}"
              + (f" = {inc_flag / inc:.0%}" if inc else ""),
          f"- Correct findings not flagged (specificity): {cor - cor_flag}/{cor}" + (f" = {(cor - cor_flag) / cor:.0%}" if cor else ""),
          f"- Flags that were real errors (PPV): {inc_flag}/{flags}" + (f" = {inc_flag / flags:.0%}" if flags else ""),
          f"- Flag rate: incorrect {inc_flag / inc:.0%} vs correct {cor_flag / cor:.0%}" if inc and cor else "", ""]

    L += ["## Rating distributions (MedGemma-stated findings)", ""]
    for name, col in (("MedGemma finding", "score_medgemma_finding"), ("CheXagent evidence", "score_chexagent_evidence"),
                      ("MedGemma change (vs prior)", "score_medgemma_change"),
                      ("CheXagent grounding (box)", "score_chexagent_grounding")):
        L.append(f"- **{name}**: {dist([num(r[col]) for r in stated])}")
        for d in sorted({dataset(r) for r in stated}):
            L.append(f"  - {d}: {dist([num(r[col]) for r in stated if dataset(r) == d])}")
    cmt = [(r["page"].split('  (')[0], r["finding"], r["comments"]) for r in rows if r["comments"].strip()]
    L += ["", "## Radiologist comments", ""] + [f"- {p} | {f}: \"{c}\"" for p, f, c in cmt]
    out = path.with_name(path.stem + "_analysis.md")
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
