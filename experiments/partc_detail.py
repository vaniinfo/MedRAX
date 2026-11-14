"""EXPLORATORY detail for Part C of the CheXagent vs MAIRA-2 document (2026-10-25), from committed readings:
per-condition AUC of every system in the three downstream experiments, the editing rule's changes by condition,
and example cases chosen by a fixed rule (first image by file name in each category, trained-model setting).

    python experiments/partc_detail.py   -> results/partc_detail.json and results/partc_detail.md
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import edv_improvement as E  # noqa: E402
import edv_longitudinal as L  # noqa: E402
import review_editing_rule as R  # noqa: E402
from maira_score_nih import auc  # noqa: E402

RESULTS = HERE / "results"
OUT_JSON, OUT_MD = RESULTS / "partc_detail.json", RESULTS / "partc_detail.md"
C = E.CONDITIONS


def setting(which, data):
    """files, y, report, CheXagent, MAIRA-2, fused CheXagent, fused MAIRA-2 per condition (kept rows only)."""
    truth, mg, chex, maira, devd = data
    if which == "a":
        files = E.split_files("test")
        pct = lambda m, c, s: np.searchsorted(devd[(m, c)], s, side="right") / len(devd[(m, c)])  # noqa: E731
        out = {}
        for c in C:
            y = np.array([truth[(f, c)] for f in files])
            m = np.array([mg[(f, c)] for f in files])
            ch = np.array([chex[(f, c)] for f in files])
            ma = np.array([maira[(f, c)] for f in files])
            out[c] = dict(files=list(files), y=y, report=m, chex=ch, maira=ma,
                          fused_chex=0.5 * m + 0.5 * pct("chex", c, ch), fused_maira=0.5 * m + 0.5 * pct("maira", c, ma))
        return out
    pairs = json.loads(L.PAIRS.read_text(encoding="utf-8"))
    files = [p["current"] for p in pairs]
    jl = lambda path: {json.loads(x)["current"]: json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()}  # noqa: E731
    a, ml, prior = jl(L.CHEX_A), jl(L.MAIRA_L), jl(L.CHEX_PRIOR)
    lab = jl(L.LABELS) if which == "b" else jl(RESULTS / "edv_trained_labels.jsonl")
    pct = lambda v: np.searchsorted(np.sort(v), v, side="right") / len(v)  # noqa: E731
    out = {}
    for c in C:
        keep = [f for f in files if a[f]["p_present"][c] is not None and prior[f]["p_present"][c] is not None]
        y = np.array([truth[(f, c)] for f in keep])
        m = np.array([lab[f]["labels"][c]["score"] for f in keep], float)
        ch = np.array([a[f]["p_present"][c] for f in keep], float)
        ma = np.array([ml[f]["findings"][c]["score"] for f in keep], float)
        out[c] = dict(files=keep, y=y, report=m, chex=ch, maira=ma, fused_chex=0.5 * m + 0.5 * pct(ch),
                      fused_maira=0.5 * m + 0.5 * pct(ma))
    return out


def edits(d):
    """Editing rule changes per condition: added/removed x agrees/disagrees with the NIH label."""
    rows = {}
    for c in C:
        stated = d[c]["report"] >= 0.5
        n = dict(added_agree=0, added_disagree=0, removed_agree=0, removed_disagree=0)
        for st, p, t in zip(stated, d[c]["chex"], d[c]["y"]):
            dec, _ = R.decide(c, float(p))
            if st and dec == "AGAINST":
                n["removed_agree" if t == 0 else "removed_disagree"] += 1
            elif (not st) and dec == "FOUND" and c in R.ADD:
                n["added_agree" if t == 1 else "added_disagree"] += 1
        rows[c] = n
    return rows


def examples(d, reports):
    """First case by file name in each category (trained-model setting)."""
    cats = {"stated, CheXagent supports, label present": lambda st, dec, t: st and dec == "FOUND" and t == 1,
            "not stated, CheXagent supports, label present (a catch)": lambda st, dec, t: (not st) and dec == "FOUND" and t == 1,
            "not stated, CheXagent supports, label absent (a false alarm)": lambda st, dec, t: (not st) and dec == "FOUND" and t == 0,
            "stated, CheXagent against, label absent (a correct removal)": lambda st, dec, t: st and dec == "AGAINST" and t == 0}
    found = {}
    for name, rule in cats.items():
        best = None
        for c in ("pleural effusion", "cardiomegaly", "pneumothorax", "consolidation", "atelectasis", "pulmonary edema"):
            for f, m, p, t in zip(d[c]["files"], d[c]["report"], d[c]["chex"], d[c]["y"]):
                dec, _ = R.decide(c, float(p))
                if rule(m >= 0.5, dec, t) and (best is None or (f, c) < (best[0], best[1])):
                    best = (f, c, float(m), float(p), dec, int(t))
        if best:
            f, c, m, p, dec, t = best
            found[name] = dict(file=f, condition=c, report_score=m, p_present=round(p, 3), decision=dec, label=t,
                               report=reports[f]["report"])
    return found


def main():
    data = E.load()
    out = {"settings": {}}
    names = {"a": "1. Base MedGemma, single image (1,500 NIH test images)",
             "b": "2. Base MedGemma, prior + current (499 NIH pairs)",
             "c": "3. Trained MedGemma, prior + current (499 NIH pairs)"}
    md = ["# EXPLORATORY: Part C detail (per condition, edits, examples)", "",
          "Computed on 2026-10-25 from committed readings. Combined-score AUCs match the pre-registered results; "
          "checker-alone AUCs are the exploratory baselines.", ""]
    for k in ("a", "b", "c"):
        d = setting(k, data)
        rows = []
        for c in C:
            y = d[c]["y"]
            pos, neg = y == 1, y == 0
            A = {s: auc(d[c][s][pos], d[c][s][neg]) for s in ("report", "fused_chex", "fused_maira", "chex", "maira")}
            rows.append(dict(condition=c, positives=int(pos.sum()), n=len(y), **{s: round(float(v), 3) for s, v in A.items()}))
        ed = edits(d)
        out["settings"][k] = dict(name=names[k], rows=rows, edits=ed)
        md += [f"## {names[k]}", "", "| condition | positives / n | report | + CheXagent | + MAIRA-2 | CheXagent alone | "
               "MAIRA-2 alone |", "|---|---|---|---|---|---|---|"]
        md += [f"| {r['condition']} | {r['positives']} / {r['n']} | {r['report']:.3f} | {r['fused_chex']:.3f} | "
               f"{r['fused_maira']:.3f} | {r['chex']:.3f} | {r['maira']:.3f} |" for r in rows]
        md += ["", "| condition | added, agrees | added, disagrees | removed, agrees | removed, disagrees |",
               "|---|---|---|---|---|"]
        md += [f"| {c} | {v['added_agree']} | {v['added_disagree']} | {v['removed_agree']} | {v['removed_disagree']} |"
               for c, v in ed.items() if any(v.values())]
        md.append("")
        if k == "c":
            reports = {json.loads(x)["current"]: json.loads(x)
                       for x in (RESULTS / "edv_trained_reports.jsonl").read_text(encoding="utf-8").splitlines()}
            out["examples"] = examples(d, reports)
    md += ["## Examples (trained model; first by file name in each category)", ""]
    for name, e in out["examples"].items():
        md += [f"- **{name}:** {e['file']}, {e['condition']}, report score {e['report_score']}, CheXagent p_present "
               f"{e['p_present']} ({e['decision']}), NIH label {e['label']}"]
    for path in (OUT_JSON, OUT_MD):
        if path.exists():
            raise SystemExit(f"{path} exists")
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
