"""Manual validation of the report reader (protocol: reader_validation_protocol.md).

    python experiments/reader_validation.py sample              -> results/reader_validation_sheet.csv (+ key)
    python experiments/reader_validation.py analyze <sheet.csv> -> results/reader_validation.md

The sheet has no reader output; the key (setting and report ID per row) is kept in a separate file.
"""

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import edv_improvement as E  # noqa: E402

RESULTS = HERE / "results"
SHEET, KEY, REPORT = (RESULTS / "reader_validation_sheet.csv", RESULTS / "reader_validation_key.json",
                      RESULTS / "reader_validation.md")
SEED, N_PER, N_BOOT = 20261027, 30, 2000
SETTINGS = {  # setting: (reports, labels, id field)
    "a": ("edv_medgemma_reports.jsonl", "edv_medgemma_labels.jsonl", "file"),
    "b": ("edv_long_medgemma_reports.jsonl", "edv_long_medgemma_labels.jsonl", "current"),
    "c": ("edv_trained_reports.jsonl", "edv_trained_labels.jsonl", "current"),
}
CODE_SCORE = {"P": 1.0, "U": 0.5, "N": 0.0, "R": 0.0}


def jl(name, key):
    return {r[key]: r for r in map(json.loads, (RESULTS / name).read_text(encoding="utf-8").splitlines())}


def sample():
    if SHEET.exists() or KEY.exists():
        raise SystemExit("sheet or key exists; not overwriting")
    rng = random.Random(SEED)
    rows = []
    for s, (rep, _, key) in SETTINGS.items():
        reports = jl(rep, key)
        for rid in rng.sample(sorted(reports), N_PER):
            rows.append((s, rid, reports[rid]["report"]))
    rng.shuffle(rows)
    with open(SHEET, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "report"] + E.CONDITIONS + ["notes"])
        for i, (_, _, text) in enumerate(rows, 1):
            w.writerow([i, text] + [""] * len(E.CONDITIONS) + [""])
    KEY.write_text(json.dumps({i: [s, rid] for i, (s, rid, _) in enumerate(rows, 1)}, indent=1), encoding="utf-8")
    print(f"{len(rows)} reports -> {SHEET}")


def kappa_linear(a, b, cats=(0.0, 0.5, 1.0)):
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    wt = np.array([[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)])
    return 1 - (wt * obs).sum() / (wt * exp).sum()


def analyze(path):
    key = {int(k): v for k, v in json.loads(KEY.read_text(encoding="utf-8")).items()}
    labels = {s: jl(lab, idf) for s, (_, lab, idf) in SETTINGS.items()}
    recs = []  # (setting, row, condition, reader score, annotator code)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s, rid = key[int(r["row"])]
            for c in E.CONDITIONS:
                code = r[c].strip().upper()
                if code not in CODE_SCORE:
                    raise SystemExit(f"row {r['row']} {c}: code {code!r} not in P/U/N/R")
                recs.append((s, int(r["row"]), c, labels[s][rid]["labels"][c]["score"], code))
    out = ["# Report reader validation (manual review)", "", "Protocol: `reader_validation_protocol.md`.", ""]
    rng = np.random.default_rng(SEED)
    for name, sub in [("pooled", recs)] + [(f"setting {s}", [x for x in recs if x[0] == s]) for s in SETTINGS]:
        rd = np.array([x[3] for x in sub])
        an = np.array([CODE_SCORE[x[4]] for x in sub])
        rows = np.array([x[1] for x in sub])
        uniq = np.unique(rows)
        agree = (rd == an).mean()
        boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(uniq, len(uniq))
            m = np.concatenate([np.flatnonzero(rows == u) for u in pick])
            boots.append((rd[m] == an[m]).mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pos, neg = an > 0, an == 0
        sens = ((rd > 0) & pos).sum() / max(pos.sum(), 1)
        spec = ((rd == 0) & neg).sum() / max(neg.sum(), 1)
        codes = np.array([x[4] for x in sub])
        err = {"negated/not mentioned read as stated": int(((rd > 0) & (codes == "N")).sum()),
               "prior-only read as current": int(((rd > 0) & (codes == "R")).sum()),
               "certainty wrong (1 vs U, or 0.5 vs P)": int(((rd == 1) & (codes == "U")).sum() +
                                                           ((rd == 0.5) & (codes == "P")).sum()),
               "missed (reader 0, annotator P or U)": int(((rd == 0) & pos).sum())}
        out += [f"## {name} ({len(uniq)} reports, {len(sub)} decisions)", "",
                f"- Exact agreement: {agree:.3f} [{lo:.3f}-{hi:.3f}]; linearly weighted kappa {kappa_linear(rd, an):.3f}",
                f"- Stated (score > 0) vs annotator P/U: sensitivity {sens:.3f} ({int(pos.sum())} annotator-stated), "
                f"specificity {spec:.3f}"] + [f"- {k}: {v}" for k, v in err.items()] + [""]
    if REPORT.exists():
        raise SystemExit(f"{REPORT} exists")
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    if sys.argv[1:2] == ["sample"]:
        sample()
    elif sys.argv[1:2] == ["analyze"] and len(sys.argv) == 3:
        analyze(sys.argv[2])
    else:
        raise SystemExit(__doc__)
