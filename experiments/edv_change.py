"""edv_change_protocol.md: change check from separate prior / current CheXagent presence verdicts (499 NIH pairs).

    .venv/Scripts/python.exe experiments/edv_change.py read      # change claims in the trained reports (Ollama, resumable)
    .venv/Scripts/python.exe experiments/edv_change.py analyze   # -> results/edv_change.md
"""

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import edv_improvement as E  # noqa: E402
import edv_longitudinal as L  # noqa: E402
import edv_trained as T  # noqa: E402
from maira_score_nih import auc  # noqa: E402

RESULTS = HERE / "results"
CLAIMS = RESULTS / "edv_change_claims.jsonl"
REPORT = RESULTS / "edv_change.md"
META = Path("D:/datasets/chestxray14/Data_Entry_2017_v2020.csv")
CONDITIONS = E.CONDITIONS
NIH_NAME = {"pleural effusion": "Effusion", "pulmonary edema": "Edema", "pleural thickening": "Pleural_Thickening"}
SEED, N_BOOT = 20261101, 2000
KINDS = ["new", "unchanged", "worse", "better", "resolved", "no change stated"]

CHANGE_PROMPT = """You read a chest X-ray report that compares a current study with a prior study.
For each condition below, say what the report states about how it changed since the prior study.
Answer with exactly one of: "new", "unchanged", "worse", "better", "resolved", "no change stated".
- "new": the report says the condition is new, newly appeared, or not seen on the prior study.
- "unchanged": stable, unchanged, persistent, similar to or again seen as on the prior study.
- "worse": increased, enlarged, worsening, progressed since the prior study.
- "better": decreased, smaller, improving, partially resolved since the prior study.
- "resolved": resolved, cleared, no longer seen.
- "no change stated": the report mentions the condition but does not say how it compares with the prior study.
"evidence": the exact words from the report (copied verbatim) that state the change; empty for "no change stated".

Report:
{report}

Conditions:
{conditions}

Answer only with JSON: {{"<condition>": {{"answer": "...", "evidence": "..."}}, ...}}"""


def jl(path, key="current"):
    return {json.loads(x)[key]: json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()}


NOT_MENTIONED = ('- "not mentioned": the report does not mention the condition, or only says it is absent without '
                 'comparing with the prior study.\n')


def read_claims(extractor, report, conditions, not_mentioned=False):
    """{condition: (claim, evidence)}; a claim whose quote is not in the report becomes "no change stated".
    not_mentioned: also offer "not mentioned" (for reading every condition, not only the stated ones)."""
    prompt = CHANGE_PROMPT.format(report=report.strip(), conditions="\n".join(f"- {c}" for c in conditions))
    kinds = KINDS + ["not mentioned"] if not_mentioned else KINDS
    if not_mentioned:
        prompt = prompt.replace('"resolved", "no change stated".', '"resolved", "no change stated", "not mentioned".')
        prompt = prompt.replace('"evidence": the exact', NOT_MENTIONED + '"evidence": the exact')
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()  # noqa: E731
    for _ in range(3):
        text = extractor.invoke(prompt).content
        m = re.search(r"\{.*\}", text, re.S)
        try:
            answer = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            answer = None
        if isinstance(answer, dict) and all(isinstance(answer.get(c), dict) for c in conditions):
            out = {}
            for c in conditions:
                claim = str(answer[c].get("answer", "")).strip().lower()
                evidence = str(answer[c].get("evidence", "")).strip()
                quotes = [norm(q) for q in re.split(r"(?<=[.;])\s+", evidence) if norm(q)]
                if claim == "not mentioned" and not_mentioned:
                    pass
                elif claim not in kinds or claim == "no change stated" or not quotes \
                        or any(q not in norm(report) for q in quotes):
                    claim = "no change stated"
                out[c] = (claim, evidence)
            return out
    raise RuntimeError(f"no usable change answer for {conditions}: {text!r}")


def read():
    from comparison_edv_pdf import make_extractor
    extractor = make_extractor("qwen3.8-edv:latest")
    reports, labels = jl(T.REPORTS), jl(T.LABELS)
    done = set(jl(CLAIMS)) if CLAIMS.exists() else set()
    todo = [f for f in reports if f not in done]
    print(f"read: {len(todo)} to go", flush=True)
    start = time.time()
    with CLAIMS.open("a", encoding="utf-8") as out:
        for n, f in enumerate(todo, 1):
            stated = [c for c in CONDITIONS if labels[f]["labels"][c]["score"] > 0]
            ans = read_claims(extractor, reports[f]["report"], stated) if stated else {}
            out.write(json.dumps({"current": f, "claims": {c: {"claim": a[0], "evidence": a[1]}
                                                           for c, a in ans.items()}}) + "\n")
            out.flush()
            if n % 25 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"read {n}/{len(todo)}  {rate:.3f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


def analyze():
    from chexagent_reliability import decide
    from medrax.evidence import EvidenceEngine
    engine = EvidenceEngine()
    pairs = json.loads(L.PAIRS.read_text(encoding="utf-8"))
    with open(META, encoding="utf-8") as fh:
        meta = {r["Image Index"]: r for r in csv.DictReader(fh)}
    has = lambda f, c: NIH_NAME.get(c, c.capitalize()) in meta[f]["Finding Labels"].split("|")  # noqa: E731
    truth, _, chex_s, _, _ = E.load()
    prior_read = jl(L.CHEX_PRIOR)
    claims = jl(CLAIMS) if CLAIMS.exists() else {}

    rows = []  # one per pair-condition
    for p in pairs:
        f, pr = p["current"], p["prior"]
        for c in CONDITIONS:
            assert has(f, c) == bool(truth[(f, c)]), (f, c)  # same current labels as the earlier analyses
            a, b = has(pr, c), has(f, c)
            ref = "new" if b and not a else "resolved" if a and not b else "persistent" if a and b else "absent"
            pp, pc = prior_read[f]["p_present"][c], chex_s[(f, c)]
            dp = decide(engine, c, pp) if pp is not None else "UNCERTAIN"
            dc = decide(engine, c, pc) if pc is not None else "UNCERTAIN"
            derived = "Uncertain" if "UNCERTAIN" in (dp, dc) else {
                ("AGAINST", "FOUND"): "New", ("FOUND", "AGAINST"): "Resolved", ("FOUND", "FOUND"): "Persistent",
                ("AGAINST", "AGAINST"): "Absent"}[(dp, dc)]
            rows.append(dict(current=f, patient=meta[f]["Patient ID"], same_view=p["same_view"], c=c, ref=ref,
                             prior_label=a, p_prior=pp, dp=dp, dc=dc, derived=derived,
                             claim=claims.get(f, {}).get("claims", {}).get(c, {}).get("claim")))
    # percentile of each prior reading among the 499 prior readings for its condition (label-free)
    by_c = defaultdict(list)
    for r in rows:
        if r["p_prior"] is not None:
            by_c[r["c"]].append(r["p_prior"])
    srt = {c: np.sort(v) for c, v in by_c.items()}
    for r in rows:
        r["pct"] = None if r["p_prior"] is None else \
            np.searchsorted(srt[r["c"]], r["p_prior"], side="right") / len(srt[r["c"]])

    patients = sorted({r["patient"] for r in rows})
    rows_of = defaultdict(list)
    for i, r in enumerate(rows):
        rows_of[r["patient"]].append(i)
    rng = np.random.default_rng(SEED)
    boot_idx = [np.concatenate([rows_of[patients[k]] for k in rng.integers(0, len(patients), len(patients))])
                for _ in range(N_BOOT)]

    def pooled_auc(sel, pos, neg):
        """AUC of the prior percentile, pos vs neg reference categories, over rows passing sel."""
        def one(idx):
            rs = [rows[i] for i in idx if sel(rows[i]) and rows[i]["pct"] is not None]
            return auc(np.array([r["pct"] for r in rs if r["ref"] == pos]),
                       np.array([r["pct"] for r in rs if r["ref"] == neg]))
        full = one(range(len(rows)))
        bs = [one(b) for b in boot_idx]
        lo, hi = np.percentile([v for v in bs if np.isfinite(v)], [2.5, 97.5])
        n_pos = sum(1 for r in rows if sel(r) and r["ref"] == pos and r["pct"] is not None)
        n_neg = sum(1 for r in rows if sel(r) and r["ref"] == neg and r["pct"] is not None)
        return full, lo, hi, n_pos, n_neg

    lines = ["# Change check (separate prior / current CheXagent verdicts): results", "",
             f"Protocol: `edv_change_protocol.md` (committed before this analysis). {len(pairs)} NIH test pairs, "
             f"{len(CONDITIONS)} conditions, {len(patients)} patients. 95% CIs from {N_BOOT} patient-level bootstrap "
             f"resamples (seed {SEED}).", ""]

    # Analysis 1 (primary)
    a1 = pooled_auc(lambda r: True, "persistent", "new")
    verdict = "supported" if a1[1] > 0.5 else "reversed" if a1[2] < 0.5 else "not supported"
    lines += ["## Analysis 1 (primary): prior-image reading, persistent vs new", "",
              "| comparison | persistent / new | AUC [95% CI] | result |", "|---|---|---|---|",
              f"| H5: pooled AUC, current labelled positive | {a1[3]} / {a1[4]} | {a1[0]:.3f} [{a1[1]:.3f}-{a1[2]:.3f}] "
              f"| {verdict} |", "",
              "Reference (NIH labels) by CheXagent's prior-image verdict, current labelled positive:", "",
              "| reference | prior Not supported | prior Supported | prior Indeterminate |", "|---|---|---|---|"]
    for ref in ("new", "persistent"):
        cnt = Counter(r["dp"] for r in rows if r["ref"] == ref)
        lines.append(f"| {ref} | {cnt['AGAINST']} | {cnt['FOUND']} | {cnt['UNCERTAIN']} |")

    # Analysis 2 (secondary)
    stated = [r for r in rows if r["claim"] is not None]
    checkable = [r for r in stated if r["claim"] != "no change stated"]
    implies_prior = {"new": False, "unchanged": True, "worse": True, "better": True, "resolved": True}

    def flagged(r):
        return (r["claim"] == "new" and r["dp"] == "FOUND") or \
               (r["claim"] in ("unchanged", "worse", "better") and r["dp"] == "AGAINST") or \
               (r["claim"] == "resolved" and r["dc"] == "FOUND")
    error = lambda r: implies_prior[r["claim"]] != r["prior_label"]  # noqa: E731
    lines += ["", "## Analysis 2 (secondary): MedGemma's change wording", "",
              f"Change claims were read for {len(stated)} stated conditions "
              f"({'reader not yet run' if not claims else f'{len(claims)} reports read'}).", "",
              "| claim | n | flagged by CheXagent | label error (prior-image implication vs prior label) |",
              "|---|---|---|---|"]
    for k in KINDS:
        rs = [r for r in stated if r["claim"] == k]
        if k == "no change stated":
            lines.append(f"| {k} | {len(rs)} | - | - |")
        else:
            lines.append(f"| {k} | {len(rs)} | {sum(map(flagged, rs))} | {sum(map(error, rs))} |")
    if len(checkable) < 20:
        lines += ["", f"{len(checkable)} checkable change claims, fewer than 20: reported descriptively only "
                      "(feasibility gate in the protocol)."]
        for r in checkable:
            ev = claims[r["current"]]["claims"][r["c"]]["evidence"]
            lines.append(f"- {r['current']} {r['c']}: {r['claim']} (\"{ev}\"); CheXagent prior {r['dp']}, current "
                         f"{r['dc']}; prior label {'yes' if r['prior_label'] else 'no'}; "
                         f"{'flagged' if flagged(r) else 'not flagged'}")
    else:
        idx_c = {id(r): i for i, r in enumerate(rows)}
        check_set = {idx_c[id(r)] for r in checkable}

        def err_diff(idx):
            rs = [rows[i] for i in idx if i in check_set]
            fl, un = [error(r) for r in rs if flagged(r)], [error(r) for r in rs if not flagged(r)]
            return (np.mean(fl) if fl else np.nan) - (np.mean(un) if un else np.nan), fl, un
        d, fl, un = err_diff(range(len(rows)))
        bs = [err_diff(b)[0] for b in boot_idx]
        lo, hi = np.percentile([v for v in bs if np.isfinite(v)], [2.5, 97.5])
        lines += ["", f"Label error rate: flagged {np.mean(fl) if fl else float('nan'):.3f} (n={len(fl)}), "
                      f"not flagged {np.mean(un) if un else float('nan'):.3f} (n={len(un)}); "
                      f"difference {d:+.3f} [{lo:+.3f} to {hi:+.3f}]."]

    # Exploratory
    sv = pooled_auc(lambda r: r["same_view"], "persistent", "new")
    rv = pooled_auc(lambda r: True, "resolved", "absent")
    lines += ["", "## Exploratory", "", "| comparison | n (pos / neg) | AUC [95% CI] |", "|---|---|---|",
              f"| persistent vs new, same-view pairs only | {sv[3]} / {sv[4]} | {sv[0]:.3f} [{sv[1]:.3f}-{sv[2]:.3f}] |",
              f"| resolved vs absent, current labelled negative | {rv[3]} / {rv[4]} | "
              f"{rv[0]:.3f} [{rv[1]:.3f}-{rv[2]:.3f}] |"]
    lines += ["", "Per condition (prior-image percentile; shown where both categories have at least 10):", "",
              "| condition | persistent / new | AUC | resolved / absent | AUC |", "|---|---|---|---|---|"]
    for c in CONDITIONS:
        cells = []
        for pos, neg in (("persistent", "new"), ("resolved", "absent")):
            p_ = [r["pct"] for r in rows if r["c"] == c and r["ref"] == pos and r["pct"] is not None]
            n_ = [r["pct"] for r in rows if r["c"] == c and r["ref"] == neg and r["pct"] is not None]
            cells += [f"{len(p_)} / {len(n_)}",
                      f"{auc(np.array(p_), np.array(n_)):.3f}" if min(len(p_), len(n_)) >= 10 else "-"]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    derived = ["New", "Resolved", "Persistent", "Absent", "Uncertain"]
    lines += ["", "Reference change (NIH labels) by derived change (CheXagent), all pair-conditions:", "",
              "| reference \\ derived | " + " | ".join(derived) + " |", "|---|" + "---|" * len(derived)]
    for ref in ("new", "resolved", "persistent", "absent"):
        cnt = Counter(r["derived"] for r in rows if r["ref"] == ref)
        lines.append(f"| {ref} | " + " | ".join(str(cnt[d]) for d in derived) + " |")
    lines += ["", "Limitations (recorded in the protocol): NIH labels text-mined per image (change from two noisy "
                  "labels; no worse/better reference); view and technique differences between prior and current; "
                  "CheXagent probably saw NIH images; cut-offs set for presence, not change; prior readings generated "
                  "earlier for option B (disclosed); trained MedGemma without the prior report; pooled percentiles."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"read": read, "analyze": analyze}[sys.argv[1]]()
