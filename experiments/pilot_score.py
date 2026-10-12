"""Freeze reviewed label sheets and score the pilot (model_comparison_pilot_protocol.md, A4).

    python experiments/pilot_score.py freeze SHEET.csv NAME
        -> experiments/results/pilot_NAME_labels.csv (no report text) + SHA-256 of SHEET.csv
    python experiments/pilot_score.py score RUNS_DIR [--preliminary]
        final:        labels ground_truth / model -> pilot_model_comparison.md
        preliminary:  labels ground_truth_auto / model_auto -> pilot_model_comparison_PRELIMINARY.md

A row's final label/side is the radiologist's where filled in, else the automatic one (so
freezing an unreviewed sheet gives the automatic labels). Counts only, no report text.
Intervals: Wilson for proportions; page-level bootstrap (2,000, seed 20261011) for paired
differences and AUC.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
RESULTS = ROOT / "experiments" / "results"
FLAGGED_PAGES = {"9", "10", "11", "12"}
MODELS = {"team_medgemma": "Team MedGemma", "medgemma15": "MedGemma 1.5", "chexagent": "CheXagent",
          "maira2": "MAIRA-2"}
SEED, BOOT = 20261011, 2000


def freeze(sheet, name):
    src = Path(sheet)
    out = RESULTS / f"pilot_{name}_labels.csv"
    with src.open(encoding="utf-8-sig") as f, out.open("w", newline="", encoding="utf-8") as g:
        rows = list(csv.DictReader(f))
        keys = (["model"] if "model" in rows[0] else []) + ["page", "condition", "label", "side"]
        w = csv.DictWriter(g, keys)
        w.writeheader()
        for r in rows:
            label = (r["radiologist_label"] or r["auto_label"]).strip().lower()
            label = "not reported" if label in ("not stated", "absent", "no") else label
            w.writerow({**({"model": r["model"]} if "model" in r else {}), "page": r["page"],
                        "condition": r["condition"], "label": label,
                        "side": (r["radiologist_side"] or r["auto_side"]).strip().lower()})
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    out.with_suffix(".sha256").write_text(f"{digest}  {src.name}\n", encoding="utf-8")
    print(f"-> {out}\n   source {src.name} sha256 {digest}")


def load(name):
    with (RESULTS / f"pilot_{name}_labels.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wilson(k, n):
    if n == 0:
        return "n/a"
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{k}/{n} = {p:.0%} [{max(0, centre - half):.0%}-{min(1, centre + half):.0%}]"


def rank_auc(pos, neg):
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def boot_pages(pages, stat, rng):
    """95% interval of stat(list of pages) over page-level resamples."""
    vals = [stat(list(rng.choice(pages, len(pages)))) for _ in range(BOOT)]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))


def score(runs_dir, preliminary):
    from chexagent_independent_screen import CONDITIONS, found
    from medrax.evidence import EvidenceEngine
    from vindr_grounding_amendment import box_sides

    runs = Path(runs_dir)
    raw = {m: json.loads((runs / f"{m}.json").read_text(encoding="utf-8")) for m in ("medgemma15", "chexagent", "maira2")}
    titles = {t.split()[1]: t for t in raw["chexagent"]}
    gt_name, model_name = ("ground_truth_auto", "model_auto") if preliminary else ("ground_truth", "model")
    gt = {(r["page"], r["condition"]): r for r in load(gt_name)}
    ml = {(r["model"], r["page"], r["condition"]): r for r in load(model_name)}
    engine = EvidenceEngine()
    rng = np.random.default_rng(SEED)

    stated = lambda m, p, c: ml[(m, p, c)]["label"] in ("present", "possible")  # noqa: E731
    q = lambda m, p, c: raw[m][titles[p]]["questions"][c]  # noqa: E731
    asked = {"chexagent": lambda p, c: q("chexagent", p, c)["score"] is not None
             and found(engine, c, q("chexagent", p, c)["score"])[0],
             "medgemma15": lambda p, c: q("medgemma15", p, c)["answer"].strip().lower().startswith("yes")}

    banner = ("**PRELIMINARY: scored on unreviewed automatic labels (ground truth and model reports). Not the "
              "pilot's final results.**" if preliminary else "Scored on the frozen, radiologist-reviewed labels.")
    lines = [f"# Pilot model comparison{' (PRELIMINARY)' if preliminary else ''}", "", banner, "",
             "Protocol: `experiments/model_comparison_pilot_protocol.md` (amendments A1-A4). 19 comparisons, "
             f"13 conditions; runs `{runs.name}`. Counts only. Intervals: 95%, Wilson for proportions, page-level "
             f"bootstrap ({BOOT}, seed {SEED}) for differences and AUC. Unconfirmed = stated, but the ground-truth "
             "report does not report it (not proof of a false positive). 19 images: read intervals, not point "
             "estimates; no ranking claims.", ""]

    def section(pages, possible_present, full):
        positive = lambda p, c: gt[(p, c)]["label"] == "present" or (  # noqa: E731
            possible_present and gt[(p, c)]["label"] == "possible")
        pages = sorted(pages, key=int)
        pos = [(p, c) for p in pages for c in CONDITIONS if positive(p, c)]
        neg = [(p, c) for p in pages for c in CONDITIONS if not positive(p, c)]
        out = [f"Ground-truth conditions: {len(pos)} (of {len(pages) * 13} page-conditions).", ""]

        # Experiment 1
        out += ["### Experiment 1: each model's own report", "",
                "| model | caught (95% CI) | difference vs Team MedGemma (95% CI) | unconfirmed | same side |",
                "|---|---|---|---|---|"]
        for m, name in MODELS.items():
            k = sum(stated(m, p, c) for p, c in pos)
            diff = "" if m == "team_medgemma" else (lambda lo_hi, d: f"{d:+.0%} [{lo_hi[0]:+.0%} to {lo_hi[1]:+.0%}]")(
                boot_pages(pages, lambda ps: (lambda P: (sum(stated(m, p, c) - stated("team_medgemma", p, c) for p, c in P)
                                                         / len(P)) if P else None)(
                    [(p, c) for p in ps for c in CONDITIONS if positive(p, c)]), rng),
                (k - sum(stated("team_medgemma", p, c) for p, c in pos)) / len(pos))
            sided = [(p, c) for p, c in pos if stated(m, p, c) and gt[(p, c)]["side"] in ("right", "left", "bilateral")
                     and ml[(m, p, c)]["side"] in ("right", "left", "bilateral")]
            same = sum(gt[k_]["side"] == ml[(m, *k_)]["side"] for k_ in sided)
            unconf = sum(stated(m, p, c) for p, c in neg)
            out.append(f"| {name} | {wilson(k, len(pos))} | {diff or '(reference)'} | {unconf} | {same}/{len(sided)} |")
        if not full:
            return out
        out += ["", "Per condition (caught / ground-truth count; counts only):", "",
                "| condition | ground truth | " + " | ".join(MODELS.values()) + " | CheXagent asked |",
                "|---|---|" + "---|" * (len(MODELS) + 1)]
        for c in CONDITIONS:
            cp = [(p, c_) for p, c_ in pos if c_ == c]
            cells = [f"{sum(stated(m, p, c_) for p, c_ in cp)}" for m in MODELS]
            if cp or any(stated(m, p, c) for m in MODELS for p in pages):
                out.append(f"| {c} | {len(cp)} | " + " | ".join(
                    f"{x}/{len(cp)} (+{sum(stated(m, p, c) for p, c_ in neg if c_ == c)} unconf.)"
                    for x, m in zip(cells, MODELS)) +
                    f" | {sum(asked['chexagent'](p, c_) for p, c_ in cp)}/{len(cp)} |")

        # Experiment 2
        out += ["", "### Experiment 2: direct question per condition", "",
                "| model | found (95% CI) | unconfirmed | AUC (95% CI) | no score |", "|---|---|---|---|---|"]
        for m, name in (("chexagent", "CheXagent (cut-offs)"), ("medgemma15", "MedGemma 1.5 (answer Yes)")):
            score_of = lambda p, c: q(m, p, c)["score"]  # noqa: E731
            auc_of = lambda ps: rank_auc(  # noqa: E731
                [score_of(p, c) for p in ps for c in CONDITIONS if positive(p, c) and score_of(p, c) is not None],
                [score_of(p, c) for p in ps for c in CONDITIONS if not positive(p, c) and score_of(p, c) is not None])
            auc = auc_of(pages)
            lo, hi = boot_pages(pages, auc_of, rng)
            out.append(f"| {name} | {wilson(sum(asked[m](p, c) for p, c in pos), len(pos))} | "
                       f"{sum(asked[m](p, c) for p, c in neg)} | "
                       f"{'n/a' if auc is None else f'{auc:.2f} [{lo:.2f}-{hi:.2f}]'} | "
                       f"{sum(score_of(p, c) is None for p in pages for c in CONDITIONS)} |")

        # Experiment 3
        cases = [(p, c) for p, c in pos if gt[(p, c)]["side"] in ("right", "left")]
        out += ["", f"### Experiment 3: location without the side ({len(cases)} one-sided ground-truth findings)", "",
                "| model | correct side (95% CI) | wrong side | both sides | not located | no answer (cut off) |",
                "|---|---|---|---|---|---|"]
        for m, name in (("chexagent", "CheXagent"), ("maira2", "MAIRA-2"), ("medgemma15", "MedGemma 1.5")):
            t = {"correct": 0, "wrong": 0, "both": 0, "none": 0, "noanswer": 0}
            for p, c in cases:
                boxes = raw[m][titles[p]]["locations"][c]["boxes"]
                if boxes is None:
                    t["noanswer"] += 1
                    continue
                sides = box_sides([b["box"] for b in boxes])
                t["none" if not sides else "both" if len(sides) > 1 else
                  "correct" if gt[(p, c)]["side"] in sides else "wrong"] += 1
            out.append(f"| {name} | {wilson(t['correct'], len(cases))} | {t['wrong']} | {t['both']} | {t['none']} | "
                       f"{t['noanswer'] if m == 'medgemma15' else '-'} |")
        return out

    lines += ["## Primary: all 19 pages, \"possible\" counted as present", ""]
    lines += section(set(titles), True, True)
    for name, pages, pp in (("Sensitivity: excluding pages 9-12", set(titles) - FLAGGED_PAGES, True),
                            ("Sensitivity: \"possible\" counted as not reported", set(titles), False),
                            ("Sensitivity: excluding pages 9-12 and \"possible\" as not reported",
                             set(titles) - FLAGGED_PAGES, False)):
        lines += ["", f"## {name}", ""] + section(pages, pp, False)
    out = RESULTS / f"pilot_model_comparison{'_PRELIMINARY' if preliminary else ''}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["freeze", "score"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--preliminary", action="store_true")
    a = ap.parse_args()
    if a.command == "freeze":
        freeze(*a.args)
    else:
        score(a.args[0], a.preliminary)
