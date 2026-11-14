"""Apply chexagent_revision_rule.md to the generated INTERPRETATION and score original vs revised
against the (preliminary) ground truth.

    python experiments/revise_interpretation.py REVIEW.json SCREEN.json REPORTS_TABLE.json
        -> logs/revised_interpretation_<time>.json (per page: revisions, revised headings)
        -> experiments/results/chexagent_revision_score.md (counts only)
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import provisional_findings as P  # noqa: E402
from chexagent_reliability import decide  # noqa: E402
from comparison_edv_pdf import INTERPRETATION_HEADINGS  # noqa: E402
from report_vs_ground_truth import ORDER, verdict  # noqa: E402

TERMS = {"cardiomegaly": r"cardiomegaly|(?:heart|cardiac silhouette)[^.]*\benlarged",
         "pleural effusion": r"effusion", "pneumothorax": r"pneumothora", "consolidation": r"consolidation",
         "pulmonary edema": r"edema", "atelectasis": r"atelecta"}
for term, q in P.TERMS.items():
    TERMS[q] = TERMS.get(q, "") + ("|" if q in TERMS else "") + re.escape(term)
NEGATION = re.compile(r"\b(no|without|not|negative for|free of)\b", re.I)
ADD = {"cardiomegaly": "HEART/MEDIASTINUM", "pleural effusion": "LUNGS/PLEURA"}


def positive(sentence, condition):
    """True if the sentence states the condition as present (term found, no negation before it)."""
    m = re.search(TERMS[condition], sentence, re.I)
    return bool(m) and not NEGATION.search(sentence[:m.start()])


def sentence_side(sentence):
    s = sentence.lower()
    if re.search(r"bilateral|both|bibasilar|bibasal", s):
        return "bilateral"
    r, l = "right" in s, "left" in s
    return "bilateral" if r and l else "right" if r else "left" if l else None


def box_sides(location):
    if not location:
        return set()
    if "both sides" in location:
        return {"right", "left"}
    return {part.strip().split(",")[0] for part in location.split(";")} & {"right", "left"}


def main(review_path, screen_path, table_path, score_name="chexagent_revision_score.md"):
    from medrax.evidence import EvidenceEngine

    engine = EvidenceEngine()
    review = {r["title"]: r for r in json.loads(Path(review_path).read_text(encoding="utf-8"))}
    screen = {p["title"]: {s["condition"]: s for s in p["screen"]}
              for p in json.loads(Path(screen_path).read_text(encoding="utf-8"))}
    table = {(r["page"], r["condition"]): r for r in json.loads(Path(table_path).read_text(encoding="utf-8"))}
    out, counts = {}, {"original": dict.fromkeys(ORDER, 0), "revised": dict.fromkeys(ORDER, 0)}
    log = []
    for title, rec in review.items():
        st = rec["structured"]
        sentences, labels = st["sentences"], {int(k): v for k, v in st["labels"].items()}
        dec = {c: decide(engine, c, s["p_present"]) for c, s in screen[title].items()}
        tags, revisions, flagged, corrected = {}, [], set(), {}
        for n, s in enumerate(sentences, 1):
            if labels.get(n) not in INTERPRETATION_HEADINGS:
                continue
            for c in dec:
                if not positive(s, c):
                    continue
                p = screen[title][c]["p_present"]
                if dec[c] == "AGAINST":
                    tags.setdefault(n, []).append(f"[Image check: {c} not supported ({p:.2f}) - recommend removal]")
                    flagged.add(c)
                    revisions.append(("flag", c, s))
                elif dec[c] == "FOUND":
                    side, sides = sentence_side(s), box_sides(screen[title][c].get("chexagent_location"))
                    opposite = {"right": "left", "left": "right"}.get(side)
                    if opposite and sides == {opposite}:
                        tags.setdefault(n, []).append(f"[Image check: {c} located on the {opposite} - confirm side]")
                        corrected[c] = opposite
                        revisions.append(("side", c, s))
        stated = {c for n, s in enumerate(sentences, 1) if labels.get(n) in INTERPRETATION_HEADINGS
                  for c in dec if positive(s, c)}
        added = {}
        for c, heading in ADD.items():
            if dec.get(c) == "FOUND" and c not in stated:
                sides = box_sides(screen[title][c].get("chexagent_location"))
                side = "bilateral" if len(sides) == 2 else next(iter(sides), "")
                what = f"{side} {c}".strip() if c == "pleural effusion" else c
                added[c] = (heading, side, f"[Image check: {what} ({screen[title][c]['p_present']:.2f})]")
                revisions.append(("add", c, what))
        revised = []
        for h in INTERPRETATION_HEADINGS:
            parts = [s + (" " + " ".join(f"**{t}**" for t in tags[n]) if n in tags else "")
                     for n, s in enumerate(sentences, 1) if labels.get(n) == h]
            parts += [f"**{a[2]}**" for c, a in added.items() if a[0] == h]
            revised.append([h, " ".join(parts) or "Not assessed."])
        out[title] = {"revisions": revisions, "revised_interpretation": revised}

        # scoring: original = generated report as read by the lookup; revised = after the rule
        for c in dec:
            r = table[(title, c)]
            orig = r["medgemma"]
            if c in flagged:
                new = "Not stated"
            elif c in corrected:
                new = f"Present ({corrected[c]})"
            elif c in added:
                new = f"Present ({added[c][1]})" if added[c][1] else "Present"
            else:
                new = orig
            vo, vn = verdict(orig, r["ground_truth"]), verdict(new, r["ground_truth"])
            if vo:
                counts["original"][vo] += 1
            if vn:
                counts["revised"][vn] += 1
            if vo != vn:
                log.append((title.split()[1], c, orig, vo, new, vn, r["ground_truth"]))

    path = ROOT / "logs" / f"revised_interpretation_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = ["# Generated interpretation revised with CheXagent: original vs revised, against the ground truth", "",
             "Rule: `chexagent_revision_rule.md` (fixed before applying; no ground truth used). Ground truth: preliminary "
             "automatic labels (as in the Excel comparison). " f"{len(review)} comparisons, 13 conditions.", "",
             "| verdict | original (MedGemma) | revised (MedGemma + CheXagent rule) |", "|---|---|---|"]
    lines += [f"| {k} | {counts['original'][k]} | {counts['revised'][k]} |" for k in ORDER]
    bad = lambda d: d["Missed"] + d["Wrong side"]  # noqa: E731
    lines += [f"| **Incorrect (missed or wrong side)** | **{bad(counts['original'])}** | **{bad(counts['revised'])}** |", "",
              "## What the rule changed (page, condition, before -> after, ground truth)", "",
              "| page | condition | original | verdict | revised | verdict | ground truth |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {p} | {c} | {o} | {vo or '-'} | {n} | {vn or '-'} | {g} |" for p, c, o, vo, n, vn, g in
              sorted(log, key=lambda x: int(x[0]))]
    (ROOT / "experiments" / "results" / score_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
