"""Exploratory pilot of the change check (edv_change_protocol.md rules) on the 20 MIMIC-CXR cases of the two
radiologist scoring forms (mixed_scoring_form_A/B_v5). Not part of the NIH protocol; descriptive only.

MIMIC has the radiologist's report of the current study, which compares with the prior, so the reference
change comes from that report (not from per-image labels). CheXagent reads the prior and the current image
separately ("Is there {condition}?"); each reading gets its Evidence Engine verdict; the two verdicts give the
derived change. The team's MedGemma report (as in the forms) is read for change claims the same way.

MIMIC text is under the PhysioNet DUA: outputs go to logs/ (git-ignored).

    .venv/Scripts/python.exe experiments/edv_change_mimic_pilot.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import edv_change as X  # noqa: E402
import mixed_scoring_pdf as M  # noqa: E402
from maira_score_nih import auc  # noqa: E402

LOGS = ROOT / "logs"
FORMS = [LOGS / "mixed_scoring_form_A_v5.json", LOGS / "mixed_scoring_form_B_v5.json"]
CACHE = LOGS / "edv_change_mimic_pilot_readings.json"  # reader + CheXagent outputs, reused if present
CONDITIONS = X.CONDITIONS
PRIOR_IMPLIED = {"new": False, "unchanged": True, "worse": True, "better": True, "resolved": True}
GROUP = {"new": "New", "unchanged": "Persistent", "worse": "Persistent", "better": "Persistent",
         "resolved": "Resolved"}


def cases():
    wanted = [(s, x["source"], x["source_title"]) for s, f in zip("AB", FORMS)
              for x in json.loads(f.read_text(encoding="utf-8")) if x["dataset"] == "MIMIC-CXR"]
    by_key = {(c["source"], c["rec"]["title"]): c for c in M.load_mimic()}
    out = []
    for form, source, title in wanted:
        c = by_key[(source, title)]
        current = Path(c["rec"]["image"])
        out.append(dict(form=form, title=title, source=source, current=str(current),
                        prior=str(current.with_name("prior.jpg")), subject=current.parent.name.split("_subject_")[-1],
                        generated=c["page"]["generated_text"], truth=c["page"]["truth_text"]))
    assert len(out) == 20 and all(Path(c["prior"]).exists() for c in out)
    return out


def run(cs):
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    from comparison_edv_pdf import make_extractor
    extractor = make_extractor("qwen3.8-edv:latest")
    for c in cs:
        r = cache.setdefault(c["current"], {})
        for key, text in (("truth_claims", c["truth"]), ("medgemma_claims", c["generated"])):
            if key not in r:
                r[key] = {k: list(v) for k, v in X.read_claims(extractor, text, CONDITIONS, not_mentioned=True).items()}
                CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        print(f"read {c['title']}", flush=True)
    M.unload_ollama("qwen3.8-edv:latest")
    tool = None
    for c in cs:
        r = cache[c["current"]]
        for key, path in (("p_prior", c["prior"]), ("p_current", c["current"])):
            if key in r:
                continue
            if tool is None:
                from medrax.tools.xray_vqa import XRayVQATool
                tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
            r[key] = {}
            for cond in CONDITIONS:
                _, yn = tool._generate_response([path], f"Is there {cond}?", 8)
                r[key][cond] = yn["p_yes"] if yn else None
            CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        print(f"CheXagent {c['title']}", flush=True)
    return cache


def analyze(cs, cache):
    from chexagent_reliability import decide
    from medrax.evidence import EvidenceEngine
    engine = EvidenceEngine()
    rows = []
    for c in cs:
        r = cache[c["current"]]
        for cond in CONDITIONS:
            pp, pc = r["p_prior"][cond], r["p_current"][cond]
            dp = decide(engine, cond, pp) if pp is not None else "UNCERTAIN"
            dc = decide(engine, cond, pc) if pc is not None else "UNCERTAIN"
            derived = "Uncertain" if "UNCERTAIN" in (dp, dc) else {
                ("AGAINST", "FOUND"): "New", ("FOUND", "AGAINST"): "Resolved", ("FOUND", "FOUND"): "Persistent",
                ("AGAINST", "AGAINST"): "Absent"}[(dp, dc)]
            gt, mg = r["truth_claims"][cond], r["medgemma_claims"][cond]
            rows.append(dict(case=f"{c['form']}: {c['title']}", subject=c["subject"], cond=cond, p_prior=pp,
                             p_current=pc, dp=dp, dc=dc, derived=derived, gt=gt[0], gt_ev=gt[1], mg=mg[0],
                             mg_ev=mg[1]))

    def flagged(claim, r):
        return (claim == "new" and r["dp"] == "FOUND") or \
               (claim in ("unchanged", "worse", "better") and r["dp"] == "AGAINST") or \
               (claim == "resolved" and r["dc"] == "FOUND")
    short = {"FOUND": "Supported", "AGAINST": "Not supported", "UNCERTAIN": "Indeterminate"}
    gt_change = [r for r in rows if r["gt"] in PRIOR_IMPLIED]
    mg_change = [r for r in rows if r["mg"] in PRIOR_IMPLIED]
    lines = ["# Change check, MIMIC-CXR pilot (20 cases of the scoring forms): results", "",
             "Exploratory and descriptive; not part of `edv_change_protocol.md` (NIH). Reference change = what the "
             "radiologist's report of the current study says about each condition compared with the prior (read by "
             "the local model with a verbatim quote check). CheXagent read each image separately; verdicts from the "
             "Evidence Engine (same cut-offs, provisional cut-offs and fallback as everywhere else). "
             f"{len(cs)} cases, {len({c['subject'] for c in cs})} patients, {len(CONDITIONS)} conditions. "
             "MIMIC-CXR text: PhysioNet DUA, do not share outside credentialed people.", "",
             "## Change wording found", "", "| claim | radiologist report | MedGemma report |", "|---|---|---|"]
    for k in X.KINDS + ["not mentioned"]:
        lines.append(f"| {k} | {sum(r['gt'] == k for r in rows)} | {sum(r['mg'] == k for r in rows)} |")

    # 1. CheXagent's prior-image verdict vs what the radiologist's change wording implies about the prior
    lines += ["", "## 1. Prior image: CheXagent's verdict vs the radiologist's change wording", "",
              "Radiologist says new (absent on prior) or unchanged/worse/better/resolved (present on prior).", "",
              "| radiologist implies on prior | CheXagent prior Supported | Not supported | Indeterminate |",
              "|---|---|---|---|"]
    for implied, name in ((True, "present"), (False, "absent")):
        cnt = Counter(r["dp"] for r in gt_change if PRIOR_IMPLIED[r["gt"]] == implied)
        lines.append(f"| {name} | {cnt['FOUND']} | {cnt['AGAINST']} | {cnt['UNCERTAIN']} |")
    pos = [r["p_prior"] for r in gt_change if PRIOR_IMPLIED[r["gt"]] and r["gt"] != "resolved" and r["p_prior"] is not None]
    neg = [r["p_prior"] for r in gt_change if r["gt"] == "new" and r["p_prior"] is not None]
    lines += ["", f"Persistent (unchanged/worse/better) vs new, prior-image p_present: {len(pos)} vs {len(neg)}; AUC "
                  f"{auc(np.array(pos), np.array(neg)):.3f} (raw scores pooled over conditions; too few for an "
                  "interval)." if pos and neg else
              f"Persistent vs new: {len(pos)} vs {len(neg)} - too few to compute an AUC."]

    # 2. derived change vs the radiologist's change
    derived = ["New", "Resolved", "Persistent", "Absent", "Uncertain"]
    lines += ["", "## 2. Derived change (CheXagent) vs the radiologist's change wording", "",
              "| radiologist \\ CheXagent | " + " | ".join(derived) + " |", "|---|" + "---|" * len(derived)]
    for g in ("New", "Persistent", "Resolved"):
        cnt = Counter(r["derived"] for r in gt_change if GROUP[r["gt"]] == g)
        lines.append(f"| {g} | " + " | ".join(str(cnt[d]) for d in derived) + " |")

    # 3. MedGemma's change claims, CheXagent's flags, and the radiologist's report as reference
    def mg_wrong(r):  # MedGemma's implication about the prior contradicts the radiologist's
        return PRIOR_IMPLIED[r["mg"]] != PRIOR_IMPLIED[r["gt"]] if r["gt"] in PRIOR_IMPLIED else None
    lines += ["", "## 3. MedGemma's change claims checked by CheXagent", "",
              f"{len(mg_change)} change claims in the MedGemma reports; {sum(flagged(r['mg'], r) for r in mg_change)} "
              "flagged by CheXagent.", ""]
    judged = [r for r in mg_change if mg_wrong(r) is not None]
    for name, sel in (("flagged", True), ("not flagged", False)):
        rs = [r for r in judged if flagged(r["mg"], r) == sel]
        lines.append(f"- {name}: {len(rs)} with a radiologist change statement to compare; "
                     f"{sum(map(mg_wrong, rs))} contradict it about the prior image.")
    lines += [f"- {len(mg_change) - len(judged)} MedGemma change claims have no change statement in the "
              "radiologist's report to compare with.", "",
              "| case | condition | MedGemma | radiologist | CheXagent prior / current (p) | derived | flag |",
              "|---|---|---|---|---|---|---|"]
    fmt = lambda p: "-" if p is None else f"{p:.2f}"  # noqa: E731
    for r in rows:
        if r["mg"] in PRIOR_IMPLIED or r["gt"] in PRIOR_IMPLIED:
            lines.append(f"| {r['case']} | {r['cond']} | {r['mg']}" + (f" (\"{r['mg_ev']}\")" if r["mg_ev"] else "")
                         + f" | {r['gt']}" + (f" (\"{r['gt_ev']}\")" if r["gt_ev"] else "")
                         + f" | {short[r['dp']]} ({fmt(r['p_prior'])}) / {short[r['dc']]} ({fmt(r['p_current'])}) "
                         f"| {r['derived']} | {'FLAG' if r['mg'] in PRIOR_IMPLIED and flagged(r['mg'], r) else ''} |")
    lines += ["", "Limitations: 20 cases from 13 patients (several cases per patient), chosen for the scoring forms "
                  "(conditions MedGemma reports), not at random; the reference is the reader model's reading of the "
                  "radiologist's wording; the cut-offs were set for presence, not change; no intervals."]
    out = LOGS / f"edv_change_mimic_pilot_{time.strftime('%Y%m%d-%H%M%S')}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    cs = cases()
    analyze(cs, run(cs))
