"""Change wording in three reports of the 19 MIMIC-CXR comparisons: the radiologist's (reference), the team's
MedGemma (as in the scoring form) and CheXagent's own longitudinal report (prior + current images,
chexagent_longitudinal_report.py). Each is read for the 13 conditions with the quote-checked change reader
(edv_change.read_claims, "not mentioned" allowed). Descriptive; nothing is regenerated.

MIMIC text is under the PhysioNet DUA: outputs go to logs/ (git-ignored).

    .venv/Scripts/python.exe experiments/change_three_reports_mimic.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import edv_change as X  # noqa: E402
from comparison_edv_pdf import make_extractor, parse_page  # noqa: E402
from pypdf import PdfReader  # noqa: E402

LOGS = ROOT / "logs"
SOURCE = Path("D:/datasets/held_out_eval_images/subset_comparisons_clean_filtered.pdf")
CHEX = LOGS / "chexagent_screen_20261011-065557_chexagent_longitudinal.json"
CACHE = LOGS / "change_three_reports_mimic_readings.json"
CONDITIONS = X.CONDITIONS
GROUP = {"new": "new", "unchanged": "persistent", "worse": "persistent", "better": "persistent",
         "resolved": "resolved"}
CHANGE = set(GROUP)


def main():
    chex = json.loads(CHEX.read_text(encoding="utf-8"))["reports"]
    pages = [parse_page(p.extract_text()) for p in PdfReader(str(SOURCE)).pages]
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    extractor = make_extractor("qwen3.8-edv:latest")
    for page in pages:
        r = cache.setdefault(page["title"], {})
        texts = {"radiologist": page["truth_text"], "medgemma": page["generated_text"], "chexagent": chex[page["title"]]}
        for who, text in texts.items():
            if who not in r:
                r[who] = {c: list(v) for c, v in X.read_claims(extractor, text, CONDITIONS, not_mentioned=True).items()}
                CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        print(f"read {page['title']}", flush=True)

    models = ("medgemma", "chexagent")
    names = {"medgemma": "MedGemma (trained)", "chexagent": "CheXagent (longitudinal report)"}
    rows = [(page["title"], c, {w: cache[page["title"]][w][c] for w in ("radiologist",) + models})
            for page in pages for c in CONDITIONS]
    ref = [(t, c, v) for t, c, v in rows if v["radiologist"][0] in CHANGE]
    lines = ["# Change wording: radiologist vs MedGemma vs CheXagent (19 MIMIC-CXR comparisons)", "",
             "Descriptive. Each report read for 13 conditions by the local reader with a verbatim quote check. "
             "CheXagent's report: prior + current images, asked to write the findings and describe change from the "
             "prior (about 2,061 input tokens, slightly over its 2,048 design length). MIMIC-CXR text: PhysioNet DUA.",
             "", "## Change wording found (condition mentions over 19 x 13)", "",
             "| claim | radiologist | " + " | ".join(names[m] for m in models) + " |", "|---|---|---|---|"]
    for k in X.KINDS + ["not mentioned"]:
        lines.append(f"| {k} | " + " | ".join(str(sum(v[w][0] == k for _, _, v in rows))
                                             for w in ("radiologist",) + models) + " |")

    lines += ["", f"## Where the radiologist states a change ({len(ref)} condition mentions)", "",
              "| | " + " | ".join(names[m] for m in models) + " |", "|---|---|---|"]
    def count(f):
        return " | ".join(str(sum(f(v[m][0], v["radiologist"][0]) for _, _, v in ref)) for m in models)
    lines += [f"| mentions the condition at all | {count(lambda a, g: a != 'not mentioned')} |",
              f"| states a change for it | {count(lambda a, g: a in CHANGE)} |",
              f"| same change group (new / persistent / resolved) | "
              f"{count(lambda a, g: a in CHANGE and GROUP[a] == GROUP[g])} |",
              f"| exactly the same change word | {count(lambda a, g: a == g)} |",
              f"| a different change group | {count(lambda a, g: a in CHANGE and GROUP[a] != GROUP[g])} |"]
    extra = {m: [(t, c, v) for t, c, v in rows if v[m][0] in CHANGE and v["radiologist"][0] == "not mentioned"]
             for m in models}
    lines += ["", "## Change claims for conditions the radiologist does not mention", ""] + \
        [f"- {names[m]}: {len(extra[m])}" for m in models]

    lines += ["", "## Every condition with a change statement in any of the three reports", "",
              "| comparison | condition | radiologist | MedGemma | CheXagent |", "|---|---|---|---|---|"]
    cell = lambda a: a[0] + (f" (\"{a[1]}\")" if a[1] else "")  # noqa: E731
    for t, c, v in rows:
        if any(v[w][0] in CHANGE for w in v):
            lines.append(f"| {t} | {c} | {cell(v['radiologist'])} | {cell(v['medgemma'])} | {cell(v['chexagent'])} |")
    lines += ["", "Limitations: 19 comparisons from a handful of patients (several per patient); the reference is the "
                  "reader model's reading of the radiologist's wording, and the reader sometimes attaches a general "
                  "sentence (\"findings unchanged\") to a specific condition; no intervals."]
    out = LOGS / f"change_three_reports_mimic_{time.strftime('%Y%m%d-%H%M%S')}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
