"""MAIRA-2 grounding coverage expansion for the regional (pleural effusion, atelectasis, consolidation) and focal
(pneumothorax) findings of the returned radiologist forms - for the visualization report only.

Same inference as the previous experiment: vindr_grounding_amendment.maira_locator() (microsoft/maira-2 at revision
MAIRA_REVISION, 8-bit, greedy decoding, max_new_tokens 150), report-blinded: the prompt is the condition name only
(mimic19_grounding_compare.phrase(): side words removed, then capitalized). MedGemma's report or side, the reference
report and the radiologist scores are never used. Raw outputs kept as returned (including empty and unparsable).

Existing results are reused only where image path + condition + model revision + settings match: the 19-case cache
logs/mimic19_maira2_grounding.json was produced by the same function and revision. Everything is written to a NEW
versioned cache (logs/maira2_grounding_v2_<stamp>.json); no existing file is modified.

    .venv/Scripts/python.exe experiments/maira2_coverage_expansion.py
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
LOGS = ROOT / "logs"
CONDS = ("pleural effusion", "atelectasis", "consolidation", "pneumothorax")


def targets():
    import focused_conditions_report as FC
    from mimic19_grounding_compare import phrase
    FC.CONDS = CONDS
    entries, *_ = FC.restore()
    out = {}
    for e in entries:
        out.setdefault((e["current"], phrase(e["finding"])), e["condition"])
    return out


def main():
    from vindr_grounding_amendment import MAIRA_REVISION, maira_locator
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = LOGS / f"maira2_grounding_v2_{stamp}.json"
    old = json.loads((LOGS / "mimic19_maira2_grounding.json").read_text(encoding="utf-8"))
    tg = targets()
    cache = {"meta": {"model": "microsoft/maira-2", "revision": MAIRA_REVISION, "quantization": "8-bit",
                      "generation": "greedy, max_new_tokens=150", "prompt": "condition name only, capitalized",
                      "function": "vindr_grounding_amendment.maira_locator", "created": stamp,
                      "reused_from": "logs/mimic19_maira2_grounding.json (same function and revision)"},
             "results": {}}
    todo = []
    for (img, ph), cond in sorted(tg.items()):
        key = f"{img}|{ph}"
        if key in old:
            cache["results"][key] = {**old[key], "image": img, "condition": cond, "phrase": ph, "source": "reused"}
        else:
            todo.append((key, img, ph, cond))
    print(f"{len(tg)} image/condition combinations: {len(tg) - len(todo)} reused, {len(todo)} to run", flush=True)
    path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    loc = maira_locator() if todo else None
    for n, (key, img, ph, cond) in enumerate(todo, 1):
        t = time.time()
        try:
            r = loc(img, ph)
        except Exception as ex:  # keep failures visible
            r = {"prompt": ph.capitalize(), "raw": "", "boxes": [], "parse_error": True, "error": repr(ex)}
        cache["results"][key] = {**r, "image": img, "condition": cond, "phrase": ph, "source": "new run"}
        path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        print(f"{n}/{len(todo)} {cond} {Path(img).parent.name or Path(img).name}: {len(r['boxes'])} boxes "
              f"({time.time() - t:.0f}s)", flush=True)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
