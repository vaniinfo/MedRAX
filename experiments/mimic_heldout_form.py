"""Radiologist scoring form of 20 held-out MIMIC-CXR prior + current pairs (the team's held-out set, never used to
train the trained MedGemma), built exactly like chexpert_plus_form.py. MIMIC-CXR: PhysioNet DUA; outputs go to
logs/ (git-ignored).

Input: the 60 pairs Jay sent (D:/datasets/held_out_eval_images/subset.csv; images/images/row_NNNN/before.jpg and
after.jpg, row_NNNN = the NNNN-th data row of subset.csv). Candidates, fixed before any report is generated
(report text only, the rules of chexpert_plus_pairs.py / mimic_heldout_pairs.py): patients not in the 28 MIMIC
cases already on disk; the current report compares with the prior and states one of the six conditions without a
negation; non-empty prior report; one pair per patient; stratified by condition (rarest first, up to 5 each, seeded).
The trained MedGemma gets the prior report and indication as given in the held-out CSV (its training format).

    .venv/Scripts/python.exe -I experiments/mimic_heldout_form.py prepare
    .venv-medgemma/Scripts/python.exe experiments/mimic_heldout_form.py generate
    .venv/Scripts/python.exe experiments/mimic_heldout_form.py build [suffix]

Sets 2 and 3 (two more forms of 20): the 40 pairs of subset.csv not in set 1, with the rules relaxed so that all 60
pairs are used - a patient may appear more than once, the 6 patients already on disk are allowed, and pairs whose
report states none of the six conditions (or whose MedGemma report states none of the 13) are kept. Shuffled
(seeded), ordered by the number of six conditions in the radiologist's report and dealt alternately to the two sets.

    .venv/Scripts/python.exe -I experiments/mimic_heldout_form.py prepare_rest
    .venv-medgemma/Scripts/python.exe experiments/mimic_heldout_form.py generate 2      (then 3)
    .venv/Scripts/python.exe experiments/mimic_heldout_form.py build 2 [suffix]        (then 3)
"""

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chexpert_plus_form as F  # noqa: E402

BASE = Path("D:/datasets/held_out_eval_images")
SUBSET = BASE / "subset.csv"
F.IMAGES = BASE / "images" / "images"
F.INPUTS = F.LOGS / "mimic_heldout_inputs.json"
F.REPORTS = F.LOGS / "mimic_heldout_trained_reports.jsonl"
F.NAME, F.SLUG = "MIMIC-CXR held-out", "mimic_heldout"
SEED = 20261102


def prepare():
    from chexpert_plus_pairs import COMPARES, TERMS, stated
    if F.INPUTS.exists():
        raise SystemExit(f"{F.INPUTS} exists; not overwriting")
    rows = list(csv.DictReader(SUBSET.open(encoding="utf-8")))
    used = {p.name.split("_subject_")[-1] for d in ("held_out_eval_images", "new_16_matched_20261017",
                                                     "new_17_cases_20261017")
            for p in (BASE / d).iterdir() if "_subject_" in p.name}
    pairs = []
    for k, r in enumerate(rows):
        six = stated(r["report_after"])
        folder = F.IMAGES / f"row_{k:04d}"
        if r["subject_id"] in used or not six or not r["report_before"].strip() or not COMPARES.search(r["report_after"]):
            continue
        assert (folder / "before.jpg").exists() and (folder / "after.jpg").exists(), folder
        pairs.append({"patient": r["subject_id"], "row": k, "prior": r["dicom_id_before"], "current": r["dicom_id_after"],
                      "prior_png": f"row_{k:04d}/before.jpg", "current_png": f"row_{k:04d}/after.jpg",
                      "six": sorted(six), "prior_report": r["report_before"].strip(),
                      "indication": r["context_indication"].strip() or "Not provided.",
                      "truth_text": r["report_after"].strip()})
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    counts = Counter(c for q in pairs for c in q["six"])
    chosen, taken = [], set()
    for c in sorted(TERMS, key=lambda c: counts[c]):
        for q in pairs:
            if sum(c in x["six"] for x in chosen) >= 5:
                break
            if c in q["six"] and q["patient"] not in taken:
                chosen.append(q)
                taken.add(q["patient"])
    for q in pairs:
        if q["patient"] not in taken:
            chosen.append(q)
            taken.add(q["patient"])
    F.INPUTS.write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    print(f"{len(pairs)} eligible pairs; {len(chosen)} candidates (one per patient) -> {F.INPUTS}; "
          f"by condition {dict(Counter(c for q in chosen for c in q['six']))}")


def entry(k, r, six):
    return {"patient": r["subject_id"], "row": k, "prior": r["dicom_id_before"], "current": r["dicom_id_after"],
            "prior_png": f"row_{k:04d}/before.jpg", "current_png": f"row_{k:04d}/after.jpg", "six": sorted(six),
            "prior_report": r["report_before"].strip(), "indication": r["context_indication"].strip() or "Not provided.",
            "truth_text": r["report_after"].strip()}


def prepare_rest():
    from chexpert_plus_pairs import stated
    outs = {n: F.LOGS / f"mimic_heldout_inputs_{n}.json" for n in (2, 3)}
    if any(o.exists() for o in outs.values()):
        raise SystemExit(f"{outs} exist; not overwriting")
    first = {x["current"] for x in json.loads((F.LOGS / "mimic_heldout_scoring_form.json").read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(SUBSET.open(encoding="utf-8")))
    rest = [entry(k, r, stated(r["report_after"])) for k, r in enumerate(rows) if r["dicom_id_after"] not in first]
    rng = random.Random(SEED + 1)
    rng.shuffle(rest)
    rest.sort(key=lambda q: -len(q["six"]))
    for i, n in enumerate((2, 3)):
        part = rest[i::2]
        outs[n].write_text(json.dumps(part, indent=1), encoding="utf-8")
        print(f"set {n}: {len(part)} pairs, {len({q['patient'] for q in part})} patients, "
              f"by condition {dict(Counter(c for q in part for c in q['six']))} -> {outs[n]}")


def use_set(n):
    """Paths and names of set n (2 or 3); set 1 keeps the defaults above."""
    F.INPUTS = F.LOGS / f"mimic_heldout_inputs_{n}.json"
    F.NAME, F.SLUG = f"MIMIC-CXR held-out, set {n}", f"mimic_heldout_{n}"
    F.SKIP_EMPTY = False


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage in ("generate", "build") and len(sys.argv) > 2 and sys.argv[2] in ("2", "3"):
        use_set(int(sys.argv.pop(2)))
    {"prepare": prepare, "prepare_rest": prepare_rest, "generate": F.generate, "build": F.build}[stage]()
