"""30 prior + current pairs (20 for a scoring form + 10 spares) from the team's held-out MIMIC-CXR pairs
(held_out_val_set.csv), chosen from the radiologists' report text only, so that only their 60 images need to be
fetched. Same rules as chexpert_plus_pairs.py:
- patients not already in the 28 MIMIC cases with images on disk (held_out_eval_images, new_16/17 folders);
- the current report compares with the prior and states at least one of the six conditions in a sentence without
  a negation (keyword screen), and the prior report is not empty;
- one pair per patient; stratified by condition (rarest first, up to 5 per condition, seeded), filled to 30.
MIMIC-CXR text and images: PhysioNet DUA. Outputs go to logs/ (git-ignored).

    .venv/Scripts/python.exe -I experiments/mimic_heldout_pairs.py
"""

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chexpert_plus_pairs import COMPARES, TERMS, stated  # noqa: E402

BASE = Path("D:/datasets/held_out_eval_images")
CSV = BASE / "held_out_val_set.csv"
OUT = Path(__file__).resolve().parents[1] / "logs" / "mimic_heldout_candidates.json"
SEED, N = 20261102, 30


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting")
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    used = {p.name.split("_subject_")[-1] for d in ("held_out_eval_images", "new_16_matched_20261017",
                                                     "new_17_cases_20261017")
            for p in (BASE / d).iterdir() if "_subject_" in p.name}
    pairs = []
    for r in rows:
        six = stated(r["report_after"])
        if r["subject_id"] not in used and six and r["report_before"].strip() and COMPARES.search(r["report_after"]):
            pairs.append({**r, "six": sorted(six)})
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    counts = Counter(c for q in pairs for c in q["six"])
    chosen, taken = [], set()
    for c in sorted(TERMS, key=lambda c: counts[c]):
        for q in pairs:
            if len(chosen) >= N or sum(c in x["six"] for x in chosen) >= 5:
                break
            if c in q["six"] and q["subject_id"] not in taken:
                chosen.append(q)
                taken.add(q["subject_id"])
    for q in pairs:
        if len(chosen) >= N:
            break
        if q["subject_id"] not in taken:
            chosen.append(q)
            taken.add(q["subject_id"])
    OUT.write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    files = [q[k] for q in chosen for k in ("image_path_before", "image_path_after")]
    OUT.with_suffix(".files.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
    print(f"{len(pairs)} eligible pairs; {len(chosen)} chosen from {len(taken)} patients; "
          f"by condition {dict(Counter(c for q in chosen for c in q['six']))}")
    print(f"{len(files)} image paths -> {OUT.with_suffix('.files.txt')}")


if __name__ == "__main__":
    main()
