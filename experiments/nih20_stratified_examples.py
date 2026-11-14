"""Two disjoint sets of NIH prior + current pairs for the radiologist scoring forms, stratified by the NIH labels of
the current image so that each set covers every condition: at least MIN_PER condition-labelled cases per condition
per set (fewer only if the 499 pairs do not have enough), the rest filled with random labelled pairs, plus N_NORMAL
pairs with no label. Cases are drawn at random within each condition (fixed seed); only NIH labels are used, never
model outputs, so the sets are not selected for how well MedGemma or CheXagent did.

Each page's table lists every condition MedGemma states, every condition CheXagent finds that the report does not
state, and every NIH-labelled condition neither of them reported ("missed by both").

    python experiments/nih20_stratified_examples.py   -> logs/nih20_set_A.json, logs/nih20_set_B.json
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import edv_improvement as E  # noqa: E402
import edv_longitudinal as L  # noqa: E402
from edv_longitudinal_pdf import table  # noqa: E402
from medrax.evidence import EvidenceEngine  # noqa: E402

SEED, N, N_NORMAL, MIN_PER = 20261101, 20, 2, 2
OUT = {"A": ROOT / "logs" / "nih20_set_A.json", "B": ROOT / "logs" / "nih20_set_B.json"}


def main():
    for p in OUT.values():
        if p.exists():
            raise SystemExit(f"{p} already exists; not overwriting")
    truth, *_ = E.load()
    pairs = sorted(L.pairs(), key=lambda p: p["current"])
    labels = {p["current"]: {c for c in L.CONDITIONS if truth[(p["current"], c)]} for p in pairs}
    rng = random.Random(SEED)
    sets = {"A": [], "B": []}
    used = set()

    def take(pool, s):
        pool = [p for p in pool if p["current"] not in used]
        if not pool:
            return False
        p = rng.choice(pool)
        used.add(p["current"])
        sets[s].append(p)
        return True

    def count(s):
        return Counter(c for p in sets[s] for c in labels[p["current"]])

    # rarest conditions first; alternate the two sets so they get equal shares
    order = sorted(L.CONDITIONS, key=lambda c: sum(c in v for v in labels.values()))
    for c in order:
        for s in ("A", "B") * MIN_PER:
            if count(s)[c] < MIN_PER and len(sets[s]) < N - N_NORMAL:
                take([p for p in pairs if c in labels[p["current"]]], s)
    abnormal = [p for p in pairs if labels[p["current"]]]
    normal = [p for p in pairs if not labels[p["current"]]]
    for s in ("A", "B"):
        while len(sets[s]) < N - N_NORMAL and take(abnormal, s):
            pass
        for _ in range(N_NORMAL):
            take(normal, s)

    engine = EvidenceEngine()
    jl = lambda path: {json.loads(x)["current"]: json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()}  # noqa: E731
    chex, reports, lab = jl(L.CHEX_A), jl(L.REPORTS), jl(L.LABELS)
    for s, chosen in sets.items():
        chosen.sort(key=lambda p: p["current"])
        record = []
        for k, pr in enumerate(chosen, 1):
            f = pr["current"]
            rows, _ = table(engine, reports[f]["report"], lab[f]["labels"], chex[f], truth, f)
            listed = {r["finding"] for r in rows}
            for c in L.CONDITIONS:  # labelled by NIH but reported by neither model
                if truth[(f, c)] and c not in listed:
                    rows.append({"finding": c, "name": c.capitalize(), "status": "Missed",
                                 "p_present": chex[f]["p_present"][c], "ground_truth": "Present (NIH label)"})
            record.append({"example": k, **pr, "nih_labels": sorted(labels[f]), "rows": rows})
        OUT[s].write_text(json.dumps(record, indent=1), encoding="utf-8")
        cov = count(s)
        print(f"set {s}: {len(chosen)} pairs ({sum(not labels[p['current']] for p in chosen)} normal); "
              f"cases per condition: {dict(sorted(cov.items()))}; table rows {sum(len(r['rows']) for r in record)}")


if __name__ == "__main__":
    main()
