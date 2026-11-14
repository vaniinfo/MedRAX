"""A random set of NIH prior + current pairs for a second radiologist scoring form, in the format of
edv_longitudinal_examples_*.json (for nih20_scoring_pdf.py).

Sampling: random.Random(SEED).sample over the 499 longitudinal test pairs sorted by current file name, after
excluding the pairs already used in the first (label-selected) example set. No labels or results are used in the
selection.

    python experiments/nih20_random_examples.py --exclude logs/edv_longitudinal_examples_20261020-065035.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import edv_improvement as E  # noqa: E402
import edv_longitudinal as L  # noqa: E402
from edv_longitudinal_pdf import table  # noqa: E402
from medrax.evidence import EvidenceEngine  # noqa: E402

SEED = 20261031


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "logs" / "nih20_random_examples.json"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; not overwriting")
    used = {e["current"] for e in json.loads(Path(args.exclude).read_text(encoding="utf-8"))}
    pairs = sorted((p for p in L.pairs() if p["current"] not in used), key=lambda p: p["current"])
    chosen = sorted(random.Random(SEED).sample(pairs, args.n), key=lambda p: p["current"])

    truth, *_ = E.load()
    engine = EvidenceEngine()
    jl = lambda path: {json.loads(x)["current"]: json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()}  # noqa: E731
    chex, reports, labels = jl(L.CHEX_A), jl(L.REPORTS), jl(L.LABELS)
    record = []
    for k, pr in enumerate(chosen, 1):
        f = pr["current"]
        rows, _ = table(engine, reports[f]["report"], labels[f]["labels"], chex[f], truth, f)
        record.append({"example": k, **pr, "rows": rows})
        print(k, f, [(r["finding"], r["status"]) for r in rows])
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"{len(record)} random pairs (seed {SEED}, {len(used)} excluded) -> {out}")


if __name__ == "__main__":
    main()
