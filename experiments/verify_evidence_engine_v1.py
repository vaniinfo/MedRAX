"""Check that the Evidence Engine v1.0 runtime reproduces the D2/D3 research code exactly.

For every NIH test case (random + supplement), every finding, both engines (NIH primary,
Indiana comparator) and all three rules, the runtime engine (medrax/evidence, which only
loads the frozen bundle) must give the same state and reason as experiments/
evidence_engine_d3.classify. Nothing is fitted or tuned.

    python experiments/verify_evidence_engine_v1.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evidence_engine_d1_d2 as E  # noqa: E402
import evidence_engine_d3 as D  # noqa: E402
from medrax.evidence import EvidenceEngine  # noqa: E402


def main():
    E.verify()
    ind_data, ind_split, nih_data, nih_split = D.load_data()
    R, Sup = nih_data["random"], nih_data["supplement"]
    frozen = json.loads(D.FROZEN_BANDS.read_text(encoding="utf-8"))["findings"]
    indiana = json.loads((E.RESULTS / "frozen_dev_params.json").read_text(encoding="utf-8"))["findings"]
    nih = json.loads((E.RESULTS / "nih_frozen_dev_params.json").read_text(encoding="utf-8"))["findings"]
    ind_dev = [i for i in sorted(ind_data) if ind_split[i] == "dev"]
    nih_dev_r = [i for i in sorted(R) if nih_split["random"][i] == "dev"]
    rng = np.random.default_rng(E.SEED)
    sources = {}
    for f in E.S.FINDINGS:
        sources[("comparator", f)] = E.Source("Indiana", f, indiana[f]["platt"], *D.arr(ind_data, ind_dev, f), rng)
        sources[("primary", f)] = E.Source("NIH", f, nih[f]["platt"], *D.arr(R, nih_dev_r, f), rng)

    test = ([(i, R) for i in sorted(R) if nih_split["random"][i] == "test"]
            + [(i, Sup) for i in sorted(Sup) if nih_split["supplement"][i] == "test"])
    checked, mismatches, states = 0, [], Counter()
    for engine_name in ("primary", "comparator"):
        for rule_name, support_lr, refute_lr in E.RULES:
            engine = EvidenceEngine(engine=engine_name, rule=rule_name)
            for f in E.S.FINDINGS:
                p = np.array([d[i][f][0] for i, d in test])
                _, ref_states, ref_reasons = D.classify(
                    sources[(engine_name, f)], p, support_lr, refute_lr, frozen[f]["bands"])
                for k, pk in enumerate(p):
                    ev = engine.assess(f, float(pk))
                    checked += 1
                    if (ev.state, ev.reason) != (ref_states[k], ref_reasons[k]):
                        mismatches.append((engine_name, rule_name, f, float(pk), ev.state, ev.reason,
                                           ref_states[k], ref_reasons[k]))
                    if engine_name == "primary" and rule_name == E.RULES[0][0]:
                        states[ev.state] += 1

    print(f"checked {checked} (engine, rule, finding, case) assessments")
    print(f"mismatches: {len(mismatches)}")
    for m in mismatches[:10]:
        print("  ", m)
    print("primary engine, primary rule, NIH test states:", dict(states))

    e = EvidenceEngine()
    print("example:", e.assess("pleural effusion", 0.95, prior=0.05).to_dict())
    print("unvalidated finding:", e.assess("pneumonia", 0.9).to_dict())
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
