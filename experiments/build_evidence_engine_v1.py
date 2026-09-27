"""Freeze Evidence Engine v1.0: write the parameter bundle the engine loads at run time.

Rebuilds the LR machinery exactly as D2/D3 did (same seed and construction order, same
frozen inputs, each SHA-256 verified) and stores everything the engine needs, so the engine
never refits anything:
  - primary engine: NIH-derived Platt LR + NIH-development empirical bands;
  - comparator:     Indiana-derived Platt LR + the same NIH-development empirical bands
                    (the configuration evaluated as "Indiana" in D3).
Reads development data only; the NIH test split is not read.

    python experiments/build_evidence_engine_v1.py
"""

import hashlib
import json
import os
import stat
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evidence_engine_d1_d2 as E  # noqa: E402
import evidence_engine_d3 as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "medrax" / "evidence"
BUNDLE = OUT_DIR / "evidence_engine_v1.0.json"
BANDS_SHA = "f6fb7df6f27b19f335774f9705265b91e69eed1af2e4a1c4f4cfdd3f9c7956f6"


def source_record(src, bands):
    """Everything the engine needs for one finding under one LR source."""
    grid_logit = E.logit(E.GRID)
    cutpoints = {}
    for name, support_lr, refute_lr in E.RULES:
        p_s, p_r = src.cutpoints(support_lr, refute_lr)
        cutpoints[name] = {"support_lr": support_lr, "refute_lr": refute_lr,
                           "p_support": p_s, "p_refute": p_r}
    return {
        "platt": {"coef": src.a, "intercept": src.b},
        "dev_prevalence": float(src.pi),
        "dev_positives": src.dev_pos,
        "dev_n": int(len(src.dev_p)),
        # support: at least SUPPORT_MIN development scores at or beyond p in its tail
        "support": {"p_min": float(src.sorted[E.SUPPORT_MIN - 1]),
                    "p_max": float(src.sorted[-E.SUPPORT_MIN])},
        "cutpoints": cutpoints,
        # LR 95% CI from Platt refits on the development set, as natural-log LR on the grid
        "log_lr_ci": {"lower": np.round(np.log(src.lower), 6).tolist(),
                      "upper": np.round(np.log(src.upper), 6).tolist()},
        "empirical_bands": [{"lo": b["lo"], "hi": b["hi"], "k_pos": b["k_pos"],
                             "k_neg": b["k_neg"], "uci": b["uci"]} for b in bands],
        "_grid_check": float(grid_logit[0]),
    }


def main():
    E.verify()
    band_sha = hashlib.sha256(D.FROZEN_BANDS.read_bytes()).hexdigest()
    assert band_sha == BANDS_SHA, f"nih_dev_empirical_bands.json changed: {band_sha}"
    frozen_bands = json.loads(D.FROZEN_BANDS.read_text(encoding="utf-8"))["findings"]

    ind_data, ind_split, nih_data, nih_split = D.load_data()
    R = nih_data["random"]
    indiana = json.loads((E.RESULTS / "frozen_dev_params.json").read_text(encoding="utf-8"))["findings"]
    nih = json.loads((E.RESULTS / "nih_frozen_dev_params.json").read_text(encoding="utf-8"))["findings"]
    ind_dev = [i for i in sorted(ind_data) if ind_split[i] == "dev"]
    nih_dev_r = [i for i in sorted(R) if nih_split["random"][i] == "dev"]

    rng = np.random.default_rng(E.SEED)  # same seed and order as D2/D3
    primary, comparator = {}, {}
    for f in E.S.FINDINGS:
        ind_src = E.Source("Indiana", f, indiana[f]["platt"], *D.arr(ind_data, ind_dev, f), rng)
        nih_src = E.Source("NIH", f, nih[f]["platt"], *D.arr(R, nih_dev_r, f), rng)
        primary[f] = source_record(nih_src, frozen_bands[f]["bands"])
        comparator[f] = source_record(ind_src, frozen_bands[f]["bands"])
    for rec in list(primary.values()) + list(comparator.values()):
        rec.pop("_grid_check")

    bundle = {
        "engine_id": "evidence-engine-v1.0",
        "created": datetime.now().isoformat(timespec="seconds"),
        "input": "CheXagent p_present: Yes/(Yes+No) over the first answer token of 'Is there {finding}?'",
        "model": "StanfordAIMI/CheXagent-2-3b @ 8f19b53a2eceda4c33b0acec6c81fbc293ad80d0",
        "findings": list(E.S.FINDINGS),
        "rules": {"primary": "primary 5 / 0.2",
                  "sensitivity": ["sensitivity 2 / 0.5", "sensitivity 10 / 0.1"]},
        "min_dev_positives": E.MIN_DEV_POSITIVES,
        "support_min": E.SUPPORT_MIN,
        "grid": {"space": "logit(p)", "start": -9.5, "stop": 9.5, "num": len(E.GRID)},
        "band_edges": D.BANDS,
        "engines": {
            "primary": {"name": "NIH", "platt_source": "NIH ChestX-ray14 development (random part)",
                        "bands_source": "NIH ChestX-ray14 development (random + supplement)",
                        "findings": primary},
            "comparator": {"name": "Indiana", "platt_source": "Indiana (Open-i) development split",
                           "bands_source": "NIH ChestX-ray14 development (random + supplement)",
                           "findings": comparator},
        },
        "provenance": {
            "commits": {"ppresent_validation": "14b4dde (tag ppresent-validation-v1)",
                        "d1_d2_protocol": "47c6b64", "d3_protocol": "aee83a3", "d3_results": "3643e42"},
            "inputs_sha256": {**E.EXPECTED_SHA256, "nih_dev_empirical_bands.json": BANDS_SHA},
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if BUNDLE.exists():
        raise SystemExit(f"{BUNDLE} already exists; v1.0 is frozen and is not rebuilt")
    BUNDLE.write_text(json.dumps(bundle, indent=1), encoding="utf-8")
    os.chmod(BUNDLE, stat.S_IREAD)
    print("wrote", BUNDLE, hashlib.sha256(BUNDLE.read_bytes()).hexdigest())
    for f in E.S.FINDINGS:
        c = primary[f]["cutpoints"]["primary 5 / 0.2"]
        print(f"  {f:17} p_support {c['p_support']}  p_refute {c['p_refute']}  support {primary[f]['support']}")


if __name__ == "__main__":
    main()
