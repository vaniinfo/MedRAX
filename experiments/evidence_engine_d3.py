"""D3 for the Evidence Engine: dual-evidence REFUTING rule.

Protocol: experiments/evidence_engine_d3_design.md (committed before this run).

Phase 1 builds empirical LR bands from NIH development data only and freezes them to
results/nih_dev_empirical_bands.json (read-only). Phase 2 reads the bands only from that
file and evaluates the rule once on the NIH test split used in D2. The Platt LR machinery
is rebuilt exactly as in D2 (same seed and construction order), so its REFUTING and
SUPPORTING boundaries are D2's.

    python experiments/evidence_engine_d3.py
"""

import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).parent))
import evidence_engine_d1_d2 as E  # noqa: E402
import ppresent_nih as N  # noqa: E402
import ppresent_split as S  # noqa: E402

RESULTS = Path(__file__).parent / "results"
FROZEN_BANDS = RESULTS / "nih_dev_empirical_bands.json"
REPORT = RESULTS / "evidence_engine_d3.md"
BANDS = E.BANDS  # fixed in D2, before any band was built from development data
RULES = [("primary 5 / 0.2", 5.0, 0.2), ("sensitivity 2 / 0.5", 2.0, 0.5),
         ("sensitivity 10 / 0.1", 10.0, 0.1)]
REASONS = ["SUPPORTING", "REFUTING", "WEAK_EVIDENCE", "REFUTING_NOT_EMPIRICALLY_CONFIRMED",
           "OUTSIDE_SUPPORT", "FEW_DEV_POSITIVES"]


def cp_upper(k, n):
    return 1.0 if k >= n else float(beta.ppf(0.975, k + 1, n - k))


def cp_lower(k, n):
    return 0.0 if k <= 0 else float(beta.ppf(0.025, k, n - k + 1))


def band_of(p):
    return np.clip(np.searchsorted(BANDS, np.asarray(p, float), side="right") - 1, 0, len(BANDS) - 2)


def load_data():
    ind_data = S.load()
    ind_split = json.loads((RESULTS / "split_indiana_70_30.json").read_text(encoding="utf-8"))
    nih_data, _ = N.load_readings()
    nih_split = json.loads((RESULTS / "split_nih_70_30.json").read_text(encoding="utf-8"))
    return ind_data, ind_split, nih_data, nih_split


def arr(data, images, f):
    return (np.array([data[i][f][0] for i in images], float),
            np.array([data[i][f][1] for i in images], int))


def build_bands(nih_data, nih_split):
    """Phase 1: empirical LR bands from NIH development (random + supplement) only."""
    R, Sup = nih_data["random"], nih_data["supplement"]
    dev = ([(i, R) for i in sorted(R) if nih_split["random"][i] == "dev"]
           + [(i, Sup) for i in sorted(Sup) if nih_split["supplement"][i] == "dev"])
    frozen = {"source": "NIH development split, random + supplement", "n_images": len(dev),
              "band_edges": BANDS, "findings": {}}
    for f in S.FINDINGS:
        p = np.array([d[i][f][0] for i, d in dev])
        y = np.array([d[i][f][1] for i, d in dev])
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        idx = band_of(p)
        bands = []
        for b in range(len(BANDS) - 1):
            k_pos, k_neg = int(((idx == b) & (y == 1)).sum()), int(((idx == b) & (y == 0)).sum())
            lower_neg = cp_lower(k_neg, n_neg)
            bands.append({
                "lo": BANDS[b], "hi": BANDS[b + 1], "k_pos": k_pos, "k_neg": k_neg,
                "lr": (k_pos / n_pos) / (k_neg / n_neg) if k_neg else None,
                "uci": cp_upper(k_pos, n_pos) / lower_neg if lower_neg > 0 else None,
            })
        frozen["findings"][f] = {"n_pos": n_pos, "n_neg": n_neg, "bands": bands}
    FROZEN_BANDS.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    os.chmod(FROZEN_BANDS, stat.S_IREAD)


def classify(source, p, support_lr, refute_lr, bands):
    """States and internal reasons under the dual-evidence REFUTING rule."""
    base = source.states(p, support_lr, refute_lr)  # D2 rule: Platt only
    uci = np.array([b["uci"] if b["uci"] is not None else math.inf for b in bands])
    confirmed = uci[band_of(p)] <= refute_lr
    states = base.copy()
    reasons = np.empty(len(p), dtype=object)
    reasons[base == "SUPPORTING"] = "SUPPORTING"
    reasons[base == "UNCERTAIN"] = "WEAK_EVIDENCE"
    refute = base == "REFUTING"
    reasons[refute & confirmed] = "REFUTING"
    states[refute & ~confirmed] = "UNCERTAIN"
    reasons[refute & ~confirmed] = "REFUTING_NOT_EMPIRICALLY_CONFIRMED"
    insufficient = base == "INSUFFICIENT DATA"
    reasons[insufficient] = ("FEW_DEV_POSITIVES" if source.dev_pos < E.MIN_DEV_POSITIVES
                             else "OUTSIDE_SUPPORT")
    return base, states, reasons


def refute_cell(states, y, bidx, criterion):
    point = E.state_lr(states, y, "REFUTING")
    if math.isnan(point):
        return "0%, –", "0 / %d" % int(y.sum())
    lo, hi = E.ci([E.state_lr(states[b], y[b], "REFUTING") for b in bidx])
    mark = " ✓✓" if hi <= criterion else (" ✓" if point <= criterion else " ✗")
    cov = (states == "REFUTING").mean()
    missed = int(((states == "REFUTING") & (y == 1)).sum())
    return (f"{cov:.1%}, {E.fmt_lr(point)} [{E.fmt_lr(lo)}–{E.fmt_lr(hi)}]{mark}",
            f"{missed} / {int(y.sum())} ({missed / y.sum():.0%})")


def main():
    E.verify()
    ind_data, ind_split, nih_data, nih_split = load_data()

    # Phase 1 (development only), then freeze
    if FROZEN_BANDS.exists():
        print(f"using existing {FROZEN_BANDS.name}; not rebuilding")
    else:
        build_bands(nih_data, nih_split)
    band_sha = hashlib.sha256(FROZEN_BANDS.read_bytes()).hexdigest()
    frozen = json.loads(FROZEN_BANDS.read_text(encoding="utf-8"))["findings"]

    # Platt LR sources, rebuilt exactly as in D2 (same seed and order)
    indiana = json.loads((RESULTS / "frozen_dev_params.json").read_text(encoding="utf-8"))["findings"]
    nih = json.loads((RESULTS / "nih_frozen_dev_params.json").read_text(encoding="utf-8"))["findings"]
    R, Sup = nih_data["random"], nih_data["supplement"]
    ind_dev = [i for i in sorted(ind_data) if ind_split[i] == "dev"]
    nih_dev_r = [i for i in sorted(R) if nih_split["random"][i] == "dev"]
    rng = np.random.default_rng(E.SEED)
    sources = {}
    for f in S.FINDINGS:
        sources[("Indiana", f)] = E.Source("Indiana", f, indiana[f]["platt"], *arr(ind_data, ind_dev, f), rng)
        sources[("NIH", f)] = E.Source("NIH", f, nih[f]["platt"], *arr(R, nih_dev_r, f), rng)

    # Phase 2: evaluate once on the NIH test split
    test_r = [i for i in sorted(R) if nih_split["random"][i] == "test"]
    test_s = [i for i in sorted(Sup) if nih_split["supplement"][i] == "test"]
    both = {**R, **Sup}
    rng_test = np.random.default_rng(E.SEED + 3)

    L = ["# Evidence Engine D3: dual-evidence REFUTING rule", "",
         "Protocol: `experiments/evidence_engine_d3_design.md` (committed before this run). "
         f"Empirical bands built from NIH development only and frozen to `{FROZEN_BANDS.name}` "
         f"(SHA-256 {band_sha[:12]}…) before the test split was read. Platt boundaries identical "
         "to D2. Test CIs: 2,000-sample image-level bootstrap.", "",
         "## Frozen empirical bands (NIH development, random + supplement)", "",
         "Per band: positives / negatives, empirical LR, exact upper 95% bound (Clopper–Pearson). "
         "✓ = upper bound ≤ 0.2 (a Platt-REFUTING case in this band can be confirmed).", "",
         "| Finding | Dev pos / neg | " + " | ".join(
             f"{b['lo']:g}–{min(b['hi'], 1):g}" for b in frozen[S.FINDINGS[0]]["bands"]) + " |",
         "|---|---|" + "---|" * (len(BANDS) - 1)]
    for f in S.FINDINGS:
        cells = []
        for b in frozen[f]["bands"]:
            if b["k_pos"] + b["k_neg"] == 0:
                cells.append("–")
                continue
            lr = "–" if b["lr"] is None else E.fmt_lr(b["lr"])
            uci = "∞" if b["uci"] is None else E.fmt_lr(b["uci"])
            ok = " ✓" if b["uci"] is not None and b["uci"] <= 0.2 else ""
            cells.append(f"{b['k_pos']}/{b['k_neg']}: {lr} ≤{uci}{ok}")
        L.append(f"| {f} | {frozen[f]['n_pos']} / {frozen[f]['n_neg']} | " + " | ".join(cells) + " |")

    def comparison(title, images, data, rule=RULES[0]):
        name, ls, lr_ = rule
        out = ["", f"## {title}", "",
               f"Rule: {name}. REFUTING state: coverage (share of all cases), empirical LR [95% CI] "
               f"(✓ point ≤ {lr_:g}; ✓✓ upper CI ≤ {lr_:g}), and positives classified REFUTING "
               "(false reassurance).", "",
               "| Finding | Platt source | Pos / neg | D2 Platt-only REFUTING | positives in it | "
               "**D3 dual-evidence REFUTING** | positives in it |",
               "|---|---|---|---|---|---|---|"]
        bidx = [rng_test.integers(0, len(images), len(images)) for _ in range(E.N_BOOT)]
        for f in S.FINDINGS:
            p, y = arr(data, images, f)
            for src in ("NIH", "Indiana"):
                base, states, _ = classify(sources[(src, f)], p, ls, lr_, frozen[f]["bands"])
                d2, d2_missed = refute_cell(base, y, bidx, lr_)
                d3, d3_missed = refute_cell(states, y, bidx, lr_)
                out.append(f"| {f} | {src} | {int(y.sum())} / {int((1 - y).sum())} | "
                           f"{d2} | {d2_missed} | **{d3}** | {d3_missed} |")
        return out

    L += comparison("Primary: NIH test, random part (natural prevalence)", test_r, R)
    L += comparison("Secondary: NIH test, random + supplement", test_r + test_s, both)

    L += ["", "## Internal reasons (NIH test, random part; NIH Platt source; primary rule)", "",
          "| Finding | " + " | ".join(REASONS) + " |", "|---|" + "---|" * len(REASONS)]
    for f in S.FINDINGS:
        p, _ = arr(R, test_r, f)
        _, _, reasons = classify(sources[("NIH", f)], p, 5.0, 0.2, frozen[f]["bands"])
        L.append(f"| {f} | " + " | ".join(f"{(reasons == r).mean():.1%}" for r in REASONS) + " |")

    for rule in RULES[1:]:
        L += comparison(f"Sensitivity analysis: NIH test, random + supplement", test_r + test_s,
                        both, rule)

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
