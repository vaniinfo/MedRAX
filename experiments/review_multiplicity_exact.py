"""EXPLORATORY: E2 multiplicity from the pre-registered bootstrap draws themselves (third internal review,
2026-10-25). Replaces the CI-based approximation in review_multiplicity.py.

The E2 scripts (maira_score_external.py for A1, maira_score_all_conditions.py for A2) use a fixed seed and a
fixed loop order, so their 2,000 resamples per comparison can be regenerated exactly. This script
  1. replays them and checks that each reproduced 95% CI equals the committed one (verification);
  2. computes a two-sided bootstrap p-value per comparison, p = min(1, 2 * (1 + k) / (B + 1)), where k is the
     number of finite resampled differences on the far side of zero (<= 0 for a positive estimate);
  3. applies Holm across all 29 comparisons and, separately, across the 28 with at least 10 positives,
     with an independent Holm implementation checked against statsmodels if it is installed.

    python experiments/review_multiplicity_exact.py   -> results/review_multiplicity_exact.md
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import maira_score_all_conditions as A2  # noqa: E402
import maira_score_external as A1  # noqa: E402
from build_detection_grounding_doc import AUC_ROWS  # noqa: E402
from maira_score_nih import auc  # noqa: E402

OUT = HERE / "results" / "review_multiplicity_exact.md"


def replay(rng, y, c, m, n_boot):
    bc, bm = [], []
    for _ in range(n_boot):
        b = rng.integers(0, len(y), len(y))
        yb = y[b]
        bc.append(auc(c[b][yb == 1], c[b][yb == 0]))
        bm.append(auc(m[b][yb == 1], m[b][yb == 0]))
    bc, bm = np.array(bc), np.array(bm)
    ok = np.isfinite(bc) & np.isfinite(bm)
    d = (bc - bm)[ok]
    return auc(c[y == 1], c[y == 0]) - auc(m[y == 1], m[y == 0]), d


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    adj, running = np.empty_like(p), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def main():
    rows = []  # (dataset, condition, positives, estimate, lo, hi, p)
    rng = np.random.default_rng(A1.SEED)
    for ds in ("NIH test", "Indiana", "VinDr"):
        for finding, y, c, m in A1.paired(ds):
            est, d = replay(rng, y, c, m, A1.N_BOOT)
            rows.append((ds, finding, int(y.sum()), est, d))
    rng = np.random.default_rng(A2.SEED)
    for ds, finding, y, c, m in A2.paired():
        est, d = replay(rng, y, c, m, A2.N_BOOT)
        rows.append((ds, finding, int(y.sum()), est, d))
    # verification against the committed table (AUC_ROWS is parsed from it)
    committed = {}
    for r in AUC_ROWS:
        ds = {"NIH": "NIH test"}.get(r[0], r[0])
        cond = r[1].rstrip("*").lower()
        committed[(ds, cond)] = r[5]
    lines = ["# EXPLORATORY: E2 multiplicity from the replayed pre-registered bootstrap draws", "",
             "Run on 2026-10-25 after all pre-registered results were known. Method: docstring of "
             "`review_multiplicity_exact.py`.", "",
             "| dataset | condition | positives | difference [replayed 95% CI] | committed | matches | resamples on far "
             "side of 0 | bootstrap p | Holm, all 29 | Holm, >= 10 positives |", "|---|---|---|---|---|---|---|---|---|---|"]
    est = [r[3] for r in rows]
    k = [int(((r[4] <= 0) if r[3] > 0 else (r[4] >= 0)).sum()) for r in rows]
    p = [min(1.0, 2 * (1 + kk) / (len(r[4]) + 1)) for kk, r in zip(k, rows)]
    p_all = holm(p)
    elig = [i for i, r in enumerate(rows) if r[2] >= 10]
    p_el = dict(zip(elig, holm([p[i] for i in elig])))
    mism = 0
    for i, (ds, cond, pos, e, d) in enumerate(rows):
        lo, hi = np.percentile(d, [2.5, 97.5])
        mine = f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        com = committed.get((ds, cond), "missing")
        ok = com == mine
        mism += not ok
        pe = f"{p_el[i]:.4f}" if i in p_el else "not counted"
        lines.append(f"| {ds} | {cond} | {pos} | {mine} | {com} | {'yes' if ok else 'NO'} | {k[i]} | {p[i]:.4f} | "
                     f"{p_all[i]:.4f} | {pe} |")
    try:
        from statsmodels.stats.multitest import multipletests
        sm = multipletests(p, alpha=0.05, method="holm")[1]
        check = f"statsmodels Holm agrees: {bool(np.allclose(sm, p_all))}."
    except ImportError:
        check = "statsmodels not installed; Holm checked only by this script's implementation."
    lines += ["", f"Reproduced CIs matching the committed table: {len(rows) - mism} of {len(rows)}. {check}",
              f"Significant at 0.05 after Holm: {int((p_all < 0.05).sum())} of {len(rows)} (all); "
              f"{sum(p_el[i] < 0.05 for i in elig)} of {len(elig)} (at least 10 positives). The smallest attainable p "
              f"is {2 / 2001:.4f}, so with 29 tests Holm's first threshold is 0.05 / 29 = {0.05 / 29:.4f}.", ""]
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
