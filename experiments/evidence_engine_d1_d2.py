"""D1/D2 for the Evidence Engine (protocol: experiments/evidence_engine_design.md).

D1: to what extent is the Indiana -> NIH calibration shift explained by prevalence shift?
D2: does a likelihood ratio learned on one dataset keep its meaning on another?

Uses only committed readings and frozen parameters, each verified against its SHA-256;
no model is rerun and nothing frozen is modified.

    python experiments/evidence_engine_d1_d2.py
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
import ppresent_nih as N  # noqa: E402
import ppresent_split as S  # noqa: E402

RESULTS = Path(__file__).parent / "results"
REPORT = RESULTS / "evidence_engine_d1_d2.md"
EXPECTED_SHA256 = {  # from results/REPRODUCE.md (tag ppresent-validation-v1)
    "frozen_dev_params.json": "3c14b9441b81095b68b8f896ee3457ddeb7a995efe4e2882ab77d9255c989d14",
    "split_indiana_70_30.json": "8148ad9c4670b27075398477e88508bd675f25f44ec8ca86340b99fc37e0e4b7",
    "nih_frozen_dev_params.json": "e256455ec871cbb7d12806ae6a14c8f077942b9fbfe120ac3574869a43cac776",
    "split_nih_70_30.json": "79e83a88c7bd447f548d306eea079d64e74fb948e5d2afcc39c8145ba032e41b",
    "ppresent_indiana.jsonl": "5ddccb37affadd778c4e848a53ef74370d656556ecbb4eae440c9837c4d60aa0",
    "ppresent_nih.jsonl": "8344d22b3145e0855e3e4be193f368b20f523c500e643062131408acb7d312c8",
}

SEED = 20260927
N_REFIT = 1000        # bootstrap refits of Platt on the source development set (LR CI)
N_BOOT = 2000         # bootstrap resamples of the test set (reported CIs)
MIN_DEV_POSITIVES = 20
SUPPORT_MIN = 10      # development scores required at or beyond p in that tail
RULES = [("primary 5 / 0.2", 5.0, 0.2), ("sensitivity 2 / 0.5", 2.0, 0.5),
         ("sensitivity 10 / 0.1", 10.0, 0.1)]
STATES = ["SUPPORTING", "UNCERTAIN", "REFUTING", "INSUFFICIENT DATA"]
GRID = 1 / (1 + np.exp(-np.linspace(-9.5, 9.5, 1901)))
BANDS = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001]
LR_CAP = 1e6


def verify():
    for name, expected in EXPECTED_SHA256.items():
        actual = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
        assert actual == expected, f"{name} changed: {actual} != {expected}"


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def odds(x):
    return x / (1 - x)


def fit_platt(p, y):
    m = LogisticRegression(C=np.inf, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


class Source:
    """LR function for one finding, learned on one dataset's development set."""

    def __init__(self, name, finding, platt, dev_p, dev_y, rng):
        self.name, self.finding = name, finding
        self.a, self.b = platt["coef"], platt["intercept"]
        self.dev_p, self.dev_y = np.asarray(dev_p, float), np.asarray(dev_y, int)
        self.pi = self.dev_y.mean()
        self.dev_pos = int(self.dev_y.sum())
        self.median = float(np.median(self.dev_p))
        self.sorted = np.sort(self.dev_p)
        # LR uncertainty: refit Platt on bootstrap resamples of the development set
        a_k, b_k, pi_k = [], [], []
        n = len(self.dev_p)
        while len(a_k) < N_REFIT:
            idx = rng.integers(0, n, n)
            y = self.dev_y[idx]
            if 0 < y.sum() < n:
                a, b = fit_platt(self.dev_p[idx], y)
                a_k.append(a), b_k.append(b), pi_k.append(y.mean())
        a_k, b_k, pi_k = map(np.array, (a_k, b_k, pi_k))
        # Work in log-LR so extreme scores cannot overflow; percentiles commute with exp
        log_curves = (a_k[:, None] * logit(GRID)[None, :] + b_k[:, None]
                      - np.log(odds(pi_k))[:, None])
        self.lower = np.exp(np.clip(np.percentile(log_curves, 2.5, axis=0), -50, 50))
        self.upper = np.exp(np.clip(np.percentile(log_curves, 97.5, axis=0), -50, 50))

    def lr(self, p):
        return np.exp(np.clip(self.a * logit(p) + self.b - np.log(odds(self.pi)), -50, 50))

    def lr_ci(self, p):
        return np.interp(logit(p), logit(GRID), self.lower), np.interp(logit(p), logit(GRID), self.upper)

    def cutpoints(self, support_lr, refute_lr):
        above = GRID[self.lower >= support_lr]
        below = GRID[self.upper <= refute_lr]
        return (float(above.min()) if len(above) else None,
                float(below.max()) if len(below) else None)

    def in_support(self, p):
        p = np.asarray(p, float)
        n_ge = len(self.sorted) - np.searchsorted(self.sorted, p, side="left")
        n_le = np.searchsorted(self.sorted, p, side="right")
        return np.where(p >= self.median, n_ge >= SUPPORT_MIN, n_le >= SUPPORT_MIN)

    def states(self, p, support_lr, refute_lr):
        p = np.asarray(p, float)
        if self.dev_pos < MIN_DEV_POSITIVES:
            return np.full(len(p), "INSUFFICIENT DATA", dtype=object)
        p_s, p_r = self.cutpoints(support_lr, refute_lr)
        out = np.full(len(p), "UNCERTAIN", dtype=object)
        if p_s is not None:
            out[p >= p_s] = "SUPPORTING"
        if p_r is not None:
            out[p <= p_r] = "REFUTING"
        out[~self.in_support(p)] = "INSUFFICIENT DATA"
        return out


def ci(values):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (math.nan, math.nan)


def fmt_lr(x):
    if math.isnan(x):
        return "–"
    if x >= 1000:
        return "≥1000"
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def state_lr(states, y, state):
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return math.nan
    tp = (states[pos] == state).mean()
    fp = (states[neg] == state).mean()
    if fp == 0:
        return LR_CAP if tp > 0 else math.nan
    return tp / fp


def main():
    verify()
    rng = np.random.default_rng(SEED)

    indiana = json.loads((RESULTS / "frozen_dev_params.json").read_text(encoding="utf-8"))["findings"]
    nih = json.loads((RESULTS / "nih_frozen_dev_params.json").read_text(encoding="utf-8"))["findings"]
    ind_data = S.load()
    ind_split = json.loads((RESULTS / "split_indiana_70_30.json").read_text(encoding="utf-8"))
    nih_data, _ = N.load_readings()
    nih_split = json.loads((RESULTS / "split_nih_70_30.json").read_text(encoding="utf-8"))
    R, Sup = nih_data["random"], nih_data["supplement"]
    ind_dev = [i for i in sorted(ind_data) if ind_split[i] == "dev"]
    nih_dev_r = [i for i in sorted(R) if nih_split["random"][i] == "dev"]
    test_r = [i for i in sorted(R) if nih_split["random"][i] == "test"]
    test_s = [i for i in sorted(Sup) if nih_split["supplement"][i] == "test"]
    both = {**R, **Sup}

    def arr(data, images, f):
        return (np.array([data[i][f][0] for i in images], float),
                np.array([data[i][f][1] for i in images], int))

    sources = {}
    for f in S.FINDINGS:
        sources[("Indiana", f)] = Source("Indiana", f, indiana[f]["platt"], *arr(ind_data, ind_dev, f), rng)
        sources[("NIH", f)] = Source("NIH", f, nih[f]["platt"], *arr(R, nih_dev_r, f), rng)

    L = [
        "# Evidence Engine D1/D2 results",
        "",
        "Protocol: `experiments/evidence_engine_design.md` (committed before this run). Inputs "
        "SHA-256 verified against `results/REPRODUCE.md`. NIH test: random part n = "
        f"{len(test_r)} (natural prevalence), supplement n = {len(test_s)}. LR CIs from {N_REFIT} "
        f"Platt refits on the source development set; reported CIs from {N_BOOT} image-level "
        f"bootstrap resamples of the test set. Seed {SEED}.",
    ]

    # ---------------- D1 ----------------
    L += ["", "## D1: how much of the Indiana → NIH calibration shift is prevalence shift?", "",
          "NIH test, random part. B = Indiana Platt; B′ = Indiana LR × NIH development prevalence "
          "(aggregate count only); C = NIH Platt. Brier (lower is better) with 95% CI. "
          "Share explained = (B − B′) / (B − C).", "",
          "| Finding | NIH dev prevalence | Brier raw | Brier B | Brier B′ | Brier C | B − B′ (prevalence) | "
          "B′ − C (residual) | Share explained |", "|---|---|---|---|---|---|---|---|---|"]
    diag = []
    boots = [rng.integers(0, len(test_r), len(test_r)) for _ in range(N_BOOT)]
    for f in S.FINDINGS:
        p, y = arr(R, test_r, f)
        si, sn = sources[("Indiana", f)], sources[("NIH", f)]
        q_b = 1 / (1 + np.exp(-(si.a * logit(p) + si.b)))
        post_odds = odds(sn.pi) * si.lr(p)
        q_bp = post_odds / (1 + post_odds)
        q_c = 1 / (1 + np.exp(-(sn.a * logit(p) + sn.b)))

        def stats(idx):
            yy = y[idx]
            br = {k: np.mean((q[idx] - yy) ** 2) for k, q in
                  (("raw", p), ("B", q_b), ("Bp", q_bp), ("C", q_c))}
            gap = br["B"] - br["C"]
            return {**br, "d1": br["B"] - br["Bp"], "d2": br["Bp"] - br["C"],
                    "share": (br["B"] - br["Bp"]) / gap if gap > 1e-9 else math.nan}

        pt = stats(np.arange(len(p)))
        bs = [stats(b) for b in boots]
        c = {k: ci([b[k] for b in bs]) for k in pt}
        L.append(
            f"| {f} | {sn.pi:.3f} | " + " | ".join(
                f"{pt[k]:.3f} [{c[k][0]:.3f}–{c[k][1]:.3f}]" for k in ("raw", "B", "Bp", "C", "d1", "d2"))
            + f" | {pt['share']:.0%} [{c['share'][0]:.0%}–{c['share'][1]:.0%}] |")
        cells = []
        for q in (q_b, q_bp, q_c):
            slope = fit_platt(q, y)[0]
            cells.append(f"{q.mean() - y.mean():+.3f} / {slope:.2f}")
        diag.append(f"| {f} | {y.mean():.3f} | " + " | ".join(cells) + " |")
    L += ["", "Calibration diagnostics (NIH test, random part): calibration-in-the-large "
          "(mean predicted − observed; 0 is ideal) / calibration slope (1 is ideal).", "",
          "| Finding | Observed prevalence | B | B′ | C |", "|---|---|---|---|---|", *diag]

    # ---------------- D2 ----------------
    L += ["", "## D2: does the LR keep its meaning across datasets?", "",
          "### State boundaries (raw p_present) under the primary rule", "",
          "SUPPORTING if p ≥ p_S (lower 95% CI of LR ≥ 5); REFUTING if p ≤ p_R (upper 95% CI of "
          "LR ≤ 0.2); outside support → INSUFFICIENT DATA.", "",
          "| Finding | Source | Dev positives | p_S | p_R | Support range (p) |",
          "|---|---|---|---|---|---|"]
    for f in S.FINDINGS:
        for src in ("Indiana", "NIH"):
            s = sources[(src, f)]
            p_s, p_r = s.cutpoints(5, 0.2)
            lo = s.sorted[SUPPORT_MIN - 1]
            hi = s.sorted[-SUPPORT_MIN]
            L.append(f"| {f} | {src} | {s.dev_pos} | {p_s if p_s is None else f'{p_s:.3f}'} | "
                     f"{p_r if p_r is None else f'{p_r:.4f}'} | {lo:.4f}–{hi:.3f} |")

    def d2_table(title, images, data):
        out = ["", f"### {title}", "",
               "Empirical LR = P(state | finding) / P(state | no finding), 95% bootstrap CI; "
               "coverage = share of all cases in the state. ✓ = point estimate meets the "
               "criterion (SUPPORTING ≥ 5, REFUTING ≤ 0.2); ✓✓ = the CI also meets it.", "",
               "| Finding | Source | Pos / neg | SUPPORTING: coverage, LR | UNCERTAIN: coverage, LR | "
               "REFUTING: coverage, LR | INSUFFICIENT |", "|---|---|---|---|---|---|---|"]
        bidx = [rng.integers(0, len(images), len(images)) for _ in range(N_BOOT)]
        for f in S.FINDINGS:
            p, y = arr(data, images, f)
            for src in ("Indiana", "NIH"):
                st = sources[(src, f)].states(p, 5, 0.2)
                cells = []
                for state in STATES[:3]:
                    point = state_lr(st, y, state)
                    lo, hi = ci([state_lr(st[b], y[b], state) for b in bidx])
                    mark = ""
                    if state == "SUPPORTING" and not math.isnan(point):
                        mark = " ✓✓" if lo >= 5 else (" ✓" if point >= 5 else " ✗")
                    if state == "REFUTING" and not math.isnan(point):
                        mark = " ✓✓" if hi <= 0.2 else (" ✓" if point <= 0.2 else " ✗")
                    cov = (st == state).mean()
                    cells.append(f"{cov:.0%}, {fmt_lr(point)} [{fmt_lr(lo)}–{fmt_lr(hi)}]{mark}"
                                 if (st == state).any() else f"{cov:.0%}, –")
                insuff = (st == "INSUFFICIENT DATA").mean()
                out.append(f"| {f} | {src} | {int(y.sum())} / {int((1 - y).sum())} | "
                           + " | ".join(cells) + f" | {insuff:.0%} |")
        return out

    L += d2_table("Primary: NIH test, random part (natural prevalence)", test_r, R)
    L += d2_table("Secondary: NIH test, random + supplement", test_r + test_s, both)

    # Sensitivity analyses
    L += ["", "### Pre-registered sensitivity analyses (NIH test, random + supplement)", "",
          "Coverage and empirical LR of SUPPORTING / REFUTING under each rule.", "",
          "| Finding | Source | Rule | UNCERTAIN coverage | SUPPORTING: coverage, LR | REFUTING: coverage, LR |",
          "|---|---|---|---|---|---|"]
    images = test_r + test_s
    for f in S.FINDINGS:
        p, y = arr(both, images, f)
        for src in ("Indiana", "NIH"):
            for name, ls, lr_ in RULES:
                st = sources[(src, f)].states(p, ls, lr_)
                cells = []
                for state in ("SUPPORTING", "REFUTING"):
                    cells.append(f"{(st == state).mean():.0%}, {fmt_lr(state_lr(st, y, state))}")
                L.append(f"| {f} | {src} | {name} | {(st == 'UNCERTAIN').mean():.0%} | "
                         + " | ".join(cells) + " |")

    # Platt check against empirical binned LR
    L += ["", "### Platt LR vs empirical LR by score band (NIH test, random + supplement)", "",
          "Empirical LR = P(band | finding) / P(band | no finding) with 95% CI; Platt LR = mean "
          "Platt-derived LR of the cases in the band (Indiana / NIH source). ✓ = Platt LR within "
          "the empirical CI.", "",
          "| Finding | " + " | ".join(f"{lo:g}–{min(hi, 1):g}" for lo, hi in zip(BANDS[:-1], BANDS[1:])) + " |",
          "|---|" + "---|" * (len(BANDS) - 1)]
    bidx = [rng.integers(0, len(images), len(images)) for _ in range(N_BOOT)]
    for f in S.FINDINGS:
        p, y = arr(both, images, f)
        cells = []
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            band = np.where((p >= lo) & (p < hi), "IN", "OUT").astype(object)
            if not (band == "IN").any():
                cells.append("–")
                continue
            emp = state_lr(band, y, "IN")
            elo, ehi = ci([state_lr(band[b], y[b], "IN") for b in bidx])
            marks = []
            for src in ("Indiana", "NIH"):
                plr = float(np.mean(sources[(src, f)].lr(p[band == "IN"])))
                ok = "✓" if elo <= plr <= ehi else "✗"
                marks.append(f"{fmt_lr(plr)}{ok}")
            cells.append(f"{fmt_lr(emp)} [{fmt_lr(elo)}–{fmt_lr(ehi)}] · {' / '.join(marks)} "
                         f"(n={(band == 'IN').sum()})")
        L.append(f"| {f} | " + " | ".join(cells) + " |")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
