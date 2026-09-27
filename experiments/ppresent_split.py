"""Held-out test of per-finding p_present thresholds and calibration (Indiana eval set).

Uses the readings from experiments/ppresent_indiana.py (no new model inference).

Protocol, fixed before looking at any test result:
  1. Stratified image-level 70/30 split (iterative multi-label stratification, fixed
     seed): all six findings of an X-ray stay in the same split.
  2. On the 70% development set only, per finding:
       - threshold T = argmax Youden's J (sensitivity + specificity - 1) over the observed
         p_present values, rule "positive if p >= T"; ties go to the lowest threshold;
       - Platt calibration: logistic regression of the label on logit(p_present).
  3. Freeze: T and the calibration parameters are written to frozen_dev_params.json
     before the test set is read; the test evaluation takes them only from that file.
  4. On the 30% test set, with 2,000-sample image-level bootstrap 95% CIs:
       - AUC;
       - sensitivity / specificity at the universal 0.50 and at the frozen T, on the same
         images, and the paired difference (T - 0.50);
       - calibration: Brier score and expected calibration error (ECE, 5 equal-width
         bins) for raw vs calibrated p_present, and a reliability table.
  Nothing is retuned after seeing the test results.

    python experiments/ppresent_split.py
"""

import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RESULTS = Path(__file__).parent / "results"
READINGS = RESULTS / "ppresent_indiana.jsonl"
SPLIT_FILE = RESULTS / "split_indiana_70_30.json"
FROZEN = RESULTS / "frozen_dev_params.json"
REPORT = RESULTS / "ppresent_split.md"

SEED = 20260927
DEV_FRACTION = 0.7
BOOTSTRAP = 2000
FINDINGS = ["cardiomegaly", "pleural effusion", "pneumothorax", "consolidation",
            "pulmonary edema", "atelectasis"]
BANDS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]


def load():
    """Per image: {finding: (p_present, truth)}."""
    data = {}
    for line in READINGS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        data.setdefault(r["file"], {})[r["finding"]] = (r["p_present"], r["truth"])
    return data


def stratified_split(data):
    """Iterative stratification (Sechidis et al., 2011) into dev/test by image."""
    rng = random.Random(SEED)
    images = sorted(data)
    labels = {img: {f for f in FINDINGS if data[img][f][1]} for img in images}
    ratios = {"dev": DEV_FRACTION, "test": 1 - DEV_FRACTION}
    want_size = {s: r * len(images) for s, r in ratios.items()}
    want_label = {s: {f: r * sum(f in labels[i] for i in images) for f in FINDINGS}
                  for s, r in ratios.items()}
    assigned = {}

    def place(img, split):
        assigned[img] = split
        want_size[split] -= 1
        for f in labels[img]:
            want_label[split][f] -= 1

    while True:
        remaining = {f: [i for i in images if i not in assigned and f in labels[i]]
                     for f in FINDINGS}
        remaining = {f: imgs for f, imgs in remaining.items() if imgs}
        if not remaining:
            break
        label = min(remaining, key=lambda f: len(remaining[f]))  # rarest label first
        todo = remaining[label]
        rng.shuffle(todo)
        for img in todo:
            best = max(ratios, key=lambda s: (want_label[s][label], want_size[s], rng.random()))
            place(img, best)

    rest = [i for i in images if i not in assigned]  # images with no finding
    rng.shuffle(rest)
    for img in rest:
        place(img, max(ratios, key=lambda s: (want_size[s], rng.random())))
    return assigned


def arrays(data, images, finding):
    p = np.array([data[i][finding][0] for i in images], dtype=float)
    y = np.array([data[i][finding][1] for i in images], dtype=int)
    return p, y


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sens_spec(p, y, t):
    pred = p >= t
    pos, neg = y == 1, y == 0
    sens = pred[pos].mean() if pos.any() else math.nan
    spec = (~pred[neg]).mean() if neg.any() else math.nan
    return sens, spec


def youden_threshold(p, y):
    best_t, best_j = None, -1.0
    for t in np.unique(p):  # ascending, so ties keep the lowest threshold
        sens, spec = sens_spec(p, y, t)
        if sens + spec - 1 > best_j + 1e-12:
            best_t, best_j = float(t), sens + spec - 1
    return best_t, best_j


def calibrate(params, p):
    return 1 / (1 + np.exp(-(params["coef"] * logit(p) + params["intercept"])))


def ece(prob, y, bins=5):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if m.any():
            total += m.mean() * abs(prob[m].mean() - y[m].mean())
    return total


def ci(values):
    v = np.array([x for x in values if not math.isnan(x)])
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (math.nan, math.nan)


def fmt(x, lo, hi, digits=2):
    return f"{x:.{digits}f} [{lo:.{digits}f}–{hi:.{digits}f}]"


def develop(data, split):
    """Fit thresholds and calibration on the development set only, then freeze them."""
    dev = [i for i in sorted(data) if split[i] == "dev"]
    frozen = {"created": datetime.now().isoformat(timespec="seconds"), "seed": SEED,
              "dev_images": len(dev), "findings": {}}
    for f in FINDINGS:
        p, y = arrays(data, dev, f)
        t, j = youden_threshold(p, y)
        model = LogisticRegression(penalty=None, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        frozen["findings"][f] = {
            "threshold": t, "dev_youden_j": round(j, 4),
            "platt": {"coef": float(model.coef_[0][0]), "intercept": float(model.intercept_[0])},
            "dev_pos": int(y.sum()), "dev_neg": int((1 - y).sum()),
        }
    FROZEN.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    return frozen


def evaluate(data, split):
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))  # only source of dev parameters
    test = [i for i in sorted(data) if split[i] == "test"]
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(test), len(test)) for _ in range(BOOTSTRAP)]

    lines = [
        "# Held-out test: universal 0.50 vs per-finding threshold for p_present",
        "",
        f"Split: stratified by image, 70/30, seed {SEED} "
        f"({sum(v == 'dev' for v in split.values())} development / {len(test)} test images). "
        f"Thresholds (Youden's J) and Platt calibration fitted on development only and frozen "
        f"at {frozen['created']} (`{FROZEN.name}`) before the test set was evaluated. "
        f"95% CIs: {BOOTSTRAP}-sample image-level bootstrap of the test set.",
        "",
        "**Enriched corpus** (studies selected per finding): AUC and sensitivity/specificity "
        "transfer; absolute positive rates are optimistic for a real clinic.",
        "",
        "## Sample sizes",
        "",
        "| Finding | Dev pos / neg | Test pos / neg |",
        "|---|---|---|",
    ]
    for f in FINDINGS:
        _, y = arrays(data, test, f)
        d = frozen["findings"][f]
        lines.append(f"| {f} | {d['dev_pos']} / {d['dev_neg']} | {int(y.sum())} / {int((1 - y).sum())} |")

    lines += ["", "## Discrimination and the threshold comparison (test set)", "",
              "| Finding | AUC | Frozen T | Sens @0.50 | Sens @T | Δ sens (T − 0.50) | "
              "Spec @0.50 | Spec @T | Δ spec (T − 0.50) | Δ Youden J |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    calib_rows, reliability = [], []
    for f in FINDINGS:
        p, y = arrays(data, test, f)
        t = frozen["findings"][f]["threshold"]
        params = frozen["findings"][f]["platt"]
        cal = calibrate(params, p)

        def stats(idx):
            pb, yb, cb = p[idx], y[idx], cal[idx]
            s5, c5 = sens_spec(pb, yb, 0.5)
            st, ct = sens_spec(pb, yb, t)
            a = roc_auc_score(yb, pb) if 0 < yb.sum() < len(yb) else math.nan
            return {"auc": a, "s5": s5, "st": st, "c5": c5, "ct": ct,
                    "ds": st - s5, "dc": ct - c5, "dj": (st + ct) - (s5 + c5),
                    "brier_raw": np.mean((pb - yb) ** 2), "brier_cal": np.mean((cb - yb) ** 2),
                    "ece_raw": ece(pb, yb), "ece_cal": ece(cb, yb)}

        point = stats(np.arange(len(test)))
        boot = [stats(b) for b in boots]
        c = {k: ci([b[k] for b in boot]) for k in point}
        lines.append(
            f"| {f} | {fmt(point['auc'], *c['auc'])} | {t:.3f} | "
            f"{fmt(point['s5'], *c['s5'])} | {fmt(point['st'], *c['st'])} | {fmt(point['ds'], *c['ds'])} | "
            f"{fmt(point['c5'], *c['c5'])} | {fmt(point['ct'], *c['ct'])} | {fmt(point['dc'], *c['dc'])} | "
            f"{fmt(point['dj'], *c['dj'])} |"
        )
        calib_rows.append(
            f"| {f} | {fmt(point['brier_raw'], *c['brier_raw'], 3)} | {fmt(point['brier_cal'], *c['brier_cal'], 3)} | "
            f"{fmt(point['ece_raw'], *c['ece_raw'], 3)} | {fmt(point['ece_cal'], *c['ece_cal'], 3)} |"
        )
        cells = []
        for lo, hi in BANDS:
            m = (p >= lo) & (p < hi)
            cells.append(f"{y[m].mean():.0%} obs / {cal[m].mean():.0%} cal (n={m.sum()})" if m.any() else "–")
        reliability.append(f"| {f} | " + " | ".join(cells) + " |")

    lines += ["", "Δ columns are paired on the same test images; a CI excluding 0 means the "
              "per-finding threshold changes that metric beyond bootstrap noise.", "",
              "## Calibration (test set; lower is better)", "",
              "| Finding | Brier raw p | Brier calibrated | ECE raw p | ECE calibrated |",
              "|---|---|---|---|---|", *calib_rows, "",
              "## Reliability on the test set, by raw p_present band", "",
              "Observed positive rate vs mean Platt-calibrated probability in each raw band.", "",
              "| Finding | " + " | ".join(f"{lo:.1f}–{min(hi, 1):.1f}" for lo, hi in BANDS) + " |",
              "|---|" + "---|" * len(BANDS), *reliability]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    data = load()
    assert all(len(v) == len(FINDINGS) for v in data.values()), "incomplete readings"
    split = stratified_split(data)
    SPLIT_FILE.write_text(json.dumps(split, indent=1, sort_keys=True), encoding="utf-8")
    develop(data, split)
    evaluate(data, split)


if __name__ == "__main__":
    main()
