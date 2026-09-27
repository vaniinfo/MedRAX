"""External validation of CheXagent p_present on NIH ChestX-ray14.

    python experiments/ppresent_nih.py sample      # draw the 5,000-image sample (once)
    python experiments/ppresent_nih.py supplement  # draw the rare-finding supplement (once)
    python experiments/ppresent_nih.py run         # CheXagent readings; resumable
    python experiments/ppresent_nih.py analyze     # experiments A, B, C (see below)

Sample: one random on-disk image per patient, then 5,000 random patients (fixed seed), so
prevalence is NIH's natural rate and no patient appears twice. The manifest is written
once and made read-only.

Supplement: per-patient sampling leaves few positives for the rare findings, so 150 extra
patients per rare finding (edema, consolidation, pneumothorax, cardiomegaly) are drawn
from patients NOT in the random sample, each contributing one image positive for that
finding. Readings are tagged "random" or "supplement". The supplement is enriched by
construction, so it is used only for prevalence-independent analyses (AUC, thresholds,
sensitivity) and never for calibration or positive rates.

Questions (all scored with the same query as Indiana, "Is there {finding}?"):
  A. Indiana -> Indiana: reference, from experiments/ppresent_split.py.
  B. Indiana -> NIH: the FROZEN Indiana thresholds and Platt calibration
     (results/frozen_dev_params.json, checked against its SHA-256), applied unchanged.
  C. NIH -> NIH: thresholds and calibration fitted on a 70% NIH development split
     (stratified, one image per patient so image split = patient split), frozen to
     results/nih_frozen_dev_params.json, tested on the 30% NIH test split.
B and C are compared on the same NIH test images. Labels: NIH NLP-mined report labels
(noisy), mapped to the six findings below.
"""

import argparse
import hashlib
import json
import os
import random
import stat
import sys
import time
import warnings
from pathlib import Path

NIH = Path("D:/datasets/chestxray14")
RESULTS = Path(__file__).parent / "results"
MANIFEST = RESULTS / "nih_sample_5000.json"
SUPPLEMENT = RESULTS / "nih_supplement.json"
SUPPLEMENT_PER_FINDING = 150
RARE = ["pulmonary edema", "consolidation", "pneumothorax", "cardiomegaly"]
READINGS = RESULTS / "ppresent_nih.jsonl"
INDIANA_FROZEN = RESULTS / "frozen_dev_params.json"
INDIANA_SHA = RESULTS / "frozen_dev_params.sha256"
NIH_SPLIT = RESULTS / "split_nih_70_30.json"
NIH_FROZEN = RESULTS / "nih_frozen_dev_params.json"
REPORT = RESULTS / "ppresent_nih.md"

SEED = 20260927
N_SAMPLE = 5000
# NIH label -> finding name used for Indiana
LABELS = {
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "pleural effusion",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "consolidation",
    "Edema": "pulmonary edema",
    "Atelectasis": "atelectasis",
}
FINDINGS = list(LABELS.values())


def sample():
    import pandas as pd

    if MANIFEST.exists():
        print(f"{MANIFEST.name} already exists; not redrawing")
        return
    df = pd.read_csv(NIH / "Data_Entry_2017_v2020.csv")
    df = df[df["Image Index"].isin(set(os.listdir(NIH / "images")))]
    rng = random.Random(SEED)
    per_patient = {}
    for pid, group in df.groupby("Patient ID"):
        per_patient[int(pid)] = group.iloc[rng.randrange(len(group))]
    patients = sorted(per_patient)
    rng.shuffle(patients)
    chosen = sorted(patients[:N_SAMPLE])
    rows = []
    for pid in chosen:
        r = per_patient[pid]
        labels = set(r["Finding Labels"].split("|"))
        rows.append({
            "file": r["Image Index"], "patient": pid, "view": r["View Position"],
            "truth": {finding: int(label in labels) for label, finding in LABELS.items()},
        })
    RESULTS.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps({"seed": SEED, "n": len(rows), "images": rows}, indent=1),
                        encoding="utf-8")
    os.chmod(MANIFEST, stat.S_IREAD)
    print(f"sampled {len(rows)} images from {len(chosen)} patients")
    for f in FINDINGS:
        print(f"  {f:17} {sum(r['truth'][f] for r in rows)} positives")
    print("  views:", {v: sum(r['view'] == v for r in rows) for v in ("PA", "AP")})


def supplement():
    import pandas as pd

    if SUPPLEMENT.exists():
        print(f"{SUPPLEMENT.name} already exists; not redrawing")
        return
    sampled = {r["patient"] for r in json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]}
    df = pd.read_csv(NIH / "Data_Entry_2017_v2020.csv")
    df = df[df["Image Index"].isin(set(os.listdir(NIH / "images")))]
    df = df[~df["Patient ID"].isin(sampled)]
    rng = random.Random(SEED + 1)
    by_finding = {finding: label for label, finding in LABELS.items()}
    used, rows = set(), []
    for finding in RARE:
        positives = df[df["Finding Labels"].str.split("|").apply(lambda ls: by_finding[finding] in ls)]
        patients = sorted(set(positives["Patient ID"]) - used)
        rng.shuffle(patients)
        for pid in patients[:SUPPLEMENT_PER_FINDING]:
            group = positives[positives["Patient ID"] == pid]
            r = group.iloc[rng.randrange(len(group))]
            labels = set(r["Finding Labels"].split("|"))
            rows.append({
                "file": r["Image Index"], "patient": int(pid), "view": r["View Position"],
                "selected_for": finding,
                "truth": {f: int(label in labels) for label, f in LABELS.items()},
            })
            used.add(pid)
    SUPPLEMENT.write_text(json.dumps({"seed": SEED + 1, "n": len(rows), "images": rows}, indent=1),
                          encoding="utf-8")
    os.chmod(SUPPLEMENT, stat.S_IREAD)
    print(f"supplement: {len(rows)} images from {len(used)} patients not in the random sample")
    for f in FINDINGS:
        print(f"  {f:17} {sum(r['truth'][f] for r in rows)} positives")


def run():
    warnings.filterwarnings("ignore")
    from transformers import logging

    logging.set_verbosity_error()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from medrax.tools.xray_vqa import XRayVQATool

    # Random sample first, so it completes before the supplement if the run is cut short
    rows = [{**r, "set": "random"} for r in json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]]
    if SUPPLEMENT.exists():
        rows += [{**r, "set": "supplement"}
                 for r in json.loads(SUPPLEMENT.read_text(encoding="utf-8"))["images"]]
    done = set()
    if READINGS.exists():
        for line in READINGS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["file"], r["finding"]))
    todo = [(r, f) for r in rows for f in FINDINGS if (r["file"], f) not in done]
    print(f"{len(done)} readings done, {len(todo)} to go", flush=True)
    if not todo:
        return

    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    started = time.time()
    with READINGS.open("a", encoding="utf-8") as out:
        for i, (r, finding) in enumerate(todo, 1):
            text, yes_no = tool._generate_response(
                [str(NIH / "images" / r["file"])], f"Is there {finding}?", 8
            )
            out.write(json.dumps({
                "file": r["file"], "set": r["set"], "view": r["view"], "finding": finding,
                "truth": r["truth"][finding], "answer": text,
                "p_present": yes_no["p_yes"] if yes_no else None,
            }) + "\n")
            out.flush()
            if i % 300 == 0 or i == len(todo):
                rate = i / (time.time() - started)
                print(f"{i}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - i) / rate / 60:.0f} min",
                      flush=True)


SENS_TARGETS = (0.80, 0.90, 0.95)
BOOTSTRAP = 2000


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_readings():
    """{set: {file: {finding: (p, truth)}}} for images with all six readings, and views."""
    data, views = {"random": {}, "supplement": {}}, {}
    for line in READINGS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["p_present"] is None:
            continue
        data[r["set"]].setdefault(r["file"], {})[r["finding"]] = (r["p_present"], r["truth"])
        views[r["file"]] = r["view"]
    for s in data:
        data[s] = {f: v for f, v in data[s].items() if len(v) == len(FINDINGS)}
    return data, views


def sensitivity_threshold(p, y, target):
    """Highest threshold whose sensitivity (p >= t) reaches the target."""
    for t in sorted(set(p.tolist()), reverse=True):
        if (p[y == 1] >= t).mean() >= target:
            return float(t)
    return float(min(p))


def analyze(dry_run=False):
    global NIH_SPLIT, NIH_FROZEN, REPORT
    if dry_run:
        # Exercise the code on partial readings without creating the real frozen artifacts
        tmp = Path(os.environ.get("TEMP", "/tmp"))
        NIH_SPLIT, NIH_FROZEN, REPORT = (tmp / "dry_split_nih.json", tmp / "dry_nih_frozen.json",
                                         tmp / "dry_ppresent_nih.md")
        for path in (NIH_SPLIT, NIH_FROZEN):
            if path.exists():
                os.chmod(path, stat.S_IWRITE)
                path.unlink()
    sys.path.insert(0, str(Path(__file__).parent))
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import ppresent_split as S

    # B uses the Indiana parameters only if they are byte-identical to the frozen file
    expected = INDIANA_SHA.read_text(encoding="utf-8").split()[0]
    actual = sha256(INDIANA_FROZEN)
    assert actual == expected, f"frozen_dev_params.json changed: {actual} != {expected}"
    indiana = json.loads(INDIANA_FROZEN.read_text(encoding="utf-8"))["findings"]

    data, views = load_readings()
    R, Sup = data["random"], data["supplement"]
    n_random = json.loads(MANIFEST.read_text(encoding="utf-8"))["n"]
    n_supp = json.loads(SUPPLEMENT.read_text(encoding="utf-8"))["n"]
    if not dry_run and (len(R) != n_random or len(Sup) != n_supp):
        raise SystemExit(f"readings incomplete ({len(R)}/{n_random} random, {len(Sup)}/{n_supp} "
                         "supplement): refusing to split and freeze. Use --dry-run to test.")
    split_r, split_s = S.stratified_split(R), S.stratified_split(Sup)
    split = {"random": split_r, "supplement": split_s}
    if NIH_SPLIT.exists():
        assert json.loads(NIH_SPLIT.read_text(encoding="utf-8")) == split, "NIH split changed"
    else:
        NIH_SPLIT.write_text(json.dumps(split, indent=1, sort_keys=True), encoding="utf-8")
        os.chmod(NIH_SPLIT, stat.S_IREAD)
    dev_r = [i for i in sorted(R) if split_r[i] == "dev"]
    test_r = [i for i in sorted(R) if split_r[i] == "test"]
    dev_s = [i for i in sorted(Sup) if split_s[i] == "dev"]
    test_s = [i for i in sorted(Sup) if split_s[i] == "test"]
    both = {**R, **Sup}

    def arr(images, f):
        return (np.array([both[i][f][0] for i in images], float),
                np.array([both[i][f][1] for i in images], int))

    # C: fit on NIH development and freeze before scoring the test set
    if NIH_FROZEN.exists():
        nih = json.loads(NIH_FROZEN.read_text(encoding="utf-8"))
        print(f"using existing {NIH_FROZEN.name} (frozen {nih['created']}); not refitting")
    else:
        nih = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": S.SEED,
               "dev_random": len(dev_r), "dev_supplement": len(dev_s), "findings": {}}
        for f in FINDINGS:
            p, y = arr(dev_r + dev_s, f)
            t, j = S.youden_threshold(p, y)
            pr, yr = arr(dev_r, f)
            # Unregularized logistic fit, as for Indiana (C=inf replaces the deprecated penalty=None)
            model = LogisticRegression(C=np.inf, max_iter=1000).fit(S.logit(pr).reshape(-1, 1), yr)
            nih["findings"][f] = {
                "threshold": t, "dev_youden_j": round(j, 4),
                "sens_targets": {str(s): sensitivity_threshold(p, y, s) for s in SENS_TARGETS},
                "platt": {"coef": float(model.coef_[0][0]), "intercept": float(model.intercept_[0])},
                "dev_pos": int(y.sum()), "dev_neg": int((1 - y).sum()),
            }
        NIH_FROZEN.write_text(json.dumps(nih, indent=1), encoding="utf-8")
        os.chmod(NIH_FROZEN, stat.S_IREAD)

    rng = np.random.default_rng(S.SEED)
    fmt, ci = S.fmt, S.ci
    L = [
        "# External validation of CheXagent p_present: Indiana → NIH ChestX-ray14",
        "",
        f"NIH random sample: {len(R)} images (one per patient, natural prevalence); "
        f"supplement: {len(Sup)} rare-finding images from other patients (enriched; used only "
        "for AUC and thresholds, never for calibration). NIH split: stratified 70/30 by image = by "
        f"patient, seed {S.SEED}. Indiana parameters: `{INDIANA_FROZEN.name}`, SHA-256 verified "
        f"({actual[:12]}…). NIH parameters frozen at {nih['created']} (`{NIH_FROZEN.name}`) before "
        f"test scoring. 95% CIs: {BOOTSTRAP}-sample image-level bootstrap. NIH labels are NLP-mined "
        "from reports (noisy).",
        "",
        "## Sample sizes (positives / negatives)",
        "",
        "| Finding | NIH dev random | NIH dev suppl. | NIH test random | NIH test suppl. |",
        "|---|---|---|---|---|",
    ]
    for f in FINDINGS:
        cells = []
        for imgs in (dev_r, dev_s, test_r, test_s):
            _, y = arr(imgs, f)
            cells.append(f"{int(y.sum())} / {int((1 - y).sum())}")
        L.append(f"| {f} | " + " | ".join(cells) + " |")

    # A: Indiana -> Indiana reference (Indiana held-out test, same frozen parameters)
    ind_data = S.load()
    ind_split = json.loads((RESULTS / "split_indiana_70_30.json").read_text(encoding="utf-8"))
    ind_test = [i for i in sorted(ind_data) if ind_split[i] == "test"]

    def indiana_arr(f):
        return (np.array([ind_data[i][f][0] for i in ind_test], float),
                np.array([ind_data[i][f][1] for i in ind_test], int))

    L += ["", "## A / B / C at a glance", "",
          "AUC and Brier score (lower is better). A: Indiana test, Indiana calibration. "
          "B: NIH random sample (all 5,000 are unseen by B), Indiana calibration. "
          "C: NIH test (random part), NIH calibration.", "",
          "| Finding | A AUC | A Brier raw → cal | B AUC | B Brier raw → Indiana-cal | C Brier NIH-cal |",
          "|---|---|---|---|---|---|"]
    for f in FINDINGS:
        pa, ya = indiana_arr(f)
        cal_a = S.calibrate(indiana[f]["platt"], pa)
        pb, yb = arr(sorted(R), f)
        cal_b = S.calibrate(indiana[f]["platt"], pb)
        pc, yc = arr(test_r, f)
        cal_c = S.calibrate(nih["findings"][f]["platt"], pc)
        L.append(
            f"| {f} | {roc_auc_score(ya, pa):.2f} | {np.mean((pa - ya) ** 2):.3f} → {np.mean((cal_a - ya) ** 2):.3f} | "
            f"{roc_auc_score(yb, pb):.2f} | {np.mean((pb - yb) ** 2):.3f} → {np.mean((cal_b - yb) ** 2):.3f} | "
            f"{np.mean((cal_c - yc) ** 2):.3f} |"
        )

    # Calibration transfer: B vs C on the same NIH test images (random part only)
    L += ["", "## Calibration on the NIH test set (random part; natural prevalence)", "",
          "Same images for all columns. Δ = Indiana-cal − NIH-cal (positive: NIH recalibration "
          "is better).", "",
          "| Finding | Brier raw | Brier Indiana-cal (B) | Brier NIH-cal (C) | Δ Brier | ECE raw | "
          "ECE Indiana-cal | ECE NIH-cal |", "|---|---|---|---|---|---|---|---|"]
    boots_r = [rng.integers(0, len(test_r), len(test_r)) for _ in range(BOOTSTRAP)]
    reliability = []
    for f in FINDINGS:
        p, y = arr(test_r, f)
        cb = S.calibrate(indiana[f]["platt"], p)
        cc = S.calibrate(nih["findings"][f]["platt"], p)

        def cal_stats(idx):
            pp, yy, bb, cc_ = p[idx], y[idx], cb[idx], cc[idx]
            return {"raw": np.mean((pp - yy) ** 2), "b": np.mean((bb - yy) ** 2),
                    "c": np.mean((cc_ - yy) ** 2), "d": np.mean((bb - yy) ** 2) - np.mean((cc_ - yy) ** 2),
                    "eraw": S.ece(pp, yy), "eb": S.ece(bb, yy), "ec": S.ece(cc_, yy)}

        pt = cal_stats(np.arange(len(test_r)))
        bs = [cal_stats(b) for b in boots_r]
        c = {k: ci([b[k] for b in bs]) for k in pt}
        L.append(f"| {f} | {fmt(pt['raw'], *c['raw'], 3)} | {fmt(pt['b'], *c['b'], 3)} | "
                 f"{fmt(pt['c'], *c['c'], 3)} | {fmt(pt['d'], *c['d'], 3)} | {pt['eraw']:.3f} | "
                 f"{pt['eb']:.3f} | {pt['ec']:.3f} |")
        cells = []
        for lo, hi in S.BANDS:
            m = (p >= lo) & (p < hi)
            cells.append(f"{y[m].mean():.0%} / {cb[m].mean():.0%} / {cc[m].mean():.0%} (n={m.sum()})"
                         if m.any() else "–")
        reliability.append(f"| {f} | " + " | ".join(cells) + " |")
    L += ["", "ECE is biased upward under bootstrap, so its intervals are omitted; use the Brier "
          "intervals.", "", "### Reliability by raw p_present band (NIH test, random part)", "",
          "Observed positive rate / mean Indiana-calibrated / mean NIH-calibrated.", "",
          "| Finding | " + " | ".join(f"{lo:.1f}–{min(hi, 1):.1f}" for lo, hi in S.BANDS) + " |",
          "|---|" + "---|" * len(S.BANDS), *reliability]

    # Operating points on the NIH test set (random + supplement: prevalence-independent)
    test_all = test_r + test_s
    boots_all = [rng.integers(0, len(test_all), len(test_all)) for _ in range(BOOTSTRAP)]
    L += ["", "## Operating points on the NIH test set (random + supplement)", "",
          "Sensitivity / specificity with 95% CI. Thresholds: universal 0.50; Indiana-frozen "
          "Youden T; NIH-dev Youden T; NIH-dev thresholds reaching a target sensitivity.", "",
          "| Finding | Test AUC | Rule | Threshold | Sensitivity | Specificity |",
          "|---|---|---|---|---|---|"]
    for f in FINDINGS:
        p, y = arr(test_all, f)
        rules = [("universal 0.50", 0.5), ("Indiana Youden (B)", indiana[f]["threshold"]),
                 ("NIH Youden (C)", nih["findings"][f]["threshold"])]
        rules += [(f"NIH sens ≥ {s:.2f}", nih["findings"][f]["sens_targets"][str(s)]) for s in SENS_TARGETS]
        aucs = ci([roc_auc_score(y[b], p[b]) for b in boots_all if 0 < y[b].sum() < len(b)])
        for k, (name, t) in enumerate(rules):
            ss = [S.sens_spec(p[b], y[b], t) for b in boots_all]
            sens, spec = S.sens_spec(p, y, t)
            first = f"| {f} | {fmt(roc_auc_score(y, p), *aucs)} |" if k == 0 else "| | |"
            L.append(f"{first} {name} | {t:.3f} | {fmt(sens, *ci([s for s, _ in ss]))} | "
                     f"{fmt(spec, *ci([c for _, c in ss]))} |")

    # View: PA vs AP
    L += ["", "## AUC by view (NIH test, random + supplement)", "",
          "| Finding | PA AUC (pos/neg) | AP AUC (pos/neg) |", "|---|---|---|"]
    for f in FINDINGS:
        cells = []
        for v in ("PA", "AP"):
            imgs = [i for i in test_all if views[i] == v]
            p, y = arr(imgs, f)
            cells.append(f"{roc_auc_score(y, p):.2f} ({int(y.sum())}/{int((1 - y).sum())})"
                         if 0 < y.sum() < len(y) else "–")
        L.append(f"| {f} | " + " | ".join(cells) + " |")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sample", "supplement", "run", "analyze"])
    parser.add_argument("--dry-run", action="store_true",
                        help="analyze partial readings, writing only temporary files")
    args = parser.parse_args()
    if args.command == "analyze":
        analyze(dry_run=args.dry_run)
    else:
        {"sample": sample, "supplement": supplement, "run": run}[args.command]()
