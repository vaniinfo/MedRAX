"""Final external validation of Evidence Engine v1.0 on VinDr-CXR (test split).

Protocol: experiments/vindr_external_validation_design.md (committed before the data was on
this machine). The engine is used unchanged from tag evidence-engine-v1.0.

    python experiments/vindr_external_validation.py prepare [--data DIR]   # labels + PNGs
    python experiments/vindr_external_validation.py run [--data DIR]       # CheXagent readings
    python experiments/vindr_external_validation.py analyze                # endpoints

DIR defaults to $VINDR_DIR or D:/datasets/vindr-cxr (PhysioNet VinDr-CXR v1.0.0 layout:
test/*.dicom and annotations/image_labels_test.csv).
"""

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
RESULTS = HERE / "results"
MANIFEST = RESULTS / "vindr_test_manifest.json"
READINGS = RESULTS / "vindr_readings.jsonl"
REPORT = RESULTS / "vindr_external_validation.md"

SEED = 20260927
N_BOOT = 2000
MIN_POSITIVES = 20
LABELS = {  # VinDr-CXR label -> Evidence Engine finding
    "Cardiomegaly": "cardiomegaly",
    "Pleural effusion": "pleural effusion",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "consolidation",
    "Edema": "pulmonary edema",
    "Atelectasis": "atelectasis",
}
FINDINGS = list(LABELS.values())


def data_dir(arg):
    return Path(arg or os.environ.get("VINDR_DIR", "D:/datasets/vindr-cxr"))


def dicom_to_png(src: Path, dst: Path):
    """The pre-registered conversion: VOI LUT, MONOCHROME1 inversion, min-max to 8-bit."""
    import pydicom
    from PIL import Image
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds = pydicom.dcmread(str(src))
    img = apply_voi_lut(ds.pixel_array, ds).astype(np.float64)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    img = (img - img.min()) / max(img.max() - img.min(), 1e-9) * 255.0
    Image.fromarray(img.astype(np.uint8)).save(dst)


def prepare(data):
    import pandas as pd

    if MANIFEST.exists():
        print(f"{MANIFEST.name} already exists; not rebuilding")
        return
    label_files = list(data.rglob("image_labels_test.csv"))
    assert len(label_files) == 1, f"expected one image_labels_test.csv under {data}, found {label_files}"
    labels = pd.read_csv(label_files[0])
    missing = [c for c in LABELS if c not in labels.columns]
    assert not missing, f"label columns missing: {missing}; columns are {list(labels.columns)}"
    if labels["image_id"].duplicated().any():
        raise SystemExit("image_labels_test.csv has several rows per image; the protocol assumes one "
                         "consensus row per image, so stop and decide before continuing")

    dicoms = {p.stem: p for p in data.rglob("*.dicom")} or {p.stem: p for p in data.rglob("*.dcm")}
    png_dir = data / "png_test"
    png_dir.mkdir(exist_ok=True)
    rows, failures = [], []
    records = labels.to_dict("records")
    for k, r in enumerate(records, 1):
        image_id = r["image_id"]
        src = dicoms.get(image_id)
        dst = png_dir / f"{image_id}.png"
        try:
            if src is None:
                raise FileNotFoundError("no DICOM for this image_id")
            if not dst.exists():
                dicom_to_png(src, dst)
        except Exception as e:  # listed and excluded, as pre-registered
            failures.append({"image_id": image_id, "error": f"{type(e).__name__}: {e}"})
            continue
        rows.append({"image_id": image_id, "png": str(dst),
                     "truth": {f: int(r[lab]) for lab, f in LABELS.items()}})
        if k % 250 == 0:
            print(f"{k}/{len(records)} converted", flush=True)

    RESULTS.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps({"labels_file": str(label_files[0]), "n": len(rows),
                                    "failures": failures, "images": rows}, indent=1), encoding="utf-8")
    os.chmod(MANIFEST, stat.S_IREAD)
    print(f"manifest: {len(rows)} images, {len(failures)} excluded (could not be decoded or found)")
    for f in FINDINGS:
        pos = sum(r["truth"][f] for r in rows)
        print(f"  {f:17} {pos} positives  {'evaluable' if pos >= MIN_POSITIVES else 'NOT evaluable (<20)'}")


def run(data):
    warnings.filterwarnings("ignore")
    from transformers import logging

    logging.set_verbosity_error()
    sys.path.insert(0, str(ROOT))
    from medrax.tools.xray_vqa import XRayVQATool

    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]
    done = set()
    if READINGS.exists():
        for line in READINGS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["image_id"], r["finding"]))
    todo = [(r, f) for r in rows for f in FINDINGS if (r["image_id"], f) not in done]
    print(f"{len(done)} readings done, {len(todo)} to go", flush=True)
    if not todo:
        return
    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    started = time.time()
    with READINGS.open("a", encoding="utf-8") as out:
        for i, (r, finding) in enumerate(todo, 1):
            text, yes_no = tool._generate_response([r["png"]], f"Is there {finding}?", 8)
            out.write(json.dumps({"image_id": r["image_id"], "finding": finding,
                                  "truth": r["truth"][finding], "answer": text,
                                  "p_present": yes_no["p_yes"] if yes_no else None}) + "\n")
            out.flush()
            if i % 300 == 0 or i == len(todo):
                rate = i / (time.time() - started)
                print(f"{i}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - i) / rate / 60:.0f} min", flush=True)


STATES = ["SUPPORTING", "UNCERTAIN", "REFUTING", "INSUFFICIENT DATA"]
REASONS = ["WEAK_EVIDENCE", "REFUTING_NOT_EMPIRICALLY_CONFIRMED", "OUTSIDE_SUPPORT",
           "FEW_DEV_POSITIVES", "FINDING_NOT_VALIDATED"]


def state_lr(states, y, state):
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return math.nan
    tp, fp = (states[pos] == state).mean(), (states[neg] == state).mean()
    if fp == 0:
        return 1e6 if tp > 0 else math.nan
    return tp / fp


def ci(values):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (math.nan, math.nan)


def fmt_lr(x):
    if math.isnan(x):
        return "–"
    return "≥1000" if x >= 1000 else (f"{x:.2f}" if x < 10 else f"{x:.1f}")


def analyze():
    sys.path.insert(0, str(ROOT))
    from sklearn.metrics import roc_auc_score

    from medrax.evidence import EvidenceEngine  # verifies the frozen checksums on load

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["images"]
    readings = {}
    for line in READINGS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        readings[(r["image_id"], r["finding"])] = r
    expected = len(rows) * len(FINDINGS)
    if len(readings) != expected or any(r["p_present"] is None for r in readings.values()):
        raise SystemExit(f"readings incomplete ({len(readings)}/{expected}); refusing to analyze")

    engines = {"NIH (primary)": "primary", "Indiana (comparator)": "comparator"}
    rules = ["primary 5 / 0.2", "sensitivity 2 / 0.5", "sensitivity 10 / 0.1"]
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(rows), len(rows)) for _ in range(N_BOOT)]

    L = ["# Final external validation of Evidence Engine v1.0 on VinDr-CXR", "",
         "Protocol: `experiments/vindr_external_validation_design.md` (committed before the data "
         "was on this machine). Engine: tag `evidence-engine-v1.0`, loaded unchanged with its "
         f"checksums verified. VinDr-CXR test split: {len(rows)} images "
         f"({len(manifest['failures'])} excluded as undecodable), 5-radiologist consensus labels. "
         f"CIs: {N_BOOT}-sample image-level bootstrap, seed {SEED}.", "",
         "## Findings: positives and evaluability", "",
         "| Finding | Positives / negatives | Evaluable (≥ 20 positives) | p_present AUC [95% CI] (descriptive) |",
         "|---|---|---|---|"]
    data = {}
    for f in FINDINGS:
        p = np.array([readings[(r["image_id"], f)]["p_present"] for r in rows])
        y = np.array([r["truth"][f] for r in rows])
        data[f] = (p, y)
        evaluable = y.sum() >= MIN_POSITIVES
        if 0 < y.sum() < len(y):
            aucs = ci([roc_auc_score(y[b], p[b]) for b in boots if 0 < y[b].sum() < len(b)])
            auc = f"{roc_auc_score(y, p):.3f} [{aucs[0]:.3f}–{aucs[1]:.3f}]"
        else:
            auc = "–"
        L.append(f"| {f} | {int(y.sum())} / {int((1 - y).sum())} | {'yes' if evaluable else 'no'} | {auc} |")

    for rule in rules:
        primary = rule == rules[0]
        L += ["", f"## {'Primary endpoints' if primary else 'Descriptive'}: rule {rule}", "",
              "Empirical LR = P(state | finding) / P(state | no finding) [95% CI]. SUPPORTING: ✓ point "
              "≥ criterion, ✓✓ lower CI ≥ criterion. REFUTING: ✓ point ≤ criterion, ✓✓ upper CI ≤ "
              "criterion. ✗ = not met. Coverage = share of all images in the state.", "",
              "| Finding | Engine | SUPPORTING: coverage, LR | UNCERTAIN coverage | REFUTING: coverage, LR | "
              "Positives assigned to REFUTING | INSUFFICIENT coverage |",
              "|---|---|---|---|---|---|---|"]
        support_lr, refute_lr = (5, 0.2) if primary else ((2, 0.5) if "2 /" in rule else (10, 0.1))
        reason_rows = []
        for f in FINDINGS:
            p, y = data[f]
            if y.sum() < MIN_POSITIVES:
                L.append(f"| {f} | – | not evaluable ({int(y.sum())} positives) | | | | |")
                continue
            for label, key in engines.items():
                engine = EvidenceEngine(engine=key, rule=rule)
                ev = [engine.assess(f, float(x)) for x in p]
                states = np.array([e.state for e in ev], dtype=object)
                reasons = np.array([e.reason for e in ev], dtype=object)
                cells = {}
                for state, crit in (("SUPPORTING", support_lr), ("REFUTING", refute_lr)):
                    point = state_lr(states, y, state)
                    if math.isnan(point):
                        cells[state] = f"{(states == state).mean():.1%}, not observed"
                        continue
                    lo, hi = ci([state_lr(states[b], y[b], state) for b in boots])
                    if state == "SUPPORTING":
                        mark = "✓✓" if lo >= crit else ("✓" if point >= crit else "✗")
                    else:
                        mark = "✓✓" if hi <= crit else ("✓" if point <= crit else "✗")
                    cells[state] = (f"{(states == state).mean():.1%}, {fmt_lr(point)} "
                                    f"[{fmt_lr(lo)}–{fmt_lr(hi)}] {mark}")
                assigned = int(((states == "REFUTING") & (y == 1)).sum())
                L.append(f"| {f} | {label} | {cells['SUPPORTING']} | {(states == 'UNCERTAIN').mean():.1%} | "
                         f"{cells['REFUTING']} | {assigned} / {int(y.sum())} ({assigned / y.sum():.0%}) | "
                         f"{(states == 'INSUFFICIENT DATA').mean():.1%} |")
                if primary:
                    reason_rows.append(f"| {f} | {label} | " + " | ".join(
                        f"{(reasons == r).mean():.1%}" for r in REASONS) + " |")
        if primary:
            L += ["", "### UNCERTAIN / INSUFFICIENT DATA by reason (primary rule)", "",
                  "| Finding | Engine | " + " | ".join(REASONS) + " |",
                  "|---|---|" + "---|" * len(REASONS), *reason_rows]

    L += ["", "## Limitation (stated in advance)", "",
          "VinDr-CXR played no role in the engine's development, but CheXagent may have been "
          "trained on it; untouched by the engine is not the same as unseen by CheXagent."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "analyze"])
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    if args.command == "analyze":
        analyze()
    else:
        {"prepare": prepare, "run": run}[args.command](data_dir(args.data))
