"""Measure CheXagent's p_present against ground truth on the Indiana (Open-i) eval set.

For every image and finding, asks CheXagent "Is there {finding}?" and records p_present,
the Yes-vs-No softmax over its first answer token (XRayVQATool._yes_no_confidence). The
result says whether p_present separates positives from negatives, where a threshold
belongs, and what a given p_present is worth, i.e. what the confidence engine should be.

Ground truth comes from cxr-edv's compare_indiana.truth_for (MeSH problem terms mapped to
six findings), imported rather than copied so both projects share one definition.

    python experiments/ppresent_indiana.py run       # resumable; appends to the jsonl
    python experiments/ppresent_indiana.py analyze   # writes the markdown report

Caveat: the corpus was selected per finding (see its selected_for column), so prevalence
is enriched. AUC and sensitivity/specificity transfer to other settings; positive rates per
band (PPV-like numbers) are optimistic for a real clinic.
"""

import argparse
import csv
import json
import random
import sys
import time
import warnings
from pathlib import Path

DATA = Path("C:/Users/hari/_workarea_/MedRAX/data/indiana_eval")
CXR_EDV = Path("C:/Users/hari/_workarea_/xray/cxr-edv")
RESULTS = Path(__file__).parent / "results"
READINGS = RESULTS / "ppresent_indiana.jsonl"
REPORT = RESULTS / "ppresent_indiana.md"

sys.path.insert(0, str(CXR_EDV))
from compare_indiana import truth_for  # noqa: E402
from edv.findings import FINDINGS  # noqa: E402


def load_truth():
    with (DATA / "GROUND_TRUTH.csv").open(newline="", encoding="utf-8") as handle:
        return {row["file"]: truth_for(row) for row in csv.DictReader(handle)}


def run():
    warnings.filterwarnings("ignore")
    from transformers import logging

    logging.set_verbosity_error()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from medrax.tools.xray_vqa import XRayVQATool

    truth = load_truth()
    RESULTS.mkdir(exist_ok=True)
    done = set()
    if READINGS.exists():
        for line in READINGS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["file"], r["finding"]))

    todo = [(f, fi) for f in sorted(truth) for fi in FINDINGS if (f, fi) not in done]
    print(f"{len(done)} readings done, {len(todo)} to go", flush=True)
    if not todo:
        return

    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    started = time.time()
    with READINGS.open("a", encoding="utf-8") as out:
        for i, (file, finding) in enumerate(todo, 1):
            text, yes_no = tool._generate_response(
                [str(DATA / file)], f"Is there {finding}?", 8
            )
            out.write(json.dumps({
                "file": file,
                "finding": finding,
                "truth": truth[file][finding],
                "answer": text,
                "p_present": yes_no["p_yes"] if yes_no else None,
            }) + "\n")
            out.flush()
            if i % 60 == 0 or i == len(todo):
                rate = i / (time.time() - started)
                print(f"{i}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - i) / rate / 60:.1f} min",
                      flush=True)


def auc(pos, neg):
    """Probability a random positive scores above a random negative (ties count half)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def bootstrap_auc(pos, neg, reps=1000, seed=0):
    rng = random.Random(seed)
    values = sorted(
        auc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg]) for _ in range(reps)
    )
    return values[int(0.025 * reps)], values[int(0.975 * reps) - 1]


def confusion(rows, threshold):
    tp = sum(r["p"] >= threshold and r["truth"] for r in rows)
    fp = sum(r["p"] >= threshold and not r["truth"] for r in rows)
    fn = sum(r["p"] < threshold and r["truth"] for r in rows)
    tn = sum(r["p"] < threshold and not r["truth"] for r in rows)
    return tp, fp, fn, tn


def analyze():
    readings = [json.loads(l) for l in READINGS.read_text(encoding="utf-8").splitlines()]
    readings = [{**r, "p": r["p_present"]} for r in readings if r["p_present"] is not None]
    bands = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]

    lines = [
        "# CheXagent p_present vs ground truth — Indiana eval set",
        "",
        f"{len({r['file'] for r in readings})} frontal images, {len(readings)} readings. "
        "Query: `Is there {finding}?`; p_present = Yes/(Yes+No) over the first answer token. "
        "Truth: cxr-edv `truth_for` (MeSH problem terms).",
        "",
        "**Enriched corpus** (selected per finding): AUC and sensitivity/specificity "
        "transfer; the observed positive rate per band is optimistic for a real clinic.",
        "",
        "## Discrimination",
        "",
        "| Finding | Pos / Neg | AUC [95% CI] | Sens @0.5 | Spec @0.5 | Best threshold (Youden) | Sens / Spec there |",
        "|---|---|---|---|---|---|---|",
    ]
    for finding in FINDINGS:
        rows = [r for r in readings if r["finding"] == finding]
        pos = [r["p"] for r in rows if r["truth"]]
        neg = [r["p"] for r in rows if not r["truth"]]
        a = auc(pos, neg)
        lo, hi = bootstrap_auc(pos, neg) if pos and neg else (None, None)
        tp, fp, fn, tn = confusion(rows, 0.5)
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        best = max(
            (t / 100 for t in range(1, 100)),
            key=lambda t: (lambda c: c[0] / max(c[0] + c[2], 1) + c[3] / max(c[3] + c[1], 1))(
                confusion(rows, t)
            ),
        )
        btp, bfp, bfn, btn = confusion(rows, best)
        lines.append(
            f"| {finding} | {len(pos)} / {len(neg)} | "
            f"{a:.3f} [{lo:.3f}–{hi:.3f}] | {sens:.2f} | {spec:.2f} | {best:.2f} | "
            f"{btp / max(btp + bfn, 1):.2f} / {btn / max(btn + bfp, 1):.2f} |"
        )

    lines += [
        "",
        "## What a p_present value is worth (observed positive rate per band)",
        "",
        "| Finding | " + " | ".join(f"{lo:.1f}–{min(hi, 1):.1f}" for lo, hi in bands) + " |",
        "|---|" + "---|" * len(bands),
    ]
    for finding in FINDINGS:
        rows = [r for r in readings if r["finding"] == finding]
        cells = []
        for lo, hi in bands:
            in_band = [r for r in rows if lo <= r["p"] < hi]
            if in_band:
                rate = sum(r["truth"] for r in in_band) / len(in_band)
                cells.append(f"{rate:.0%} (n={len(in_band)})")
            else:
                cells.append("–")
        lines.append(f"| {finding} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Read across a row: if p_present were calibrated, the positive rate in each band "
        "would sit inside that band (e.g. ~80% in 0.7–0.9).",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "analyze"])
    {"run": run, "analyze": analyze}[parser.parse_args().command]()
