"""Amendment A1 of maira_score_design.md: standalone CheXagent vs MAIRA-2 presence scores on three
held-out labelled sets (NIH test split, Indiana, VinDr).

    python experiments/maira_score_external.py score     # MAIRA-2 readings for Indiana and VinDr (resumable)
    python experiments/maira_score_external.py analyze   # AUC per model, paired difference, 95% CI
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
from maira_score_nih import SENTENCES, auc, sample  # noqa: E402

RESULTS = HERE / "results"
INDIANA = Path("C:/Users/hari/_workarea_/MedRAX/data/indiana_eval")
SEED, N_BOOT = 20261017, 2000
READINGS = {"Indiana": RESULTS / "maira_score_indiana.jsonl", "VinDr": RESULTS / "maira_score_vindr.jsonl"}
REPORT = RESULTS / "maira_vs_chexagent_external.md"


def images():
    """{dataset: [(image id, path)]}"""
    indiana = sorted({json.loads(l)["file"] for l in (RESULTS / "ppresent_indiana.jsonl").read_text(encoding="utf-8").splitlines()})
    vindr = json.loads((RESULTS / "vindr_test_manifest.json").read_text(encoding="utf-8"))["images"]
    return {"Indiana": [(f, INDIANA / f) for f in indiana], "VinDr": [(im["image_id"], Path(im["png"])) for im in vindr]}


def score():
    from PIL import Image

    from maira_score_nih import make_scorer

    scorer = None
    for name, items in images().items():
        path = READINGS[name]
        done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        todo = [(i, p) for i, p in items if i not in done]
        print(f"{name}: {len(done)} done, {len(todo)} to go", flush=True)
        if not todo:
            continue
        scorer = scorer or make_scorer()
        start = time.time()
        with path.open("a", encoding="utf-8") as out:
            for n, (i, p) in enumerate(todo, 1):
                out.write(json.dumps({"id": i, "findings": scorer(Image.open(p).convert("RGB"))}) + "\n")
                out.flush()
                if n % 200 == 0 or n == len(todo):
                    rate = n / (time.time() - start)
                    print(f"{name} {n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)


def paired(ds):
    """[(finding, truth array, chexagent array, maira array)] for one dataset."""
    out = []
    if ds == "NIH test":
        test = {f for f, s in sample() if s == "test"}
        maira = {json.loads(l)["file"]: json.loads(l)["findings"]
                 for l in (RESULTS / "maira_score_nih.jsonl").read_text(encoding="utf-8").splitlines()}
        chex = [json.loads(l) for l in (RESULTS / "ppresent_nih.jsonl").read_text(encoding="utf-8").splitlines()]
        chex = [r for r in chex if r["set"] == "random" and r["file"] in test]
        key = "file"
    else:
        maira = {json.loads(l)["id"]: json.loads(l)["findings"] for l in READINGS[ds].read_text(encoding="utf-8").splitlines()}
        src = "ppresent_indiana.jsonl" if ds == "Indiana" else "vindr_readings.jsonl"
        chex = [json.loads(l) for l in (RESULTS / src).read_text(encoding="utf-8").splitlines()]
        key = "file" if ds == "Indiana" else "image_id"
    for finding in SENTENCES:
        if ds == "VinDr" and finding == "pulmonary edema":
            continue  # not labelled in the Kaggle-derived data
        rows = [r for r in chex if r["finding"] == finding]
        missing = [r[key] for r in rows if r[key] not in maira]
        if missing:
            raise SystemExit(f"{ds}: {len(missing)} images without a MAIRA-2 reading")
        out.append((finding, np.array([r["truth"] for r in rows]), np.array([r["p_present"] for r in rows]),
                    np.array([maira[r[key]][finding]["score"] for r in rows])))
    return out


def analyze():
    rng = np.random.default_rng(SEED)
    lines = ["# Standalone presence scores: CheXagent vs MAIRA-2 on three held-out labelled sets", "",
             "Amendment A1 of `maira_score_design.md` (committed before these readings and analyses). AUC: how often "
             "a model scores an image with the finding above one without it (0.5 = chance). 95% CIs from "
             f"{N_BOOT} image-level bootstrap resamples (seed {SEED}); the difference is paired (same images). "
             "Hypothesis: CheXagent's AUC is higher - supported if the difference's CI lies above 0.", "",
             "Caveat: CheXagent's training data probably include public NIH and VinDr images; MAIRA-2's published "
             "training sets do not. Indiana is the least affected set.", "",
             "| dataset | finding | positives / n | CheXagent AUC [95% CI] | MAIRA-2 AUC [95% CI] | "
             "CheXagent minus MAIRA-2 [95% CI] | hypothesis |", "|---|---|---|---|---|---|---|"]
    tally = {"supported": 0, "not supported": 0, "reversed": 0}
    for ds in ("NIH test", "Indiana", "VinDr"):
        for finding, y, c, m in paired(ds):
            pos = int(y.sum())
            ac, am = auc(c[y == 1], c[y == 0]), auc(m[y == 1], m[y == 0])
            bc, bm = [], []
            for _ in range(N_BOOT):
                b = rng.integers(0, len(y), len(y))
                yb = y[b]
                bc.append(auc(c[b][yb == 1], c[b][yb == 0]))
                bm.append(auc(m[b][yb == 1], m[b][yb == 0]))
            bc, bm = np.array(bc), np.array(bm)
            ok = np.isfinite(bc) & np.isfinite(bm)
            (cl, ch), (ml, mh), (dl, dh) = (np.percentile(v[ok], [2.5, 97.5]) for v in (bc, bm, bc - bm))
            verdict = "supported" if dl > 0 else "reversed" if dh < 0 else "not supported"
            tally[verdict] += 1
            note = " (fewer than 10 positives)" if pos < 10 else ""
            lines.append(f"| {ds} | {finding} | {pos} / {len(y)} | {ac:.3f} [{cl:.3f}-{ch:.3f}] | {am:.3f} [{ml:.3f}-{mh:.3f}] | "
                         f"{ac - am:+.3f} [{dl:+.3f} to {dh:+.3f}] | {verdict}{note} |")
    lines += ["", f"Summary over {sum(tally.values())} comparisons: supported {tally['supported']}, not supported "
                  f"{tally['not supported']}, reversed {tally['reversed']}."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"score": score, "analyze": analyze}[sys.argv[1]]()
