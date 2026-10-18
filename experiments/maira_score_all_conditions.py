"""Amendment A2 of maira_score_design.md: CheXagent vs MAIRA-2 presence scores for the seven further
conditions (NIH test; VinDr for the five it labels), and one report covering all 13 conditions
(A1's six-finding results listed with them, not recomputed).

    python experiments/maira_score_all_conditions.py score       # MAIRA-2 (NIH test, VinDr) and CheXagent (VinDr)
    python experiments/maira_score_all_conditions.py analyze     # -> results/maira_vs_chexagent_all_conditions.md
"""

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import provisional_findings as P  # noqa: E402
from maira_score_nih import NIH, auc, sample  # noqa: E402

RESULTS = HERE / "results"
SEED, N_BOOT = 20261017, 2000  # as in A1
SENTENCES7 = {
    "pneumonia": ("There is pneumonia.", "There is no pneumonia."),
    "infiltration": ("There is infiltration.", "There is no infiltration."),
    "emphysema": ("There is emphysema.", "There is no emphysema."),
    "nodule": ("There is a nodule.", "There is no nodule."),
    "mass": ("There is a mass.", "There is no mass."),
    "fibrosis": ("There is fibrosis.", "There is no fibrosis."),
    "pleural thickening": ("There is pleural thickening.", "There is no pleural thickening."),
}
VINDR_CLASS = {"infiltration": "Infiltration", "nodule": "Nodule/Mass", "mass": "Nodule/Mass",
               "fibrosis": "Pulmonary fibrosis", "pleural thickening": "Pleural thickening"}
MAIRA = {"NIH test": RESULTS / "maira_score7_nih_test.jsonl", "VinDr": RESULTS / "maira_score7_vindr.jsonl"}
CHEX_VINDR = RESULTS / "chexagent_vindr_extra.jsonl"
REPORT = RESULTS / "maira_vs_chexagent_all_conditions.md"


def vindr():
    """[(image id, png)] of the 3,000 VinDr images and {(id, condition): 0/1} (>= 2 of 3 readers)."""
    images = json.loads((RESULTS / "vindr_test_manifest.json").read_text(encoding="utf-8"))["images"]
    ids = {im["image_id"] for im in images}
    votes = defaultdict(set)
    with open("D:/datasets/vindr-kaggle/train.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["image_id"] in ids:
                votes[(r["image_id"], r["class_name"])].add(r["rad_id"])
    truth = {(i, c): int(len(votes.get((i, cls), ())) >= 2) for i in ids for c, cls in VINDR_CLASS.items()}
    return [(im["image_id"], im["png"]) for im in images], truth


def done_ids(path):
    return {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()


def score():
    import torch
    from PIL import Image

    sets = {"NIH test": [(f, NIH / f) for f, s in sample() if s == "test"], "VinDr": vindr()[0]}
    todo = {k: [(i, p) for i, p in v if i not in done_ids(MAIRA[k])] for k, v in sets.items()}
    if any(todo.values()):
        from maira_score_nih import make_scorer
        scorer = make_scorer(SENTENCES7)
        for name, items in todo.items():
            start = time.time()
            with MAIRA[name].open("a", encoding="utf-8") as out:
                for n, (i, p) in enumerate(items, 1):
                    out.write(json.dumps({"id": i, "findings": scorer(Image.open(p).convert("RGB"))}) + "\n")
                    out.flush()
                    if n % 200 == 0 or n == len(items):
                        rate = n / (time.time() - start)
                        print(f"MAIRA-2 {name} {n}/{len(items)}  {rate:.2f}/s  eta {(len(items) - n) / rate / 60:.0f} min",
                              flush=True)
        del scorer
        torch.cuda.empty_cache()
    todo = [(i, p) for i, p in vindr()[0] if i not in done_ids(CHEX_VINDR)]
    if todo:
        from medrax.tools.xray_vqa import XRayVQATool
        tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
        start = time.time()
        with CHEX_VINDR.open("a", encoding="utf-8") as out:
            for n, (i, p) in enumerate(todo, 1):
                rec = {"id": i, "p_present": {}}
                for c in VINDR_CLASS:
                    _, yes_no = tool._generate_response([p], f"Is there {c}?", 8)
                    rec["p_present"][c] = yes_no["p_yes"] if yes_no else None
                out.write(json.dumps(rec) + "\n")
                out.flush()
                if n % 200 == 0 or n == len(todo):
                    rate = n / (time.time() - start)
                    print(f"CheXagent VinDr {n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:.0f} min",
                          flush=True)


def paired():
    """[(dataset, condition, truth, chexagent, maira)] for A2."""
    out = []
    test = [f for f, s in sample() if s == "test"]
    _, labels = P.sample()
    chex = {(r["file"], r["finding"]): r["p_present"]
            for r in map(json.loads, (RESULTS / "provisional_readings.jsonl").read_text(encoding="utf-8").splitlines())}
    maira = {json.loads(l)["id"]: json.loads(l)["findings"] for l in MAIRA["NIH test"].read_text(encoding="utf-8").splitlines()}
    for c, label in P.FINDINGS.items():
        files = [f for f in test if chex.get((f, c)) is not None]
        out.append(("NIH test", c, np.array([int(label in labels[f]) for f in files]),
                    np.array([chex[(f, c)] for f in files]), np.array([maira[f][c]["score"] for f in files])))
    images, truth = vindr()
    cv = {json.loads(l)["id"]: json.loads(l)["p_present"] for l in CHEX_VINDR.read_text(encoding="utf-8").splitlines()}
    mv = {json.loads(l)["id"]: json.loads(l)["findings"] for l in MAIRA["VinDr"].read_text(encoding="utf-8").splitlines()}
    for c in VINDR_CLASS:
        ids = [i for i, _ in images if cv[i][c] is not None]
        out.append(("VinDr", c, np.array([truth[(i, c)] for i in ids]), np.array([cv[i][c] for i in ids]),
                    np.array([mv[i][c]["score"] for i in ids])))
    return out


def analyze():
    rng = np.random.default_rng(SEED)
    a1 = (RESULTS / "maira_vs_chexagent_external.md").read_text(encoding="utf-8")
    a1_rows = [l for l in a1.splitlines() if l.startswith("| ") and not l.startswith("| dataset")]
    new_rows, tally = [], {"supported": 0, "not supported": 0, "reversed": 0}
    for ds, c, y, ch, ma in paired():
        pos = int(y.sum())
        ac, am = auc(ch[y == 1], ch[y == 0]), auc(ma[y == 1], ma[y == 0])
        bc, bm = [], []
        for _ in range(N_BOOT):
            b = rng.integers(0, len(y), len(y))
            yb = y[b]
            bc.append(auc(ch[b][yb == 1], ch[b][yb == 0]))
            bm.append(auc(ma[b][yb == 1], ma[b][yb == 0]))
        bc, bm = np.array(bc), np.array(bm)
        ok = np.isfinite(bc) & np.isfinite(bm)
        (cl, chh), (ml, mh), (dl, dh) = (np.percentile(v[ok], [2.5, 97.5]) for v in (bc, bm, bc - bm))
        verdict = "supported" if dl > 0 else "reversed" if dh < 0 else "not supported"
        tally[verdict] += 1
        label = c + (" (VinDr Nodule/Mass label)" if ds == "VinDr" and c in ("nodule", "mass") else "")
        new_rows.append(f"| {ds} | {label} | {pos} / {len(y)} | {ac:.3f} [{cl:.3f}-{chh:.3f}] | {am:.3f} [{ml:.3f}-{mh:.3f}] | "
                        f"{ac - am:+.3f} [{dl:+.3f} to {dh:+.3f}] | {verdict}{' (fewer than 10 positives)' if pos < 10 else ''} |")
    a1_tally = {k: sum(f"| {k}" in l for l in a1_rows) for k in tally}
    a1_tally["supported"] = sum("| supported" in l for l in a1_rows)
    lines = ["# CheXagent vs MAIRA-2 presence scores, all 13 conditions", "",
             "Amendments A1 (six findings) and A2 (seven further conditions) of `maira_score_design.md`, both committed "
             "before their readings. AUC: how often a model scores an image with the condition above one without it "
             f"(0.5 = chance). Paired 95% CIs from {N_BOOT} image-level bootstrap resamples. Hypothesis: CheXagent's AUC is "
             "higher - supported if the difference's CI lies above 0.", "",
             "Caveats: CheXagent's training data probably include public NIH and VinDr images; MAIRA-2's do not. VinDr "
             "labels nodule and mass together, so both are scored against that one label. NIH labels are NLP-mined.", "",
             "| dataset | condition | positives / n | CheXagent AUC [95% CI] | MAIRA-2 AUC [95% CI] | "
             "CheXagent minus MAIRA-2 [95% CI] | hypothesis |", "|---|---|---|---|---|---|---|",
             "| **Six findings (A1)** | | | | | | |"] + a1_rows + ["| **Seven further conditions (A2)** | | | | | | |"] + new_rows
    total = {k: a1_tally[k] + tally[k] for k in tally}
    lines += ["", f"Six findings (A1): {len(a1_rows)} comparisons, supported {a1_tally['supported']}.",
              f"Seven further conditions (A2): {len(new_rows)} comparisons, supported {tally['supported']}, not supported "
              f"{tally['not supported']}, reversed {tally['reversed']}.",
              f"All conditions: {len(a1_rows) + len(new_rows)} comparisons, supported {total['supported']}."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"score": score, "analyze": analyze}[sys.argv[1]]()
