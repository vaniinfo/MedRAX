"""EXPLORATORY: is CheXagent-DenseNet disagreement a signal that CheXagent is wrong?

Development splits only (Indiana dev, NIH dev: random + supplement); test splits, the frozen
Evidence Engine bundle and VinDr are not touched or changed.

Second model: TorchXRayVision densenet121-res224-chex, trained on CheXpert only. The repo's
default "all" weights were trained on NIH and Open-i (Indiana), so they would be
contaminated on these data.

For each (image, finding):
  - CheXagent: p_present from the existing readings (ppresent_*.jsonl); its call is
    p_present >= the frozen development threshold of the dataset (Youden, frozen params);
  - DenseNet: its output for the finding;
  - disagreement d = |percentile rank of CheXagent - percentile rank of DenseNet| within the
    dataset's development set and finding (0 = same rank, 1 = opposite ends).
Questions: does d predict that CheXagent's call is wrong (AUROC with 95% bootstrap CI)?
How often is CheXagent wrong when the models agree vs disagree? And for the Evidence Engine
states (NIH primary engine on NIH, Indiana comparator on Indiana): how often are SUPPORTING
and REFUTING wrong when the models agree vs disagree?

    python experiments/disagreement_signal.py score     # DenseNet scores (cached)
    python experiments/disagreement_signal.py analyze   # writes experiments/results/disagreement_dev.md
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "experiments" / "results"
SCORES = RESULTS / "densenet_chex_dev_scores.jsonl"
OUT = RESULTS / "disagreement_dev.md"
INDIANA_DIR = Path("C:/Users/hari/_workarea_/MedRAX/data/indiana_eval")
NIH_DIR = Path("D:/datasets/chestxray14/images")
FINDINGS = ["cardiomegaly", "pleural effusion", "pneumothorax", "consolidation", "pulmonary edema", "atelectasis"]
XRV_NAMES = {"cardiomegaly": "Cardiomegaly", "pleural effusion": "Effusion", "pneumothorax": "Pneumothorax",
             "consolidation": "Consolidation", "pulmonary edema": "Edema", "atelectasis": "Atelectasis"}
SEED = 20261003


def dev_readings():
    """{(dataset, file): {finding: (truth, p_present)}} for the development splits."""
    out = {}
    ind_split = json.loads((RESULTS / "split_indiana_70_30.json").read_text(encoding="utf-8"))
    for line in (RESULTS / "ppresent_indiana.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if ind_split.get(r["file"]) == "dev" and r["p_present"] is not None:
            out.setdefault(("indiana", r["file"]), {})[r["finding"]] = (r["truth"], r["p_present"])
    nih_split = json.loads((RESULTS / "split_nih_70_30.json").read_text(encoding="utf-8"))
    for line in (RESULTS / "ppresent_nih.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if nih_split[r["set"]].get(r["file"]) == "dev" and r["p_present"] is not None:
            out.setdefault(("nih", r["file"]), {})[r["finding"]] = (r["truth"], r["p_present"])
    return out


def score():
    import skimage.io
    import torch
    import torchvision
    import torchxrayvision as xrv

    done = set()
    if SCORES.exists():
        done = {(r["dataset"], r["file"]) for r in map(json.loads, SCORES.read_text(encoding="utf-8").splitlines())}
    todo = [k for k in dev_readings() if k not in done]
    print(f"{len(done)} scored, {len(todo)} to go", flush=True)
    model = xrv.models.DenseNet(weights="densenet121-res224-chex").cuda().eval()
    index = {f: model.pathologies.index(XRV_NAMES[f]) for f in FINDINGS}
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

    def load(key):
        dataset, file = key
        img = skimage.io.imread(str((INDIANA_DIR if dataset == "indiana" else NIH_DIR) / file))
        if img.ndim == 3:
            img = img[..., :3].mean(2)
        img = xrv.datasets.normalize(img, 255)
        return transform(img[None])

    with SCORES.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), 32):
            batch = todo[start:start + 32]
            x = torch.from_numpy(np.stack([load(k) for k in batch])).float().cuda()
            with torch.inference_mode():
                y = model(x).cpu().numpy()
            for (dataset, file), row in zip(batch, y):
                out.write(json.dumps({"dataset": dataset, "file": file,
                                      **{f: round(float(row[index[f]]), 5) for f in FINDINGS}}) + "\n")
            print(f"{start + len(batch)}/{len(todo)}", flush=True)


def auroc(score, label):
    score, label = np.asarray(score), np.asarray(label)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_ci(fn, *arrays, n=2000):
    rng = np.random.default_rng(SEED)
    k = len(arrays[0])
    vals = [fn(*(a[idx] for a in arrays)) for idx in (rng.integers(0, k, k) for _ in range(n))]
    vals = [v for v in vals if not np.isnan(v)]
    return np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))


def pct_rank(x):
    return (np.argsort(np.argsort(x)) + 0.5) / len(x)


def analyze():
    from medrax.evidence import EvidenceEngine

    readings = dev_readings()
    dense = {(r["dataset"], r["file"]): r for r in map(json.loads, SCORES.read_text(encoding="utf-8").splitlines())}
    params = {"indiana": json.loads((RESULTS / "frozen_dev_params.json").read_text(encoding="utf-8"))["findings"],
              "nih": json.loads((RESULTS / "nih_frozen_dev_params.json").read_text(encoding="utf-8"))["findings"]}
    engines = {"indiana": EvidenceEngine(engine="comparator"), "nih": EvidenceEngine(engine="primary")}

    lines = ["# EXPLORATORY: CheXagent-DenseNet disagreement as an error signal (development splits)", "",
             "DenseNet: TorchXRayVision densenet121-res224-chex (CheXpert only). CheXagent call: "
             "p_present >= frozen development threshold. Disagreement d = |percentile-rank difference| "
             "within dataset and finding. 'Disagree' = top quartile of d. 95% CIs: 2,000-sample "
             "image-level bootstrap. These thresholds were fitted on these same development data.", ""]
    pooled = {"d": [], "err": [], "state": [], "truth": []}
    for dataset in ("indiana", "nih"):
        lines += [f"## {dataset.upper()} development", "",
                  "| finding | n | positives | DenseNet AUC | CheXagent AUC | Spearman | CheXagent errors | "
                  "AUROC of d for CheXagent error (95% CI) | error rate: agree / disagree |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for f in FINDINGS:
            keys = [k for k in readings if k[0] == dataset and f in readings[k] and k in dense]
            truth = np.array([readings[k][f][0] for k in keys])
            p = np.array([readings[k][f][1] for k in keys])
            q = np.array([dense[k][f] for k in keys])
            d = np.abs(pct_rank(p) - pct_rank(q))
            call = (p >= params[dataset][f]["threshold"]).astype(int)
            err = (call != truth).astype(int)
            lo, hi = boot_ci(auroc, d, err)
            top = d >= np.quantile(d, 0.75)
            rho = np.corrcoef(pct_rank(p), pct_rank(q))[0, 1]
            lines.append(f"| {f} | {len(keys)} | {truth.sum()} | {auroc(q, truth):.3f} | {auroc(p, truth):.3f} | "
                         f"{rho:.2f} | {err.sum()} | {auroc(d, err):.3f} ({lo:.3f}-{hi:.3f}) | "
                         f"{err[~top].mean():.1%} / {err[top].mean():.1%} |")
            states = [engines[dataset].assess(f, float(x)).state for x in p]
            pooled["d"] += list(d >= np.quantile(d, 0.75))
            pooled["err"] += list(err)
            pooled["state"] += states
            pooled["truth"] += list(truth)
        lines.append("")
        # Directional second opinion: does DenseNet's score separate CheXagent's wrong calls
        # from its right ones, within its positive calls and within its negative calls?
        lines += [f"### {dataset.upper()}: directional (added after the first table; exploratory)", "",
                  "Among CheXagent's positive calls: AUROC of a LOW DenseNet score for a false positive. "
                  "Among its negative calls: AUROC of a HIGH DenseNet score for a missed finding.", "",
                  "| finding | positive calls (false) | AUROC for false positive (95% CI) | "
                  "negative calls (missed) | AUROC for missed finding (95% CI) |", "|---|---|---|---|---|"]
        for f in FINDINGS:
            keys = [k for k in readings if k[0] == dataset and f in readings[k] and k in dense]
            truth = np.array([readings[k][f][0] for k in keys])
            p = np.array([readings[k][f][1] for k in keys])
            q = np.array([dense[k][f] for k in keys])
            call = p >= params[dataset][f]["threshold"]
            cells = []
            for mask, signal, wrong in ((call, -q, truth == 0), (~call, q, truth == 1)):
                s, w = signal[mask], wrong[mask].astype(int)
                lo, hi = boot_ci(auroc, s, w)
                cells += [f"{mask.sum()} ({w.sum()})", f"{auroc(s, w):.3f} ({lo:.3f}-{hi:.3f})"]
            lines.append(f"| {f} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Evidence Engine states, both datasets pooled", "",
              "How often a state is wrong, when the two models agree (bottom 3 quartiles of d) vs disagree (top quartile).", "",
              "| state | agree: n, wrong | disagree: n, wrong |", "|---|---|---|"]
    top, state, truth = (np.array(pooled[k]) for k in ("d", "state", "truth"))
    for s, wrong_if in (("SUPPORTING", 0), ("REFUTING", 1)):
        cells = []
        for mask in (~top, top):
            m = mask & (state == s)
            cells.append(f"{m.sum()}, {(truth[m] == wrong_if).mean():.1%}" if m.sum() else "0, -")
        lines.append(f"| {s} | {cells[0]} | {cells[1]} |")
    m = state == "UNCERTAIN"
    lines.append(f"| UNCERTAIN (share in group) | {(m & ~top).sum() / (~top).sum():.1%} | {(m & top).sum() / top.sum():.1%} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    {"score": score, "analyze": analyze}[sys.argv[1]]()
