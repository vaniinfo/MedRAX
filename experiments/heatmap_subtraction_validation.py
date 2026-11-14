"""Validation of the two pilots, following experiments/heatmap_subtraction_validation_protocol.md (criteria were set
before running). Parameters are frozen from the pilots; nothing is tuned. NIH/VinDr only -> experiments/results/.

    .venv/Scripts/python.exe experiments/heatmap_subtraction_validation.py [biovil|subtraction|all]
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import condition_grounding_models as CG  # noqa: E402
import subtraction_pilot as SP  # noqa: E402
from anatomical_measurements import auc, auc_ci, segment_all  # noqa: E402

RESULTS = HERE / "results"
NIH, VINDR = CG.NIH, CG.VINDR
OPAC = {"Effusion", "Atelectasis", "Consolidation", "Pneumonia", "Infiltration"}
MIN_N = 15


def lung_mask(g):
    im = Image.fromarray(g)
    im.thumbnail((1024, 1024))
    m = segment_all(np.asarray(im, np.float32))
    lungs = (m["Left Lung"] | m["Right Lung"]).astype(np.uint8)
    return cv2.resize(lungs, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def biovil():
    import pandas as pd
    import torch
    from biovil_t_heatmap_pilot import load_engine
    eng = load_engine("cuda" if torch.cuda.is_available() else "cpu")
    jl = lambda f: [json.loads(x) for x in open(RESULTS / f, encoding="utf-8")]  # noqa: E731
    sets = {}
    vc = {(r["image_id"], r["finding"]): r["boxes"] for r in jl("vindr_grounding_chexagent.jsonl")
          if r["condition"] == "blind"}
    vm = {(r["image_id"], r["finding"]): r["boxes"] for r in jl("vindr_grounding_maira2.jsonl")
          if r["condition"] == "blind"}
    for c in json.loads((RESULTS / "vindr_grounding_cases.json").read_text(encoding="utf-8"))["cases"]:
        k = (c["image_id"], c["finding"])
        sets.setdefault(f"VinDr {c['finding']}", []).append(dict(img=c["png"], prompt=c["finding"],
                                                                 truth=c["truth_boxes"], chex=vc.get(k),
                                                                 maira=vm.get(k)))
    b = pd.read_csv(NIH / "BBox_List_2017.csv")
    b.columns = ["img", "lab", "x", "y", "w", "h"] + list(b.columns[6:])
    gc = {(r["file"], r["label"]): r["boxes"] for r in jl("grounding_chexagent.jsonl")}
    gm = {(r["file"], r["label"]): r["boxes"] for r in jl("grounding_maira2.jsonl")}
    for lab, prompt in (("Effusion", "pleural effusion"), ("Atelectasis", "atelectasis"),
                        ("Pneumothorax", "pneumothorax"), ("Pneumonia", "pneumonia")):
        for img, g in b[b.lab == lab].groupby("img"):
            if not (NIH / "images" / img).exists():
                continue
            truth = [[r.x / 10.24, r.y / 10.24, (r.x + r.w) / 10.24, (r.y + r.h) / 10.24] for r in g.itertuples()]
            sets.setdefault(f"NIH {lab.lower()}", []).append(dict(img=str(NIH / "images" / img), prompt=prompt,
                                                                  truth=truth, chex=gc.get((img, lab)),
                                                                  maira=gm.get((img, lab))))
    out = {}
    for name, cases in sets.items():
        rows = []
        for c in cases:
            sim = eng.get_similarity_map_from_raw_data(Path(c["img"]), c["prompt"], interpolation="bilinear")
            g = np.array(Image.open(c["img"]).convert("L"))
            if g.shape != sim.shape:
                continue
            truth = CG.box_mask(c["truth"], g.shape)
            v = np.nan_to_num(sim, nan=-np.inf)
            py, px = np.unravel_index(np.argmax(v), v.shape)
            fin = np.isfinite(sim)
            top = fin & (sim >= np.quantile(sim[fin], 0.9))
            lungs = lung_mask(g)
            r = {"hit": bool(truth[py, px]), "side_ok": CG.side(top) == CG.side(truth),
                 "baseline": float((lungs & truth).sum() / max(1, lungs.sum())),
                 "truth_in_crop": float((truth & fin).sum() / max(1, truth.sum()))}
            for mname, boxes in (("CheXagent", c["chex"]), ("MAIRA-2", c["maira"])):
                if boxes is None:
                    r[mname] = None          # not run for this image
                else:
                    mm = CG.metrics(CG.box_mask(boxes, g.shape), truth, CG.box_point(boxes, g.shape))
                    r[mname] = {"hit": mm["hit"], "side_ok": mm["side_ok"], "empty": mm["empty"]}
            rows.append(r)
        hit = CG.boot_mean([float(x["hit"]) for x in rows])
        base = float(np.mean([x["baseline"] for x in rows]))
        res = {"n": len(rows), "hit_rate": hit, "lung_baseline": base,
               "side_agreement": CG.boot_mean([float(x["side_ok"]) for x in rows]),
               "truth_inside_crop_mean": float(np.mean([x["truth_in_crop"] for x in rows])),
               "criterion_met": bool(len(rows) >= MIN_N and hit[1] > base),
               "descriptive_only": len(rows) < MIN_N}
        for mname in ("CheXagent", "MAIRA-2"):
            pr = [(float(x["hit"]), float(x[mname]["hit"])) for x in rows if x[mname] is not None]
            res[mname] = {"n": len(pr), "hit_rate": CG.boot_mean([y for _, y in pr]),
                          "empty_rate": float(np.mean([x[mname]["empty"] for x in rows if x[mname] is not None]))
                          if pr else None,
                          "biovil_minus_model_hit": CG.boot_diff([a for a, _ in pr], [y for _, y in pr])}
        out[name] = res
        print(f"{name:26s} n={len(rows):3d} hit {hit[0]:.2f} [{hit[1]:.2f},{hit[2]:.2f}] baseline {base:.2f} "
              f"met={res['criterion_met']} | CheX {res['CheXagent']['hit_rate'][0]} MAIRA {res['MAIRA-2']['hit_rate'][0]}",
              flush=True)
    return out


def subtraction():
    import pandas as pd
    from skimage.exposure import match_histograms
    d = pd.read_csv(NIH / "Data_Entry_2017_v2020.csv").set_index("Image Index")
    labels = lambda n: set(d.loc[n, "Finding Labels"].split("|")) - {"No Finding"}  # noqa: E731
    rows = []
    for p in json.loads((RESULTS / "edv_long_pairs.json").read_text(encoding="utf-8")):
        if not p["same_view"]:
            continue
        a, b = labels(p["prior"]), labels(p["current"])
        new, res_ = (b - a) & OPAC, (a - b) & OPAC
        grp = ("NEW" if new and not res_ else "RESOLVED" if res_ and not new else "STABLE" if a == b else
               "PTX_NEW" if "Pneumothorax" in b - a else "OTHER")
        if "Pneumothorax" in b - a:
            grp = "PTX_NEW"
        if grp == "OTHER":
            continue
        cur, prior = SP.square(NIH / "images" / p["current"]), SP.square(NIH / "images" / p["prior"])
        lc, lcr, lcl = SP.lungs(cur)
        lp, _, _ = SP.lungs(prior)
        if lc.sum() < 1000 or lp.sum() < 1000 or not lcr.any() or not lcl.any():
            rows.append(dict(pair=p, group=grp, failed="lung segmentation"))
            continue
        prior_m = match_histograms(prior, cur).astype(np.float32)
        W, method = SP.register(prior_m, cur, lp, lc)
        warped = cv2.warpAffine(prior_m, W, (SP.N, SP.N), flags=cv2.INTER_LINEAR)
        lpw = cv2.warpAffine(lp.astype(np.uint8), W, (SP.N, SP.N), flags=cv2.INTER_NEAREST) > 0
        zc, zp = np.zeros_like(cur), np.zeros_like(cur)
        for m in (lcr, lcl):
            zc[m] = (cur[m] - cur[m].mean()) / (cur[m].std() + 1e-6)
            zp[m] = (warped[m] - warped[m].mean()) / (warped[m].std() + 1e-6)
        diff = cv2.GaussianBlur(zc - zp, (0, 0), SP.SIGMA)
        dens = max(float((diff[m] > SP.THR).mean()) for m in (lcr, lcl))
        luc = max(float((diff[m] < -SP.THR).mean()) for m in (lcr, lcl))
        rows.append(dict(pair=p, group=grp, new=sorted(new), resolved=sorted(res_), denser=dens, lucent=luc,
                         dice=SP.dice(lpw, lc), method=method))
        if len(rows) % 50 == 0:
            print(len(rows), flush=True)
    ok = [r for r in rows if "failed" not in r]
    out = {"n_pairs": len(rows), "failed": len(rows) - len(ok),
           "groups": {g: sum(r["group"] == g for r in rows) for g in ("NEW", "RESOLVED", "STABLE", "PTX_NEW")},
           "registration_ok_rate": float(np.mean([r["dice"] >= SP.DICE_MIN for r in ok]))}
    for tag, sub in (("all", ok), ("registration_ok", [r for r in ok if r["dice"] >= SP.DICE_MIN])):
        for name, g, score in (("B1 NEW vs STABLE (denser)", "NEW", "denser"),
                               ("B2 RESOLVED vs STABLE (lucent)", "RESOLVED", "lucent")):
            pos = [r[score] for r in sub if r["group"] == g]
            neg = [r[score] for r in sub if r["group"] == "STABLE"]
            a = auc(pos, neg) if pos and neg else None
            ci = auc_ci(pos, neg) if pos and neg else None
            out[f"{name} [{tag}]"] = {"n_pos": len(pos), "n_neg": len(neg), "auc": a, "ci": ci,
                                      "median_pos": float(np.median(pos)) if pos else None,
                                      "median_neg": float(np.median(neg)) if neg else None,
                                      "criterion_met": bool(ci and ci[0] > 0.6)}
            print(f"{name} [{tag}] pos {len(pos)} neg {len(neg)} AUC {a} CI {ci}", flush=True)
    ptx = [r["lucent"] for r in ok if r["group"] == "PTX_NEW"]
    out["pneumothorax_new_descriptive"] = {"n": len(ptx), "lucent": ptx,
                                           "stable_lucent_median": float(np.median(
                                               [r["lucent"] for r in ok if r["group"] == "STABLE"]))}
    out["rows"] = rows
    return out


def main(which):
    st = time.strftime("%Y%m%d-%H%M%S")
    out = {"protocol": "experiments/heatmap_subtraction_validation_protocol.md", "stamp": st}
    if which in ("biovil", "all"):
        out["biovil_t"] = biovil()
    if which in ("subtraction", "all"):
        out["subtraction"] = subtraction()
    p = RESULTS / f"heatmap_subtraction_validation_{which}_{st}.json"
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
