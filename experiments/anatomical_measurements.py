"""FEASIBILITY: quantitative anatomical measurements for the radiologist evaluation report (presentation only).

Goal: better information for the radiologist reader. Nothing here changes a model, a prediction, a CheXagent
verdict, the Evidence Engine, a box, a report or any radiologist score. New time-stamped files only; MIMIC-CXR
content goes to logs/ (git-ignored); experiments/results/ receives NIH (public) numbers only.

Placement rule: every measurement is computed for EVERY image (prior and current of all 19 comparisons) from the
anatomy alone. Neither MedGemma's findings nor the reference report decide where segmentation or measurements go.
The finding name only decides which already-computed measurement is shown next to that finding in the report.

Measurements (torchxrayvision PSPNet, ChestX-Det anatomy: lungs, heart, hila, trachea, spine)
  Phase 1 (global)
    cardiomegaly      cardiothoracic ratio (CTR) = (MRD + MLD) / widest outer-lung thoracic width at or above the
                      diaphragm domes, midline from the spine; lines drawn. Projection (AP/PA) and rotation flagged; conventional 0.50 reference
                      mentioned only for PA films that pass every check.
    low lung volumes  projected lung area / thoracic width^2, lung height (apex to diaphragm dome) / thoracic width,
                      hemidiaphragm dome levels; prior vs current only when technically comparable.
    hyperinflation    the same plus diaphragm dome height ratio (dome height above the costophrenic-cardiophrenic
                      chord / chord length: flatter diaphragm = smaller). No diagnostic thresholds.
  Phase 2 (regional, exploratory)
    pleural effusion  costophrenic angle (degrees) and costophrenic recess depth per side.
    atelectasis       lung area share per side, hilum height per side, trachea and heart shift from the midline.
    consolidation     no anatomical measure; candidate opacity location from a pathology classifier's saliency
                      (Grad-CAM, torchxrayvision DenseNet 'all'), labelled model-generated, not a boundary.
  Lung segmentation alone is never used to draw a disease boundary.

Quality control (per image; a measurement is NOT displayed when its landmarks fail)
  lungs: both present (>= 2% of the image each), main component >= 85% of the mask, right lung on the image left,
         apex and base not cut by the image edge, area ratio within 1:3.
  heart: present (>= 0.5%), main component >= 85%, centre between the two lungs, borders inside the thoracic width.
  midline: spine visible over the thorax; rotation index = (thoracic mid-point - spine midline) / thoracic width;
           'rotated' when beyond the 95th percentile of NIH PA No-Finding images (set in the NIH stage).
  diaphragm/costophrenic: enough lung-base columns to fit both lines.
  prior vs current: compared only if both pass QC, neither is rotated and the projection is known and identical.

NIH feasibility check (PRE-STATED before any run; public NIH ChestX-ray14 images; patients used anywhere else in
experiments/results are excluded; seed 20261108)
  Samples per view (PA, AP): Cardiomegaly, Effusion, Atelectasis, Emphysema (hyperinflation proxy) 150 each,
  No Finding 250. Each measurement is compared positive vs No Finding (AUC, bootstrap 95% CI).
  A measurement is called FEASIBLE FOR DISPLAY when QC lets it be computed in >= 80% of images AND the AUC's lower
  95% bound is > 0.60 in that view. Otherwise it is 'not supported by this check' and is not shown in the report.
  Low lung volumes has no NIH label: its measurements are shown as descriptive only, with that stated.
  Saliency (phase 2): pointing game on NIH boxes (Effusion, Atelectasis; Pneumonia and Infiltrate as consolidation
  proxies): hit = saliency peak inside the radiologist box; baseline = the share of the segmented lung area inside
  the box (hit rate of a random lung point). FEASIBLE when the hit rate's lower 95% bound exceeds the baseline mean.
  Caveat: the DenseNet was trained on data that includes NIH, so this check is optimistic. (Amended after the first
  run, which found 0 box images: all NIH box patients are project patients; the box check includes them because it
  tunes nothing.)

    .venv/Scripts/python.exe experiments/anatomical_measurements.py nih      # feasibility check (NIH)
    .venv/Scripts/python.exe experiments/anatomical_measurements.py mimic    # 19 comparisons: overlays + report
"""

import json
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
LOGS = ROOT / "logs"
RESULTS = HERE / "results"
NIH_DIR = Path("D:/datasets/chestxray14")
SEED = 20261108
TARGETS = ['Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula', 'Left Lung', 'Right Lung',
           'Left Hilus Pulmonis', 'Right Hilus Pulmonis', 'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum',
           'Weasand', 'Spine']
ANAT = "Anatomical measurement (segmentation-derived) - not a diagnosis"
MODEL = "Model-generated candidate location - not confirmed disease extent"
SAL = "Classifier saliency (model-generated) - candidate location, not a disease boundary"
CAM_CLASSES = ("Effusion", "Atelectasis", "Consolidation")

# ------------------------------------------------------------------ models
_SEG = _CLS = None


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def segment_all(img):
    """{target: bool mask} at the image's own size; zero-padded to a square (no crop), 512 for the model."""
    global _SEG
    import torch
    import torchxrayvision as xrv
    if _SEG is None:
        _SEG = xrv.baseline_models.chestx_det.PSPNet().eval().to(device())
    img = np.asarray(img, np.float32)
    h, w = img.shape
    s = max(h, w)
    pad = np.zeros((s, s), np.float32)
    top, left = (s - h) // 2, (s - w) // 2
    pad[top:top + h, left:left + w] = img
    x = xrv.datasets.XRayResizer(512, engine="cv2")(xrv.datasets.normalize(pad, 255)[None])
    with torch.no_grad():
        out = torch.sigmoid(_SEG(torch.from_numpy(x)[None].to(device())))[0].cpu().numpy()
    masks = {}
    for i, name in enumerate(TARGETS):
        m = np.array(Image.fromarray((out[i] > 0.5).astype(np.uint8) * 255).resize((s, s), Image.NEAREST)) > 0
        masks[name] = m[top:top + h, left:left + w]
    return masks


def gradcam(img, classes=CAM_CLASSES):
    """{class: (cam at image size, 0-1)} from torchxrayvision DenseNet 'all' (padded to square, 224)."""
    global _CLS
    import torch
    import torch.nn.functional as F
    import torchxrayvision as xrv
    if _CLS is None:
        _CLS = xrv.models.DenseNet(weights="densenet121-res224-all").eval().to(device())
        _CLS.op_threshs = None
    img = np.asarray(img, np.float32)
    h, w = img.shape
    s = max(h, w)
    pad = np.zeros((s, s), np.float32)
    top, left = (s - h) // 2, (s - w) // 2
    pad[top:top + h, left:left + w] = img
    x = torch.from_numpy(xrv.datasets.XRayResizer(224, engine="cv2")(xrv.datasets.normalize(pad, 255)[None]))[None]
    x = x.to(device()).requires_grad_(False)
    out = {}
    for c in classes:
        feats = _CLS.features(x)
        feats.retain_grad()
        logit = _CLS.classifier(F.adaptive_avg_pool2d(F.relu(feats), 1).flatten(1))[0, _CLS.pathologies.index(c)]
        _CLS.zero_grad()
        logit.backward()
        wts = feats.grad.mean((2, 3), keepdim=True)
        cam = F.relu((wts * feats).sum(1))[0].detach().cpu().numpy()
        cam = np.array(Image.fromarray(cam.astype(np.float32)).resize((s, s), Image.BILINEAR))[top:top + h, left:left + w]
        out[c] = cam / cam.max() if cam.max() > 0 else cam
    return out


# ------------------------------------------------------------------ geometry
def largest(m):
    from scipy.ndimage import label
    lab, n = label(m)
    if n == 0:
        return m, 0.0
    sizes = np.bincount(lab.ravel())[1:]
    k = sizes.argmax() + 1
    return lab == k, float(sizes.max() / sizes.sum())


def base_contour(m):
    """{x: lowest lung row} for each column of the mask."""
    ys, xs = np.nonzero(m)
    out = {}
    for x, y in zip(xs, ys):
        if y > out.get(x, -1):
            out[x] = y
    return out


def measure(masks, rot_limit=None):
    """All measurements + QC + drawing geometry for one image. Values are scale-free (divided by thoracic width)."""
    h, w = masks["Heart"].shape
    R = {"qc": {}, "geo": {}, "val": {}, "why": {}}
    qc, geo, val, why = R["qc"], R["geo"], R["val"], R["why"]
    lungs = {}
    lung_issues = []
    for side, name in (("R", "Right Lung"), ("L", "Left Lung")):
        m, frac = largest(masks[name])
        lungs[side] = m
        if m.sum() < 0.02 * h * w:
            lung_issues.append(f"{name.lower()} not found")
        elif frac < 0.85:
            lung_issues.append(f"{name.lower()} fragmented")
        else:
            ys = np.nonzero(m)[0]
            if ys.min() <= 1 or ys.max() >= h - 2:
                lung_issues.append(f"{name.lower()} cut by the image edge")
    if not lung_issues:
        cxR, cxL = np.nonzero(lungs["R"])[1].mean(), np.nonzero(lungs["L"])[1].mean()
        if cxR >= cxL:
            lung_issues.append("lungs in unexpected order (image flipped?)")
        ar = lungs["R"].sum() / lungs["L"].sum()
        if not 1 / 3 <= ar <= 3:
            lung_issues.append(f"lung area ratio {ar:.2f} outside 1:3 (segmentation failure likely)")
    qc["lungs"] = not lung_issues
    why["lungs"] = "; ".join(lung_issues)
    if not qc["lungs"]:
        for k in ("heart", "midline", "base_R", "base_L"):
            qc[k], why[k] = False, "lungs failed QC"
        R["qc"] = {k: bool(v) for k, v in qc.items()}
        return R
    # thoracic width: widest row with both lungs (outer lung edges)
    def dome_row(m):
        bc = base_contour(m)
        cols = np.array(sorted(bc))
        wd = cols.max() - cols.min()
        cen = cols[(cols > cols.min() + 0.2 * wd) & (cols < cols.max() - 0.2 * wd)]
        return min(bc[x] for x in cen) if len(cen) else h
    ylim = max(dome_row(lungs["R"]), dome_row(lungs["L"]))   # widest width at or above the lower dome level
    best = (0, None, None, None)
    for y in range(min(h, ylim + 1)):
        xr, xl = np.nonzero(lungs["R"][y])[0], np.nonzero(lungs["L"][y])[0]
        if xr.size and xl.size and xl.max() - xr.min() > best[0]:
            best = (xl.max() - xr.min(), y, xr.min(), xl.max())
    tw, ty, tx0, tx1 = best
    geo["thorax"] = (int(tx0), int(tx1), int(ty))
    val["thoracic_width_frac_image"] = tw / w
    # midline from the spine within the lung rows
    ylo = min(np.nonzero(lungs["R"])[0].min(), np.nonzero(lungs["L"])[0].min())
    yhi = max(np.nonzero(lungs["R"])[0].max(), np.nonzero(lungs["L"])[0].max())
    sp = masks["Spine"].copy()
    sp[:ylo] = False
    sp[yhi:] = False
    if sp.sum() > 0.002 * h * w and np.ptp(np.nonzero(sp)[0]) > 0.3 * (yhi - ylo):
        mid = float(np.median(np.nonzero(sp)[1]))
        rot = ((tx0 + tx1) / 2 - mid) / tw
        val["rotation_index"] = rot
        qc["midline"] = rot_limit is None or abs(rot) <= rot_limit
        why["midline"] = "" if qc["midline"] else f"rotated (rotation index {rot:+.3f}, limit {rot_limit:.3f})"
    else:
        mid = (tx0 + tx1) / 2
        qc["midline"], why["midline"] = False, "spine not identified over the thorax (rotation cannot be checked)"
    geo["mid"] = mid
    # heart and CTR
    hm, hfrac = largest(masks["Heart"])
    issues = []
    if hm.sum() < 0.005 * h * w:
        issues.append("heart not found")
    elif hfrac < 0.85:
        issues.append("heart fragmented")
    else:
        ys, xs = np.nonzero(hm)
        if not (np.nonzero(lungs["R"])[1].mean() < xs.mean() < np.nonzero(lungs["L"])[1].mean()):
            issues.append("heart not between the lungs")
        if xs.min() < tx0 or xs.max() > tx1:
            issues.append("heart border outside the thoracic width")
        if not (xs.min() < mid < xs.max()):
            issues.append("heart does not cross the midline")
    qc["heart"] = not issues
    why["heart"] = "; ".join(issues)
    if qc["heart"]:
        ys, xs = np.nonzero(hm)
        yr, yl = ys[xs == xs.min()].mean(), ys[xs == xs.max()].mean()
        mrd, mld = mid - xs.min(), xs.max() - mid
        val["ctr"] = (mrd + mld) / tw
        val["heart_shift"] = ((xs.min() + xs.max()) / 2 - mid) / tw
        geo["heart"] = (int(xs.min()), int(yr), int(xs.max()), int(yl))
        geo["heart_mask"] = hm
    # per-lung: apex, base contour, dome, costophrenic and cardiophrenic points, angle, area, hilum
    area = {}
    for side in ("R", "L"):
        m = lungs[side]
        ys, xs = np.nonzero(m)
        apex = ys.min()
        area[side] = m.sum()
        bc = base_contour(m)
        cols = np.array(sorted(bc))
        width = cols.max() - cols.min()
        lateral = cols[cols <= cols.min() + 0.25 * width] if side == "R" else cols[cols >= cols.max() - 0.25 * width]
        medial = cols[cols >= cols.max() - 0.25 * width] if side == "R" else cols[cols <= cols.min() + 0.25 * width]
        central = cols[(cols > cols.min() + 0.2 * width) & (cols < cols.max() - 0.2 * width)]
        if min(len(lateral), len(medial), len(central)) < 10:
            qc[f"base_{side}"], why[f"base_{side}"] = False, "lung base too narrow to measure"
            continue
        cpx = max(lateral, key=lambda x: bc[x])
        cdx = max(medial, key=lambda x: bc[x])
        dx = min(central, key=lambda x: bc[x])
        cp, cd, dome = np.array([cpx, bc[cpx]], float), np.array([cdx, bc[cdx]], float), np.array([dx, bc[dx]], float)
        chord = cd - cp
        clen = np.hypot(*chord)
        # signed perpendicular distance of the dome above the chord (positive = dome above the line)
        cross = (chord[0] * (dome[1] - cp[1]) - chord[1] * (dome[0] - cp[0])) / clen
        dome_h = -cross if side == "R" else cross
        # costophrenic angle: lateral wall (outermost x per row, lowest 25% of lung height above the CP point)
        # versus the lateral lung base
        hgt = bc[cpx] - apex
        rows = np.arange(int(bc[cpx] - 0.25 * hgt), int(bc[cpx] - 0.03 * hgt))
        wall = [(y, (np.nonzero(m[y])[0].min() if side == "R" else np.nonzero(m[y])[0].max())) for y in rows
                if m[y].any()]
        basepts = [(x, bc[x]) for x in lateral]
        ok = len(wall) >= 8 and len(basepts) >= 8 and dome[1] < cp[1]
        qc[f"base_{side}"] = ok
        why[f"base_{side}"] = "" if ok else "costophrenic region not identifiable"
        val[f"lung_height_{side}"] = (dome[1] - apex) / tw
        val[f"dome_ratio_{side}"] = dome_h / clen
        val[f"recess_depth_{side}"] = (cp[1] - dome[1]) / tw
        val[f"dome_y_{side}"] = dome[1]
        geo[f"lung_{side}"] = dict(apex=(int(xs[ys == apex].mean()), int(apex)), cp=cp.tolist(), cd=cd.tolist(),
                                   dome=dome.tolist())
        if ok:
            a, b = np.polyfit([p[0] for p in wall], [p[1] for p in wall], 1)   # x = a*y + b
            c, d = np.polyfit([p[0] for p in basepts], [p[1] for p in basepts], 1)  # y = c*x + d
            up = np.array([-a, -1.0]) / np.hypot(a, 1)
            med = (np.array([1.0, c]) if side == "R" else np.array([-1.0, -c])) / np.hypot(1, c)
            val[f"cp_angle_{side}"] = float(np.degrees(np.arccos(np.clip(up @ med, -1, 1))))
            geo[f"lung_{side}"]["wall"] = [(a * y + b, y) for y in (rows[0], rows[-1])]
            geo[f"lung_{side}"]["base"] = [(x, c * x + d) for x in (min(lateral), max(lateral))]
        hil, _ = largest(masks["Right Hilus Pulmonis" if side == "R" else "Left Hilus Pulmonis"])
        if hil.sum() > 0.0005 * h * w:
            hy, hx = np.nonzero(hil)
            val[f"hilum_level_{side}"] = (hy.mean() - apex) / max(1.0, dome[1] - apex)
            geo[f"hilum_{side}"] = (float(hx.mean()), float(hy.mean()))
    lung_union = lungs["R"] | lungs["L"]
    val["lung_area"] = lung_union.sum() / tw ** 2
    val["lung_share_R"] = area["R"] / (area["R"] + area["L"])
    if "lung_height_R" in val and "lung_height_L" in val:
        val["lung_height"] = (val["lung_height_R"] + val["lung_height_L"]) / 2
        val["dome_diff"] = (val["dome_y_L"] - val["dome_y_R"]) / tw   # + = right dome higher
    if "dome_ratio_R" in val and "dome_ratio_L" in val:
        val["dome_ratio"] = (val["dome_ratio_R"] + val["dome_ratio_L"]) / 2
    tr, _ = largest(masks["Weasand"])
    if tr.sum() > 0.0005 * h * w:
        ty_, tx_ = np.nonzero(tr)
        val["trachea_shift"] = (tx_.mean() - mid) / tw
        geo["trachea"] = (float(tx_.mean()), float(ty_.mean()))
    geo["lungs"] = lung_union
    R["qc"] = {k: bool(v) for k, v in qc.items()}
    return R


def zone_of(point, masks):
    import condition_aware_grounding as G
    regions = G.zone_regions(masks)
    x, y = int(point[0]), int(point[1])
    for (lung, third), r in regions.items():
        if r[y, x]:
            return f"{lung.split()[0].lower()} {third} zone"
    return None


# ------------------------------------------------------------------ statistics
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    from scipy.stats import rankdata
    r = rankdata(allv)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_ci(pos, neg, n=1000):
    rng = np.random.default_rng(SEED)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def prop_ci(k, n):
    from scipy.stats import beta
    return [float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


# Each NIH test: (condition label, measurement key, direction: +1 = larger in the condition, needs)
TESTS = [("Cardiomegaly", "ctr", +1, ("lungs", "heart", "midline")),
         ("Emphysema", "lung_height", +1, ("lungs", "base_R", "base_L")),
         ("Emphysema", "lung_area", +1, ("lungs",)),
         ("Emphysema", "dome_ratio", -1, ("lungs", "base_R", "base_L")),
         ("Effusion", "cp_angle_max", +1, ("lungs", "base_R", "base_L")),
         ("Effusion", "recess_depth_min", -1, ("lungs", "base_R", "base_L")),
         ("Atelectasis", "lung_share_dev", +1, ("lungs",)),
         ("Atelectasis", "hilum_asym", +1, ("lungs", "base_R", "base_L")),
         ("Atelectasis", "trachea_shift_abs", +1, ("lungs", "midline"))]


def derived(v):
    out = dict(v)
    if "cp_angle_R" in v and "cp_angle_L" in v:
        out["cp_angle_max"] = max(v["cp_angle_R"], v["cp_angle_L"])
    if "recess_depth_R" in v and "recess_depth_L" in v:
        out["recess_depth_min"] = min(v["recess_depth_R"], v["recess_depth_L"])
    if "lung_share_R" in v:
        out["lung_share_dev"] = abs(v["lung_share_R"] - 0.53)
    if "hilum_level_R" in v and "hilum_level_L" in v:
        out["hilum_asym"] = abs(v["hilum_level_R"] - v["hilum_level_L"])
    if "trachea_shift" in v:
        out["trachea_shift_abs"] = abs(v["trachea_shift"])
    return out


# ------------------------------------------------------------------ NIH stage
def nih():
    import pandas as pd
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"anatomical_measurements_nih_{stamp}.json"
    used = set()
    for f in RESULTS.glob("*.json*"):
        used |= set(re.findall(r"(\d{8})_\d{3}\.png", f.read_text(encoding="utf-8", errors="ignore")))
    d = pd.read_csv(NIH_DIR / "Data_Entry_2017_v2020.csv")
    d["pid"] = d["Image Index"].str[:8]
    d = d[~d.pid.isin(used)]
    d = d[d["Image Index"].map(lambda n: (NIH_DIR / "images" / n).exists())]
    print(f"{len(used)} project patients excluded; {len(d)} candidate images", flush=True)
    rng = np.random.default_rng(SEED)
    sample = []
    for view in ("PA", "AP"):
        dv = d[d["View Position"] == view]
        for lab, n in (("Cardiomegaly", 150), ("Effusion", 150), ("Atelectasis", 150), ("Emphysema", 150),
                       ("No Finding", 250)):
            pool = dv[dv["Finding Labels"] == "No Finding"] if lab == "No Finding" else \
                dv[dv["Finding Labels"].str.split("|").map(lambda s: lab in s)]
            pick = pool.sample(min(n, len(pool)), random_state=int(rng.integers(1e9)))
            sample += [(r["Image Index"], view, lab, r["Finding Labels"]) for _, r in pick.iterrows()]
    rows = []
    for k, (name, view, lab, labels) in enumerate(sample):
        img = Image.open(NIH_DIR / "images" / name).convert("L")
        img.thumbnail((1024, 1024))
        r = measure(segment_all(np.array(img)))
        rows.append({"image": name, "view": view, "group": lab, "labels": labels,
                     "qc": {q: bool(v) for q, v in r["qc"].items()}, "why": r["why"],
                     "val": {a: float(b) for a, b in r["val"].items()}})
        if k % 200 == 0:
            print(f"{k}/{len(sample)}", flush=True)
    # rotation limit: 95th percentile of |rotation index| in PA No Finding
    rot = [abs(r["val"]["rotation_index"]) for r in rows if r["view"] == "PA" and r["group"] == "No Finding"
           and "rotation_index" in r["val"]]
    rot_limit = float(np.percentile(rot, 95))
    for r in rows:
        if "rotation_index" in r["val"]:
            r["qc"]["midline"] = bool(abs(r["val"]["rotation_index"]) <= rot_limit)
    tests = []
    for view in ("PA", "AP"):
        for lab, key, sign, needs in TESTS:
            grp = [r for r in rows if r["view"] == view and r["group"] in (lab, "No Finding")]
            ok = [r for r in grp if all(r["qc"].get(q) for q in needs) and key in derived(r["val"])]
            pos = [sign * derived(r["val"])[key] for r in ok if r["group"] == lab]
            neg = [sign * derived(r["val"])[key] for r in ok if r["group"] == "No Finding"]
            a = auc(pos, neg)
            ci = auc_ci(pos, neg) if a is not None else None
            comp = len(ok) / max(1, len(grp))
            tests.append({"view": view, "condition": lab, "measure": key, "direction": sign, "n_pos": len(pos),
                          "n_neg": len(neg), "computable": comp, "auc": a, "ci": ci,
                          "median_pos": float(np.median(pos) * sign) if pos else None,
                          "median_neg": float(np.median(neg) * sign) if neg else None,
                          "feasible": bool(comp >= 0.8 and ci and ci[0] > 0.6)})
            print(f"{view} {lab:12s} {key:18s} computable {comp:.0%} AUC {a if a is None else round(a, 3)} {ci}",
                  flush=True)
    qc_rates = {}
    for view in ("PA", "AP"):
        vr = [r for r in rows if r["view"] == view]
        qc_rates[view] = {q: float(np.mean([bool(r["qc"].get(q)) for r in vr]))
                          for q in ("lungs", "heart", "midline", "base_R", "base_L")}
    fails = {}
    for r in rows:
        for q, msg in r["why"].items():
            if msg and msg != "lungs failed QC":
                fails[msg.split(" (")[0]] = fails.get(msg.split(" (")[0], 0) + 1
    point = pointing_game(d, used)
    json.dump({"rot_limit": rot_limit, "n_images": len(rows), "excluded_patients": len(used), "tests": tests,
               "qc_rates": qc_rates, "qc_failure_reasons": fails, "pointing": point, "rows": rows},
              out.open("w", encoding="utf-8"), indent=1)
    print(f"-> {out}")


def pointing_game(d, used):
    import pandas as pd
    import condition_aware_grounding as G
    b = pd.read_csv(NIH_DIR / "BBox_List_2017.csv")
    b = b.rename(columns={b.columns[2]: "x", b.columns[3]: "y", b.columns[4]: "w", b.columns[5]: "h"})
    # project patients are NOT excluded here: every NIH box image was already used for display checks, and
    # this check tunes nothing (it only decides whether saliency is shown)
    b = b[b["Image Index"].map(lambda n: (NIH_DIR / "images" / n).exists())]
    cls_for = {"Effusion": "Effusion", "Atelectasis": "Atelectasis", "Pneumonia": "Consolidation",
               "Infiltrate": "Consolidation"}
    res = {}
    for lab, cls in cls_for.items():
        hits, base = [], []
        for _, r in b[b["Finding Label"] == lab].iterrows():
            img = np.array(Image.open(NIH_DIR / "images" / r["Image Index"]).convert("L"))
            if img.shape != (1024, 1024):
                continue
            cam = gradcam(img, (cls,))[cls]
            yy, xx = np.unravel_index(cam.argmax(), cam.shape)
            inside = r["x"] <= xx <= r["x"] + r["w"] and r["y"] <= yy <= r["y"] + r["h"]
            m = segment_all(img)
            lungs = m["Left Lung"] | m["Right Lung"]
            box = np.zeros(img.shape, bool)
            box[int(r["y"]):int(r["y"] + r["h"]), int(r["x"]):int(r["x"] + r["w"])] = True
            hits.append(inside)
            base.append(float((lungs & box).sum() / max(1, lungs.sum())))
        k, n = int(sum(hits)), len(hits)
        ci = prop_ci(k, n)
        res[lab] = {"cam_class": cls, "n": n, "hit_rate": k / max(1, n), "ci": ci,
                    "baseline": float(np.mean(base)) if base else None,
                    "feasible": bool(n and ci[0] > np.mean(base))}
        print(f"pointing {lab}: {k}/{n} hit, baseline {res[lab]['baseline']}", flush=True)
    return res


# ------------------------------------------------------------------ MIMIC stage
def projections():
    """Subject-level AP/PA from the held-out CSV, only when every row of that subject agrees on both images."""
    import pandas as pd
    d = pd.read_csv("D:/datasets/held_out_eval_images/held_out_val_set.csv")
    out = {}
    for sid, g in d.groupby("subject_id"):
        views = set(g.view_position_before) | set(g.view_position_after)
        out[str(sid)] = views.pop() if len(views) == 1 else None
    return out


def comparable(rp, rc, proj):
    if not (rp["qc"]["lungs"] and rc["qc"]["lungs"]):
        return False, "segmentation failed QC on one image"
    if not (rp["qc"]["midline"] and rc["qc"]["midline"]):
        return False, "rotation not checkable or rotated on one image"
    if proj is None:
        return False, "projection not recorded (cannot confirm same projection)"
    return True, f"both {proj}, both pass QC"


def fmt(v, k, d=2):
    return "-" if v.get(k) is None else f"{v[k]:.{d}f}"


def show_lines(R, proj, nih_ok):
    """Text lines per measurement group; nothing shown when landmarks fail."""
    q, v, w = R["qc"], R["val"], R["why"]
    out = {}
    if q["lungs"] and q["heart"] and q["midline"]:
        rel = "" if proj == "PA" else (" | AP film: heart magnified, the PA reference (0.50) does not apply"
                                       if proj == "AP" else " | projection unknown: treat as possibly AP")
        out["ctr"] = [f"Cardiothoracic ratio {v['ctr']:.2f}" + (" (conventional PA reference 0.50)" if proj == "PA"
                                                                 else "") + rel]
    else:
        out["ctr"] = ["CTR not shown: " + "; ".join(x for x in (w.get("lungs"), w.get("heart"), w.get("midline")) if x)]
    if q["lungs"] and q.get("base_R") and q.get("base_L"):
        out["volume"] = [f"Lung height / thoracic width {v['lung_height']:.2f} (R {v['lung_height_R']:.2f}, "
                         f"L {v['lung_height_L']:.2f}); projected lung area / width^2 {v['lung_area']:.2f}",
                         f"Diaphragm domes: right {'higher' if v['dome_diff'] > 0 else 'lower'} by "
                         f"{abs(v['dome_diff']):.2f} x thoracic width"]
        out["dome"] = [f"Diaphragm dome height ratio R {v['dome_ratio_R']:.2f}, L {v['dome_ratio_L']:.2f} "
                       "(smaller = flatter)"]
        out["cp"] = [f"Costophrenic angle R {v['cp_angle_R']:.0f} deg, L {v['cp_angle_L']:.0f} deg; recess depth R "
                     f"{v['recess_depth_R']:.2f}, L {v['recess_depth_L']:.2f}"]
    else:
        msg = "; ".join(x for x in (w.get("lungs"), w.get("base_R"), w.get("base_L")) if x)
        out["volume"] = out["dome"] = out["cp"] = [f"Not shown: {msg}"]
    if q["lungs"]:
        s = [f"Lung area share R {v['lung_share_R']:.2f} / L {1 - v['lung_share_R']:.2f}"]
        if "hilum_level_R" in v and "hilum_level_L" in v:
            s.append(f"Hilum level (0 apex - 1 dome) R {v['hilum_level_R']:.2f}, L {v['hilum_level_L']:.2f}")
        if q["midline"] and "trachea_shift" in v:
            s.append(f"Trachea shift {v['trachea_shift']:+.2f}, heart shift {v.get('heart_shift', float('nan')):+.2f} "
                     "x width (+ = to patient's left)")
        out["atel"] = s
    else:
        out["atel"] = [f"Not shown: {w['lungs']}"]
    for k in list(out):
        if k in nih_ok and not nih_ok[k]:
            out[k] = [f"Not shown: did not pass the NIH feasibility check ({nih_ok.get(k + '_why', '')})"]
    return out


GROUP_FOR = [("cardiomegaly", "ctr"), ("low lung volumes", "volume"), ("hyperinflation", "dome"),
             ("effusion", "cp"), ("atelectasis", "atel"), ("consolidation", "sal"), ("pneumonia", "sal"),
             ("opacit", "sal")]


def group_for(finding):
    f = finding.lower()
    return next((g for k, g in GROUP_FOR if k in f), None)


def draw_anatomy(ax, arr, R, groups, title=""):
    """Anatomical overlays in blue/cyan only (model pathology localizations are drawn elsewhere, in red/orange)."""
    import condition_aware_grounding as G
    ax.imshow(arr, cmap="gray")
    ax.axis("off")
    ax.set_title(title, fontsize=5.8, loc="left")
    g, q = R["geo"], R["qc"]
    if not q["lungs"]:
        ax.text(0.5, 0.5, "segmentation failed QC\nno measurement shown", color="yellow", ha="center",
                transform=ax.transAxes, fontsize=7, weight="bold")
        return
    ax.imshow(np.ma.masked_where(~G.outline(g["lungs"]), np.ones_like(arr)), cmap="winter", alpha=0.9)
    tx0, tx1, ty = g["thorax"]
    if "ctr" in groups and q["heart"] and q["midline"]:
        ax.imshow(np.ma.masked_where(~G.outline(g["heart_mask"]), np.ones_like(arr)), cmap="spring", alpha=0.9)
        x0, yr, x1, yl = g["heart"]
        m = g["mid"]
        ax.axvline(m, color="yellow", lw=0.7, ls="--")
        ax.plot([x0, m], [yr, yr], color="yellow", lw=1.8, zorder=4)
        ax.plot([m, x1], [yl, yl], color="yellow", lw=1.8, zorder=4)
        ax.plot([tx0, tx1], [ty, ty], color="cyan", lw=1.0, zorder=3)
    if {"volume", "dome", "cp"} & set(groups):
        for s in ("R", "L"):
            L = g.get(f"lung_{s}")
            if not L or not q.get(f"base_{s}"):
                continue
            if "volume" in groups:
                ax.plot([L["apex"][0], L["apex"][0]], [L["apex"][1], L["dome"][1]], color="yellow", lw=1)
                ax.plot([tx0, tx1], [ty, ty], color="cyan", lw=0.8)
            if "dome" in groups or "volume" in groups:
                ax.plot([L["cp"][0], L["cd"][0]], [L["cp"][1], L["cd"][1]], color="orange" if False else "yellow",
                        lw=0.9, ls=":")
                ax.plot(*L["dome"], marker="o", ms=2.5, color="yellow")
            if "cp" in groups and "wall" in L:
                ax.plot([p[0] for p in L["wall"]], [p[1] for p in L["wall"]], color="yellow", lw=1)
                ax.plot([p[0] for p in L["base"]], [p[1] for p in L["base"]], color="yellow", lw=1)
    if "atel" in groups:
        ax.axvline(g["mid"], color="yellow", lw=0.7, ls="--")
        for s in ("R", "L"):
            if f"hilum_{s}" in g:
                ax.plot(*g[f"hilum_{s}"], marker="x", ms=4, color="yellow")
        if "trachea" in g:
            ax.plot(*g["trachea"], marker="^", ms=4, color="yellow")


def mimic():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    from pypdf import PdfReader
    import comparison_edv_pdf as C
    from comparison_edv_pdf import parse_page
    import condition_aware_grounding as G

    nihf = sorted(RESULTS.glob("anatomical_measurements_nih_*.json"))[-1]
    N = json.loads(nihf.read_text(encoding="utf-8"))
    rot_limit = N["rot_limit"]
    feas = {(t["view"], t["measure"]): t for t in N["tests"]}
    # a group is shown when its primary measure passed the NIH check in PA or AP
    primary = {"ctr": "ctr", "dome": "dome_ratio", "cp": "cp_angle_max", "atel": "lung_share_dev"}
    nih_ok = {}
    for gname, key in primary.items():
        ts = [feas.get((v, key)) for v in ("PA", "AP")]
        nih_ok[gname] = any(t and t["feasible"] for t in ts)
        nih_ok[gname + "_why"] = ", ".join(f"{t['view']} AUC {t['auc']:.2f} [{t['ci'][0]:.2f}-{t['ci'][1]:.2f}]"
                                           for t in ts if t and t["auc"] is not None)
    sal_ok = N["pointing"]
    proj = projections()
    rec = json.loads((LOGS / "radiologist_requested_plus_edv_scoring_form_v26.json").read_text(encoding="utf-8"))
    pages = {p["title"]: p for p in (parse_page(x.extract_text()) for x in PdfReader(
        "D:/datasets/held_out_eval_images/subset_comparisons_clean_filtered.pdf").pages)}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    data = []
    for r in rec:
        cur = Path(r["image"])
        sid = cur.parent.name.split("_subject_")[1]
        imgs, res, cams = {}, {}, {}
        for which, p in (("prior", cur.with_name("prior.jpg")), ("current", cur)):
            im = Image.open(p).convert("L")
            im.thumbnail((1024, 1024))
            imgs[which] = np.array(im)
            res[which] = measure(segment_all(imgs[which]), rot_limit)
            cams[which] = gradcam(imgs[which])
        pr = proj.get(sid)
        comp, why = comparable(res["prior"], res["current"], pr)
        data.append(dict(r=r, sid=sid, imgs=imgs, res=res, cams=cams, proj=pr, comp=comp, comp_why=why))
        print(f"{r['title']}: proj {pr}, QC cur {res['current']['qc']}, comparable {comp}", flush=True)
    LOGS.mkdir(exist_ok=True)
    json.dump([{"title": x["r"]["title"], "projection": x["proj"], "comparable": x["comp"], "why": x["comp_why"],
                **{w: {"qc": x["res"][w]["qc"], "why": x["res"][w]["why"],
                       "val": {a: float(b) for a, b in x["res"][w]["val"].items()}} for w in ("prior", "current")}}
               for x in data], (LOGS / f"anatomical_measurements_mimic19_{stamp}.json").open("w", encoding="utf-8"),
              indent=1)

    W_IN, H_IN = 11.7, 8.3
    wrap = lambda s, n: "\n".join(textwrap.wrap(" ".join(str(s).split()), n))  # noqa: E731

    def change_line(x, key, label, d=2):
        vp, vc = x["res"]["prior"]["val"], x["res"]["current"]["val"]
        if key not in vp or key not in vc:
            return f"{label}: not available on both images"
        if not x["comp"]:
            return f"{label}: prior {vp[key]:.{d}f}, current {vc[key]:.{d}f} - NOT compared ({x['comp_why']})"
        return f"{label}: prior {vp[key]:.{d}f} -> current {vc[key]:.{d}f} ({vc[key] - vp[key]:+.{d}f})"

    # ---------------- 1. overlays + prior/current (all 19)
    ov = LOGS / f"anatomical_measurements_overlays_{stamp}.pdf"
    with PdfPages(ov) as pdf:
        fig = plt.figure(figsize=(W_IN, H_IN))
        fig.text(0.03, 0.97, feasibility_text(N, nihf), fontsize=6.6, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        for x in data:
            fig = plt.figure(figsize=(W_IN, H_IN))
            fig.text(0.015, 0.985, f"{x['r']['title']} - anatomical measurements (computed for every image; not "
                     f"placed by any finding). Projection: {x['proj'] or 'not recorded'}", fontsize=10,
                     weight="bold", va="top")
            for i, w in enumerate(("prior", "current")):
                for j, (grp, cap) in enumerate(((("ctr",), "Heart + CTR lines"),
                                                (("volume",), "Lung height + thoracic width"),
                                                (("dome", "cp"), "Dome chords + costophrenic lines"),
                                                (("atel",), "Midline, hila (x), trachea (^)"))):
                    ax = fig.add_axes([0.015 + j * 0.16, 0.5 - i * 0.46, 0.155, 0.43])
                    draw_anatomy(ax, x["imgs"][w], x["res"][w], grp, f"{w.upper()}: {cap}")
            lines = [f"QC prior: " + ("pass" if all(x['res']['prior']['qc'].values()) else
                                      "; ".join(f"{k}: {v}" for k, v in x['res']['prior']['why'].items() if v)),
                     f"QC current: " + ("pass" if all(x['res']['current']['qc'].values()) else
                                        "; ".join(f"{k}: {v}" for k, v in x['res']['current']['why'].items() if v)),
                     f"Prior vs current comparable: {'yes' if x['comp'] else 'no'} ({x['comp_why']})", ""]
            for key, lab in (("ctr", "CTR"), ("lung_height", "Lung height/width"), ("lung_area", "Lung area/width^2"),
                             ("dome_ratio", "Dome ratio (mean)"), ("cp_angle_R", "CP angle R"),
                             ("cp_angle_L", "CP angle L"), ("lung_share_R", "Right lung share"),
                             ("rotation_index", "Rotation index")):
                lines.append(change_line(x, key, lab, 0 if "angle" in key else 2))
            fig.text(0.665, 0.95, "\n".join(sum((textwrap.wrap(s, 62) or [""] for s in lines), [])), fontsize=6.2,
                     va="top", family="monospace")
            fig.text(0.015, 0.012, f"{ANAT}. Blue/cyan = segmentation outlines; yellow = measurement lines. Values are "
                     "divided by thoracic width (scale-free). Not displayed when landmarks fail QC.", fontsize=6)
            pdf.savefig(fig)
            plt.close(fig)
        # failure cases
        fig = plt.figure(figsize=(W_IN, H_IN))
        fl = ["FAILURE CASES AND QC (MIMIC-19, prior + current = 38 images)", ""]
        for x in data:
            for w in ("prior", "current"):
                bad = {k: v for k, v in x["res"][w]["why"].items() if v}
                if bad:
                    fl.append(f"{x['r']['title']} {w}: " + "; ".join(f"{k}: {v}" for k, v in bad.items()))
        fl += ["", "Comparisons with prior/current NOT compared: " +
               ", ".join(x["r"]["title"].split()[1] + f" ({x['comp_why']})" for x in data if not x["comp"])]
        fig.text(0.03, 0.97, "\n".join(sum((textwrap.wrap(s, 150) or [""] for s in fl), [])), fontsize=6.2,
                 va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
    print(f"overlays -> {ov}")

    # ---------------- 2. redesigned sample report (cases chosen by finding type, model outputs only)
    want = ["cardiomegaly", "low lung volumes", "hyperinflation", "effusion", "atelectasis", "opacit|pneumonia"]
    picked = []
    for pat in want:
        for x in data:
            if x not in picked and any(re.search(pat, f) for f in x["r"]["findings"]):
                picked.append(x)
                break
    C.DROPDOWNS = list(C.DROPDOWNS) + ["visualization"]
    rep = LOGS / f"anatomical_measurements_report_sample_{stamp}.pdf"
    fields, titles, pno = [], [], 0
    mm = lambda fx, fy, fw, fh: (fx * W_IN * 25.4, (1 - fy - fh) * H_IN * 25.4, fw * W_IN * 25.4, fh * H_IN * 25.4)  # noqa: E731
    VERD = {"Supported": "Agree", "Indeterminate": "Not agree (inconclusive)", "Requires confirmation": "Not agree"}
    with PdfPages(rep) as pdf:
        for x in picked:
            r = x["r"]
            page = pages[r["title"]]
            rows = [i for i in r["review"]]
            for start in range(0, len(rows), 2):
                fig = plt.figure(figsize=(W_IN, H_IN))
                pno += 1
                titles.append(f"{r['title']} (measurement report)")
                fig.text(0.015, 0.985, f"{r['title']} - radiologist report with anatomical measurements (prototype)",
                         fontsize=10.5, weight="bold", va="top")
                for k, w in enumerate(("prior", "current")):
                    ax = fig.add_axes([0.015 + k * 0.105, 0.70, 0.1, 0.255])
                    ax.imshow(x["imgs"][w], cmap="gray")
                    ax.axis("off")
                    ax.set_title(w.capitalize(), fontsize=7.5, loc="left")
                fig.text(0.235, 0.965, "MedGemma report (unchanged)", fontsize=7, weight="bold", va="top")
                fig.text(0.235, 0.948, wrap(page["generated_text"], 92), fontsize=5.3, va="top", family="monospace")
                fig.text(0.62, 0.965, "Reference report (unchanged)", fontsize=7, weight="bold", va="top")
                fig.text(0.62, 0.948, wrap(page["truth_text"], 92), fontsize=5.3, va="top", family="monospace")
                fig.text(0.015, 0.685, f"Projection: {x['proj'] or 'not recorded'}.  Prior vs current measurements "
                         f"comparable: {'yes' if x['comp'] else 'no - ' + x['comp_why']}.", fontsize=6.6)
                for ri, it in enumerate(rows[start:start + 2]):
                    y0 = 0.36 - ri * 0.335
                    f = it["finding"]
                    ch = r.get("change_claims", {}).get(f) or it.get("change") or ["not stated", ""]
                    fig.text(0.015, y0 + 0.3, f"{it['name']}", fontsize=8, weight="bold", va="top")
                    fig.text(0.015, y0 + 0.28, wrap(f"MedGemma interval change: {ch[0]}" +
                                                    (f' - "{ch[1]}"' if ch[1] else ""), 30), fontsize=5.8, va="top")
                    fig.text(0.015, y0 + 0.2, wrap("Presence verification (CheXagent, report-blinded, unchanged): "
                                                   f"{VERD.get(it['status'], it['status'])}"
                                                   + (f", p_present {it['p_present']:.2f}" if it.get('p_present')
                                                      else ""), 30), fontsize=5.8, va="top",
                             color="#1a7f37" if it["status"] == "Supported" else "#b42318")
                    grp = group_for(f)
                    # panel A: anatomical measurement (prior and current)
                    for k, w in enumerate(("prior", "current")):
                        ax = fig.add_axes([0.16 + k * 0.125, y0 + 0.01, 0.12, 0.27])
                        if grp in (None, "sal"):
                            draw_anatomy(ax, x["imgs"][w], x["res"][w], (), f"{w}: lung outline only")
                        elif grp in nih_ok and not nih_ok[grp]:
                            fb = ("volume",) if grp == "dome" else ()
                            draw_anatomy(ax, x["imgs"][w], x["res"][w], fb,
                                         f"{w}: " + ("lung height (descriptive)" if fb else "lung outline only"))
                        else:
                            draw_anatomy(ax, x["imgs"][w], x["res"][w], (grp,), f"{w}: anatomical measurement")
                    # panel B: model-generated localization (original CheXagent box; saliency for consolidation)
                    vp = next((v for v in r["visual_proof"] if v["finding"] == f and v.get("boxes")), None)
                    ax = fig.add_axes([0.415, y0 + 0.01, 0.12, 0.27])
                    ax.imshow(x["imgs"]["current"], cmap="gray")
                    ax.axis("off")
                    for b in (vp or {}).get("boxes", []):
                        x1, y1, x2, y2 = G.box_px(b["box_normalized"], x["imgs"]["current"].shape)
                        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.4))
                    ax.set_title(wrap("Original CheXagent box (unchanged) - " + MODEL if vp else
                                      "CheXagent: no box recorded", 38), fontsize=5, loc="left")
                    if grp == "sal" and sal_ok.get("Pneumonia", {}).get("feasible"):
                        ax2 = fig.add_axes([0.54, y0 + 0.01, 0.12, 0.27])
                        cam = x["cams"]["current"]["Consolidation"]
                        ax2.imshow(x["imgs"]["current"], cmap="gray")
                        ax2.contour(cam, levels=[0.7], colors="orange", linewidths=1.2)
                        ax2.axis("off")
                        ax2.set_title(wrap(SAL, 38), fontsize=5, loc="left")
                    # measurement text and limitations
                    lines = show_lines(x["res"]["current"], x["proj"], nih_ok).get(grp, []) if grp and grp != "sal" \
                        else []
                    if grp == "ctr":
                        lines += [change_line(x, "ctr", "CTR"),
                                  "Prior-current CTR differences are not validated: in these 19 comparisons the CTR "
                                  "of the same patient varied by up to 0.13 between films, and a visibly wrong heart "
                                  "outline can pass QC. Check the outline before using the value."]
                    elif grp == "volume":
                        lines += ["Descriptive only: no labelled data to validate these for low lung volumes.",
                                  change_line(x, "lung_height", "Lung height/width"),
                                  change_line(x, "lung_area", "Lung area/width^2"),
                                  "Area or height differences are NOT proof of low lung volumes (inspiration, "
                                  "positioning, projection, body habitus)."]
                    elif grp == "dome":
                        lines += show_lines(x["res"]["current"], x["proj"], nih_ok)["volume"]
                        lines += [change_line(x, "lung_height", "Lung height/width"),
                                  "Lung height and area: descriptive only; NOT validated for hyperinflation (NIH "
                                  "emphysema AUC about 0.5)."]
                        if nih_ok["dome"]:
                            lines.append(change_line(x, "dome_ratio", "Dome ratio"))
                    elif grp == "cp":
                        lines += [change_line(x, "cp_angle_R", "CP angle R", 0),
                                  change_line(x, "cp_angle_L", "CP angle L", 0),
                                  "Fluid location is not inferred from lung segmentation; the angle describes the "
                                  "lung outline only."]
                    elif grp == "atel" and nih_ok["atel"]:
                        lines.append(change_line(x, "lung_share_R", "Right lung share"))
                    elif grp == "sal":
                        pt = sal_ok.get("Pneumonia", {})
                        cam = x["cams"]["current"]["Consolidation"]
                        yy, xx = np.unravel_index(cam.argmax(), cam.shape)
                        z = zone_of((xx, yy), segment_all(x["imgs"]["current"])) if x["res"]["current"]["qc"]["lungs"] \
                            else None
                        lines += ([f"Saliency peak: {z or 'outside the segmented lungs / segmentation failed'}"]
                                  if pt.get("feasible") else
                                  ["Candidate opacity location (classifier saliency) NOT shown: it failed the NIH "
                                   "check (peaks mostly at the image's lower edge, outside the lungs)."]) + [
                                  f"NIH pointing check (pneumonia boxes): hit {pt.get('hit_rate', 0):.0%} vs lung "
                                  f"baseline {pt.get('baseline') or 0:.0%} - "
                                  + ("passed" if pt.get("feasible") else "NOT passed: treat as unreliable")]
                    else:
                        lines.append("No anatomical measurement exists for this finding; lung outline shown for "
                                     "context only.")
                    lines.append("Limitations: segmentation can fail on rotated, portable or low-contrast films; "
                                 "outer lung edges approximate the inner rib margins; no value is a diagnosis.")
                    fig.text(0.675, y0 + 0.3, "Anatomical measurements", fontsize=6.6, weight="bold", va="top",
                             color="#1f4e9c")
                    fig.text(0.675, y0 + 0.283, "segmentation-derived - not a diagnosis", fontsize=5.6, va="top",
                             color="#1f4e9c", style="italic")
                    fig.text(0.675, y0 + 0.265, "\n".join(sum((textwrap.wrap(s, 50) for s in lines), [])),
                             fontsize=5.5, va="top")
                    labs = (("medgemma", "Finding accuracy"), ("change", "Interval-change accuracy"),
                            ("chexagent", "CheXagent presence evidence"), ("visualization", "Visualization usefulness"))
                    for j, (_, lab) in enumerate(labs):
                        fig.text(0.853, y0 + 0.262 - j * 0.04, lab, fontsize=5.4, va="top")
                    fields.append({"page": pno, "row": ri + 1, "finding": it["name"], "comment_rect": None,
                                   "evidence": VERD.get(it["status"], it["status"]), "ground_truth": "",
                                   "tooltip": None, "tooltip_rect": None, "gt_comment": "", "comment": "",
                                   "medgemma_change": "",
                                   "scores": [(kk, mm(0.955, y0 + 0.244 - j * 0.04, 0.032, 0.022))
                                              for j, (kk, _) in enumerate(labs)]})
                fig.text(0.015, 0.008, "Blue/cyan/yellow = anatomy and measurements (segmentation). Red = CheXagent "
                         "box, orange = classifier saliency: model-generated candidate locations, not disease extent. "
                         "Rating fields are new and empty.", fontsize=6)
                pdf.savefig(fig)
                plt.close(fig)
    C.add_form_fields(rep, fields, titles)
    print(f"report -> {rep} ({len(picked)} comparisons, {len(fields)} findings)")


def feasibility_text(N, path):
    L = ["FEASIBLE MEASUREMENTS BY CONDITION - NIH check (" + path.name + ")",
         f"{N['n_images']} NIH images; {N['excluded_patients']} project patients excluded. Rotation limit (95th pct "
         f"|rotation index|, PA No Finding) = {N['rot_limit']:.3f}. Positive vs No Finding (optimistic: easy controls).",
         "", f"{'View':4s} {'Condition':12s} {'Measure':18s} {'Comput.':>7s} {'AUC':>5s} {'95% CI':>13s} "
         f"{'med pos':>8s} {'med neg':>8s}  Verdict"]
    for t in N["tests"]:
        L.append(f"{t['view']:4s} {t['condition']:12s} {t['measure']:18s} {t['computable']:7.0%} "
                 + (f"{t['auc']:5.2f} {t['ci'][0]:6.2f}-{t['ci'][1]:<6.2f} {t['median_pos']:8.3f} {t['median_neg']:8.3f}"
                    if t["auc"] is not None else f"{'-':>5s} {'-':>13s} {'-':>8s} {'-':>8s}")
                 + ("  FEASIBLE" if t["feasible"] else "  not supported"))
    L += ["", "Low lung volumes: no NIH label - measurements are descriptive only (not validated).", "",
          "Classifier saliency pointing game (peak inside the radiologist box; baseline = lung share inside box):"]
    for lab, p in N["pointing"].items():
        L.append(f"  {lab:12s} ({p['cam_class']:13s}) n={p['n']:3d} hit {p['hit_rate']:.0%} "
                 f"[{p['ci'][0]:.0%}-{p['ci'][1]:.0%}] baseline {p['baseline'] or 0:.0%}  "
                 + ("FEASIBLE" if p["feasible"] else "not supported"))
    L += ["", "QC pass rates: " + "; ".join(f"{v}: " + ", ".join(f"{k} {r:.0%}" for k, r in q.items())
                                          for v, q in N["qc_rates"].items())]
    return "\n".join(L)


if __name__ == "__main__":
    {"nih": nih, "mimic": mimic}[sys.argv[1]]()
