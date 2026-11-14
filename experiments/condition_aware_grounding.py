"""EXPLORATORY: condition-aware display of model-predicted locations (follows grounding_display_case13.py).

Nothing here changes CheXagent's predictions, the Evidence Engine, the scoring forms or any scoring result. The
boxes shown are the ones already recorded (CheXagent located each finding with a side-free phrase: report-blinded;
MedGemma's side is never given to CheXagent). Segmentation, zones, heat maps and measurements are DISPLAY AIDS: they
are never used as evidence that a condition exists.

Display approaches (by condition):
- GLOBAL  (low lung volumes, hyperinflation, emphysema*, pulmonary edema*: both lungs; cardiomegaly: heart):
  segmentation outline + descriptive measurements (lung height / thorax width, cardiothoracic ratio, change vs prior);
  the original box is kept for comparison.            (*added here: diffuse, bilateral by nature)
- REGIONAL (pleural effusion, atelectasis, consolidation, pneumonia, infiltration, opacities, pleural thickening):
  the original box plus the lung zone(s) it falls in (upper / middle / lower third of the segmented lung), labelled
  "anatomical region", not a disease boundary.
- FOCAL   (pneumothorax, nodule, mass and other localized findings): the original box, with the lung outline as context.

Steps:
    .venv/Scripts/python.exe experiments/condition_aware_grounding.py validate   # coordinate tests + NIH boxes
    .venv/Scripts/python.exe experiments/condition_aware_grounding.py gallery    # MIMIC-19 display pages
    .venv/Scripts/python.exe experiments/condition_aware_grounding.py review     # blinded radiologist review form
    .venv/Scripts/python.exe experiments/condition_aware_grounding.py report     # comparison report
Outputs: experiments/results/condition_aware_grounding_nih.json (NIH only, public data); everything with MIMIC-CXR
images in logs/ (git-ignored), new time-stamped files, nothing overwritten.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
LOGS = ROOT / "logs"
RESULTS = HERE / "results"
NIH_DIR = Path("D:/datasets/chestxray14/images")
NIH_OUT = RESULTS / "condition_aware_grounding_nih.json"
GALLERY_CACHE = LOGS / "condition_aware_grounding_gallery.json"
SEED, N_BOOT = 20261108, 2000
GLOBAL_LUNG = ("low lung volumes", "hyperinflation", "emphysema", "pulmonary edema")
GLOBAL_HEART = ("cardiomegaly",)
REGIONAL = ("pleural effusion", "atelectasis", "consolidation", "pneumonia", "infiltration", "opacities",
            "pleural thickening", "effusion", "infiltrate")
DISCLAIMER = "Display aid only: segmentation, zones and measurements do not confirm that the condition exists."


def category(finding):
    f = finding.lower()
    if any(k in f for k in GLOBAL_HEART):
        return "global-heart"
    if any(k in f for k in GLOBAL_LUNG):
        return "global-lung"
    if any(k in f for k in REGIONAL):
        return "regional"
    return "focal"


# ---------------------------------------------------------------- segmentation and geometry
_SEG = None


def segment(img):
    """{'Right Lung', 'Left Lung', 'Heart': bool mask} at img's own size (2-D uint8/float array). The image is
    zero-padded to a square (no cropping), resized to 512 for the model, and the masks are mapped back."""
    global _SEG
    import torch
    import torchxrayvision as xrv
    if _SEG is None:
        _SEG = xrv.baseline_models.chestx_det.PSPNet().eval().to("cuda" if torch.cuda.is_available() else "cpu")
    dev = next(_SEG.parameters()).device
    img = np.asarray(img, np.float32)
    h, w = img.shape
    side = max(h, w)
    pad = np.zeros((side, side), np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    pad[top:top + h, left:left + w] = img
    x = xrv.datasets.XRayResizer(512)(xrv.datasets.normalize(pad, 255)[None])
    with torch.no_grad():
        out = torch.sigmoid(_SEG(torch.from_numpy(x)[None].to(dev)))[0].cpu().numpy()
    masks = {}
    for name, i in (("Left Lung", 4), ("Right Lung", 5), ("Heart", 8)):
        m = np.array(Image.fromarray((out[i] > 0.5).astype(np.uint8) * 255).resize((side, side), Image.NEAREST)) > 0
        masks[name] = m[top:top + h, left:left + w]
    return masks


def box_px(box, shape):
    """Normalized [x1, y1, x2, y2] (0-100, as CheXagent and the forms store them) -> pixel ints, the same formula as
    XRayVQATool._visualize_boxes."""
    h, w = shape
    x1, y1, x2, y2 = box
    return int(x1 / 100 * w), int(y1 / 100 * h), int(x2 / 100 * w), int(y2 / 100 * h)


def box_mask(box, shape):
    m = np.zeros(shape, bool)
    x1, y1, x2, y2 = box_px(box, shape)
    m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
    return m


def zone_regions(masks):
    """{(lung, third): mask}; thirds of each lung's vertical extent (upper / middle / lower)."""
    out = {}
    for lung in ("Right Lung", "Left Lung"):
        m = masks[lung]
        ys = np.nonzero(m)[0]
        if ys.size == 0:
            continue
        lo, hi = ys.min(), ys.max() + 1
        rows = np.arange(m.shape[0])[:, None]
        for k, name in enumerate(("upper", "middle", "lower")):
            a, b = lo + k * (hi - lo) / 3, lo + (k + 1) * (hi - lo) / 3
            out[(lung, name)] = m & (rows >= a) & (rows < b)
    return out


def zones_for_boxes(boxes, masks, min_share=0.2):
    """Union of the lung zones the boxes fall in: a zone is included if it holds at least min_share of the box's area
    inside the lungs. Returns (mask, [zone names]); empty if no box touches a lung."""
    regions = zone_regions(masks)
    shape = masks["Heart"].shape
    union, names = np.zeros(shape, bool), []
    for b in boxes:
        bm = box_mask(b, shape)
        inside = sum(int((bm & r).sum()) for r in regions.values())
        for key, r in regions.items():
            ov = int((bm & r).sum())
            if inside and ov >= min_share * inside:
                union |= r
                names.append(f"{key[0].split()[0].lower()} {key[1]} zone")
    return union, list(dict.fromkeys(names))


def outline(mask):
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask, iterations=3)


def measures(masks):
    """Descriptive geometry from the segmentation (not validated thresholds)."""
    lungs = masks["Left Lung"] | masks["Right Lung"]
    ys, xs = np.nonzero(lungs)
    if xs.size == 0:
        return {}
    tw = xs.max() - xs.min()
    hy, hx = np.nonzero(masks["Heart"])
    heights = [np.ptp(np.nonzero(masks[k])[0]) if masks[k].any() else 0 for k in ("Right Lung", "Left Lung")]
    return {"lung_height_to_thorax_width": float(max(heights) / tw) if tw else None,
            "cardiothoracic_ratio": float(np.ptp(hx) / tw) if tw and hx.size else None,
            "lung_area_to_thorax_box": float(lungs.sum() / ((ys.max() - ys.min()) * tw)) if tw else None}


def center_in(box, mask):
    h, w = mask.shape
    x1, y1, x2, y2 = box_px(box, (h, w))
    cy, cx = min(h - 1, max(0, (y1 + y2) // 2)), min(w - 1, max(0, (x1 + x2) // 2))
    return bool(mask[cy, cx])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return [xs.min() / w * 100, ys.min() / h * 100, (xs.max() + 1) / w * 100, (ys.max() + 1) / h * 100] \
        if xs.size else None


def side_of(mask_or_box, shape):
    """'right' / 'left' / 'both' in patient terms (patient's right = image left)."""
    if isinstance(mask_or_box, np.ndarray):
        xs = np.nonzero(mask_or_box)[1]
        if xs.size == 0:
            return None
        lo, hi = xs.min() / shape[1], xs.max() / shape[1]
    else:
        lo, hi = mask_or_box[0] / 100, mask_or_box[2] / 100
    return "both" if lo < 0.45 and hi > 0.55 else "right" if hi <= 0.55 else "left"


# ---------------------------------------------------------------- 1. validation
def validate():
    from grounding_recall import cases
    if NIH_OUT.exists():
        raise SystemExit(f"{NIH_OUT} exists; not overwriting")
    tests = {}
    # (a) box conversion: same formula as the drawing code; corners land inside the image for non-square shapes
    for shape in ((3056, 2544), (2544, 3056), (1024, 1024)):
        x1, y1, x2, y2 = box_px([0, 0, 100, 100], shape)
        assert (x2, y2) == (shape[1], shape[0]) and (x1, y1) == (0, 0), shape
        assert box_px([50, 25, 50, 25], shape) == (shape[1] // 2, shape[0] // 4, shape[1] // 2, shape[0] // 4)
    tests["box conversion (normalized 0-100 -> pixels, 3 shapes)"] = "passed"
    # (b) scale: a MIMIC image segmented at full size and as a 1,024-px thumbnail gives the same masks
    rec = json.loads((LOGS / "radiologist_requested_plus_edv_scoring_form_v26.json").read_text(encoding="utf-8"))
    scale = []
    for r in rec[:5]:
        full = np.array(Image.open(r["image"]).convert("L"))
        th = Image.fromarray(full)
        th.thumbnail((1024, 1024))
        m_full, m_small = segment(full), segment(np.array(th))
        for k in m_full:
            up = np.array(Image.fromarray(m_small[k].astype(np.uint8)).resize(full.shape[::-1], Image.NEAREST)) > 0
            inter, uni = (up & m_full[k]).sum(), (up | m_full[k]).sum()
            scale.append(inter / uni if uni else 1.0)
    tests["scale: full-size vs 1,024-px masks, mean IoU over 5 MIMIC images x 3 organs"] = round(float(np.mean(scale)), 3)
    # (c) laterality: flipping the image swaps the lung labels
    full = np.array(Image.open(rec[0]["image"]).convert("L").resize((1024, 1024)))
    m, mf = segment(full), segment(full[:, ::-1])
    sw = [(m["Right Lung"] & mf["Left Lung"][:, ::-1]).sum() / max(1, (m["Right Lung"] | mf["Left Lung"][:, ::-1]).sum()),
          (m["Left Lung"] & mf["Right Lung"][:, ::-1]).sum() / max(1, (m["Left Lung"] | mf["Right Lung"][:, ::-1]).sum())]
    tests["laterality: horizontal flip swaps right/left lung masks (IoU)"] = [round(float(v), 3) for v in sw]

    chex = {(json.loads(l)["file"], json.loads(l)["label"]): json.loads(l)["boxes"]
            for l in (RESULTS / "grounding_chexagent.jsonl").read_text(encoding="utf-8").splitlines()}
    maira = {(json.loads(l)["file"], json.loads(l)["label"]): json.loads(l)["boxes"]
             for l in (RESULTS / "grounding_maira2.jsonl").read_text(encoding="utf-8").splitlines()}
    rows, lat_ok, seg_cache = [], [], {}
    for n, c in enumerate(cases(), 1):
        f = c["file"]
        if f not in seg_cache:
            img = np.array(Image.open(NIH_DIR / f).convert("L"))
            masks = segment(img)
            seg_cache.clear()
            seg_cache[f] = masks
            lat_ok.append(bool(np.nonzero(masks["Left Lung"])[1].mean() > np.nonzero(masks["Right Lung"])[1].mean())
                          if masks["Left Lung"].any() and masks["Right Lung"].any() else None)
        masks = seg_cache[f]
        shape = masks["Heart"].shape
        lungs = masks["Left Lung"] | masks["Right Lung"]
        truth = c["truth"]
        boxes = chex.get((f, c["label"])) or []
        cat = category(c["phrase"])
        row = {"file": f, "label": c["label"], "category": cat, "located": bool(boxes),
               "truth_center_in_lungs_or_heart": center_in(truth, lungs | masks["Heart"])}
        tb = box_mask(truth, shape)
        row["truth_area_share_of_lungs"] = float((tb & lungs).sum() / max(1, lungs.sum()))
        if boxes:
            bm = np.zeros(shape, bool)
            for b in boxes:
                bm |= box_mask(b, shape)
            row["box_hit"] = center_in(truth, bm)
            row["box_iou"] = max(iou(truth, b) for b in boxes)
            row["box_share_of_lungs"] = float((bm & lungs).sum() / max(1, lungs.sum()))
            zm, zn = zones_for_boxes(boxes, masks)
            row["zones"] = zn
            row["zone_hit"] = center_in(truth, zm) if zm.any() else False
            row["zone_share_of_lungs"] = float(zm.sum() / max(1, lungs.sum()))
            truth_side = side_of(truth, shape)
            row["truth_side"] = truth_side
            row["box_side"] = side_of(np.asarray(bm), shape)
            row["zone_side"] = side_of(zm, shape) if zm.any() else None
        if cat == "global-heart":
            hb = mask_bbox(masks["Heart"])
            row["heart_iou"] = iou(truth, hb) if hb else 0.0
            row["heart_hit"] = center_in(truth, masks["Heart"])
            mb = maira.get((f, c["label"])) or []
            row["maira_iou"] = max((iou(truth, b) for b in mb), default=0.0)
        rows.append(row)
        if n % 100 == 0:
            print(f"validate {n}", flush=True)
    tests["laterality: patient's left lung lies on the image right (NIH images)"] = \
        f"{sum(v is True for v in lat_ok)}/{sum(v is not None for v in lat_ok)}"
    tests["radiologist boxes whose centre lies in the segmented lungs or heart"] = \
        f"{sum(r['truth_center_in_lungs_or_heart'] for r in rows)}/{len(rows)}"
    NIH_OUT.write_text(json.dumps({"tests": tests, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(tests, indent=1))
    print(f"{len(rows)} NIH radiologist boxes -> {NIH_OUT}")


def boot_ci(vals, seed=SEED):
    vals = np.asarray(vals, float)
    if vals.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    b = [vals[rng.integers(0, vals.size, vals.size)].mean() for _ in range(N_BOOT)]
    return tuple(np.percentile(b, [2.5, 97.5]))


# ---------------------------------------------------------------- 2. gallery (MIMIC-19 findings)
CAL = None


def calibration_status(finding):
    global CAL
    import provisional_findings as P
    from comparison_edv_pdf import SIX, validated_name
    v = validated_name(finding) or finding
    if v in SIX:
        return "calibrated (Evidence Engine v1.0)"
    if any(k in finding.lower() for k in P.TERMS):
        return "provisional cut-offs"
    return "no cut-off (not one of the 13 conditions)"


def gallery_items():
    rec = json.loads((LOGS / "radiologist_requested_plus_edv_scoring_form_v26.json").read_text(encoding="utf-8"))
    from openpyxl import load_workbook
    book = sorted(Path("D:/datasets/doctor_response").glob("doctor_responses_*.xlsx"))[-1]
    ws = load_workbook(book, read_only=True)["All rows"]
    head = [c.value for c in next(ws.iter_rows(max_row=1))]
    scores = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        d = dict(zip(head, r))
        if d["Form"] == "MIMIC-19" and d["Row type"] == "MedGemma stated":
            scores.setdefault((d["Case"], d["Condition"]), {})[d["Doctor"]] = d["Score: CheXagent grounding"]
    items = []
    for r in rec:
        case = int(r["title"].split()[1])
        for i in r["review"]:
            vp = next((v for v in r["visual_proof"] if v["finding"] == i["finding"] and v.get("boxes")), None)
            if not vp:
                continue
            from comparison_edv_pdf import validated_name
            cond = validated_name(i["finding"]) or i["finding"]
            items.append({"case": case, "title": r["title"], "image": r["image"], "finding": i["finding"],
                          "name": i["name"], "category": category(i["finding"]), "status": i["status"],
                          "p_present": i.get("p_present"), "calibration": calibration_status(i["finding"]),
                          "boxes": [b["box_normalized"] for b in vp["boxes"]],
                          "original_location": i.get("image_location") or "; ".join(
                              b["location"].replace("patient's ", "") for b in vp["boxes"]),
                          "grounding_scores": scores.get((case, cond), {})})
    return items


def render_pair(item, size=1024):
    """(original image array, overlays) where overlays describe the original and the experimental display."""
    img = Image.open(item["image"]).convert("L")
    img.thumbnail((size, size))
    arr = np.array(img)
    masks = segment(arr)
    prior = Path(item["image"]).with_name("prior.jpg")
    exp = {"category": item["category"]}
    lungs = masks["Left Lung"] | masks["Right Lung"]
    if item["category"] == "global-heart":
        exp.update(mask=masks["Heart"], label="Heart (segmentation outline)")
    elif item["category"] == "global-lung":
        exp.update(mask=lungs, label="Both lungs (segmentation outline)")
    elif item["category"] == "regional":
        zm, zn = zones_for_boxes(item["boxes"], masks)
        exp.update(mask=zm, label="Anatomical region: " + ("; ".join(zn) or "box outside the segmented lungs"),
                   keep_box=True)
    else:
        exp.update(mask=lungs, label="Lungs for anatomical context", keep_box=True, faint=True)
    m = measures(masks)
    if item["category"].startswith("global") and prior.exists():
        p = np.array(Image.open(prior).convert("L"))
        pimg = Image.fromarray(p)
        pimg.thumbnail((size, size))
        exp["prior_measures"] = measures(segment(np.array(pimg)))
    exp["measures"] = m
    return arr, masks, exp


def draw_panel(ax, arr, item, exp=None, show_box=True, fontsize=8):
    from matplotlib.patches import Rectangle
    h, w = arr.shape
    ax.imshow(arr, cmap="gray")
    ax.axis("off")
    if exp is not None and exp.get("mask") is not None and exp["mask"].any():
        if exp.get("faint"):
            ax.imshow(np.ma.masked_where(~outline(exp["mask"]), np.ones_like(arr)), cmap="winter", alpha=0.9)
        elif exp["category"] == "regional":
            ax.imshow(np.ma.masked_where(~exp["mask"], np.ones_like(arr)), cmap="autumn", alpha=0.30)
            ax.imshow(np.ma.masked_where(~outline(exp["mask"]), np.ones_like(arr)), cmap="autumn", alpha=0.9)
        else:
            ax.imshow(np.ma.masked_where(~exp["mask"], np.ones_like(arr)), cmap="winter", alpha=0.25)
            ax.imshow(np.ma.masked_where(~outline(exp["mask"]), np.ones_like(arr)), cmap="winter", alpha=0.95)
    if show_box:
        for b in item["boxes"]:
            x1, y1, x2, y2 = box_px(b, (h, w))
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.8))


def info_lines(item, exp, blinded=False):
    verdict = {"Supported": "Agree (Supported)", "Indeterminate": "Not agree (Indeterminate: between cut-offs)",
               "Requires confirmation": "Not agree (evidence against, or side differs)"}.get(item["status"], item["status"])
    lines = [f"Finding (MedGemma): {item['name']}    |    display type: {exp['category']}",
             f"Presence prediction (CheXagent, report-blinded 'Is there ...?'): p_present = "
             + (f"{item['p_present']:.3f}" if item["p_present"] is not None else "n/a"),
             f"Calibration status: {item['calibration']}",
             f"Evidence verdict (as on the form): {verdict}"]
    if not blinded:
        lines.append(f"Original localization (CheXagent box, side-free phrase): {item['original_location']}")
        lines.append(f"Experimental visualization: {exp['label']}")
    m, pm = exp.get("measures") or {}, exp.get("prior_measures")
    if exp["category"].startswith("global") and m:
        f2 = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        s = (f"Measurements (descriptive, not validated): lung height / thorax width "
             f"{f2(m.get('lung_height_to_thorax_width'))}; cardiothoracic ratio {f2(m.get('cardiothoracic_ratio'))}")
        if pm:
            s += (f"; prior {f2(pm.get('lung_height_to_thorax_width'))} / {f2(pm.get('cardiothoracic_ratio'))}"
                  " (positioning may differ)")
        lines.append(s)
    lines.append(DISCLAIMER)
    return lines


def gallery():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    items = gallery_items()
    out = LOGS / f"condition_aware_grounding_gallery_{time.strftime('%Y%m%d-%H%M%S')}.pdf"
    summary = []
    with PdfPages(out) as pdf:
        for it in items:
            arr, masks, exp = render_pair(it)
            fig = plt.figure(figsize=(11.7, 8.3))
            gs = it["grounding_scores"]
            fig.suptitle(f"{it['title']} - {it['name']}   (doctors' grounding score for the original box: "
                         + ", ".join(f"{k.replace('Dr ', '')} {v if v is not None else 'N/A'}" for k, v in gs.items())
                         + ")", fontsize=10, x=0.02, ha="left")
            a1 = fig.add_axes([0.02, 0.22, 0.46, 0.68])
            draw_panel(a1, arr, it, None, True)
            a1.set_title("Original: CheXagent box", fontsize=9, loc="left")
            a2 = fig.add_axes([0.52, 0.22, 0.46, 0.68])
            draw_panel(a2, arr, it, exp, exp.get("keep_box", False) or True)
            a2.set_title(f"Experimental: {exp['label']}" + (" (original box kept in red)" if True else ""),
                         fontsize=9, loc="left")
            fig.text(0.02, 0.19, "\n".join(info_lines(it, exp)), fontsize=8, va="top", family="monospace")
            pdf.savefig(fig)
            plt.close(fig)
            summary.append({k: it[k] for k in ("case", "finding", "category", "status", "p_present", "calibration",
                                                  "original_location", "grounding_scores")}
                           | {"experimental": exp["label"], "measures": exp.get("measures"),
                              "prior_measures": exp.get("prior_measures")})
    GALLERY_CACHE.with_name(out.stem + ".json").write_text(json.dumps(summary, indent=1, default=float),
                                                         encoding="utf-8")
    print(f"{len(items)} findings -> {out}")


# ---------------------------------------------------------------- 3. blinded review form
def review():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import comparison_edv_pdf as C
    items = gallery_items()
    rng = random.Random(SEED)
    rng.shuffle(items)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = LOGS / f"condition_aware_grounding_review_{stamp}.pdf"
    key_path = LOGS / f"condition_aware_grounding_review_{stamp}_KEY.json"
    W_IN, H_IN = 11.7, 8.3
    mm = lambda fx, fy, fw, fh: (fx * W_IN * 25.4, (1 - fy - fh) * H_IN * 25.4, fw * W_IN * 25.4, fh * H_IN * 25.4)  # noqa: E731
    fields, titles, key = [], [], []
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(W_IN, H_IN))
        fig.text(0.05, 0.9, "Blinded review: two ways of showing where a finding is", fontsize=16, weight="bold")
        fig.text(0.05, 0.84, "\n".join([
            "Each page shows the same chest X-ray twice, as Display X and Display Y. One is the current display and",
            "one is an experimental display; which is which varies from page to page.",
            "",
            "For EACH display, rate two things separately (1 = poor, 5 = excellent; leave blank if you cannot assess):",
            "  1. Easier to interpret: how quickly and clearly the display tells you where to look.",
            "  2. Localization clinically accurate: whether the marked location matches where the finding is on",
            "     this image (if the finding is absent, rate whether the display misleads).",
            "Then give an overall preference: 1 = Display X clearly better ... 3 = no preference ... 5 = Display Y.",
            "",
            "Shaded regions, outlines and measurements are display aids. They do not mean the condition is present.",
            "The text above the images (presence score, calibration, verdict) is the same for both displays."]),
            fontsize=10, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        titles.append("Instructions")
        for n, it in enumerate(items, 1):
            arr, masks, exp = render_pair(it)
            exp_first = rng.random() < 0.5
            key.append({"page": n + 1, "case": it["case"], "finding": it["finding"], "category": it["category"],
                        "display_X": "experimental" if exp_first else "original",
                        "display_Y": "original" if exp_first else "experimental",
                        "original_grounding_scores": it["grounding_scores"]})
            fig = plt.figure(figsize=(W_IN, H_IN))
            fig.suptitle(f"Item {n} of {len(items)} - {it['title']}", fontsize=10, x=0.02, ha="left")
            fig.text(0.02, 0.945, "\n".join(info_lines(it, exp, blinded=True)[:-1]), fontsize=7.5, va="top",
                     family="monospace")
            for k, (label, experimental) in enumerate((("Display X", exp_first), ("Display Y", not exp_first))):
                ax = fig.add_axes([0.02 + k * 0.49, 0.13, 0.46, 0.66])
                draw_panel(ax, arr, it, exp if experimental else None, True)
                ax.set_title(label, fontsize=10, loc="left", weight="bold")
                fig.text(0.02 + k * 0.49, 0.10, "Easier to interpret (1-5):", fontsize=8)
                fig.text(0.02 + k * 0.49 + 0.23, 0.10, "Localization accurate (1-5):", fontsize=8)
            fig.text(0.02, 0.045, "Overall preference (1 = X better, 3 = none, 5 = Y better):", fontsize=8)
            fig.text(0.50, 0.045, "Comment:", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
            titles.append(f"Item {n}")
            dd = lambda fx, fy: mm(fx, fy, 0.045, 0.028)  # noqa: E731
            fields.append({"page": n + 1, "row": 1, "finding": f"item {n}", "evidence": "", "ground_truth": "",
                           "tooltip": None, "tooltip_rect": None, "gt_comment": "", "comment": "",
                           "medgemma_change": "",
                           "scores": [("x_interpret", dd(0.165, 0.092)), ("x_accurate", dd(0.40, 0.092)),
                                      ("y_interpret", dd(0.655, 0.092)), ("y_accurate", dd(0.89, 0.092)),
                                      ("preference", dd(0.40, 0.037))],
                           "comment_rect": mm(0.56, 0.01, 0.42, 0.06)})
    for k in ("x_interpret", "x_accurate", "y_interpret", "y_accurate", "preference"):
        if k not in C.DROPDOWNS:
            C.DROPDOWNS.append(k)
    C.add_form_fields(out, fields, titles)
    key_path.write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(items)} items -> {out}\nanswer key (do not send) -> {key_path}")


# ---------------------------------------------------------------- 4. comparison report
def report():
    data = json.loads(NIH_OUT.read_text(encoding="utf-8"))
    rows = data["rows"]
    L = ["# Condition-aware grounding display: comparison report (exploratory)", "",
         f"Built {time.strftime('%Y-%m-%d %H:%M')} by experiments/condition_aware_grounding.py. CheXagent's predictions, "
         "the Evidence Engine and all scoring results are unchanged; the boxes are the recorded report-blinded ones "
         "(side-free phrase). Segmentation (torchxrayvision PSPNet, ChestX-Det), lung zones and measurements are "
         "display aids, not evidence that a condition exists.", "",
         "## 1. Coordinate checks", "", "| test | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in data["tests"].items()]
    L += ["", "## 2. Automated comparison on 947 NIH radiologist boxes", "",
          "Hit = the centre of the radiologist's box falls inside the display (the box, or the shaded region). A larger "
          "display hits more easily, so the share of the lungs it covers is reported next to it: a display is only "
          "better if it hits more without covering most of the lungs. 95% CIs: bootstrap over findings "
          f"({N_BOOT} resamples).", "",
          "| condition (NIH label) | display type | n located / n | original box: hit | lung zone: hit | "
          "box share of lungs | zone share of lungs | zone side = truth side |", "|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if not v else f"{100 * np.mean(v):.0f}% [{100 * boot_ci(v)[0]:.0f}-{100 * boot_ci(v)[1]:.0f}]"  # noqa: E731
    for label in sorted({r["label"] for r in rows}):
        rr = [r for r in rows if r["label"] == label]
        loc = [r for r in rr if r["located"]]
        side = [r["zone_side"] == r["truth_side"] or r["zone_side"] == "both" for r in loc if r.get("zone_side")]             if rr[0]["category"] != "global-heart" else []
        L.append(f"| {label} | {rr[0]['category']} | {len(loc)}/{len(rr)} | {fmt([r['box_hit'] for r in loc])} | "
                 f"{fmt([r['zone_hit'] for r in loc])} | {100 * np.mean([r['box_share_of_lungs'] for r in loc]):.0f}% | "
                 f"{100 * np.mean([r['zone_share_of_lungs'] for r in loc]):.0f}% | "
                 f"{(f'{100 * np.mean(side):.0f}%' if side else '- (heart)')} |" if loc else f"| {label} | - | 0/{len(rr)} | | | | | |")
    hearts = [r for r in rows if r["category"] == "global-heart"]
    if hearts:
        loc = [r for r in hearts if r["located"]]
        L += ["", "### Cardiomegaly (global, heart): which outline matches the radiologist's box", "",
              "| display | mean IoU with radiologist box | IoU >= 0.5 | centre hit |", "|---|---|---|---|",
              f"| CheXagent box (located {len(loc)}/{len(hearts)}) | {np.mean([r['box_iou'] for r in loc]):.2f} | "
              f"{fmt([r['box_iou'] >= 0.5 for r in loc])} | {fmt([r['box_hit'] for r in loc])} |",
              f"| Heart segmentation outline (all {len(hearts)}) | {np.mean([r['heart_iou'] for r in hearts]):.2f} | "
              f"{fmt([r['heart_iou'] >= 0.5 for r in hearts])} | {fmt([r['heart_hit'] for r in hearts])} |",
              f"| MAIRA-2 box (all {len(hearts)}) | {np.mean([r['maira_iou'] for r in hearts]):.2f} | "
              f"{fmt([r['maira_iou'] >= 0.5 for r in hearts])} | - |"]
    fails = [r for r in rows if r["located"] and r["category"] == "regional" and not r["zone_hit"]]
    L += ["", "## 3. Where the zone display fails (regional findings)", "",
          f"- {len(fails)} of {sum(r['located'] and r['category'] == 'regional' for r in rows)} located regional "
          "findings: the radiologist's box centre is outside the shaded zone(s).",
          f"  - box outside the segmented lungs (no zone): {sum(not r['zones'] for r in fails)}",
          f"  - zone on the wrong side: "
          f"{sum(bool(r['zones']) and r.get('zone_side') not in (r.get('truth_side'), 'both') for r in fails)}",
          f"  - right side, wrong third (zone too coarse or box misplaced vertically): "
          f"{sum(bool(r['zones']) and r.get('zone_side') in (r.get('truth_side'), 'both') for r in fails)}",
          f"- Radiologist boxes centred outside the segmented lungs and heart (segmentation or lateral/pleural "
          f"findings at the edge): {sum(not r['truth_center_in_lungs_or_heart'] for r in rows)}",
          "- Global lung findings (low lung volumes, hyperinflation) have no radiologist boxes in NIH, so they are "
          "judged only in the blinded review and the gallery.",
          "- Focal findings: the display is unchanged (original box), so its accuracy is the box's; zones are too "
          "coarse to localize nodules and masses.", "",
          "## 4. Still to do before adoption", "",
          "- Blinded radiologist review (logs/condition_aware_grounding_review_*.pdf; key kept separately): per "
          "display, 'easier to interpret' and 'localization clinically accurate' rated separately, plus preference.",
          "- Adopt a display for a condition only if the review favours it on both questions."]
    out = RESULTS / f"condition_aware_grounding_report_{time.strftime('%Y%m%d-%H%M%S')}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"-> {out}")


if __name__ == "__main__":
    {"validate": validate, "gallery": gallery, "review": review, "report": report}[sys.argv[1]]()
