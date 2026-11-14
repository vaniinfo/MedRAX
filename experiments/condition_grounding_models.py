"""EXPERIMENT: condition-specific localization models vs CheXagent / MAIRA-2 grounding (presentation research only).

Nothing here changes MedGemma, CheXagent presence verification, the Evidence Engine, any model output already
recorded, any score, report or form. New files only (MIMIC-CXR content in logs/; public NIH/VinDr numbers in
experiments/results/). Grad-CAM / attention maps are not used. Localization is never guided by MedGemma's findings,
its claimed side or the reference report: each model runs on the whole current image.

Models (weights in C:/Users/hari/model-weights, loaded without remote code; HF revisions recorded in the output)
  ptx_vasandesh  vasandesh/pneumothorax-unet-efficientnet-b4  U-Net/EfficientNet-B4 (segmentation_models_pytorch),
                 SIIM-ACR; mask + classification; ImageNet normalization, 3 channels, 512 (zero-padded to square here).
  ptx_ianpan     ianpan/pneumothorax-cxr  EfficientNetV2-S + U-Net (reviewed unet.py loaded by path; timm encoder);
                 mask + classification; 1 channel, stretched to 512, scaled to [-1, 1] (as in its modeling.py).
                 Training data undocumented.
  pna_ianpan     ianpan/pneumonia-cxr  same architecture; RSNA Pneumonia + SIIM-FISABIO-RSNA COVID-19 (boxes converted
                 to ellipses) -> lung-opacity mask + classification.
Region = probability > 0.5 (no native threshold documented), mapped back to the original image by reversing the
padding or stretching.

Validation (Phase 3; truth = radiologist boxes; pre-stated metrics: region IoU with the union of truth boxes [primary],
pointing hit [mask peak / largest-box centre inside truth], share of the prediction inside truth, share of truth
covered, side agreement, empty prediction; image-level AUC vs negatives of the same dataset)
  independent: VinDr-derived (3-reader) pneumothorax (15 images) and consolidation (25 images, proxy for the pneumonia
               model's lung opacity), with the existing report-blinded CheXagent and MAIRA-2 boxes on the same images.
  optimistic:  NIH pneumothorax / pneumonia boxes (SIIM-ACR and RSNA are drawn from NIH: possible training overlap),
               with the existing CheXagent and MAIRA-2 boxes.
Truth boxes are boxes, not masks, so Dice against pixel masks is not available. Secondary metric (added after the
first run showed that tight masks are penalized against loose truth boxes): the same metrics after converting each
mask to the bounding boxes of its connected regions ('as boxes').
Caveat: CheXagent's training data (CheXinstruct) includes VinDr-CXR grounding, so the VinDr comparison is independent
for the condition-specific models and MAIRA-2 but optimistic for CheXagent.

    .venv/Scripts/python.exe experiments/condition_grounding_models.py validate
    .venv/Scripts/python.exe experiments/condition_grounding_models.py cases      # forms: run, stability, gallery
"""

import importlib.util
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
MW = Path("C:/Users/hari/model-weights")
NIH = Path("D:/datasets/chestxray14")
VINDR = Path("D:/datasets/vindr-cxr-kaggle-derived")
SEED = 20261108
REVISIONS = {"ptx_vasandesh": "vasandesh/pneumothorax-unet-efficientnet-b4@d9a9fb73",
             "ptx_ianpan": "ianpan/pneumothorax-cxr@f00a7af5", "pna_ianpan": "ianpan/pneumonia-cxr@1ab2c071"}
LABEL = {"ptx_vasandesh": "SIIM-ACR pneumothorax U-Net (vasandesh) - segmentation mask",
         "ptx_ianpan": "Pneumothorax U-Net (ianpan) - segmentation mask",
         "pna_ianpan": "Pneumonia / lung-opacity U-Net (ianpan; ellipse-trained) - segmentation mask"}
MODELS_FOR = {"pneumothorax": ("ptx_vasandesh", "ptx_ianpan"),
              **{c: ("pna_ianpan",) for c in ("pneumonia", "consolidation", "infiltration", "opacities",
                                               "lung opacities", "patchy opacity")}}
_M = {}


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ model loading
def load(name):
    if name in _M:
        return _M[name]
    import torch
    import torch.nn as nn
    if name == "ptx_vasandesh":
        import segmentation_models_pytorch as smp

        class PTX(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1)
                self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 1))

            def forward(self, x):
                d = self.unet.decoder(self.unet.encoder(x))
                return self.unet.segmentation_head(d), self.cls_head(d)
        m = PTX()
        m.load_state_dict(torch.load(MW / "siim_ptx_vasandesh/pytorch_model.bin", map_location="cpu",
                                     weights_only=True), strict=True)
    else:
        from safetensors.torch import load_file
        from timm import create_model
        d = MW / ("ianpan_pneumothorax-cxr" if name == "ptx_ianpan" else "ianpan_pneumonia-cxr")
        spec = importlib.util.spec_from_file_location(f"unet_{name}", d / "unet.py")   # reviewed, model code only
        U = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(U)
        c = json.loads((d / "config.json").read_text())

        class IP(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = create_model(c["backbone"], features_only=True, pretrained=False, in_chans=c["in_chans"])
                self.decoder = U.UnetDecoder(c["decoder_n_blocks"], c["decoder_channels"], c["encoder_channels"],
                                             c["decoder_center_block"], c["decoder_norm_layer"],
                                             c["decoder_attention_type"])
                self.segmentation_head = U.SegmentationHead(c["decoder_channels"][-1], c["seg_num_classes"],
                                                            size=c["img_size"])
                self.pooling = nn.AdaptiveAvgPool2d(1)
                self.dropout = nn.Dropout(c["cls_dropout"])
                self.classifier = nn.Linear(c["feature_dim"], c["cls_num_classes"])

            def forward(self, x):
                f = self.encoder(x)
                seg = self.segmentation_head(self.decoder(f)[-1])
                b, n = f[-1].shape[:2]
                return seg, self.classifier(self.pooling(f[-1]).reshape(b, n))
        m = IP()
        m.load_state_dict(load_file(str(d / "model.safetensors")), strict=True)
    _M[name] = m.eval().to(device())
    return _M[name]


def native_geometry(name):
    return "pad" if name == "ptx_vasandesh" else "stretch"


def predict(name, gray, geometry=None, gamma=1.0, hflip=False):
    """(probability map at the original size, classification probability). Preprocessing reversed exactly."""
    import torch
    geometry = geometry or native_geometry(name)
    g = np.asarray(gray, np.float32)
    if gamma != 1.0:
        g = 255.0 * (g / 255.0) ** gamma
    if hflip:
        g = g[:, ::-1]
    h, w = g.shape
    if geometry == "pad":
        s = max(h, w)
        canvas = np.zeros((s, s), np.float32)
        top, left = (s - h) // 2, (s - w) // 2
        canvas[top:top + h, left:left + w] = g
    else:
        canvas, top, left, s = g, 0, 0, None
    x = np.array(Image.fromarray(canvas).resize((512, 512), Image.BILINEAR))
    if name == "ptx_vasandesh":
        x = np.stack([x / 255.0] * 3)
        x = (x - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None]
    else:
        x = ((x / 255.0) - 0.5) * 2.0
        x = x[None]
    m = load(name)
    with torch.no_grad():
        seg, cls = m(torch.from_numpy(x[None].astype(np.float32)).to(device()))
    p = torch.sigmoid(seg)[0, 0].float().cpu().numpy()
    if geometry == "pad":
        p = np.array(Image.fromarray(p).resize((s, s), Image.BILINEAR))[top:top + h, left:left + w]
    else:
        p = np.array(Image.fromarray(p).resize((w, h), Image.BILINEAR))
    if hflip:
        p = p[:, ::-1]
    return np.ascontiguousarray(p), float(torch.sigmoid(cls)[0, 0])


# ------------------------------------------------------------------ regions and metrics
def box_mask(boxes, shape):
    m = np.zeros(shape, bool)
    h, w = shape
    for b in boxes or []:
        x1, y1, x2, y2 = b
        m[max(0, int(y1 / 100 * h)):max(0, int(y2 / 100 * h)), max(0, int(x1 / 100 * w)):max(0, int(x2 / 100 * w))] = True
    return m


def region_from_prob(p, thr=0.5, min_frac=0.0005):
    from scipy.ndimage import label
    m = p > thr
    lab, n = label(m)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        keep = [i + 1 for i, s in enumerate(sizes) if s >= min_frac * m.size]
        m = np.isin(lab, keep)
    return m


def side(mask):
    if not mask.any():
        return None
    xs = np.nonzero(mask)[1] / mask.shape[1]
    r, l = (xs < 0.5).mean(), (xs >= 0.5).mean()   # patient's right = image left
    return "both" if min(r, l) > 0.2 else "right" if r > l else "left"


def metrics(pred, truth, point):
    inter = (pred & truth).sum()
    union = (pred | truth).sum()
    return {"empty": not pred.any(), "iou": float(inter / union) if union else 0.0,
            "precision": float(inter / pred.sum()) if pred.any() else None,
            "recall": float(inter / truth.sum()) if truth.any() else None,
            "hit": bool(point is not None and truth[min(truth.shape[0] - 1, int(point[1])),
                                                   min(truth.shape[1] - 1, int(point[0]))]),
            "side_ok": (side(pred) == side(truth)) if pred.any() else None, "area": float(pred.mean())}


def mask_to_boxes(reg):
    """Bounding box (0-100 coordinates) of each connected region of a mask."""
    from scipy.ndimage import find_objects, label
    lab, n = label(reg)
    h, w = reg.shape
    return [[sl[1].start / w * 100, sl[0].start / h * 100, sl[1].stop / w * 100, sl[0].stop / h * 100]
            for sl in find_objects(lab) if sl is not None]


def box_point(boxes, shape):
    if not boxes:
        return None
    b = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return ((b[0] + b[2]) / 200 * shape[1], (b[1] + b[3]) / 200 * shape[0])


def boot_mean(v, n=2000):
    v = np.array([x for x in v if x is not None], float)
    if not len(v):
        return None, None, None
    rng = np.random.default_rng(SEED)
    bs = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_diff(a, b, n=2000):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, None, None
    d = np.array([x - y for x, y in pairs], float)
    rng = np.random.default_rng(SEED)
    bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ------------------------------------------------------------------ Phase 3
def validate():
    import pandas as pd
    from anatomical_measurements import auc, auc_ci
    jl = lambda f: [json.loads(l) for l in open(RESULTS / f, encoding="utf-8")]  # noqa: E731
    sets = []
    vcases = {(c["image_id"], c["finding"]): c for c in json.loads(
        (RESULTS / "vindr_grounding_cases.json").read_text(encoding="utf-8"))["cases"]}
    vc = {(r["image_id"], r["finding"]): r["boxes"] for r in jl("vindr_grounding_chexagent.jsonl")
          if r["condition"] == "blind"}
    vm = {(r["image_id"], r["finding"]): r["boxes"] for r in jl("vindr_grounding_maira2.jsonl")
          if r["condition"] == "blind"}
    labels = pd.read_csv(VINDR / "annotations/image_labels_test.csv")
    for finding, models in (("pneumothorax", ("ptx_vasandesh", "ptx_ianpan")), ("consolidation", ("pna_ianpan",))):
        cases = [dict(img=c["png"], truth=c["truth_boxes"], chex=vc.get(k), maira=vm.get(k))
                 for k, c in vcases.items() if k[1] == finding]
        lab_col = next(c for c in labels.columns if c.lower().startswith(finding[:6]))
        per_img = labels.groupby("image_id")[[lab_col]].max()   # negatives: condition absent for every reader
        neg_ids = per_img[per_img[lab_col] == 0].sample(100, random_state=SEED).index.tolist()
        sets.append(dict(name=f"VinDr {finding} (independent)", models=models, cases=cases,
                         neg=[str(VINDR / "png_test" / f"{i}.png") for i in neg_ids], truth_frame="norm"))
    b = pd.read_csv(NIH / "BBox_List_2017.csv")
    b.columns = ["img", "lab", "x", "y", "w", "h"] + list(b.columns[6:])
    gc = {(r["file"], r["label"]): r["boxes"] for r in jl("grounding_chexagent.jsonl")}
    gm = {(r["file"], r["label"]): r["boxes"] for r in jl("grounding_maira2.jsonl")}
    d = pd.read_csv(NIH / "Data_Entry_2017_v2020.csv")
    negs = d[(d["Finding Labels"] == "No Finding") & d["Image Index"].map(lambda n: (NIH / "images" / n).exists())] \
        .sample(100, random_state=SEED)["Image Index"].tolist()
    for lab, models in (("Pneumothorax", ("ptx_vasandesh", "ptx_ianpan")), ("Pneumonia", ("pna_ianpan",))):
        cases = []
        for img, g in b[b.lab == lab].groupby("img"):
            if (img, lab) not in gc or not (NIH / "images" / img).exists():
                continue
            truth = [[r.x / 10.24, r.y / 10.24, (r.x + r.w) / 10.24, (r.y + r.h) / 10.24] for r in g.itertuples()]
            cases.append(dict(img=str(NIH / "images" / img), truth=truth, chex=gc[(img, lab)], maira=gm.get((img, lab))))
        sets.append(dict(name=f"NIH {lab.lower()} (optimistic: possible training overlap)", models=models,
                         cases=cases, neg=[str(NIH / "images" / n) for n in negs]))
    out = {"revisions": REVISIONS, "threshold": 0.5, "sets": []}
    for S in sets:
        rows = {m: [] for m in ("CheXagent", "MAIRA-2") + S["models"] + tuple(f"{m} (as boxes)" for m in S["models"])}
        cls_pos = {m: [] for m in S["models"]}
        for c in S["cases"]:
            g = np.array(Image.open(c["img"]).convert("L"))
            if max(g.shape) > 1600:
                im = Image.fromarray(g)
                im.thumbnail((1600, 1600))
                g = np.array(im)
            truth = box_mask(c["truth"], g.shape)
            for mname, boxes in (("CheXagent", c["chex"]), ("MAIRA-2", c["maira"])):
                rows[mname].append(metrics(box_mask(boxes, g.shape), truth, box_point(boxes, g.shape)))
            for m in S["models"]:
                p, pc = predict(m, g)
                yy, xx = np.unravel_index(p.argmax(), p.shape)
                reg = region_from_prob(p)
                rows[m].append(metrics(reg, truth, (xx, yy)))
                rows[f"{m} (as boxes)"].append(metrics(box_mask(mask_to_boxes(reg), g.shape), truth, (xx, yy)))
                cls_pos[m].append(pc)
        neg_p = {m: [] for m in S["models"]}
        for f in S["neg"]:
            g = np.array(Image.open(f).convert("L"))
            if max(g.shape) > 1600:
                im = Image.fromarray(g)
                im.thumbnail((1600, 1600))
                g = np.array(im)
            for m in S["models"]:
                neg_p[m].append(predict(m, g)[1])
        res = {"name": S["name"], "n": len(S["cases"]), "methods": {}}
        for mname, rs in rows.items():
            r = {k: boot_mean([x[k] if x[k] is None else float(x[k]) for x in rs])
                 for k in ("iou", "hit", "precision", "recall", "side_ok", "area")}
            r["empty_rate"] = float(np.mean([x["empty"] for x in rs])) if rs else None
            if mname in cls_pos:
                r["auc"] = auc(cls_pos[mname], neg_p[mname])
                r["auc_ci"] = auc_ci(cls_pos[mname], neg_p[mname])
            res["methods"][mname] = r
        for m in S["models"] + tuple(f"{m} (as boxes)" for m in S["models"]):
            for ref in ("CheXagent", "MAIRA-2"):
                res.setdefault("paired_iou_diff", {})[f"{m} - {ref}"] = boot_diff(
                    [x["iou"] for x in rows[m]], [x["iou"] for x in rows[ref]])
                res.setdefault("paired_hit_diff", {})[f"{m} - {ref}"] = boot_diff(
                    [float(x["hit"]) for x in rows[m]], [float(x["hit"]) for x in rows[ref]])
        out["sets"].append(res)
        print(json.dumps({"set": res["name"], "n": res["n"], **{k: {kk: (v[kk] if not isinstance(v[kk], tuple) else
                                                                         round(v[kk][0], 3) if v[kk][0] is not None
                                                                         else None)
                                                                    for kk in ("iou", "hit", "side_ok", "empty_rate")}
                                                                for k, v in res["methods"].items()}}), flush=True)
    f = RESULTS / f"condition_grounding_validation_{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {f}")


# ------------------------------------------------------------------ Phase 2 + 4
def report_texts():
    """{image key: (MedGemma report, reference report)} for MIMIC sets and NIH (generated text from the form build)."""
    import pickle
    import pandas as pd
    from pypdf import PdfReader
    from comparison_edv_pdf import parse_page
    H = Path("D:/datasets/held_out_eval_images")
    out = {}
    pdfs = {n: {p["title"]: p for p in (parse_page(x.extract_text()) for x in PdfReader(str(H / n)).pages)}
            for n in ("subset_comparisons_clean_filtered.pdf", "subset_comparisons_16_matched.pdf")}
    for r in json.loads((LOGS / "radiologist_requested_plus_edv_scoring_form_v26.json").read_text(encoding="utf-8")):
        p = pdfs["subset_comparisons_clean_filtered.pdf"][r["title"]]
        out[r["image"]] = (p["generated_text"], p["truth_text"])
    for f in pickle.load(open(LOGS / "mixed_scoring_phase1_v8.pkl", "rb")).values():
        for pg in f:
            if "rec" in pg:
                p = pdfs[pg["source"]][pg["rec"]["title"]]
                out.setdefault(pg["rec"]["image"], (p["generated_text"], p["truth_text"]))
            elif "pair" in pg:
                pd_ = pg["page_draw"]
                out[pg["pair"]["current"]] = (" ".join(t for _, t in pd_.get("generated", [])),
                                              " ".join(t for _, t in pd_.get("truth", [])))
    rep = {json.loads(l)["current"]: json.loads(l)["report"] for l in open(LOGS / "mimic_heldout_trained_reports.jsonl",
                                                                         encoding="utf-8")}
    subset = pd.read_csv(H / "subset.csv")
    for p in json.loads((LOGS / "mimic_heldout_inputs_2.json").read_text(encoding="utf-8")):
        out[p["current"]] = (rep.get(p["current"], "(not found)"), str(subset.report_after[p["row"]]))
    return out


def cases():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    import focused_conditions_report as FC
    import anatomical_measurements as A
    from mimic19_grounding_compare import phrase
    FC.CONDS = tuple(MODELS_FOR)
    entries, page_comments, v26, book, n_wb = FC.restore()
    texts = report_texts()
    maira = json.loads((LOGS / "mimic19_maira2_grounding.json").read_text(encoding="utf-8"))
    items = {}
    for e in entries:
        items.setdefault((e["key"], e["condition"]), []).append(e)
    order = sorted(items, key=lambda k: (k[1] != "pneumothorax", items[k][0]["dataset"], k[0]))
    readers, results = {}, []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = LOGS / f"condition_grounding_gallery_{stamp}.pdf"
    W, H = 11.7, 8.3
    wrap = lambda s, n: "\n".join(textwrap.wrap(" ".join(str(s).split()), n))  # noqa: E731
    with PdfPages(out) as pdf:
        for key in order:
            es = sorted(items[key], key=lambda e: (e["doctor"], e["form"]))
            e0, cond = es[0], key[1]
            ims = {}
            for w_ in ("prior", "current"):
                im = Image.open(e0[w_]).convert("L")
                im.thumbnail((1024, 1024))
                ims[w_] = np.array(im)
            cur = ims["current"]
            seg = A.segment_all(cur)
            lungs = seg["Left Lung"] | seg["Right Lung"]
            from scipy.ndimage import binary_fill_holes
            from skimage.morphology import convex_hull_image
            thorax = convex_hull_image(lungs) if lungs.any() else np.ones_like(lungs)
            mres = {}
            for m in MODELS_FOR[cond]:
                p, pc = predict(m, cur)
                reg = region_from_prob(p)
                stab = {}
                for vname, kw in (("geometry swap", {"geometry": "stretch" if native_geometry(m) == "pad" else "pad"}),
                                  ("gamma 0.8", {"gamma": 0.8}), ("gamma 1.25", {"gamma": 1.25}),
                                  ("horizontal flip (mapped back)", {"hflip": True})):
                    pv, pcv = predict(m, cur, **kw)
                    rv = region_from_prob(pv)
                    u = (reg | rv).sum()
                    stab[vname] = {"dice": float(2 * (reg & rv).sum() / (reg.sum() + rv.sum())) if reg.sum() + rv.sum()
                                   else 1.0, "p": round(pcv, 3), "side": side(rv)}
                from scipy.ndimage import label
                ncomp = label(reg)[1]
                flags = []
                if not reg.any():
                    flags.append("no region at 0.5")
                elif reg.mean() < 0.005:
                    flags.append("small region (<0.5% of image)")
                if reg.mean() > 0.25 or (ncomp >= 3 and side(reg) == "both"):
                    flags.append("diffuse region")
                if reg.any() and (reg & ~thorax).sum() / reg.sum() > 0.5:
                    flags.append("mostly outside the thorax")
                if any(v["dice"] < 0.5 for v in stab.values()) and reg.any():
                    flags.append("unstable under preprocessing (Dice < 0.5)")
                if any(v["side"] and side(reg) and v["side"] != side(reg) for v in stab.values()):
                    flags.append("side changes under preprocessing")
                if any(bool(v["side"]) != bool(side(reg)) for v in stab.values()):
                    flags.append("region appears/disappears under preprocessing")
                mres[m] = dict(p=pc, prob=p, reg=reg, side=side(reg), area=float(reg.mean()), stab=stab, flags=flags,
                               outside=float((reg & ~thorax).sum() / reg.sum()) if reg.any() else None)
            bi = FC.box_image(e0, v26, readers)
            chex = bi[1] if bi and bi[0] == "boxes" else None
            mb = (maira.get(f"{e0['current']}|{phrase(e0['finding'])}") or {}).get("boxes") if e0.get("title19") else None
            sides = {"CheXagent": side(box_mask(chex, cur.shape)) if chex else None,
                     "MAIRA-2": side(box_mask(mb, cur.shape)) if mb else None,
                     **{m: r["side"] for m, r in mres.items()}}
            known = {k: v for k, v in sides.items() if v}
            disagree = len({v for v in known.values() if v != "both"}) > 1
            # ---- page
            fig = plt.figure(figsize=(W, H))
            name = e0["name"].split(" (")[0] if e0["row_type"] == "MedGemma stated" else cond.capitalize()
            fig.text(0.012, 0.988, f"{name} ({e0['row_type']}) - {e0['dataset']}", fontsize=12, weight="bold", va="top")
            fig.text(0.012, 0.962, "; ".join(sorted({f"{FC.FORM_NAME[e['form']]} {e['form_case']}" for e in es})),
                     fontsize=7, va="top")
            panels = [("Prior (original)", ims["prior"], None), ("D. Current - original, no annotations", cur, None)]
            if bi and bi[0] == "embedded":
                panels.append((f"A. CheXagent - exact box image from the {FC.FORM_NAME[e0['form']]} (all page boxes)",
                               np.array(bi[1]), "embedded"))
            else:
                panels.append(("A. CheXagent grounding (box)" if chex else "A. CheXagent: no box for this row",
                               cur, ("box", chex or [], "red")))
            panels.append(("B. MAIRA-2 grounding (box)" if mb else
                           ("B. MAIRA-2: no box returned" if e0.get("title19") else "B. MAIRA-2: not run for this set"),
                           cur, ("box", mb or [], "yellow")))
            for m in MODELS_FOR[cond]:
                panels.append((f"C. {LABEL[m]}; p = {mres[m]['p']:.2f}", cur, ("mask", m)))
            pw = 0.97 / len(panels)
            for j, (cap, arr, spec) in enumerate(panels):
                ax = fig.add_axes([0.012 + j * pw, 0.5, pw - 0.006, 0.42])
                ax.imshow(arr, cmap=None if spec == "embedded" else "gray")
                ax.axis("off")
                if spec and spec != "embedded" and spec[0] == "box":
                    for b in spec[1]:
                        x1, y1, x2, y2 = b[0] / 100 * cur.shape[1], b[1] / 100 * cur.shape[0], \
                            b[2] / 100 * cur.shape[1], b[3] / 100 * cur.shape[0]
                        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=spec[2], lw=1.4))
                elif spec and spec != "embedded" and spec[0] == "mask":
                    r = mres[spec[1]]
                    if r["reg"].any():
                        ax.contour(r["reg"].astype(float), levels=[0.5], colors="orange", linewidths=1.2)
                    else:
                        ax.text(0.5, 0.05, "no region > 0.5", color="yellow", ha="center", transform=ax.transAxes,
                                fontsize=7)
                ax.set_title(wrap(cap, int(pw * 260)), fontsize=6, loc="left")
            # model output summary
            L = [f"Sides (patient): " + ", ".join(f"{k} {v or '-'}" for k, v in sides.items())
                 + ("   ** METHODS DISAGREE ON SIDE **" if disagree else "")]
            for m, r in mres.items():
                L.append(f"{m}: p={r['p']:.2f}; region {100 * r['area']:.1f}% of image, side {r['side'] or '-'}"
                         + (f", {100 * r['outside']:.0f}% outside thorax" if r["outside"] is not None else "")
                         + "; stability Dice " + ", ".join(f"{k} {v['dice']:.2f}" for k, v in r["stab"].items())
                         + ("; FLAGS: " + "; ".join(r["flags"]) if r["flags"] else ""))
            fig.text(0.012, 0.485, "\n".join(sum((textwrap.wrap(s, 190) for s in L), [])), fontsize=6.3, va="top",
                     family="monospace")
            mg, ref = texts.get(e0["key"], texts.get(Path(e0["current"]).name, ("(not found)", "(not found)")))
            fig.text(0.012, 0.39, "MedGemma report", fontsize=7.5, weight="bold", va="top")
            fig.text(0.012, 0.372, wrap(mg, 120)[:1400], fontsize=5.6, va="top", family="monospace")
            fig.text(0.5, 0.39, "Reference report", fontsize=7.5, weight="bold", va="top")
            fig.text(0.5, 0.372, wrap(ref, 120)[:1400], fontsize=5.6, va="top", family="monospace")
            hist = [f"{e['doctor'].replace('Siddarth', 'Siddharth')} ({FC.FORM_NAME[e['form']]} {e['form_case']}): "
                    + "/".join("-" if e["scores"][k] is None else str(e["scores"][k])
                               for k in ("medgemma", "change", "chexagent", "grounding"))
                    + (f' "{e["comment"]}"' if e["comment"] else "") for e in es]
            fig.text(0.012, 0.12, "Historical doctor scores (MedGemma finding / change / CheXagent evidence / "
                     "CheXagent grounding) - context only, not ratings of the new models", fontsize=7, weight="bold",
                     va="top")
            fig.text(0.012, 0.1, "\n".join(sum((textwrap.wrap(s, 170) for s in hist), [])), fontsize=6.2, va="top")
            fig.text(0.012, 0.006, "All localizations are model-generated candidate regions, not confirmed disease "
                     "extent. Each method in its own panel; predictions unmodified (mask = probability > 0.5).",
                     fontsize=5.8)
            pdf.savefig(fig)
            plt.close(fig)
            results.append({"condition": cond, "row_type": e0["row_type"], "dataset": e0["dataset"],
                            "forms": "; ".join(f"{e['doctor']} {e['form']} {e['form_case']}" for e in es),
                            "sides": sides, "side_disagreement": disagree,
                            "chexagent_box": "per-finding" if chex else "embedded image" if bi else "none",
                            "maira": bool(mb),
                            "models": {m: {k: r[k] for k in ("p", "side", "area", "outside", "stab", "flags")}
                                       for m, r in mres.items()}})
            print(cond, e0["dataset"], {m: (round(r["p"], 2), r["side"], r["flags"]) for m, r in mres.items()},
                  "DISAGREE" if disagree else "", flush=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"-> {out} ({len(order)} findings)")


if __name__ == "__main__":
    {"validate": validate, "cases": cases}[sys.argv[1]]()
