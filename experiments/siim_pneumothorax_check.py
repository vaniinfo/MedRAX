"""EXPLORATORY: a SIIM-ACR pneumothorax segmentation model on the pneumothorax findings of the radiologist forms.

Model: vasandesh/pneumothorax-unet-efficientnet-b4 (Hugging Face, MIT; U-Net with EfficientNet-B4 encoder, trained on
SIIM-ACR; card: classification AUC 0.946, segmentation Dice 0.51 on positives). Weights loaded with
torch.load(weights_only=True) from C:/Users/hari/model-weights/siim_ptx_vasandesh (no remote code). Output: a
pneumothorax probability map (model-generated candidate region, not a confirmed boundary) and a classification
probability.

Nothing else changes: model predictions, boxes, Evidence Engine, scores and reports stay as they are. New files only.

    .venv/Scripts/python.exe experiments/siim_pneumothorax_check.py sanity   # NIH boxes + No Finding (optimistic:
                                                                              # SIIM images come from NIH)
    .venv/Scripts/python.exe experiments/siim_pneumothorax_check.py cases    # every pneumothorax row of the forms
"""

import json
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
NIH = Path("D:/datasets/chestxray14")
WEIGHTS = Path("C:/Users/hari/model-weights/siim_ptx_vasandesh/pytorch_model.bin")
SEED = 20261108
_M = None


def model():
    global _M
    if _M is None:
        import torch
        import torch.nn as nn
        import segmentation_models_pytorch as smp

        class PTX(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1)
                self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 1))

            def forward(self, x):
                f = self.unet.encoder(x)
                d = self.unet.decoder(f)
                return self.unet.segmentation_head(d), self.cls_head(d)
        m = PTX()
        m.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True)
        _M = m.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return _M


def predict(gray, norm="imagenet"):
    """(probability map at the image's size, classification probability). Zero-padded to a square, 512 input."""
    import torch
    g = np.asarray(gray, np.float32) / 255.0
    h, w = g.shape
    s = max(h, w)
    pad = np.zeros((s, s), np.float32)
    top, left = (s - h) // 2, (s - w) // 2
    pad[top:top + h, left:left + w] = g
    x = np.array(Image.fromarray(pad).resize((512, 512), Image.BILINEAR))
    x = np.stack([x] * 3)
    if norm == "imagenet":
        x = (x - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None]
    m = model()
    with torch.no_grad():
        seg, cls = m(torch.from_numpy(x[None].astype(np.float32)).to(next(m.parameters()).device))
    prob = torch.sigmoid(seg)[0, 0].cpu().numpy()
    prob = np.array(Image.fromarray(prob).resize((s, s), Image.BILINEAR))[top:top + h, left:left + w]
    return prob, float(torch.sigmoid(cls)[0, 0])


def sanity():
    import pandas as pd
    from anatomical_measurements import auc, auc_ci, prop_ci
    b = pd.read_csv(NIH / "BBox_List_2017.csv")
    b.columns = ["img", "lab", "x", "y", "w", "h"] + list(b.columns[6:])
    b = b[(b.lab == "Pneumothorax") & b.img.map(lambda n: (NIH / "images" / n).exists())]
    d = pd.read_csv(NIH / "Data_Entry_2017_v2020.csv")
    neg = d[(d["Finding Labels"] == "No Finding") & d["Image Index"].map(lambda n: (NIH / "images" / n).exists())] \
        .sample(100, random_state=SEED)["Image Index"].tolist()
    out = {}
    for norm in ("imagenet", "plain"):
        pos_p, hits, inbox, side_ok = [], [], [], []
        for _, r in b.iterrows():
            img = np.array(Image.open(NIH / "images" / r.img).convert("L"))
            if img.shape != (1024, 1024):
                continue
            prob, p = predict(img, norm)
            pos_p.append(p)
            yy, xx = np.unravel_index(prob.argmax(), prob.shape)
            hits.append(r.x <= xx <= r.x + r.w and r.y <= yy <= r.y + r.h)
            m = prob > 0.5
            box = np.zeros_like(m)
            box[int(r.y):int(r.y + r.h), int(r.x):int(r.x + r.w)] = True
            inbox.append(float((m & box).sum() / m.sum()) if m.sum() else None)
            if m.sum():
                side_ok.append((np.nonzero(m)[1].mean() < 512) == (r.x + r.w / 2 < 512))
        neg_p = [predict(np.array(Image.open(NIH / "images" / n).convert("L")), norm)[1] for n in neg]
        a = auc(pos_p, neg_p)
        k = int(sum(hits))
        out[norm] = {"n_pos": len(pos_p), "n_neg": len(neg_p), "auc": a, "auc_ci": auc_ci(pos_p, neg_p),
                     "peak_in_box": k / len(hits), "peak_in_box_ci": prop_ci(k, len(hits)),
                     "mask_share_in_box_median": float(np.median([v for v in inbox if v is not None]))
                     if any(v is not None for v in inbox) else None,
                     "mask_present_pos": float(np.mean([v is not None for v in inbox])),
                     "side_correct": float(np.mean(side_ok)) if side_ok else None,
                     "median_p_pos": float(np.median(pos_p)), "median_p_neg": float(np.median(neg_p))}
        print(norm, json.dumps(out[norm]), flush=True)
    f = RESULTS / f"siim_pneumothorax_sanity_{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({"note": "NIH pneumothorax boxes vs 100 No Finding; optimistic because SIIM-ACR images "
                             "are drawn from NIH (possible training overlap).", **out}, indent=1), encoding="utf-8")
    print(f"-> {f}")


def cases():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    import condition_aware_grounding as G
    import focused_conditions_report as FC
    import anatomical_measurements as A
    sn = json.loads(sorted(RESULTS.glob("siim_pneumothorax_sanity_*.json"))[-1].read_text(encoding="utf-8"))
    norm = max(("imagenet", "plain"), key=lambda k: sn[k]["auc"])
    FC.CONDS = ("pneumothorax",)
    entries, page_comments, v26, book, n_wb = FC.restore()
    maira = json.loads((LOGS / "mimic19_maira2_grounding.json").read_text(encoding="utf-8"))
    items = {}
    for e in entries:
        items.setdefault((e["key"], e["row_type"]), []).append(e)
    order = sorted(items, key=lambda k: (items[k][0]["dataset"], k[0]))
    readers, rows = {}, []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = LOGS / f"siim_pneumothorax_cases_{stamp}.pdf"
    W, H = 11.7, 8.3
    wrap = lambda s, n: "\n".join(textwrap.wrap(" ".join(str(s).split()), n))  # noqa: E731
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(W, H))
        s0 = sn[norm]
        fig.text(0.04, 0.95, "\n".join([
            "SIIM-ACR PNEUMOTHORAX SEGMENTATION MODEL ON THE RADIOLOGIST FORMS (exploratory)", "",
            "Model: vasandesh/pneumothorax-unet-efficientnet-b4 (U-Net, EfficientNet-B4, trained on SIIM-ACR; MIT).",
            "Card: classification AUC 0.946, segmentation Dice 0.51 on positive cases.",
            f"Sanity check on NIH ({norm} normalization): {s0['n_pos']} pneumothorax box images vs {s0['n_neg']} No "
            f"Finding:", f"  classification AUC {s0['auc']:.2f} ({s0['auc_ci'][0]:.2f}-{s0['auc_ci'][1]:.2f}); "
            f"probability-map peak inside the radiologist box {s0['peak_in_box']:.0%};",
            f"  predicted region on the box's side {s0['side_correct']:.0%} (of images with a region). Optimistic: "
            "SIIM images come from NIH.", "",
            "Each page: prior and current with the model's pneumothorax probability (contour at 0.5, orange; "
            "model-generated", "candidate region, not a confirmed boundary), the original CheXagent box (red), MAIRA-2 "
            "(yellow, 19-case set only),", "the classification probability, MedGemma's claim, the reference and the "
            "doctors' scores and comments.", "", "Nothing is changed in the models, boxes, Evidence Engine, scores or "
            "reports. MIMIC-CXR content: do not distribute."]), fontsize=9, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        for key in order:
            es = sorted(items[key], key=lambda e: (e["doctor"], e["form"]))
            e0 = es[0]
            ims, res = {}, {}
            for w_ in ("prior", "current"):
                im = Image.open(e0[w_]).convert("L")
                im.thumbnail((1024, 1024))
                ims[w_] = np.array(im)
                res[w_] = predict(ims[w_], norm)
            prob, pc = res["current"]
            m = prob > 0.5
            masks = A.segment_all(ims["current"])
            side = G.side_of(m, m.shape) if m.sum() > 50 else None
            fig = plt.figure(figsize=(W, H))
            name = e0["name"].split(" (")[0] if e0["row_type"] == "MedGemma stated" else "Pneumothorax"
            fig.text(0.015, 0.985, f"{name} - {e0['row_type']}", fontsize=13, weight="bold", va="top")
            fig.text(0.015, 0.955, f"{e0['dataset']}; " + "; ".join(sorted({f"{FC.FORM_NAME[e['form']]} "
                     f"{e['form_case']}" for e in es})), fontsize=7.5, va="top")
            for j, w_ in enumerate(("prior", "current")):
                ax = fig.add_axes([0.015 + j * 0.245, 0.33, 0.24, 0.58])
                ax.imshow(ims[w_], cmap="gray")
                ax.axis("off")
                pr, pp = res[w_]
                if (pr > 0.5).any():
                    ax.contour(pr, levels=[0.5], colors="orange", linewidths=1.3)
                ax.set_title(f"{w_.capitalize()}: SIIM model p(pneumothorax) = {pp:.2f}", fontsize=8, loc="left")
            ax = fig.add_axes([0.505, 0.33, 0.24, 0.58])
            ax.imshow(ims["current"], cmap="gray")
            ax.axis("off")
            bi = FC.box_image(e0, v26, readers)
            if bi and bi[0] == "boxes":
                for b in bi[1]:
                    x1, y1, x2, y2 = G.box_px(b, ims["current"].shape)
                    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
            from mimic19_grounding_compare import phrase
            mb = (maira.get(f"{e0['current']}|{phrase(e0['finding'])}") or {}).get("boxes") or []
            for b in mb:
                x1, y1, x2, y2 = G.box_px(b, ims["current"].shape)
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="yellow", lw=1.3, ls="--"))
            if (prob > 0.5).any():
                ax.contour(prob, levels=[0.5], colors="orange", linewidths=1.0)
            ax.set_title(wrap("Current: CheXagent box (red)" + (" | MAIRA-2 (yellow)" if mb else "") +
                              " | SIIM region (orange)" + ("" if bi and bi[0] == "boxes" else
                                                            " - CheXagent box not recorded for this row"), 52),
                         fontsize=7, loc="left")
            lungs = masks["Left Lung"] | masks["Right Lung"]
            inside = float((m & lungs).sum() / m.sum()) if m.sum() else None
            facts = [f"MedGemma: {e0['name'] if e0['row_type'] == 'MedGemma stated' else 'Not detected'}",
                     f"MedGemma interval change: {e0['change'] or '-'}",
                     f"CheXagent shown: {e0['chexagent_shown']}" + (f" (p_present {e0['p_present']:.2f})"
                                                                    if e0.get('p_present') else ""),
                     f"Reference: {e0['reference'] or '-'}" + (f' - "{e0["reference_quote"]}"'
                                                              if e0["reference_quote"] else ""), "",
                     f"SIIM model, current: p = {pc:.2f}; region at 0.5: " +
                     (f"{100 * m.mean():.1f}% of the image, side {side}, {100 * inside:.0f}% inside the segmented "
                      "lungs" if m.sum() > 50 else "none"),
                     f"SIIM model, prior: p = {res['prior'][1]:.2f}"]
            fig.text(0.76, 0.91, "Finding and model outputs", fontsize=8.5, weight="bold", va="top")
            fig.text(0.76, 0.89, "\n".join(sum((textwrap.wrap(s, 46) or [""] for s in facts), [])), fontsize=6.6,
                     va="top")
            fig.text(0.015, 0.3, "Previous radiologist evaluation (original scores 1-5: MedGemma finding / interval "
                     "change / CheXagent evidence / grounding; '-' not asked)", fontsize=8.5, weight="bold", va="top")
            L = []
            for e in es:
                sc = "/".join("-" if e["scores"][k] is None else str(e["scores"][k])
                              for k in ("medgemma", "change", "chexagent", "grounding"))
                L.append(f"{e['doctor'].replace('Siddarth', 'Siddharth')} ({FC.FORM_NAME[e['form']]} {e['form_case']}): "
                         f"{sc}" + (f'  comment: "{e["comment"]}"' if e["comment"] else ""))
            fig.text(0.015, 0.27, "\n".join(sum((textwrap.wrap(s, 150) for s in L), [])), fontsize=7, va="top")
            fig.text(0.015, 0.012, "Orange = SIIM-ACR model probability contour (model-generated candidate region, not "
                     "a confirmed boundary). Red = CheXagent, yellow dashed = MAIRA-2 (model-generated). Exploratory; "
                     "not a diagnosis.", fontsize=6.3)
            pdf.savefig(fig)
            plt.close(fig)
            rows.append({"dataset": e0["dataset"], "forms": "; ".join(f"{e['doctor']} {e['form']} {e['form_case']}"
                                                                        for e in es),
                         "row_type": e0["row_type"], "medgemma": e0["name"], "reference": e0["reference"],
                         "siim_p_current": round(pc, 3), "siim_p_prior": round(res["prior"][1], 3),
                         "siim_side": side, "region_pct": round(100 * float(m.mean()), 2)})
    (out.with_suffix(".json")).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"-> {out} ({len(order)} findings, normalization {norm})")
    for r in rows:
        print(r)


if __name__ == "__main__":
    {"sanity": sanity, "cases": cases}[sys.argv[1]]()
