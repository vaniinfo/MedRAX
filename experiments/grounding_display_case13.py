"""EXPLORATORY: ways to show WHERE a finding is, tried on one case (MIMIC-CXR Comparison 13 of 19, "low lung
volumes"), after the radiologists' feedback that boxes are too small for diffuse findings. Nothing here changes the
models, the Evidence Engine or any form; outputs go to logs/ (MIMIC-CXR content, PhysioNet DUA).

Displays, all on the current image:
 A. CheXagent's box as shown to the doctors (from the form's record).
 B. Anatomy: both lungs and the heart outlined by a segmentation model (torchxrayvision PSPNet, ChestX-Det).
 C. Zone snapping: each CheXagent box mapped to the upper / middle / lower third of the segmented lung it falls in,
    and the whole zone shaded.
 D. Occlusion map ("PET-like"): the report-blinded question "Is there low lung volumes?" asked again with each
    patch of the image blurred; colour = how much the Yes probability drops (where CheXagent's answer comes from).
 E. Attention map: CheXagent's attention from the answer position to its 32 x 32 image tokens (layers 16-31).
 F. Box density: CheXagent asked to locate the finding under six wordings; the boxes summed into a heat map.
 G. MAIRA-2 phrase grounding of the same phrase (second model).
 H. Measurements: lung field height and area from the segmentation, current vs prior, and the cardiothoracic ratio.

    .venv/Scripts/python.exe experiments/grounding_display_case13.py
"""

import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
LOGS = ROOT / "logs"
TITLE, FINDING = "Comparison 13 of 19", "low lung volumes"
PHRASINGS = ["low lung volumes", "low lung volume", "decreased lung volumes", "hypoinflated lungs",
             "poor inspiration", "small lungs"]
CACHE = LOGS / "grounding_display_case13.json"


def load_case():
    rec = [r for r in json.loads((LOGS / "radiologist_requested_plus_edv_scoring_form_v26.json").read_text(
        encoding="utf-8")) if r["title"].startswith(TITLE)][0]
    from pypdf import PdfReader
    from comparison_edv_pdf import parse_page
    page = [parse_page(p.extract_text()) for p in
            PdfReader("D:/datasets/held_out_eval_images/subset_comparisons_clean_filtered.pdf").pages]
    page = [p for p in page if p["title"].startswith(TITLE)][0]
    return rec, page


def segment(path):
    """{organ: mask} at the image's own size; the image is padded to a square first (no cropping)."""
    import torch
    import torchxrayvision as xrv
    img = np.array(Image.open(path).convert("L")).astype(np.float32)
    h, w = img.shape
    side = max(h, w)
    pad = np.zeros((side, side), np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    pad[top:top + h, left:left + w] = img
    x = xrv.datasets.normalize(pad, 255)[None]
    x = torch.from_numpy(xrv.datasets.XRayResizer(512)(x))[None]
    model = xrv.baseline_models.chestx_det.PSPNet().eval()
    with torch.no_grad():
        out = torch.sigmoid(model(x))[0].numpy()
    names = {"Left Lung": 4, "Right Lung": 5, "Heart": 8}
    masks = {}
    for k, i in names.items():
        m = np.array(Image.fromarray((out[i] > 0.5).astype(np.uint8) * 255).resize((side, side), Image.NEAREST)) > 0
        masks[k] = m[top:top + h, left:left + w]
    return masks


def measures(masks):
    """Lung height, area and cardiothoracic ratio (pixels / fractions); descriptive, not validated."""
    lungs = masks["Left Lung"] | masks["Right Lung"]
    ys, xs = np.nonzero(lungs)
    out = {"lung_area_px": int(lungs.sum()), "image_area_px": lungs.size,
           "lung_area_fraction": float(lungs.sum() / lungs.size)}
    for k in ("Right Lung", "Left Lung"):
        y, x = np.nonzero(masks[k])
        out[f"{k} height_px"] = int(y.max() - y.min()) if y.size else 0
    out["thorax_width_px"] = int(xs.max() - xs.min()) if xs.size else 0
    hy, hx = np.nonzero(masks["Heart"])
    out["heart_width_px"] = int(hx.max() - hx.min()) if hx.size else 0
    out["cardiothoracic_ratio"] = out["heart_width_px"] / out["thorax_width_px"] if out["thorax_width_px"] else None
    out["lung_height_to_width"] = (max(out["Right Lung height_px"], out["Left Lung height_px"]) /
                                   out["thorax_width_px"]) if out["thorax_width_px"] else None
    return out


def chexagent_parts(cur, small):
    """Occlusion map, attention map, box density, and the baseline answer (one CheXagent session)."""
    import torch
    from medrax.tools.xray_vqa import XRayVQATool
    t = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    q = f"Is there {FINDING}?"
    base_text, base = t._generate_response([small], q, 8)
    p0 = base["p_yes"]

    # D. occlusion: blur one patch at a time (patch = 1/8 of each side, stride 1/16)
    im = Image.open(small).convert("RGB")
    W, H = im.size
    blurred = im.filter(ImageFilter.GaussianBlur(radius=W / 25))
    n, work = 16, Path(tempfile.mkdtemp(prefix="occl_", dir=ROOT / "temp"))
    drop, count = np.zeros((n, n)), np.zeros((n, n))
    pw, ph = W // 8, H // 8
    for gy in range(n - 1):
        for gx in range(n - 1):
            x0, y0 = gx * W // n, gy * H // n
            test = im.copy()
            test.paste(blurred.crop((x0, y0, x0 + pw, y0 + ph)), (x0, y0))
            f = work / f"o_{gy}_{gx}.png"
            test.save(f)
            _, yn = t._generate_response([str(f)], q, 8)
            p = yn["p_yes"] if yn else p0
            drop[gy:gy + 2, gx:gx + 2] += p0 - p
            count[gy:gy + 2, gx:gx + 2] += 1
    occlusion = drop / np.maximum(count, 1)

    # E. attention from the answer position to the 1,024 image tokens (32 x 32 grid over the squashed 512 x 512 input)
    qq = t.tokenizer.from_list_format([{"image": small}, {"text": q}])
    ids = t.tokenizer.apply_chat_template([{"from": "system", "value": "You are a helpful assistant."},
                                           {"from": "human", "value": qq}], add_generation_prompt=True,
                                          return_tensors="pt").to("cuda")
    s = int((ids[0] == t.tokenizer.img_start_id).nonzero())
    with torch.no_grad():
        att = t.model(ids, output_attentions=True).attentions
    a = torch.stack([x[0, :, -1, s + 1:s + 1025].float().mean(0) for x in att[16:]]).mean(0)
    attention = a.reshape(32, 32).cpu().numpy()

    # F. box density over wordings
    from medrax.tools.medgemma import describe_location
    density, located = np.zeros((100, 100)), []
    for ph_ in PHRASINGS:
        proof, _ = t.locate_findings(small, [ph_])
        for qd in proof:
            for b in qd["boxes"] if qd["localized"] else []:
                x1, y1, x2, y2 = [int(v) for v in b["box_normalized"]]
                density[y1:y2, x1:x2] += 1
                located.append((ph_, b["box_normalized"], describe_location(b["box_normalized"])))
    del t
    torch.cuda.empty_cache()
    return {"answer": base_text, "p_present": p0, "occlusion": occlusion.tolist(), "attention": attention.tolist(),
            "density": density.tolist(), "located": located}


def maira_part(small):
    import torch
    from vindr_grounding_amendment import maira_locator
    loc = maira_locator()
    out = {ph_: loc(small, ph_) for ph_ in ("low lung volumes", "lungs")}
    torch.cuda.empty_cache()
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    from scipy.ndimage import zoom

    rec, page = load_case()
    cur, pri = rec["image"], rec["image"].replace("current.jpg", "prior.jpg")
    work = Path(tempfile.mkdtemp(prefix="case13_", dir=ROOT / "temp"))
    small = work / "current_1024.png"
    im = Image.open(cur).convert("L")
    im.thumbnail((1024, 1024))
    im.save(small)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if "chexagent" not in cache:
        t0 = time.time()
        cache["chexagent"] = chexagent_parts(cur, str(small))
        cache["chexagent_seconds"] = round(time.time() - t0)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    if "maira" not in cache:
        cache["maira"] = maira_part(str(small))
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    cx = cache["chexagent"]
    masks_c, masks_p = segment(str(small)), segment(pri)
    m_c, m_p = measures(masks_c), measures(masks_p)

    arr = np.array(im)
    H, W = arr.shape
    boxes = [b["box_normalized"] for v in rec["visual_proof"] if v["finding"] == FINDING for b in v.get("boxes", [])]

    def base(ax, title):
        ax.imshow(arr, cmap="gray")
        ax.set_title(title, fontsize=9, loc="left")
        ax.axis("off")

    def heat(ax, m, title, cmap="jet"):
        base(ax, title)
        hm = np.clip(np.array(m, float), 0, None)
        if hm.max() > 0:
            hm = hm / hm.max()
        ax.imshow(zoom(hm, (H / hm.shape[0], W / hm.shape[1]), order=1), cmap=cmap, alpha=0.45,
                  vmin=0, vmax=1)

    out = LOGS / f"grounding_display_case13_{time.strftime('%Y%m%d-%H%M%S')}.pdf"
    with PdfPages(out) as pdf:
        # page 1: the case
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle(f"{TITLE}: how to show '{FINDING}' - one-case exploration (not a validated result)",
                     fontsize=12, x=0.02, ha="left")
        for k, (p_, lab) in enumerate(((pri, "Prior"), (cur, "Current"))):
            ax = fig.add_axes([0.02 + k * 0.25, 0.30, 0.24, 0.6])
            ax.imshow(Image.open(p_).convert("L"), cmap="gray")
            ax.set_title(lab, fontsize=10)
            ax.axis("off")
        doc = ("Dr Franco and Dr Siddarth (19-case form): MedGemma 'Low lung volumes' 1/1 (both say NOT present); "
               "CheXagent 'Not agree' 5/5; CheXagent box 1/1. The reference report does not mention low lung volumes. "
               "So in this case the finding is probably absent: the displays below are compared for how they "
               "communicate location (and absence), not to prove a finding.")
        txt = (f"MedGemma report (trained): {page['generated_text'][:700]}\n\nReference report: "
               f"{page['truth_text'][:700]}\n\n{doc}\n\nCheXagent, report-blinded 'Is there {FINDING}?': "
               f"answer {cx['answer']!r}, p_present {cx['p_present']:.3f}. 'Low lung volumes' is not one of the 13 "
               "calibrated conditions, so no cut-off applies (shown as Indeterminate / Not agree on the form).")
        fig.text(0.53, 0.9, txt, fontsize=7.5, va="top", wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # page 2: the eight displays
        fig, axs = plt.subplots(2, 4, figsize=(16.5, 9.5))
        fig.suptitle(f"{TITLE} - '{FINDING}': eight ways to show location (current image)", fontsize=12, x=0.01,
                     ha="left")
        ax = axs[0, 0]
        base(ax, "A. CheXagent box (as shown to doctors)\nboth doctors: box 1/5")
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(Rectangle((x1 * W / 100, y1 * H / 100), (x2 - x1) * W / 100, (y2 - y1) * H / 100,
                                   fill=False, ec="red", lw=2))
        ax = axs[0, 1]
        base(ax, "B. Anatomy: both lungs (blue) and heart (orange)\nsegmentation model")
        ax.imshow(np.ma.masked_where(~(masks_c["Left Lung"] | masks_c["Right Lung"]), np.ones_like(arr)),
                  cmap="winter", alpha=0.35)
        ax.imshow(np.ma.masked_where(~masks_c["Heart"], np.ones_like(arr)), cmap="autumn", alpha=0.35)
        ax = axs[0, 2]
        base(ax, "C. Zone snapping: CheXagent boxes ->\nwhole lung zone shaded")
        zone_img = np.zeros_like(arr, bool)
        zones = []
        for x1, y1, x2, y2 in boxes:
            cxp, cyp = (x1 + x2) / 200 * W, (y1 + y2) / 200 * H
            for lung in ("Right Lung", "Left Lung"):
                y, x = np.nonzero(masks_c[lung])
                if x.size and x.min() <= cxp <= x.max():
                    third = np.clip(int(3 * (cyp - y.min()) / max(1, y.max() - y.min())), 0, 2)
                    lo = y.min() + third * (y.max() - y.min()) / 3
                    hi = y.min() + (third + 1) * (y.max() - y.min()) / 3
                    rows = (np.arange(H)[:, None] >= lo) & (np.arange(H)[:, None] < hi)
                    zone_img |= masks_c[lung] & rows
                    zones.append(f"{lung.split()[0].lower()} {['upper', 'middle', 'lower'][third]} zone")
        ax.imshow(np.ma.masked_where(~zone_img, np.ones_like(arr)), cmap="autumn", alpha=0.4)
        ax.text(5, H - 10, "; ".join(dict.fromkeys(zones)) or "-", color="yellow", fontsize=8)
        heat(axs[0, 3], cx["occlusion"], f"D. Occlusion map ('PET-like'): where blurring\nlowers CheXagent's Yes "
                                         f"(base p={cx['p_present']:.2f}; max drop {np.max(cx['occlusion']):.3f})")
        heat(axs[1, 0], cx["attention"], "E. CheXagent attention to image patches\n(answer token, layers 16-31)")
        heat(axs[1, 1], cx["density"], f"F. Box density: CheXagent located the finding\nunder {len(PHRASINGS)} wordings "
                                       f"({len(cx['located'])} boxes)", cmap="hot")
        ax = axs[1, 2]
        base(ax, "G. MAIRA-2 phrase grounding (second model)\nred: 'low lung volumes'; cyan: 'lungs'")
        for ph_, col in (("low lung volumes", "red"), ("lungs", "cyan")):
            for x1, y1, x2, y2 in cache["maira"][ph_]["boxes"]:
                ax.add_patch(Rectangle((x1 * W / 100, y1 * H / 100), (x2 - x1) * W / 100, (y2 - y1) * H / 100,
                                       fill=False, ec=col, lw=2))
        ax = axs[1, 3]
        ax.axis("off")
        ax.set_title("H. Measurements (segmentation; descriptive)", fontsize=9, loc="left")
        ratio = m_c["lung_area_fraction"] / m_p["lung_area_fraction"] if m_p["lung_area_fraction"] else float("nan")
        ax.text(0, 0.95, "\n".join([
            f"Lung area / image: current {m_c['lung_area_fraction']:.2f}, prior {m_p['lung_area_fraction']:.2f}",
            f"   -> current is {100 * (ratio - 1):+.0f}% vs prior (positioning differs)",
            f"Lung height / thorax width: current {m_c['lung_height_to_width']:.2f}, "
            f"prior {m_p['lung_height_to_width']:.2f}",
            f"Cardiothoracic ratio (heart / thorax width): current {m_c['cardiothoracic_ratio']:.2f}, "
            f"prior {m_p['cardiothoracic_ratio']:.2f}",
            "",
            "Low lung volumes is judged from inspiration:",
            "diaphragm level / number of ribs above it,",
            "not a local abnormality. A whole-lung display (B)",
            "plus a height or rib-count measure fits it better",
            "than a box. Thresholds here are NOT validated."]), fontsize=8, va="top", family="monospace")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        # page 3: comparison table
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("Which display fits which kind of finding (from this one case; to be tested on more cases "
                     "and by the radiologists)", fontsize=11, x=0.02, ha="left")
        rows = [
            ["A. Single box", "CheXagent", "Small boxes; both doctors rated it 1", "Focal findings (nodule, mass, "
             "pneumothorax apex, devices)", "Instant"],
            ["B. Anatomy outline", "Segmentation (PSPNet)", "Whole lungs / heart, no model guess", "Global findings "
             "(low lung volumes, hyperinflation, edema); cardiomegaly (heart)", "~1 s"],
            ["C. Zone snapping", "CheXagent + segmentation", "Shades the lung zone the box falls in", "Regional "
             "findings (effusion, atelectasis, consolidation); makes side obvious", "~1 s"],
            ["D. Occlusion map", "CheXagent, report-blinded", "Shows which regions drive the Yes answer; uses the "
             "same question and score", "Explaining CheXagent's evidence; slow", f"~{cache['chexagent_seconds']} s"],
            ["E. Attention map", "CheXagent internals", "Fast heat map; attention is known to be noisy", "Quick "
             "overview; needs validation", "~1 s"],
            ["F. Box density", "CheXagent, 6 wordings", "Soft region from many boxes; robust to one bad box",
             "Regional findings", "~10 s"],
            ["G. Second model", "MAIRA-2", "Independent location; better on cardiomegaly in our grounding test",
             "Cross-check; route by condition", "~10 s"],
            ["H. Measurement", "Segmentation", "A number (lung height, area vs prior, CTR) instead of a picture",
             "Low lung volumes, cardiomegaly, change from prior", "~1 s"],
        ]
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.85])
        ax.axis("off")
        tb = ax.table(cellText=rows, colLabels=["Display", "Source", "What it shows", "Best for", "Time"],
                      loc="upper left", colWidths=[0.14, 0.16, 0.30, 0.32, 0.08], cellLoc="left")
        tb.auto_set_font_size(False)
        tb.set_fontsize(8)
        tb.scale(1, 2.6)
        pdf.savefig(fig)
        plt.close(fig)
    cache["measures"] = {"current": m_c, "prior": m_p}
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    print(f"p_present {cx['p_present']:.3f}; occlusion max drop {np.max(cx['occlusion']):.3f}; "
          f"boxes located {len(cx['located'])}; MAIRA-2 boxes "
          f"{ {k: len(v['boxes']) for k, v in cache['maira'].items()} }; measures current {m_c} prior {m_p}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
