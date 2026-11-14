"""Pilot (idea 2): BioViL-T text-guided similarity heat maps on a few findings from the regional/focal PDFs.

Exploratory only - not validated, not shown to radiologists, no claim of diagnostic accuracy.
  Model: microsoft/BiomedVLP-BioViL-T (MIT), revision 692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23, in
         C:/Users/hari/model-weights/biovil_t. Image weights MD5 a83080e2f23aa584a4f2b24c39b1bb64 (= the value
         published in hi-ml-multimodal), loaded with torch.load(weights_only=True); text weights from safetensors.
  Code:  hi-ml-multimodal 0.2.2 (MIT, reviewed), loaded by path from C:/Users/hari/model-weights/himl_src/x;
         no trust_remote_code. SimpleITK (only used there for NIfTI) is stubbed.
  Prompt: the condition name only ('pleural effusion', ...): report-blinded, as for MAIRA-2. MedGemma's text or
          side and the reference report are never used to steer the map.
  Map:   cosine similarity between each image patch embedding (14x14 grid, Gaussian-smoothed, sigma 1.5) and the
         prompt embedding; resize 512 + centre crop 448, as in the library. Pixels outside the crop are NaN (not
         analysed). Display: RdBu_r, fixed -1..+1, alpha 0.5 (library convention).
No existing prediction, box, score, report or file is changed. MIMIC-CXR content -> logs/ only.

    .venv/Scripts/python.exe experiments/biovil_t_heatmap_pilot.py
"""

import json
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = ROOT / "logs"
WEIGHTS = Path("C:/Users/hari/model-weights/biovil_t")
HIML = Path("C:/Users/hari/model-weights/himl_src/x")
REVISION = "692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23"
IMG_MD5 = "a83080e2f23aa584a4f2b24c39b1bb64"
# (report, v1 page) - chosen for: abnormality explicitly present in the reference (mostly), a mix of Group A
# (both boxes) and Group B (MAIRA-2 only), and one negative control (reference: effusion explicitly absent)
PICK = [("regional", 5), ("regional", 8), ("regional", 11), ("regional", 35), ("regional", 56),
        ("regional", 72), ("regional", 46), ("focal", 5), ("focal", 8)]

sys.path[:0] = [str(ROOT), str(HERE)]


def load_engine(device):
    import hashlib
    assert hashlib.md5((WEIGHTS / "biovil_t_image_model_proj_size_128.pt").read_bytes()).hexdigest() == IMG_MD5
    sys.modules.setdefault("SimpleITK", types.ModuleType("SimpleITK"))  # only used for .nii.gz in the library
    sys.path.insert(0, str(HIML))
    from health_multimodal.image import ImageInferenceEngine, ImageModel
    from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
    from health_multimodal.image.model.types import ImageEncoderType
    from health_multimodal.text import TextInferenceEngine
    from health_multimodal.text.model import CXRBertModel, CXRBertTokenizer
    from health_multimodal.vlp import ImageTextInferenceEngine
    im = ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE, joint_feature_size=128,
                    pretrained_model_path=None)
    sd = torch.load(WEIGHTS / "biovil_t_image_model_proj_size_128.pt", map_location="cpu", weights_only=True)
    im.load_state_dict(sd)
    tok = CXRBertTokenizer.from_pretrained(str(WEIGHTS))
    txt = CXRBertModel.from_pretrained(str(WEIGHTS), use_safetensors=True)
    eng = ImageTextInferenceEngine(
        image_inference_engine=ImageInferenceEngine(im, create_chest_xray_transform_for_inference(512, 448)),
        text_inference_engine=TextInferenceEngine(tokenizer=tok, text_model=txt))
    eng.to(device)
    return eng


def summarize(sim, boxes_by_model):
    """Descriptive only (no ground truth): peak position, image side, share of the top-10% map inside each box."""
    h, w = sim.shape
    v = np.nan_to_num(sim, nan=-np.inf)
    py, px = np.unravel_index(np.argmax(v), v.shape)
    fin = sim[np.isfinite(sim)]
    top = np.isfinite(sim) & (sim >= np.quantile(fin, 0.9))
    out = {"max": float(fin.max()), "min": float(fin.min()), "peak_xy_pct": [round(100 * px / w, 1),
                                                                             round(100 * py / h, 1)],
           "peak_side": "patient right" if px < 0.45 * w else "patient left" if px > 0.55 * w else "midline",
           "top10_side": {"patient right": float(top[:, : w // 2].sum() / top.sum()),
                          "patient left": float(top[:, w // 2:].sum() / top.sum())}}
    for name, boxes in boxes_by_model.items():
        if not boxes:
            continue
        m = np.zeros_like(top)
        for b in boxes:
            m[int(b[1] / 100 * h):int(np.ceil(b[3] / 100 * h)), int(b[0] / 100 * w):int(np.ceil(b[2] / 100 * w))] = 1
        out[f"top10_in_{name}_box"] = float((top & m).sum() / top.sum())
        out[f"peak_in_{name}_box"] = bool(m[py, px])
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    import regional_focal_reports as R
    import regional_focal_reports_v2 as R2
    import reference_sentence_audit as RA
    import visualization_evaluation_report as VE

    st = time.strftime("%Y%m%d-%H%M%S")
    F, *_ = R.build()
    pages = R2.v1_page_numbers(F)
    sel = {pages[id(f)]: f for f in F}
    texts = VE.reference_texts()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eng = load_engine(device)

    W, H = 11.7, 8.3
    wrap = lambda s, n: "\n".join(textwrap.wrap(" ".join(str(s).split()), n))  # noqa: E731
    results = []
    out = LOGS / f"biovil_t_heatmap_pilot_{st}.pdf"
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(W, H))
        fig.text(0.04, 0.95, "\n".join([
            "PILOT - BioViL-T TEXT-GUIDED HEAT MAPS (idea 2)  -  exploratory, not validated, not for radiologists",
            "", wrap("Exploratory only: not validated, not shown to radiologists, no claim of diagnostic accuracy. "
                     "No existing prediction, box, score or report is changed.", 110), "",
            f"Model microsoft/BiomedVLP-BioViL-T @ {REVISION[:12]} (MIT); code hi-ml-multimodal 0.2.2 by path.",
            "Prompt = condition name only (report-blinded). Map = patch-to-prompt cosine similarity, -1..+1:",
            "  red = image region resembles the prompt; blue = does not. Grey margins = outside the model's centre crop.",
            "The map shows where the model's features match the words; it is NOT a disease boundary, NOT a presence",
            "verdict, and was not checked against radiologist-drawn boxes (required before any radiologist use).",
            "CheXagent / MAIRA-2 boxes are shown for orientation only; they are not ground truth either.",
            "Patient's right is on the image left.", "",
            f"{len(PICK)} findings: " + "; ".join(f"{a} p. {b}" for a, b in PICK), "",
            "MIMIC-CXR content (PhysioNet DUA): do not distribute."]), fontsize=8.6, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        for rep, pg in PICK:
            f = sel[(rep, pg)]
            e0, cond = f["e0"], f["key"][1]
            sim = eng.get_similarity_map_from_raw_data(Path(e0["current"]), cond, interpolation="bilinear")
            cur = np.array(Image.open(e0["current"]).convert("L"))
            assert sim.shape == cur.shape, (sim.shape, cur.shape)
            boxes = {"CheXagent": f["cx_coords"] or [], "MAIRA-2": f["mx"] or []}
            s = summarize(sim, boxes)
            if e0["dataset"].startswith("NIH"):
                ref = None
                ref_label = ("NIH label present (text-mined; no report text)" if e0["reference"] == "Yes" else
                             "NIH label not assigned (text-mined; no report text)")
            else:
                ref = RA.classify(cond, texts.get(e0["key"], ""))
                ref_label = RA.describe(ref)
            results.append({"report": rep, "v1_page": pg, "condition": cond, "row_type": e0["row_type"],
                            "dataset": e0["dataset"], "group": f["group"], "prompt": cond,
                            "reference": ref_label,
                            "reference_sentences": [x[1] for x in ref["evidence"]] if ref else [],
                            "chexagent_side": f["cx_side"], "maira_side": f["mx_side"],
                            "chexagent_box_recorded": bool(f["cx_coords"]), **s})
            fig = plt.figure(figsize=(W, H))
            fig.text(0.012, 0.985, f"{cond.capitalize()} - {e0['row_type']}   ({rep} p. {pg}; Group {f['group']}; "
                     f"{e0['dataset']})", fontsize=12, weight="bold", va="top")
            panels = [("Current (original)", None)]
            if f["cx_coords"]:
                panels.append(("CheXagent - original box (red)", ("box", f["cx_coords"], "red")))
            elif f["bi"] and f["bi"][0] == "embedded":
                panels.append(("CheXagent - exact form image (all boxes of that page)", ("rgb", np.array(f["bi"][1]))))
            else:
                panels.append(("CheXagent - no box for this finding", ("box", [], "red")))
            panels.append(("MAIRA-2 - box (yellow)" if f["mx"] else "MAIRA-2 - no box", ("box", f["mx"] or [],
                                                                                         "yellow")))
            panels.append((f"BioViL-T - similarity to '{cond}' (EXPLORATORY)", ("heat", sim)))
            pw = 0.976 / len(panels)
            for j, (cap, spec) in enumerate(panels):
                ax = fig.add_axes([0.012 + j * pw, 0.42, pw - 0.008, 0.5])
                if spec and spec[0] == "rgb":
                    ax.imshow(spec[1])
                else:
                    ax.imshow(cur, cmap="gray")
                if spec and spec[0] == "box":
                    for b in spec[1]:
                        x1, y1 = b[0] / 100 * cur.shape[1], b[1] / 100 * cur.shape[0]
                        ax.add_patch(Rectangle((x1, y1), (b[2] - b[0]) / 100 * cur.shape[1],
                                               (b[3] - b[1]) / 100 * cur.shape[0], fill=False, ec=spec[2], lw=1.4))
                if spec and spec[0] == "heat":
                    hm = ax.imshow(spec[1], cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.5)
                    py, px = s["peak_xy_pct"][1] / 100 * cur.shape[0], s["peak_xy_pct"][0] / 100 * cur.shape[1]
                    ax.plot(px, py, marker="+", color="black", ms=10, mew=1.5)
                    cb = fig.add_axes([0.012 + j * pw, 0.395, pw - 0.008, 0.012])
                    fig.colorbar(hm, cax=cb, orientation="horizontal").ax.tick_params(labelsize=6)
                ax.axis("off")
                ax.set_title(wrap(cap, 48), fontsize=7.5, loc="left")
                ax.text(0.0, -0.01, "R (patient right)", transform=ax.transAxes, fontsize=6, va="top")
                ax.text(1.0, -0.01, "patient left (L)", transform=ax.transAxes, fontsize=6, va="top", ha="right")
            lines = [f"Reference: {ref_label}"]
            for sent in results[-1]["reference_sentences"][:3]:
                lines.append(f'   "{sent}"')
            lines += [f"MedGemma: {e0['name'] if e0['row_type'] == 'MedGemma stated' else 'not stated'}; "
                      f"CheXagent box side: {f['cx_side'] or '-'}; MAIRA-2 box side: {f['mx_side'] or '-'}",
                      f"BioViL-T map: peak similarity {s['max']:.2f} at the {s['peak_side']} (+); top-10% of the map "
                      f"{100 * s['top10_side']['patient right']:.0f}% on the patient's right, "
                      f"{100 * s['top10_side']['patient left']:.0f}% on the left"
                      + "".join(f"; {100 * s[f'top10_in_{k}_box']:.0f}% inside the {k} box" for k in boxes
                                if f"top10_in_{k}_box" in s),
                      "Descriptive only: no box here is ground truth, and similarity is not probability of disease."]
            fig.text(0.012, 0.33, "\n".join(wrap(x, 175) for x in lines), fontsize=7.5, va="top")
            pdf.savefig(fig)
            plt.close(fig)
    meta = {"model": "microsoft/BiomedVLP-BioViL-T", "revision": REVISION, "image_weights_md5": IMG_MD5,
            "code": "hi-ml-multimodal 0.2.2 (by path)", "prompt": "condition name only", "device": device,
            "transform": "resize 512, centre crop 448", "smoothing": "gaussian sigma 1.5 on 14x14 patch grid",
            "results": results}
    js = out.with_suffix(".json")
    js.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(out)
    print(js)
    for r in results:
        print(f"{r['report']:8s} p{r['v1_page']:<3d} {r['condition']:17s} grp {r['group']} ref: {r['reference'][:45]:45s}"
              f" peak {r['max']:.2f} {r['peak_side']:13s} R/L top10 {r['top10_side']['patient right']:.2f}/"
              f"{r['top10_side']['patient left']:.2f} cx {r['chexagent_side']} mx {r['maira_side']}"
              f" inCX {r.get('top10_in_CheXagent_box', '-')} inMX {r.get('top10_in_MAIRA-2_box', '-')}")


if __name__ == "__main__":
    main()
