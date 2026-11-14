"""Pilot (idea 3): prior-vs-current temporal subtraction on the same findings as the BioViL-T pilot.

Exploratory only - not validated, not shown to radiologists, no claim of diagnostic accuracy. No new model: lungs from
the existing torchxrayvision PSPNet (anatomical_measurements.segment_all). No existing file is changed; MIMIC-CXR
content -> logs/ only.

Per finding:
  1. prior and current: grey, zero-padded to a square, resized to 768 px; prior histogram-matched to current.
  2. lung masks (left + right lung) on both.
  3. registration of the prior onto the current: initial scale/shift from the lung-mask bounding boxes, refined by
     ECC affine (OpenCV) on blurred images inside the dilated current lung mask; the initial transform is kept if
     ECC fails or makes the lung overlap worse.
  4. each image z-scored inside each lung separately (current lung masks; cancels exposure/contrast differences),
     difference = current - warped prior in SD units, Gaussian-smoothed (sigma 6 px, suppresses rib-edge
     misregistration). Side effect: a uniform change over a whole lung is also cancelled; only local change remains.
     shown inside the current lungs dilated by 25 px. Red = denser (whiter) now; blue = more lucent (darker) now.
  QC: lung-mask Dice (warped prior vs current) and lung-region correlation before/after. Dice < 0.85 -> flagged
  'registration poor: map unreliable'. Breathing depth, rotation, AP/PA and exposure differences cause artefacts.

    .venv/Scripts/python.exe experiments/subtraction_pilot.py
"""

import json
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = ROOT / "logs"
sys.path[:0] = [str(ROOT), str(HERE)]
from biovil_t_heatmap_pilot import PICK  # noqa: E402

N = 768
SIGMA = 6
VMAX = 1.5
THR = 0.75  # SD units: "changed" pixel for the per-lung summary
DICE_MIN = 0.85


def square(path):
    im = np.asarray(Image.open(path).convert("L"), np.float32)
    h, w = im.shape
    s = max(h, w)
    pad = np.zeros((s, s), np.float32)
    pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
    return cv2.resize(pad, (N, N), interpolation=cv2.INTER_AREA)


def lungs(img):
    import anatomical_measurements as A
    m = A.segment_all(np.clip(img, 0, 255))  # INTER_AREA can overshoot 255 by rounding
    return m["Right Lung"] | m["Left Lung"], m["Right Lung"], m["Left Lung"]


def dice(a, b):
    return float(2 * (a & b).sum() / max(1, a.sum() + b.sum()))


def corr(a, b, m):
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-9))


def register(prior, cur, lp, lc):
    """Affine 2x3 mapping prior -> current; returns (warp, method)."""
    def bbox(m):
        ys, xs = np.nonzero(m)
        return xs.min(), ys.min(), xs.max(), ys.max()
    px1, py1, px2, py2 = bbox(lp)
    cx1, cy1, cx2, cy2 = bbox(lc)
    sx, sy = (cx2 - cx1) / max(1, px2 - px1), (cy2 - cy1) / max(1, py2 - py1)
    init = np.array([[sx, 0, cx1 - sx * px1], [0, sy, cy1 - sy * py1]], np.float32)
    warp_mask = lambda W: cv2.warpAffine(lp.astype(np.uint8), W, (N, N), flags=cv2.INTER_NEAREST) > 0  # noqa: E731
    d0 = dice(warp_mask(init), lc)
    a = cv2.GaussianBlur(cur, (0, 0), 3)
    b = cv2.GaussianBlur(prior, (0, 0), 3)
    mask = cv2.dilate(lc.astype(np.uint8), np.ones((31, 31), np.uint8))
    # ECC (template = current, input = prior) estimates W with current(x) ~ prior(W x): a current -> prior map,
    # so it starts from the inverse of the prior -> current initial transform and is inverted back afterwards
    W = cv2.invertAffineTransform(init).astype(np.float32)
    try:
        _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_AFFINE,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), mask, 5)
        W = cv2.invertAffineTransform(W)
        if dice(warp_mask(W), lc) < d0:
            return init, "lung-box scale/shift (ECC made overlap worse)"
        return W, "lung-box init + ECC affine"
    except cv2.error:
        return init, "lung-box scale/shift (ECC did not converge)"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from skimage.exposure import match_histograms
    import regional_focal_reports as R
    import regional_focal_reports_v2 as R2
    import reference_sentence_audit as RA
    import visualization_evaluation_report as VE

    st = time.strftime("%Y%m%d-%H%M%S")
    F, *_ = R.build()
    pages = R2.v1_page_numbers(F)
    sel = {pages[id(f)]: f for f in F}
    refs = VE.reference_texts()
    W_, H_ = 11.7, 8.3
    wrap = lambda s, n: "\n".join(textwrap.wrap(" ".join(str(s).split()), n))  # noqa: E731
    results = []
    out = LOGS / f"subtraction_pilot_{st}.pdf"
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(W_, H_))
        fig.text(0.04, 0.95, "\n".join([
            "PILOT - PRIOR-vs-CURRENT SUBTRACTION (idea 3)  -  exploratory, not validated, not for radiologists", "",
            wrap("Exploratory only: not validated, not shown to radiologists, no claim of diagnostic accuracy. No new "
                 "model; no existing prediction, box, score or report is changed.", 110), "",
            "Prior aligned to current (lung-mask scale/shift + ECC affine), histogram-matched, then subtracted and",
            f"each lung z-scored on both films (cancels exposure), smoothed (sigma {SIGMA} px at {N} px); shown inside the current lungs.",
            "  red  = denser / whiter now than on the prior (e.g. new or larger opacity or effusion)",
            "  blue = more lucent / darker now (e.g. cleared opacity, more air - pneumothorax, hyperinflation)",
            f"Registration QC on every page: lung-mask Dice after alignment (< {DICE_MIN} = map unreliable) and lung",
            "correlation before/after. Breathing depth, rotation, AP/PA and exposure differences leave artefacts,",
            "typically along the diaphragm, heart border and ribs. Patient's right is on the image left.", "",
            f"{len(PICK)} findings (same as the BioViL-T pilot): " + "; ".join(f"{a} p. {b}" for a, b in PICK), "",
            "MIMIC-CXR content (PhysioNet DUA): do not distribute."]), fontsize=8.6, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        for rep, pg in PICK:
            f = sel[(rep, pg)]
            e0, cond = f["e0"], f["key"][1]
            cur, prior = square(e0["current"]), square(e0["prior"])
            lc, lcr, lcl = lungs(cur)
            lp, _, _ = lungs(prior)
            prior_m = match_histograms(prior, cur).astype(np.float32)
            Wm, method = register(prior_m, cur, lp, lc)
            warped = cv2.warpAffine(prior_m, Wm, (N, N), flags=cv2.INTER_LINEAR)
            lpw = cv2.warpAffine(lp.astype(np.uint8), Wm, (N, N), flags=cv2.INTER_NEAREST) > 0
            region = cv2.dilate(lc.astype(np.uint8), np.ones((51, 51), np.uint8)) > 0
            zc, zp = np.zeros_like(cur), np.zeros_like(cur)
            for m in (lcr, lcl):
                zc[m] = (cur[m] - cur[m].mean()) / (cur[m].std() + 1e-6)
                zp[m] = (warped[m] - warped[m].mean()) / (warped[m].std() + 1e-6)
            diff = cv2.GaussianBlur(zc - zp, (0, 0), SIGMA)
            region = lc
            shown = np.where(region, diff, np.nan)
            qc = {"lung_dice_before": dice(lp, lc), "lung_dice_after": dice(lpw, lc),
                  "corr_before": corr(cur, prior_m, lc), "corr_after": corr(cur, warped, lc), "method": method}
            ok = qc["lung_dice_after"] >= DICE_MIN
            # per lung (current masks; image left = patient right): mean change, and share of strong change
            per = {}
            for side, m in (("patient right", lcr), ("patient left", lcl)):
                d = diff[m]
                per[side] = {"mean": float(d.mean()), "denser_pct": float(100 * (d > THR).mean()),
                             "lucent_pct": float(100 * (d < -THR).mean())}
            if e0["dataset"].startswith("NIH"):
                ref_label, sents = ("NIH label present" if e0["reference"] == "Yes" else "NIH label not assigned"), []
            else:
                res = RA.classify(cond, refs.get(e0["key"], ""))
                ref_label, sents = RA.describe(res), [x[1] for x in res["evidence"]]
            results.append({"report": rep, "v1_page": pg, "condition": cond, "dataset": e0["dataset"],
                            "medgemma_change": e0.get("change"), "reference": ref_label,
                            "reference_sentences": sents, "registration_ok": ok, **qc, "per_lung": per})
            fig = plt.figure(figsize=(W_, H_))
            fig.text(0.012, 0.985, f"{cond.capitalize()} - {e0['row_type']}   ({rep} p. {pg}; {e0['dataset']})",
                     fontsize=12, weight="bold", va="top")
            if not ok:
                fig.text(0.012, 0.952, f"REGISTRATION POOR (lung Dice {qc['lung_dice_after']:.2f}): MAP UNRELIABLE",
                         fontsize=10, weight="bold", color="#b42318", va="top")
            panels = [("Prior (original)", prior, None), ("Prior aligned to current", warped, "lc"),
                      ("Current (original)", cur, "lc"), ("Subtraction: current - aligned prior (EXPLORATORY)", cur,
                                                          "diff")]
            pw = 0.976 / len(panels)
            for j, (cap, arr, spec) in enumerate(panels):
                ax = fig.add_axes([0.012 + j * pw, 0.42, pw - 0.008, 0.5])
                ax.imshow(arr, cmap="gray")
                if spec == "lc":
                    ax.contour(lc.astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
                if spec == "diff":
                    hm = ax.imshow(shown, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, alpha=0.65)
                    cb = fig.add_axes([0.012 + j * pw, 0.395, pw - 0.008, 0.012])
                    c = fig.colorbar(hm, cax=cb, orientation="horizontal")
                    c.ax.tick_params(labelsize=6)
                    c.set_label("blue = more lucent now      red = denser now", fontsize=6)
                ax.axis("off")
                ax.set_title(wrap(cap, 48), fontsize=7.5, loc="left")
                ax.text(0.0, -0.01, "R (patient right)", transform=ax.transAxes, fontsize=6, va="top")
                ax.text(1.0, -0.01, "patient left (L)", transform=ax.transAxes, fontsize=6, va="top", ha="right")
            lines = [f"Reference (current report): {ref_label}"] + [f'   "{s_}"' for s_ in sents[:3]]
            lines += [f"MedGemma interval change: {e0.get('change') or '-'}",
                      f"Registration: {method}; lung Dice {qc['lung_dice_before']:.2f} -> {qc['lung_dice_after']:.2f}; "
                      f"lung correlation {qc['corr_before']:.2f} -> {qc['corr_after']:.2f}"
                      + ("" if ok else "  ** below threshold: map unreliable **"),
                      f"Change inside each lung (share of lung pixels changed by > {THR} SD (lung-normalized)): "
                      + "; ".join(f"{k}: denser {v['denser_pct']:.0f}%, more lucent {v['lucent_pct']:.0f}%"
                                  for k, v in per.items()),
                      "Descriptive only: subtraction shows grey-level change after alignment, including artefacts; it "
                      "is not a diagnosis and was not checked against radiologist-marked change."]
            fig.text(0.012, 0.33, "\n".join(wrap(x, 175) for x in lines), fontsize=7.5, va="top")
            pdf.savefig(fig)
            plt.close(fig)
    js = out.with_suffix(".json")
    js.write_text(json.dumps({"size": N, "sigma": SIGMA, "vmax": VMAX, "dice_min": DICE_MIN, "results": results},
                             indent=1, default=str), encoding="utf-8")
    print(out)
    for r in results:
        p = r["per_lung"]
        print(f"{r['report']:8s} p{r['v1_page']:<3d} {r['condition']:17s} dice {r['lung_dice_before']:.2f}->"
              f"{r['lung_dice_after']:.2f} corr {r['corr_before']:.2f}->{r['corr_after']:.2f} ok {r['registration_ok']}"
              f" | R +{p['patient right']['denser_pct']:.0f}/-{p['patient right']['lucent_pct']:.0f}"
              f"  L +{p['patient left']['denser_pct']:.0f}/-{p['patient left']['lucent_pct']:.0f}"
              f" | MG change: {str(r['medgemma_change'])[:40]} | ref: {r['reference'][:40]}")


if __name__ == "__main__":
    main()
