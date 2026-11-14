"""Credibility figures for the manuscript: study flow, laterality examples, calibration, pre-registration timeline.

    python experiments/manuscript_extra_figures.py   -> experiments/results/figures/manuscript_*.png

Example cases are chosen by a fixed rule (first one-sided case of each finding in image-ID order), not by result.
"""

import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as pa  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
FIGS = RESULTS / "figures"
INK, MUTED, LINE = "#16233a", "#5f6470", "#1f2d44"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


# ------------------------------------------------------------------ 1. study flow
def flow_figure(path):
    fig, ax = plt.subplots(figsize=(12, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f4f2", bold=False, size=9.5):
        ax.add_patch(pa.Rectangle((x, y), w, h, fc=fc, ec=LINE, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def down(x, y1, y2):
        ax.annotate("", (x, y2), (x, y1), arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.2, mutation_scale=12))

    def side(x1, y, x2, text):
        ax.annotate("", (x2, y), (x1, y), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, mutation_scale=10))

    cols = [(1, 31, "NIH ChestX-ray14"), (35, 30, "Indiana (Open-i)"), (68, 31, "VinDr-CXR (Kaggle VinBigData)")]
    for x, w, t in cols:
        ax.text(x + w / 2, 97, t, ha="center", fontsize=12, fontweight="bold", color=INK)
    # NIH
    box(1, 84, 31, 9, "112,120 images\n30,805 patients")
    down(16.5, 84, 77)
    box(1, 68, 31, 9, "Random sample: one image per patient,\n5,000 patients (fixed seed)")
    down(16.5, 68, 61)
    box(1, 52, 31, 9, "Random 70/30 split, no shared patients\nDevelopment 3,500 · Test 1,500")
    down(9, 52, 43)
    down(24, 52, 43)
    box(1, 31, 15, 12, "Test 1,500\nPresence (E2)\nFusion, editing\n(E4 a)", fc="#e7f6ef", size=9)
    box(17, 31, 15, 12, "Development\n3,500\nCalibration;\nexploratory", fc="#eef3fb", size=9)
    down(8.5, 31, 23)
    box(1, 12, 15, 11, "499 test images\nwith an earlier\nimage (prior outside\nthe sample): E4 b, c", fc="#e7f6ef", size=8.6)
    box(17, 12, 15, 11, "984 radiologist\nboxes (880 images)\n− 37 on test images\n= 947 boxes: E3", fc="#e7f6ef", size=8.6)
    # Indiana
    box(35, 84, 30, 9, "Open-i frontal images with reports\nand coded problem / MeSH terms")
    down(50, 84, 77)
    box(35, 68, 30, 9, "Evaluation set selected per condition\n(enriched): 544 images, 6 conditions")
    down(50, 68, 61)
    box(35, 52, 30, 9, "All 544 used\n(also our earlier CheXagent\ndevelopment set, 2026-09-27)")
    down(50, 52, 43)
    box(35, 31, 30, 12, "Presence (E2)\nCheXagent and MAIRA-2\nagainst coded labels", fc="#e7f6ef")
    # VinDr
    box(68, 84, 31, 9, "15,000 training images,\neach read by 3 radiologists")
    down(83.5, 84, 77)
    box(68, 68, 31, 9, "3,000 sampled by a label-independent\nhash ranking, frozen before\nany label was read")
    down(83.5, 68, 61)
    box(68, 52, 31, 9, "Label: present if ≥2 of 3 readers\n(3,000 used; none excluded)")
    down(76, 52, 43)
    down(91, 52, 43)
    box(68, 31, 15, 12, "3,000 images\nPresence (E2)", fc="#e7f6ef", size=9)
    box(84, 31, 15, 12, "170 findings with\na consensus side:\neffusion, consolidation,\npneumothorax,\natelectasis",
        size=8.2)
    down(91.5, 31, 23)
    box(84, 12, 15, 11, "− 1 side conflict\n− 16 bilateral\n= 153 one-sided:\nE1 laterality", fc="#e7f6ef", size=8.6)
    ax.text(1, 3, "Green boxes: sets used in the reported analyses. E1 laterality, E2 presence, E3 grounding,\n"
                  "E4 downstream (a: single image; b, c: prior + current, base and fine-tuned MedGemma).", fontsize=9,
            color=MUTED, linespacing=1.4)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ 2. laterality examples
def laterality_examples(path):
    cases = json.loads((RESULTS / "vindr_grounding_cases.json").read_text(encoding="utf-8"))["cases"]
    rec = {}
    for model, f in (("CheXagent", "vindr_grounding_chexagent.jsonl"), ("MAIRA-2", "vindr_grounding_maira2.jsonl")):
        for r in map(json.loads, (RESULTS / f).read_text(encoding="utf-8").splitlines()):
            rec[(model, r["image_id"], r["finding"], r["condition"])] = r
    chosen = []
    for finding in ("pleural effusion", "consolidation", "pneumothorax", "atelectasis"):
        c = sorted((c for c in cases if c["unilateral"] and c["finding"] == finding), key=lambda c: c["image_id"])[0]
        chosen.append(c)
    conds = [("blind", "Side-free prompt"), ("true", "Prompt names the true side"),
             ("wrong", "Prompt names the opposite side")]
    fig, axs = plt.subplots(len(chosen), 3, figsize=(10.5, 3.7 * len(chosen)))
    for i, c in enumerate(chosen):
        im = Image.open(c["png"]).convert("L")
        im.thumbnail((700, 700))
        W, H = im.size
        opp = {"right": "left", "left": "right"}[c["side"]]
        for j, (cond, title) in enumerate(conds):
            ax = axs[i, j]
            ax.imshow(im, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for b in c["truth_boxes"]:
                x1, y1, x2, y2 = b
                ax.add_patch(pa.Rectangle((x1 / 100 * W, y1 / 100 * H), (x2 - x1) / 100 * W, (y2 - y1) / 100 * H,
                                          fill=False, ec=GREEN, lw=1.4))
            for model, col, ls in (("CheXagent", BLUE, "-"), ("MAIRA-2", ORANGE, "--")):
                r = rec.get((model, c["image_id"], c["finding"], cond))
                for b in (r or {}).get("boxes") or []:
                    x1, y1, x2, y2 = b
                    ax.add_patch(pa.Rectangle((x1 / 100 * W, y1 / 100 * H), (x2 - x1) / 100 * W, (y2 - y1) / 100 * H,
                                              fill=False, ec=col, lw=2.2, ls=ls))
            named = {"blind": "no side", "true": c["side"], "wrong": opp}[cond]
            if i == 0:
                ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
            ax.text(0.02, 0.98, f"prompt side: {named}", transform=ax.transAxes, va="top", fontsize=8.5,
                    color="white", bbox=dict(fc="#000000a0", ec="none", pad=2))
        axs[i, 0].set_ylabel(f"{c['finding'].capitalize()}\ntrue side: patient's {c['side']}", fontsize=10,
                             color=INK)
    handles = [pa.Patch(fill=False, ec=GREEN, lw=1.4, label="Radiologists (VinDr readers)"),
               pa.Patch(fill=False, ec=BLUE, lw=2.2, label="CheXagent"),
               pa.Patch(fill=False, ec=ORANGE, lw=2.2, ls="--", label="MAIRA-2")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Laterality conditioning in example cases (first one-sided case of each finding by image ID)",
                 fontsize=12, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.957, "Images are shown in standard orientation: the patient's right is on the image's left. "
                          "Boxes: all three VinDr readers (green), CheXagent (blue), MAIRA-2 (dashed orange).",
             fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.945))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [c["image_id"] for c in chosen]


# ------------------------------------------------------------------ 3. calibration
def calibration_figure(path):
    eng = json.loads((ROOT / "medrax" / "evidence" / "evidence_engine_v1.0.json").read_text(encoding="utf-8"))
    g = eng["grid"]
    z = np.linspace(g["start"], g["stop"], g["num"])
    split = json.loads((RESULTS / "split_nih_70_30.json").read_text(encoding="utf-8"))["random"]
    test = {}
    for r in map(json.loads, (RESULTS / "ppresent_nih.jsonl").read_text(encoding="utf-8").splitlines()):
        if r["set"] == "random" and split.get(r["file"]) == "test" and r["p_present"] is not None:
            test.setdefault(r["finding"], []).append((r["p_present"], r["truth"]))
    edges = eng["band_edges"]
    sys.path.insert(0, str(ROOT))
    from medrax.evidence import EvidenceEngine
    engine = EvidenceEngine()
    grid = np.round(np.arange(0, 1.00001, 0.0005), 4)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.6), sharey=True)
    for ax, (name, f) in zip(axs.flat, eng["engines"]["primary"]["findings"].items()):
        a, b = f["platt"]["coef"], f["platt"]["intercept"]
        pi = f["dev_prevalence"]
        lr = np.exp(a * z + b) / (pi / (1 - pi))  # odds(q) / odds(prevalence)
        lo, hi = np.exp(f["log_lr_ci"]["lower"]), np.exp(f["log_lr_ci"]["upper"])
        ax.fill_between(z, lo, hi, color=BLUE, alpha=0.18, lw=0, label="Platt LR, 95% CI (development)")
        ax.plot(z, lr, color=BLUE, lw=1.8)
        rows = test.get(name, [])
        p = np.array([x[0] for x in rows])
        y = np.array([x[1] for x in rows])
        npos, nneg = (y == 1).sum(), (y == 0).sum()
        for k in range(len(edges) - 1):
            m = (p >= edges[k]) & ((p < edges[k + 1]) | (edges[k + 1] >= 1))
            tp, fp = (m & (y == 1)).sum(), (m & (y == 0)).sum()
            if tp == 0 or fp == 0:
                continue
            elr = (tp / npos) / (fp / nneg)
            se = np.sqrt(1 / tp - 1 / npos + 1 / fp - 1 / nneg)
            zc = np.log(np.clip(np.median(p[m]), 1e-4, 1 - 1e-4) / (1 - np.clip(np.median(p[m]), 1e-4, 1 - 1e-4)))
            ax.errorbar([zc], [elr], yerr=[[elr - elr * np.exp(-1.96 * se)], [elr * np.exp(1.96 * se) - elr]],
                        fmt="o", color=ORANGE, ms=5, capsize=2, lw=1.2)
        for t, col in ((5, GREEN), (0.2, "#c0392b")):
            ax.axhline(t, color=col, ls="--", lw=1)
        states = [engine.assess(name, float(q)).state for q in grid]
        sup = [q for q, st in zip(grid, states) if st == "SUPPORTING"]
        ref = [q for q, st in zip(grid, states) if st == "REFUTING"]
        ax.set_yscale("log")
        ax.set_ylim(0.005, 300)
        ax.set_xlim(-8, 6)
        ticks = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99]
        ax.set_xticks([np.log(t / (1 - t)) for t in ticks], [str(t) for t in ticks], fontsize=9)
        ax.set_title(f"{name.capitalize()} ({npos} test positives)", fontsize=11, color=INK, loc="left")
        ref_txt = f"REFUTING up to p = {max(ref):g}" if ref else "no REFUTING region (not empirically confirmed)"
        ax.text(0.02, 0.97, f"Engine: SUPPORTING from p = {min(sup):g}\n{ref_txt}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)
        ax.grid(color="#e1e0d9", lw=0.6)
    for ax in axs[:, 0]:
        ax.set_ylabel("Likelihood ratio (log scale)")
    for ax in axs[1]:
        ax.set_xlabel("CheXagent p_present (logit scale)")
    handles = [pa.Patch(color=BLUE, alpha=0.35, label="Frozen engine: Platt LR with 95% CI (NIH development)"),
               plt.Line2D([], [], marker="o", color=ORANGE, ls="", label="Empirical LR per band, NIH test split (95% CI)"),
               plt.Line2D([], [], color=GREEN, ls="--", label="LR 5 (SUPPORTING threshold)"),
               plt.Line2D([], [], color="#c0392b", ls="--", label="LR 0.2 (REFUTING threshold)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("Evidence Engine v1.0: likelihood ratios fitted on development data and observed on the test split",
                 fontsize=12.5, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ 4. pre-registration timeline
TIMELINE = [  # (row label, [(kind, commit, "YYYY-MM-DD HH:MM", label)])
    ("Evidence Engine v1.0", [("P", "47c6b64", "2026-09-27 16:36", "design"), ("F", "a277a8a", "2026-09-27 16:46", "frozen")]),
    ("Engine validation on VinDr (Part A)", [("P", "5280b43", "2026-09-27 17:02", ""), ("R", "4357857", "2026-10-05 12:38", "")]),
    ("E1 laterality (VinDr)", [("P", "daeb6fa", "2026-10-04 07:44", ""), ("A", "73a9d4f", "2026-10-04 08:19", "data source"),
                               ("R", "ede4255", "2026-10-05 10:54", "")]),
    ("Provisional cut-offs", [("P", "a4c832d", "2026-10-09 09:39", ""), ("R", "2eefcb8", "2026-10-09 13:11", "")]),
    ("Editing rule", [("P", "8d615fd", "2026-10-13 21:51", "")]),
    ("E2 presence", [("P", "79578de", "2026-10-13 06:50", ""), ("A", "9459f6d", "2026-10-17 22:05", "A1"),
                     ("R", "2270725", "2026-10-17 23:09", ""), ("A", "9a14988", "2026-10-18 07:16", "A2"),
                     ("R", "992ce10", "2026-10-18 10:04", "")]),
    ("E4 a: base, single image", [("P", "3189f6f", "2026-10-18 17:24", ""), ("A", "ae21a7f", "2026-10-18 18:55", "A1"),
                                  ("R", "d7e8b36", "2026-10-19 16:49", "")]),
    ("E4 b: base, prior + current", [("P", "ed54d95", "2026-10-19 17:53", ""), ("R", "2ca38e5", "2026-10-20 06:44", "")]),
    ("E3 grounding (NIH boxes)", [("P", "6106a1d", "2026-10-24 07:15", ""), ("R", "cd3ba76", "2026-10-24 07:49", "")]),
    ("E4 c: fine-tuned, prior + current", [("P", "9362ffb", "2026-10-24 22:30", ""), ("R", "dbe2f1e", "2026-10-25 06:45", "")]),
]


def _fmt_gap(minutes):
    if minutes < 60:
        return f"{minutes:.0f} min"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} h"
    return f"{minutes / 1440:.1f} days"


def timeline_figure(path):
    fig, ax = plt.subplots(figsize=(13, 6.6))
    style = {"P": ("o", INK, "protocol committed"), "F": ("D", INK, "specification frozen"),
             "A": ("^", ORANGE, "amendment committed"), "R": ("s", GREEN, "results committed")}
    end = dt.datetime(2026, 10, 26, 12)
    for i, (label, events) in enumerate(TIMELINE):
        y = len(TIMELINE) - 1 - i
        ts = [dt.datetime.strptime(e[2], "%Y-%m-%d %H:%M") for e in events]
        ax.plot([min(ts), max(ts)], [y, y], color="#b9bcc4", lw=2, zorder=1)
        gaps = []
        last_rule = None
        for (kind, commit, _, note), t in zip(events, ts):
            mk, col, _ = style[kind]
            ax.scatter([t], [y], marker=mk, s=70, color=col, zorder=3, edgecolor="white", linewidth=0.8)
            if kind in ("P", "A", "F"):
                last_rule = (t, note or {"P": "protocol", "F": "freeze", "A": "amendment"}[kind])
            elif last_rule:
                gaps.append(f"{last_rule[1]} → results {_fmt_gap((t - last_rule[0]).total_seconds() / 60)}")
        if not gaps:
            gaps = ["no results of its own (applied in E4)"] if label == "Editing rule" else \
                   ["frozen 10 min after the design"] if label.startswith("Evidence Engine") else []
        ax.text(end, y, "; ".join(gaps), va="center", fontsize=8.8, color=MUTED)
    ax.set_yticks(range(len(TIMELINE)), [t[0] for t in reversed(TIMELINE)], fontsize=10)
    ax.set_xlim(dt.datetime(2026, 9, 26), dt.datetime(2026, 11, 9))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    ax.set_xticks([dt.datetime(2026, 9, 27) + dt.timedelta(days=d) for d in range(0, 30, 4)])
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    ax.axvline(end - dt.timedelta(hours=8), color="#d9d9d9", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], marker=m, color=c, ls="", ms=8, label=n) for m, c, n in style.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.04))
    ax.set_title("Pre-registration timeline (git commit times, 2026): every protocol and amendment precedes its "
                 "results", fontsize=12, fontweight="bold", color=INK, loc="left")
    fig.text(0.01, 0.012, "Right: time from the last protocol or amendment to each results commit. Short gaps reflect "
                          "runs of minutes to hours; in E2 A1, MAIRA-2's NIH scores existed but had not been analysed, "
                          "as the amendment states.", fontsize=8.6, color=MUTED)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    flow_figure(FIGS / "manuscript_flow.png")
    ids = laterality_examples(FIGS / "manuscript_laterality_examples.png")
    calibration_figure(FIGS / "manuscript_calibration.png")
    timeline_figure(FIGS / "manuscript_timeline.png")
    print("example cases:", ids)
    sys.exit(0)
