"""Example figures for grounding_recall_report.md, in both directions: per condition, up to six NIH
cases where one model's box hits the radiologist's box (IoU >= 0.5) and the other's does not (IoU <
0.3 or no box), largest IoU gap first (ties: file name). Selected examples; each figure states how
often each pattern occurred.

    python experiments/grounding_examples_figures.py
        -> results/figures/grounding_{chexagent,maira2}_better_<label>.png
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as pa  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import grounding_recall as G  # noqa: E402

FIGS = HERE / "results" / "figures"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
NAMES = {"Effusion": "Pleural effusion", "Infiltrate": "Infiltration"}


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    cs = G.cases()
    rec = {m: {(json.loads(l)["file"], json.loads(l)["label"]): json.loads(l)
               for l in p.read_text(encoding="utf-8").splitlines()} for m, p in G.OUT.items()}
    rows = []
    for c in cs:
        k = (c["file"], c["label"])
        rows.append((c, rec["CheXagent"][k], rec["MAIRA-2"][k],
                     G.measures(rec["CheXagent"][k], c)["best"], G.measures(rec["MAIRA-2"][k], c)["best"]))
    for winner, loser, wi, li in (("CheXagent", "MAIRA-2", 3, 4), ("MAIRA-2", "CheXagent", 4, 3)):
        for label in G.PHRASE:
            figure(rows, label, winner, loser, wi, li)
    # MAIRA-2's strength (cardiomegaly) is at IoU 0.5: CheXagent's heart boxes mostly overlap 0.3-0.5,
    # so the 0.3 'miss' line hides it; this figure splits both models at 0.5
    figure(rows, "Cardiomegaly", "MAIRA-2", "CheXagent", 4, 3, miss=0.5)
    for label in G.PHRASE:
        lab = [r for r in rows if r[0]["label"] == label]
        print(f"{label}: at IoU 0.5 only CheXagent hits {sum(r[3] >= 0.5 > r[4] for r in lab)}, "
              f"only MAIRA-2 hits {sum(r[4] >= 0.5 > r[3] for r in lab)}")


def figure(rows, label, winner, loser, wi, li, miss=0.3):
    lab = [r for r in rows if r[0]["label"] == label]
    better = sorted([r for r in lab if r[wi] >= 0.5 and r[li] < miss], key=lambda r: (-(r[wi] - r[li]), r[0]["file"]))
    reverse = [r for r in lab if r[li] >= 0.5 and r[wi] < miss]
    if not better:
        print(f"{label}: no case where {winner} hits and {loser} misses")
        return
    shown = better[:6]
    cols = 3
    nrow = (len(shown) + cols - 1) // cols
    fig, axs = plt.subplots(nrow, cols, figsize=(13.5, 5.3 * nrow + 1.0), squeeze=False)
    for ax in axs.flat:
        ax.set_axis_off()
    for ax, (c, ra, rm, ia, im) in zip(axs.flat, shown):
        img = Image.open(G.NIH_DIR / c["file"]).convert("L")
        W, H = img.size
        ax.imshow(img, cmap="gray")

        def draw(b, col, ls="-", lw=2.2):
            ax.add_patch(pa.Rectangle((b[0] * W / 100, b[1] * H / 100), (b[2] - b[0]) * W / 100,
                                      (b[3] - b[1]) * H / 100, fill=False, ec=col, lw=lw, ls=ls))
        draw(c["truth"], GREEN, lw=2.8)
        for b in ra["boxes"]:
            draw(b, BLUE)
        for b in rm["boxes"]:
            draw(b, ORANGE, "--")
        ax.set_title(f"{c['file']}\nCheXagent IoU {ia:.2f}" + ("" if ra["boxes"] else " (no box)")
                     + f"  ·  MAIRA-2 IoU {im:.2f}" + ("" if rm["boxes"] else " (no box)"), fontsize=10.5, loc="left")
    name = NAMES.get(label, label)
    fig.legend(handles=[pa.Patch(fc="none", ec=GREEN, lw=2.8, label="Radiologist box (NIH)"),
                        pa.Patch(fc="none", ec=BLUE, lw=2.2, label="CheXagent"),
                        pa.Patch(fc="none", ec=ORANGE, lw=2.2, ls="--", label="MAIRA-2")],
               loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle(f"{name}: cases where {winner}'s box hits and {loser}'s misses", fontsize=14,
                 fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 1 - 0.55 / (5.3 * nrow + 1.0) - 0.035,
             f"Of {len(lab)} {name.lower()} cases: {winner} hit (IoU >= 0.5) and {loser} missed (IoU < {miss}) in "
             f"{len(better)}; the reverse in {len(reverse)}. Selected examples, largest gap first "
             f"({len(shown)} shown).", fontsize=10, color="#52514e")
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - 1.95 / (5.3 * nrow + 1.0), bottom=0.7 / (5.3 * nrow + 1.0),
                        wspace=0.08, hspace=0.22)
    tag = "chexagent" if winner == "CheXagent" else "maira2"
    out = FIGS / f"grounding_{tag}_better_{label.lower()}{'' if miss == 0.3 else '_iou05'}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"{label}: {winner} better {len(better)}, reverse {len(reverse)} -> {out.name}")

if __name__ == "__main__":
    main()
