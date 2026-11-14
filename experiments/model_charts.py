"""Slide-ready charts of model performance (numbers read from the committed results files).

    python experiments/model_charts.py   -> logs/charts_<time>/{1_nih_auc,2_caught,3_side}.png

1. MAIRA-2 vs CheXagent AUC per finding, NIH development split (maira_score_nih.md).
2. Ground-truth findings caught per approach, 19 comparisons (pilot_model_comparison_PRELIMINARY.md).
3. Side without being told the side, 12 one-sided findings (same file).
Palette: dataviz reference palette, validated (blue, orange, aqua; light surface).
"""

import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK, "axes.facecolor": SURFACE,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def table(text, heading):
    """Data rows of the first markdown table at or after the line starting with heading."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(heading))
    i = next(j for j in range(start, len(lines)) if lines[j].startswith("|"))
    block = []
    while i < len(lines) and lines[i].startswith("|"):
        block.append(lines[i])
        i += 1
    return [[c.strip() for c in l.strip("|").split("|")] for l in block[2:]]  # skip header and --- row


def nums(cell):
    """Unsigned numbers in a cell ('0.769 [0.724-0.811]' -> [0.769, 0.724, 0.811])."""
    return [float(x) for x in re.findall(r"\d*\.\d+|\d+", cell)]


def frame(ax, title, subtitle, pad=26):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=pad)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_auc(out):
    text = (RESULTS / "maira_score_nih.md").read_text(encoding="utf-8")
    rows = table(text, "| finding")
    rows = sorted(rows, key=lambda r: nums(r[3])[0])  # by CheXagent AUC, best at top
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, r in enumerate(rows):
        m, ml, mh = nums(r[2])[:3]
        c, cl, ch = nums(r[3])[:3]
        ax.plot([m, c], [i, i], color=GRID, linewidth=2, zorder=1, solid_capstyle="round")
        for v, lo, hi, col in ((m, ml, mh, ORANGE), (c, cl, ch, BLUE)):
            ax.plot([lo, hi], [i, i], color=col, linewidth=2, alpha=0.45, solid_capstyle="round", zorder=2)
            ax.scatter([v], [i], s=70, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(c + 0.012, i + 0.22, f"{c:.2f}", color=INK, fontsize=9)
        ax.text(m - 0.012, i + 0.22, f"{m:.2f}", color=INK, fontsize=9, ha="right")
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.505, -0.75, "chance", color=MUTED, fontsize=8.5)
    ax.set_yticks(range(len(rows)), [r[0].capitalize() for r in rows])
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(-0.9, len(rows) - 0.4)
    ax.set_xlabel("AUC (1.0 = perfect separation of present vs absent)")
    ax.scatter([], [], s=70, color=BLUE, label="CheXagent (direct question)")
    ax.scatter([], [], s=70, color=ORANGE, label="MAIRA-2 (sentence likelihood)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    frame(ax, "CheXagent separates present from absent better than MAIRA-2 on every finding",
          "AUC with 95% interval, 3,500 NIH chest X-rays with labels (development split)")
    fig.tight_layout()
    fig.savefig(out / "1_nih_auc.png", dpi=200)
    plt.close(fig)


def chart_caught(out):
    text = (RESULTS / "pilot_model_comparison_PRELIMINARY.md").read_text(encoding="utf-8")
    primary = text.split("## Sensitivity")[0]
    e1 = table(primary, "### Experiment 1")
    e2 = table(primary, "### Experiment 2")
    rows = [(f"{r[0]} — own report", r[1], r[3]) for r in e1] + \
           [("CheXagent — asked directly", e2[0][1], e2[0][2])]
    rows = sorted(rows, key=lambda r: nums(r[1])[2])  # by % caught
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (name, cell, unconf) in enumerate(rows):
        k, n, pct, lo, hi = nums(cell)[:5]
        col = BLUE if "asked directly" in name else GREY
        ax.plot([lo, hi], [i, i], color=col, linewidth=2.5, alpha=0.5 if col == BLUE else 1, solid_capstyle="round")
        ax.scatter([pct], [i], s=80, color=col if col == BLUE else MUTED, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(hi + 1.5, i, f"{int(k)}/{int(n)} caught  ·  {unconf} not in ground truth", va="center",
                color=INK if col == BLUE else INK2, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Ground-truth findings caught (%)")
    frame(ax, "Asking CheXagent directly catches the most ground-truth findings",
          "PRELIMINARY (unreviewed labels) · 19 comparisons, 32 ground-truth findings · dot = %, line = 95% interval")
    fig.tight_layout()
    fig.savefig(out / "2_caught.png", dpi=200)
    plt.close(fig)


def chart_side(out):
    text = (RESULTS / "pilot_model_comparison_PRELIMINARY.md").read_text(encoding="utf-8")
    primary = text.split("## Sensitivity")[0]
    e3 = table(primary, "### Experiment 3")
    n_cases = int(re.search(r"Experiment 3: location without the side \((\d+)", primary).group(1))
    cats = [("Correct side", BLUE), ("Wrong side", ORANGE), ("Both sides", AQUA), ("Not located / no answer", GREY)]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    rows = list(reversed(e3))
    for i, r in enumerate(rows):
        correct = int(nums(r[1])[0])
        wrong, both, none = int(r[2]), int(r[3]), int(r[4])
        noanswer = int(r[5]) if r[5].strip().isdigit() else 0
        left = 0
        for (label, col), v in zip(cats, (correct, wrong, both, none + noanswer)):
            if v:
                ax.barh(i, v - 0.08, left=left + 0.04, color=col, height=0.55)  # gap between segments
                ax.text(left + v / 2, i, str(v), ha="center", va="center", color=INK, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
            left += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, n_cases)
    ax.set_xlabel(f"One-sided ground-truth findings (of {n_cases})")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=col, label=label) for label, col in cats], loc="upper center",
              bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False, fontsize=9)
    frame(ax, "CheXagent gets the side right most often; MAIRA-2 more often picks the wrong side",
          "PRELIMINARY · each model located the finding by name only (no side given)")
    fig.tight_layout()
    fig.savefig(out / "3_side.png", dpi=200)
    plt.close(fig)


def chart_external(out, source="maira_vs_chexagent_external.md", name="4_external_auc.png", scale=1.0):
    """CheXagent vs MAIRA-2 AUC on the three held-out sets, one panel each. scale < 1 shrinks the canvas,
    so text appears larger when the figure is fitted to a page."""
    rows = [r for r in table((RESULTS / source).read_text(encoding="utf-8"), "| dataset") if not r[0].startswith("**")]
    for r in rows:
        r[1] = r[1].replace(" (VinDr Nodule/Mass label)", "*")
    sets = list(dict.fromkeys(r[0] for r in rows))
    findings = list(dict.fromkeys(r[1] for r in rows))
    findings = [f.rstrip("*") for f in findings]
    findings = list(dict.fromkeys(findings))
    for r in rows:
        r[1] = r[1].rstrip("*")
    fig, axes = plt.subplots(1, len(sets), figsize=(13 * scale, (2.2 + 0.42 * len(findings)) * max(scale, 0.85)), sharey=True)
    for ax, ds in zip(axes, sets):
        for r in [r for r in rows if r[0] == ds]:
            i = len(findings) - 1 - findings.index(r[1])
            c, cl, ch = nums(r[3])[:3]
            m, ml, mh = nums(r[4])[:3]
            ax.plot([m, c], [i, i], color=GRID, linewidth=2, zorder=1)
            for v, lo, hi, col in ((m, ml, mh, ORANGE), (c, cl, ch, BLUE)):
                ax.plot([lo, hi], [i, i], color=col, linewidth=2, alpha=0.45, solid_capstyle="round", zorder=2)
                ax.scatter([v], [i], s=55, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if not r[6].startswith("supported"):
                ax.text(1.0, i + 0.28, "not significant", color=MUTED, fontsize=7.5, ha="right")
            pos, n_img = (int(v) for v in re.findall(r"\d+", r[2])[:2])
            ax.text(0.19, i, f"{pos} of {n_img:,}", color=INK2, fontsize=8, va="center", ha="left")
        ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        n = re.search(r"/ (\d+)", next(r[2] for r in rows if r[0] == ds)).group(1)
        ax.set_title(f"{ds} ({int(n):,} images)", fontsize=10.5, color=INK, loc="left")
        ax.set_xlim(0.18, 1.02)
        ax.set_ylim(-0.7, len(findings) - 0.3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("AUC")
    axes[0].set_yticks(range(len(findings)), [f.capitalize() for f in reversed(findings)])
    axes[0].scatter([], [], s=55, color=BLUE, label="CheXagent")
    axes[0].scatter([], [], s=55, color=ORANGE, label="MAIRA-2")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False, ncol=2, fontsize=10)
    supported = sum(r[6].startswith("supported") for r in rows)
    fig.suptitle(f"CheXagent separates present from absent better than MAIRA-2 in {supported} of {len(rows)} comparisons",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.99)
    fig.text(0.01, 0.885, "AUC with 95% interval on three labelled evaluation sets; dashed line = chance (0.5). "
                          "Gaps: a condition not labelled in that dataset.\nLeft numbers: images with the condition of all "
                          "images tested. *VinDr labels nodule and mass together.", fontsize=9.5, color=INK2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    fig.savefig(out / name, dpi=200)
    plt.close(fig)


def chart_difference(out, source="maira_vs_chexagent_all_conditions.md", name="6_difference_forest.png"):
    """Forest plot: CheXagent minus MAIRA-2 AUC with 95% CI per dataset and condition; line at 0."""
    rows = [r for r in table((RESULTS / source).read_text(encoding="utf-8"), "| dataset") if not r[0].startswith("**")]
    order = ["cardiomegaly", "pleural effusion", "pneumothorax", "consolidation", "pulmonary edema", "atelectasis",
             "pneumonia", "infiltration", "emphysema", "nodule", "mass", "fibrosis", "pleural thickening"]
    sets = ["NIH test", "Indiana", "VinDr"]
    rows.sort(key=lambda r: (sets.index(r[0]), order.index(r[1].split(" (")[0])))
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(rows) + 2.6))
    y, ticks, labels, group_y = 0, [], [], []
    for ds in sets:
        group = [r for r in rows if r[0] == ds]
        group_y.append((y, ds, len(group)))
        for r in group:
            d, lo, hi = (float(v) for v in re.findall(r"[-+]\d*\.\d+", r[5])[:3])
            supported = r[6].startswith("supported")
            col = BLUE if supported else MUTED
            ax.plot([lo, hi], [-y, -y], color=col, linewidth=2, alpha=0.55 if supported else 1, solid_capstyle="round")
            ax.scatter([d], [-y], s=45, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if not supported:
                note = "too few positives" if "fewer" in r[6] else "not significant"
                ax.text(hi + 0.015, -y, note, va="center", color=MUTED, fontsize=8)
            ticks.append(-y)
            cond = r[1].split(" (")[0]
            pos, n = (int(v) for v in re.findall(r"\d+", r[2])[:2])
            labels.append(cond.capitalize() + ("*" if "Nodule/Mass" in r[1] else "") + f"  ({pos} of {n:,})")
            y += 1
        y += 0.8
    for y0, ds, n in group_y:
        n_ds = int(re.findall(r"\d+", next(r[2] for r in rows if r[0] == ds))[1])
        from matplotlib.transforms import blended_transform_factory
        ax.text(-0.01, -y0 + 0.75, f"{ds} ({n_ds:,} images)", fontsize=10, fontweight="bold", color=INK, ha="right",
                transform=blended_transform_factory(ax.transAxes, ax.transData))
    ax.axvline(0, color=INK2, linewidth=1)
    ax.text(0.005, 1.0, "MAIRA-2 better  ←  0  →  CheXagent better", transform=ax.get_xaxis_transform(),
            fontsize=8.5, color=INK2, ha="left", va="bottom")
    ax.set_yticks(ticks, labels)
    ax.set_xlim(-0.13, 0.75)
    ax.set_xlabel("Difference in AUC (CheXagent minus MAIRA-2), with 95% interval")
    supported = sum(r[6].startswith("supported") for r in rows)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    # tall figure: title and subtitle placed in figure coordinates so they never overlap
    fig.suptitle(f"CheXagent's AUC is higher in {supported} of {len(rows)} comparisons, and lower in none",
                 x=0.02, ha="left", y=0.995, fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.968, "Blue: interval entirely above 0 (supported). Grey: interval includes 0. (n of N) = images with the condition "
                          "of all images tested. *VinDr labels nodule and mass together.", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out / name, dpi=200)
    plt.close(fig)


def chart_edv_improvement(out, name="7_edv_improvement_auc.png"):
    """MedGemma alone vs + CheXagent vs + MAIRA-2: AUC per condition with 95% CIs, NIH test split
    (systems and bootstrap as in edv_improvement.py)."""
    import sys
    import numpy as np
    sys.path[:0] = [str(ROOT / "experiments")]
    import edv_improvement as E
    from maira_score_nih import auc

    truth, mg, chex, maira, dev = E.load()
    files = E.split_files("test")
    pct = lambda model, c, s: np.searchsorted(dev[(model, c)], s, side="right") / len(dev[(model, c)])  # noqa: E731
    arr = lambda d: {c: np.array([d[(f, c)] for f in files]) for c in E.CONDITIONS}  # noqa: E731
    y, m, ch, ma = arr(truth), arr(mg), arr(chex), arr(maira)
    systems = {"MedGemma alone": (m, MUTED),
               "MedGemma + CheXagent": ({c: 0.5 * m[c] + 0.5 * pct("chex", c, ch[c]) for c in E.CONDITIONS}, BLUE),
               "MedGemma + MAIRA-2": ({c: 0.5 * m[c] + 0.5 * pct("maira", c, ma[c]) for c in E.CONDITIONS}, ORANGE)}
    main = [c for c in E.CONDITIONS if y[c].sum() >= 10]
    rng = np.random.default_rng(E.SEED)
    boots = [rng.integers(0, len(files), len(files)) for _ in range(E.N_BOOT)]

    def stats(v, c):
        a = auc(v[c][y[c] == 1], v[c][y[c] == 0])
        bs = [auc(v[c][b][y[c][b] == 1], v[c][b][y[c][b] == 0]) for b in boots]
        return a, *np.percentile([x for x in bs if np.isfinite(x)], [2.5, 97.5])

    rows = ["Mean (12 conditions)"] + E.CONDITIONS
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(rows) + 2.2))
    for i, label in enumerate(rows):
        yy = len(rows) - 1 - i
        for s, (v, col) in systems.items():
            if i == 0:
                per = {c: stats(v, c)[0] for c in main}
                a = np.mean(list(per.values()))
                bs = [np.nanmean([auc(v[c][b][y[c][b] == 1], v[c][b][y[c][b] == 0]) for c in main]) for b in boots[:500]]
                lo, hi = np.percentile(bs, [2.5, 97.5])
            else:
                a, lo, hi = stats(v, label)
            ax.plot([lo, hi], [yy, yy], color=col, linewidth=2, alpha=0.45, solid_capstyle="round", zorder=2)
            ax.scatter([a], [yy], s=60 if i else 85, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axhline(len(rows) - 1.5, color=GRID, linewidth=1)
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    labels = [r if i == 0 else r.capitalize() + (" †" if r not in main else "") for i, r in enumerate(rows)]
    ax.set_yticks(range(len(rows)), list(reversed(labels)))
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUC (1.0 = perfect separation of present vs absent)")
    for s, (_, col) in systems.items():
        ax.scatter([], [], s=60, color=col, label=s)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=3, frameon=False, fontsize=9.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.suptitle("Adding EDV raises MedGemma's AUC on every condition; CheXagent more than MAIRA-2",
                 x=0.02, ha="left", y=0.995, fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.945, f"Base MedGemma 1.5 reports on {len(files):,} labelled NIH test images; AUC with 95% interval; "
                          "dashed line = chance. † fewer than 10 positives (not in the mean).", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / name, dpi=200)
    plt.close(fig)


def chart_hypotheses(out, name="8_hypotheses_three_experiments.png",
                     title="Adding CheXagent's image check improves MedGemma reports in all three experiments"):
    """H1/H2 (H3/H3b for the trained model) in all three EDV experiments: mean AUC per system (left) and
    the paired differences with 95% CIs (right); numbers from edv_improvement.md (base MedGemma, single
    image), edv_longitudinal.md (base, prior + current) and edv_trained.md (team's trained MedGemma)."""
    signed = lambda cell: [float(x) for x in re.findall(r"[+-]?\d*\.\d+", cell)]  # noqa: E731
    specs = (("edv_improvement.md", "Base MedGemma, single image (1,500 images; 12 conditions)", "o", False,
              "## Primary: mean AUC", ("MedGemma alone", "MedGemma + CheXagent", "MedGemma + MAIRA-2")),
             ("edv_longitudinal.md", "Base MedGemma, prior + current (499 pairs; 9 conditions)", "D", True,
              "## Mean AUC", ("MedGemma-L alone", "MedGemma-L + CheXagent A", "MedGemma-L + MAIRA-2-L")),
             ("edv_trained.md", "Team's trained MedGemma, prior + current (499 pairs; 9 conditions)", "s", True,
              "## Mean AUC", ("Trained MedGemma alone", "Trained + CheXagent A", "Trained + MAIRA-2-L")))
    runs = []
    for source, label, marker, filled, heading, sys_names in specs:
        text = (RESULTS / source).read_text(encoding="utf-8")
        means = {r[0]: nums(r[1]) for r in table(text, heading)}
        cells = [[c.strip() for c in l.strip().strip("|").split("|")] for l in text.splitlines() if l.startswith("| H")]
        hyp = [signed(c[1]) for c in cells][:2]  # H1, H2 / H1-L, H2-L / H3, H3b, in that order
        runs.append((label, marker, filled, [means[n] for n in sys_names], hyp))

    offs = [0.24, 0.0, -0.24]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    systems = [("MedGemma alone", MUTED), ("MedGemma + CheXagent", BLUE), ("MedGemma + MAIRA-2", ORANGE)]
    for j, (label, marker, filled, means, _) in enumerate(runs):
        for i, ((sname, col), (a, lo, hi)) in enumerate(zip(systems, means)):
            yy = len(systems) - 1 - i + offs[j]
            a1.plot([lo, hi], [yy, yy], color=col, linewidth=2, alpha=0.45, solid_capstyle="round")
            a1.scatter([a], [yy], s=60, marker=marker, color=col if filled else SURFACE, edgecolor=col, linewidth=1.8,
                       zorder=3)
            a1.text(hi + 0.012, yy, f"{a:.3f}", va="center", fontsize=8, color=INK2)
    a1.set_yticks(range(len(systems)), [s for s, _ in reversed(systems)])
    a1.set_xlim(0.5, 0.95)
    a1.set_xlabel("Mean AUC over conditions with at least 10 positives (95% interval)")
    frame(a1, "Mean AUC", "", pad=8)
    hyps = [("CheXagent added\nvs MedGemma alone\n(H1 / H3)", 0), ("CheXagent added\nvs MAIRA-2 added\n(H2 / H3b)", 1)]
    for j, (label, marker, filled, _, diffs) in enumerate(runs):
        for i, (hname, k) in enumerate(hyps):
            d, lo, hi = diffs[k]
            yy = len(hyps) - 1 - i + offs[j] * 0.8
            a2.plot([lo, hi], [yy, yy], color=BLUE, linewidth=2.2, alpha=0.5, solid_capstyle="round")
            a2.scatter([d], [yy], s=60, marker=marker, color=BLUE if filled else SURFACE, edgecolor=BLUE, linewidth=1.8,
                       zorder=3)
            a2.text(hi + 0.008, yy, f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=8, color=INK2)
    a2.axvline(0, color=INK2, linewidth=1.2)
    a2.text(0.006, 1.6, "no difference (supported = whole interval right of this line)", fontsize=8, color=MUTED)
    a2.set_yticks(range(len(hyps)), [h for h, _ in reversed(hyps)])
    a2.set_ylim(-0.55, 1.75)
    a2.set_xlim(-0.02, 0.52)
    a2.set_xlabel("Difference in mean AUC (95% interval)")
    frame(a2, "All hypotheses supported in all three", "", pad=8)
    for label, marker, filled, _, _ in runs:
        a1.scatter([], [], s=55, marker=marker, color=MUTED if filled else SURFACE, edgecolor=MUTED, linewidth=1.8,
                   label=label)
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title,
                 x=0.02, ha="left", y=0.99, fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.93, "Labelled NIH test images. Base = Google's MedGemma 1.5; trained = the team's model. "
                         "Image-level bootstrap, 2,000 resamples.", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out / name, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    import sys
    if "--hypotheses" in sys.argv:
        out = ROOT / "logs" / f"charts_{time.strftime('%Y%m%d-%H%M%S')}"
        out.mkdir(parents=True)
        chart_hypotheses(out)
        print(f"-> {out}")
        sys.exit()
    if "--edv-improvement" in sys.argv:
        out = ROOT / "logs" / f"charts_{time.strftime('%Y%m%d-%H%M%S')}"
        out.mkdir(parents=True)
        chart_edv_improvement(out)
        print(f"-> {out}")
        sys.exit()
    if "--difference" in sys.argv:
        out = ROOT / "logs" / f"charts_{time.strftime('%Y%m%d-%H%M%S')}"
        out.mkdir(parents=True)
        chart_difference(out)
        print(f"-> {out}")
        sys.exit()
    out = ROOT / "logs" / f"charts_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    if "--external" in sys.argv:
        chart_external(out)
    elif "--all-conditions" in sys.argv:
        chart_external(out, "maira_vs_chexagent_all_conditions.md", "5_all_conditions_auc.png")
    else:
        chart_auc(out)
        chart_caught(out)
        chart_side(out)
    print(f"-> {out}")
