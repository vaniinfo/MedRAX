"""Manuscript draft (Word), revision 13 (2026-11-02; earlier: revision 2 after internal review, 2026-10-25).

Numbers come from the committed results files and from experiments/results/review_analyses.md
(exploratory analyses requested in review). Builds the method-overview and laterality-conditioning
figures and a larger-font presence chart into experiments/results/figures/.

    python experiments/build_manuscript.py   -> logs/manuscript_draft_<time>.docx
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as pa  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.shared import Pt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_detection_grounding_doc import AUC_ROWS, FIG, Doc  # noqa: E402

ROOT = HERE.parent
BLUE, ORANGE, GREEN, MUTED, INK = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#0b0b0b"


# ------------------------------------------------------------------ figures

def overview_figure(path):
    """Figure 1: A, laterality conditioning and evidence accuracy; B, downstream use."""
    INK, MUTED, LINE = "#16233a", "#5f6470", "#1f2d44"
    GREY, BLUE, GREEN, ORANGE, PURPLE, YELLOW, TEAL = ("#ececec", "#dbe7ef", "#dff1e8", "#fde6d8", "#e6dcef",
                                                        "#fff4d1", "#d4ebe3")
    fig, ax = plt.subplots(figsize=(12, 13.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.5, 110)
    ax.axis("off")

    def box(x, y, w, h, title, body="", fc=GREY, size=10.5):
        ax.add_patch(pa.Rectangle((x, y), w, h, fc=fc, ec=LINE, lw=1.3))
        if body:
            ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=size, fontweight="bold",
                    color=INK)
            ax.text(x + w / 2, y + h - 4.1, body, ha="center", va="top", fontsize=size - 1.2, color=INK,
                    linespacing=1.35)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size, fontweight="bold",
                    color=INK)

    def path_arrow(points):
        xs, ys = zip(*points)
        ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=LINE, lw=1.3, solid_capstyle="butt")
        ax.annotate("", points[-1], points[-2], arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3,
                                                                 mutation_scale=13))

    # ======================= Panel A
    ax.text(0, 108.5, "A   Does the prompt decide the evidence? (pre-registered; CheXagent and MAIRA-2)",
            fontsize=13, fontweight="bold", color=INK, va="top")
    ax.text(0, 105.2, "Same image and checker; only the localisation prompt differs. 153 one-sided VinDr-CXR "
                      "findings.", fontsize=10, color=MUTED, va="top")
    rows = [
        ("Side-free prompt", "\"Locate pleural effusion\"", BLUE,
         "Side of the returned box is\ncompared with the true side afterwards",
         "Box on true / opposite side:\nCheXagent 99% / 14%; MAIRA-2 54% / 41%\n"
         "LR+ 6.73 (4.78-10.52) vs 1.31 (inconclusive)"),
        ("Prompt names the true side", "\"Locate right pleural effusion\"", GREY,
         "Box on the named side", "CheXagent 100%, MAIRA-2 99%"),
        ("Prompt names the opposite side", "\"Locate left pleural effusion\"", ORANGE,
         "Box on the named (wrong) side", "CheXagent 86%, MAIRA-2 99%"),
    ]
    for i, (t, q, fc, mid, res) in enumerate(rows):
        y = 91 - i * 11.5
        box(1, y, 27, 9.5, t, q, fc=fc, size=10.5)
        box(36, y, 26, 9.5, "Checker returns a box", mid, fc="#f6f6f6", size=10.5)
        box(70, y, 29, 9.5, "Result", res, fc=TEAL if i == 0 else "#fbe3e3", size=10.5)
        path_arrow([(28, y + 4.75), (36, y + 4.75)])
        path_arrow([(62, y + 4.75), (70, y + 4.75)])
    ax.add_patch(pa.Rectangle((1, 58.3), 98, 6.7, fc="#f6f6f6", ec=LINE, lw=1.0, ls=(0, (4, 3))))
    ax.text(50, 61.65, "Also evaluated with no report involved: report-blinded presence scores for 13 conditions "
                       "against dataset labels\n(NIH 1,500 · Indiana 544 · VinDr 3,000 images) and boxes against "
                       "947 NIH radiologist boxes.", ha="center", va="center", fontsize=10, color=INK,
            linespacing=1.4)

    # ======================= Panel B
    ax.plot([0, 100], [55.6, 55.6], color="#c9ccd2", lw=1)
    ax.text(0, 54.2, "B   Downstream application to generated reports (NIH; presence only)", fontsize=13,
            fontweight="bold", color=INK, va="top")
    box(1, 44, 20, 6, "Report prompt*", fc=GREY)
    box(25, 44, 18, 6, "Current X-ray", fc=GREY)
    box(47, 44, 18, 6, "Prior X-ray†", fc=GREY)
    box(16, 32, 32, 8, "MedGemma", "base, or fine-tuned for comparison reports", fc=BLUE)
    box(56, 32, 24, 8, "Generated report", "findings and impression", fc="#f3f6f8")
    path_arrow([(11, 44), (11, 42), (24, 42), (24, 40)])
    path_arrow([(34, 44), (34, 40)])
    path_arrow([(56, 44), (56, 42), (42, 42), (42, 40)])
    path_arrow([(48, 36), (56, 36)])

    # images to the checker (no report): along the top-right
    ax.plot([34, 34], [50, 52.2], color=LINE, lw=1.3)
    ax.plot([56, 56], [50, 52.2], color=LINE, lw=1.3)
    ax.plot([34, 92], [52.2, 52.2], color=LINE, lw=1.3)
    path_arrow([(92, 52.2), (92, 26)])
    ax.text(90.5, 47, "images only:\nno report text,\nno assertion", ha="right", va="center", fontsize=9.5,
            color=MUTED, linespacing=1.3)

    box(1, 14, 38, 12, "Report reader (Qwen)",
        "condition scores from the report text:\n1 present · 0.5 possible · 0 not stated\n"
        "an answer whose quote is not in the report\ncounts as not stated", fc=YELLOW)
    path_arrow([(68, 32), (68, 29.5), (20, 29.5), (20, 26)])

    box(54, 15, 44, 11, "Report-blinded checker",
        "CheXagent or MAIRA-2\n\"Is there {condition}?\" for 13 fixed conditions\nno report text, no polarity",
        fc=PURPLE)
    box(54, 4.5, 44, 8, "Presence evidence",
        "CheXagent: calibrated likelihood-ratio states‡\nMAIRA-2: uncalibrated score", fc="#efe9f5")
    path_arrow([(76, 15), (76, 12.5)])

    box(1, 0, 46, 10.5, "Compare with NIH labels",
        "score fusion (both checkers) · editing rule (CheXagent)\nwith report-only and checker-only baselines\n"
        "side not compared here (NIH has no side labels)", fc=GREEN)
    path_arrow([(20, 14), (20, 10.5)])
    path_arrow([(54, 8.5), (47, 8.5)])

    fig.text(0.02, 0.012, "* Base MedGemma: report prompt. Fine-tuned MedGemma: its training input format, with "
                          "placeholders for the prior report and indication.\n† Paired settings only.   "
                          "‡ Frozen engine for six conditions; provisional cut-offs for seven.\nAutomatic report "
                          "correction is not established: most additions by the editing rule disagreed with the "
                          "reference labels.", fontsize=8.8, color=MUTED, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def conditioning_figure(path):
    data = {"CheXagent": {"bt": 99, "bw": 14, "ct": 100, "cw": 86}, "MAIRA-2": {"bt": 54, "bw": 41, "ct": 99, "cw": 99}}
    keys = ["bt", "bw", "ct", "cw"]
    labels = ["Side-free:\ntrue side", "Side-free:\nopposite side", "Prompt names\ntrue side", "Prompt names\nopposite side"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (model, vals) in zip(axs, data.items()):
        bars = ax.bar(range(4), [vals[k] for k in keys], color=[GREEN, "#a8dcc6", BLUE, ORANGE], width=0.65)
        for b, k in zip(bars, keys):
            ax.text(b.get_x() + b.get_width() / 2, vals[k] + 2, f"{vals[k]}%", ha="center", fontsize=11)
        ax.set_xticks(range(4), labels, fontsize=10)
        ax.set_title(model, fontsize=13, fontweight="bold", loc="left")
        ax.set_ylim(0, 112)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("Box on the side in question (%)", fontsize=11)
    fig.suptitle("When the prompt names a side, both checkers localise that side, whether true or not",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.875, "153 one-sided VinDr-CXR findings (pre-registered). A box crossing the midline counts for "
                          "both sides, so the side-free columns can overlap.", fontsize=9.5, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def presence_chart(out_dir):
    """Presence AUC by dataset (chart 5) on a smaller canvas, so its labels are larger on the page."""
    import model_charts as MC
    name = "manuscript_fig3_presence_auc.png"
    MC.chart_external(out_dir, "maira_vs_chexagent_all_conditions.md", name, scale=0.62)
    return out_dir / name


def fusion_chart(out_dir):
    """Mean AUC of report alone, each checker alone and each fused score, per setting, with 95% CIs
    (image-level), read from results/review_analyses.md."""
    import re
    text = (HERE / "results" / "review_analyses.md").read_text(encoding="utf-8")
    blocks = re.split(r"^### ", text, flags=re.M)[1:4]
    systems = ["report alone", "CheXagent alone", "report + CheXagent", "MAIRA-2 alone", "report + MAIRA-2"]
    shown = ["Report alone", "CheXagent alone", "Fused: report + CheXagent", "MAIRA-2 alone", "Fused: report + MAIRA-2"]
    vals = []  # per setting: {system: (auc, lo, hi)}
    for b in blocks:
        v = {}
        for sname in systems:
            m = re.search(r"^\| " + re.escape(sname) + r" \| ([\d.]+) \| ([\d.]+)-([\d.]+) \|", b, flags=re.M)
            v[sname] = tuple(float(x) for x in m.groups())
        vals.append(v)
    settings = [("(a) Base MedGemma, single image (1,500)", "o", "#ffffff"),
                ("(b) Base MedGemma, prior + current (499)", "D", None),
                ("(c) Trained MedGemma, prior + current (499)", "s", None)]
    colors = [MUTED, BLUE, BLUE, ORANGE, ORANGE]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, sname in enumerate(systems):
        yc = len(systems) - 1 - i
        for j, ((label, marker, face), v) in enumerate(zip(settings, vals)):
            a, lo, hi = v[sname]
            y = yc + 0.22 - 0.22 * j
            ax.plot([lo, hi], [y, y], color=colors[i], alpha=0.45, lw=2.5, solid_capstyle="round")
            ax.scatter([a], [y], marker=marker, s=70, color=face or colors[i], edgecolor=colors[i], linewidth=1.8,
                       zorder=3, label=label if i == 0 else None)
            ax.text(hi + 0.006, y, f"{a:.3f}", va="center", fontsize=10, color="#52514e")
    ax.set_yticks(range(len(systems)), list(reversed(shown)), fontsize=12)
    ax.set_xlim(0.5, 0.95)
    ax.set_xlabel("Mean AUC over conditions with at least 10 positives (95% CI)", fontsize=11)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", color="#e1e0d9")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    from matplotlib.lines import Line2D
    proxies = [Line2D([], [], marker=mk, linestyle="", markersize=8, markeredgewidth=1.8, markeredgecolor=MUTED,
                      markerfacecolor=face or MUTED) for _, mk, face in settings]
    ax.legend(proxies, [lab for lab, _, _ in settings], loc="lower center", bbox_to_anchor=(0.42, -0.44),
              ncol=1, frameon=False, fontsize=10)
    fig.suptitle("Adding CheXagent raises detection well above MedGemma's findings alone; CheXagent alone is similar",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    name = "manuscript_fig4_fusion_with_baselines.png"
    fig.savefig(out_dir / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_dir / name


def tidy_tables(document):
    """Repeat each table's header row on new pages and keep rows from splitting across pages."""
    for t in document.tables:
        for i, row in enumerate(t.rows):
            trPr = row._tr.get_or_add_trPr()
            trPr.append(OxmlElement("w:cantSplit"))
            if i == 0:
                trPr.append(OxmlElement("w:tblHeader"))
            if len(t.rows) <= 15 and i < len(t.rows) - 1:  # keep short tables on one page
                for cell in row.cells:
                    for par in cell.paragraphs:
                        par.paragraph_format.keep_with_next = True


# ------------------------------------------------------------------ manuscript

def build(d, fig1, fig2, fig3, fig4):
    t = d.d.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Longitudinal Chest X-ray Report Generation and Report-Blinded Verification Using Medical "
                  "Vision-Language Models")
    r.bold = True
    r.font.size = Pt(16)
    a = d.d.add_paragraph()
    a.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = a.add_run("[Authors and affiliations]   ·   DRAFT, revision 13, for internal review   ·   November 2026")
    r.italic = True
    r.font.size = Pt(10)

    # ---------------- abstract
    d.h("Abstract", 1)
    d.p("**Purpose.** To fine-tune MedGemma for longitudinal chest X-ray reporting and to test whether "
        "report-blinded CheXagent evidence improves identification of conditions in generated reports, and "
        "whether naming a laterality biases visual grounding.")
    d.p("**Materials and Methods.** MedGemma was fine-tuned on MIMIC-CXR comparison reports to describe the "
        "current radiograph relative to the prior. CheXagent assessed 13 conditions and their location without "
        "access to the report. On 499 NIH ChestX-ray14 prior-current pairs, condition scores read from base and "
        "fine-tuned MedGemma reports were evaluated alone and combined with CheXagent evidence; grounding was "
        "compared with 947 radiologist boxes. In a pre-registered experiment, CheXagent and MAIRA-2 localized 153 "
        "unilateral VinDr-CXR findings with no, the correct or the incorrect side named. Two radiologists "
        "independently evaluated a selected sample of 100 prior-current pairs (60 held-out MIMIC-CXR, 20 CheXpert "
        "Plus, 20 NIH ChestX-ray14), rating on a five-point scale the accuracy of MedGemma findings, CheXagent "
        "diagnostic evidence, MedGemma longitudinal change descriptions and CheXagent anatomical grounding.")
    d.p("**Results.** Fine-tuned MedGemma referred to the prior examination in 487 of 499 reports (97.6%), "
        "versus 3 of 499 (0.6%) for base MedGemma. Adding CheXagent evidence increased mean AUC from 0.593 to "
        "0.837 for fine-tuned MedGemma (difference, 0.244; 95% CI: 0.220, 0.269) and from 0.545 to 0.835 for "
        "base MedGemma; CheXagent alone achieved 0.840. With the incorrect side named, CheXagent returned that "
        "side in 86% of findings and MAIRA-2 in 99%; without a named side, CheXagent's box side discriminated "
        "correct from incorrect sides (positive likelihood ratio, 6.73; 95% CI: 4.78, 10.52). [Radiologist "
        "evaluation results.]")
    d.p("**Conclusion.** Fine-tuned MedGemma produced comparison-style reports, and report-blinded CheXagent "
        "evidence substantially improved condition discrimination over MedGemma findings, though not beyond "
        "CheXagent alone. Because naming a side biased grounding, verification evidence should be collected "
        "without the asserted attribute and used to flag findings for review rather than to edit reports.")

    # ---------------- introduction
    d.h("1. Introduction", 1)
    d.p("Generative vision-language models can draft chest X-ray reports [MAIRA-2; MedGemma; CheXagent]. "
        "Their drafts can contain the errors a reader must catch before sign-off: conditions stated but "
        "absent, conditions missed, and correct findings placed on the wrong side. Wrong-side statements are "
        "a recognised patient-safety error in radiology reports [Kathait 2024; CorBenchX], and "
        "language models answer left-right questions poorly [Other Left]. Validating each stated "
        "finding against the image is one safeguard; for it to help, the check must not simply repeat the "
        "draft. Clinical reports also compare the current study with a prior one, which the base "
        "MedGemma rarely did in our data (3 of 499 reports).")
    d.p("**The system studied.** We address both halves of this problem in one workflow (Figure 1): (1) a "
        "MedGemma model fine-tuned on MIMIC-CXR comparison reports reads the prior and current radiographs, "
        "with the prior report and clinical indication, and writes a longitudinal report of the current study; "
        "(2) CheXagent independently evaluates a predefined list of chest X-ray conditions without receiving "
        "MedGemma's report or its assertions. For each condition, CheXagent produces a presence score that is "
        "interpreted using calibrated likelihood ratios; where localization is available, its predicted bounding "
        "box provides separate anatomical evidence. (3) This evidence is subsequently compared with the findings "
        "extracted from MedGemma's report to identify potentially unsupported statements and omitted "
        "conditions, which are flagged for the radiologist rather than edited automatically. Because the checker "
        "never sees the claim it is checked against, it cannot simply agree with it; this distinguishes the "
        "approach from passing each generated claim to another model and asking whether it agrees. Generation and "
        "verification are evaluated together: the generator's longitudinal reporting and findings, the "
        "reliability of the checker's presence and localization evidence, and the effect of that evidence on "
        "the generated findings. Two radiologists assess the clinical accuracy of MedGemma's reported findings "
        "and longitudinal change descriptions, as well as CheXagent's diagnostic evidence and anatomical "
        "grounding, using a five-point rating scale (Section 3.8).")
    d.p("Agent frameworks have begun to add such validation. CXRAgent [CXRAgent], built around a central "
        "multimodal \"Director\" model that calls chest X-ray tools (MAIRA-2 phrase grounding among them), "
        "introduced an Evidence-Driven Validator (EDV). For each diagnostic statement a tool produces, a "
        "multimodal model given the statement and the image writes the supporting and refuting visual "
        "evidence and a credibility judgement; the current version of the paper rates confidence High, Medium "
        "or Low and re-invokes another tool for low-rated statements. EDV improved downstream accuracy and "
        "report metrics [CXRAgent]. We build on its idea of tying each finding to image evidence.")
    d.p("In that design, the validating model receives the statement it checks. Tools that localise "
        "findings receive a phrase as well. We ask whether such inputs shape the returned evidence, using "
        "laterality in grounding models as a test case: a grounding model asked to localise \"left pleural "
        "effusion\" may localise the left side because the phrase says so, whatever the image shows. We "
        "measure this directly, and evaluate an alternative in which evidence is collected without the report's "
        "assertion. We did not test EDV's multimodal validator itself.")
    d.p("Dedicated fact-checking models are a second line of prior work. Mahmood et al. trained an "
        "image-text model on about 27 million synthetic real and perturbed finding-location pairs to judge "
        "whether each finding in a generated report is real and to predict its location [Mahmood 2025]. "
        "Checking both presence and location against the image has therefore been done. Our question is "
        "different: we study general-purpose checkers that are not trained for fact-checking, measure in "
        "controlled experiments how their evidence depends on the assertion in the prompt, and evaluate "
        "evidence collected without it. We did not test dedicated fact-checking models, so our results do "
        "not show whether they share this behaviour.")
    d.p("More recently, CLEAR [Zou 2026] investigated visually unsupported clinical statements using "
        "phrase-conditioned grounding and localized counterfactual image perturbations. Rather than treating "
        "grounding alone as proof of a finding, CLEAR evaluates how the model's responses and image-phrase "
        "alignment change when candidate image regions are perturbed. Our study examines a complementary "
        "question: whether the assertion supplied to a verifier influences the evidence it produces. We evaluate "
        "this effect experimentally through laterality-conditioned prompting, and assess evidence obtained "
        "without supplying the asserted side.")
    d.p("**Terminology.** *Report-blinded evidence:* the checker receives the image and the name of a condition "
        "from a fixed list, never the report text or whether the report asserts the condition. The checker "
        "still knows which condition is being tested. *Laterality-blinded localisation:* the localisation "
        "prompt names the condition but not its side; the returned box's side is compared with the asserted "
        "side afterwards. *Laterality-conditioned localisation:* the prompt includes a side.")
    d.p("**Research questions.**")
    d.table(["Research question", "Focus"], [
        ("RQ1 - Longitudinal reporting", "Does fine-tuning MedGemma improve longitudinal report generation, and "
         "how accurate are its findings and interval-change descriptions?"),
        ("RQ2 - Verification conditioning", "Does supplying an asserted laterality bias visual grounding, and can "
         "laterality-blinded localization provide informative evidence?"),
        ("RQ3 - Evidence reliability", "How accurately does report-blinded CheXagent identify conditions and "
         "localize findings, including comparison with an alternative checker (MAIRA-2)?"),
        ("RQ4 - Downstream integration", "Does combining MedGemma findings with CheXagent evidence improve (a) "
         "condition identification - discrimination of present from absent conditions - and (b) report "
         "auditing - detection of unsupported statements and clinically important omissions in the generated "
         "report - beyond either model alone?"),
    ], widths=[4.5, 12], size=8.5, caption="Research questions.")
    d.p("How to score MAIRA-2, which has no native confidence score, is treated as part of RQ3's methodology "
        "rather than as a separate question. RQ4 separates two objectives that are not identical: improving "
        "condition identification, measured against reference labels, and auditing a report, which is the task "
        "in the proposed clinical workflow and is addressed by the radiologist evaluation.")
    d.table(["", "EDV (CXRAgent)", "Fact-checking model (Mahmood 2025)", "This work"], [
        ("Checker", "Multimodal language model (as the Director)", "Dedicated model trained on synthetic "
         "real / perturbed findings", "Off-the-shelf CheXagent or MAIRA-2, compared head-to-head"),
        ("Checker input", "The tool's statement and the image", "The image and the report finding "
         "(polarity and finding; whether the stated location is an input is not clear from the paper)",
         "The image and a condition name; no report assertion; no side in localisation"),
        ("Evidence produced", "Written supporting / refuting evidence; High / Medium / Low", "Real / fake "
         "label and a predicted box", "CheXagent: presence likelihood ratio with 95% CI. MAIRA-2: an "
         "uncalibrated presence score. Both: a box and its side"),
        ("Decision rule", "Judgement by the multimodal model", "Learned classifier output",
         "CheXagent only: predefined, frozen thresholds (Evidence Engine v1.0)"),
        ("Dependence on the assertion", "Not studied", "Not studied", "Measured for laterality (E1)"),
        ("Evaluation", "Downstream task accuracy and report metrics", "Accuracy, box overlap and agreement "
         "with ground-truth error scores", "Against labels and radiologist boxes (pre-registered); downstream "
         "fusion and editing, with checker-only baselines"),
    ], widths=[2.8, 4.2, 4.6, 4.9], size=8, caption="Prior approaches to checking generated findings against "
                                                   "the image, and this work (CXRAgent arXiv v2 and code at "
                                                   "commit 0514b489; Mahmood et al., MICCAI 2025).")
    d.p("**Contributions.**")
    d.bullets([
        "A MedGemma 1.5 4B model fine-tuned on MIMIC-CXR comparison reports for longitudinal reporting, which "
        "referred to the prior study in 487 of 499 NIH reports (base MedGemma: 3 of 499) and identified "
        "labelled conditions better than base MedGemma (mean AUC 0.593 vs 0.545), with a radiologist evaluation "
        "of its findings and described changes.",
        "A controlled, pre-registered measurement of laterality conditioning: naming a side in the "
        "localisation prompt largely determines the side two general-purpose chest X-ray models return, while "
        "laterality-blinded localisation by CheXagent discriminates true from false sides.",
        "A pre-registered comparison of two checkers' report-blinded evidence for presence (3 datasets, 13 "
        "conditions) and grounding (947 radiologist boxes), with a training-provenance analysis and a "
        "robustness analysis of how to score a model that only writes reports.",
        "Pre-registered downstream evaluations showing that adding report-blinded CheXagent evidence improves "
        "identification of conditions over MedGemma's findings alone, for base and fine-tuned MedGemma (+0.24 "
        "to +0.31 mean AUC), more than MAIRA-2 does; checker-only baselines locate the gain in CheXagent's "
        "detection, and an audit of automatic edits shows that the evidence should flag rather than edit.",
        "A frozen evidence-calibration engine, with protocols committed before the outputs they analyse.",
    ])

    # ---------------- related work
    d.h("2. Related work", 1)
    d.p("**Chest X-ray vision-language models.** CheXagent is an instruction-tuned chest X-ray model that "
        "answers questions and localises findings [CheXagent]. MAIRA-2 generates grounded findings from "
        "current and prior images [MAIRA-2]. MedGemma is a medical vision-language model that can draft "
        "reports [MedGemma; MedGemma 1.5].")
    d.p("**Evaluating generated reports against a reference.** Clinical-efficacy measures compare labels "
        "extracted from generated and reference reports [CheXbert]; entity-relation measures compare clinical "
        "entities and relations [RadGraph]; language-model judges count clinically significant errors "
        "[GREEN]; RadFact checks sentence entailment, including boxes [MAIRA-2]. All need a reference report.")
    d.p("**Checking generated findings against the image.** Mahmood et al. trained a dedicated "
        "fact-checking model that labels each report finding real or fake and predicts its box, evaluated on "
        "ChestX-ray8, VinDr-CXR, MS-CXR and Chest ImaGenome [Mahmood 2025]; the same group pairs it with "
        "language-model correction [Mahmood 2024]. Counter-evidence verification localises each claim with a "
        "grounding model, masks the region and tests whether the claim persists [CoEV]. Other methods turn "
        "claims into visual questions [UniVRSE], use the generator's hidden states [ReXTrust], re-measure "
        "specific findings with dedicated modules [FactCheXcker], or benchmark error correction, including "
        "side confusion [CorBenchX]. In the methods we found, the checker receives the claim it checks.")
    d.p("**Agentic systems and validation.** Agent frameworks coordinate chest X-ray tools under a "
        "language-model planner [MedRAX; CXRAgent]; our implementation builds on MedRAX. CXRAgent's EDV gives "
        "the statement and the image to a multimodal model. RadAgents prompts MAIRA-2 with a side-free "
        "pneumothorax phrase and assigns the side afterwards from lung masks, for its own reading rather than "
        "to audit a report [RadAgents]. In medical visual question answering, models that verify a given "
        "answer show agreement bias [Verification Mirage], and agents defer to findings supplied to them "
        "[Deference].")
    d.p("**Prompt dependence and laterality.** Medical vision-language models change answers when a "
        "diagnosis or premise is suggested [EchoBench; sycophancy], and swapping anatomical words in chest "
        "X-ray questions changes answers [HEAL-MedVQA]. Natural-image grounding models still return boxes for "
        "referring expressions whose attributes do not match the image [C-REC], and chest X-ray grounding degrades on "
        "flipped images [AnatomiX]. We found no controlled measurement of whether chest X-ray grounding "
        "models localise the side named in the prompt. Collecting verification evidence without the claim "
        "has a text-only precedent in factored chain-of-verification [CoVe].")
    d.p("**Confidence from vision-language models.** First-token yes/no probabilities from chest X-ray models "
        "have been calibrated with conformal prediction [CONRep]; studies of medical vision-language "
        "confidence stress errors at high confidence [Calibrated triage]. Head-to-head grounding comparisons "
        "of CheXagent and MAIRA-2 exist on other datasets and tasks, with rankings that depend on the task "
        "and prompt [RadVLM; AnatomiX; CheXOne].")
    d.p("**Prior-aware report generation.** MAIRA-2 conditions on prior images and reports [MAIRA-2]; "
        "Longitudinal-MIMIC and later methods use prior reports [Longitudinal-MIMIC]. Supervised fine-tuning "
        "followed by GRPO has been applied to chest X-ray reports [EditGRPO; CheXOne], including for interval "
        "change [STRIVE]; reinforcement learning after supervised training can reduce transfer to NIH "
        "[ChexReason], and generated reports can contain hallucinated comparisons [Banerjee DPO].")

    # ---------------- methods
    d.h("3. Methods", 1)
    d.h("3.1 Design and pre-registration", 2)
    d.fig(fig1, "Study overview. (A) Laterality conditioning: the same checker localised the same one-sided "
                "VinDr-CXR findings with a side-free prompt, a prompt naming the true side and a prompt naming the "
                "opposite side; report-blinded presence scores and boxes were also evaluated against reference "
                "labels and radiologist boxes. (B) Downstream application: MedGemma reports were read for 13 "
                "conditions, and report-blinded presence evidence was combined with them or used to edit them, "
                "evaluated against NIH labels with report-only and checker-only baselines.", width_cm=16)
    d.p("**Pre-registration**, as used here: for each main analysis, a protocol stating data, models, prompts, "
        "measures, hypotheses and decision rules was committed to version control before the model outputs it "
        "analyses were generated, or, where an amendment says so, generated but not yet read. Changes were made "
        "only as dated amendments. Supplementary S9 lists each protocol with its commit, date and the "
        "development evidence already known when it was written. Analyses added after results were known, "
        "including the checker-only baselines and patient-clustered intervals added in review, are labelled "
        "exploratory.")

    d.h("3.2 Datasets and patient separation", 2)
    d.table(["Dataset", "Images used", "Labels", "Analyses"], [
        ("NIH ChestX-ray14 [NIH]", "Random sample of 5,000 images, one per patient; 3,500 development, "
         "1,500 test", "Image-level labels text-mined by the dataset authors from the original reports (not "
         "released); 13 of 14 labels (hernia excluded)", "Presence (test); "
         "calibration and exploratory checks (development)"),
        ("NIH prior + current pairs", "The 499 test images whose patient has an earlier image; the prior is "
         "outside the 5,000 sample", "Current image's labels", "Longitudinal setting of E4"),
        ("NIH box subset", "947 radiologist boxes on 847 images (699 patients); test-split images excluded",
         "Radiologist boxes", "Grounding"),
        ("Indiana (Open-i) [Open-i]", "544 frontal images, selected per condition (enriched)", "Open-i's "
         "manually coded problem and MeSH terms for each report, mapped to 6 conditions by a fixed term list", "Presence; earlier CheXagent development (see text)"),
        ("VinDr-CXR [VinDr], Kaggle VinBigData release", "3,000 training-set images, each read independently by 3 radiologists",
         "Our rule (pre-registered): present if at least 2 of the 3 marked it; their boxes", "Presence; laterality (one-sided "
         "findings)"),
        ("MIMIC-CXR [MIMIC-CXR], the team's held-out pairs", "60 prior + current pairs (47 patients) from the "
         "held-out set, never used to train the fine-tuned MedGemma", "The radiologist's report of the current "
         "study (with its comparison to the prior)", "Radiologist evaluation"),
        ("CheXpert Plus [CheXpert Plus]", "20 prior + current pairs (consecutive frontal studies, one per patient) "
         "from 76,846 eligible", "The radiologist's report of the current study", "Radiologist evaluation"),
    ], widths=[3.6, 5, 4.4, 3.5], caption="Datasets.", size=8.5)
    d.fig(FIG / "manuscript_flow.png", "Study flow. How each analysis set was formed from the three datasets, with "
                                       "exclusions and the experiments that use each set.", width_cm=16)
    d.p("**Patient separation (exploratory audit).** The NIH sample contains one image per patient, so "
        "development (3,500 patients) and test (1,500 patients) share no patients, and no test patient "
        "contributes more than one test image. None of the 499 prior images is in either split. The 947 "
        "grounding findings come from 699 patients; 42 of these patients also have a different image in the "
        "test split. With one image per patient, patient-level and image-level resampling are the same design "
        "for the NIH presence and fusion analyses (Supplementary S7). The VinDr and Indiana files used provide no patient identifiers.")
    d.p("**Our own prior use of the evaluation data.** Exclusion from a model's training is different from "
        "exclusion from our research process. The same 544 Indiana images were our first CheXagent development "
        "set (2026-09-27): CheXagent's presence AUCs on them, a 70/30 split and an Indiana-fitted comparator "
        "calibration were produced then, before the E2 protocol. The decisions to use CheXagent as the default "
        "checker and \"Is there {condition}?\" as its query were made after those results were seen. No "
        "threshold or calibration used in any reported analysis was fitted on Indiana (the engine is fitted on "
        "NIH development data), and MAIRA-2 was first run on Indiana under the E2 protocol. Indiana is "
        "therefore held out from both models' documented training, but not untouched by our process. VinDr "
        "was first read for the engine's external validation, pre-registered on 2026-09-27.")

    d.h("3.3 Models and training provenance", 2)
    d.table(["Role", "Model", "Version and runtime"], [
        ("Checker", "CheXagent-2-3b [CheXagent]", "Snapshot 8f19b53a; bf16; transformers 4.46; greedy"),
        ("Checker", "MAIRA-2 [MAIRA-2]", "Revision 795a2b1c; bf16 (8-bit in the laterality test); greedy"),
        ("Report generator", "MedGemma 1.5 4B, base [MedGemma 1.5]", "Ollama 0.40.2 model medgemma1.5:4b, digest "
         "e3cf59767d36, Q4_K_M 4-bit quantisation; temperature 0. Its correspondence to Google's "
         "google/medgemma-1.5-4b-it weights was not verified"),
        ("Report generator", "MedGemma 1.5 4B, team-trained [MedGemma 1.5]", "Supervised then reinforcement-learning "
         "fine-tuning on MIMIC-CXR comparison reports [MIMIC-CXR]; merged weights, revision e346cb1; bf16; "
         "transformers 4.57.3; greedy"),
        ("Report reader", "Qwen 3.5, 27.3B parameters", "Ollama 0.40.2, digest 67f762aeb910, Q4_K_M; "
         "temperature 0; thinking disabled"),
    ], widths=[3, 4.8, 8.7], caption="Models, versions and runtimes.", size=8.5)
    d.table(["Dataset", "CheXagent development", "MAIRA-2 development", "Overlap with our evaluation images"], [
        ("NIH ChestX-ray14", "Included (supervised instruction data) [CheXagent]",
         "Image encoder (RAD-DINO-MAIRA-2) self-supervised on 112,120 NIH images; not in report training "
         "[MAIRA-2]", "Unknown for CheXagent; likely for MAIRA-2's image encoder, which used the full dataset"),
        ("VinDr-CXR", "Included (supervised instruction data) [CheXagent]", "Not listed", "Unknown; our images "
         "come from the public training release"),
        ("Open-i (Indiana)", "Excluded; held out for evaluation [CheXagent]", "Not trained on (model card)",
         "None documented"),
        ("MIMIC-CXR", "Included", "Included", "Not evaluated here"),
    ], widths=[3, 4.6, 4.6, 4.3], size=8.5,
       caption="Training provenance, from each model's paper or model card. That the CheXagent-2-3b checkpoint "
               "follows the data description of the CheXagent paper is assumed, not verified.")

    d.p("**Fine-tuned report generator.** The team fine-tuned MedGemma 1.5 4B to write comparison reports. "
        "Each training example pairs a prior and a current MIMIC-CXR image with the prior report and the "
        "clinical indication, and the target is the radiologist's report of the current study, which describes "
        "interval change and gives an impression (input format in Supplementary S1). Training used supervised "
        "fine-tuning followed by reinforcement learning (GRPO); the adapters were merged into the published "
        "weights (vaniinfoorg/medgemma_trained, revision e346cb1). [To be completed by the training team: "
        "number of studies and patients, split, reward function, hyperparameters, compute, and how the "
        "checkpoint was selected.] Fine-tuning used MIMIC-CXR only, so no NIH image was used in fine-tuning; "
        "MedGemma's own pretraining data are described in [MedGemma 1.5].")

    d.h("3.4 Evidence collection", 2)
    d.p("**Presence.** For every image, each checker is queried for all 13 conditions, regardless of what the "
        "report states. CheXagent: \"Is there {condition}?\"; the score (p_present) is the probability of "
        "\"Yes\" relative to \"No\" as the first answer token. MAIRA-2 has no question-answering mode, so its "
        "score is computed by teacher forcing after its standard report prompt. For a positive sentence "
        "(\"There is {a} {condition}.\") and a negative one (\"There is no {condition}.\"), we sum the token "
        "log-probabilities, including the final period, without an end-of-sequence token and without length "
        "normalisation. The score is P(positive) / (P(positive) + P(negative)). The processor-generated prompt "
        "for a single image contains the phrase \"in comparison to the prior frontal image\" even when no "
        "prior is supplied; we did not assess its effect.")
    d.p("**Localisation.** \"Please locate the following phrase: {condition}\" (CheXagent) or MAIRA-2's "
        "phrase-grounding input with the capitalised condition name. Side words are never included, except in "
        "the laterality-conditioned arms of E1. Exact prompts are in Supplementary S1.")
    d.p("**Box side.** Each box is assigned a side from its centre (the patient's right is the image's left). "
        "A box whose left edge lies left of 40% of the image width and whose right edge lies right of 60% "
        "counts for both sides. A case agrees with a side if that side is among its boxes' sides. A case "
        "without boxes agrees with neither side.")

    d.h("3.5 Evidence calibration (Evidence Engine v1.0)", 2)
    d.p("For six principal conditions, Platt calibration on NIH development data maps p_present to a "
        "calibrated probability q at the development prevalence π: logit q = a·logit(p) + b. The likelihood "
        "ratio is LR(p) = odds(q(p)) / odds(π), with a 95% CI from 1,000 Platt refits on bootstrap resamples, "
        "stored on a fixed grid. States: SUPPORTING if the lower 95% bound of LR is at least 5; REFUTING if "
        "its upper bound is at most 0.2 and the upper bound of an empirical-band LR is also at most 0.2 (bands "
        "with edges 0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1; exact Clopper-Pearson bound); otherwise "
        "UNCERTAIN. INSUFFICIENT DATA is returned when fewer than 10 development scores lie at or beyond p in "
        "its tail, or the condition has fewer than 20 development positives. The specification, parameters and "
        "checksums were frozen (tag evidence-engine-v1.0, 2026-09-27).")
    d.p("**Provisional cut-offs for seven further conditions** (pneumonia, infiltration, emphysema, nodule, "
        "mass, fibrosis, pleural thickening) were fitted the same way on NIH development data (design committed "
        "2026-10-09 before any reading; values committed the same day). A state was used only if its "
        "empirical likelihood ratio passed a pre-specified check on the NIH test split, which is also the E4 "
        "test set; the check used labels and CheXagent scores but no report. Five SUPPORTING states (emphysema, "
        "nodule, mass, fibrosis, pleural thickening) and one REFUTING state (nodule) passed (Supplementary S8). "
        "They enter E4 only through the editing rule. Because their selection used the E4 test images, the "
        "editing rule is not fully independent of its test set.")

    d.h("3.6 Experiments", 2)
    d.p("**E1, laterality conditioning (VinDr-CXR, pre-registered).** For 153 findings marked on one side only "
        "(pleural effusion 104, consolidation 24, pneumothorax 15, atelectasis 10), each checker localised the "
        "finding three times: with no side, with the true side and with the opposite side. Primary endpoint: "
        "LR+ = P(box on the side in question | that side is true) / P(box on the side in question | that side "
        "is false) for side-free prompts, supported if its lower 95% bound exceeds 1. Co-primary: the increase "
        "in opposite-side agreement from side-free to side-naming prompts (one-sided exact McNemar, "
        "Holm-adjusted across the two models). CIs are clustered by image.")
    d.p("**E2, presence (pre-registered).** Paired AUC difference, CheXagent minus MAIRA-2, per dataset and "
        "condition (29 comparisons with labels). Criterion: 95% CI above 0. Comparisons with fewer than 10 "
        "positives are reported but not counted.")
    d.p("**E3, grounding (pre-registered).** Each checker localised each of 947 boxed NIH findings from the "
        "side-free condition name. Hit: the best of its boxes has IoU at least t with the radiologist's box. "
        "Primary: recall at IoU 0.5 (two-sided). Secondary: IoU 0.3, pointing game, union-of-boxes recall. "
        "Bootstrap resamples keep all findings of one image together.")
    d.p("**E4, downstream use with generated reports.** MedGemma reports were read for 13 conditions by the "
        "report reader, which had to quote the sentence it relied on; an answer whose quote was not in the "
        "report counted as not stated. Report score: 1 stated, 0.5 possible, 0 not stated.")
    d.bullets([
        "*Score fusion (pre-registered):* each checker's score is converted to its percentile among that "
        "checker's scores, without labels; fused score = 0.5 × report score + 0.5 × percentile. Hypotheses: "
        "fused (CheXagent) exceeds report alone, and fused (CheXagent) exceeds fused (MAIRA-2), in mean AUC over "
        "conditions with at least 10 positives.",
        "*Checker-only baselines (exploratory, added in review):* each checker's score alone on the same images, "
        "and fused minus checker alone.",
        "*Report editing (pre-registered secondary):* a condition the report states (score at least 0.5) is "
        "removed if CheXagent's evidence is against it, and pleural effusion or cardiomegaly is added if the "
        "evidence supports it. The rule was fixed on 2026-10-13 (chexagent_revision_rule.md), before the E4 "
        "protocol, by reference to the decision function chexagent_reliability.decide. For the six engine "
        "conditions, evidence is SUPPORTING or REFUTING as defined above. Where the engine returns "
        "INSUFFICIENT DATA because p_present lies outside the development support, that function overrides it: "
        "p_present at least 0.5 counts as supporting, below 0.5 as against. This fallback is not part of Evidence "
        "Engine v1.0 and was not stated in the E4 protocol itself. For the other seven conditions, the "
        "provisional states are used. The rule's design (removal on against; additions limited to the two "
        "conditions) was justified by reliability figures that included NIH test-split images. Summaries "
        "follow each setting's protocol: in (a), sensitivity and specificity pooled over all 13 conditions; in "
        "(c), their means over the 9 conditions with at least 10 positives; balanced accuracy is their mean.",
    ])
    d.p("Settings: (a) base MedGemma, 1,500 single test images, percentiles from the development split; (b) base "
        "MedGemma on the 499-pair subset, with prior and current images given to all models, percentiles within "
        "the 499; (c) team-trained MedGemma on the same 499 pairs, in its training input format, with "
        "placeholders (\"Not available.\", \"Not provided.\") for the prior report and indication, which NIH "
        "lacks. Settings (b) and (c) use a subset of (a)'s images and share checker readings, so the three are "
        "related evaluations, not independent replications.")
    d.p("**E5, robustness (exploratory).** Alternative MAIRA-2 scores; side named in the localisation phrase "
        "(40 NIH pneumothorax cases); CheXagent given two images versus one; grounding without cardiomegaly.")

    d.h("3.7 Statistics and multiplicity", 2)
    d.p("AUC is the Mann-Whitney statistic. CIs for AUCs, recalls and paired differences use 2,000 bootstrap "
        "resamples with fixed seeds; proportions use Wilson intervals; discordant pairs use exact McNemar "
        "tests. Means over conditions include conditions with at least 10 positives.")
    d.p("**Multiplicity.** E1's co-primary test was Holm-adjusted across the two models, as pre-registered. The "
        "29 presence comparisons (E2) and the six fusion hypotheses (E4) were pre-specified as individual tests "
        "without adjustment, and are reported that way. Two exploratory Holm adjustments were added in review "
        "(Supplementary S7), both from bootstrap p-values, two-sided: p = min(1, 2(1 + k) / (B + 1)), where k is "
        "the number of the B = 2,000 resampled differences on the far side of zero. For E2, the pre-registered "
        "analysis scripts use fixed seeds and a fixed order, so we regenerated their exact resamples; this "
        "reproduced all 29 committed CIs to three decimals. Holm was applied across the 29. For E4, 2,000 new "
        "resamples were drawn and Holm applied across the six hypotheses. With B = 2,000, the smallest attainable "
        "p is 0.001, which limits how small adjusted values can be.")
    d.p("These p-values invert the percentile bootstrap interval. Without the +1 terms, p is the smallest "
        "level at which the two-sided percentile interval excludes zero, so the tests agree with the reported "
        "confidence intervals [Davison-Hinkley]. They rest on the same assumption as those intervals: that the "
        "bootstrap distribution approximates the sampling distribution of the difference. They are not "
        "permutation tests under a null hypothesis. The +1 terms follow the usual conservative convention for "
        "Monte Carlo p-values, so that no p is reported as zero [North 2002]. Adjusted values are exploratory "
        "and are reported only to show that the conclusions do not depend on the lack of adjustment.")

    d.h("3.8 Radiologist evaluation", 2)
    d.p("**What the evaluation measures.** The radiologists rate four separate capabilities, not overall report "
        "quality: (1) whether each finding MedGemma states is present on the current image; (2) whether "
        "CheXagent's report-blinded verdict on that finding (Agree or Not agree) is correct; (3) whether the "
        "change MedGemma describes relative to the prior examination (new, unchanged, worse, better, resolved, or "
        "none stated) is correct; and (4) whether CheXagent's box marks the finding's anatomical location. "
        "Ratings are made per finding, so each capability is assessed on the same findings and can be reported "
        "separately. Conditions found only by CheXagent, and conditions in the reference report that neither "
        "model stated, are rated as well, which measures CheXagent's added findings and both models' misses.")
    d.p("**Cases.** 100 prior and current chest X-ray pairs: 60 from the team's held-out MIMIC-CXR pairs (never "
        "used to train the fine-tuned MedGemma), 20 from CheXpert Plus [CheXpert Plus] and 20 from the 499 NIH "
        "pairs. MIMIC-CXR and CheXpert Plus pairs were chosen from the radiologists' report text only, before any "
        "model output was generated: consecutive frontal studies of one patient, a current report that compares "
        "with the prior, and (for the first MIMIC-CXR set and CheXpert Plus) a current report stating at least "
        "one of cardiomegaly, pleural effusion, pneumothorax, consolidation, pulmonary edema or atelectasis. NIH "
        "pairs were chosen by their labels for the same six conditions. The selection is enriched for these "
        "conditions and is not a random sample.")
    d.p("**Model outputs shown.** For each pair, the fine-tuned MedGemma wrote a report from the prior image, "
        "the prior report and the clinical indication, and the current image (its training format; NIH pairs "
        "used placeholder text for the prior report and indication). The language-model reader extracted the "
        "13 conditions it stated and the change it reported for each (new, unchanged, worse, better, resolved, "
        "or no change stated, with a verbatim quote). CheXagent, without the report, answered \"Is there "
        "{condition}?\" for each condition [current image only; NIH pairs: prior and current images], and its "
        "Evidence Engine verdict was shown as Agree (supported) or Not agree. Conditions CheXagent found that "
        "MedGemma did not state, and conditions in the reference report that neither model stated, were listed "
        "as separate rows. CheXagent boxed each finding MedGemma stated on the current image.")
    d.p("**Readers and grading.** Two radiologists [qualifications and years of experience] graded the 100 "
        "prior and current pairs, divided between them without overlap; each pair was read by one radiologist. "
        "Each page showed the prior, current and boxed images, the reference report, MedGemma's report and a "
        "table with one row per finding. For every finding the radiologist rated four dimensions on a "
        "five-point scale (1, poor; 5, excellent): MedGemma finding accuracy, CheXagent diagnostic evidence "
        "accuracy, MedGemma longitudinal change accuracy and CheXagent grounding accuracy, with free-text "
        "comments. Scores were entered in fillable "
        "PDF forms and read automatically. [Analysis: score distributions per dimension and dataset; "
        "no inter-reader agreement, as each pair had one reader.] MIMIC-CXR material was shared "
        "only under the PhysioNet data use agreement and CheXpert Plus under its research use agreement.")


    # ---------------- results
    d.h("4. Results", 1)
    d.h("4.1 Longitudinal report generation (RQ1)", 2)
    d.p("Longitudinal reporting and the report-only comparison below were exploratory (descriptive) analyses. "
        "On the 499 NIH prior-current pairs, which come from a dataset and hospital not used in fine-tuning, "
        "the fine-tuned MedGemma wrote comparison-style reports: 487 of 499 referred to the prior study, against "
        "3 of 499 for base MedGemma given the same two images and an instruction to note change (Table 6). Its "
        "reports also identified labelled conditions better: mean AUC of the report score 0.593 vs 0.545 "
        "(+0.049, +0.025 to +0.073). The largest per-condition differences were consolidation, atelectasis, "
        "pneumothorax and pleural effusion. Both models rarely mentioned emphysema, nodules, fibrosis or "
        "pleural thickening, so their scores for those conditions were near chance.")
    d.table(["Measure (499 NIH pairs)", "Base MedGemma", "Fine-tuned MedGemma"], [
        ("Reports referring to the prior study", "3", "487"),
        ("Mean AUC of report score (9 conditions)", "0.545", "0.593"),
        ("Pleural effusion AUC (45 positives)", "0.775", "0.818"),
        ("Atelectasis AUC (41 positives)", "0.553", "0.665"),
        ("Consolidation AUC (17 positives)", "0.514", "0.669"),
        ("Pneumothorax AUC (16 positives)", "0.499", "0.586"),
        ("Fusion with CheXagent: gain over CheXagent alone", "-0.004", "-0.003"),
    ], widths=[7.5, 4, 4], size=8.5,
       caption="Base and fine-tuned MedGemma on the same NIH pairs. Per-condition values are point estimates "
               "without CIs; the mean difference is +0.049 (+0.025 to +0.073).")
    d.p("Three cautions apply. The two models differ in quantisation (base: 4-bit through Ollama; fine-tuned: "
        "bf16 through transformers), runtime and input format, so the comparison does not isolate the effect of "
        "fine-tuning. The fine-tuned model ran without the prior report and indication it was trained with. And "
        "the NIH labels describe the current image only, so the correctness of the interval-change statements "
        "was not assessed. With CheXagent's evidence added, both generators reached similar detection (0.835 and "
        "0.837; Table 8), because CheXagent's evidence carries most of the discrimination.")

    d.h("4.2 Naming a side determines the side checkers return (RQ2)", 2)
    d.fig(fig2, "Side agreement on 153 one-sided VinDr-CXR findings, by prompt.", width_cm=16)
    d.p("When the prompt named the opposite side, CheXagent localised that side in 86% of cases (79-91%) and "
        "MAIRA-2 in 99% (96-100%). Opposite-side agreement rose from 14% to 86% (CheXagent) and from 41% to 99% "
        "(MAIRA-2) between side-free and side-naming prompts (109/0 and 89/0 discordant pairs; Holm-adjusted p "
        "< 10^-26 for both). With side-free prompts, CheXagent's boxes covered the true side in 99% of cases and "
        "the opposite side in 14%; the two overlap because boxes crossing the midline count for both sides. "
        "This gives LR+ 6.73 (4.78-10.52; supported). For MAIRA-2 the figures were 54% and 41% (LR+ 1.31, "
        "0.99-1.80), so the evidence of side discrimination was inconclusive. In an exploratory check on 40 NIH "
        "pneumothorax cases, both checkers localised whichever side the phrase named, true or opposite, in 40 of "
        "40 cases. With side-free prompts, MAIRA-2 placed most single boxes on the patient's right.")
    d.fig(FIG / "manuscript_laterality_examples.png",
          "Example cases from the laterality experiment, chosen by a fixed rule rather than by result: the first one-sided case "
          "of each finding in image-ID order (all four happen to be right-sided). Green: the three VinDr readers' "
          "boxes; blue: CheXagent; dashed orange: MAIRA-2. Columns: side-free prompt, prompt naming the true side, "
          "prompt naming the opposite side.", width_cm=12)

    d.h("4.3 Report-blinded presence evidence (RQ3)", 2)
    d.fig(fig3, "Presence AUC of CheXagent (blue) and MAIRA-2 (orange) with 95% CIs, by dataset and condition. "
                "Left numbers: positives / images.")
    d.p("CheXagent's AUC point estimate exceeded MAIRA-2's in all 29 comparisons; the pre-registered criterion "
        "was met in 26 (Supplementary S3). The criterion was not met for NIH consolidation (+0.101, -0.004 to "
        "+0.211), NIH pneumonia (9 positives) or NIH infiltration (both models near chance). After the "
        "exploratory Holm adjustment across the 29, all 26 remained significant (adjusted p 0.029 for each, a "
        "value set by the bootstrap's resolution rather than by the size of the differences). On Indiana, "
        "which neither model's documented training data include, the criterion was met for all 6 conditions "
        "(differences +0.091 to +0.349); CheXagent's results on these images were known to us beforehand "
        "(Section 3.2). On NIH and VinDr, which CheXagent's development data include, differences may be "
        "inflated (Section 5).")
    d.p("**Scoring MAIRA-2 (exploratory).** For one image, MAIRA-2's own report described an effusion that "
        "the fixed sentence pair scored low. On 300 development images, a score from 10 sampled reports "
        "tracked MAIRA-2's own statements most closely (AUC 0.959 against its greedy report), but detected "
        "NIH-labelled conditions less well than the pre-registered score (mean AUC 0.777 vs 0.828; CheXagent "
        "0.863 on the same images). Alternatives helped only for pleural effusion (0.900 vs 0.864). The "
        "pre-registered score was the best of the four we tested at detecting labelled conditions "
        "(Supplementary S5).")

    d.h("4.4 Grounding (RQ3)", 2)
    d.p("Localization is measured separately from presence: a box can be anatomically plausible even when the "
        "corresponding diagnosis is wrong, and a correct diagnosis can be poorly localized.")
    d.table(["Measure", "CheXagent", "MAIRA-2", "Difference [95% CI]"], [
        ("Recall at IoU 0.5 (primary)", "23.3%", "19.0%", "+4.3 [+1.7, +7.0]"),
        ("Recall at IoU 0.3", "52.5%", "30.3%", "+22.2 [+18.8, +25.6]"),
        ("Pointing game", "68.5%", "40.0%", ""),
        ("Mean IoU", "0.31", "0.22", ""),
        ("No box returned", "9 (1%)", "127 (13%)", ""),
    ], widths=[5, 3, 3, 4.5], caption="Grounding on 947 NIH radiologist boxes (pre-registered; image-clustered CIs).")
    d.p("At the strict standard, both checkers matched about one in five radiologist boxes. CheXagent located "
        "more findings at lenient standards. MAIRA-2 more often returned no box (74 of 116 pneumonia cases) "
        "but matched cardiomegaly boxes better (IoU 0.5 recall 100% vs 70%). Excluding cardiomegaly "
        "(exploratory), recall at IoU 0.5 was 15.1% vs 4.8%. Patient-clustered intervals were similar (+1.5 to "
        "+7.3 at IoU 0.5). Example cases in both directions are in Supplementary S6.")

    d.h("4.5 Downstream use with generated reports (RQ4)", 2)
    d.p("The analyses in this section address condition identification (RQ4a): discrimination of present from "
        "absent conditions against reference labels. They do not by themselves show that final reports are more "
        "accurate; report auditing (RQ4b) is assessed in the radiologist evaluation (Section 4.7).")
    d.p("**CheXagent evidence improves identification of conditions over MedGemma's findings.** All six "
        "pre-registered hypotheses were supported (Table 8; Figure 6). Adding CheXagent's evidence raised mean "
        "AUC over MedGemma's findings alone by +0.306, +0.291 and +0.244 in settings a, b and c, including +0.244 "
        "(0.593 to 0.837) for the fine-tuned MedGemma, and by +0.144 to +0.162 more than adding MAIRA-2's. With "
        "the exploratory Holm adjustment across the six, each adjusted p was 0.006 (two-sided; no resampled "
        "difference at or below zero).")
    d.p("**Where the improvement comes from.** The checker-only baselines added in review show that the gain is "
        "CheXagent's own detection: in every setting, CheXagent's score alone was as discriminative as the "
        "combination (difference -0.002, -0.004 and -0.003; 95% CIs -0.003 to -0.002, -0.007 to -0.002 and "
        "-0.008 to +0.004). MedGemma's findings added no further discrimination on top of CheXagent. Fusion with MAIRA-2 modestly improved on MAIRA-2 alone, with intervals above zero "
        "in all three settings (+0.016, +0.007, +0.027; Supplementary S7). For "
        "some averaged conditions the report score was the same for every image (emphysema, fibrosis and "
        "pleural thickening in setting a), so the fused ranking there came entirely from the checker.")
    d.table(["Setting", "Report alone", "CheXagent alone*", "Fused, CheXagent", "MAIRA-2 alone*", "Fused, MAIRA-2"], [
        ("(a) Base, single image (1,500)", "0.557", "0.866", "0.863", "0.695", "0.711"),
        ("(b) Base, prior + current (499)", "0.545", "0.840", "0.835", "0.667", "0.673"),
        ("(c) Trained, prior + current (499)", "0.593", "0.840", "0.837", "0.667", "0.693"),
    ], widths=[4.4, 2.2, 2.4, 2.4, 2.4, 2.4], size=8.5,
       caption="Mean AUC over conditions with at least 10 positives (12 in a; 9 in b and c). *Exploratory "
               "baselines added in review. Pre-registered differences: fused CheXagent minus report alone "
               "+0.306, +0.291, +0.244; fused CheXagent minus fused MAIRA-2 +0.152, +0.162, +0.144 (all CIs "
               "above 0).")
    d.fig(fig4, "Mean AUC of the report score alone, each checker alone (exploratory baselines) and each fused "
                  "score (pre-registered), in the three settings, with 95% CIs.", width_cm=15)
    d.p("**Report editing.** The yes/no rule changed balanced accuracy by +0.015 (+0.008 to +0.021) with base "
        "MedGemma in setting (a): pooled sensitivity rose from 0.113 to 0.145 and specificity fell from 0.989 to "
        "0.986. All 81 changes were additions of pleural effusion or cardiomegaly; 22 of them were correct by "
        "the NIH labels and 59 were not. With the trained MedGemma in setting (c), the change was -0.0005 "
        "(95% CI -0.0010 to -0.0001, recomputed at four decimals), "
        "because the rule rarely triggered (one correct removal and seven incorrect additions). Changing the "
        "summary to the other setting's definition did not change either conclusion (+0.019 in a; -0.0005 in c). "
        "These are the only measurements here of changed report content.")
    d.p("**Sensitivity of the editing rule (exploratory).** Removing the provisional states, the fallback, or "
        "both left the results identical in both settings (Supplementary S7). Both parts did produce decisions: "
        "in setting (a), the fallback decided 22 image-condition pairs and the provisional states 429. None of "
        "these decisions changed a report. Provisional supporting states cannot add a condition, the nodule "
        "refuting state never met a stated nodule, and the fallback's decisions fell on conditions already "
        "stated or not eligible for addition. Every change came from the frozen engine's own states, so the "
        "provisional states and the fallback did not affect the edits. The overall editing policy was itself "
        "informed by reliability figures that included NIH test images (Section 3.6).")
    d.h("4.6 Ablations and failure cases", 2)
    d.bullets([
        "**Prior image for the checker:** giving CheXagent both images produced no clear improvement over the "
        "current image alone (+0.005, -0.002 to +0.012); averaging separate prior and current readings was "
        "slightly worse (-0.014, -0.026 to -0.002).",
        "**Conditions reports omit:** both MedGemma versions rarely mentioned emphysema, nodules, fibrosis or "
        "pleural thickening.",
        "**Box failures:** MAIRA-2 placed boxes on the wrong side or returned none; CheXagent's heart boxes "
        "often covered only the lower cardiac silhouette; NIH gives one box per finding even when bilateral, "
        "penalising checkers that box both sides.",
    ])

    d.h("4.7 Radiologist evaluation", 2)
    d.p("[Results pending: per-dimension score distributions (median and interquartile range, proportion scored "
        "4-5) for MedGemma finding accuracy, CheXagent evidence accuracy, MedGemma change accuracy and CheXagent "
        "grounding accuracy, by dataset; agreement between CheXagent's verdict and the radiologists' scores.]")

    # ---------------- discussion
    d.h("5. Discussion", 1)
    d.p("**What the results establish.** In both checkers, naming a side in the localisation prompt largely "
        "determined the side returned. We measured this for laterality in two models and do not claim it for "
        "every form of claim checking. A validator that supplies the asserted side to its grounding tool can "
        "therefore return a box that restates the assertion. Without the side, CheXagent's localisation "
        "discriminated true from false sides, and its presence scores discriminated conditions across three "
        "datasets better than MAIRA-2's.")
    d.p("**What the downstream results show.** Report-blinded CheXagent evidence, added to MedGemma's findings, "
        "substantially improved identification of conditions against reference labels, for both the base and the "
        "fine-tuned generator, and more than MAIRA-2's evidence did. The improvement is CheXagent's own detection: "
        "CheXagent alone performed as well as the combination. In practice this means the evidence contributes "
        "most where the report is silent or wrong, which is what a flag for the reader needs. What we did not "
        "show is that automatically edited reports are more accurate: the tested editing rule changed little "
        "(+0.015 or none), and interval-change accuracy and clinical benefit were not assessed.")
    d.p("**The editing rule should not edit reports automatically.** Of the 81 changes the rule made to base "
        "MedGemma reports, all were additions of pleural effusion or cardiomegaly, and 59 (73%) disagreed with "
        "the NIH reference labels. In a reading workflow, about three of every four automatic additions would "
        "have inserted a finding the reference labels do not support. These are disagreements with "
        "text-mined labels, not radiologist-adjudicated errors, and some may be label errors. Even so, the "
        "small gain in balanced accuracy (+0.015) hides a low precision for the changes themselves. The safer "
        "use of report-blinded evidence is to flag findings for the reader rather than to change the report.")
    d.p("**The fine-tuned report generator.** Fine-tuning on comparison reports changed what MedGemma "
        "writes: on an unseen dataset it described interval change almost every time and its reports identified "
        "labelled conditions better. Its main purpose, accurate statements of change, is what our NIH data cannot "
        "test. A held-out MIMIC-CXR evaluation against radiologists' comparison reports, and a comparison with "
        "the base model run in bf16 with the same input format, would isolate the effect of fine-tuning.")
    d.p("**Relation to the original EDV.** Our laterality experiment tested grounding models, not the "
        "multimodal model that performs EDV, so it does not show that EDV restates claims. It does show that "
        "MAIRA-2, one of the tools CXRAgent can call, localised a stated opposite side in 99% of cases, and that "
        "its side-free side evidence was inconclusive. Whether EDV's validator similarly follows sides stated "
        "in the tool output is untested. The two approaches also express evidence differently. EDV's High, "
        "Medium and Low labels can be evaluated, for example by the proportion of correct statements in each "
        "level, but they carry no numerical probability or likelihood ratio without an additional mapping. Our "
        "evidence is stated on a quantitative scale with predefined thresholds. CXRAgent's gains were measured "
        "on different tasks and metrics, so the results are not directly comparable. A head-to-head comparison "
        "of statement-given and report-blinded validation inside one agent would be informative.")
    d.p("**Relation to dedicated fact-checking models.** A model trained on perturbed findings, as in "
        "Mahmood et al., is a different design from the general-purpose checkers we studied. Our results do "
        "not show whether it follows a stated location or polarity; testing it with the same true and false "
        "statements would be informative.")
    d.p("**Alternative explanations.**")
    d.bullets([
        "*Training exposure.* Both checkers have documented exposure to NIH images: CheXagent's supervised, "
        "MAIRA-2's image-only, through its encoder. CheXagent's also includes VinDr-CXR (Table 5). Overlap with "
        "our specific images is unknown for CheXagent. High VinDr AUCs are consistent with such overlap but do "
        "not establish it. Indiana, excluded from both, gave the same direction of result. CheXagent's "
        "instruction data (CheXinstruct) also include MIMIC-CXR and CheXpert [CheXagent], so the radiologist "
        "evaluation pairs from those datasets may contain images CheXagent saw in training; agreement with "
        "CheXagent there may be optimistic. The fine-tuned MedGemma was trained on MIMIC-CXR, but not on the "
        "held-out pairs.",
        "*Label quality.* NIH labels are text-mined; VinDr and Indiana labels are better but imperfect.",
        "*Coarse report scores and the reader.* Three-valued report scores create many ties, which limits how "
        "finely they rank images, though they can still reach AUC 1.0. Reader errors would also lower the "
        "report-only AUCs and could change how much the report adds in fusion.",
        "*Scoring MAIRA-2.* The pre-registered score was the best of four tested, but understated MAIRA-2 for "
        "effusion.",
        "*Box conventions.* Recall depends on how radiologists drew boxes, which favoured MAIRA-2's whole-heart "
        "boxes.",
    ])
    d.p("**Limitations.**")
    d.bullets([
        "The report reader has not yet been validated against manual review. Requiring a quote makes the reader "
        "cite text that exists, but a real quote can still be misread (a negated, hedged or prior-only finding "
        "read as present), and stated findings can be missed. A blinded manual review of 90 reports (1,170 "
        "decisions), measuring presence, negation, uncertainty and temporal errors, has been specified before "
        "annotation (Supplementary S9) and is pending.",
        "Base and trained MedGemma differ in quantisation, runtime and input format.",
        "The trained MedGemma ran without the prior report and indication it was trained with, and on NIH "
        "rather than MIMIC-CXR images.",
        "Grounding was evaluated with the condition known to be present.",
        "Several conditions had few positives.",
        "Settings (b) and (c) reuse (a)'s images and checker readings.",
        "No reader study was performed.",
    ])
    d.p("**Future work.** Planned next, with protocols written before any reading: whether a stated "
        "assertion changes CheXagent's presence evidence, and how well blinded and assertion-given evidence "
        "separate true from false statements; image-ablation controls; more checkers and datasets for "
        "laterality; and an EDV-style validator given true or false statements. With a radiologist co-author: "
        "manual validation of the report reader and a reader study of drafts reviewed with and without "
        "evidence flags. Comparing the checkers' own free-text reports, read by the same reader (pre-registered "
        "and in progress), would test MAIRA-2 in its native output. For the fine-tuned MedGemma, two experiments "
        "are planned: base MedGemma in bf16 with the identical input format on the same NIH pairs, which removes "
        "the quantisation and format confounds; and interval-change accuracy against radiologists' comparison "
        "reports on a held-out MIMIC-CXR set, which needs the images and confirmation that the set was not used "
        "for checkpoint selection. Fusion or editing rules that "
        "add value beyond the checker alone remain to be designed.")

    d.h("6. Conclusion", 1)
    d.p("For two chest X-ray models, naming a side in the localisation prompt largely determined the side they "
        "returned. Laterality evidence for validating generated findings should therefore be collected without "
        "the asserted side. Report-blinded evidence from CheXagent was informative for presence and side and "
        "outperformed MAIRA-2's. Added to MedGemma's findings, base or fine-tuned, it substantially improved "
        "identification of conditions; the gain came from CheXagent's own detection, and most automatic additions "
        "by a simple editing rule disagreed with reference labels. Report-blinded evidence is therefore best used "
        "to flag findings for the reader rather than to edit reports. In an exploratory application, a MedGemma fine-tuned on "
        "comparison reports wrote longitudinal reports on an unseen dataset; whether its statements of change "
        "are correct remains to be tested.")

    # ---------------- supplementary
    d.page_break()
    d.h("Supplementary material", 1)
    d.h("S1. Exact prompts", 2)
    report_prompt = (ROOT / "medrax" / "docs" / "medgemma_report_prompt.txt").read_text(encoding="utf-8").strip()
    from comparison_edv_pdf import GT_PROMPT  # the report reader's prompt
    from edv_improvement import CONDITIONS, EXTRA_RULES
    for label, text in (
            ("CheXagent presence", "Is there {condition}?"),
            ("CheXagent localisation", "Please locate the following phrase: {condition}"),
            ("CheXagent two-image presence (setting b)", "The first image is the prior chest X-ray and the second is "
             "the current chest X-ray. Is there {condition} in the current chest X-ray?"),
            ("MAIRA-2 scoring prefix (processor-generated)", "You are an expert radiology assistant tasked with "
             "interpreting a chest X-ray study. USER: Given the current frontal image only <image> Provide a "
             "description of the findings in the radiology study in comparison to the prior frontal image. "
             "INDICATION: N/A TECHNIQUE: N/A COMPARISON: N/A ASSISTANT:"),
            ("MAIRA-2 scored sentences", "\"There is {a} {condition}.\" / \"There is no {condition}.\""),
            ("MAIRA-2 grounding", "phrase-grounding input with the capitalised condition name, e.g. \"Pleural effusion\""),
            ("Base MedGemma report prompt (settings a and b)", report_prompt),
            ("Prefix added before the report prompt in setting b", "The first image is the PRIOR study and the second "
             "is the CURRENT study; report on the CURRENT study and note change from the prior."),
            ("Trained MedGemma (setting c)", "Prior chest X-ray: <image> Prior report: Not available. Current chest "
             "X-ray: <image> Clinical indication for current study: Not provided. Compare the current chest X-ray to "
             "the prior chest X-ray. Describe any interval change in findings and provide an impression."),
            ("Report reader (Qwen), exactly as sent; {generated report} is replaced by the MedGemma report",
             GT_PROMPT.format(truth="{generated report}", findings="\n".join(f"- {c}" for c in CONDITIONS),
                              extra="".join(f"\n- {r}" for r in EXTRA_RULES))),
    ):
        d.p(f"**{label}:**", after=1)
        d.quote(text)
    d.p("The prompt was written for comparing a report with a ground-truth report; in E4 the generated report "
        "fills the report slot. The rule \"Subcutaneous emphysema (air in the soft tissues) is not "
        "emphysema.\" was added by amendment A1 of the E4 protocol, before any report was read, and was used in "
        "all three settings. A \"present\" answer whose quoted evidence is not found in the report is counted "
        "as not stated; \"(possible)\" answers score 0.5.")

    d.h("S2. Configurations", 2)
    d.bullets([
        "Hardware: one NVIDIA RTX 3090 (24 GB).",
        "Decoding: greedy in all pre-registered runs. MAIRA-2 sampled reports (exploratory): temperature 1, "
        "10 per image.",
        "Images: NIH 1024 × 1024 PNG; VinDr converted from DICOM; base MedGemma inputs reduced to at most "
        "1024 px; other models use their own processors.",
        "Bootstrap: 2,000 resamples; seeds fixed in each protocol.",
    ])

    d.h("S3. Presence AUC, all comparisons", 2)
    d.table(["Dataset", "Condition", "Positives / n", "CheXagent", "MAIRA-2", "Difference [95% CI]", "Criterion met?"],
            AUC_ROWS, widths=[1.7, 2.9, 2.4, 1.8, 1.8, 3.6, 2.8], size=8,
            caption="Presence AUC (pre-registered). *VinDr labels nodule and mass together.")

    d.h("S4. Grounding by condition", 2)
    d.table(["Condition", "Cases", "IoU 0.5 C / M", "IoU 0.3 C / M", "Pointing game C / M"], [
        ("Atelectasis", "173", "12% / 2%", "42% / 16%", "61% / 31%"),
        ("Pleural effusion", "150", "6% / 3%", "38% / 28%", "67% / 41%"),
        ("Cardiomegaly", "141", "70% / 100%", "99% / 100%", "99% / 100%"),
        ("Infiltration", "117", "21% / 11%", "58% / 29%", "74% / 44%"),
        ("Pneumonia", "116", "17% / 4%", "53% / 12%", "78% / 16%"),
        ("Pneumothorax", "93", "33% / 13%", "55% / 22%", "68% / 32%"),
        ("Mass", "81", "21% / 1%", "47% / 7%", "63% / 26%"),
        ("Nodule", "76", "1% / 0%", "11% / 3%", "17% / 3%"),
    ], widths=[3.5, 1.6, 3.6, 3.6, 3.6], caption="Grounding recall by condition (C = CheXagent, M = MAIRA-2).")

    d.h("S5. MAIRA-2 scoring robustness (exploratory)", 2)
    d.table(["MAIRA-2 score", "Tracks its own report (AUC)", "Detects NIH labels (mean AUC)"], [
        ("Pre-registered sentence pair", "0.883", "0.828"),
        ("Own phrasings", "0.838", "0.685"),
        ("Own phrasings, filtered", "0.878", "0.761"),
        ("Share of 10 sampled reports", "0.959", "0.777"),
        ("Single report, yes/no", "-", "0.654"),
        ("CheXagent p_present (reference)", "", "0.863"),
    ], widths=[6.5, 4.5, 5], caption="300 NIH development images; mean over 5 conditions with at least 10 positives.")

    d.h("S6. Grounding examples", 2)
    d.fig(FIG / "grounding_examples.png", "Overview of grounding outcomes (green: radiologist; blue: CheXagent; "
                                          "dashed orange: MAIRA-2).")
    d.fig(FIG / "grounding_maira2_better_cardiomegaly_iou05.png", "Cases where MAIRA-2's box is better: cardiomegaly.")
    d.fig(FIG / "grounding_chexagent_better_pneumothorax.png", "Cases where CheXagent's box is better: pneumothorax.")

    d.h("S7. Checker-only baselines and patient-clustered intervals (exploratory)", 2)
    d.table(["Setting", "Fused CheXagent minus CheXagent alone", "Fused MAIRA-2 minus MAIRA-2 alone",
             "CheXagent alone minus MAIRA-2 alone"], [
        ("(a)", "-0.002 [-0.003, -0.002]", "+0.016 [+0.005, +0.028]", "+0.170 [+0.140, +0.202]"),
        ("(b)", "-0.004 [-0.007, -0.002]", "+0.007 [+0.001, +0.012]", "+0.173 [+0.132, +0.215]"),
        ("(c)", "-0.003 [-0.008, +0.004]", "+0.027 [+0.010, +0.047]", "+0.173 [+0.133, +0.212]"),
    ], widths=[1.8, 4.9, 4.9, 4.9], size=8.5,
       caption="Paired differences in mean AUC, image-level bootstrap.")
    d.p("Each patient contributes exactly one image to these analyses, so patient-level and image-level "
        "resampling are the same sampling design. The patient-clustered intervals we computed differ by at most "
        "0.005 only because they use different random draws (Monte Carlo variation). Grounding is different: "
        "one image can carry several boxes, so its intervals keep an image's findings together, and the "
        "patient-clustered analysis there groups 947 findings into 699 patients (IoU 0.5: +1.5 to +7.3).")
    d.table(["Rule variant", "(a) change in balanced accuracy", "(c) change in balanced accuracy"], [
        ("As reported", "+0.0148 [+0.008, +0.022]", "-0.0005 [-0.0010, -0.0001]"),
        ("Without provisional states", "+0.0148 [+0.008, +0.022]", "-0.0005 [-0.0010, -0.0001]"),
        ("Without the fallback", "+0.0148 [+0.008, +0.022]", "-0.0005 [-0.0010, -0.0001]"),
        ("Frozen engine only (neither)", "+0.0148 [+0.008, +0.022]", "-0.0005 [-0.0010, -0.0001]"),
    ], widths=[5.5, 5.5, 5.5], size=8.5,
       caption="Editing rule sensitivity (exploratory; each setting's protocol summary; new bootstrap draws, so "
               "CIs differ slightly from the pre-registered ones). Source: results/review_editing_rule.md.")
    d.p("**Multiplicity (exploratory).** E2: the regenerated pre-registered resamples reproduced all 29 "
        "committed CIs. Among the 26 comparisons meeting the criterion, two-sided bootstrap p-values were 0.001 "
        "for 23 and 0.002 for 3 (NIH pulmonary edema, NIH fibrosis, VinDr atelectasis); Holm across the 29 left "
        "all 26 below 0.05 (each 0.029). E4: no resampled difference was at or below zero for any of the six "
        "hypotheses, so each two-sided p was 0.001 and each Holm-adjusted p 0.006. Full tables: "
        "experiments/results/review_multiplicity_exact.md (E2) and review_multiplicity.md (E4; its one-sided "
        "values are half of these). The earlier CI-based approximation for E2 also gave 26 of 29.")

    d.h("S8. Evidence Engine v1.0 cut-offs (p_present)", 2)
    d.table(["Condition", "SUPPORTING from", "REFUTING up to"], [
        ("Cardiomegaly", "0.801", "0.014"), ("Pleural effusion", "0.743", "0.0195"),
        ("Pneumothorax", "0.4285", "0.005"), ("Consolidation", "0.8025", "none (not empirically confirmed)"),
        ("Pulmonary edema", "0.4905", "0.0005"), ("Atelectasis", "0.791", "0.0195"),
    ], widths=[5, 4, 6], caption="Frozen cut-offs of the NIH engine (rule 5 / 0.2).")
    d.table(["Condition", "SUPPORTING from (used?)", "REFUTING up to (used?)", "Development scores outside support"], [
        ("Pneumonia", "0.839 (no)", "0.0002 (no)", "INSUFFICIENT DATA -> no evidence"),
        ("Infiltration", "1.000 (no)", "none", "as above"),
        ("Emphysema", "0.784 (yes)", "0.0718 (no)", "as above"),
        ("Nodule", "0.868 (yes)", "0.0595 (yes)", "as above"),
        ("Mass", "0.871 (yes)", "0.0752 (no)", "as above"),
        ("Fibrosis", "0.822 (yes)", "0.0232 (no)", "as above"),
        ("Pleural thickening", "0.822 (yes)", "0.0470 (no)", "as above"),
    ], widths=[3.5, 4, 4, 5], size=8.5,
       caption="Provisional cut-offs (p_present), fitted on NIH development data; a state was used only if it "
               "passed the pre-specified check on the NIH test split (results/provisional_findings.md; design "
               "a4c832d, values 2eefcb8, both 2026-10-09). Only the nodule REFUTING state can remove a stated "
               "condition; provisional SUPPORTING states cannot add one, since the rule adds only pleural effusion "
               "and cardiomegaly.")
    d.fig(FIG / "manuscript_calibration.png",
          "Calibration of the frozen Evidence Engine v1.0. Blue: likelihood ratio (LR) fitted on NIH development "
          "data, with its 95% CI. Orange: empirical LR in each score band on the NIH test split, with 95% CI. "
          "Dashed lines: the SUPPORTING (LR 5) and REFUTING (LR 0.2) thresholds. The printed cut-offs are the "
          "engine's final states, including the empirical-band check for REFUTING.", width_cm=16.5)

    d.h("S9. Pre-registration log", 2)
    d.table(["Analysis", "Protocol committed", "Results committed", "Development evidence already known"], [
        ("Evidence Engine v1.0", "a277a8a, 2026-09-27 (frozen)", "-", "CheXagent on NIH and on the 544 Indiana "
         "images (Section 3.2)"),
        ("Provisional cut-offs (seven conditions)", "a4c832d, 10-09", "2eefcb8, 10-09", "NIH development data; "
         "state use checked on the NIH test split"),
        ("Editing rule", "8d615fd, 10-13 (via chexagent_reliability.decide, 9c81e21, 10-13)", "-", "Reliability of "
         "CheXagent decisions on Indiana, NIH (development and test) and VinDr"),
        ("E1 laterality (VinDr)", "daeb6fa, 10-04; data-source amendment 73a9d4f, 10-04", "ede4255, 10-05",
         "Exploratory side agreement on NIH and Indiana development data"),
        ("E2 presence", "79578de, 10-13; A1 9459f6d, 10-17; A2 9a14988, 10-18", "2270725, 10-17; 992ce10, 10-18",
         "MAIRA-2 vs CheXagent on NIH development data (six conditions)"),
        ("E3 grounding", "6106a1d, 10-24", "cd3ba76, 10-24", "Exploratory pointing game on 416 NIH boxes"),
        ("E4 setting a", "3189f6f, 10-18; A1 ae21a7f, 10-18 (reports generated, none read)", "d7e8b36, 10-19",
         "E2 presence results"),
        ("E4 setting b", "ed54d95, 10-19", "2ca38e5, 10-20", "Setting a results"),
        ("E4 setting c", "9362ffb, 10-24", "dbe2f1e, 10-25", "Settings a and b; input-format check on 3 "
         "development pairs"),
        ("Reader validation (manual review)", "pending commit, 10-25", "pending (annotation not started)",
         "All E4 results"),
        ("Own-report comparison (future work)", "bd777b0, 10-23; A1-A2 dd6bf1f, 10-25", "pending", "E2 results; "
         "exploratory MAIRA-2 effusion check"),
    ], widths=[3, 4.6, 3.4, 5.5], size=8, caption="Protocols, amendments and results (all dates 2026).")
    d.fig(FIG / "manuscript_timeline.png",
          "Pre-registration timeline from git commit times. Each protocol and amendment was committed before its "
          "results; the right-hand column gives the time from the last protocol or amendment to each results "
          "commit.", width_cm=16.5)
    d.p("The repository, including protocol files and their commit history, will be made public at [URL to be "
        "added] so these dates can be verified.")

    d.h("S10. Data availability and references", 2)
    d.p("NIH ChestX-ray14, Open-i and VinDr-CXR are public. MIMIC-CXR is credentialed (PhysioNet data use "
        "agreement). No MIMIC-CXR text or images are included; the team-trained model, derived from MIMIC-CXR, "
        "is shared only with credentialed users.")
    d.p("*References: verify every entry and complete details before submission.*")
    for ref in (
            "[CXRAgent] Lou J, Yang Y, Yu Z, Fu Z, Han W, Huang Q, Yu J. CXRAgent: Director-orchestrated multi-stage "
            "reasoning for chest X-ray interpretation. arXiv:2510.21324v2 (16 July 2026; v1 24 October 2025). Code: "
            "https://github.com/laojiahuo2003/CXRAgent, commit 0514b4897fd0 (14 July 2026). The EDV description "
            "here follows v2 and that commit.",
            "[MedRAX] Fallahpour A, et al. MedRAX: Medical reasoning agent for chest X-ray. ICML 2025; arXiv:2502.02673. "
            "https://github.com/bowang-lab/MedRAX",
            "[CheXagent] Chen Z, Varma M, Xu J, et al. A vision-language foundation model to enhance efficiency of chest "
            "X-ray interpretation. arXiv:2401.12208v2, 2024.",
            "[MAIRA-2] Bannur S, et al. MAIRA-2: Grounded radiology report generation. arXiv:2406.04449, 2024. Model "
            "card: https://huggingface.co/microsoft/maira-2; image encoder: "
            "https://huggingface.co/microsoft/rad-dino-maira-2",
            "[MedGemma 1.5] Sellergren A, et al. MedGemma 1.5 technical report. arXiv:2604.05081, 2026 (v2, 1 May "
            "2026). Model card: https://huggingface.co/google/medgemma-1.5-4b-it",
            "[MedGemma] Sellergren A, et al. MedGemma technical report. arXiv:2507.05201, 2025.",
            "[NIH] Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale chest X-ray database "
            "and benchmarks on weakly-supervised classification and localization of common thorax diseases. CVPR 2017.",
            "[VinDr] Nguyen HQ, et al. VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations. "
            "Scientific Data 9, 429 (2022).",
            "[Open-i] Demner-Fushman D, et al. Preparing a collection of radiology examinations for distribution and "
            "retrieval. JAMIA 23(2), 2016.",
            "[MIMIC-CXR] Johnson AEW, et al. MIMIC-CXR, a de-identified publicly available database of chest radiographs "
            "with free-text reports. Scientific Data 6, 317 (2019).",
            "[CheXpert Plus] Chambon P, Delbrouck JB, Sounack T, et al. CheXpert Plus: Augmenting a large chest X-ray "
            "dataset with text radiology reports, patient demographics and additional image formats. "
            "arXiv:2405.19538 (2024).",
            "[MS-CXR] Boecking B, et al. Making the most of text semantics to improve biomedical vision-language "
            "processing. ECCV 2022.",
            "[pointing game] Zhang J, et al. Top-down neural attention by excitation backprop. ECCV 2016; IJCV 2018.",
            "[CheXbert] Smit A, et al. CheXbert: Combining automatic labelers and expert annotations for accurate "
            "radiology report labeling using BERT. EMNLP 2020.",
            "[RadGraph] Jain S, et al. RadGraph: Extracting clinical entities and relations from radiology reports. "
            "NeurIPS Datasets and Benchmarks 2021.",
            "[GREEN] Ostmeier S, et al. GREEN: Generative radiology report evaluation and error notation. Findings of "
            "EMNLP 2024.",
            "[sycophancy] Sharma M, et al. Towards understanding sycophancy in language models. ICLR 2024.",
            "[Zou 2026] Zou K, Zhou N, Yang T, et al. Visually irrelevant hallucinations in medical "
            "vision-language models. arXiv:2610.38389 (2026).",
            "[Mahmood 2025] Mahmood R, Machado-Reyes D, Wu J, Kaviani P, Wong KCL, D'Souza N, Kalra M, Wang G, Yan P, "
            "Syeda-Mahmood T. Phrase-grounded fact-checking for automatically generated chest X-ray reports. MICCAI "
            "2025. https://papers.miccai.org/miccai-2025/paper/3526_paper.pdf; arXiv:2509.21356",
            "[Mahmood 2024] Mahmood R, et al. Anatomically-grounded fact checking of automated chest X-ray reports. "
            "arXiv:2412.02177, 2024.",
            "[CoEV] Zhou N, et al. Hallucination detection and correction in medical VLMs via counter-evidence "
            "verification. MICCAI 2026; arXiv:2606.18609.",
            "[UniVRSE] Liao Z, et al. Unified vision-conditioned response semantic entropy for hallucination "
            "detection in medical VLMs. arXiv:2503.20504.",
            "[ReXTrust] Hardy R, et al. ReXTrust: a model for fine-grained hallucination detection in AI-generated "
            "radiology reports. arXiv:2412.15264, 2024.",
            "[FactCheXcker] Heiman A, Zhang X, Chen E, et al. FactCheXcker: mitigating measurement hallucinations in "
            "chest X-ray report generation models. CVPR 2025; arXiv:2411.18672.",
            "[CorBenchX] Zou J, et al. CorBenchX: large-scale chest X-ray error dataset and VLM benchmark for "
            "report error correction. arXiv:2505.12057.",
            "[RadAgents] Zhang K, et al. RadAgents: multimodal agentic reasoning for chest X-ray interpretation "
            "with radiologist-like workflows. MIDL 2026; arXiv:2509.20490.",
            "[Verification Mirage] Jin R, et al. Verification mirage: mapping the reliability boundary of "
            "self-verification in medical VQA. arXiv:2605.10850, 2026.",
            "[Deference] Roy R, et al. Source-dependent deference in medical imaging agents under falsified "
            "findings: a pilot audit. arXiv:2608.29800, 2026.",
            "[EchoBench] Yuan B, Zhou Y, Wang Y, et al. EchoBench: benchmarking sycophancy in medical large "
            "vision-language models. arXiv:2509.20146, 2025.",
            "[HEAL-MedVQA] Nguyen D, Ho MK, Ta H, et al. Localizing before answering: a hallucination evaluation "
            "benchmark for grounded medical multimodal LLMs. IJCAI 2025; arXiv:2505.00744.",
            "[C-REC] Yu Z, Li R. Revisiting counterfactual problems in referring expression comprehension. CVPR "
            "2024:13438-13448. doi:10.1109/CVPR52733.2024.01276",
            "[AnatomiX] Hashmi AUR, et al. AnatomiX: an anatomy-aware grounded multimodal LLM for chest X-ray "
            "interpretation. arXiv:2601.03191, 2026.",
            "[Other Left] Wolf D, Hillenhagen H, Taskin B, et al. Your other left! Vision-language models fail to "
            "identify relative positions in medical images. MICCAI 2025; arXiv:2508.00549.",
            "[CoVe] Dhuliawala S, Komeili M, Xu J, et al. Chain-of-verification reduces hallucination in large "
            "language models. arXiv:2309.11495, 2023.",
            "[CONRep] Elyassirad D, et al. CONRep: uncertainty-aware vision-language report drafting using "
            "conformal prediction. arXiv:2602.03910; J Imaging Inform Med 2026.",
            "[Calibrated triage] Khanmohammadi R, et al. Calibrated triage, not autonomy: confidence estimation for "
            "medical vision-language models. arXiv:2606.15910, 2026.",
            "[RadVLM] Deperrois N, et al. RadVLM: a multitask conversational vision-language model for radiology. "
            "arXiv:2502.03333.",
            "[CheXOne] Zhang Y, et al. A reasoning-enabled vision-language foundation model for chest X-ray "
            "interpretation. arXiv:2604.00493, 2026.",
            "[EditGRPO] Zhang, et al. EditGRPO: reinforcement learning with post-rollout edits for clinically "
            "accurate chest X-ray report generation. AACL 2025; arXiv:2509.22812.",
            "[STRIVE] Maeng J, et al. STRIVE: multi-agent structured temporal reasoning with integrated "
            "verification for longitudinal radiology report generation. arXiv:2608.24237, 2026.",
            "[ChexReason] Berger, et al. Benchmark success, clinical failure: when reinforcement learning optimizes "
            "for benchmarks, not patients. arXiv:2512.23090, 2025.",
            "[Banerjee DPO] Banerjee O, et al. Direct preference optimization for suppressing hallucinated prior "
            "exams in radiology report generation. ML4H 2024; arXiv:2406.06496.",
            "[Longitudinal-MIMIC] Zhu Q, Mathai TS, Mukherjee P, et al. Utilizing longitudinal chest X-rays and "
            "reports to pre-fill radiology reports. arXiv:2306.08749, 2023.",
            "[Kathait 2024] Kathait AS, Garza-Frias E, Sikka T, Schultz TJ, Bizzo B, Kalra MK, Dreyer KJ. Assessing "
            "laterality errors in radiology: comparing generative artificial intelligence and natural language "
            "processing. J Am Coll Radiol 2024;21(10):1575-1582. doi:10.1016/j.jacr.2024.06.014",
            "[Davison-Hinkley] Davison AC, Hinkley DV. Bootstrap methods and their application. Cambridge "
            "University Press, 1997.",
            "[North 2002] North BV, Curtis D, Sham PC. A note on the calculation of empirical P values from Monte "
            "Carlo procedures. Am J Hum Genet 2002;71(2):439-441."):
        d.p(ref, after=2)


def main():
    figs = ROOT / "experiments" / "results" / "figures"
    fig1, fig2 = figs / "manuscript_fig1_study_overview.png", figs / "manuscript_fig2_laterality.png"
    overview_figure(fig1)
    conditioning_figure(fig2)
    fig3, fig4 = presence_chart(figs), fusion_chart(figs)
    import manuscript_extra_figures as X
    X.flow_figure(figs / "manuscript_flow.png")
    X.laterality_examples(figs / "manuscript_laterality_examples.png")
    X.calibration_figure(figs / "manuscript_calibration.png")
    X.timeline_figure(figs / "manuscript_timeline.png")
    d = Doc()
    build(d, fig1, fig2, fig3, fig4)
    tidy_tables(d.d)
    out = ROOT / "logs" / f"manuscript_draft_{time.strftime('%Y%m%d-%H%M%S')}.docx"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    d.save(out)
    print(f"{d.n_fig} figures, {d.n_tab} tables -> {out}")


if __name__ == "__main__":
    main()
