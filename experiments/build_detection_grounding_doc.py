"""Word report: how CheXagent and MAIRA-2 were compared on detection (AUC) and grounding (box location).

Written in plain language for the project team. Numbers are taken from the committed results files
(see the appendix of the document); figures from experiments/results/figures and logs/charts_*.

    python experiments/build_detection_grounding_doc.py   -> logs/CheXagent_vs_MAIRA2_detection_and_grounding_<time>.docx
"""

import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as pa  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.section import WD_ORIENT  # noqa: E402,F401
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "experiments" / "results" / "figures"
CHART5 = ROOT / "logs" / "charts_20261019-174040" / "5_all_conditions_auc.png"
CHART6 = ROOT / "logs" / "charts_20261019-173650" / "6_difference_forest.png"
CHART8 = ROOT / "logs" / "charts_20261025-064832" / "8_hypotheses_three_experiments.png"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


# ------------------------------------------------------------------ figures made here

def iou_figure(path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    cases = [("IoU = 1.0: the boxes are identical", [(2, 2, 6, 6)], (2, 2, 6, 6), None),
             ("IoU = 0.33: same size, shifted by half", [(4, 2, 8, 6)], (2, 2, 6, 6), None),
             ("Pointing game hit, but IoU = 0.11:\nthe centre is inside, the box is far too big", [(0.5, 0.5, 9.5, 9.5)],
              (3, 3, 7, 7), None)]
    for ax, (title, model, truth, _) in zip(axs, cases):
        ax.set_xlim(0, 10)
        ax.set_ylim(10, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(pa.Rectangle((truth[0], truth[1]), truth[2] - truth[0], truth[3] - truth[1], fill=True,
                                  fc=GREEN, alpha=0.18, ec=GREEN, lw=3))
        for m in model:
            ax.add_patch(pa.Rectangle((m[0], m[1]), m[2] - m[0], m[3] - m[1], fill=False, ec=BLUE, lw=2.5, ls="--"))
            ax.plot([(m[0] + m[2]) / 2], [(m[1] + m[3]) / 2], "o", color=BLUE)
        ax.set_title(title, fontsize=11, loc="left")
    handles = [pa.Patch(fc=GREEN, alpha=0.4, ec=GREEN, lw=2, label="Radiologist's box"),
               pa.Patch(fc="none", ec=BLUE, lw=2, ls="--", label="Model's box (dot = its centre)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.13, wspace=0.12)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ document helpers

class Doc:
    def __init__(self):
        self.d = Document()
        st = self.d.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(11)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        for s in self.d.sections:
            s.left_margin = s.right_margin = Cm(2.2)
            s.top_margin = s.bottom_margin = Cm(2.0)
            self._page_number(s)
        self.n_fig = 0
        self.n_tab = 0

    @staticmethod
    def _page_number(section):
        p = section.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = p.add_run("Page ")
        run.font.size = Pt(9)
        for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
            r = p.add_run()
            r.font.size = Pt(9)
            if kind:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), kind)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            r._r.append(el)

    def h(self, text, level=1):
        self.d.add_heading(text, level=level)

    def _runs(self, p, text, size=None, italic=False):
        # **bold**, *italic* and `code` markup
        for part in re.split(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|`[^`]+`)", text, flags=re.S):
            if not part:
                continue
            bold = part.startswith("**") and part.endswith("**") and len(part) > 4
            ital = not bold and part.startswith("*") and part.endswith("*") and len(part) > 2
            code = part.startswith("`") and part.endswith("`") and len(part) > 2
            body = part[2:-2] if bold else part[1:-1] if ital or code else part
            r = p.add_run(body)
            r.bold = bold
            r.italic = italic or ital
            if code:
                r.font.name = "Consolas"
            if size:
                r.font.size = Pt(size)

    def p(self, text, italic=False, size=None, after=6):
        p = self.d.add_paragraph()
        self._runs(p, text, size, italic)
        p.paragraph_format.space_after = Pt(after)
        if text.rstrip("* ").endswith(":"):
            p.paragraph_format.keep_with_next = True  # a lead-in stays with the list or quote after it
        return p

    def bullets(self, items, style="List Bullet"):
        for it in items:
            p = self.d.add_paragraph(style=style)
            self._runs(p, it)
            p.paragraph_format.space_after = Pt(2)

    def numbered(self, items):
        self.bullets(items, "List Number")

    def quote(self, text):
        p = self.d.add_paragraph()
        p.paragraph_format.left_indent = Cm(1)
        p.paragraph_format.right_indent = Cm(1)
        r = p.add_run(text)
        r.font.name = "Consolas"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.paragraph_format.space_after = Pt(6)

    def table(self, header, rows, widths=None, caption=None, size=9.5):
        if caption:
            self.n_tab += 1
            cp = self.d.add_paragraph()
            self._runs(cp, f"**Table {self.n_tab}.** {caption}", size=10)
            cp.paragraph_format.space_after = Pt(2)
            cp.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htext in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            self._runs(c.paragraphs[0], f"**{htext}**" if htext else "", size=size)
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "E7ECF3")
            c._tc.get_or_add_tcPr().append(shd)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                self._runs(cells[i].paragraphs[0], str(v), size=size)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def fig(self, path, caption, width_cm=16.5):
        self.n_fig += 1
        self.d.add_picture(str(path), width=Cm(width_cm))
        self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.d.paragraphs[-1].paragraph_format.keep_with_next = True
        cp = self.d.add_paragraph()
        self._runs(cp, f"**Figure {self.n_fig}.** {caption}", size=9.5, italic=False)
        cp.paragraph_format.space_after = Pt(10)

    def page_break(self):
        self.d.add_page_break()

    def save(self, path):
        self.d.save(str(path))


# ------------------------------------------------------------------ content

AUC_ROWS = [  # from experiments/results/maira_vs_chexagent_all_conditions.md
    ("NIH", "Cardiomegaly", "35 / 1,500", "0.936", "0.707", "+0.229 [+0.146, +0.320]", "Yes"),
    ("NIH", "Pleural effusion", "77 / 1,500", "0.885", "0.723", "+0.162 [+0.098, +0.226]", "Yes"),
    ("NIH", "Pneumothorax", "18 / 1,500", "0.894", "0.532", "+0.362 [+0.163, +0.540]", "Yes"),
    ("NIH", "Consolidation", "27 / 1,500", "0.866", "0.764", "+0.101 [-0.004, +0.211]", "No (interval includes 0)"),
    ("NIH", "Pulmonary edema", "10 / 1,500", "0.917", "0.711", "+0.206 [+0.092, +0.345]", "Yes"),
    ("NIH", "Atelectasis", "93 / 1,500", "0.894", "0.780", "+0.115 [+0.079, +0.149]", "Yes"),
    ("NIH", "Pneumonia", "9 / 1,500", "0.734", "0.622", "+0.113 [-0.019, +0.278]", "No (fewer than 10 positives)"),
    ("NIH", "Infiltration", "185 / 1,500", "0.600", "0.586", "+0.013 [-0.016, +0.044]", "No (interval includes 0)"),
    ("NIH", "Emphysema", "14 / 1,500", "0.921", "0.580", "+0.341 [+0.170, +0.519]", "Yes"),
    ("NIH", "Nodule", "85 / 1,500", "0.802", "0.687", "+0.114 [+0.057, +0.174]", "Yes"),
    ("NIH", "Mass", "67 / 1,500", "0.935", "0.852", "+0.082 [+0.033, +0.141]", "Yes"),
    ("NIH", "Fibrosis", "24 / 1,500", "0.823", "0.723", "+0.101 [+0.041, +0.166]", "Yes"),
    ("NIH", "Pleural thickening", "30 / 1,500", "0.916", "0.699", "+0.217 [+0.145, +0.294]", "Yes"),
    ("Indiana", "Cardiomegaly", "164 / 544", "0.927", "0.729", "+0.198 [+0.154, +0.245]", "Yes"),
    ("Indiana", "Pleural effusion", "120 / 544", "0.953", "0.850", "+0.103 [+0.062, +0.153]", "Yes"),
    ("Indiana", "Pneumothorax", "28 / 544", "0.961", "0.611", "+0.349 [+0.215, +0.483]", "Yes"),
    ("Indiana", "Consolidation", "109 / 544", "0.797", "0.683", "+0.115 [+0.066, +0.169]", "Yes"),
    ("Indiana", "Pulmonary edema", "45 / 544", "0.894", "0.741", "+0.152 [+0.076, +0.230]", "Yes"),
    ("Indiana", "Atelectasis", "161 / 544", "0.802", "0.711", "+0.091 [+0.050, +0.129]", "Yes"),
    ("VinDr", "Cardiomegaly", "358 / 3,000", "0.983", "0.748", "+0.236 [+0.208, +0.264]", "Yes"),
    ("VinDr", "Pleural effusion", "120 / 3,000", "0.995", "0.813", "+0.182 [+0.132, +0.231]", "Yes"),
    ("VinDr", "Pneumothorax", "15 / 3,000", "0.999", "0.505", "+0.494 [+0.335, +0.648]", "Yes"),
    ("VinDr", "Consolidation", "25 / 3,000", "0.991", "0.954", "+0.037 [+0.016, +0.064]", "Yes"),
    ("VinDr", "Atelectasis", "10 / 3,000", "0.986", "0.783", "+0.203 [+0.057, +0.379]", "Yes"),
    ("VinDr", "Infiltration", "44 / 3,000", "0.981", "0.926", "+0.055 [+0.033, +0.081]", "Yes"),
    ("VinDr", "Nodule*", "90 / 3,000", "0.967", "0.805", "+0.162 [+0.120, +0.210]", "Yes"),
    ("VinDr", "Mass*", "90 / 3,000", "0.948", "0.866", "+0.082 [+0.046, +0.123]", "Yes"),
    ("VinDr", "Fibrosis", "182 / 3,000", "0.973", "0.857", "+0.116 [+0.089, +0.145]", "Yes"),
    ("VinDr", "Pleural thickening", "178 / 3,000", "0.953", "0.676", "+0.277 [+0.229, +0.327]", "Yes"),
]


def partc_examples_figure(examples, path):
    """Four NIH X-rays from the trained-model setting, one per category, with the CheXagent decision."""
    from PIL import Image as PILImage
    fig, axs = plt.subplots(1, len(examples), figsize=(3.4 * len(examples), 5.2))
    for ax, (name, e) in zip(axs, examples.items()):
        im = PILImage.open(Path("D:/datasets/chestxray14/images") / e["file"]).convert("L")
        im.thumbnail((600, 600))
        ax.imshow(im, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        stated = {1.0: "stated", 0.5: "stated as possible", 0.0: "not stated"}[e["report_score"]]
        ax.set_title(name.replace(", label", ",\nlabel").replace(" (", "\n("), fontsize=11, color="#16233a")
        ax.text(0.5, -0.04, f"{e['file']}\n{e['condition']}: {stated} in the report\n"
                            f"CheXagent p = {e['p_present']} ({e['decision']})\nNIH label: "
                            f"{'present' if e['label'] else 'absent'}", transform=ax.transAxes, ha="center", va="top",
                fontsize=10.5, color="#16233a", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build(doc):
    d = doc
    # ---------------- title
    t = d.d.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Comparing CheXagent and MAIRA-2 as Image Checkers")
    r.bold = True
    r.font.size = Pt(22)
    t2 = d.d.add_paragraph()
    t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t2.add_run("How we measured detection (AUC) and grounding (box location), and how the image check improves "
                   "MedGemma's reports: methods, datasets, examples and results")
    r.font.size = Pt(13)
    t3 = d.d.add_paragraph()
    t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t3.add_run("MedRAX / EDV project  ·  October 2026  ·  Research results, not clinical advice")
    r.font.size = Pt(10)
    r.italic = True

    # ---------------- summary
    d.h("Summary", 1)
    d.p("Our project adds an independent **image check** (we call it EDV, for evidence-driven verification) to "
        "chest X-ray reports written by MedGemma. A second AI model looks at the same X-ray and says whether "
        "each condition in the report is really there. We considered two models for this job: **CheXagent** "
        "and **MAIRA-2**. This document explains, step by step and in plain language, how we compared them.")
    d.p("We compared two different abilities:")
    d.bullets([
        "**Detection (Part A):** can the model tell whether a condition is present on the X-ray at all? "
        "Measured with **AUC** on 3 datasets and 13 conditions.",
        "**Grounding (Part B):** when the model draws a box to show where a finding is, does the box land "
        "where a radiologist marked it? Measured with **grounding recall** on 947 radiologist-drawn boxes.",
    ])
    d.p("**Main results:**")
    d.bullets([
        "**Detection:** CheXagent separated images with a condition from images without it better than "
        "MAIRA-2 in **26 of 29** dataset-condition comparisons. In the other 3 the difference was not "
        "conclusive, and MAIRA-2 was never significantly better.",
        "**Scoring MAIRA-2 fairly:** MAIRA-2 has no built-in confidence score, so we built one. We then "
        "tested four alternative ways of scoring it; none detected disease better on average than the "
        "score we used.",
        "**Comparing what each model writes (pre-registered, first 500 images):** when both models write their "
        "own reports, they are about equally good (difference +0.020, not significant). CheXagent pulls ahead "
        "only when it is asked about each condition directly (+0.137): it then catches many more findings, at "
        "the cost of more false alarms.",
        "**Grounding:** CheXagent's boxes matched the radiologists' boxes more often: **23.3% vs 19.0%** at "
        "the strict standard (IoU 0.5) and **52.5% vs 30.3%** at the lenient standard (IoU 0.3). "
        "**MAIRA-2 was better for the heart (cardiomegaly).** Neither model draws precise boxes.",
        "**CheXagent improves MedGemma's findings (Part C):** adding CheXagent's image check raised how well "
        "MedGemma's findings identify conditions (mean AUC) from 0.557 to 0.863 with Google's base MedGemma, and "
        "from **0.593 to 0.837 with the team's trained MedGemma**; CheXagent beat MAIRA-2 as the check in all "
        "three experiments. The improvement comes from CheXagent's own detection: CheXagent alone scores about "
        "the same (0.866; 0.840). Editing the reports automatically with a simple rule did not help. The trained "
        "model described change from the prior study in **487 of 499** reports, against 3 of 499 for base "
        "MedGemma.",
        "All main tests were **pre-registered**: the method was written down and saved before any result "
        "was seen.",
    ])

    # ---------------- background
    d.h("1. Background", 1)
    d.h("1.1 The two models", 2)
    d.table(["", "CheXagent", "MAIRA-2"], [
        ("Made by", "Stanford AIMI (version CheXagent-2-3b)", "Microsoft Research"),
        ("What it is built to do", "Answer questions about a chest X-ray, write descriptions, and locate "
                                   "findings with boxes", "Write the findings section of a radiology report, "
                                                          "and locate phrases with boxes"),
        ("Can it answer \"Is there X?\"", "Yes", "No: it only writes reports"),
        ("Has a built-in confidence score", "Yes (probability of answering \"Yes\")",
         "No: we had to build one (Section 3.3)"),
        ("Draws boxes", "Yes", "Yes"),
        ("Public NIH / VinDr images in its training data", "Probably yes", "No (not in its published training data)"),
    ], widths=[4.5, 6, 6], caption="The two models compared.")

    d.h("1.2 Two different abilities", 2)
    d.p("It helps to keep two questions apart:")
    d.bullets([
        "**Detection: \"Is it there?\"** For example: does this X-ray show a pleural effusion (fluid around "
        "the lung)? This is what our image check needs most: confirming or questioning each condition a "
        "report states.",
        "**Grounding: \"Where is it?\"** For example: draw a box around the effusion. This helps a reader see "
        "what the model is pointing at, and check the side (left or right).",
    ])
    d.p("A model can be good at one and poor at the other, so we measured them separately.")

    d.h("1.3 Pre-registration", 2)
    d.p("For every main test we wrote a **protocol** first: the data, the exact method, the measures and "
        "the rule for deciding the result. We saved it in our version-control system (git) **before** "
        "running the models. Results were saved unchanged afterwards. Any change was recorded as a dated "
        "**amendment**, also before the affected run. This prevents choosing the method after seeing which "
        "one gives the nicest answer. Checks that were run after results were known are clearly labelled "
        "**exploratory** in this document.")

    # ---------------- datasets
    d.h("2. Datasets and conditions", 1)
    d.h("2.1 Datasets", 2)
    d.table(["Dataset", "Images used", "Where the labels come from", "Used for"], [
        ("**NIH ChestX-ray14** (US National Institutes of Health)",
         "A random sample of 5,000 images, split into 3,500 for development and **1,500 for testing**",
         "Text-mined automatically from the original radiology reports (so some labels are wrong)",
         "Detection (13 conditions)"),
        ("**NIH box subset**", "**947 boxes** on 847 images (8 conditions)",
         "Radiologists drew a box around each finding", "Grounding"),
        ("**Indiana University** (Open-I)", "**544** images",
         "Standard medical terms (MeSH) attached to each report", "Detection (6 conditions)"),
        ("**VinDr-CXR** (Vietnam)", "**3,000** images",
         "Each image read by 3 radiologists; a condition counts as present if at least 2 of the 3 marked it",
         "Detection (10 conditions); side test"),
    ], widths=[4, 4, 5.5, 3.5], caption="Datasets.")
    d.p("None of these images were used to tune either model in our project. The NIH development "
        "images were used only to set cut-offs for other parts of the project and for exploratory checks; "
        "all NIH detection results below are on the **1,500 test images**.")

    d.h("2.2 The 13 conditions", 2)
    d.table(["Condition", "In plain terms"], [
        ("Cardiomegaly", "Enlarged heart"),
        ("Pleural effusion", "Fluid between the lung and the chest wall"),
        ("Pneumothorax", "Air between the lung and the chest wall (collapsed lung)"),
        ("Consolidation", "Part of the lung filled with fluid or pus instead of air"),
        ("Pulmonary edema", "Fluid in the lungs, often from heart failure"),
        ("Atelectasis", "Part of the lung collapsed or not fully inflated"),
        ("Pneumonia", "Lung infection"),
        ("Infiltration", "A hazy area in the lung (a vague term in NIH's labels)"),
        ("Emphysema", "Damaged, over-inflated lungs, usually from smoking"),
        ("Nodule", "A small round spot in the lung (under 3 cm)"),
        ("Mass", "A larger spot in the lung (3 cm or more)"),
        ("Fibrosis", "Scarring of lung tissue"),
        ("Pleural thickening", "Thickened lining around the lung"),
    ], widths=[4.5, 12], caption="The 13 conditions (the NIH ChestX-ray14 labels, without hernia).")
    d.p("Indiana is labelled only for the first six. VinDr has no pulmonary edema, pneumonia or emphysema "
        "label in this comparison, and labels nodule and mass together, so both are scored against that one label.")

    # ---------------- Part A
    d.page_break()
    d.h("Part A. Detection: \"Is the condition there?\"", 1)
    d.h("3.1 What AUC measures", 2)
    d.p("For each image and condition, each model gives a **score**: a number that should be higher when the "
        "model thinks the condition is more likely. **AUC** (area under the ROC curve) asks one simple question:")
    d.quote("If we pick one image WITH the condition and one image WITHOUT it at random, how often does the "
            "model give the image with the condition the higher score?")
    d.bullets([
        "**AUC = 1.0:** always right; every image with the condition scores above every image without it.",
        "**AUC = 0.5:** no better than a coin toss.",
        "**AUC = 0.9:** right 9 times out of 10.",
    ])
    d.p("AUC looks only at the **order** of the scores, not their size. A model whose scores are all small "
        "(say 0.001 to 0.05) can still have a high AUC, as long as the images with the condition get the "
        "higher of those small numbers. This matters for MAIRA-2 (Section 3.3).")
    d.p("**95% confidence interval (CI).** Every AUC comes with a range, e.g. 0.885 [0.836-0.928]. We "
        "recomputed the AUC on 2,000 random re-draws of the images (a method called *bootstrapping*). The "
        "range covers the middle 95% of those results, and shows how much the number could move with a "
        "different set of images. For the **difference** between the two models, we re-drew the same "
        "images for both, so the comparison is paired.")
    d.p("**Decision rule (fixed in advance).** For each dataset and condition, \"CheXagent better\" was "
        "**supported** if the 95% interval of (CheXagent AUC minus MAIRA-2 AUC) lay entirely above zero. "
        "Conditions with fewer than 10 positive images were reported but not counted as supported.")

    d.h("3.2 How CheXagent's score was obtained", 2)
    d.p("CheXagent answers questions, so we asked it directly, once per condition:")
    d.quote("Is there pleural effusion?")
    d.p("A language model does not just pick an answer; it gives a probability for every possible next "
        "word. We read the probability it gave to **\"Yes\"** compared with **\"No\"** as its first word. This "
        "is CheXagent's score (we call it *p_present*), between 0 and 1. No training or tuning was involved.")
    d.p("**Example** (image 00000096_007, which NIH labels as pleural effusion): CheXagent's probability of "
        "\"Yes\" was **0.948**, so its score is 0.948.")

    d.h("3.3 How MAIRA-2's score was obtained", 2)
    d.p("MAIRA-2 **cannot answer questions**. It was built to do one thing: write the findings section of a "
        "report. So it has no \"Yes\" to read. We had to find another way to get a score from it.")
    d.p("**The idea:** instead of letting MAIRA-2 write freely, we ask how likely it would be to *begin* its "
        "findings with \"There is a pleural effusion.\" versus \"There is no pleural effusion.\"")
    d.p("**Step 1. MAIRA-2 gets the image and its standard report instruction**, exactly as designed:")
    d.quote("You are an expert radiology assistant tasked with interpreting a chest X-ray study.\n"
            "USER: Given the current frontal image only [image] Provide a description of the findings in the "
            "radiology study in comparison to the prior frontal image. INDICATION: N/A TECHNIQUE: N/A "
            "COMPARISON: N/A\nASSISTANT:")
    d.p("**Step 2. We do not let it write; we test two sentences.** A language model writes one piece of a "
        "word (a *token*) at a time, and at each step gives a probability for every possible next token. "
        "Our code supplies the tokens of a sentence we choose and records the probability MAIRA-2 gave each "
        "one. This is a standard way of scoring text with a language model (it is called *teacher "
        "forcing*). The two test sentences, split into MAIRA-2's tokens:")
    d.quote("A:  There | is | a  | ple | ural | eff | usion | .\n"
            "B:  There | is | no | ple | ural | eff | usion | .")
    d.p("The probability of a whole sentence is the product of its token probabilities. The two sentences "
        "are identical except for **\"a\"** versus **\"no\"**, so comparing them is like asking MAIRA-2: "
        "*\"having seen this X-ray and started with 'There is...', would you rather continue with 'a' or 'no'?\"*")
    d.p("**Step 3. Turn the two probabilities into one score:**")
    d.quote("MAIRA-2 score = P(sentence A) / ( P(sentence A) + P(sentence B) )")
    d.p("We did the same for all 13 conditions, each with its own pair of sentences (for example \"There is "
        "a mass.\" / \"There is no mass.\"). That is 26 short sentences per image, all scored after a single "
        "pass over the image.")
    d.p("**Example** (same image 00000096_007): MAIRA-2's log-probabilities were -20.6 for sentence A and "
        "-16.5 for sentence B. Sentence B was about 65 times more likely, so the score is "
        "1 / (1 + e^(20.6-16.5)) = **0.015**.")
    d.table(["", "CheXagent", "MAIRA-2"], [
        ("What we asked", "\"Is there pleural effusion?\"",
         "How likely: \"There is a pleural effusion.\" vs \"There is no pleural effusion.\" as the first sentence"),
        ("What came back", "\"Yes\" with probability 0.948", "Log-probability -20.6 (A) vs -16.5 (B)"),
        ("Score", "**0.948**", "**0.015**"),
    ], widths=[3.5, 5.5, 7.5], caption="How each model's score was obtained, for one image (00000096_007, "
                                         "pleural effusion).")
    d.fig(FIG / "detection_score_examples.png",
          "Pleural effusion scores on three NIH images. MAIRA-2's scores are small even when it describes the "
          "effusion in its own report, but they are still higher for the two images with effusion (0.015, "
          "0.047) than for the image without (0.000). AUC uses only this order.")
    d.p("**Why a score of 0.015 can still detect.** MAIRA-2 almost always prefers \"There is no...\", so its "
        "scores are small. What AUC checks is whether images with the condition score higher than images "
        "without it, which they do in Figure 1.")

    d.h("3.4 Was this fair to MAIRA-2? Two checks", 2)
    d.p("When we let MAIRA-2 write its normal report for image 00000096_007, it said **\"Small pleural "
        "effusions are present bilaterally.\"** So MAIRA-2 *did* see the effusion, even though our score "
        "was only 0.015. Two features of our method could under-read it:")
    d.bullets([
        "**Wording:** MAIRA-2 writes \"Small pleural effusions are present bilaterally\", not \"There is a "
        "pleural effusion.\"",
        "**Position:** we score the sentence as the *first* sentence; MAIRA-2 often starts with something "
        "else (\"The patient is status post median sternotomy...\") and mentions the effusion later.",
    ])
    d.p("**Check 1 (exploratory, 200 NIH development images, pleural effusion).** In its own free-written "
        "reports, MAIRA-2 mentioned effusion in **65 of 100** images labelled with effusion, with **4 false "
        "alarms** among 100 without. That is close to CheXagent's yes/no decision (68 and 4). So as a yes/no "
        "reader of its own reports, MAIRA-2 does about as well as CheXagent for effusion.")
    d.p("**Check 2 (exploratory, 300 other NIH development images, all conditions).** We tried three better-"
        "fitting ways of scoring MAIRA-2 and compared them with the original:")
    d.bullets([
        "**A. MAIRA-2's own phrasings:** we collected the sentences MAIRA-2 actually writes about each "
        "condition (from a separate set of 300 images), e.g. \"Small pleural effusions are present "
        "bilaterally.\" or \"No pleural effusion is evident on this single projection.\", and scored all of them.",
        "**A-strict:** the same, keeping only clean phrasings (written at least twice, about one condition, "
        "not mentioning a tube or line).",
        "**B. Ten reports:** MAIRA-2 writes 10 reports per image with some randomness; the score is the "
        "share that mention the condition (e.g. 7 of 10 = 0.7). This measures exactly what MAIRA-2 would say.",
    ])
    d.table(["MAIRA-2 score", "Matches what MAIRA-2 writes", "Detects NIH-labelled disease (AUC)"], [
        ("**Original sentence pair (used in Table 6)**", "0.883", "**0.828**"),
        ("A. Own phrasings", "0.838", "0.685"),
        ("A-strict. Clean phrasings", "0.878", "0.761"),
        ("B. Share of 10 written reports", "**0.959**", "0.777"),
        ("Its single normal report (yes/no)", "-", "0.654"),
        ("CheXagent \"Is there X?\" (for reference)", "", "0.863"),
    ], widths=[7, 4.5, 5], caption="Four ways of scoring MAIRA-2 (average over the 5 conditions with at "
                                   "least 10 positive images among 300 NIH development images; exploratory).")
    d.p("**What this showed:**")
    d.bullets([
        "Method B tracked what MAIRA-2 writes best (0.959), as intended.",
        "But **tracking its writing did not detect disease better.** The original score had the best "
        "average detection AUC (0.828). Only for pleural effusion did the alternatives help (0.864 to 0.900).",
        "**Why:** MAIRA-2 often notices a finding without writing it down. For example, on image "
        "00013094_000 (NIH: mass), none of its 11 written reports mentions a mass, yet the direct question "
        "\"There is a mass.\" scored 0.784, the 6th highest of 300 images. Asking directly reads this "
        "hidden judgement; counting what it writes loses it.",
    ])
    d.p("**Conclusion:** the original score is the best of the four ways we tried, so the comparison was "
        "not unfair to MAIRA-2 overall. It modestly understates MAIRA-2 for pleural effusion, which we "
        "report as a known limitation.")

    d.h("3.5 Detection results", 2)
    d.p("The comparison was pre-registered in two stages: first the six main conditions, then seven further "
        "conditions (amendments A1 and A2 of the protocol `maira_score_design.md`), each saved before its "
        "readings.")
    d.fig(CHART5, "AUC of CheXagent (blue) and MAIRA-2 (orange) for each condition and dataset, with 95% "
                  "intervals. Numbers on the left: images with the condition / all images tested. Dashed "
                  "line: chance (0.5).")
    d.fig(CHART6, "The difference (CheXagent minus MAIRA-2) for each comparison, with its 95% interval. "
                  "Intervals entirely to the right of 0 mean CheXagent is significantly better.")
    d.table(["Dataset", "Condition", "With condition / total", "CheXagent AUC", "MAIRA-2 AUC",
             "Difference [95% CI]", "CheXagent better?"],
            AUC_ROWS, widths=[1.7, 2.9, 2.4, 1.8, 1.8, 3.6, 2.8], size=8.5,
            caption="Detection results, all 29 dataset-condition comparisons. *VinDr labels nodule and mass "
                    "together.")
    d.p("**In words:** CheXagent was significantly better in **26 of 29** comparisons. The 3 exceptions "
        "were all on NIH: consolidation (close, interval just touches 0), pneumonia (only 9 positive images) "
        "and infiltration (both models near chance; \"infiltration\" is a vague label). MAIRA-2 was never "
        "significantly better.")
    d.p("**What this means for the project:** CheXagent gives a clean, usable confidence score. When we "
        "added CheXagent's score to MedGemma's reports on the same 1,500 NIH test images, the reports' "
        "ability to identify conditions rose from a mean AUC of 0.557 to 0.863; with MAIRA-2's score it "
        "rose to 0.711 (also pre-registered).")

    d.h("3.6 Check 3: comparing what each model writes (pre-registered)", 2)
    d.p("The AUC comparison asks each model about each condition. MAIRA-2 cannot answer questions, so we had "
        "to build a score for it (3.3). A fairer test of MAIRA-2 is to compare **what each model writes in its "
        "own report**, with no score at all. We pre-registered this test (protocol "
        "`own_words_detection_protocol.md`, saved before any report was written) and ran a first stage on a "
        "fixed random **500 of the 1,500 NIH test images**.")
    d.p("**What each model did:**")
    d.bullets([
        "**MAIRA-2's own report:** its normal findings report for the X-ray.",
        "**CheXagent's own report:** CheXagent described six areas of the X-ray (view, airway, breathing, "
        "cardiac, diaphragm, everything else); the six answers together form its report.",
        "**CheXagent's yes/no answer:** for each condition, \"Is there X?\"; we count it as \"present\" when "
        "\"Yes\" outweighs \"No\" (score of at least 0.5). This is the simple 0.5 cut, not our calibrated "
        "cut-offs, which would favour CheXagent.",
    ])
    d.p("The same reader (the Qwen model, with the quote check) read both models' reports for the 13 "
        "conditions; it never saw the X-ray or the labels. For each condition we counted findings **caught** "
        "(sensitivity) and **false alarms** (1 minus specificity), and averaged their mean, the **balanced "
        "accuracy**, over the 7 conditions with at least 10 positive images among the 500.")
    d.table(["Comparison", "Balanced accuracy difference [95% CI]", "Result"], [
        ("**H1:** CheXagent's own report vs MAIRA-2's own report", "+0.020 [-0.014, +0.053]", "No difference shown"),
        ("**H2:** CheXagent's yes/no answer vs MAIRA-2's own report", "+0.137 [+0.092, +0.180]",
         "**CheXagent better**"),
        ("**A1:** AUC on the 1 / 0.5 / 0 report scores, CheXagent report vs MAIRA-2 report",
         "+0.026 [-0.008, +0.059] (AUC difference)", "No difference shown"),
    ], widths=[8, 5, 3.5], caption="Comparing what each model says, 500 NIH test images (stage 1, pre-registered). "
                                   "Mean over 7 conditions with at least 10 positives; 2,000 bootstrap re-draws.")
    d.table(["System", "Findings caught (mean sensitivity)", "Correctly silent (mean specificity)",
             "Balanced accuracy [95% CI]"], [
        ("CheXagent's own report", "0.328", "0.959", "0.643 [0.604, 0.680]"),
        ("CheXagent's yes/no answer", "0.621", "0.899", "0.760 [0.715, 0.803]"),
        ("MAIRA-2's own report", "0.290", "0.956", "0.623 [0.587, 0.661]"),
    ], widths=[5, 4, 4, 4], caption="The three systems on the same 500 images.")
    d.p("**What this showed:**")
    d.bullets([
        "**When both write reports, they are about equal.** CheXagent's report was slightly ahead, but not "
        "significantly (+0.020).",
        "**CheXagent's advantage comes from being asked directly.** Its yes/no answers caught about twice as "
        "many findings as MAIRA-2's report (0.621 vs 0.290), but raised more false alarms (specificity 0.899 "
        "vs 0.956; MAIRA-2 significantly better on this).",
        "**This supports the AUC comparison but narrows what it means:** CheXagent is the better checker when "
        "a validator asks it about each condition, which is how the image check works. It does not show that "
        "CheXagent describes X-rays better than MAIRA-2.",
        "Counting \"possible\" mentions as stated gave the same picture (H1 +0.027, not significant; H2 "
        "+0.138, CheXagent better).",
    ])
    d.p("**Caveats:** first stage only (500 of 1,500 images, NIH labels, 7 conditions in the average); the "
        "same automatic reader scored both reports and has not yet been checked by a radiologist; CheXagent's "
        "six-area prompts are our choice of how to make it write a report.")

    # ---------------- Part B
    d.page_break()
    d.h("Part B. Grounding: \"Where is the condition?\"", 1)
    d.h("4.1 Why we tested this", 2)
    d.p("Our professor asked about **grounding recall**: of the findings a radiologist outlined on the "
        "images, what share does the model's box land on? This is separate from detection. We ran a new "
        "pre-registered experiment (protocol saved before any box was produced).")

    d.h("4.2 How boxes are compared", 2)
    d.p("**IoU (Intersection over Union)** measures how much two boxes overlap:")
    d.quote("IoU = (area where the two boxes overlap) / (area covered by either box)")
    d.bullets([
        "**1.0** = identical boxes; **0** = the boxes do not touch.",
        "IoU is lowered by a box that is too small, too large, or in the wrong place.",
        "**Recall at IoU 0.5** = the share of radiologist boxes that the model's box overlaps with IoU of at "
        "least 0.5. This is the most common standard, and our main measure.",
        "**Recall at IoU 0.3** = the same with a lower bar, often used for chest X-rays because radiologists' "
        "boxes are themselves approximate.",
        "**Pointing game** = a hit if the *centre* of the model's box is inside the radiologist's box. It "
        "ignores box size and is the most lenient measure.",
    ])
    iou_path = FIG / "iou_explained.png"
    iou_figure(iou_path)
    d.fig(iou_path, "How IoU and the pointing game work. Left: perfect match. Middle: a box of the right size "
                    "shifted by half its width has IoU 0.33 (a hit at the 0.3 standard, a miss at 0.5). Right: "
                    "a box far too big still wins the pointing game, but its IoU is low.", width_cm=15)

    d.h("4.3 Data", 2)
    d.p("NIH ChestX-ray14 includes images where radiologists drew a box around a finding. We used every such "
        "box whose image we have, except images in our NIH test split (kept for other experiments): **947 "
        "cases** (one image, one finding, one radiologist box) on **847 images**. All NIH images are 1024 x "
        "1024 pixels, so box positions can be compared directly.")
    d.table(["Condition", "Cases", "Phrase given to the models"], [
        ("Atelectasis", "173", "atelectasis"), ("Pleural effusion", "150", "pleural effusion"),
        ("Cardiomegaly", "141", "cardiomegaly"), ("Infiltration", "117", "infiltration"),
        ("Pneumonia", "116", "pneumonia"), ("Pneumothorax", "93", "pneumothorax"), ("Mass", "81", "mass"),
        ("Nodule", "76", "nodule"), ("**Total**", "**947**", ""),
    ], widths=[5, 3, 6], caption="Grounding cases.")

    d.h("4.4 How each model was asked", 2)
    d.p("Both models got **only the condition name, with no side** (\"pleural effusion\", not \"left pleural "
        "effusion\"). An earlier test showed that if a side is named, both models simply box that side, "
        "even when it is wrong (40 of 40 times each). Each model was told the condition is there, so this "
        "tests **location only, not detection**.")
    d.table(["", "CheXagent", "MAIRA-2"], [
        ("Instruction", "Please locate the following phrase: pleural effusion",
         "Its built-in phrase-grounding mode, given the phrase \"Pleural effusion\""),
        ("Real answer (examples)", "<|ref|> Pleural effusion <|/ref|> <|box|> (10,20),(47,85) <|/box|>",
         "<obj> Mass.<box><x26><y45><x38><y60></box></obj>"),
        ("How coordinates were read", "Numbers are percent of image width and height",
         "Grid positions (0-100), converted with MAIRA-2's own conversion function"),
        ("Settings", "As used in our pipeline", "Full precision, fixed model version, no randomness"),
    ], widths=[3.5, 6.5, 6.5], caption="How each model was asked to draw a box.")
    d.p("A model may give several boxes; we used **its best box** (highest IoU). We also report a version "
        "that combines all its boxes, which penalises extra boxes. **No box counts as a miss.**")

    d.h("4.5 Grounding results", 2)
    d.table(["Measure", "CheXagent", "MAIRA-2", "Difference [95% CI]", "Result"], [
        ("**Recall at IoU 0.5 (main)**", "23.3% (221/947)", "19.0% (180/947)", "+4.3% [+1.7%, +7.0%]",
         "CheXagent better"),
        ("**Recall at IoU 0.3**", "52.5% (497/947)", "30.3% (287/947)", "+22.2% [+18.8%, +25.6%]",
         "CheXagent better"),
        ("Pointing game", "68.5%", "40.0%", "", ""),
        ("Average IoU", "0.31", "0.22", "", ""),
        ("Combined-box recall at IoU 0.5 / 0.3", "23.2% / 49.3%", "18.7% / 29.0%", "", ""),
        ("Boxes per case (average)", "1.26", "0.92", "", ""),
        ("Cases with no box", "9 (1%)", "127 (13%)", "", ""),
    ], widths=[5, 3, 3, 3.5, 2.5], caption="Grounding results, 947 NIH radiologist boxes (pre-registered).")
    d.table(["Condition", "Cases", "Recall IoU 0.5\nCheXagent / MAIRA-2", "Recall IoU 0.3\nCheXagent / MAIRA-2",
             "Pointing game\nCheXagent / MAIRA-2"], [
        ("Atelectasis", "173", "12% / 2%", "42% / 16%", "61% / 31%"),
        ("Pleural effusion", "150", "6% / 3%", "38% / 28%", "67% / 41%"),
        ("Cardiomegaly", "141", "70% / **100%**", "99% / 100%", "99% / 100%"),
        ("Infiltration", "117", "21% / 11%", "58% / 29%", "74% / 44%"),
        ("Pneumonia", "116", "17% / 4%", "53% / 12%", "78% / 16%"),
        ("Pneumothorax", "93", "33% / 13%", "55% / 22%", "68% / 32%"),
        ("Mass", "81", "21% / 1%", "47% / 7%", "63% / 26%"),
        ("Nodule", "76", "1% / 0%", "11% / 3%", "17% / 3%"),
    ], widths=[3.3, 1.6, 3.9, 3.9, 3.9], caption="Grounding by condition.")
    d.p("**In words:**")
    d.bullets([
        "**CheXagent's boxes match the radiologists' more often**, as pre-registered. The lead is small at "
        "the strict standard (+4.3%) and large at the lenient one (+22.2%).",
        "**Neither model draws precise boxes**: at IoU 0.5, only about 1 in 5 radiologist boxes is matched. "
        "CheXagent usually points at the right area (pointing game 68.5%) but sizes its box differently.",
        "**MAIRA-2 is better for the heart (cardiomegaly):** 100% vs 70% at IoU 0.5. Without cardiomegaly "
        "(806 cases; not pre-registered), recall at IoU 0.5 is 15.1% for CheXagent vs 4.8% for MAIRA-2.",
        "**MAIRA-2 often gives no box:** 127 cases (13%), including 74 of the 116 pneumonia cases.",
        "**Nodules are hard for both** (11% vs 3% even at IoU 0.3).",
    ])

    d.h("4.6 Example images", 2)
    d.p("In every figure: **green** = radiologist's box, **blue** = CheXagent, **dashed orange** = MAIRA-2, "
        "with each model's IoU. Figure 5 gives an overview. Figures 6 onwards show, for each condition, up "
        "to six cases picked by a fixed rule: one model's box hits (IoU at least 0.5) and the other's misses "
        "(IoU under 0.3, or no box), largest gap first. These are **selected examples**; each figure states "
        "how often its pattern occurred. To be fair, we show both directions.")
    d.fig(FIG / "grounding_examples.png",
          "Overview. A: mass, CheXagent hits, MAIRA-2's box far too small. B: cardiomegaly, MAIRA-2 closer. "
          "C: pneumothorax, MAIRA-2 boxes the wrong side. D: effusion, CheXagent boxes both sides (NIH has "
          "one box). E: mass, MAIRA-2 gives no box. F: nodule, both miss.")
    d.table(["Condition", "CheXagent hits, MAIRA-2 misses", "MAIRA-2 hits, CheXagent misses",
             "Split at IoU 0.5: only CheXagent / only MAIRA-2"], [
        ("Atelectasis", "16", "3", "20 / 3"), ("Pleural effusion", "6", "3", "9 / 5"),
        ("Cardiomegaly", "0", "1", "**0 / 42**"), ("Infiltration", "11", "2", "18 / 7"),
        ("Pneumonia", "17", "1", "17 / 2"), ("Pneumothorax", "17", "1", "21 / 2"), ("Mass", "16", "0", "17 / 1"),
        ("Nodule", "1", "0", "1 / 0"), ("**All**", "**84**", "**11**", "**103 / 62**"),
    ], widths=[3.5, 4, 4, 5], caption="How often each pattern occurred (947 cases).")
    d.p("Cardiomegaly needs the last column: CheXagent's heart boxes almost always land in the right place "
        "(99% at IoU 0.3) but usually overlap only 0.3-0.5, so MAIRA-2's real advantage shows at the 0.5 line "
        "(42 cases, never the reverse). The cardiomegaly figure below uses that line for both models.")

    d.h("Where MAIRA-2's box is better", 3)
    for f, cap in (
            ("grounding_maira2_better_cardiomegaly_iou05.png",
             "Cardiomegaly (6 of 42): MAIRA-2's box matches the whole heart; CheXagent's covers only its lower "
             "part, or is missing."),
            ("grounding_maira2_better_effusion.png",
             "Pleural effusion (all 3 cases): CheXagent's boxes are much smaller than the effusion."),
            ("grounding_maira2_better_atelectasis.png", "Atelectasis (all 3 cases)."),
            ("grounding_maira2_better_infiltrate.png", "Infiltration (both cases)."),
            ("grounding_maira2_better_pneumonia.png", "Pneumonia (the one case)."),
            ("grounding_maira2_better_pneumothorax.png", "Pneumothorax (the one case)."),
            ("grounding_maira2_better_cardiomegaly.png", "Cardiomegaly by the strict rule (the one case; "
                                                         "CheXagent gave no box).")):
        d.fig(FIG / f, cap)

    d.h("Where CheXagent's box is better", 3)
    for f, cap in (
            ("grounding_chexagent_better_pneumothorax.png",
             "Pneumothorax (6 of 17): in all six the pneumothorax is at the top of the patient's left chest; "
             "MAIRA-2 boxes the same region on the patient's right."),
            ("grounding_chexagent_better_mass.png",
             "Mass (6 of 16): MAIRA-2's box is elsewhere, much too small, or missing."),
            ("grounding_chexagent_better_pneumonia.png",
             "Pneumonia (6 of 17): in 5 of the 6, MAIRA-2 gave no box."),
            ("grounding_chexagent_better_atelectasis.png", "Atelectasis (6 of 16)."),
            ("grounding_chexagent_better_infiltrate.png", "Infiltration (6 of 11)."),
            ("grounding_chexagent_better_effusion.png", "Pleural effusion (all 6)."),
            ("grounding_chexagent_better_nodule.png", "Nodule (the one case).")):
        d.fig(FIG / f, cap)

    d.h("4.7 Side: does the box land on the correct side?", 2)
    d.p("A separate pre-registered test on **VinDr-CXR** (153 findings on one side only: pleural effusion, "
        "consolidation, atelectasis, pneumothorax) asked whether the side of each model's box tells us the "
        "true side. We used a **likelihood ratio (LR+)**: how much more often the box agrees with the true "
        "side than with the wrong side. 1 means no information; above 2 is useful.")
    d.table(["", "CheXagent", "MAIRA-2"], [
        ("Box agrees with the true side", "99%", "54%"),
        ("Box agrees with the wrong side", "14%", "41%"),
        ("Likelihood ratio (95% CI)", "**6.73 (4.78-10.52)**: informative", "1.31 (0.99-1.80): almost no information"),
    ], widths=[5.5, 5.5, 5.5], caption="Side test, VinDr-CXR (pre-registered).")
    d.p("When no side is named, MAIRA-2 tends to box the patient's right, which matches the pneumothorax "
        "examples above.")

    # ---------------- Part C
    d.page_break()
    d.h("Part C. Does the image check improve MedGemma's findings?", 1)
    d.h("5.1 The question", 2)
    d.p("Parts A and B compared the two checker models on their own. Part C asks the question the project is "
        "really about: **when CheXagent's independent check is added to the findings in a report written by "
        "MedGemma, are conditions identified better?** We also repeated the test with MAIRA-2 as the checker, "
        "and with the team's own trained MedGemma. Afterwards, we added a comparison that reviewers asked for: "
        "**the checker on its own**, without the report.")
    d.h("5.2 How a report and a check are combined", 2)
    d.numbered([
        "**Read the report.** A local language model (Qwen) reads MedGemma's report for each of the 13 "
        "conditions and must quote the sentence it relied on. Each condition gets a report score: **1** if "
        "the report states it, **0.5** if it says it is possible, **0** if it does not mention it.",
        "**Put the checker's score on a common scale.** CheXagent's and MAIRA-2's scores are on different "
        "scales, so each is converted to a *percentile*: where that image ranks among all images for that "
        "condition (0 = lowest score, 1 = highest). This uses no labels.",
        "**Average the two:** combined score = half the report score + half the checker's percentile.",
        "**Measure** the mean AUC of the report score alone and of the combined score, against the NIH labels.",
    ])
    d.p("The checker never sees the report: it scores the X-ray on its own (claim-blinded).")
    d.h("5.3 Three experiments", 2)
    d.table(["Experiment", "Report writer", "Images", "What the report writer was given"], [
        ("1. Single image", "Google's base MedGemma 1.5", "1,500 NIH test images",
         "The current X-ray and the project's report prompt"),
        ("2. Prior + current", "Google's base MedGemma 1.5", "499 NIH test pairs (a patient's earlier and current X-ray)",
         "Both X-rays; the checkers were also given both"),
        ("3. Trained model", "**The team's trained MedGemma** (fine-tuned on MIMIC-CXR comparison reports, "
         "then trained further with reinforcement learning)", "The same 499 NIH pairs",
         "Its training format: prior X-ray, prior report, current X-ray, clinical indication. NIH has no "
         "reports, so 'Not available.' and 'Not provided.' were used as placeholders"),
    ], widths=[2.8, 4.5, 4, 5.2], caption="The three experiments (each pre-registered before its reports were generated).")
    d.h("5.4 Results", 2)
    d.fig(FIG / "manuscript_fig4_fusion_with_baselines.png",
          "Mean AUC of MedGemma's findings alone, each checker alone and each combination, in all three "
          "experiments, with 95% intervals. Hollow circles: base MedGemma, single image. Diamonds: base "
          "MedGemma, prior + current. Squares: the team's trained MedGemma, prior + current.", width_cm=15)
    d.table(["Experiment", "MedGemma alone", "+ CheXagent", "+ MAIRA-2", "CheXagent added vs alone",
             "CheXagent vs MAIRA-2 added"], [
        ("1. Base, single image", "0.557", "**0.863**", "0.711", "+0.306 [+0.284, +0.328]", "+0.152 [+0.121, +0.184]"),
        ("2. Base, prior + current", "0.545", "**0.835**", "0.673", "+0.291 [+0.264, +0.316]", "+0.162 [+0.124, +0.201]"),
        ("3. Trained, prior + current", "**0.593**", "**0.837**", "0.693", "+0.244 [+0.220, +0.269]", "+0.144 [+0.111, +0.178]"),
    ], widths=[3.4, 2.2, 2, 2, 3.4, 3.4], size=8.5,
       caption="Mean AUC and the pre-registered differences with 95% intervals. All six hypotheses supported.")
    d.table(["Experiment", "CheXagent alone", "MedGemma + CheXagent", "Combined minus CheXagent alone",
             "MAIRA-2 alone", "Combined minus MAIRA-2 alone"], [
        ("1. Base, single image", "0.866", "0.863", "-0.002 [-0.003, -0.002]", "0.695", "+0.016 [+0.005, +0.028]"),
        ("2. Base, prior + current", "0.840", "0.835", "-0.004 [-0.007, -0.002]", "0.667", "+0.007 [+0.001, +0.012]"),
        ("3. Trained, prior + current", "0.840", "0.837", "-0.003 [-0.008, +0.004]", "0.667", "+0.027 [+0.010, +0.047]"),
    ], widths=[3.4, 2.2, 2.6, 3.4, 2.2, 3.4], size=8.5,
       caption="The checker on its own (exploratory comparison, added after the results were known).")
    d.p("**In words:**")
    d.bullets([
        "**Adding CheXagent's check improves MedGemma's findings in every experiment,** including with the "
        "team's trained model (0.593 to 0.837): conditions are identified much better than from the report "
        "alone.",
        "**The improvement comes from CheXagent's own detection.** CheXagent on its own scores about the same "
        "as the combination (0.840 vs 0.837 with the trained model): the report adds no further detection on "
        "top of CheXagent. With the weaker MAIRA-2, the report does add a little (+0.007 to +0.027).",
        "**CheXagent is the better checker in every experiment** (by 0.14 to 0.16 over MAIRA-2).",
        "**The trained model detects better than base MedGemma on its own:** 0.593 vs 0.545 (difference "
        "+0.049, 95% CI +0.025 to +0.073; exploratory). It is ahead on pleural effusion, consolidation, "
        "atelectasis, pneumothorax and edema.",
        "**The trained model writes comparison reports:** 487 of 499 of its reports describe change from the "
        "prior study, against 3 of 499 for base MedGemma.",
        "**With CheXagent added, trained and base end level** (0.837 vs 0.835): the check dominates detection, "
        "so the trained model's advantage shows mainly in the report itself.",
    ])
    d.h("5.5 A result that did not improve: editing the reports automatically", 2)
    d.p("We also tested a simple rule that actually changes the report: remove a condition the report states if "
        "CheXagent is confidently against it, and add pleural effusion or cardiomegaly if CheXagent confidently "
        "finds them.")
    d.bullets([
        "**Base MedGemma (single image):** balanced accuracy rose slightly (+0.015, 95% CI +0.008 to +0.021), but "
        "all 81 changes were additions, and **59 of them disagreed with the NIH labels** (22 agreed).",
        "**Trained MedGemma:** only 8 changes (1 correct removal, 7 additions that disagreed with the labels); "
        "balanced accuracy changed by -0.0005 (95% CI -0.0010 to -0.0001). The rule rarely triggered because the "
        "trained model already mentions most conditions CheXagent strongly supports.",
    ])
    d.p("**So CheXagent's evidence should flag findings for a reader, not rewrite reports automatically.** "
        "The NIH labels are text-mined, so some of these disagreements may be label errors; a radiologist's "
        "review would settle this.")
    detail = json.loads((ROOT / "experiments" / "results" / "partc_detail.json").read_text(encoding="utf-8"))
    d.h("5.6 Results for each condition", 2)
    d.p("The tables below give every condition in each experiment: the AUC of MedGemma's findings alone, with "
        "each checker added, and each checker on its own. Conditions with fewer than 10 positive images are "
        "marked * and were not included in the averages. The combined scores were pre-registered; the "
        "checker-alone columns are the exploratory comparison.")
    for k, st in detail["settings"].items():
        rows = []
        for r in st["rows"]:
            name = r["condition"].capitalize() + ("*" if r["positives"] < 10 else "")
            best = max(r["fused_chex"], r["fused_maira"])
            rows.append((name, f"{r['positives']} / {r['n']:,}", f"{r['report']:.3f}",
                         (f"**{r['fused_chex']:.3f}**" if r["fused_chex"] == best else f"{r['fused_chex']:.3f}"),
                         (f"**{r['fused_maira']:.3f}**" if r["fused_maira"] == best and r["fused_maira"] > r["fused_chex"]
                          else f"{r['fused_maira']:.3f}"), f"{r['chex']:.3f}", f"{r['maira']:.3f}"))
        d.table(["Condition", "With condition / total", "MedGemma alone", "+ CheXagent", "+ MAIRA-2",
                 "CheXagent alone", "MAIRA-2 alone"], rows, widths=[3.3, 2.6, 2.1, 2.1, 2.1, 2.1, 2.1], size=8.5,
                caption=f"Experiment {st['name']}: AUC per condition. Bold: the better combination.")
    d.p("**What the condition tables show:**")
    d.bullets([
        "**The gain is largest where MedGemma rarely mentions the condition.** Emphysema, fibrosis, pleural "
        "thickening, nodule and mass sit at or near 0.5 for MedGemma alone (it seldom writes about them), and "
        "CheXagent lifts them to 0.78-0.94.",
        "**Where MedGemma's report is already informative, the gain is smaller:** pleural effusion with the "
        "trained model goes from 0.818 to 0.850.",
        "**The trained model's report is better than base MedGemma's** for cardiomegaly, pleural effusion, "
        "pneumothorax, consolidation, edema and atelectasis, which is where it writes most.",
        "**MAIRA-2 helps a few conditions** (consolidation, atelectasis, mass) but adds little for emphysema, "
        "pneumothorax with base MedGemma, or pleural thickening.",
        "**Infiltration stays weak for every system** (about 0.6): a vague NIH label.",
    ])
    d.h("5.7 Where the automatic edits went", 2)
    ed_rows = []
    for k, st in detail["settings"].items():
        for c, v in st["edits"].items():
            if any(v.values()):
                ed_rows.append((st["name"].split(" (")[0], c.capitalize(), str(v["added_agree"]),
                                str(v["added_disagree"]), str(v["removed_agree"]), str(v["removed_disagree"])))
    d.table(["Experiment", "Condition", "Added, NIH label agrees", "Added, label disagrees",
             "Removed, label agrees", "Removed, label disagrees"], ed_rows,
            widths=[4.6, 2.6, 2.3, 2.3, 2.3, 2.3], size=8.5,
            caption="Every change the editing rule made, by condition, compared with the NIH labels.")
    d.p("Nearly all edits were additions of cardiomegaly or pleural effusion, and most additions disagreed with "
        "the NIH labels (base model, single image: 59 of 81). Removals were very rare (one in total). The rule "
        "only ever adds these two conditions; other CheXagent findings were not used to change the report.")
    d.h("5.8 Example cases", 2)
    ex = detail["examples"]
    ex_png = FIG / "partc_examples_trained.png"
    partc_examples_figure(ex, ex_png)
    d.fig(ex_png, "Example cases with the trained MedGemma, chosen by a fixed rule: the first image by file name "
                  "in each category. CheXagent never saw the report.", width_cm=16.5)
    d.table(["Case", "What the report said", "CheXagent", "NIH label", "Meaning"], [
        (ex[k]["file"] + "\n" + ex[k]["condition"],
         {1.0: "states it", 0.5: "says it is possible", 0.0: "does not mention it"}[ex[k]["report_score"]],
         f"p = {ex[k]['p_present']} ({ex[k]['decision']})", "present" if ex[k]["label"] else "absent", meaning)
        for k, meaning in zip(ex, [
            "Agreement: the check supports what the report says.",
            "A catch: the report missed a finding that CheXagent flags. A reader would be prompted to look again.",
            "A false alarm: CheXagent flags a finding that the label does not support.",
            "A correct challenge: the report states a finding that CheXagent and the label do not support.",
        ])
    ], widths=[3.6, 3, 3, 1.9, 5], size=8.5, caption="The four example cases in words.")
    labs = {json.loads(x)["current"]: json.loads(x)["labels"] for x in
            (ROOT / "experiments" / "results" / "edv_trained_labels.jsonl").read_text(encoding="utf-8").splitlines()}
    d.p("**What the report said about the condition** (the reader's verbatim quote from the trained model's "
        "report, or the start of the report when the condition is not mentioned):")
    for k in ex:
        e = ex[k]
        quote = labs[e["file"]][e["condition"]]["evidence"].strip()
        if quote:
            d.quote(f"{e['file']} ({e['condition']}): \"{quote}\"")
        else:
            text = " ".join(e["report"].replace("**", "").replace("* ", "").split())
            hits = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if e["condition"] in x.lower()]
            if hits:
                d.quote(f"{e['file']} ({e['condition']} not stated; the report negates it): \"{hits[0]}\"")
            else:
                d.quote(f"{e['file']} ({e['condition']} not mentioned). Report begins: "
                        f"{text[:300]}{'...' if len(text) > 300 else ''}")
    d.p("**Note on the last case:** the trained model's report only says the cardiomediastinal silhouette is "
        "\"unchanged\" from a prior that showed cardiomegaly. The automatic reader counted this as stating "
        "cardiomegaly. That is a temporal reading error by the reader, not a statement in the report, and it is "
        "the kind of error the planned radiologist review of the reader will measure. The example was kept "
        "because the cases were chosen by a fixed rule.")
    d.h("5.9 Caveats for Part C", 2)
    d.bullets([
        "**What \"improves\" means here:** conditions are identified better when CheXagent's score is combined "
        "with MedGemma's findings, measured against NIH labels. It does not mean the report text itself got "
        "better: CheXagent alone does as well, and automatic editing did not help.",
        "**The report score has only three levels** (0, 0.5, 1), which limits its AUC; a continuous checker "
        "score can rank images more finely.",
        "**The trained model ran outside its training setting:** placeholder prior report and indication, and "
        "NIH rather than MIMIC images. Its own score is probably an underestimate.",
        "**Describing change is not the same as describing it correctly.** NIH labels say what is present on "
        "the current X-ray, not what changed. Whether the trained model's change statements are right needs "
        "the radiologists' comparison reports in the team's MIMIC held-out set (planned).",
        "**The trained model's report-writing advantage** has not yet been tested against MAIRA-2, which is "
        "also built for comparison reports (planned).",
    ])

    # ---------------- together
    d.h("6. Putting it together", 1)
    d.table(["Ability", "Test", "Result"], [
        ("**Detection**: is it there?", "AUC, 13 conditions, 3 datasets (pre-registered)",
         "CheXagent better in 26 of 29; never MAIRA-2"),
        ("Fairness of MAIRA-2's score", "4 scoring methods, NIH development images (exploratory)",
         "Original score detects best; effusion modestly understated"),
        ("**What each model writes**", "Own reports vs CheXagent's yes/no, 500 NIH test images (pre-registered, "
         "stage 1)", "Reports about equal (+0.020, n.s.); CheXagent's yes/no answer better (+0.137)"),
        ("**Grounding**: where is it?", "947 NIH radiologist boxes (pre-registered)",
         "CheXagent better overall (+4.3% at IoU 0.5, +22.2% at 0.3); MAIRA-2 better for the heart"),
        ("**Side**: which side?", "153 one-sided VinDr findings (pre-registered)",
         "CheXagent's side informative (LR+ 6.73); MAIRA-2's not (1.31)"),
        ("**Improving MedGemma's findings**", "3 experiments on NIH, base and trained MedGemma (pre-registered)",
         "CheXagent improves detection over the report alone in every one (+0.24 to +0.31 mean AUC) and beats "
         "MAIRA-2 as the check; CheXagent alone does as well; automatic editing did not help"),
    ], widths=[4, 6, 6.5], caption="Summary of all tests.")
    d.p("**Why we use CheXagent for the image check:** it gives a usable confidence score (needed for the "
        "Supported / Indeterminate / Requires confirmation decisions), it detects conditions better across "
        "datasets, and its box side carries information. Its boxes show roughly where a finding is, not its "
        "exact outline.")

    d.h("7. Limitations", 1)
    d.bullets([
        "**Training data overlap.** CheXagent's paper lists NIH and VinDr among its training data, and excludes "
        "the Indiana (Open-I) images, which it holds out for evaluation. MAIRA-2's image encoder was trained, "
        "without labels, on NIH images. Whether our specific images were seen is unknown. Indiana, excluded from "
        "both, gave the same result (CheXagent better in 6 of 6).",
        "**NIH labels are text-mined** from reports and contain errors.",
        "**MAIRA-2 has no native score**; our score is the best of four we tried, but it modestly understates "
        "MAIRA-2 for pleural effusion.",
        "**NIH has one box per finding,** even when a finding is on both sides; a model that boxes both sides "
        "can score low.",
        "**Box style:** radiologists' boxes follow NIH's own conventions. A model whose style matches them "
        "(MAIRA-2 for the heart) gains.",
        "**Grounding was tested with the condition named**; in real use, a model that wrongly thinks a "
        "condition is present will also draw a box.",
        "**Small numbers** for some conditions (e.g. 9 pneumonia images in NIH detection; 76 nodule boxes).",
        "**Part C** runs the trained model without its usual prior report and indication, and the report "
        "score's three levels make part of the AUC gain mechanical (see 5.6).",
        "**Exploratory checks** (the MAIRA-2 scoring checks, the analysis without cardiomegaly, the example "
        "figures) were run after results were known and support, rather than replace, the pre-registered tests.",
    ])

    d.h("8. Glossary", 1)
    d.table(["Term", "Meaning"], [
        ("AUC", "Probability that a random image with the condition scores higher than a random image without it (0.5 = chance, 1 = perfect)."),
        ("95% confidence interval (CI)", "The range where the true value most likely lies, from 2,000 re-draws of the images."),
        ("Bootstrap", "Repeating a calculation on many random re-draws of the data to see how much the result varies."),
        ("Pre-registration", "Writing down and saving the method and decision rule before seeing any result."),
        ("Exploratory", "A check run after results were known; useful, but weaker evidence."),
        ("Token", "A word or piece of a word, the unit a language model reads and writes."),
        ("Teacher forcing", "Feeding a language model a chosen sentence and reading how probable it found each token."),
        ("p_present", "CheXagent's score: its probability of answering \"Yes\" to \"Is there X?\"."),
        ("Grounding", "A model drawing a box to show where a finding is."),
        ("IoU", "Overlap of two boxes: shared area divided by combined area (0 to 1)."),
        ("Recall at IoU t", "Share of radiologist boxes matched by the model's box with IoU of at least t."),
        ("Pointing game", "A hit if the centre of the model's box lies inside the radiologist's box."),
        ("Likelihood ratio (LR+)", "How much more often a sign appears when something is true than when it is false; 1 = no information."),
        ("McNemar test", "A test that compares two models on the same cases by looking only at the cases where they disagree."),
        ("EDV", "Evidence-driven verification: our independent image check of MedGemma's reports."),
        ("Percentile", "Where one score ranks among all scores: 0 = lowest, 1 = highest."),
        ("Claim-blinded", "The checker is not told what the report says, so it cannot simply agree with it."),
        ("Trained MedGemma", "The team's version of MedGemma, fine-tuned to write comparison reports from a prior and a current X-ray."),
    ], widths=[4.5, 12], caption="Glossary.")

    d.h("Appendix: where the numbers come from", 1)
    d.table(["What", "File (in the project repository)"], [
        ("Detection protocol (pre-registered)", "experiments/maira_score_design.md (amendments A1, A2)"),
        ("Detection results, all conditions", "experiments/results/maira_vs_chexagent_all_conditions.md"),
        ("MAIRA-2 score code", "experiments/maira_score_nih.py"),
        ("MAIRA-2 scoring checks (exploratory)", "experiments/results/maira_score_validity.md, "
                                                 "experiments/results/maira_score_methods_dev.md"),
        ("MAIRA-2 scoring explained", "experiments/maira_scoring_method_explained.md"),
        ("Own-report comparison, stage 1 (pre-registered)", "experiments/own_words_detection_protocol.md, "
                                                             "experiments/results/own_words_detection_stage1.md"),
        ("Part C, experiment 1 (base, single image)", "experiments/edv_improvement_protocol.md, experiments/results/edv_improvement.md"),
        ("Part C, experiment 2 (base, prior + current)", "experiments/edv_longitudinal_protocol.md, experiments/results/edv_longitudinal.md"),
        ("Part C, experiment 3 (trained model)", "experiments/edv_trained_protocol.md, experiments/results/edv_trained.md"),
        ("Part C detail: per condition, edits, examples (exploratory)", "experiments/partc_detail.py, "
                                                                         "experiments/results/partc_detail.md"),
        ("Grounding protocol (pre-registered)", "experiments/grounding_recall_protocol.md"),
        ("Grounding results and every box", "experiments/results/grounding_recall.md, grounding_chexagent.jsonl, "
                                            "grounding_maira2.jsonl"),
        ("Grounding report", "experiments/grounding_recall_report.md"),
        ("Side test (VinDr, pre-registered)", "experiments/results/vindr_grounding_amendment.md"),
        ("Figures", "experiments/results/figures/, logs/charts_20261019-174040/, logs/charts_20261019-173650/, "
                    "logs/charts_20261025-064832/"),
    ], widths=[6, 10.5], caption="Source files.")


def main():
    doc = Doc()
    build(doc)
    out = ROOT / "logs" / f"CheXagent_vs_MAIRA2_detection_and_grounding_{time.strftime('%Y%m%d-%H%M%S')}.docx"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    doc.save(out)
    print(f"{doc.n_fig} figures, {doc.n_tab} tables -> {out}")


if __name__ == "__main__":
    main()
