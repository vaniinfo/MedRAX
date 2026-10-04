"""PREVIEW: one page comparing CheXagent's own report with MedGemma's and the gold report.

CheXagent is asked for each section separately (one task per question works best); MedGemma
gets its usual report prompt. No visual proof: CheXagent cannot check its own report.

    python experiments/preview_chexagent_report.py [IMAGE]    # writes temp/chexagent_report_<image>.pdf
"""

import re
import sys
from datetime import datetime
from pathlib import Path

from fpdf.fonts import FontFace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from medrax.tools.medgemma import MedGemmaTool  # noqa: E402
from medrax.tools.xray_vqa import XRayVQATool  # noqa: E402
from medrax.utils.flow_trace import ground_truth, split_report  # noqa: E402
from preview_onepage_report import GREY, OnePage  # noqa: E402

DEFAULT_IMAGE = ROOT / "demo" / "000a0e37-937e40b9-b1318438-ba4e64cd-fb7f8641.png"
QUESTIONS = {
    "FINDINGS:": "Write the findings section of the radiology report for this chest X-ray.",
    "IMPRESSION:": "Write the impression section of the radiology report for this chest X-ray.",
    "VIEW:": "What is the view of this chest X-ray?",
}

# Hand comparison for the default image (000a0e37), from the three reports
NOTES = [
    ("Pleural effusion", "bilateral, right > left", "bilateral", "moderate bilateral, more on the right"),
    ("Pneumothorax", "small right apical", "\"No pneumothorax\"", "tiny right apical"),
    ("Atelectasis", "compressive, under the effusions", "right lower lobe", "not stated (wet read: consolidations)"),
    ("Vessels", "elevated venous pressure", "not mentioned", "vascular engorgement indeterminate"),
    ("Devices", "not mentioned", "sternotomy wires, valve", "post-operative mediastinum"),
]


def chexagent_text(raw):
    """One line per sentence, anatomy tag kept, bold markers dropped."""
    parts = re.split(r"\s*(?=\[[^\]]+\])", raw.replace("**", "").strip())
    return "\n".join(p.strip() for p in parts if p.strip())


def main():
    image = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    chexagent = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    chex = {}
    for header, question in QUESTIONS.items():
        raw, _ = chexagent._generate_response([str(image)], question, 512)
        chex[header] = chexagent_text(raw)
    medgemma, _ = MedGemmaTool()._run(str(image), "(preview)")
    med_findings, med_impression = split_report(medgemma["response"])
    truth = ground_truth(image.name, str(image))

    pdf = OnePage()
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin
    x0 = pdf.l_margin

    pdf.set_xy(x0, 10)
    pdf.font(15, True)
    pdf.cell(W / 2, 7, "Chest X-ray Report: CheXagent")
    pdf.font(7.5, color=GREY)
    pdf.cell(W / 2, 7, f"{image.name}  |  {datetime.now():%Y-%m-%d %H:%M}", align="R")
    pdf.line(x0, 18, x0 + W, 18)

    # Image and CheXagent's answer on the view
    pdf.image_box(str(image), x0, 21, 70, 70, "Chest X-ray")
    tx = x0 + 78
    y = pdf.title_bar(tx, 21, W - 78, "Models compared")
    pdf.set_xy(tx, y)
    pdf.font(8.5)
    pdf.multi_cell(W - 78, 4.2,
                   "CheXagent-2-3b: each section asked separately (findings, impression, view).\n"
                   "MedGemma 1.5 (medgemma1.5:4b): fixed report prompt, as in medgemma_vs_gold.csv.\n"
                   f"CheXagent's answer to \"What is the view of this chest X-ray?\": {chex['VIEW:']}",
                   align="L")

    # Three reports side by side
    top = 21 + 70 + 7
    col_w = (W - 8) / 3
    columns = [
        ("CheXagent", [("FINDINGS:", chex["FINDINGS:"]), ("IMPRESSION:", chex["IMPRESSION:"])]),
        ("MedGemma 1.5", [("FINDINGS:", med_findings), ("IMPRESSION:", med_impression or "-")]),
        ("Ground Truth", truth[1] if truth else [("", "Not available")]),
    ]
    ends = []
    for i, (title, sections) in enumerate(columns):
        x = x0 + i * (col_w + 4)
        y = pdf.title_bar(x, top, col_w, title)
        ends.append(pdf.text_block(x, y, col_w, 95, sections))

    # Comparison table (hand-made for the default image only)
    if image == DEFAULT_IMAGE:
        top = max(ends) + 4
        y = pdf.title_bar(x0, top, W, "Comparison")
        pdf.set_xy(x0, y)
        pdf.set_draw_color(200, 200, 200)
        pdf.font(8)
        with pdf.table(col_widths=(30, 50, 46, 60), line_height=4, text_align="LEFT",
                       headings_style=FontFace(emphasis="BOLD", fill_color=(235, 235, 235))) as table:
            table.row(["", "CheXagent", "MedGemma 1.5", "Ground Truth"])
            for row in NOTES:
                table.row(list(row))
        pdf.set_xy(x0, pdf.get_y() + 3)
        pdf.font(7.5, color=GREY)
        pdf.multi_cell(W, 3.6,
                       "CheXagent describes change over time (\"interval increase\") from a single image with "
                       "no prior study, and may have been trained on MIMIC-CXR, so its agreement with this "
                       "gold report may be inflated.", align="L")

    out = ROOT / "temp" / f"chexagent_report_{image.stem[:8]}.pdf"
    pdf.output(str(out))
    print(f"CheXagent findings:\n{chex['FINDINGS:']}\n\nCheXagent impression:\n{chex['IMPRESSION:']}")
    print(f"\nPDF: {out}")


if __name__ == "__main__":
    main()
