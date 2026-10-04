"""PDF of the kinds of queries CheXagent answers, run on one image, with its outputs.

Every query goes through XRayVQATool._run, exactly as in the CheXagent UI: the answer text,
p_present for yes/no questions (and a box search when the answer is Yes), and any boxes,
drawn on the image. Page 1 summarises all queries; the following pages give each answer in
full with its annotated image.

    python experiments/chexagent_queries_report.py [IMAGE]    # writes temp/chexagent_queries_<image>.pdf
"""

import re
import sys
from datetime import datetime
from pathlib import Path

from fpdf.fonts import FontFace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrax.tools.medgemma import describe_location  # noqa: E402
from medrax.tools.xray_vqa import XRayVQATool  # noqa: E402
from medrax.utils.report_pdf import _GREY, _ReportPDF, _images  # noqa: E402

DEFAULT_IMAGE = ROOT / "demo" / "000a0e37-937e40b9-b1318438-ba4e64cd-fb7f8641.png"

# (task, query); grouped as in the CheXagent task list
QUERIES = [
    ("Presence (yes/no)", "Is there pleural effusion?"),
    ("Presence (yes/no)", "Is there pneumothorax?"),
    ("Presence (yes/no)", "Is there atelectasis?"),
    ("Presence (yes/no)", "Is there cardiomegaly?"),
    ("Presence (yes/no)", "Is there pneumonia?"),
    ("Locate a finding", "Please locate the following phrase: pleural effusion"),
    ("Locate a finding", "Please locate the following phrase: pneumothorax"),
    ("Where is it", "Where is the pleural effusion?"),
    ("Free question", "Does this chest X-ray show signs of pneumonia?"),
    ("Findings section", "Write the findings section of the radiology report for this chest X-ray."),
    ("Impression section", "Write the impression section of the radiology report for this chest X-ray."),
    ("View", "What is the view of this chest X-ray?"),
    ("Disease classification", "Which of the following findings are present in this chest X-ray: "
                               "atelectasis, cardiomegaly, consolidation, edema, pleural effusion, pneumothorax?"),
    ("Anatomy description", "Describe the heart size and the mediastinum."),
    ("Lines, tubes and devices", "Are there any support devices, lines or tubes? Describe them."),
    ("Locate a device", "Please locate the following phrase: sternotomy wires"),
]
NOT_RUN = [("Compare two images", "Compare these two chest X-rays: has the pleural effusion changed?",
            "needs two images of the same patient; only one is available")]


def clean(text):
    """One line per anatomy-tagged sentence, bold markers dropped."""
    parts = re.split(r"\s*(?=\[[^\]]+\])", text.replace("**", "").strip())
    return "\n".join(p.strip() for p in parts if p.strip()) or "(empty answer)"


def run(tool, image, query):
    output, _ = tool._run([str(image)], query)
    answer = clean(output.get("response", output.get("error", "")))
    extra = []
    if output.get("yes_no_confidence"):
        extra.append(f"p_present = {output['yes_no_confidence']['p_yes']}")
    if output.get("auto_location"):
        extra.append(f"answer was Yes, so CheXagent was also asked: "
                     f"\"{output['auto_location']['prompt']}\"")
    for box in output.get("boxes", []):
        p = box.get("p_present")
        extra.append(f"box {box['box_normalized']} ({describe_location(box['box_normalized'])}), "
                     f"label \"{box['phrase']}\"" + (f", p_present {p}" if p is not None else ""))
    return {"answer": answer, "extra": extra, "image": output.get("visualization_path")}


def short(text, n=110):
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def main():
    image = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    results = []
    for task, query in QUERIES:
        results.append((task, query, run(tool, image, query)))
        print(f"{task}: {query}", flush=True)

    pdf = _ReportPDF()
    pdf.add_page()
    pdf.font(17, True)
    pdf.cell(0, 9, "CheXagent: queries and answers", new_x="LMARGIN", new_y="NEXT")
    pdf.paragraph(f"{image.name}   |   {datetime.now():%Y-%m-%d %H:%M}   |   CheXagent-2-3b, "
                  "one task per query", 9, color=_GREY, gap=3)
    _images(pdf, [("Chest X-ray", str(image))])

    pdf.heading("Summary")
    pdf.set_draw_color(200, 200, 200)
    pdf.font(8)
    with pdf.table(col_widths=(8, 34, 62, 76), line_height=4, text_align="LEFT",
                   headings_style=FontFace(emphasis="BOLD", fill_color=(235, 235, 235))) as table:
        table.row(["#", "Task", "Query", "CheXagent's answer"])
        for i, (task, query, r) in enumerate(results, 1):
            answer = short(r["answer"]) + (f"  [{r['extra'][0]}]" if r["extra"] and "p_present" in r["extra"][0] else "")
            table.row([str(i), task, query, answer])
        for task, query, why in NOT_RUN:
            table.row(["-", task, query, f"not run: {why}"])

    pdf.add_page()
    pdf.heading("Answers in full")
    for i, (task, query, r) in enumerate(results, 1):
        pdf.keep_space(30)
        pdf.paragraph(f"{i}. {task}", 11, bold=True, gap=0)
        pdf.paragraph(f"Query: {query}", 9, color=_GREY, gap=0.5)
        pdf.paragraph(r["answer"], 10, gap=0.5)
        for line in r["extra"]:
            pdf.paragraph(f"- {line}", 9, gap=0)
        if r["image"]:
            pdf.ln(1)
            pdf.keep_space(85)
            _images(pdf, [("Boxes drawn by CheXagent", r["image"])])
        pdf.ln(3)
    for task, query, why in NOT_RUN:
        pdf.paragraph(f"- {task}", 11, bold=True, gap=0)
        pdf.paragraph(f"Query: {query}", 9, color=_GREY, gap=0.5)
        pdf.paragraph(f"Not run: {why}.", 10)

    out = ROOT / "temp" / f"chexagent_queries_{image.stem[:8]}.pdf"
    pdf.output(str(out))
    print(f"\nPDF: {out}")


if __name__ == "__main__":
    main()
