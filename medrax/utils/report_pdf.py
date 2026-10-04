"""PDF version of the final report shown in the UI (see flow_trace.build_report).

Layout: header, the X-ray and CheXagent's annotated image side by side, the CheXpert
checklist as checkboxes, then the generated report, the ground truth report and the VQA
report (EDV).
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fpdf import FPDF
from PIL import Image

# A Unicode TrueType font (reports contain characters such as "—" and "≥"); Helvetica,
# which only covers Latin-1, is the fallback
_FONT_FILES = {"": "C:/Windows/Fonts/arial.ttf", "B": "C:/Windows/Fonts/arialbd.ttf"}

_GREY = (110, 110, 110)
_BLACK = (0, 0, 0)
_GREEN = (0, 110, 50)
_RED = (170, 30, 30)


class _ReportPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(True, margin=15)
        self.family = "Helvetica"
        if all(Path(f).is_file() for f in _FONT_FILES.values()):
            for style, path in _FONT_FILES.items():
                self.add_font("Arial", style, path)
            self.family = "Arial"

    def text_safe(self, text: str) -> str:
        if self.family == "Arial":
            return text
        return text.encode("latin-1", "replace").decode("latin-1")

    def font(self, size: float, bold: bool = False, color: Tuple[int, int, int] = _BLACK):
        self.set_font(self.family, "B" if bold else "", size)
        self.set_text_color(*color)

    def keep_space(self, height: float):
        """Start a new page unless height mm are left, so a heading stays with its text."""
        if self.get_y() + height > self.h - self.b_margin:
            self.add_page()

    def paragraph(self, text: str, size: float = 10, bold: bool = False, color=_BLACK, gap: float = 1):
        if bold:
            self.keep_space(15)  # a label: keep it with the first lines that follow
        self.font(size, bold, color)
        self.multi_cell(0, size * 0.5, self.text_safe(text), align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(gap)

    def heading(self, text: str, color=_BLACK):
        self.keep_space(25)
        self.ln(3)
        self.font(13, True, color)
        self.cell(0, 7, self.text_safe(text), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 180, 180)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def sections(self, sections: List[Tuple[str, str]]):
        for header, text in sections:
            self.paragraph(header, 10, bold=True, gap=0)
            self.paragraph(text, 10, gap=2)


def _images(pdf: _ReportPDF, images: List[Tuple[str, Optional[str]]]):
    """Draw up to two captioned images side by side (one image is centred)."""
    images = [(caption, path) for caption, path in images if path and Path(path).is_file()]
    if not images:
        return
    box_w = (pdf.w - pdf.l_margin - pdf.r_margin - 6) / 2
    box_h = 80
    # Centre the row of image boxes
    start = pdf.l_margin + (pdf.w - pdf.l_margin - pdf.r_margin - len(images) * box_w
                            - (len(images) - 1) * 6) / 2
    top = pdf.get_y()
    row_h = 0  # height of the tallest image, so captions sit right under the images
    for caption, path in images:
        with Image.open(path) as image:
            row_h = max(row_h, image.height * min(box_w / image.width, box_h / image.height))
    for i, (caption, path) in enumerate(images):
        # Embed a reduced copy (~1000 px, JPEG) so the PDF stays small
        image = Image.open(path).convert("RGB")
        image.thumbnail((1000, 1000))
        scale = min(box_w / image.width, box_h / image.height)
        w, h = image.width * scale, image.height * scale
        left = start + i * (box_w + 6)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        pdf.image(buffer, x=left + (box_w - w) / 2, y=top, w=w, h=h)
        pdf.set_xy(left, top + row_h + 1)
        pdf.font(8, color=_GREY)
        pdf.cell(box_w, 4, pdf.text_safe(caption), align="C")
    pdf.set_xy(pdf.l_margin, top + row_h + 7)


def _checklist(pdf: _ReportPDF, marks: List[Tuple[str, str]]):
    """Checklist as two columns of checkboxes."""
    col_w = (pdf.w - pdf.l_margin - pdf.r_margin) / 2
    rows = (len(marks) + 1) // 2
    top = pdf.get_y()
    box, row_h = 3.8, 6.5
    pdf.set_draw_color(*_BLACK)
    for i, (label, mark) in enumerate(marks):
        col, row = divmod(i, rows)
        x = pdf.l_margin + col * col_w
        y = top + row * row_h
        pdf.rect(x, y + 0.8, box, box)
        if mark.strip():
            pdf.font(9, True, _RED if mark == "?" else _BLACK)
            pdf.set_xy(x, y + 0.8)
            pdf.cell(box, box, mark, align="C")
        pdf.font(10, bold=mark == "X")
        pdf.set_xy(x + box + 2, y)
        pdf.cell(col_w - box - 2, 5.4, pdf.text_safe(label))
    pdf.set_xy(pdf.l_margin, top + rows * row_h + 1)


def write_report_pdf(data: Dict[str, Any], path: Path) -> Path:
    """Write the final report (flow_trace.build_report data) to a PDF file."""
    return write_reports_pdf([data], path)


def write_reports_pdf(reports: List[Dict[str, Any]], path: Path, title: Optional[str] = None) -> Path:
    """Write several final reports to one PDF, each starting on a new page. With a title,
    the first page lists the reports and their page numbers."""
    pdf = _ReportPDF()
    if title:
        pdf.add_page()
        pdf.font(17, True)
        pdf.cell(0, 9, pdf.text_safe(title), new_x="LMARGIN", new_y="NEXT")
        pdf.paragraph(f"{len(reports)} images   |   {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                      9, color=_GREY, gap=4)
        # Clickable index; each link is pointed at its report's page below
        links = [(data["image_name"], pdf.add_link()) for data in reports]
        for name, link in links:
            pdf.font(10)
            pdf.cell(0, 6, pdf.text_safe(name), link=link, new_x="LMARGIN", new_y="NEXT")
    for i, data in enumerate(reports):
        pdf.add_page()
        if title:
            pdf.set_link(links[i][1], page=pdf.page)
        _add_report(pdf, data)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path


def _add_report(pdf: _ReportPDF, data: Dict[str, Any]) -> None:
    """One report, starting at the current position."""
    pdf.font(17, True)
    pdf.cell(0, 9, "Chest X-ray Report", new_x="LMARGIN", new_y="NEXT")
    details = [data["image_name"], datetime.now().strftime("%Y-%m-%d %H:%M")]
    if data.get("medgemma_model"):
        details.append(f"MedGemma: {data['medgemma_model']}")
    pdf.paragraph("   |   ".join(details), 9, color=_GREY, gap=3)

    _images(pdf, [("Chest X-ray", data.get("image_path")),
                  ("CheXagent visual proof", data.get("annotated_path"))])

    pdf.heading("Checklist")
    _checklist(pdf, data["checklist"])

    pdf.heading("Generated Report (MedGemma)")
    pdf.sections(data["generated"])

    pdf.heading("Ground Truth Report")
    if data["truth"]:
        pdf.sections(data["truth"])
    else:
        pdf.paragraph("Not available (no Indiana or MIMIC-CXR gold report for this image)")

    pdf.heading("VQA Report (EDV)")
    pdf.paragraph("Visual Proof:", 10, bold=True, gap=0)
    for line in data["visual_proof"]:
        pdf.paragraph(line, 10)
    pdf.ln(1)
    if "evidence_sections" in data:
        # Alternative evidence layout: [(heading, colour, lines), ...]
        for heading, color, lines in data["evidence_sections"]:
            pdf.paragraph(heading, 10, bold=True, color=color, gap=0)
            for line in lines:
                pdf.paragraph(line, 10)
            pdf.ln(1)
        return
    pdf.paragraph("Supporting Evidence:", 10, bold=True, color=_GREEN, gap=0)
    for line in data["supporting"]:
        pdf.paragraph(line, 10)
    pdf.ln(1)
    pdf.paragraph("Refuting Evidence:", 10, bold=True, color=_RED, gap=0)
    for line in data["refuting"]:
        pdf.paragraph(line, 10)
