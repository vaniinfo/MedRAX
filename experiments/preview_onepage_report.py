"""PREVIEW: a one-page version of the report, for one image.

Layout (A4, one page): header; the X-ray, CheXagent's boxed image and the checklist side by
side; MedGemma's report next to the ground truth report; an evidence table with one row per
finding MedGemma reports (agreement level, CheXagent score, box location, location check).
CheXagent checks only MedGemma's findings. Levels use the agreement rule of
preview_agreement_report.py.
Text is shrunk to fit if needed, so the report never runs onto a second page.

    python experiments/preview_onepage_report.py [IMAGE]     # writes temp/onepage_<image>.pdf
"""

import io
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fpdf import FPDF
from fpdf.fonts import FontFace
from langchain_openai import ChatOpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from medrax.evidence import EvidenceEngine  # noqa: E402
from medrax.tools.medgemma import MedGemmaTool  # noqa: E402
from medrax.tools.xray_vqa import XRayVQATool  # noqa: E402
from medrax.utils import flow_trace  # noqa: E402
from preview_agreement_report import cutoffs, described_location  # noqa: E402

DEFAULT_IMAGE = ROOT / "demo" / "000a0e37-937e40b9-b1318438-ba4e64cd-fb7f8641.png"
FONTS = {"": "C:/Windows/Fonts/arial.ttf", "B": "C:/Windows/Fonts/arialbd.ttf"}
GREY, BLACK = (110, 110, 110), (0, 0, 0)
LEVEL_COLOURS = {"High": (0, 110, 50), "Medium": (175, 105, 0), "Low": (170, 30, 30)}
NAMES = {"effusion": "pleural effusion", "edema": "pulmonary edema"}


def strong_yes(engine, finding, p):
    ev = engine.assess(finding, p)
    return ev.state == "SUPPORTING" or (ev.reason == "OUTSIDE_SUPPORT" and p >= 0.5)


def evidence_rows(engine, report, output):
    """One row per finding MedGemma reports: finding, level, score, where, location check."""
    rows = []
    for proof in output.get("visual_proof", []):
        finding = NAMES.get(proof["finding"], proof["finding"])
        name = proof["finding"].capitalize()
        if "boxes" not in proof:
            rows.append((name, "Medium", "not boxed", "-", "-"))
            continue
        p = proof["boxes"][0]["p_present"]
        ev = engine.assess(finding, p)
        yes_cut, no_cut = cutoffs(engine, finding) if ev.reason != "FINDING_NOT_VALIDATED" else (None, None)
        if strong_yes(engine, finding, p):
            level, score = "High", f"{p}  strong yes (from {yes_cut})"
        elif ev.state == "REFUTING" or ev.reason == "OUTSIDE_SUPPORT":
            level, score = "Low", f"{p}  strong no (below {no_cut})"
        elif ev.reason == "FINDING_NOT_VALIDATED":
            level, score = "Medium", f"{p}  no validated cut-off"
        else:
            level, score = "Medium", f"{p}  undecided (strong yes from {yes_cut})"
        where = "; ".join(b["location"].replace("patient's ", "") for b in proof["boxes"])
        side, zone = described_location(report, proof["finding"])
        if side is None and zone is None:
            check = "MedGemma gives no location"
        else:
            box_text = " ".join(b["location"] for b in proof["boxes"])
            sides = {"right" if "right" in b["location"] else "left" if "left" in b["location"] else "both"
                     for b in proof["boxes"]}
            side_ok = side is None or (sides >= {"right", "left"} or "both" in sides if side == "both"
                                       else side in sides or "both" in sides)
            zone_ok = zone is None or zone in box_text
            said = " ".join(w for w in (side, zone) if w)
            check = f"agrees ({said})" if side_ok and zone_ok else f"differs (MedGemma: {said})"
        rows.append((name, level, score, where, check))
    return rows


class OnePage(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_margins(12, 10, 12)
        self.set_auto_page_break(False)
        for style, path in FONTS.items():
            self.add_font("Arial", style, path)

    def font(self, size, bold=False, color=BLACK):
        self.set_font("Arial", "B" if bold else "", size)
        self.set_text_color(*color)

    def title_bar(self, x, y, w, text, size=10):
        self.set_xy(x, y)
        self.font(size, True)
        self.cell(w, 5, text)
        self.set_draw_color(180, 180, 180)
        self.line(x, y + 5.5, x + w, y + 5.5)
        return y + 7

    def image_box(self, path, x, y, w, h, caption):
        image = Image.open(path).convert("RGB")
        size = getattr(self, "image_max", 900)
        image.thumbnail((size, size))
        scale = min(w / image.width, h / image.height)
        iw, ih = image.width * scale, image.height * scale
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92 if size > 900 else 85)
        self.image(buffer, x=x + (w - iw) / 2, y=y, w=iw, h=ih)
        self.set_xy(x, y + h + 0.5)
        self.font(7, color=GREY)
        self.cell(w, 3, caption, align="C")

    def _height(self, w, size, sections):
        line_h = size * 0.42
        height = 0
        for header, text in sections:
            self.font(size)
            height += line_h + line_h * len(self.multi_cell(w, line_h, text, dry_run=True, output="LINES")) + 1
        return height

    def text_block(self, x, y, w, max_h, sections):
        """Header/text sections in a column, shrinking the font until they fit max_h; text
        that still does not fit at the smallest size is cut and ends with "…"."""
        for size in (8.5, 8, 7.5, 7, 6.5, 6):
            if self._height(w, size, sections) <= max_h:
                break
        sections = [list(s) for s in sections]
        while self._height(w, size, sections) > max_h:
            longest = max(sections, key=lambda s: len(s[1]))
            words = longest[1].rstrip("…").split()
            if len(words) < 5:
                break
            longest[1] = " ".join(words[: int(len(words) * 0.9)]) + "…"
        line_h = size * 0.42
        for header, text in sections:
            self.set_xy(x, y)
            self.font(size, True)
            self.cell(w, line_h, header)
            y += line_h
            self.set_xy(x, y)
            self.font(size)
            self.multi_cell(w, line_h, text, align="L")
            y = self.get_y() + 1
        return y


def write_onepage(data, rows, path):
    write_onepages([(data, rows)], path)


def write_onepages(reports, path, title=None):
    """Several one-page reports in one PDF; with a title, a clickable index comes first."""
    pdf = OnePage()
    links = []
    if title:
        pdf.add_page()
        pdf.set_xy(pdf.l_margin, 12)
        pdf.font(15, True)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.font(8, color=GREY)
        pdf.cell(0, 5, f"{len(reports)} images  |  {datetime.now():%Y-%m-%d %H:%M}  |  one page per image",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        for data, _ in reports:
            if pdf.get_y() > pdf.h - 15:
                pdf.add_page()
                pdf.set_y(12)
            link = pdf.add_link()
            links.append(link)
            pdf.font(8.5)
            pdf.cell(0, 4.6, data["image_name"], link=link, new_x="LMARGIN", new_y="NEXT")
    for i, (data, rows) in enumerate(reports):
        pdf.add_page()
        if links:
            pdf.set_link(links[i], page=pdf.page)
        draw_onepage(pdf, data, rows)
    pdf.output(str(path))


def draw_onepage(pdf, data, rows):
    W = pdf.w - pdf.l_margin - pdf.r_margin
    x0 = pdf.l_margin

    # Header
    pdf.set_xy(x0, 10)
    pdf.font(15, True)
    pdf.cell(W / 2, 7, "Chest X-ray Report")
    pdf.font(7.5, color=GREY)
    pdf.cell(W / 2, 7, f"{data['image_name']}  |  {datetime.now():%Y-%m-%d %H:%M}  |  "
                       f"MedGemma: {data['medgemma_model']}", align="R")
    pdf.set_draw_color(0, 0, 0)
    pdf.line(x0, 18, x0 + W, 18)

    # Row 1: images and checklist
    top, img_w, img_h = 21, 55, 64
    pdf.image_box(data["image_path"], x0, top, img_w, img_h, "Chest X-ray")
    if data.get("annotated_path"):
        pdf.image_box(data["annotated_path"], x0 + img_w + 4, top, img_w, img_h, "CheXagent visual proof")
    cx = x0 + 2 * img_w + 12
    cw = W - (cx - x0)
    y = pdf.title_bar(cx, top, cw, "Checklist", 9)
    row_h = (img_h - 5) / len(data["checklist"])
    for label, mark in data["checklist"]:
        pdf.set_draw_color(0, 0, 0)
        pdf.rect(cx, y + 0.5, 3, 3)
        if mark.strip():
            pdf.font(7, True, LEVEL_COLOURS["Low"] if mark == "?" else BLACK)
            pdf.set_xy(cx, y + 0.5)
            pdf.cell(3, 3, mark, align="C")
        pdf.font(8, bold=mark == "X")
        pdf.set_xy(cx + 4.5, y)
        pdf.cell(cw - 5, 4, label)
        y += row_h

    # Row 2: generated vs ground truth
    top = 21 + img_h + 6
    col_w = (W - 6) / 2
    y_left = pdf.title_bar(x0, top, col_w, "Generated Report (MedGemma)")
    y_right = pdf.title_bar(x0 + col_w + 6, top, col_w, "Ground Truth Report")
    truth = data["truth"] or [("", "Not available (no Indiana or MIMIC-CXR gold report for this image)")]
    end_left = pdf.text_block(x0, y_left, col_w, 82, data["generated"])
    end_right = pdf.text_block(x0 + col_w + 6, y_right, col_w, 82, truth)

    # Row 3: evidence check
    top = max(end_left, end_right) + 4
    y = pdf.title_bar(x0, top, W, "Evidence check (EDV)")
    pdf.set_xy(x0, y)
    pdf.set_draw_color(200, 200, 200)
    pdf.font(8)
    with pdf.table(col_widths=(32, 18, 52, 48, 36), line_height=4, text_align="LEFT",
                   headings_style=FontFace(emphasis="BOLD", fill_color=(235, 235, 235)),
                   first_row_as_headings=True) as table:
        table.row(["Finding", "Level", "CheXagent score", "CheXagent box", "Location vs MedGemma"])
        for name, level, score, where, check in rows or [("None reported", "-", "-", "-", "-")]:
            row = table.row()
            row.cell(name)
            row.cell(level, style=FontFace(emphasis="BOLD", color=LEVEL_COLOURS.get(level, BLACK)))
            row.cell(score)
            row.cell(where)
            row.cell(check)


def main():
    load_dotenv()
    image = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    extractor = ChatOpenAI(model="gpt-4o", temperature=0.7, top_p=0.95)
    chexagent = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    tool = MedGemmaTool(locator=chexagent, extractor=extractor)
    engine = EvidenceEngine()

    with flow_trace.FlowTrace(image.name, image_path=str(image), trace_dir=ROOT / "temp" / "preview") as trace:
        output, _ = tool._run(str(image), "(preview: no user question)")
        data = flow_trace.final_report(trace, extractor, write_pdf=False)[0]["data"]
    rows = evidence_rows(engine, output["response"], output)

    out = ROOT / "temp" / f"onepage_{image.stem[:8]}.pdf"
    write_onepage(data, rows, out)
    for r in rows:
        print(r)
    print(f"\nOne-page preview: {out}")


if __name__ == "__main__":
    main()
