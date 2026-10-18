"""Add an EDV section to a prior/current comparison PDF, one page per comparison.

Input: a PDF with one comparison per page (Prior and Current X-ray images, "Generated Report",
"Ground Truth Report"), e.g. subset_comparisons_clean.pdf. For each page:
  - the page's header, images and both reports are copied as they are;
  - the findings the Generated Report states are listed (local Qwen extractor, as in the
    pipeline; the text stays on this machine);
  - CheXagent checks each finding on the CURRENT image only (locate with a side-free phrase,
    then p_present), and each finding gets a High / Medium / Low evidence level (Evidence
    Engine cut-offs, as in experiments/preview_onepage_report.py).

    python experiments/comparison_edv_pdf.py PDF [--out logs/comparisons_edv_<time>.pdf]

Images are the ones embedded in the PDF (low resolution); pdfimages must be on PATH or in
C:/tools/poppler/poppler-24.07.0/Library/bin.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from fpdf.fonts import FontFace  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from medrax.evidence import EvidenceEngine  # noqa: E402
from medrax.tools.medgemma import MedGemmaTool, describe_location  # noqa: E402
from medrax.tools.xray_vqa import XRayVQATool  # noqa: E402
from preview_agreement_report import cutoffs  # noqa: E402
from preview_onepage_report import BLACK, GREY, NAMES, OnePage  # noqa: E402
from provisional_findings import EXPORT as PROVISIONAL_EXPORT  # noqa: E402
from provisional_findings import assess as provisional_assess  # noqa: E402

# Imaging Evidence Review: status label -> text colour; "Requires confirmation" is a priority item
STATUS_STYLE = {
    "Supported": (0, 110, 50),
    "Review characterization": (40, 90, 160),
    "Requires confirmation": (185, 60, 20),
    "Indeterminate": (95, 95, 130),
    "Not assessed": (120, 120, 120),
    "Consider": (40, 90, 160),  # CheXagent's own finding, not in the generated report
}
PRIORITY = "Requires confirmation"

# Where to look, for "Recommend direct review of ..." (by validated finding)
REVIEW_REGION = {"pleural effusion": "{side}pleural space", "pneumothorax": "{side}pleural space",
                 "atelectasis": "{side}lung", "consolidation": "{side}lung",
                 "cardiomegaly": "cardiac silhouette", "pulmonary edema": "lungs"}
# What the image check does not establish, for supported findings
NOT_VERIFIED = {"pleural effusion": "Effusion size and interval change have not been independently verified.",
                "pneumothorax": "Pneumothorax size and interval change have not been independently verified.",
                "atelectasis": "Extent and interval change have not been independently verified.",
                "consolidation": "Cause and interval change have not been independently verified.",
                "cardiomegaly": "Degree of enlargement and interval change have not been independently verified."}


VALIDATED_TERMS = [("pleural effusion", "pleural effusion"), ("effusion", "pleural effusion"),
                   ("pneumothorax", "pneumothorax"), ("atelectasis", "atelectasis"),
                   ("consolidation", "consolidation"), ("cardiomegaly", "cardiomegaly"),
                   ("edema", "pulmonary edema")]


def validated_name(finding):
    """Evidence Engine finding for a stated finding with descriptors, e.g. 'plate-like
    atelectasis' -> 'atelectasis'; the name itself if it is not one of the six."""
    f = NAMES.get(finding, finding)
    return next((v for term, v in VALIDATED_TERMS if term in f), f)


def report_side(report, finding):
    """Side the report gives for this finding: only a side word right next to it
    ('left lower lobe atelectasis', 'right-sided effusion', 'atelectasis in the right base'),
    so a neighbouring finding's side is not borrowed. None if no side is given."""
    head = re.escape(finding.split()[-1].lower().rstrip("s"))
    text = report.lower()
    before = r"\b(right|left|bilateral|bibasilar|bibasal|both)(?:[- ]sided)?\s+(?:[a-z-]+\s+){0,3}?" + head
    after = head + r"\w*\s+(?:in|of|at|on|within|involving)\s+(?:the\s+)?(?:[a-z-]+\s+){0,1}?(right|left|both)\b"
    words = re.findall(before, text) + re.findall(after, text)
    sides = {"both" if w in ("bilateral", "bibasilar", "bibasal", "both") else w for w in words}
    if not sides:
        return None
    return "both" if "both" in sides or sides == {"right", "left"} else sides.pop()


SIX = ["cardiomegaly", "pleural effusion", "pneumothorax", "consolidation", "pulmonary edema", "atelectasis"]
DEVICE_WORDS = ("clip", "line", "tube", "catheter", "pacemaker", "pacer", "wire", "port", "stent", "device",
                "valve", "picc", "stimulator", "hardware", "fusion", "lead", "drain", "sternotomy")


_CUTOFF_CACHE = {}


def _cutoffs(engine, finding):
    if finding not in _CUTOFF_CACHE:
        _CUTOFF_CACHE[finding] = cutoffs(engine, finding)
    return _CUTOFF_CACHE[finding]


def is_device(finding):
    """Whole-word match, so e.g. 'effusion' is not taken for 'fusion' (spinal fusion hardware)."""
    return any(re.search(r"\b" + re.escape(w) + r"s?\b", finding.lower()) for w in DEVICE_WORDS)


def default_cutoffs(engine):
    """(strong yes from, strong no up to) for findings without calibration: the strictest
    values among the six validated findings, so a default call is never easier than a
    validated one."""
    cuts = [_cutoffs(engine, f) for f in SIX]
    return max(c[0] for c in cuts if c[0] is not None), min(c[1] for c in cuts if c[1] is not None)


def provisional_cutoff(q):
    """Support cut-off of a provisional finding, or None if SUPPORTING is not used for it."""
    f = json.loads(PROVISIONAL_EXPORT.read_text(encoding="utf-8"))["findings"][q]
    return f["p_support"] if f["use_supporting"] else None


def review_items(engine, report, output, defaults=None):
    """One radiologist-facing item per finding the report states, from the EDV rules:
    Evidence Engine state of CheXagent's score (claim-blinded) and box laterality vs report."""
    items = []
    for proof in output.get("visual_proof", []):
        if is_device(proof["finding"]):
            continue  # devices (clips, lines, tubes...) are not conditions to verify
        finding = validated_name(proof["finding"])
        side = report_side(report, proof["finding"])
        side_word = {"both": "bilateral", "right": "right", "left": "left"}.get(side)
        phrase = f"{side_word} {proof['finding']}" if side_word else proof["finding"]  # e.g. "left pneumothorax"
        name = phrase[0].upper() + phrase[1:]
        region = REVIEW_REGION.get(finding, "{side}region of interest").format(
            side=f"{side_word} " if side_word in ("right", "left") else "")
        confirm = f"Recommend direct review of the {region} before finalizing the report."
        record = {"finding": proof["finding"], "report_side": side}
        if "boxes" not in proof:
            items.append({**record, "name": name, "status": PRIORITY,
                          "text": f"The automated image check could not locate the reported {phrase} on the image. {confirm}"})
            continue
        p = proof["boxes"][0]["p_present"]
        where = "; ".join(b["location"].replace("patient's ", "") for b in proof["boxes"])
        record.update(p_present=p, image_location=where)
        ev = engine.assess(finding, p)
        record.update(engine_state=ev.state, engine_reason=ev.reason, lr=ev.lr)
        if ev.reason == "FINDING_NOT_VALIDATED":
            prov = provisional_assess(proof["finding"], p)
            if prov is not None:
                # Provisional cut-offs (NIH, provisional_findings_v0.json) for seven more findings;
                # recorded in the .json, not shown in the report
                state, q = prov
                cut = provisional_cutoff(q)
                record.update(cutoff="provisional-v0", provisional_finding=q, provisional_state=state,
                              support_from=cut)
                as_ = "" if q in proof["finding"].lower() else f" (assessed as {q})"
                if state == "SUPPORTING":
                    items.append({**record, "name": name, "status": "Supported",
                                  "text": f"The automated image check supports the reported {phrase}{as_}."})
                elif state == "REFUTING":
                    items.append({**record, "name": name, "status": PRIORITY,
                                  "text": f"The automated image check does not support the reported {phrase}{as_}. {confirm}"})
                else:
                    items.append({**record, "name": name, "status": "Indeterminate",
                                  "text": f"The automated image check is inconclusive for the reported {phrase}{as_}. "
                                          f"Radiologist assessment recommended."})
                continue
            if defaults is None:
                items.append({**record, "name": name, "status": "Not assessed",
                              "text": "No validated automated image check for this finding; radiologist interpretation required."})
                continue
            # Default (uncalibrated) cut-offs for conditions beyond the six; not shown in the report,
            # recorded in the .json ("cutoff": "default")
            d_yes, d_no = defaults
            record.update(cutoff="default", default_cutoffs=[d_yes, d_no], support_from=d_yes)
            if p >= d_yes:
                items.append({**record, "name": name, "status": "Supported",
                              "text": f"The automated image check supports the reported {phrase}."})
            elif p <= d_no:
                items.append({**record, "name": name, "status": PRIORITY,
                              "text": f"The automated image check does not support the reported {phrase}. {confirm}"})
            else:
                items.append({**record, "name": name, "status": "Indeterminate",
                              "text": f"The automated image check is inconclusive for the reported {phrase}. "
                                      f"Radiologist assessment recommended."})
            continue
        strong_yes = ev.state == "SUPPORTING" or (ev.reason == "OUTSIDE_SUPPORT" and p >= 0.5)
        strong_no = ev.state == "REFUTING" or (ev.reason == "OUTSIDE_SUPPORT" and p < 0.5)
        record.update(support_from=_cutoffs(engine, finding)[0])
        # Laterality check against the side the report gives (if any); zones are not compared
        # (the box zones are coarse thirds of the image)
        sides = {"right" if "right" in b["location"] else "left" if "left" in b["location"] else "both"
                 for b in proof["boxes"]}
        side_ok = side is None or (sides >= {"right", "left"} or "both" in sides if side == "both"
                                   else side in sides or "both" in sides)
        image_side = "both sides" if len(sides) > 1 or "both" in sides else f"the {next(iter(sides))}"
        side_note = "" if side_ok else f" The automated image check localizes it to {image_side}."
        if strong_yes and side_ok and finding == "pulmonary edema":
            items.append({**record, "name": name, "status": "Supported",
                          "text": "The automated image check supports pulmonary edema. Distinction from "
                                  "vascular congestion and severity are not assessed."})
        elif strong_yes and side_ok:
            # "a right pleural effusion", "bilateral pleural effusions", "cardiomegaly", "atelectasis"
            countable = finding in ("pleural effusion", "pneumothorax")
            what = (phrase + ("" if phrase.endswith("s") else "s") if side == "both" and countable
                    else f"a {phrase}" if countable else phrase)
            items.append({**record, "name": name, "status": "Supported",
                          "text": f"The automated image check supports {what}. " + NOT_VERIFIED.get(finding, "")})
        elif strong_yes:
            items.append({**record, "name": name, "status": PRIORITY,
                          "text": f"The automated image check supports the finding on {image_side} rather than the "
                                  f"reported {side_word} side. Recommend confirming laterality before finalizing the report."})
        elif strong_no:
            items.append({**record, "name": name, "status": PRIORITY,
                          "text": f"The automated image check does not support the reported {phrase}. {confirm}"})
        elif ev.lr is not None and ev.lr < 1:
            items.append({**record, "name": name, "status": PRIORITY,
                          "text": f"The automated image check finds limited support for the reported {phrase}.{side_note} {confirm}"})
        else:
            items.append({**record, "name": name, "status": "Indeterminate",
                          "text": f"The automated image check finds some support for the reported {phrase}, but not enough to confirm it."
                                  f"{side_note} Radiologist assessment recommended."})
    return items

PDFIMAGES = shutil.which("pdfimages") or "C:/tools/poppler/poppler-24.07.0/Library/bin/pdfimages.exe"
HEADERS = {"FINDINGS:", "IMPRESSION:", "FINDINGS", "FINDING", "IMPRESSION"}


def sections(lines):
    """[(header, text)] from report lines: header lines start a section; a line ending with
    ':' is kept as its own paragraph (e.g. "Comparison of current chest X-ray to prior ...:")."""
    out, header, body = [], "", []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # A bare heading word inside a sentence ("... The / FINDINGS / are unchanged ...") is text
        if line in HEADERS and not line.endswith(":") and body and not body[-1].endswith((".", ":")):
            body.append(line)
            continue
        if line in HEADERS:
            if body or header:
                out.append((header, " ".join(body)))
            header, body = line, []
        elif line.endswith(":") and not body:
            out.append((header, line))
            header = ""
        else:
            body.append(line)
    if body or header:
        out.append((header, " ".join(body)))
    return out


def parse_page(text):
    # The source page number sometimes runs into the last line ("...here, ___.Page 20")
    lines = [l.rstrip() for l in re.sub(r"(?<=\S)(Page \d+)\s*$", r"\n\1", text.rstrip()).splitlines()]
    title = next(l.strip() for l in lines if l.strip().startswith("Comparison "))
    row = next((l.strip() for l in lines if l.strip().startswith("Row ")), "")
    g = next(i for i, l in enumerate(lines) if l.strip() == "Generated Report")
    t = next(i for i, l in enumerate(lines) if l.strip() == "Ground Truth Report")
    end = next((i for i in range(len(lines) - 1, t, -1) if re.fullmatch(r"\s*Page \d+\s*", lines[i])), len(lines))
    return {"title": title, "row": row, "generated": sections(lines[g + 1:t]),
            "truth": sections(lines[t + 1:end]), "generated_text": "\n".join(lines[g + 1:t]),
            "truth_text": "\n".join(lines[t + 1:end])}


NOT_IN_GT = "NOT IN GROUND TRUTH"
GT_PROMPT = """Below is a chest X-ray radiology report (the ground truth) and a list of findings taken from another report of the same study.

For each finding, decide whether the ground-truth report states that this condition is PRESENT on the current study.
- Synonyms and descriptions count: "atelectatic changes" = atelectasis; "heart is mildly enlarged" = cardiomegaly; "layering fluid" = pleural effusion; "airspace process" = opacity.
- Negated ("no pneumothorax"), resolved or not mentioned conditions are NOT present.
- Vascular congestion or engorgement is not pulmonary edema; scarring is not atelectasis.{extra}
- Judge the condition only, not its side.

Answer for each finding:
- if present: where the ground-truth report places it, in a few words taken from the report (e.g. "Left lower lobe", "Right base", "Bilateral", "Right upper quadrant"). If it gives no location, write "Present, location not stated". If the wording is hedged (may, possible, could, suggest), add " (possible)".
- if not present: "NOT IN GROUND TRUTH"
For a present finding, also copy the exact words of the ground-truth report that state it ("evidence"); for a finding that is not present, evidence is "".

Reply with JSON only, one key per finding, keys exactly as given: {{"finding": {{"answer": "...", "evidence": "..."}}, ...}}

GROUND-TRUTH REPORT:
{truth}

FINDINGS:
{findings}"""


def ground_truth_locations(extractor, truth_text, findings, with_evidence=False, extra_rules=""):
    """{finding: where the ground-truth report places it, or NOT_IN_GT} (local LLM; the
    findings are given without the generated report's side, so the side comes from the
    ground truth). A "present" answer whose quoted evidence is not in the report is taken as
    NOT_IN_GT, so an answer the model made up does not reach the table."""
    if not findings:
        return {}
    # extra_rules: additional "- ..." lines for one analysis; empty by default, so the prompt is unchanged
    prompt = GT_PROMPT.format(truth=truth_text.strip(), findings="\n".join(f"- {f}" for f in findings),
                              extra="".join(f"\n- {r}" for r in ([extra_rules] if isinstance(extra_rules, str) and extra_rules
                                                                  else extra_rules or [])))
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()  # noqa: E731
    for _ in range(3):
        text = extractor.invoke(prompt).content
        m = re.search(r"\{.*\}", text, re.S)
        try:
            answer = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            answer = None
        if isinstance(answer, dict) and all(isinstance(answer.get(f), dict) for f in findings):
            out = {}
            for f in findings:
                where = str(answer[f].get("answer", "")).strip() or NOT_IN_GT
                evidence = str(answer[f].get("evidence", "")).strip()
                # The model may join several sentences into one quote: each must be in the report
                quotes = [norm(q) for q in re.split(r"(?<=[.;])\s+", evidence) if norm(q)]
                rejected = where != NOT_IN_GT and (not quotes or any(q not in norm(truth_text) for q in quotes))
                if rejected:
                    where = NOT_IN_GT
                # with_evidence: (answer, quoted evidence, answer the quote check rejected or None)
                out[f] = (where, evidence, str(answer[f].get("answer", "")) if rejected else None) \
                    if with_evidence else where
            return out
    raise RuntimeError(f"no usable ground-truth answer for {findings}: {text!r}")


def mark(text, phrases):
    """text with each phrase (case-insensitive, any whitespace) wrapped as bold underline markdown."""
    for phrase in phrases:
        for piece in re.split(r"(?<=[.;])\s+", phrase.strip()):
            words = piece.strip(" .;").split()
            if len(words) < 2:
                continue
            pattern = r"\s+".join(re.escape(wd) for wd in words)
            text = re.sub(pattern, lambda m: f"--**{m.group(0)}**--", text, count=1, flags=re.I)
    return text


def draw_sections(pdf, x, y, w, max_h, secs, highlights=None):
    """Sections in a column, font shrunk to fit max_h (empty headers are not printed). Phrases in
    `highlights` are drawn in bold underline."""
    md = bool(highlights)
    if md:
        secs = [(header, mark(text, highlights)) for header, text in secs]

    def height(size):
        lh, h = size * 0.42, 0
        for header, text in secs:
            pdf.font(size)
            h += (lh if header else 0) + lh * len(pdf.multi_cell(w, lh, text, dry_run=True, output="LINES",
                                                                  markdown=md)) + 1
        return h
    size = next((s for s in (8.5, 8, 7.5, 7, 6.5, 6, 5.5) if height(s) <= max_h), 5.5)
    lh = size * 0.42
    for header, text in secs:
        if header:
            pdf.set_xy(x, y)
            pdf.font(size, True)
            pdf.cell(w, lh, header)
            y += lh
        pdf.set_xy(x, y)
        pdf.font(size)
        pdf.multi_cell(w, lh, text, align="L", markdown=md)
        y = pdf.get_y() + 1
    return y


def draw_review(pdf, x, y, w, items):
    """'IMAGING EVIDENCE REVIEW:' section of the generated report: heading with a small italic
    note beside it, then one entry per finding ("Name - Status" and a sentence)."""
    pdf.set_xy(x, y)
    pdf.font(8.5, True)
    heading = "IMAGING EVIDENCE REVIEW:"
    pdf.cell(pdf.get_string_width(heading) + 2, 4, heading)
    if "ariali" not in pdf.fonts:
        pdf.add_font("Arial", "I", "C:/Windows/Fonts/ariali.ttf")
    pdf.set_font("Arial", "I", 6.5)
    pdf.set_text_color(*GREY)
    pdf.cell(90, 4, "Secondary image-based assessment of the generated findings")
    y += 5.6
    if not items:
        pdf.set_xy(x, y)
        pdf.font(8.2)
        pdf.multi_cell(w, 3.8, "No abnormal findings are stated in the generated report.", align="L")
        return pdf.get_y() + 1
    for item in items:
        pdf.set_xy(x, y)
        pdf.font(8.6, True)
        pdf.cell(pdf.get_string_width(item["name"]) + 1, 4.2, item["name"])
        pdf.font(8.6)
        pdf.cell(pdf.get_string_width(" — ") + 0.5, 4.2, " — ")
        pdf.font(8.6, True, STATUS_STYLE[item["status"].replace(" (default cut-off)", "")])
        pdf.cell(60, 4.2, item["status"])
        y += 4.4
        pdf.set_xy(x, y)
        pdf.font(8.2)
        pdf.multi_cell(w, 3.8, item["text"], align="L")
        y = pdf.get_y() + 1.8
    return y


def draw_page(pdf, *args, **kwargs):
    """One page; the image row is made shorter if the text would otherwise run into the legend
    (each height is tried on a scratch document first)."""
    for box_h in (62, 56, 50, 44, 38):
        if _draw_page(OnePage(), *args, box_h=box_h, **kwargs):
            break
    _draw_page(pdf, *args, box_h=box_h, **kwargs)


def _draw_page(pdf, page, prior, current, annotated, items, structured=None, box_h=62, show_reports=True,
               gt_highlights=None):
    """Draws the page; True if everything fits above the legend."""
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin
    x0 = pdf.l_margin
    pdf.set_xy(x0, 9)
    pdf.font(15, True)
    pdf.cell(W, 7, page["title"], align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.font(8, color=GREY)
    pdf.cell(W, 4, page["row"], align="C")

    top = 22
    images = [("Prior", prior), ("Current", current)] + ([("Visual Proof", annotated)] if annotated else [])
    box_w = (W - 4 * (len(images) - 1)) / len(images)
    for i, (caption, path) in enumerate(images):
        pdf.image_box(path, x0 + i * (box_w + 4), top, box_w, box_h, caption)

    y = top + box_h + 6
    if show_reports:
        y = pdf.title_bar(x0, y, W, "Generated Report")
        y = draw_sections(pdf, x0, y, W, 46, page["generated"]) + 1.5
        y += 1.5
    if structured:
        # a thin rule sets the template apart from what is above it
        pdf.set_draw_color(205, 208, 214)
        pdf.line(x0, y + 0.5, x0 + W, y + 0.5)
        y = draw_structured(pdf, x0, y + 2, W, 70 if show_reports else pdf.h - 16 - y, structured) + 2.5
    if not show_reports:
        # template-only version: no ground truth, comparison table or legend; page number only
        pdf.set_xy(x0, pdf.h - 12)
        pdf.font(6.8, color=GREY)
        pdf.cell(W, 3.2, f"Page {pdf.page_no()} of {{nb}}", align="R")
        return y <= pdf.h - 14
    legend_top = pdf.h - 38
    top = y
    y = pdf.title_bar(x0, y, W, "Ground Truth Report")
    if gt_highlights:
        pdf.set_xy(x0 + pdf.get_string_width("Ground Truth Report") + 6, top + 0.6)
        pdf.set_font("Arial", "I", 6.5)
        pdf.set_text_color(*GREY)
        pdf.cell(150, 4, "Bold underlined: the ground-truth text that each [Image check] change in the "
                         "interpretation is checked against")
    table_h = 7 + draw_table(pdf, x0, 0, W, items, dry_run=True) + 3  # title bar + table + gap
    y = draw_sections(pdf, x0, y, W, max(legend_top - 3 - table_h - y, 20), page["truth"],
                      highlights=gt_highlights) + 3
    y = pdf.title_bar(x0, y, W, "Finding Comparison")
    y += draw_table(pdf, x0, y, W, items)
    draw_legend(pdf, x0, legend_top, W)
    return y <= legend_top - 1


TABLE_COLS = [("MedGemma finding", 0.32), ("CheXagent evidence", 0.28), ("Ground truth location", 0.40)]


def draw_table(pdf, x, y, w, items, dry_run=False):
    """MedGemma finding | CheXagent evidence (review status) | ground-truth location.
    Returns the table height."""
    size, lh, pad = 8, 3.6, 1.3
    # Status with CheXagent's score (p_present) in brackets, e.g. "Supported (0.80)"
    rows = [(i["name"], i["status"] + (f" ({i['p_present']:.2f})" if i.get("p_present") is not None
                                       else " (not located)"),
             i.get("ground_truth", "")) for i in items] or \
           [("No abnormal findings stated", "", "")]
    widths = [w * f for _, f in TABLE_COLS]

    def height(cells, bold=False):
        pdf.font(size, bold)
        return pad * 2 + lh * max(len(pdf.multi_cell(cw - 2 * pad, lh, c or " ", dry_run=True, output="LINES"))
                                  for c, cw in zip(cells, widths))

    total = height([c for c, _ in TABLE_COLS], True) + sum(height(r) for r in rows)
    if dry_run:
        return total
    pdf.set_draw_color(200, 204, 212)
    for n, cells in enumerate([[c for c, _ in TABLE_COLS]] + rows):
        h = height(cells, n == 0)
        cx = x
        for k, (text, cw) in enumerate(zip(cells, widths)):
            if n == 0:
                pdf.set_fill_color(236, 239, 244)
            pdf.rect(cx, y, cw, h, style="DF" if n == 0 else "D")
            if n == 0:
                pdf.font(size, True)
            elif k == 1 and text.split(" (")[0] in STATUS_STYLE:
                pdf.font(size, True, STATUS_STYLE[text.split(" (")[0]])
            elif k == 2 and text == NOT_IN_GT:
                pdf.font(size, True, GREY)
            else:
                pdf.font(size)
            pdf.set_xy(cx + pad, y + pad)
            pdf.multi_cell(cw - 2 * pad, lh, text, align="L")
            cx += cw
        y += h
    return total


LEGEND = [
    ("Supported", "the automated image check scores the finding at or above its support cut-off (a likelihood "
                  "ratio of at least 5) and locates it on the reported side."),
    ("Indeterminate", "the score leans towards the finding but is below the support cut-off: not enough evidence "
                      "either way."),
    ("Requires confirmation", "the score leans against the finding, the finding is located on the opposite "
                              "side, or it could not be located."),
    ("Consider", "the automated image check found this condition on its own (score at its cut-off), but the generated "
                 "report does not mention it; a prompt for radiologist review, not part of the report."),
]


def draw_legend(pdf, x, y, w):
    pdf.set_draw_color(205, 208, 214)
    pdf.line(x, y, x + w, y)
    y += 1.5
    for status, text in LEGEND:
        pdf.set_xy(x, y)
        pdf.font(6.8, True, STATUS_STYLE.get(status, BLACK))
        label = status + ": "
        lw = pdf.get_string_width(label)
        pdf.cell(lw, 3.2, label)
        pdf.font(6.8, color=GREY)
        pdf.multi_cell(w - lw, 3.2, text, align="L")
        y = pdf.get_y() + 0.6
    pdf.set_xy(x, y + 0.6)
    pdf.font(6.8, color=GREY)
    pdf.cell(w, 3.2, f"Page {pdf.page_no()} of {{nb}}", align="R")


def row_image(images_dir, row):
    """current.jpg of the folder for a page's "Row N" label (row_NN_subject_...)."""
    n = int(re.search(r"\d+", row).group(0))
    matches = sorted(Path(images_dir).glob(f"row_{n:02d}_*/current.jpg"))
    if len(matches) != 1:
        raise SystemExit(f"{row}: expected one row_{n:02d}_*/current.jpg in {images_dir}, found {len(matches)}")
    return str(matches[0])


def shrink(path, work, size=1200):
    """Downscaled copy of a full-resolution visual proof, for the PDF."""
    from PIL import Image
    if path is None:
        return None
    image = Image.open(path)
    image.thumbnail((size, size))
    out = work / (Path(path).stem + "_small.png")
    image.save(out)
    return str(out)


INTERPRETATION_PROMPT = ROOT / "medrax" / "docs" / "medgemma_interpretation_prompt.txt"
INTERPRETATION_HEADINGS = ["LINES/TUBES/IMPLANTS", "LUNGS/PLEURA", "HEART/MEDIASTINUM",
                           "BONES/SOFT TISSUES/UPPER ABDOMEN"]


def add_interpretation(rec, model, sorter):
    """Structured interpretation written by MedGemma from the prior and current images (same
    settings as the report tool: RGB, shrunk to fit 1024x1024, greedy, 450 tokens), into
    rec["interpretation"]; reused if already there. Only the four requested headings are kept
    (MedGemma sometimes adds FINDINGS/IMPRESSION); the raw output is saved too."""
    if rec.get("interpretation"):
        return
    import base64
    import io

    from langchain_core.messages import HumanMessage
    from PIL import Image

    def data_url(path):
        image = Image.open(path).convert("RGB")
        image.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    current = Path(rec["image"])
    prior = current.with_name("prior.jpg")
    prompt = INTERPRETATION_PROMPT.read_text(encoding="utf-8").strip()
    llm = ChatOpenAI(model=model, temperature=0, base_url="http://localhost:11434/v1", api_key="ollama")
    messages = [HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": data_url(prior)}},
        {"type": "image_url", "image_url": {"url": data_url(current)}},
        {"type": "text", "text": prompt}])]
    raw = llm.invoke(messages, max_tokens=450).content
    sections = parse_interpretation(raw)
    reformatted = None
    if len(sections) < 3:
        # MedGemma sometimes writes its usual FINDINGS/IMPRESSION report instead. A follow-up turn
        # with the images breaks it (Ollama) and text-only it cannot sort, so the local text model
        # (sorter) puts MedGemma's FINDINGS sentences under the headings; any sentence that is not
        # word for word in MedGemma's answer is dropped, so the content stays MedGemma's.
        findings = re.split(r"\**\s*IMPRESSION\s*:", raw)[0]
        findings = re.sub(r"^\s*\**\s*FINDINGS\s*:\**", "", findings).strip()
        reformatted = sorter.invoke(
            "Sort each sentence of this chest X-ray report under exactly one of these headings, each "
            "heading starting a new line: " + ", ".join(h + ":" for h in INTERPRETATION_HEADINGS)
            + " Copy the sentences word for word; do not add, change or remove any. Write \"Not commented "
            "on.\" under a heading no sentence belongs to.\n\nREPORT:\n" + findings, max_tokens=450).content
        norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()  # noqa: E731
        kept = {}
        for head, text in parse_interpretation(reformatted).items():
            sentences = [s for s in re.split(r"(?<=\.)\s+", text)
                         if norm(s) and (norm(s) in norm(findings) or norm(s) == "not commented on")]
            if sentences:
                kept[head] = " ".join(sentences)
        sections = kept
    rec["interpretation"] = {"model": model, "prompt_file": INTERPRETATION_PROMPT.name,
                             "images": [str(prior), str(current)], "raw": raw, "reformatted": reformatted,
                             "sorted_by": getattr(sorter, "model_name", None) if reformatted else None,
                             "sections": [[h, sections.get(h, "Not commented on.")]
                                          for h in INTERPRETATION_HEADINGS]}
    print(f"  interpretation: {rec['interpretation']['sections']}", flush=True)


STRUCTURED_PROMPT = """Below are numbered sentences from the FINDINGS of a chest X-ray report. Assign each sentence to exactly one heading:
COMPARISON (only says what the study is compared with)
TECHNIQUE (only about how the image was taken: view, position, portable, rotation, inspiration)
LINES/TUBES/IMPLANTS (catheters, lines, tubes, pacemakers, clips, other devices or hardware)
LUNGS/PLEURA (lungs, airspace, atelectasis, edema, effusion, pneumothorax, lung volumes)
HEART/MEDIASTINUM (heart size, cardiac silhouette, mediastinum, hila, aorta)
BONES/SOFT TISSUES/UPPER ABDOMEN (bones, spine, soft tissues, free air, abdomen)
A sentence that mixes topics goes under the heading of its main finding.
Reply with JSON only: {{"1": "HEADING", "2": "HEADING", ...}}

SENTENCES:
{sentences}"""
STRUCTURED_SUBSECTIONS = INTERPRETATION_HEADINGS
# A heading the generated report has no sentence for (wording chosen by the user)
NO_STATEMENT = "Not assessed."


def add_structured(rec, page, sorter):
    """The radiologist's template filled from the generated report as it is: IMPRESSION word for
    word, and each FINDINGS sentence placed, unchanged, under one heading (the local text model
    only labels numbered sentences; the sentences themselves are copied here). Into
    rec["structured"]; reused if already there."""
    if rec.get("structured"):
        return
    findings = " ".join(t for h, t in page["generated"] if "IMPRESSION" not in h)
    impression = " ".join(t for h, t in page["generated"] if "IMPRESSION" in h).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.:])\s+(?=[A-Z0-9])", re.sub(r"\s+", " ", findings)) if s.strip()]
    heads = STRUCTURED_SUBSECTIONS + ["COMPARISON", "TECHNIQUE"]
    labels = {}
    for _ in range(3):
        text = sorter.invoke(STRUCTURED_PROMPT.format(
            sentences="\n".join(f"{n}. {s}" for n, s in enumerate(sentences, 1)))).content
        m = re.search(r"\{.*\}", text, re.S)
        try:
            answer = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            answer = {}
        labels = {int(k): v.strip().upper() for k, v in answer.items()
                  if str(k).isdigit() and isinstance(v, str) and v.strip().upper() in heads}
        if set(labels) == set(range(1, len(sentences) + 1)):
            break
    else:
        raise RuntimeError(f"{rec['title']}: could not place every sentence: {labels}")
    placed = {h: " ".join(s for n, s in enumerate(sentences, 1) if labels[n] == h) for h in heads}
    rec["structured"] = {
        "sorted_by": getattr(sorter, "model_name", None), "sentences": sentences, "labels": labels,
        "fields": [["COMPARISON", placed["COMPARISON"] or "Prior chest radiograph."],
                   ["IMPRESSION", impression or "Not stated."],
                   ["CLINICAL INDICATION", "Not provided."],
                   ["TECHNIQUE", placed["TECHNIQUE"] or "Frontal radiograph of the chest."]],
        "interpretation": [[h, placed[h] or NO_STATEMENT] for h in STRUCTURED_SUBSECTIONS]}
    print(f"  structured: {[(h, labels[n]) for n, h in enumerate(sentences, 1)]}", flush=True)


def draw_structured(pdf, x, y, w, max_h, structured):
    """The radiologist's template below the generated report, in its order: EXAMINATION,
    COMPARISON (the report's comparison sentence), IMPRESSION (word for word), CLINICAL
    INDICATION, TECHNIQUE, INTERPRETATION with its four sub-headings, and ATTENDING RADIOLOGIST
    marked as an unreviewed draft; font shrunk to fit max_h."""
    fields = dict(structured["fields"])
    lines = [(0, "**EXAMINATION:** Chest radiograph.")] + \
            [(0, f"**{h}:** {fields[h]}") for h in ("COMPARISON", "IMPRESSION", "CLINICAL INDICATION", "TECHNIQUE")] + \
            [(0, "**INTERPRETATION:**")] + \
            [(4, f"**{h}:** {NO_STATEMENT if t in ('Not commented on.', 'No statement in the AI-generated draft.') else t}")  # older saved runs
             for h, t in structured["interpretation"]] + \
            [(0, "**ATTENDING RADIOLOGIST:** AI-generated draft — Not reviewed by a radiologist")]

    def height(size):
        pdf.font(size)
        return sum(size * 0.42 * len(pdf.multi_cell(w - indent, size * 0.42, l, markdown=True, dry_run=True,
                                                      output="LINES")) + 0.9 for indent, l in lines)
    size = next((s for s in (8.5, 8, 7.5, 7, 6.5) if height(s) <= max_h), 6.5)
    for indent, line in lines:
        pdf.set_xy(x + indent, y)
        pdf.font(size)
        pdf.multi_cell(w - indent, size * 0.42, line, markdown=True, align="L")
        y = pdf.get_y() + 0.9
    return y


def parse_interpretation(raw):
    """{heading: text} for the four headings; other headings (FINDINGS, INTERPRETATION) and
    everything from an IMPRESSION on are dropped."""
    raw = re.split(r"\**\s*IMPRESSION\s*:", raw)[0]
    pattern = r"^\s*\**\s*(" + "|".join(re.escape(h) for h in INTERPRETATION_HEADINGS) + \
              r"|[A-Z][A-Z /]+)\s*:\**\s*(.*)$"
    sections, head = {}, None
    for line in raw.splitlines():
        m = re.match(pattern, line)
        if m:
            head = m.group(1).strip() if m.group(1).strip() in INTERPRETATION_HEADINGS else None
            line = m.group(2)
        if head and line.strip():
            sections[head] = (sections.get(head, "") + " " + line.strip()).strip()
    return sections


def draw_interpretation(pdf, x, y, w, max_h, sections):
    """'Interpretation' section: "HEADING: text" lines, font shrunk to fit max_h."""
    lines = [f"**{h}:** {t}" for h, t in sections]

    def height(size):
        pdf.font(size)
        return sum(size * 0.42 * len(pdf.multi_cell(w, size * 0.42, l, markdown=True, dry_run=True,
                                                      output="LINES")) + 0.8 for l in lines)
    size = next((s for s in (8.5, 8, 7.5, 7, 6.5) if height(s) <= max_h), 6.5)
    for line in lines:
        pdf.set_xy(x, y)
        pdf.font(size)
        pdf.multi_cell(w, size * 0.42, line, markdown=True, align="L")
        y = pdf.get_y() + 0.8
    return y


_MAIRA = None


def add_maira(rec):
    """MAIRA-2 phrase grounding for each reviewed finding, side-free phrase (as in VinDr Part B),
    on the image CheXagent read; into rec["maira"] and each review item (reused if already there).
    Location only: MAIRA-2 gives no presence score."""
    global _MAIRA
    maira = rec.get("maira") or {}
    for f in sorted({i["finding"] for i in rec["review"]} - set(maira)):
        if _MAIRA is None:
            from vindr_grounding_amendment import maira_locator
            _MAIRA = maira_locator()
        result = _MAIRA(rec["image"], f)
        where = "; ".join(dict.fromkeys(
            describe_location(b).replace("patient's ", "").replace(",", "") for b in result["boxes"]))
        maira[f] = {**result, "location": where[0].upper() + where[1:] if where else "Not located"}
    rec["maira"] = maira
    for i in rec["review"]:
        i["maira_location"] = maira[i["finding"]]["location"]
    print(f"  MAIRA-2: {({f: v['location'] for f, v in maira.items()})}", flush=True)


def make_extractor(model):
    return ChatOpenAI(model=model, temperature=0, base_url="http://localhost:11434/v1",
                      api_key="ollama", extra_body={"think": False})


def add_ground_truth(extractor, page, rec, redo=False):
    """Ground-truth location per reviewed finding, into rec["ground_truth"] and each review item
    (reused from the .json if already there)."""
    gt = {} if redo else rec.get("ground_truth") or {}
    missing = sorted({i["finding"] for i in rec["review"]} - set(gt))
    if missing:
        gt.update(ground_truth_locations(extractor, page["truth_text"], missing))
    rec["ground_truth"] = gt
    for i in rec["review"]:
        i["ground_truth"] = gt[i["finding"]]
    print(f"  ground truth: {gt}", flush=True)


def add_suggestions(extractor, page, rec, screen, truth=None):
    """CheXagent's own findings from an independent screen (chexagent_independent_screen.py: "Is there
    X?" for 13 conditions, found = at its cut-off) that the generated report does not state, as
    "Consider" items with their ground-truth location: from `truth` ({(title, condition): location},
    the saved lookup the other analyses use) if given, else looked up and cached in rec["ground_truth"]."""
    from chexagent_independent_screen import canonical

    stated = {canonical(f) for f in rec.get("findings", [])} | {canonical(i["finding"]) for i in rec["review"]}
    found = [s for s in screen.get(rec["title"], []) if s["chexagent_found"] and s["condition"] not in stated]
    gt = dict(rec.get("ground_truth") or {})
    if truth is not None:
        gt.update({s["condition"]: truth[(rec["title"], s["condition"])] for s in found})
    else:
        missing = sorted({s["condition"] for s in found} - set(gt))
        if missing:
            gt.update(ground_truth_locations(extractor, page["truth_text"], missing))
        rec["ground_truth"] = gt
    rec["suggestions"] = [{
        "finding": s["condition"], "status": "Consider", "p_present": s["p_present"],
        "name": f"{s['condition'].capitalize()} (not in report; CheXagent: "
                f"{'; '.join(dict.fromkeys(s.get('chexagent_location', '').split('; ')))})",
        "ground_truth": gt[s["condition"]]} for s in found]
    print(f"  suggestions: {[(i['finding'], round(i['p_present'], 2), i['ground_truth']) for i in rec['suggestions']]}",
          flush=True)


def render_from_json(args):
    """Layout-only rebuild of a previous run: findings, review statuses and boxes from its .json."""
    from types import SimpleNamespace

    work = Path(tempfile.mkdtemp(prefix="comparisons_", dir=ROOT / "temp"))
    subprocess.run([PDFIMAGES, "-png", args.pdf, str(work / "img")], check=True)
    reader = PdfReader(args.pdf)
    record = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    drawer = SimpleNamespace(temp_dir=work)
    engine = EvidenceEngine()
    defaults = default_cutoffs(engine) if args.default_cutoffs else None
    revised = json.loads(Path(args.revisions_json).read_text(encoding="utf-8")) if args.revisions_json else None
    evidence = None
    if args.highlight_gt:  # pilot ground-truth review sheet: quoted ground-truth sentence per page/condition
        import csv
        with open(args.highlight_gt, encoding="utf-8-sig") as f:
            # keyed by the page's "Row N" label (the sheet's page numbers are positions in its PDF)
            evidence = {(r["row_folder_subject"], r["condition"]): r["auto_evidence"] for r in csv.DictReader(f)}
    screen = truth = None
    if args.screen_json:
        screen = {p["title"]: p["screen"] for p in json.loads(Path(args.screen_json).read_text(encoding="utf-8"))}
    if args.truth_json:  # saved ground-truth lookup for all conditions (screen_three_way.py output)
        truth = {(r["page"], r["condition"]): r["ground_truth"]
                 for r in json.loads(Path(args.truth_json).read_text(encoding="utf-8"))}
    extractor = make_extractor(args.ollama_model)
    if defaults:
        print(f"default cut-offs: strong yes >= {defaults[0]}, strong no <= {defaults[1]}")
    pdf = OnePage()
    skip = {int(r) for r in args.skip_rows.split(",")} if args.skip_rows else set()
    kept = []
    for k, (p, rec) in enumerate(zip(reader.pages, record)):
        page = parse_page(p.extract_text())
        assert page["title"] == rec["title"], f"page {k + 1}: {page['title']} vs {rec['title']}"
        if int(re.search(r"\d+", page["row"]).group(0)) in skip:
            print(f"{page['title']} ({page['row']}): left out (--skip-rows)")
            continue
        kept.append(rec)
        prior, current = (str(work / f"img-{2 * k + i:03d}.png") for i in (0, 1))
        boxes = [{"phrase": v["finding"], "box_normalized": b["box_normalized"], "p_present": b["p_present"]}
                 for v in rec["visual_proof"] if not is_device(v["finding"]) for b in v.get("boxes", [])]
        # Boxes on the image CheXagent read (full resolution if the run used --images-dir)
        annotated = shrink(XRayVQATool._visualize_boxes(drawer, rec.get("image") or current, boxes), work) \
            if boxes else None
        if args.default_cutoffs:
            rec["review"] = review_items(engine, page["generated_text"], {"visual_proof": rec["visual_proof"]},
                                         defaults)
            print(f"{rec['title']}: {[(i['name'], i['status']) for i in rec['review']]}")
        add_ground_truth(extractor, page, rec, redo=args.redo_ground_truth)
        if args.maira:
            add_maira(rec)
        if args.interpretation:
            add_interpretation(rec, args.medgemma_model, extractor)
        if args.structured:
            add_structured(rec, page, extractor)
        if screen is not None:
            add_suggestions(extractor, page, rec, screen, truth)
        items = rec["review"] + (rec.get("suggestions", []) if screen is not None else [])
        structured = rec.get("structured")
        highlights = None
        if revised is not None and structured:  # revise_interpretation.py output: tagged headings
            structured = {**structured, "interpretation": revised[rec["title"]]["revised_interpretation"]}
            if evidence is not None:  # ground-truth sentences behind each changed condition
                changed = {c for _, c, _ in revised[rec["title"]]["revisions"]}
                highlights = [evidence[(page["row"], c)] for c in sorted(changed) if evidence.get((page["row"], c))]
        draw_page(pdf, page, prior, current, annotated, items, structured, show_reports=not args.hide_reports,
                  gt_highlights=highlights)
    out = Path(args.out) if args.out else ROOT / "logs" / f"comparisons_review_{time.strftime('%Y%m%d-%H%M%S')}.pdf"
    if out.exists():
        raise SystemExit(f"{out} already exists; not overwriting")
    pdf.output(str(out))
    # all pages are kept in the .json (it must line up with the source PDF for later rebuilds)
    out.with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"{len(kept)} pages re-rendered from {args.from_json} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ollama-model", default="qwen3.8-edv:latest")  # qwen3.5 broken by Ollama 0.40 migration
    ap.add_argument("--default-cutoffs", action="store_true",
                    help="conditions beyond the six get the strictest validated cut-offs, labelled (default cut-off)")
    ap.add_argument("--images-dir", default=None,
                    help="folder with row_NN_*/current.jpg (full resolution) for CheXagent, matched to each "
                         "page's 'Row N'; the page still shows the PDF's own images")
    ap.add_argument("--interpretation", action="store_true",
                    help="with --from-json: add a structured Interpretation section written by MedGemma "
                         "from the prior and current images (the generated report is not changed)")
    ap.add_argument("--medgemma-model", default="medgemma1.5:4b")
    ap.add_argument("--screen-json", default=None,
                    help="with --from-json: an independent CheXagent screen (chexagent_independent_screen.py); "
                         "conditions it finds that the report does not state are added as 'Consider' rows")
    ap.add_argument("--revisions-json", default=None,
                    help="with --structured: draw the INTERPRETATION revised by revise_interpretation.py "
                         "(image-check tags inline) instead of the original")
    ap.add_argument("--highlight-gt", default=None,
                    help="with --revisions-json: ground-truth review sheet (pilot_ground_truth_sheet.py CSV); the "
                         "ground-truth sentence behind each changed condition is drawn in bold underline")
    ap.add_argument("--truth-json", default=None,
                    help="with --screen-json: saved ground-truth lookup for all conditions (…_three_way.json), so "
                         "'Consider' rows use the same ground truth as the other analyses")
    ap.add_argument("--skip-rows", default=None,
                    help="with --from-json: comma-separated 'Row N' numbers to leave out, e.g. 10,12")
    ap.add_argument("--hide-reports", action="store_true",
                    help="with --from-json: template-only pages (images, the radiologist's template, page "
                         "number); no generated/ground-truth reports, Finding Comparison table or legend")
    ap.add_argument("--structured", action="store_true",
                    help="with --from-json: add the radiologist's report template, filled from the generated "
                         "report's findings and impression as they are")
    ap.add_argument("--maira", action="store_true",
                    help="with --from-json: add a MAIRA-2 location column (phrase grounding, side-free phrase)")
    ap.add_argument("--redo-ground-truth", action="store_true",
                    help="with --from-json: look up the ground-truth locations again instead of reusing them")
    ap.add_argument("--from-json", default=None,
                    help="re-render from a previous run's .json (same findings, scores and statuses; "
                         "boxes redrawn from the saved coordinates; no model calls)")
    args = ap.parse_args()
    if args.from_json:
        return render_from_json(args)

    work = Path(tempfile.mkdtemp(prefix="comparisons_", dir=ROOT / "temp"))
    subprocess.run([PDFIMAGES, "-png", args.pdf, str(work / "img")], check=True)
    reader = PdfReader(args.pdf)
    extractor = make_extractor(args.ollama_model)
    chexagent = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    tool = MedGemmaTool(locator=chexagent, extractor=extractor)  # only its finding extraction is used
    engine = EvidenceEngine()

    pages, record = [], []
    for k, p in enumerate(reader.pages):
        page = parse_page(p.extract_text())
        prior, current = (str(work / f"img-{2 * k + i:03d}.png") for i in (0, 1))
        # CheXagent reads the full-resolution current image if given, else the PDF's embedded one
        analysed = row_image(args.images_dir, page["row"]) if args.images_dir else current
        # Devices are not reviewed, so they are not located (no boxes on the visual proof either)
        findings = [f for f in tool._extract_findings(page["generated_text"]) if not is_device(f)]
        output, annotated = {"visual_proof": []}, None
        if findings:
            proof, annotated = chexagent.locate_findings(analysed, findings)
            annotated = shrink(annotated, work)
            output["visual_proof"] = [
                {"finding": q["finding"],
                 "boxes": [{"box_normalized": b["box_normalized"], "location": describe_location(b["box_normalized"]),
                            "p_present": b.get("p_present")} for b in q["boxes"]]}
                if q["localized"] else {"finding": q["finding"], "status": "not localized"}
                for q in proof]
        items = review_items(engine, page["generated_text"], output,
                             default_cutoffs(engine) if args.default_cutoffs else None)
        rec = {"page": k + 1, "title": page["title"], "image": analysed, "findings": findings,
               "visual_proof": output["visual_proof"], "review": items}
        add_ground_truth(extractor, page, rec)
        pages.append((page, prior, current, annotated, items))
        record.append(rec)
        print(f"{page['title']}: {[(i['name'], i['status']) for i in items]}", flush=True)

    pdf = OnePage()
    for args_ in pages:
        draw_page(pdf, *args_)
    out = Path(args.out) if args.out else ROOT / "logs" / f"comparisons_review_{time.strftime('%Y%m%d-%H%M%S')}.pdf"
    if out.exists():
        raise SystemExit(f"{out} already exists; not overwriting")
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    out.with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"\n{len(pages)} pages -> {out}")


if __name__ == "__main__":
    main()
