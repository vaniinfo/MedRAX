"""Per-request flow trace of the MedGemma -> extractor -> CheXagent pipeline.

One log file per user message, named after the image, recording what goes into and comes out
of each model: the director's tool calls, MedGemma's opinion, the findings the extractor
lists, every CheXagent query (the exact decoded input), its raw grounded output and, for
yes/no questions, the top vocabulary tokens behind p_present. It ends with a final report:
CheXpert checklist, generated report, ground truth report (Indiana images) and the EDV
visual proof.

The active trace is a module-level global (the UI serves one local user). Tools call
current() and log only if a trace is active, so they behave the same without one.
"""

import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

TRACE_DIR = Path("logs/flow_traces")
INDIANA_REPORTS = Path("C:/Users/hari/_workarea_/MedRAX/data/indiana_kaggle/indiana_reports.csv")
# MIMIC-CXR studies: gold_report is the ground truth; the images are demo/<dicom_id>.png
DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
MIMIC_GOLD = DEMO_DIR / "medgemma_vs_gold.csv"

_current: Optional["FlowTrace"] = None


def current() -> Optional["FlowTrace"]:
    """The active trace, or None."""
    return _current


class FlowTrace:
    """A log file for one request; use as a context manager to make it the active trace."""

    def __init__(
        self, image_name: Optional[str], image_path: Optional[str] = None, trace_dir: Path = TRACE_DIR
    ):
        self.image_name = image_name or "no-image"
        self.image_path = image_path  # the file the tools read, for the ground truth lookup
        stem = re.sub(r"[^\w.-]+", "_", Path(self.image_name).stem) or "image"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = trace_dir / f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}.log"
        self.file = self.path.open("w", encoding="utf-8")
        # MedGemma results, kept for the final report
        self.medgemma_runs: List[Dict[str, Any]] = []

    def __enter__(self) -> "FlowTrace":
        global _current
        if _current is not None:  # a request that was cut off (e.g. the browser closed)
            _current.file.close()
        _current = self
        return self

    def __exit__(self, *exc) -> None:
        global _current
        if _current is self:
            _current = None
        self.file.close()

    def log(self, text: str = "") -> None:
        self.file.write(text + "\n")
        self.file.flush()

    def block(self, title: str, text: Any) -> None:
        self.log(f"  {title}:")
        for line in str(text).splitlines() or [""]:
            self.log(f"    | {line}")

    def section(self, title: str) -> None:
        """Start a step: "[title]", or "[step] detail" for titles written as "step] detail"."""
        self.log(f"\n[{title}" + ("" if "]" in title else "]"))


# ---------------------------------------------------------------- final report

CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices",
]

CHECKLIST_PROMPT = (
    "Below is a chest X-ray report. Label it with these CheXpert labels: "
    + ", ".join(CHEXPERT_LABELS) + ".\n"
    "A label is \"present\" if the report states it is seen, \"uncertain\" if the report "
    "calls it possible, suspected or cannot exclude it. Labels the report states are absent, "
    "or does not mention, are left out. Lung Opacity covers any opacity or increased density "
    "in the lungs. Output JSON only: {\"present\": [...], \"uncertain\": [...]} using the exact "
    "label names.\n\nReport:\n"
)

# Evidence Engine v1.0 names for the findings it validates
ENGINE_NAMES = {"effusion": "pleural effusion", "edema": "pulmonary edema"}


def checklist(trace: FlowTrace, extractor, report: str):
    """CheXpert checklist of a report (text-only call to the extractor)."""
    reply = extractor.invoke([HumanMessage(content=CHECKLIST_PROMPT + report)], max_tokens=200)
    trace.section(f"Checklist labelling of MedGemma's report] model: {extractor.model_name}")
    trace.block("output (raw)", reply.content)
    match = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", reply.content), re.S)
    try:
        labels = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        labels = {}
    known = {label.lower(): label for label in CHEXPERT_LABELS}
    present = {known[x.lower()] for x in labels.get("present", []) if x.lower() in known}
    uncertain = {known[x.lower()] for x in labels.get("uncertain", []) if x.lower() in known} - present
    return present, uncertain


def ground_truth(image_name: str, image_path: Optional[str] = None):
    """Ground truth report for this image, or None.

    Returns (source, [(section, text), ...]) with FINDINGS and IMPRESSION first. Looks up:
      - Indiana (Open-i) reports, by file name "<uid>_IM-..." or "<label>_CXR<uid>.png";
      - MIMIC-CXR gold reports (demo/medgemma_vs_gold.csv), by the dicom_id in the file name,
        or else by comparing the image with the demo/<dicom_id>.png files (uploads are often
        renamed, e.g. "1.png").
    """
    name = Path(image_name).name
    match = re.match(r"(\d+)_IM-", name) or re.match(r"\w+_CXR(\d+)\.", name)
    if match and INDIANA_REPORTS.is_file():
        with INDIANA_REPORTS.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["uid"] == match.group(1):
                    return (f"Indiana (Open-i) report, uid {row['uid']}",
                            [("FINDINGS:", row["findings"]), ("IMPRESSION:", row["impression"])])

    if MIMIC_GOLD.is_file():
        with MIMIC_GOLD.open(encoding="utf-8") as f:
            rows = {row["dicom_id"]: row for row in csv.DictReader(f)}
        dicom_id = next((d for d in rows if d in name), None)
        if dicom_id is None and image_path:
            dicom_id = _same_image(image_path, [d for d in rows if (DEMO_DIR / f"{d}.png").is_file()])
        if dicom_id:
            row = rows[dicom_id]
            return (f"MIMIC-CXR gold_report ({MIMIC_GOLD.name}), study {row['study_id']}, "
                    f"dicom_id {dicom_id}", mimic_sections(row["gold_report"]))
    return None


def _same_image(image_path: str, dicom_ids: List[str]) -> Optional[str]:
    """dicom_id of the demo image showing the same picture as image_path, if any."""
    from PIL import Image
    import numpy as np

    def thumb(path):
        return np.asarray(Image.open(path).convert("L").resize((32, 32)), dtype=float)

    try:
        target = thumb(image_path)
    except Exception:
        return None
    for dicom_id in dicom_ids:
        if np.abs(thumb(DEMO_DIR / f"{dicom_id}.png") - target).mean() < 2:
            return dicom_id
    return None


# Administrative sections of a MIMIC-CXR report, left out of the ground truth shown
_MIMIC_ADMIN = {"EXAMINATION", "EXAM", "INDICATION", "HISTORY", "CLINICAL HISTORY", "COMPARISON",
                "COMPARISONS", "TECHNIQUE", "REASON FOR EXAM", "REASON FOR EXAMINATION",
                "TYPE OF EXAMINATION", "CLINICAL INFORMATION", "STUDY"}


def mimic_sections(report: str) -> List[tuple]:
    """FINDINGS, IMPRESSION and other clinical sections of a MIMIC-CXR report's final report
    (the wet read before it is dropped).

    Reports without a FINDINGS header put the observations in unlabelled paragraphs (after an
    administrative section such as REASON FOR EXAMINATION) or under headers such as
    "PA AND LATERAL VIEWS OF THE CHEST"; those become the findings.
    """
    text = "\n" + report.split("FINAL REPORT", 1)[-1]
    parts = re.split(r"\n[ \t]*([A-Z][A-Z /&]*[A-Z]):", text)
    flat = lambda s: " ".join(s.split())
    narrative = [parts[0]]  # text outside any clinical section
    sections = {}
    for header, body in zip(parts[1::2], parts[2::2]):
        if header in _MIMIC_ADMIN:
            # Only the first paragraph belongs to an administrative header
            narrative += re.split(r"\n\s*\n", body)[1:]
        elif flat(body):
            sections[header] = flat(body)
    narrative = flat(" ".join(narrative))

    findings = sections.pop("FINDINGS", None)
    impression = sections.pop("IMPRESSION", None) or sections.pop("CONCLUSION", None) or "-"
    if findings is None:
        # Unlabelled observations and non-standard sections are the findings
        findings = flat(" ".join([narrative] + list(sections.values()))) or "-"
        sections = {}
    return ([("FINDINGS:", findings), ("IMPRESSION:", impression)]
            + [(f"{header}:", body) for header, body in sections.items()])


def split_report(text: str):
    """FINDINGS and IMPRESSION sections of a report (the whole text as findings otherwise)."""
    text = re.sub(r"\*\*", "", text)
    match = re.search(r"FINDINGS:?\s*(.*?)\s*IMPRESSION:?\s*(.*)", text, re.S | re.I)
    return (match.group(1).strip(), match.group(2).strip()) if match else (text.strip(), "")


_REPORT_TITLES = {"Checklist", "Generated Report (MedGemma)", "Ground Truth Report", "VQA Report (EDV)"}
_REPORT_HEADINGS = {"FINDINGS:", "IMPRESSION:", "Visual Proof:", "Supporting Evidence:", "Refuting Evidence:"}


def to_markdown(report: str) -> str:
    """Render a final report (plain text, as logged) for the chat UI."""
    out = []
    for line in report.splitlines():
        if line in _REPORT_TITLES:
            out.append(f"\n### {line}\n")
        elif line in _REPORT_HEADINGS or re.fullmatch(r"[A-Z][A-Z /&]*:", line):
            out.append(f"**{line}**  ")
        elif line.startswith("Annotated image:"):
            continue  # the UI already shows the annotated image
        elif line.startswith("["):
            out.append(f"`{line[:3]}` {line[3:]}  ")  # keep checkboxes aligned and literal
        else:
            out.append(f"{line}  " if line else "")
    return "\n".join(out).strip()


def build_report(trace: FlowTrace, extractor, run: Dict[str, Any], engine) -> Dict[str, Any]:
    """The final report of one MedGemma run as data, shared by the text and PDF versions."""
    output = run["output"]
    present, uncertain = checklist(trace, extractor, output["response"])
    marks = [(label, "X" if label in present else "?" if label in uncertain else " ")
             for label in CHEXPERT_LABELS]
    no_finding = not ((present | uncertain) - {"Support Devices"})
    marks.append(("No Finding", "X" if no_finding else " "))

    findings, impression = split_report(output["response"])
    truth = ground_truth(trace.image_name, trace.image_path)
    if truth:
        trace.section(f"Ground truth lookup] {truth[0]}")  # in the log, not in the report

    proof_lines, supporting, refuting = [], [], []
    for proof in output.get("visual_proof", []):
        name = proof["finding"].capitalize()
        if "boxes" not in proof:
            instead = proof.get("chexagent_boxed_instead")
            proof_lines.append(f"{name}: not localized by CheXagent"
                               + (f" (it boxed {', '.join(instead)} instead)" if instead else ""))
            refuting.append(f"{name}: stated by MedGemma, but CheXagent did not box it under "
                            "this name" + (f" (boxed {', '.join(instead)})" if instead else ""))
            continue
        boxes = " and ".join(f"{b['box_normalized']} ({b['location']})" for b in proof["boxes"])
        p = proof["boxes"][0]["p_present"]
        side = "above" if p is not None and p >= 0.5 else "below"
        proof_lines.append(f"{name}: Box{'es' if len(proof['boxes']) > 1 else ''} at {boxes} "
                           f"with p_present={p} ({side} 0.5)")
        if p is not None:
            # Evidence Engine state: in the log only, not in the report
            ev = engine.assess(ENGINE_NAMES.get(proof["finding"], proof["finding"]), p)
            trace.section(f"Evidence Engine v1.0] {proof['finding']}: p_present={p} -> "
                          f"{ev.state} ({ev.reason})")
        if p is not None and p >= 0.5:
            supporting.append(f"{name}: CheXagent localized it and answers Yes, p_present={p}")
        else:
            refuting.append(f"{name}: CheXagent draws a box when asked to locate it, but answers "
                            f"No to 'Is there {proof['finding']}?', p_present={p}")
    stated = bool(output.get("visual_proof"))
    if not stated:
        proof_lines.append("(no findings stated by MedGemma to verify)")

    return {
        "image_name": trace.image_name,
        "image_path": trace.image_path,
        "annotated_path": output.get("visualization_path"),
        "medgemma_model": run.get("model"),
        "checklist": marks,
        "generated": [("FINDINGS:", findings), ("IMPRESSION:", impression or "-")],
        "truth": [(header, text or "-") for header, text in truth[1]] if truth else None,
        "visual_proof": proof_lines,
        "supporting": [f"- {s}" for s in supporting] or [
            "None: CheXagent supports none of the findings MedGemma stated" if stated
            else "None: MedGemma stated no finding to verify"],
        "refuting": [f"- {r}" for r in refuting] or [
            "None: every finding MedGemma stated was localized by CheXagent with p_present >= 0.5"
            if stated else "None: MedGemma stated no finding to verify"],
    }


def report_text(data: Dict[str, Any]) -> str:
    """Plain-text report, as logged and (via to_markdown) shown in the UI."""
    lines = ["Checklist", "FINDINGS:"] + [f"[{mark}] {label}" for label, mark in data["checklist"]]
    lines += ["", "Generated Report (MedGemma)", ""]
    for header, text in data["generated"]:
        lines += [header, text]
    lines += ["", "Ground Truth Report", ""]
    if data["truth"]:
        for header, text in data["truth"]:
            lines += [header, text]
    else:
        lines.append("Not available (no Indiana or MIMIC-CXR gold report for this image)")
    lines += ["", "", "VQA Report (EDV)", "", "Visual Proof:"] + data["visual_proof"]
    lines += ["", "Supporting Evidence:"] + data["supporting"]
    lines += ["", "Refuting Evidence:"] + data["refuting"]
    if data["annotated_path"]:
        lines += ["", f"Annotated image: {data['annotated_path']}"]
    return "\n".join(lines)


def final_report(
    trace: FlowTrace, extractor, summary: Optional[str] = None, write_pdf: bool = True
) -> List[Dict[str, Any]]:
    """Log the final report for each MedGemma run of this request, save it as a PDF next to
    the log (unless write_pdf is False), and return {"text", "pdf", "data"} per report."""
    from medrax.evidence import EvidenceEngine
    from medrax.utils.report_pdf import write_report_pdf

    engine = EvidenceEngine()
    reports = []
    for i, run in enumerate(trace.medgemma_runs):
        banner = ["", "#" * 100, f"FINAL REPORT: {trace.image_name}", "#" * 100]
        if "response" not in run["output"]:
            trace.log("\n".join(banner + [f"MedGemma failed: {run['output'].get('error')}"]))
            continue
        data = build_report(trace, extractor, run, engine)
        text = report_text(data)
        trace.log("\n".join(banner + ["", text]))

        pdf = None
        if write_pdf:
            pdf = trace.path.with_name(trace.path.stem + (f"_{i + 1}" if i else "") + ".pdf")
            try:
                write_report_pdf(data, pdf)
                trace.log(f"\nPDF report: {pdf}")
            except Exception as e:
                trace.log(f"\nPDF report failed: {e!r}")
                pdf = None
        reports.append({"text": text, "pdf": pdf, "data": data})

    if summary:
        trace.log("\n" + "#" * 100)
        trace.log("DIRECTOR SUMMARY SHOWN IN THE UI")
        trace.log("#" * 100)
        trace.log(summary)
    return reports
