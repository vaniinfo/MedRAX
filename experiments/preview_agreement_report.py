"""PREVIEW: the report with agreement-based confidence levels instead of the 0.5 cut-off.

For one image, runs the normal pipeline (MedGemma report -> findings -> CheXagent proof ->
checklist), then replaces Supporting / Refuting Evidence with a level per finding MedGemma
reports, using the Evidence Engine's per-condition cut-offs (option 3):

  High    MedGemma reports it and CheXagent gives a strong yes
  Medium  MedGemma reports it and CheXagent is undecided (or the finding has no validated
          cut-off, or CheXagent could not box it)
  Low     MedGemma reports it but CheXagent gives a strong no

Each line also says whether CheXagent's box agrees with the side / zone MedGemma describes.
CheXagent checks only the findings MedGemma reports. Nothing in the UI or the normal report
changes; output goes to temp/.

    python experiments/preview_agreement_report.py [IMAGE]
"""

import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrax.evidence import EvidenceEngine  # noqa: E402
from medrax.tools.medgemma import MedGemmaTool  # noqa: E402
from medrax.tools.xray_vqa import XRayVQATool  # noqa: E402
from medrax.utils import flow_trace  # noqa: E402
from medrax.utils.report_pdf import write_report_pdf  # noqa: E402

DEFAULT_IMAGE = ROOT / "demo" / "00fe920d-adff507b-b407525b-7c4ef10f-54bf740b.png"
GREEN, AMBER, RED = (0, 110, 50), (175, 105, 0), (170, 30, 30)


def cutoffs(engine, finding):
    """(lowest score called a strong yes, highest score called a strong no) for a finding."""
    ps = np.round(np.arange(0.0005, 1.0, 0.0005), 4)
    states = [engine.assess(finding, float(p)).state for p in ps]
    yes = [p for p, s in zip(ps, states) if s == "SUPPORTING"]
    no = [p for p, s in zip(ps, states) if s == "REFUTING"]
    return (min(yes) if yes else None), (max(no) if no else None)


def described_location(report, finding):
    """Side and zone MedGemma gives for a finding, from the sentences that mention it."""
    head = finding.split()[-1]
    sentences = [s.lower() for s in re.split(r"(?<=[.\n])\s*", report) if head in s.lower()]
    text = " ".join(sentences)
    side = ("both" if re.search(r"bilateral|both|bibasilar|bibasal", text) else
            "right" if "right" in text else "left" if "left" in text else None)
    zone = ("lower" if re.search(r"lower|base|basal|basilar", text) else
            "upper" if re.search(r"upper|apex|apical", text) else
            "middle" if re.search(r"\bmid|middle", text) else None)
    return side, zone


def location_note(report, proof):
    side, zone = described_location(report, proof["finding"])
    if side is None and zone is None:
        return "MedGemma gives no location"
    locations = [b["location"] for b in proof["boxes"]]
    box_sides = {"both" if l.startswith("both") else "right" if "right" in l else "left" for l in locations}
    side_ok = side is None or (box_sides >= {"right", "left"} or "both" in box_sides
                               if side == "both" else side in box_sides or "both" in box_sides)
    zone_ok = zone is None or any(zone in l for l in locations)
    said = " ".join(w for w in (side, zone) if w)
    where = "; ".join(locations)
    if side_ok and zone_ok:
        return f"location agrees with MedGemma ({said}): CheXagent boxes {where}"
    return f"location differs: MedGemma says {said}, CheXagent boxes {where}"


def level(engine, finding, p):
    """(level, reason) for a finding MedGemma reports, from CheXagent's score."""
    ev = engine.assess(finding, p)
    yes_cut, no_cut = cutoffs(engine, finding) if ev.reason != "FINDING_NOT_VALIDATED" else (None, None)
    if ev.state == "SUPPORTING" or (ev.reason == "OUTSIDE_SUPPORT" and p >= 0.5):
        return "High", f"CheXagent gives a strong yes (p_present {p}; strong yes from {yes_cut})"
    if ev.state == "REFUTING" or ev.reason == "OUTSIDE_SUPPORT":
        return "Low", f"CheXagent gives a strong no (p_present {p}; strong no below {no_cut})"
    if ev.reason == "FINDING_NOT_VALIDATED":
        return "Medium", f"CheXagent's score has no validated cut-off for {finding} (p_present {p})"
    cut = f"strong yes needs {yes_cut}" + (f", strong no below {no_cut}" if no_cut else "")
    return "Medium", f"CheXagent is undecided (p_present {p}; {cut})"


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
    report = output["response"]

    names = {"effusion": "pleural effusion", "edema": "pulmonary edema"}
    levels = {"High": [], "Medium": [], "Low": []}
    proof_lines = []
    for proof in output.get("visual_proof", []):
        finding = names.get(proof["finding"], proof["finding"])
        name = proof["finding"].capitalize()
        if "boxes" not in proof:
            proof_lines.append(f"{name}: not localized by CheXagent")
            levels["Medium"].append(f"- {name}: MedGemma reports it; CheXagent could not box it")
            continue
        p = proof["boxes"][0]["p_present"]
        boxes = " and ".join(f"{b['box_normalized']} ({b['location']})" for b in proof["boxes"])
        proof_lines.append(f"{name}: Box{'es' if len(proof['boxes']) > 1 else ''} at {boxes}, p_present={p}")
        lvl, why = level(engine, finding, p)
        levels[lvl].append(f"- {name}: MedGemma reports it; {why}; {location_note(report, proof)}")

    data["visual_proof"] = proof_lines or ["(no findings stated by MedGemma to verify)"]
    data["evidence_sections"] = [
        ("High: MedGemma and CheXagent agree", GREEN, levels["High"] or ["None"]),
        ("Medium: MedGemma's finding, not confirmed by CheXagent", AMBER, levels["Medium"] or ["None"]),
        ("Low: CheXagent disputes MedGemma", RED, levels["Low"] or ["None"]),
    ]
    out = ROOT / "temp" / f"preview_agreement_{image.stem[:8]}.pdf"
    write_report_pdf(data, out)

    print("\nVisual Proof:")
    print("\n".join(data["visual_proof"]))
    for heading, _, lines in data["evidence_sections"]:
        print(f"\n{heading}:")
        print("\n".join(lines))
    print(f"\nPreview PDF: {out}")


if __name__ == "__main__":
    main()
