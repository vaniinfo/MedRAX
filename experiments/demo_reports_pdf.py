"""One PDF with the final report for every image in demo/ (and demo/chest/).

Each image goes through the same pipeline as the UI, without the director's summary:
MedGemma report -> findings listed by the extractor -> CheXagent visual proof -> checklist.
DICOMs are converted to PNG first, as in the UI. A flow trace log is written per image in
logs/flow_traces/, and all reports go into one PDF with a clickable index.

Default layout: one page per image with the High / Medium / Low evidence table for the
findings MedGemma reports (experiments/preview_onepage_report.py). --layout full gives the longer
report with Supporting / Refuting Evidence.

    python experiments/demo_reports_pdf.py [--director gpt-4o|qwen] [--ollama-model qwen3.5:latest]
        [--layout onepage|full]
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from medrax.evidence import EvidenceEngine  # noqa: E402
from medrax.tools.dicom import DicomProcessorTool  # noqa: E402
from medrax.tools.medgemma import MedGemmaTool  # noqa: E402
from medrax.tools.xray_vqa import XRayVQATool  # noqa: E402
from medrax.utils import flow_trace  # noqa: E402
from medrax.utils.report_pdf import write_reports_pdf  # noqa: E402
from preview_onepage_report import evidence_rows, write_onepages  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".dcm"}


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--director", default="gpt-4o", choices=["gpt-4o", "qwen"],
                    help="model that lists MedGemma's findings and fills the checklist")
    ap.add_argument("--ollama-model", default="qwen3.5:latest")
    ap.add_argument("--out", default=None, help="PDF path (default logs/demo_reports_<time>.pdf)")
    ap.add_argument("--layout", default="onepage", choices=["onepage", "full"])
    args = ap.parse_args()
    engine = EvidenceEngine()

    if args.director == "gpt-4o":
        extractor = ChatOpenAI(model="gpt-4o", temperature=0.7, top_p=0.95)
    else:
        extractor = ChatOpenAI(model=args.ollama_model, temperature=0.7, top_p=0.95,
                               base_url="http://localhost:11434/v1", api_key="ollama")
    chexagent = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    tool = MedGemmaTool(locator=chexagent, extractor=extractor)
    dicom = DicomProcessorTool(temp_dir="temp")

    images = sorted(p for p in (ROOT / "demo").rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    # demo/<dicom_id>.png are small screenshots of the notebook's plots; use the original
    # demo/<dicom_id>.jpg instead where it exists
    originals = {p.stem for p in images if p.suffix.lower() == ".jpg"}
    images = [p for p in images if not (p.suffix.lower() == ".png" and p.stem in originals)]
    reports, failed = [], []
    for n, image in enumerate(images, 1):
        path = image
        if image.suffix.lower() == ".dcm":
            converted, _ = dicom._run(str(image))
            if "image_path" not in converted:
                failed.append((image.name, converted.get("error", "DICOM conversion failed")))
                continue
            path = Path(converted["image_path"])
        with flow_trace.FlowTrace(image.name, image_path=str(path)) as trace:
            trace.log(f"Flow trace: {image.name}   (experiments/demo_reports_pdf.py, no director)")
            trace.log(f"image passed to the tools: {path}")
            output, _ = tool._run(str(path), "(batch report: no user question)")
            done = flow_trace.final_report(trace, extractor, write_pdf=False)
        if not done:
            failed.append((image.name, output.get("error", "no report")))
        elif args.layout == "full":
            reports.append(done[0]["data"])
        else:
            rows = evidence_rows(engine, output["response"], output)
            # CheXagent checks only the findings MedGemma reports (no independent second read)
            reports.append((done[0]["data"], rows))
        print(f"[{n}/{len(images)}] {image.name}: {'ok' if done else 'FAILED'}", flush=True)

    out = Path(args.out) if args.out else ROOT / "logs" / f"demo_reports_{time.strftime('%Y%m%d-%H%M%S')}.pdf"
    if args.layout == "full":
        write_reports_pdf(reports, out, title="Chest X-ray Reports: demo images")
    else:
        write_onepages(reports, out, title="Chest X-ray Reports: demo images")
    print(f"\n{len(reports)} reports written to {out}")
    for name, error in failed:
        print(f"  failed: {name}: {error}")


if __name__ == "__main__":
    main()
