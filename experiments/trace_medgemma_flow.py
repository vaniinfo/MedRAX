"""Write the same flow trace the UI writes, for images given on the command line.

Runs MedGemmaTool._run (MedGemma opinion -> finding extraction by the director's model ->
CheXagent locate + p_present) inside a flow trace (medrax/utils/flow_trace.py), then adds
the final report. One log per image in logs/flow_traces/<image>_<time>.log.

    python experiments/trace_medgemma_flow.py IMAGE [IMAGE ...] [--director qwen|gpt-4o]
        [--ollama-model qwen3.5:latest] [--question "..."]
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medrax.tools.medgemma import MedGemmaTool  # noqa: E402
from medrax.tools.xray_vqa import XRayVQATool  # noqa: E402
from medrax.utils import flow_trace  # noqa: E402

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--director", default="qwen", choices=["qwen", "gpt-4o"])
    ap.add_argument("--ollama-model", default="qwen3.5:latest")
    ap.add_argument("--question", default="(trace script: no user question)",
                    help="only recorded; MedGemma always gets its fixed report prompt")
    args = ap.parse_args()

    if args.director == "gpt-4o":
        extractor = ChatOpenAI(model="gpt-4o", temperature=0.7, top_p=0.95)
    else:
        extractor = ChatOpenAI(model=args.ollama_model, temperature=0.7, top_p=0.95,
                               base_url="http://localhost:11434/v1", api_key="ollama")
    chexagent = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    tool = MedGemmaTool(locator=chexagent, extractor=extractor)

    for image in args.images:
        with flow_trace.FlowTrace(Path(image).name, image_path=image) as trace:
            trace.log(f"Flow trace: {trace.image_name}   (experiments/trace_medgemma_flow.py, no director)")
            trace.log(f"image passed to the tools: {image}")
            tool._run(image, args.question)
            reports = flow_trace.final_report(trace, extractor)
        print(f"{image}: {trace.path}" + "".join(f", {r['pdf']}" for r in reports if r["pdf"]))


if __name__ == "__main__":
    main()
