#!/usr/bin/env python
"""Ask the agent one question from the command line, without the Gradio UI.

Same agent, same tools, same validator, same message shape the UI builds -- it calls
main.initialize_agent rather than assembling a parallel one, so what you see here is
what the UI would have done. Useful for reading a transcript without a browser, for
diffing two runs, and for scripting a batch of questions.

    python scripts/ask.py IMAGE "Does this chest X-ray contain a pulmonary edema?"
    python scripts/ask.py IMAGE "..." --no-image      # tool outputs only, no radiograph
    python scripts/ask.py IMAGE "..." --quiet         # final answer only

The radiograph is sent as base64 by default, exactly as the UI sends it. --no-image
withholds it, which is how to see what the Director concludes from validated evidence
alone.
"""
import argparse
import base64
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("image", help="path to a chest X-ray")
    parser.add_argument("question", help="what to ask about it")
    parser.add_argument("--no-image", action="store_true",
                        help="withhold the radiograph; the Director sees only tool output")
    parser.add_argument("--quiet", action="store_true",
                        help="print the final answer only, not the validation blocks")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if not os.path.isfile(args.image):
        return f"no such image: {args.image}"

    from dotenv import load_dotenv
    load_dotenv()
    import main as medrax

    device = medrax.select_device()
    model_dir = os.getenv("MEDRAX_MODEL_DIR", os.path.expanduser("~/model-weights"))
    selected = [name.strip() for name in os.getenv("MEDRAX_TOOLS", "").split(",") if name.strip()]
    if not selected:
        selected = ["ImageVisualizerTool", "DicomProcessorTool", "ChestXRayClassifierTool",
                    "ChestXRaySegmentationTool", "ChestXRayReportGeneratorTool", "XRayVQATool"]

    openai_kwargs = {}
    if key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = key
    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    agent, _ = medrax.initialize_agent(
        "medrax/docs/system_prompts.txt", tools_to_use=selected, model_dir=model_dir,
        temp_dir="temp", device=device, model=args.model,
        temperature=args.temperature, top_p=0.95, openai_kwargs=openai_kwargs)

    # The message shape interface.py builds: the path for the tools, the image itself
    # for the Director, then the question.
    messages = [{"role": "user", "content": f"image_path: {args.image}"}]
    if not args.no_image:
        with open(args.image, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": args.question}]})

    print(f"\n{'=' * 78}\nQ: {args.question}\n  image: {args.image}"
          f"{'  (withheld from the Director)' if args.no_image else ''}\n{'=' * 78}",
          flush=True)

    start = time.time()
    final = None
    for event in agent.workflow.stream(
            {"messages": messages}, {"configurable": {"thread_id": str(time.time())}}):
        if not isinstance(event, dict):
            continue
        if "execute" in event and not args.quiet:
            for message in event["execute"]["messages"]:
                print(f"\n--- {message.name} ---\n{str(message.content)[:1400]}", flush=True)
        if "process" in event:
            content = event["process"]["messages"][-1].content
            if content:
                final = content

    print(f"\n{'=' * 78}\nDIRECTOR ANSWER  ({time.time() - start:.0f}s)\n{'=' * 78}")
    print(final or "(no answer)")
    print("\nThe per-tool validation and CROSS-TOOL SYNTHESIS blocks are in logs/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
