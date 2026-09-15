#!/usr/bin/env python
"""Which tools has the Director actually called, across all logged sessions?

Loading a tool is not using it. GPT-4o picks tools per question from their description
strings, so a tool can be loaded every run and never chosen -- LlavaMedTool's own
description tells the model it "may not be as reliable for detailed chest X-ray
analysis", which discourages selection.

    python scripts/tool_usage.py
"""
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

TOOL_CLASSES = {
    "chest_xray_expert": "XRayVQATool (CheXagent)",
    "chest_xray_classifier": "ChestXRayClassifierTool (DenseNet-121)",
    "chest_xray_report_generator": "ChestXRayReportGeneratorTool (SwinV2 x2)",
    "chest_xray_segmentation": "ChestXRaySegmentationTool (PSPNet)",
    "image_visualizer": "ImageVisualizerTool",
    "dicom_processor": "DicomProcessorTool",
    "llava_med_qa": "LlavaMedTool (LLaVA-Med 7B)",
    "xray_phrase_grounding": "XRayPhraseGroundingTool (MAIRA-2)",
    "chest_xray_generator": "ChestXRayGeneratorTool (RoentGen)",
}

files = sorted(glob.glob("logs/tool_calls_*.json"))
if not files:
    print("No logs/tool_calls_*.json found. Ask the agent something first.")
    sys.exit(0)

counts, errors, validated = collections.Counter(), collections.Counter(), collections.Counter()
for path in files:
    try:
        entries = json.load(open(path))
    except Exception:
        continue
    for entry in entries:
        name = entry.get("name", "?")
        counts[name] += 1
        content = str(entry.get("content", ""))
        if "'error'" in content or "error_details" in content:
            errors[name] += 1
        if entry.get("validation"):
            validated[name] += 1

print(f"{len(files)} logged turns\n")
print(f"{'tool':42s} {'calls':>6s} {'errors':>7s} {'validated':>10s}")
for name, n in counts.most_common():
    label = TOOL_CLASSES.get(name, name)
    print(f"{label:42s} {n:6d} {errors[name]:7d} {validated[name]:10d}")

never = [label for key, label in TOOL_CLASSES.items() if key not in counts]
if never:
    print("\nNever called:")
    for label in never:
        print(f"  - {label}")
    print("\nA tool is never called either because it failed to load (check the startup\n"
          "line) or because its description did not persuade the orchestrator to pick it.\n"
          "Name it directly in your question to force it.")
