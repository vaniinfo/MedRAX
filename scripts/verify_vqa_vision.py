#!/usr/bin/env python
"""Verify that the CheXagent VQA tool is actually reading its input image.

CheXagent can fail in a way that produces no error: it loads, generates fluent
clinical text, and ignores the X-ray entirely, returning the same answer for every
input. A single plausible-looking output is NOT evidence that the vision path works.

This script asserts that different images produce different answers. Run it after any
change to the transformers version, the model revision, or the device/dtype.

    python scripts/verify_vqa_vision.py            # exit 0 = healthy, 1 = broken
    MEDRAX_DEVICE=cpu python scripts/verify_vqa_vision.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

FINDINGS = 'Given the indication: "", write a structured findings section for the CXR.'
VIEW = "What is the view of this chest X-ray? Options: (a) PA, (b) AP, (c) LATERAL"
REAL = ["demo/chest/normal1.jpg", "demo/chest/pneumonia1.jpg", "demo/chest/effusion1.png"]


def main() -> int:
    from medrax.tools import XRayVQATool

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    real = [p for p in REAL if os.path.isfile(p)]
    if len(real) < 2:
        print(f"FAIL: need >=2 sample X-rays, found {real}")
        return 1

    os.makedirs("temp", exist_ok=True)
    blank = "temp/_verify_blank.png"
    Image.fromarray(np.zeros((512, 512, 3), np.uint8)).save(blank)

    device = os.getenv("MEDRAX_DEVICE") or ("cuda" if _cuda() else "mps" if _mps() else "cpu")
    print(f"device: {device}")
    tool = XRayVQATool(cache_dir=os.getenv("MEDRAX_MODEL_DIR",
                                           os.path.expanduser("~/model-weights")),
                       device=device)

    def ask(path, prompt, n=60):
        out, _ = tool._run(image_paths=[path], prompt=prompt, max_new_tokens=n)
        return str(out.get("response", out.get("error", ""))).strip()

    failures = []

    # 1. two different real X-rays must not give identical findings
    a, b = ask(real[0], FINDINGS), ask(real[1], FINDINGS)
    print(f"\nfindings[{real[0]}]:\n  {a[:150]}")
    print(f"findings[{real[1]}]:\n  {b[:150]}")
    if a == b:
        failures.append("two different X-rays produced IDENTICAL findings text")

    # 2. a blank image must not give the same answer as a real X-ray
    blank_ans, real_ans = ask(blank, VIEW, 16), ask(real[0], VIEW, 16)
    print(f"\nview[blank]: {blank_ans!r}\nview[{real[0]}]: {real_ans!r}")
    if blank_ans == real_ans:
        failures.append("a blank image and a real X-ray produced the SAME view answer")

    print()
    if failures:
        print("FAIL: the vision path is not working. Answers do not depend on the image.")
        for f in failures:
            print(f"  - {f}")
        print("\nMost likely cause: an incompatible transformers version "
              "(CheXagent-2-3b needs 4.40.x).")
        return 1
    print("PASS: answers vary with the input image; the vision path is live.")
    return 0


def _cuda():
    import torch
    return torch.cuda.is_available()


def _mps():
    import torch
    return torch.backends.mps.is_available()


if __name__ == "__main__":
    sys.exit(main())
