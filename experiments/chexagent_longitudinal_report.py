"""CheXagent's own longitudinal report (prior + current image) for each comparison, read for the
13 conditions with the same quote-checked lookup as the ground truth.

Prompt (chosen on page 8 between the area-by-area prompts and this instruction, because only
this one described change from the prior): see PROMPT. Note: two images plus the prompt are
about 2,061 input tokens, slightly over CheXagent's 2,048 design length.

    python experiments/chexagent_longitudinal_report.py PDF --images-dir DIR --out OUT.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from pypdf import PdfReader  # noqa: E402

from chexagent_independent_screen import CONDITIONS  # noqa: E402
from comparison_edv_pdf import NOT_IN_GT, ground_truth_locations, make_extractor, parse_page, row_image  # noqa: E402

PROMPT = ("The first image is the prior chest X-ray and the second is the current chest X-ray. Write the FINDINGS "
          "section of a radiology report for the current chest X-ray, and describe any change compared with the prior.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ollama-model", default="qwen3.8-edv:latest")
    args = ap.parse_args()
    out = Path(args.out)
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    if not data.get("reports"):
        import torch

        from medrax.tools.xray_vqa import XRayVQATool

        tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
        reports = {}
        for p in PdfReader(args.pdf).pages:
            page = parse_page(p.extract_text())
            current = Path(row_image(args.images_dir, page["row"]))
            reports[page["title"]] = tool._generate_response([str(current.with_name("prior.jpg")), str(current)],
                                                             PROMPT, 300)[0].strip()
            print(f"{page['title']}: {reports[page['title']][:120]}", flush=True)
        data = {"prompt": PROMPT, "reports": reports}
        out.write_text(json.dumps(data, indent=1), encoding="utf-8")
        del tool
        torch.cuda.empty_cache()

    extractor = make_extractor(args.ollama_model)
    labels = {}
    for title, text in data["reports"].items():
        found = ground_truth_locations(extractor, text, CONDITIONS)
        labels[title] = {c: ("Not stated" if v == NOT_IN_GT else v) for c, v in found.items()}
        print(f"{title}: {[c for c, v in labels[title].items() if v != 'Not stated']}", flush=True)
    data["labels"] = labels
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
