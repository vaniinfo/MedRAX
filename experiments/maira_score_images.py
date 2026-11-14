"""MAIRA-2 presence score (maira_score_design.md; same prompt, sentences and settings as the NIH
readings) for each comparison's full-resolution current image, six findings.

    python experiments/maira_score_images.py PDF --images-dir DIR --out OUT.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from pypdf import PdfReader  # noqa: E402

from comparison_edv_pdf import parse_page, row_image  # noqa: E402
from maira_score_nih import SENTENCES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    from vindr_grounding_amendment import MAIRA_REVISION

    kw = dict(cache_dir="C:/Users/hari/model-weights", trust_remote_code=True, revision=MAIRA_REVISION)
    model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", device_map="cuda", torch_dtype=torch.bfloat16,
                                                 **kw).eval()
    proc = AutoProcessor.from_pretrained("microsoft/maira-2", **kw)
    tok = proc.tokenizer
    encoded = {s: tok(s, add_special_tokens=False)["input_ids"] for pair in SENTENCES.values() for s in pair}
    out = {}
    for p in PdfReader(args.pdf).pages:
        page = parse_page(p.extract_text())
        image = Image.open(row_image(args.images_dir, page["row"])).convert("RGB")
        inp = {k: v.cuda() for k, v in proc.format_and_preprocess_reporting_input(
            current_frontal=image, current_lateral=None, prior_frontal=None, indication=None, technique=None,
            comparison=None, prior_report=None, get_grounding=False, return_tensors="pt").items()}
        with torch.no_grad():
            o = model(**inp, use_cache=True)
            cache, n_prefix = o.past_key_values, o.past_key_values[0][0].shape[-2]
            first = torch.log_softmax(o.logits[0, -1].float(), -1)
            lp = {}
            for s, c in encoded.items():
                o2 = model(input_ids=torch.tensor([c]).cuda(), past_key_values=cache, use_cache=True,
                           attention_mask=torch.ones(1, n_prefix + len(c), dtype=torch.long).cuda())
                l2 = torch.log_softmax(o2.logits[0].float(), -1)
                lp[s] = first[c[0]].item() + sum(l2[t, c[t + 1]].item() for t in range(len(c) - 1))
        out[page["title"]] = {f: 1 / (1 + math.exp(-(lp[pos] - lp[neg]))) for f, (pos, neg) in SENTENCES.items()}
        print(page["title"], {f: round(v, 3) for f, v in out[page["title"]].items()}, flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
