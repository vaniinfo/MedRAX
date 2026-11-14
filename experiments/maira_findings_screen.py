"""MAIRA-2 column for the MedGemma / CheXagent / ground-truth table (screen_three_way.py).

MAIRA-2 cannot answer "Is there X?", so it is screened through its own report: for each page it
writes a FINDINGS section from the full-resolution current and prior frontal images (no prior
report, indication, technique or comparison; 8-bit, greedy, as in VinDr Part B). The 13 screened
conditions are then looked up in that report with the same quote-checked lookup used for the
ground truth. MAIRA-2 never sees the generated report or the ground truth.

    python experiments/maira_findings_screen.py THREE_WAY.json PDF --images-dir DIR
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


def prior_reports(pages, images_dir):
    """{title: ground-truth report of the row whose current image is this row's prior image}, for
    rows whose prior study is in the set (images matched by correlation >= 0.99)."""
    import numpy as np
    from PIL import Image

    def vec(path):
        a = np.asarray(Image.open(path).convert("L").resize((64, 64)), float)
        a -= a.mean()
        return a / (np.linalg.norm(a) + 1e-9)
    current = {page["title"]: Path(row_image(images_dir, page["row"])) for page in pages}
    vectors = {t: vec(p) for t, p in current.items()}
    truth = {page["title"]: page["truth_text"] for page in pages}
    out = {}
    for t, path in current.items():
        prior = vec(path.with_name("prior.jpg"))
        score, source = max((float((prior * v).sum()), s) for s, v in vectors.items() if s != t)
        if score >= 0.99:
            out[t] = {"from": source, "report": truth[source]}
    return out


def maira_reports(pdf, images_dir, precision="8bit", use_prior_reports=False):
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    from vindr_grounding_amendment import MAIRA_REVISION

    kw = dict(cache_dir="C:/Users/hari/model-weights", trust_remote_code=True, revision=MAIRA_REVISION)
    load = (dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True)) if precision == "8bit"
            else dict(torch_dtype=torch.bfloat16))
    model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", device_map="cuda", **load, **kw).eval()
    proc = AutoProcessor.from_pretrained("microsoft/maira-2", **kw)
    pages = [parse_page(p.extract_text()) for p in PdfReader(pdf).pages]
    priors = prior_reports(pages, images_dir) if use_prior_reports else {}
    print(f"prior reports used: {({t: v['from'] for t, v in priors.items()})}", flush=True)
    reports = {}
    for page in pages:
        current = Path(row_image(images_dir, page["row"]))
        inputs = proc.format_and_preprocess_reporting_input(
            current_frontal=Image.open(current).convert("RGB"), current_lateral=None,
            prior_frontal=Image.open(current.with_name("prior.jpg")).convert("RGB"), indication=None,
            technique=None, comparison=None, prior_report=priors.get(page["title"], {}).get("report"),
            get_grounding=False, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=300, use_cache=True, do_sample=False)
        reports[page["title"]] = proc.decode(ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        print(f"{page['title']}: {reports[page['title']]}", flush=True)
    del model
    torch.cuda.empty_cache()
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("three_way_json")
    ap.add_argument("pdf")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--ollama-model", default="qwen3.8-edv:latest")
    ap.add_argument("--precision", choices=["8bit", "bf16"], default="8bit")
    ap.add_argument("--prior-reports", action="store_true",
                    help="give MAIRA-2 the prior study's report where the prior study is in the set "
                         "(its ground-truth report)")
    args = ap.parse_args()

    base = Path(args.three_way_json)
    tag = "" if args.precision == "8bit" and not args.prior_reports else \
        f"_{args.precision}" + ("_prior" if args.prior_reports else "")
    cache = base.with_name(base.stem.replace("_three_way", "") + f"_maira_reports{tag}.json")
    if cache.exists():
        reports = json.loads(cache.read_text(encoding="utf-8"))
    else:
        reports = maira_reports(args.pdf, args.images_dir, args.precision, args.prior_reports)
        cache.write_text(json.dumps(reports, indent=1), encoding="utf-8")

    extractor = make_extractor(args.ollama_model)
    record = json.loads(base.read_text(encoding="utf-8"))
    maira = {t: ground_truth_locations(extractor, text, CONDITIONS) for t, text in reports.items()}
    for r in record:
        where = maira[r["page"]][r["condition"]]
        r["maira"] = "Not stated" if where == NOT_IN_GT else where
        r["maira_report"] = reports[r["page"]]

    out = base.with_name(base.stem.replace("_three_way", f"_four_way{tag}"))
    out.with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    rows = [r for r in record if r["medgemma"] != "Not stated" or r["chexagent_found"] or r["maira"] != "Not stated"
            or r["ground_truth"] != NOT_IN_GT]
    number = lambda title: title.split()[1]  # noqa: E731  "Comparison 8 of 19" -> "8"
    lines = ["# MedGemma vs CheXagent vs MAIRA-2 (each independent) vs ground truth", "",
             "MAIRA-2: conditions its own findings report states (no score). Rows where any column has the condition.",
             "", "| page | condition | MedGemma | CheXagent | MAIRA-2 | ground truth |", "|---|---|---|---|---|---|"]
    lines += [f"| {number(r['page'])} | {r['condition']} | {r['medgemma']} | {r['chexagent']} | {r['maira']} | "
              f"{r['ground_truth']} |" for r in rows]
    lines += ["", "## MAIRA-2 reports", ""] + [f"- **{t}**: {text}" for t, text in reports.items()]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}.md / .json")


if __name__ == "__main__":
    main()
