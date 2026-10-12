"""Model runs for model_comparison_pilot_protocol.md (Experiments 1-3; amendments A1-A3).

Raw outputs only (no scoring), one model at a time, each phase saved to
logs/pilot_runs_<time>/<model>.json and skipped if already there (pass --out-dir to resume).

    python experiments/pilot_runs.py PDF --images-dir DIR [--out-dir logs/pilot_runs_<time>]
"""

import argparse
import base64
import io
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from pypdf import PdfReader  # noqa: E402

from chexagent_independent_screen import CONDITIONS  # noqa: E402
from chexagent_report_screen import AREAS  # noqa: E402
from comparison_edv_pdf import parse_page, row_image  # noqa: E402

OLLAMA = "http://localhost:11434"
MEDGEMMA = "medgemma1.5:4b"
LONGITUDINAL = ("The first image is the PRIOR study and the second is the CURRENT study; report on the CURRENT "
                "study and note change from the prior.\n")
BOX_PROMPT = ('Where is the {c}? Output a JSON list of objects, each with a "label" and a 2D bounding box "box_2d" '
              "as [y0, x0, y1, x1], normalized to the range [0, 1] with (0, 0) the top-left and (1, 1) the "
              "bottom-right corner.")


def pages(pdf, images_dir):
    out = []
    for p in PdfReader(pdf).pages:
        page = parse_page(p.extract_text())
        current = Path(row_image(images_dir, page["row"]))
        out.append({"title": page["title"], "current": str(current), "prior": str(current.with_name("prior.jpg"))})
    return out


# ------------------------------------------------------------------ MedGemma 1.5 (Ollama)

def data_url(path):
    from PIL import Image
    image = Image.open(path).convert("RGB")
    image.thumbnail((1024, 1024))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def ollama_chat(messages, max_tokens, logprobs=False):
    body = {"model": MEDGEMMA, "temperature": 0, "max_tokens": max_tokens, "messages": messages}
    if logprobs:
        body.update(logprobs=True, top_logprobs=20)
    request = urllib.request.Request(f"{OLLAMA}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=600))["choices"][0]


def yes_score(choice):
    """A1: P(yes) / (P(yes) + P(no)) over the top-20 first-token alternatives."""
    top = ((choice.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs") or []
    yes = sum(math.exp(t["logprob"]) for t in top if re.match(r"^\s*yes", t["token"], re.I))
    no = sum(math.exp(t["logprob"]) for t in top if re.match(r"^\s*no", t["token"], re.I))
    return yes / (yes + no) if yes + no > 0 else None


def parse_boxes_yxyx(text):
    """[{label, box: [x1, y1, x2, y2] in percent}] from MedGemma's JSON list of box_2d [y0, x0, y1, x1] in 0-1."""
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return None
    boxes = []
    for item in items if isinstance(items, list) else []:
        b = item.get("box_2d") if isinstance(item, dict) else None
        if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            y0, x0, y1, x1 = b
            boxes.append({"label": item.get("label"), "box": [x0 * 100, y0 * 100, x1 * 100, y1 * 100]})
    return boxes


def run_medgemma(rows):
    prompt = (ROOT / "medrax" / "docs" / "medgemma_report_prompt.txt").read_text(encoding="utf-8").strip()
    out = {}
    for r in rows:
        current, prior = data_url(r["current"]), data_url(r["prior"])
        report = ollama_chat([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": prior}}, {"type": "image_url", "image_url": {"url": current}},
            {"type": "text", "text": LONGITUDINAL + prompt}]}], 450)["message"]["content"]
        questions, locations = {}, {}
        for c in CONDITIONS:
            choice = ollama_chat([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": current}},
                {"type": "text", "text": f"Is there {c}?"}]}], 16, logprobs=True)
            questions[c] = {"answer": choice["message"]["content"], "score": yes_score(choice)}
            text = ollama_chat([{"role": "system", "content": "You are a helpful radiology assistant."},
                                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": current}},
                                                             {"type": "text", "text": BOX_PROMPT.format(c=c)}]}],
                               300)["message"]["content"]
            locations[c] = {"raw": text, "boxes": parse_boxes_yxyx(text)}
        out[r["title"]] = {"report": report, "questions": questions, "locations": locations}
        print(f"MedGemma {r['title']}: {sum(q['answer'].lower().startswith('yes') for q in questions.values())} yes",
              flush=True)
    # free the GPU for the next model
    urllib.request.urlopen(urllib.request.Request(f"{OLLAMA}/api/generate", data=json.dumps(
        {"model": MEDGEMMA, "keep_alive": 0}).encode(), headers={"Content-Type": "application/json"}), timeout=60)
    return out


# ------------------------------------------------------------------ CheXagent

def run_chexagent(rows):
    import torch

    from medrax.tools.xray_vqa import XRayVQATool

    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    out = {}
    for r in rows:
        report = " ".join(tool._generate_response(
            [r["current"]], f'Please provide a detailed description of "{a}" in the chest X-ray', 200)[0].strip()
            for a in AREAS)
        questions, locations = {}, {}
        for c in CONDITIONS:
            text, yes_no = tool._generate_response([r["current"]], f"Is there {c}?", 8)
            questions[c] = {"answer": text, "score": yes_no["p_yes"] if yes_no else None}
            raw, _ = tool._generate_response([r["current"]], f"Please locate the following phrase: {c}", 128)
            _, boxes = tool._parse_grounding(raw)  # all boxes, as in VinDr Part B
            locations[c] = {"raw": raw, "boxes": [{"label": b["phrase"], "box": b["box_normalized"]} for b in boxes]}
        out[r["title"]] = {"report": report, "questions": questions, "locations": locations}
        print(f"CheXagent {r['title']} done", flush=True)
    del tool
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ MAIRA-2 (bf16)

def run_maira(rows):
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    from vindr_grounding_amendment import MAIRA_REVISION

    kw = dict(cache_dir="C:/Users/hari/model-weights", trust_remote_code=True, revision=MAIRA_REVISION)
    model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", device_map="cuda", torch_dtype=torch.bfloat16,
                                                 **kw).eval()
    proc = AutoProcessor.from_pretrained("microsoft/maira-2", **kw)

    def generate(inputs, n):
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=n, use_cache=True, do_sample=False)
        return proc.decode(ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    out = {}
    for r in rows:
        current = Image.open(r["current"]).convert("RGB")
        report = generate(proc.format_and_preprocess_reporting_input(
            current_frontal=current, current_lateral=None, prior_frontal=Image.open(r["prior"]).convert("RGB"),
            indication=None, technique=None, comparison=None, prior_report=None, get_grounding=False,
            return_tensors="pt"), 300)
        locations = {}
        for c in CONDITIONS:
            text = generate(proc.format_and_preprocess_phrase_grounding_input(
                frontal_image=current, phrase=c.capitalize(), return_tensors="pt"), 150)
            try:  # as in VinDr Part B (vindr_grounding_amendment.maira_locator)
                raw = [b for _, bxs in proc.convert_output_to_plaintext_or_grounded_sequence(text) or []
                       for b in bxs or []]
            except AssertionError:
                raw = [tuple((int(v) + 0.5) / 100 for v in m)
                       for m in re.findall(r"<box><x(\d+)><y(\d+)><x(\d+)><y(\d+)></box>", text)]
            boxes = [[v * 100 for v in proc.adjust_box_for_original_image_size(
                b, width=current.size[0], height=current.size[1])] for b in raw]
            locations[c] = {"raw": text, "boxes": [{"label": c, "box": b} for b in boxes]}
        out[r["title"]] = {"report": report, "locations": locations}
        print(f"MAIRA-2 {r['title']} done", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "logs" / f"pilot_runs_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = pages(args.pdf, args.images_dir)
    (out_dir / "pages.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for name, run in (("medgemma15", run_medgemma), ("chexagent", run_chexagent), ("maira2", run_maira)):
        path = out_dir / f"{name}.json"
        if path.exists():
            print(f"{name}: already done", flush=True)
            continue
        start = time.time()
        path.write_text(json.dumps(run(rows), indent=1), encoding="utf-8")
        print(f"{name}: done in {(time.time() - start) / 60:.0f} min -> {path}", flush=True)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
