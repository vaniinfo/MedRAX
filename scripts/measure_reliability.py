"""Per-tool, per-finding reliability. No LLM, no API cost -- local inference only.

Every claim this branch makes about weighting tools rests on numbers that were
guessed: the 0.40-0.60 dead zone came from eyeballing four pneumothorax images.
This measures what each tool is actually worth for each finding, so those numbers
can be derived instead.

Per image:
  classifier        one call -> 18 probabilities
  report generator  one call -> text, stance parsed per finding
  CheXagent         one call PER finding -> Yes/No + P(yes)
Results are written incrementally so the run can be interrupted and resumed.
"""
import csv, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from medrax.tools import XRayVQATool, ChestXRayClassifierTool, ChestXRayReportGeneratorTool
from medrax.agent import EvidenceValidator


def pick_device() -> str:
    """cuda on Colab or a GPU box, mps on Apple Silicon, cpu otherwise."""
    import torch
    if os.getenv("MEDRAX_DEVICE"):
        return os.environ["MEDRAX_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


DEVICE = pick_device()
CACHE = os.getenv("MEDRAX_MODEL_DIR", os.path.expanduser("~/model-weights"))
print(f"device: {DEVICE} | weights: {CACHE}", flush=True)

DATA = "data/indiana_eval"
OUT = "reliability.json"

# finding -> (MeSH substrings proving presence, classifier key)
FINDINGS = {
    "cardiomegaly":     (["cardiomegaly"], "Cardiomegaly"),
    "pleural effusion": (["pleural effusion"], "Effusion"),
    # pneumothorax omitted: the Open-i query failed and this set has only 3 positives,
    # which cannot support a sensitivity estimate. Add it once the API allows a top-up.
    "pulmonary edema":  (["pulmonary edema"], "Edema"),
    # 63 positives in this set -- better populated than edema, and it was missing
    "atelectasis":      (["atelectasis"], "Atelectasis"),
}

rows = list(csv.DictReader(open(f"{DATA}/GROUND_TRUTH.csv")))
done = {}
if os.path.exists(OUT):
    done = {r["file"]: r for r in json.load(open(OUT))}
    print(f"resuming: {len(done)} images already measured", flush=True)

vqa = XRayVQATool(cache_dir=CACHE, device=DEVICE)
clf = ChestXRayClassifierTool(device=DEVICE)
rep = ChestXRayReportGeneratorTool(cache_dir=CACHE, device=DEVICE)

results = list(done.values())
t0, n = time.time(), 0
for row in rows:
    if row["file"] in done:
        continue
    img = f"{DATA}/{row['file']}"
    if not os.path.isfile(img):
        continue
    entry = {"file": row["file"], "problems": row["problems"], "findings": {}}
    try:
        probs, _ = clf._run(img)
        report, _ = rep._run(img)
        report = str(report)
    except Exception as exc:
        print(f"  ! {row['file']}: {exc}", flush=True)
        continue
    for finding, (keys, clf_key) in FINDINGS.items():
        truth = any(k in row["problems"].lower() for k in keys)
        try:
            out, _ = vqa._run(image_paths=[img], max_new_tokens=8,
                              prompt=f"Does this chest X-ray contain a {finding}?")
            p_yes, answer = out.get("confidence"), str(out.get("response", "")).strip()
        except Exception:
            p_yes, answer = None, ""
        stance = EvidenceValidator._text_stance(report, finding)
        entry["findings"][finding] = {
            "truth": truth,
            "chexagent_answer": answer,
            "chexagent_p": p_yes,
            "classifier_p": float(probs.get(clf_key)) if probs.get(clf_key) is not None else None,
            "report_stance": (stance or {}).get("stance"),
        }
    results.append(entry)
    n += 1
    if n % 10 == 0:
        el = time.time() - t0
        remaining = len([r for r in rows if r["file"] not in done]) - n
        print(f"{n} done ({len(results)} total)  {el:.0f}s, ~{el/n*remaining:.0f}s left", flush=True)
        json.dump(results, open(OUT, "w"), indent=1)
json.dump(results, open(OUT, "w"), indent=1)
print(f"DONE {len(results)} images in {time.time()-t0:.0f}s -> {OUT}", flush=True)
