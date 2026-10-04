"""Find cases where MedGemma and CheXagent agree / disagree with the ground truth.

For the 55 MIMIC demo images (demo/<dicom_id>.jpg) and the six validated conditions:
  - truth: the gold report (demo/medgemma_vs_gold.csv), labelled present / absent / uncertain;
  - MedGemma: its report from the latest flow trace log of the image, labelled the same way
    (present or uncertain = MedGemma says yes);
  - CheXagent: p_present for "Is there {condition}?", judged with the Evidence Engine
    (option 3): strong yes / strong no / undecided; the 0.5 rule is listed too.
Labelling uses the local Qwen model through Ollama: MIMIC reports must not be sent to online
services (PhysioNet data use agreement).

    python experiments/agreement_cases.py      # writes temp/agreement_cases.md
"""

import csv
import json
import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrax.evidence import EvidenceEngine  # noqa: E402
from medrax.tools.xray_vqa import XRayVQATool  # noqa: E402
from medrax.utils.flow_trace import mimic_sections  # noqa: E402

CONDITIONS = ["cardiomegaly", "pleural effusion", "pneumothorax", "consolidation",
              "pulmonary edema", "atelectasis"]
OUT = ROOT / "temp" / "agreement_cases.md"
CACHE = ROOT / "temp" / "agreement_cases.json"

LABEL_PROMPT = (
    "Below is a chest X-ray report. For each of these conditions say whether the report states "
    "it is present: " + ", ".join(CONDITIONS) + ".\n"
    "Answer \"present\" if the report says it is seen or likely, \"uncertain\" if the report calls "
    "it possible, questionable or cannot exclude it, and \"absent\" if the report says it is not "
    "present or does not mention it. Pulmonary vascular congestion alone is not pulmonary edema. "
    "Output JSON only, one key per condition, e.g. {\"cardiomegaly\": \"absent\", ...}.\n\nReport:\n"
)


def label(llm, report):
    reply = llm.invoke([HumanMessage(content=LABEL_PROMPT + report)]).content
    match = re.search(r"\{.*\}", reply, re.S)
    labels = json.loads(match.group(0)) if match else {}
    return {c: str(labels.get(c, "absent")).lower() for c in CONDITIONS}


def medgemma_report(dicom_id):
    """MedGemma's report from the latest flow trace of this image."""
    logs = sorted((ROOT / "logs" / "flow_traces").glob(f"{dicom_id}_*.log"))
    for log in reversed(logs):
        text = log.read_text(encoding="utf-8")
        match = re.search(r"\[1 MedGemma report\].*?output \(raw\):\n((?:    \|.*\n)+)", text, re.S)
        if match:
            return "\n".join(l[6:] for l in match.group(1).splitlines())
    return None


def main():
    gold = {r["dicom_id"]: r for r in csv.DictReader(open(ROOT / "demo" / "medgemma_vs_gold.csv", encoding="utf-8"))}
    images = sorted(p for p in (ROOT / "demo").glob("*.jpg") if p.stem in gold)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}

    llm = ChatOpenAI(model="qwen3.5:latest", temperature=0, base_url="http://localhost:11434/v1",
                     api_key="ollama", extra_body={"think": False})
    chexagent = None
    engine = EvidenceEngine()
    for n, image in enumerate(images, 1):
        key = image.stem
        if key in cache:
            continue
        sections = mimic_sections(gold[key]["gold_report"])
        truth_text = "\n".join(f"{h} {t}" for h, t in sections)
        med_text = medgemma_report(key)
        if med_text is None:
            print(f"{key}: no MedGemma report in the logs, skipped")
            continue
        if chexagent is None:
            chexagent = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
        scores = {}
        for c in CONDITIONS:
            _, yes_no = chexagent._generate_response([str(image)], f"Is there {c}?", 8)
            scores[c] = yes_no["p_yes"] if yes_no else None
        cache[key] = {"truth": label(llm, truth_text), "medgemma": label(llm, med_text),
                      "scores": scores, "truth_text": truth_text, "medgemma_text": med_text}
        CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        print(f"[{n}/{len(images)}] {key[:8]} done", flush=True)

    rows = []
    for key, d in cache.items():
        for c in CONDITIONS:
            truth = d["truth"][c]
            if truth not in ("present", "absent"):
                continue
            p = d["scores"][c]
            if p is None:
                continue
            ev = engine.assess(c, p)
            strong_yes = ev.state == "SUPPORTING" or (ev.reason == "OUTSIDE_SUPPORT" and p >= 0.5)
            strong_no = ev.state == "REFUTING" or (ev.reason == "OUTSIDE_SUPPORT" and p < 0.5)
            chex = "yes" if strong_yes else "no" if strong_no else "undecided"
            med = "yes" if d["medgemma"][c] in ("present", "uncertain") else "no"
            want = "yes" if truth == "present" else "no"
            rows.append({"image": key, "condition": c, "truth": truth, "medgemma": med,
                         "chex": chex, "p": p, "half": "yes" if p >= 0.5 else "no",
                         "med_ok": med == want, "chex_ok": chex == want, "half_ok": (p >= 0.5) == (want == "yes")})

    groups = {
        "1. Both agree with the truth": lambda r: r["med_ok"] and r["chex_ok"],
        "2. MedGemma agrees, CheXagent does not": lambda r: r["med_ok"] and not r["chex_ok"],
        "3. CheXagent agrees, MedGemma does not": lambda r: not r["med_ok"] and r["chex_ok"],
        "4. Both disagree with the truth": lambda r: not r["med_ok"] and not r["chex_ok"],
    }
    lines = ["# MedGemma and CheXagent vs the gold reports (55 MIMIC images, 6 conditions)", "",
             f"{len(rows)} image-condition pairs with a clear truth (present or absent). CheXagent: "
             "option 3 (strong yes / strong no / undecided; undecided never counts as agreeing).", "",
             "| Situation | Truth present | Truth absent | Total |", "|---|---|---|---|"]
    for name, test in groups.items():
        sel = [r for r in rows if test(r)]
        pos = sum(r["truth"] == "present" for r in sel)
        lines.append(f"| {name} | {pos} | {len(sel) - pos} | {len(sel)} |")
    for name, test in groups.items():
        sel = sorted([r for r in rows if test(r)], key=lambda r: (r["truth"] != "present", r["image"]))
        lines += ["", f"## {name}", "", "| image | condition | truth | MedGemma | CheXagent (option 3) | p_present | 0.5 rule |",
                  "|---|---|---|---|---|---|---|"]
        for r in sel:
            lines.append(f"| {r['image'][:8]} | {r['condition']} | {r['truth']} | {r['medgemma']} | {r['chex']} | "
                         f"{r['p']} | {r['half']} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
