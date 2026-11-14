"""Candidate prior + current pairs from CheXpert Plus (Stanford AIMI research use agreement; not to be shared or
redistributed) for a 20-case radiologist scoring form, from the reports CSV only (no model output).

Rules, fixed before any image is downloaded:
- Same patient, consecutive studies in time (patient_report_date_order k and k + 1), both with a frontal image
  (view1_frontal), both reports with a findings or impression section; train split (PNG_train).
- The current report has an indication (clinical history / history section) and compares with a prior
  (comparison section, or "compar"/"prior"/"previous"/"interval"/"unchanged"/"stable" in its text).
- The current report states at least one of the six conditions (cardiomegaly, pleural effusion, pneumothorax,
  consolidation, pulmonary edema, atelectasis) in a sentence without a negation (no / without / negative for /
  resolved / free of / clear of ...) - a rough keyword screen; the reader model checks it later.
- One pair per patient. Stratified by condition (rarest first, seeded), 30 candidates (20 + spares).

    .venv/Scripts/python.exe -I experiments/chexpert_plus_pairs.py
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

CSV = Path("D:/datasets/chexpert_plus/df_chexpert_plus_240401.csv")
OUT = Path(__file__).resolve().parents[1] / "logs" / "chexpert_plus_candidates.json"
SEED, N = 20261102, 30
TERMS = {
    "cardiomegaly": r"cardiomegaly|(heart|cardiac silhouette|cardiomediastinal silhouette)[^.]{0,30}(enlarged|enlargement)",
    "pleural effusion": r"(pleural )?effusions?",
    "pneumothorax": r"pneumothora(x|ces)",
    "consolidation": r"consolidation",
    "pulmonary edema": r"(pulmonary |interstitial |alveolar )?edema|vascular congestion",
    "atelectasis": r"atelectasis|atelectatic",
}
NEG = re.compile(r"\b(no|without|negative for|resolved|resolution of|free of|clear of|not seen|absence of|"
                 r"no evidence of|rather than|exclude|ruled out)\b", re.I)
COMPARES = re.compile(r"compar|prior|previous|interval|unchanged|stable", re.I)


def text(r, *cols):
    return " ".join(str(r[c]) for c in cols if isinstance(r[c], str) and r[c].strip())


def stated(report):
    """The six conditions named in a sentence that has no negation word."""
    out = set()
    for sentence in re.split(r"(?<=[.;\n])\s*", report):
        if not sentence.strip() or NEG.search(sentence):
            continue
        for c, pattern in TERMS.items():
            if re.search(pattern, sentence, re.I):
                out.add(c)
    return out


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting")
    df = pd.read_csv(CSV, usecols=["path_to_image", "frontal_lateral", "deid_patient_id", "patient_report_date_order",
                                   "section_clinical_history", "section_history", "section_comparison",
                                   "section_findings", "section_impression", "split"])
    df = df[(df["split"] == "train") & (df["frontal_lateral"] == "Frontal")
            & df["path_to_image"].str.endswith("view1_frontal.jpg")]
    df = df[df[["section_findings", "section_impression"]].notna().any(axis=1)]
    by_patient = {p: g.sort_values("patient_report_date_order") for p, g in df.groupby("deid_patient_id")}
    pairs = []
    for p, g in by_patient.items():
        rows = g.to_dict("records")
        for prior, current in zip(rows, rows[1:]):
            if current["patient_report_date_order"] != prior["patient_report_date_order"] + 1:
                continue
            indication = text(current, "section_clinical_history", "section_history")
            body = text(current, "section_findings", "section_impression")
            if not indication or not (isinstance(current["section_comparison"], str) or COMPARES.search(body)):
                continue
            six = stated(body)
            if six:
                pairs.append(dict(patient=p, prior=prior["path_to_image"], current=current["path_to_image"],
                                  order=[int(prior["patient_report_date_order"]),
                                         int(current["patient_report_date_order"])], six=sorted(six)))
    print(f"{len(by_patient)} patients with a frontal study; {len(pairs)} eligible consecutive pairs")
    print("eligible pairs by condition:", dict(Counter(c for q in pairs for c in q["six"])))
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    chosen, used = [], set()
    counts = Counter(c for q in pairs for c in q["six"])
    for c in sorted(TERMS, key=lambda c: counts[c]):  # rarest first, up to 5 pairs per condition
        for q in pairs:
            if len(chosen) >= N or sum(c in x["six"] for x in chosen) >= 5:
                break
            if c in q["six"] and q["patient"] not in used:
                chosen.append(q)
                used.add(q["patient"])
    for q in pairs:  # fill with any eligible pair
        if len(chosen) >= N:
            break
        if q["patient"] not in used:
            chosen.append(q)
            used.add(q["patient"])
    png = lambda path: "PNG_train/" + path.split("train/", 1)[1].replace(".jpg", ".png")  # noqa: E731
    for q in chosen:
        q["prior_png"], q["current_png"] = png(q["prior"]), png(q["current"])
    OUT.write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    print("chosen by condition:", dict(Counter(c for q in chosen for c in q["six"])))
    files = OUT.with_suffix(".files.txt")
    files.write_text("\n".join(f for q in chosen for f in (q["prior_png"], q["current_png"])) + "\n", encoding="utf-8")
    print(f"{len(chosen)} pairs -> {OUT}; {2 * len(chosen)} image files listed in {files}")


if __name__ == "__main__":
    main()
