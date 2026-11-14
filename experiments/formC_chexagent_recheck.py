"""EXPLORATORY (post hoc, after reading Dr Franco's form C): re-runs CheXagent on the 13 form-C rows where the doctor
rated CheXagent's evidence below 3 and MedGemma's finding above 3, to see why CheXagent disagreed with the doctor.
For each row: the original query repeated on the current image; the two-image (prior + current) query; two other
wordings; and CheXagent's box with its side and zone. Nothing here changes the forms or any pre-registered result.

    .venv/Scripts/python.exe experiments/formC_chexagent_recheck.py
"""

import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
from openpyxl import load_workbook  # noqa: E402

BOOK = Path("D:/datasets/doctor_response/doctor_responses_20261106-184314.xlsx")
IMAGES = Path("D:/datasets/held_out_eval_images/images/images")
SUBSET = Path("D:/datasets/held_out_eval_images/subset.csv")
INPUTS = ROOT / "logs" / "mimic_heldout_inputs_2.json"
OUT = ROOT / "logs" / "formC_chexagent_recheck.json"
TWO = ("The first image is the prior chest X-ray and the second is the current chest X-ray. "
       "Is there {c} in the current chest X-ray?")
WORDINGS = ["Is there a {c}?", "Does this chest X-ray show {c}?"]


def main():
    from medrax.tools.medgemma import describe_location
    from medrax.tools.xray_vqa import XRayVQATool
    ws = load_workbook(BOOK)["Form C"]
    head = [c.value for c in ws[1]]
    rows = [dict(zip(head, r)) for r in ws.iter_rows(min_row=2, values_only=True)]
    sel = [r for r in rows if (r["Score: CheXagent evidence"] or 9) < 3 and (r["Score: MedGemma finding"] or 0) > 3]
    pairs = json.loads(INPUTS.read_text(encoding="utf-8"))
    sub = list(csv.DictReader(SUBSET.open(encoding="utf-8")))
    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    out = []
    for r in sel:
        p = pairs[r["Case"] - 1]
        cur, pri, c = str(IMAGES / p["current_png"]), str(IMAGES / p["prior_png"]), r["Condition"]
        meta = sub[p["row"]]
        res = {"case": r["Case"], "condition": c, "row_type": r["Row type"], "shown": r["CheXagent shown"],
               "verdict": r["CheXagent verdict (engine)"], "p_original": r["CheXagent p_present"],
               "support_cutoff": r["Support cut-off"], "against_cutoff": r["Evidence-against cut-off"],
               "reference": r["Reference (ground truth)"], "reference_quote": r["Reference quote"],
               "medgemma": r["MedGemma finding (shown)"], "score_medgemma": r["Score: MedGemma finding"],
               "score_chexagent": r["Score: CheXagent evidence"],
               "view": f"{meta['view_position_before']} -> {meta['view_position_after']}"}
        res["p_repeat"] = (tool._generate_response([cur], f"Is there {c}?", 8)[1] or {}).get("p_yes")
        res["p_two_image"] = (tool._generate_response([pri, cur], TWO.format(c=c), 8)[1] or {}).get("p_yes")
        res["p_wordings"] = [(tool._generate_response([cur], w.format(c=c), 8)[1] or {}).get("p_yes") for w in WORDINGS]
        proof, _ = tool.locate_findings(cur, [c])
        res["box"] = ["; ".join(describe_location(b["box_normalized"]).replace("patient's ", "") for b in q["boxes"])
                      for q in proof if q["localized"]] or ["not localized"]
        out.append(res)
        print(f"case {res['case']:2d} {c:18s} orig {res['p_original']:.3f} repeat {res['p_repeat']:.3f} "
              f"two-image {res['p_two_image']:.3f} wordings {[round(x, 3) for x in res['p_wordings']]} "
              f"box {res['box']} view {res['view']}", flush=True)
    if OUT.exists():
        OUT.rename(OUT.with_name(OUT.stem + f"_{time.strftime('%Y%m%d-%H%M%S')}.json"))
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
