"""Check option 3 (Evidence Engine v1.0 states) against the 0.5 cut-off on the demo MIMIC images.

For each demo/<dicom_id>.png with a gold report in demo/medgemma_vs_gold.csv, CheXagent scores
the six validated findings ("Is there {finding}?", p_present as in all experiments). Each
score is judged two ways:
  - option 3: the frozen Evidence Engine (primary, NIH-derived): SUPPORTING = present,
    REFUTING = absent, UNCERTAIN / INSUFFICIENT DATA = no call. "top" variant: scores above
    the engine's validated range (OUTSIDE_SUPPORT on the high side) counted as present;
  - the current rule: p_present >= 0.5 = present, otherwise absent.
Truth is labelled by hand from the gold reports (TRUTH below, with the report wording);
None = the report is ambiguous, left out of the scoring.

    python experiments/demo_option3_check.py        # writes temp/demo_option3_check.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medrax.evidence import EvidenceEngine  # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "demo"
OUT = Path(__file__).resolve().parents[1] / "temp" / "demo_option3_check.md"
FINDINGS = ["cardiomegaly", "pleural effusion", "pneumothorax", "consolidation",
            "pulmonary edema", "atelectasis"]

# View of each image (checked by eye); the engine was developed on frontal images only
LATERAL = {"00a85162", "00eb6d7f", "00fe920d", "0a34b2f0"}

# Hand labels from the gold reports: 1 present, 0 absent (stated absent or not mentioned),
# None ambiguous. The note gives the report wording behind each non-trivial label.
TRUTH = {
    "000a0e37": ({"cardiomegaly": 0, "pleural effusion": 1, "pneumothorax": 1, "consolidation": None,
                  "pulmonary edema": 0, "atelectasis": 0},
                 "moderate bilateral pleural effusion; 'tiny volume of right apical pneumothorax' "
                 "(wet read: no pneumothorax); consolidation only in the wet read"),
    "00a85162": ({f: 0 for f in FINDINGS}, "no acute process (emphysema)"),
    "00cd1b67": ({"cardiomegaly": 0, "pleural effusion": 1, "pneumothorax": 0, "consolidation": 0,
                  "pulmonary edema": 1, "atelectasis": None},
                 "diffuse extensive pulmonary edema; small effusion 'most likely present'; "
                 "'may be due to resolution of atelectasis'"),
    "00ce997e": ({"cardiomegaly": 0, "pleural effusion": 0, "pneumothorax": None, "consolidation": 0,
                  "pulmonary edema": 0, "atelectasis": 1},
                 "minimal bibasilar atelectasis; heart top normal; pneumothorax evaluation limited"),
    "00dd8c5b": ({f: 0 for f in FINDINGS}, "no acute process"),
    "00eb6d7f": ({"cardiomegaly": 0, "pleural effusion": 1, "pneumothorax": 0, "consolidation": 0,
                  "pulmonary edema": 0, "atelectasis": 0},
                 "some right pleural effusion; heart top-normal"),
    "00fcb75b": ({"cardiomegaly": 1, "pleural effusion": 0, "pneumothorax": 0, "consolidation": 0,
                  "pulmonary edema": None, "atelectasis": 0},
                 "mild cardiomegaly; 'very mild pulmonary vascular congestion' (not called edema)"),
    "00fe6e66": ({f: 0 for f in FINDINGS}, "no acute process"),
    "00fe920d": ({"cardiomegaly": 0, "pleural effusion": 1, "pneumothorax": 0, "consolidation": 0,
                  "pulmonary edema": 0, "atelectasis": 1},
                 "small residual right pleural effusion with right basal atelectasis"),
    "0a34b2f0": ({"cardiomegaly": 0, "pleural effusion": 0, "pneumothorax": 0, "consolidation": None,
                  "pulmonary edema": 0, "atelectasis": 0},
                 "subtle ground-glass opacities LLL, 'compatible with pneumonia' (not called consolidation)"),
}


def option3(ev):
    """(call, call counting top-of-range scores as present); call is 1, 0 or None."""
    if ev.state == "SUPPORTING":
        return 1, 1
    if ev.state == "REFUTING":
        return 0, 0
    top = ev.reason == "OUTSIDE_SUPPORT" and ev.p_present >= 0.5
    return None, (1 if top else None)


def main():
    from medrax.tools.xray_vqa import XRayVQATool

    tool = XRayVQATool(cache_dir="C:/Users/hari/model-weights", device="cuda", temp_dir="temp")
    engine = EvidenceEngine()
    rows = []
    for key, (truth, note) in TRUTH.items():
        path = next(DEMO.glob(f"{key}*.png"))
        for finding in FINDINGS:
            _, yes_no = tool._generate_response([str(path)], f"Is there {finding}?", 8)
            p = yes_no["p_yes"] if yes_no else None
            ev = engine.assess(finding, p) if p is not None else None
            call3, call3_top = option3(ev) if ev else (None, None)
            rows.append({
                "image": key, "view": "lateral" if key in LATERAL else "frontal", "finding": finding,
                "truth": truth[finding], "p": p,
                "state": f"{ev.state} ({ev.reason})" if ev else "no yes/no answer",
                "opt3": call3, "opt3_top": call3_top,
                "half": None if p is None else int(p >= 0.5),
            })
        print(f"{key} done", flush=True)

    lines = ["# Option 3 vs 0.5 cut-off on the demo MIMIC images", "",
             "Truth: hand labels from `demo/medgemma_vs_gold.csv` gold reports "
             "(ambiguous labels left out). Option 3: frozen Evidence Engine v1.0, primary.", ""]

    def score(subset, col):
        scored = [r for r in subset if r["truth"] is not None]
        calls = [r for r in scored if r[col] is not None]
        right = [r for r in calls if r[col] == r["truth"]]
        pos = [r for r in scored if r["truth"] == 1]
        pos_found = [r for r in pos if r[col] == 1]
        neg = [r for r in scored if r["truth"] == 0]
        false_pos = [r for r in neg if r[col] == 1]
        return (f"{len(calls)}/{len(scored)} called | {len(right)}/{len(calls)} of calls correct | "
                f"positives found {len(pos_found)}/{len(pos)} | false positives {len(false_pos)}/{len(neg)}")

    for view in ("frontal", "lateral"):
        subset = [r for r in rows if r["view"] == view]
        lines += [f"## {view.capitalize()} images", "",
                  f"- 0.5 cut-off: {score(subset, 'half')}",
                  f"- option 3: {score(subset, 'opt3')}",
                  f"- option 3, top-of-range counted as present: {score(subset, 'opt3_top')}", ""]

    lines += ["## Every reading", "",
              "| image | view | finding | truth | p_present | 0.5 rule | option 3 state |",
              "|---|---|---|---|---|---|---|"]
    mark = {1: "present", 0: "absent", None: "-"}
    for r in rows:
        ok = lambda c: "" if c is None or r["truth"] is None else (" ✓" if c == r["truth"] else " ✗")
        lines.append(f"| {r['image']} | {r['view']} | {r['finding']} | "
                     f"{'ambiguous' if r['truth'] is None else mark[r['truth']]} | {r['p']} | "
                     f"{mark[r['half']]}{ok(r['half'])} | {r['state']}{ok(r['opt3_top'])} |")
    lines += ["", "## Truth notes", ""] + [f"- {k}: {note}" for k, (_, note) in TRUTH.items()]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
