"""Rule-based, sentence-level audit of how a reference report describes one condition (visualization report QC).

Categories (never inferred from silence):
  explicitly present   an affirmative mention (sides recorded); 'no (significant/interval) change in X' = present
  explicitly absent    a negated mention ('no', 'without', 'negative for', 'free of', 'clear of', 'resolved', 'no
                       longer'); negation carries over a list ('no consolidation, effusion, or pneumothorax');
                       laterality-specific negation ('no right effusion') is recorded as absent on that side only
  possible/uncertain   hedged mention ('may', 'possible', 'probable', 'likely', 'could', 'cannot be excluded',
                       'suggest', 'questionable', 'versus', 'X or Y' between diagnoses); interval-only negation ('no
                       new consolidation': existing disease not excluded); or an indirect sign without the term
                       (effusion: costophrenic blunting / meniscus; atelectasis: volume loss / collapse;
                       consolidation: airspace opacity / pneumonia)
  not mentioned        none of the above; sentences with generic opacity wording are listed as 'related' only
Scope: a negation or hedge applies within its clause (split at ';', ',', 'but', 'however', 'although', 'while',
'with'). Exact sentences are returned verbatim. NIH has labels only (no report text).
"""

import re

RULES_VERSION = "2026-11-09 (v3: 'pleural air' = pneumothorax; 'pleural air or fluid' = effusion; 'suspect(ed)',"\
                " 'cannot exclude'; trailing hedges may have up to two intervening words)"
TERMS = {"pleural effusion": r"(?<!pericardial )\beffusions?\b|hydrothorax|pleural (?:air (?:or|and) )?fluid",
         "atelectasis": r"\batelecta\w*",
         "consolidation": r"\bconsolidat\w*",
         "pneumothorax": r"\b(?:hydro)?pneumothora(?:x|ces)\b|\bpleural air\b|\bair in the pleural space\b"}
INDIRECT = {"pleural effusion": r"blunt\w*\s+(?:of\s+)?(?:the\s+)?(?:\w+\s+)?costophrenic|costophrenic\s+(?:angles?\s+)?"
                                r"(?:is\s+|are\s+)?blunt\w*|\bmeniscus\b|layering fluid",
            "atelectasis": r"volume loss|\bcollapse\w*",
            "consolidation": r"airspace (?:opacit\w*|disease)|\bpneumonia\b",
            "pneumothorax": r"(?!x)x"}
RELATED = r"opacit\w*|opacification|densit(?:y|ies)|infiltrat\w*"
NEG = r"\b(?:no|without|negative for|free of|clear of|absence of|resolution of)\b"
NOT_NEG = r"\bno\s+(?:significant\s+|interval\s+|appreciable\s+)?change\b"
RESOLVED = r"\bresolved\b|\bno longer\b"
HEDGE = (r"\b(?:may|might|possibl[ey]|probabl[ey]|likely|could|cannot be (?:excluded|ruled out)|can't be excluded|"
         r"not (?:be )?excluded|suggest\w*|suspicious|questionable|versus|vs\.?|concern for|equivocal|"
         r"difficult to exclude|presumed|suspect(?:ed)?|cannot exclude|can't exclude|not exclude[ds]?|"
         r"question of)\b")
HEDGE_AFTER = (r"\b(?:cannot be (?:excluded|ruled out)|can't be excluded|not (?:be )?excluded|(?:is|are)\s+"
               r"(?:\w+\s+){0,2}(?:likely|possible|probable|suspected|questioned|not excluded)|may be present|"
               r"could be present)\b|\?")
DX = r"(?:atelecta\w*|pneumonia|consolidat\w*|effusion|scarring|aspiration|infection|edema|fibrosis)"
SIDE = {"right": r"\bright\b|\brt\b", "left": r"\bleft\b|\blt\b", "both": r"\bbilateral\w*|\bbibasilar|\bboth\b"}


def sentences(text):
    t = " ".join(str(text or "").split())
    t = re.sub(r"\b(FINDINGS|IMPRESSION|CONCLUSION)\s*:", r"\n\1:", t)
    out = []
    for block in t.split("\n"):
        out += [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z_])", block) if s.strip()]
    return out


def clauses(sentence):
    return [c for c in re.split(r";|,|\bbut\b|\bhowever\b|\balthough\b|\bwhile\b|\bwith\b", sentence, flags=re.I)
            if c.strip()]


def sides_in(text):
    s = {k for k, p in SIDE.items() if re.search(p, text, re.I)}
    return {"right", "left"} if "both" in s or {"right", "left"} <= s else s


def classify(condition, text):
    """{'category', 'present_sides', 'absent_sides', 'evidence': [(label, exact sentence)], 'related': [...]}"""
    term = TERMS[condition]
    out = {"category": "not mentioned", "present_sides": set(), "absent_sides": set(), "evidence": [], "related": []}
    labels, related = [], []
    for sent in sentences(text):
        if not re.search(term, sent, re.I):
            if re.search(INDIRECT[condition], sent, re.I) and not re.search(NEG, sent, re.I):
                labels.append("uncertain (indirect sign)")
                out["present_sides"] |= {f"possible {x}" for x in sides_in(sent)}
                out["evidence"].append(("uncertain (indirect sign, term not used)", sent))
            elif re.search(RELATED, sent, re.I):
                related.append(sent)
            continue
        carry = False
        for cl in (c for part in sent.split(";") for c in [None] + clauses(part)):
            if cl is None:          # a semicolon ends any list-carried negation
                carry = False
                continue
            m = re.search(term, cl, re.I)
            if not m:
                # a negated clause starts a list; a short list item keeps the negation going
                carry = bool(re.search(NEG, cl, re.I) and not re.search(NOT_NEG, cl, re.I)) or \
                    (carry and len(cl.split()) <= 4)
                continue
            # each mention judged on its own segment: text since the previous mention (negation / hedge before it)
            # and text up to the next mention (only trailing hedges such as 'cannot be excluded', 'is likely')
            occ = list(re.finditer(term, cl, re.I))
            for i, mo in enumerate(occ):
                before = cl[occ[i - 1].end() if i else 0:mo.start()]
                after = cl[mo.end():occ[i + 1].start() if i + 1 < len(occ) else len(cl)]
                seg = before + mo.group(0) + after
                sides = sides_in(seg) or sides_in(cl)
                stripped = re.sub(r"\b(?:or|and|nor|any|focal|the|a|right|left|bilateral|pleural|small|large)\b", "",
                                  before, flags=re.I).strip()
                listed = (carry or i > 0 and labels and labels[-1] == "absent") and not stripped
                negated = bool(re.search(NEG, before, re.I) or listed) and not re.search(NOT_NEG, seg, re.I)
                hedged = re.search(HEDGE, before, re.I) or re.search(HEDGE_AFTER, after, re.I) or \
                    re.search(DX + r"\s+or\b|\bor\s+" + DX, seg, re.I)
                if negated and re.search(r"\bnew\b", before, re.I):
                    lab = "uncertain (interval-only negation: 'no new')"
                elif negated or re.search(RESOLVED, after, re.I):
                    lab = "absent"
                    out["absent_sides"] |= sides or {"unspecified"}
                elif hedged:
                    lab = "uncertain"
                    out["present_sides"] |= {f"possible {s}" for s in sides}
                else:
                    lab = "present"
                    out["present_sides"] |= sides or {"side not stated"}
                carry = lab == "absent" or lab.startswith("uncertain (interval")
                labels.append(lab)
                out["evidence"].append((lab, sent))
    if any(lbl == "present" for lbl in labels):
        out["category"] = "explicitly present"
    elif any(lbl.startswith("uncertain") for lbl in labels):
        out["category"] = "possible/uncertain"
    elif labels:
        out["category"] = "explicitly absent"
    out["evidence"] = list(dict.fromkeys(out["evidence"]))
    out["related"] = related if out["category"] == "not mentioned" else []
    return out


def describe(res):
    c = res["category"]
    if c == "explicitly present":
        s = sorted(x for x in res["present_sides"] if not x.startswith("possible"))
        txt = "EXPLICITLY PRESENT" + (f" ({', '.join(s)})" if s else "")
        if res["absent_sides"] - {"unspecified"}:
            txt += f"; explicitly absent on the {', '.join(sorted(res['absent_sides'] - {'unspecified'}))}"
        return txt
    if c == "explicitly absent":
        a = sorted(res["absent_sides"] - {"unspecified"})
        return "EXPLICITLY ABSENT" + (f" (stated for the {' and '.join(a)} side{'s' if len(a) > 1 else ''})"
                                      if a and len(a) < 2 else "")
    if c == "possible/uncertain":
        ind = any(lbl.startswith("uncertain (indirect") for lbl, _ in res["evidence"])
        return "POSSIBLE / UNCERTAIN" + (" - indirect sign only" if ind and all(
            lbl.startswith("uncertain (indirect") for lbl, _ in res["evidence"]) else "") + \
            (f" ({', '.join(sorted(res['present_sides']))})" if res["present_sides"] else "")
    return "NOT MENTIONED in the reference report (absence not established)"
