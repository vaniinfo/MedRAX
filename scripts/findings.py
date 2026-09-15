"""The findings measured, and how each is recognised in Open-i's MeSH labels.

Shared by fetch_openi_eval_set.py, measure_reliability.py and analyze_reliability.py so
the three cannot drift apart -- which they already did once: measure_reliability.py
gained pneumothorax and consolidation while the RELIABILITY table in
medrax/agent/validator.py kept the original four findings.

  mesh  lowercased substrings of the MeSH `problems` field that prove presence
  clf   the ChestXRayClassifierTool key for the same finding

The mesh lists are not guesses. Open-i indexes with MeSH headings rather than with the
words a radiologist would say, so the obvious substring is sometimes the wrong one --
see the note on consolidation.
"""

# Below this many positives a row is noise, not a measurement. Enforced by
# analyze_reliability.py, which refuses to emit an underpowered row into RELIABILITY.
MIN_POSITIVES = 20

FINDINGS = {
    "cardiomegaly":     {"mesh": ["cardiomegaly"],     "clf": "Cardiomegaly"},
    "pleural effusion": {"mesh": ["pleural effusion"], "clf": "Effusion"},
    "pneumothorax":     {"mesh": ["pneumothorax"],     "clf": "Pneumothorax"},
    # Open-i barely uses "Consolidation" as a heading: 3 occurrences in a 175-image
    # sample, against 12 for "Airspace Disease", which is how this corpus indexes the
    # same finding. Matching the literal word alone put nearly every true consolidation
    # case in the negative class. "Opacity" (27) was considered and rejected -- it also
    # covers nodules, masses and scarring, so it would measure something broader than
    # the finding actually being asked about.
    "consolidation":    {"mesh": ["consolidation", "airspace disease"],
                         "clf": "Consolidation"},
    "pulmonary edema":  {"mesh": ["pulmonary edema"],  "clf": "Edema"},
    "atelectasis":      {"mesh": ["atelectasis"],      "clf": "Atelectasis"},
}


def is_positive(problems: str, finding: str) -> bool:
    """Ground truth for one finding, from the curated MeSH labels.

    Note the assumption: absence of the heading is taken as absence of the finding.
    Open-i's indexing is not exhaustive, so some negatives are really unlabelled
    positives. That depresses a measured AUC rather than inflating it, which is the
    safer direction to be wrong in, but it does mean these numbers are a floor.
    """
    body = (problems or "").lower()
    return any(m in body for m in FINDINGS[finding]["mesh"])
