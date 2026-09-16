"""Evidence-driven validation that runs as code, not as a prompt instruction.

Three approaches to validating tool output were compared on this repo:

1. Prompt-only (a system prompt asking the LLM to validate). The LLM decides whether
   to comply. Measured: it skipped validation entirely, and ignored an explicit
   arithmetic rule roughly half the time.
2. CXRAgent's `_explain_func` (arXiv:2510.21324): validation is a function call inside
   the tool-execution loop, so it always runs -- but the judgement is still a free-text
   LLM response with no numbers.
3. A deterministic scorer: the verdict is computed in Python.

This module is the synthesis. Validation is a forced function call (from 2), the
arithmetic is computed in code (from 3), and the LLM is used only for the one thing it
is actually needed for: describing what is visible in the image.
"""

import base64
import numbers
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

# Measured per-tool, per-finding reliability. 544 frontal chest X-rays from the Indiana
# University collection, ground truth from the curated MeSH labels, local inference only.
#
#   scripts/build_eval_set.py       selects the images (reproducible from --seed)
#   scripts/measure_reliability.py  runs the three tools over them
#   scripts/analyze_reliability.py  produces this table, with bootstrap intervals
#
# Supersedes a 212-image Open-i table whose data no longer exists and could not have been
# regenerated: Open-i redraws a ranked search on every call, so the set was never the same
# twice. The AUCs below replicate that table within ~0.06 on 2.5x the images. The
# thresholds do not replicate, which is the more important result.
#
#   threshold : the decision point maximising balanced accuracy, NOT 0.5
#   auc       : discrimination for this finding; 0.5 is chance
#
# Read the threshold interval before resting any argument on a single reading. The
# earlier table's headline claim -- that the classifier's cardiomegaly decision point is
# 0.20, making a reading of 0.46 "a clear positive" -- does not survive remeasurement at
# 0.45. Both values sit inside this row's 0.25-0.55 interval. The point estimate was never
# precise enough to settle the case that prompted it, in either direction.
#   thr_ci    : 95% bootstrap interval for that decision point. A reading inside it is
#               not a vote -- a plausible alternative threshold would flip its direction.
#   output_type: "probability" where the tool grades its answers, "binary" where it does
#               not. Measured: 43-99% of readings land in 0.05-0.95 for CheXagent and the
#               classifier, 1-6% for MedGemma. Only a graded tool earns the margin.
#   yes / no   : how often an answer in THAT DIRECTION is right at this operating point,
#               with the number of calls it rests on. Reliability is tool x finding x
#               POLARITY, because the same model differs sharply by direction: MedGemma
#               on pneumothorax is PPV 54.5% saying yes and NPV 95.9% saying no.
#               `strength` normalises against the no-skill baseline -- prevalence for a
#               positive, 1-prevalence for a negative -- the way (auc-0.5)*2 does. That
#               is what turns the flattering 95.9% NPV into 0.198, since at 5.1%
#               prevalence answering "no" to everything already scores 94.9%.
#
# AUC is population-level discrimination. It is NOT the probability that any particular
# answer is correct -- MedGemma reaches 0.792 on pneumothorax while a positive call from
# it is right about half the time. Use ppv/npv when the question is what a claim is worth.
RELIABILITY: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("chest_xray_expert", "cardiomegaly"):
        {"auc": 0.909, "threshold": 0.45, "thr_ci": (0.45, 0.60), "output_type": "probability",
         "prevalence": 0.289,
         "yes": {"ppv": 0.618, "n": 238, "strength": 0.463},
         "no": {"npv": 0.967, "n": 306, "strength": 0.887},
         },
    ("chest_xray_classifier", "cardiomegaly"):
        {"auc": 0.852, "threshold": 0.45, "thr_ci": (0.25, 0.55), "output_type": "probability",
         "prevalence": 0.289,
         "yes": {"ppv": 0.532, "n": 265, "strength": 0.342},
         "no": {"npv": 0.943, "n": 279, "strength": 0.801},
         },
    ("chest_xray_expert_gemma", "cardiomegaly"):
        {"auc": 0.889, "threshold": 0.30, "thr_ci": (0.05, 0.95), "output_type": "binary",
         "prevalence": 0.289,
         "yes": {"ppv": 0.675, "n": 191, "strength": 0.544},
         "no": {"npv": 0.921, "n": 353, "strength": 0.725},
         },
    ("chest_xray_expert", "pleural effusion"):
        {"auc": 0.951, "threshold": 0.70, "thr_ci": (0.50, 0.80), "output_type": "probability",
         "prevalence": 0.221,
         "yes": {"ppv": 0.739, "n": 134, "strength": 0.665},
         "no": {"npv": 0.949, "n": 410, "strength": 0.768},
         },
    ("chest_xray_classifier", "pleural effusion"):
        {"auc": 0.887, "threshold": 0.50, "thr_ci": (0.40, 0.60), "output_type": "probability",
         "prevalence": 0.221,
         "yes": {"ppv": 0.495, "n": 218, "strength": 0.353},
         "no": {"npv": 0.963, "n": 326, "strength": 0.833},
         },
    ("chest_xray_expert_gemma", "pleural effusion"):
        {"auc": 0.905, "threshold": 0.40, "thr_ci": (0.05, 0.95), "output_type": "binary",
         "prevalence": 0.221,
         "yes": {"ppv": 0.696, "n": 125, "strength": 0.610},
         "no": {"npv": 0.921, "n": 419, "strength": 0.643},
         },
    ("chest_xray_expert", "pneumothorax"):
        {"auc": 0.948, "threshold": 0.25, "thr_ci": (0.10, 0.60), "output_type": "probability",
         "prevalence": 0.051,
         "yes": {"ppv": 0.400, "n": 55, "strength": 0.367},
         "no": {"npv": 0.988, "n": 489, "strength": 0.762},
         },
    ("chest_xray_classifier", "pneumothorax"):
        {"auc": 0.622, "threshold": 0.35, "thr_ci": (0.05, 0.50), "output_type": "probability",
         "prevalence": 0.051,
         "yes": {"ppv": 0.083, "n": 180, "strength": 0.034},
         "no": {"npv": 0.964, "n": 364, "strength": 0.306},
         },
    ("chest_xray_expert_gemma", "pneumothorax"):
        {"auc": 0.792, "threshold": 0.05, "thr_ci": (0.05, 0.95), "output_type": "binary",
         "prevalence": 0.051,
         "yes": {"ppv": 0.545, "n": 11, "strength": 0.521},
         "no": {"npv": 0.959, "n": 533, "strength": 0.198},
         },
    ("chest_xray_expert", "consolidation"):
        {"auc": 0.810, "threshold": 0.25, "thr_ci": (0.15, 0.40), "output_type": "probability",
         "prevalence": 0.200,
         "yes": {"ppv": 0.438, "n": 176, "strength": 0.297},
         "no": {"npv": 0.913, "n": 368, "strength": 0.566},
         },
    ("chest_xray_classifier", "consolidation"):
        {"auc": 0.762, "threshold": 0.50, "thr_ci": (0.50, 0.50), "output_type": "probability",
         "prevalence": 0.200,
         "yes": {"ppv": 0.335, "n": 260, "strength": 0.168},
         "no": {"npv": 0.923, "n": 284, "strength": 0.613},
         },
    ("chest_xray_expert_gemma", "consolidation"):
        {"auc": 0.775, "threshold": 0.05, "thr_ci": (0.05, 0.10), "output_type": "binary",
         "prevalence": 0.200,
         "yes": {"ppv": 0.476, "n": 124, "strength": 0.344},
         "no": {"npv": 0.881, "n": 420, "strength": 0.406},
         },
    ("chest_xray_expert", "pulmonary edema"):
        {"auc": 0.897, "threshold": 0.50, "thr_ci": (0.20, 0.75), "output_type": "probability",
         "prevalence": 0.083,
         "yes": {"ppv": 0.461, "n": 76, "strength": 0.412},
         "no": {"npv": 0.979, "n": 468, "strength": 0.742},
         },
    ("chest_xray_classifier", "pulmonary edema"):
        {"auc": 0.807, "threshold": 0.15, "thr_ci": (0.05, 0.40), "output_type": "probability",
         "prevalence": 0.083,
         "yes": {"ppv": 0.206, "n": 170, "strength": 0.134},
         "no": {"npv": 0.973, "n": 374, "strength": 0.677},
         },
    ("chest_xray_expert_gemma", "pulmonary edema"):
        {"auc": 0.862, "threshold": 0.05, "thr_ci": (0.05, 0.95), "output_type": "binary",
         "prevalence": 0.083,
         "yes": {"ppv": 0.595, "n": 37, "strength": 0.558},
         "no": {"npv": 0.955, "n": 507, "strength": 0.452},
         },
    ("chest_xray_expert", "atelectasis"):
        {"auc": 0.819, "threshold": 0.40, "thr_ci": (0.35, 0.65), "output_type": "probability",
         "prevalence": 0.296,
         "yes": {"ppv": 0.464, "n": 317, "strength": 0.238},
         "no": {"npv": 0.938, "n": 227, "strength": 0.792},
         },
    ("chest_xray_classifier", "atelectasis"):
        {"auc": 0.699, "threshold": 0.40, "thr_ci": (0.35, 0.55), "output_type": "probability",
         "prevalence": 0.296,
         "yes": {"ppv": 0.414, "n": 314, "strength": 0.168},
         "no": {"npv": 0.865, "n": 230, "strength": 0.545},
         },
    ("chest_xray_expert_gemma", "atelectasis"):
        {"auc": 0.747, "threshold": 0.80, "thr_ci": (0.05, 0.95), "output_type": "binary",
         "prevalence": 0.296,
         "yes": {"ppv": 0.435, "n": 306, "strength": 0.197},
         "no": {"npv": 0.882, "n": 238, "strength": 0.602},
         },
}

# CheXagent beats the classifier on five of six findings, paired on the same bootstrap
# resamples. Consolidation is the exception at +0.047 (CI -0.001 to +0.100) -- within
# noise. An earlier claim that it wins on *every* finding was two AUCs compared by eye.
#
#   pneumothorax +0.325   cardiomegaly +0.056   consolidation +0.047 (no separation)
#   atelectasis  +0.118   effusion     +0.064
#   edema        +0.090


# Do two tools get the SAME films wrong? Claim synthesis needs to know whether a second
# opinion is independent evidence or an echo, and neither AUC nor PPV says which.
# Measured as the correlation of their error indicators at their own operating points.
#
# The spread matters. On pneumothorax the tools fail almost independently (0.13-0.17),
# so agreement there is real corroboration. On atelectasis they are wrong in lockstep
# (0.60-0.64) -- which is why all three called atelectasis on 3453_IM-1676, a film
# labelled normal. Three tools agreeing is not three pieces of evidence.
DEPENDENCE: Dict[Tuple[str, str, str], float] = {
    ("cardiomegaly", "chest_xray_classifier", "chest_xray_expert"): 0.508,
    ("cardiomegaly", "chest_xray_classifier", "chest_xray_expert_gemma"): 0.462,
    ("cardiomegaly", "chest_xray_expert", "chest_xray_expert_gemma"): 0.551,
    ("pleural effusion", "chest_xray_classifier", "chest_xray_expert"): 0.340,
    ("pleural effusion", "chest_xray_classifier", "chest_xray_expert_gemma"): 0.367,
    ("pleural effusion", "chest_xray_expert", "chest_xray_expert_gemma"): 0.587,
    ("pneumothorax", "chest_xray_classifier", "chest_xray_expert"): 0.125,
    ("pneumothorax", "chest_xray_classifier", "chest_xray_expert_gemma"): 0.147,
    ("pneumothorax", "chest_xray_expert", "chest_xray_expert_gemma"): 0.166,
    ("consolidation", "chest_xray_classifier", "chest_xray_expert"): 0.449,
    ("consolidation", "chest_xray_classifier", "chest_xray_expert_gemma"): 0.383,
    ("consolidation", "chest_xray_expert", "chest_xray_expert_gemma"): 0.582,
    ("pulmonary edema", "chest_xray_classifier", "chest_xray_expert"): 0.405,
    ("pulmonary edema", "chest_xray_classifier", "chest_xray_expert_gemma"): 0.226,
    ("pulmonary edema", "chest_xray_expert", "chest_xray_expert_gemma"): 0.456,
    ("atelectasis", "chest_xray_classifier", "chest_xray_expert"): 0.606,
    ("atelectasis", "chest_xray_classifier", "chest_xray_expert_gemma"): 0.635,
    ("atelectasis", "chest_xray_expert", "chest_xray_expert_gemma"): 0.604,
}

# The report generator emits no probability, so it is scored on whether its text asserts
# the finding. precision = of the reports asserting it, the fraction correct.
#
# It also contradicts itself -- asserting in FINDINGS and denying in IMPRESSION, or the
# reverse -- on 14% of cardiomegaly mentions and 11% of effusion mentions. Those reports
# are excluded from both rates here rather than forced into one column; _text_stance
# reports them as CONTRADICTS so the Director sees the disagreement instead of a verdict.
REPORT_RELIABILITY: Dict[str, Dict[str, float]] = {
    "cardiomegaly":     {"recall": 0.41, "precision": 0.65},
    "pleural effusion": {"recall": 0.58, "precision": 0.69},
    # Asserting these two is barely better than a coin flip. Weight the text accordingly.
    "pneumothorax":     {"recall": 0.46, "precision": 0.46},
    "pulmonary edema":  {"recall": 0.56, "precision": 0.27},
    "consolidation":    {"recall": 0.23, "precision": 0.56},
    # Zero negations in 544 reports: it asserts atelectasis or says nothing at all, so
    # silence is the only negative signal it offers and must not be read as one.
    "atelectasis":      {"recall": 0.30, "precision": 0.45},
}

# Fallback for pairs outside the table above -- every finding in scripts/findings.py is
# now measured, so this applies to findings nobody has evaluated at all. An unmeasured
# pair is flagged as such rather than silently treated as reliable.
UNMEASURED_THRESHOLD = 0.5

# The original guessed dead zone, surviving only where it is the best available answer:
# a pair with no measured threshold interval. For everything in RELIABILITY the band is
# that pair's own thr_ci, which is narrower for some findings and far wider for others --
# 0.15 wide for CheXagent on cardiomegaly, 0.55 wide for CheXagent on edema. Applying one
# uniform band to all of them was the mistake this whole investigation started from.
ASSUMED_DEAD_ZONE: Tuple[float, float] = (0.40, 0.60)

_DESCRIBE_SYSTEM = (
    "You are validating one claim made by a chest X-ray analysis tool, against the image.\n"
    "Report only what is visibly present in this radiograph. Do not restate the claim as "
    "if it were evidence, and do not infer from clinical history.\n"
    "Answer in exactly two lines:\n"
    "Supports: <radiographic signs you can see that support the claim, with anatomical "
    "location, or 'none visible'>\n"
    "Refutes: <contradictory signs, or expected signs that are absent, or 'none visible'>"
)


class EvidenceValidator:
    """Validates each tool result. Numbers are computed; only prose comes from the LLM."""

    # CheXagent's trained grounding template. Verified empirically against several
    # phrasings: this one and "Locate {f} in this chest X-ray." both return boxes,
    # while "Where is the {f}?" returns prose with no coordinates.
    GROUNDING_PROMPT = "Please locate the following phrase in the chest X-ray: {finding}"

    def __init__(self, model: Optional[BaseLanguageModel] = None, describe: bool = True,
                 grounder: Any = None):
        """
        Args:
            model: a vision-capable chat model used only for the descriptive half.
            describe: set False to skip that LLM call (deterministic only).
            grounder: the XRayVQATool. CheXagent can localise a named finding with
                bounding boxes, which is visual evidence from a radiology-trained
                model rather than from the generalist orchestrator.
        """
        self.model = model
        self.describe = describe and model is not None
        self.grounder = grounder
        self._grounding_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._grounding_repeat = False

    # ---------------------------------------------------------------- numbers

    # CheXagent's `confidence` is P(the answer is "Yes"), read off the generation
    # scores. It only means anything when a yes/no question was actually asked.
    _YES_NO_OPENERS = ("does ", "do ", "did ", "is ", "are ", "was ", "were ", "has ",
                       "have ", "can ", "could ", "should ", "will ", "would ")

    @classmethod
    def _is_yes_no_prompt(cls, prompt: str) -> bool:
        return cls._normalise(prompt).startswith(cls._YES_NO_OPENERS)

    @staticmethod
    def _probabilities(result: Any, yes_no: bool = True) -> List[Tuple[str, float]]:
        """Pull every probability a tool reported. Tools return (output, metadata).

        PATCH: `yes_no` guards the CheXagent confidence. Observed on a real run, the
        Director ignored its instruction to use the trained template and asked
        "Identify any abnormalities in this chest X-ray." CheXagent replied "No
        abnormalities detected." with confidence=0.0002 -- the probability of a "Yes"
        token that was never on offer. The validator read that noise as a maximal
        negative vote, margin 1.00, on a question with no yes/no answer to be
        confident about. A value that cannot mean what its name says is dropped.
        """
        payload = result[0] if isinstance(result, tuple) and result else result
        if not isinstance(payload, dict):
            return []
        found: List[Tuple[str, float]] = []
        confidence = EvidenceValidator._as_probability(payload.get("confidence"))
        if confidence is not None and yes_no:
            answer = str(payload.get("response", "")).strip()
            found.append((f"P(yes) [answered {answer!r}]", confidence))
        for key, value in payload.items():
            if key == "confidence":
                continue
            probability = EvidenceValidator._as_probability(value)
            if probability is not None:
                found.append((key, probability))
        return found

    @staticmethod
    def _as_probability(value: Any) -> Optional[float]:
        """Coerce a value to a probability, or None.

        PATCH: must not use isinstance(value, float). torchxrayvision returns
        numpy.float32, for which that test is False -- which silently meant the
        classifier's probabilities were never parsed and the dead-zone rule never
        applied to the one tool it was written for.
        """
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return None
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return None
        return as_float if 0.0 <= as_float <= 1.0 else None

    @staticmethod
    def _reliability(tool: str, finding: Optional[str]) -> Optional[Dict[str, float]]:
        return RELIABILITY.get((tool, finding or ""))

    @classmethod
    def _classify(cls, p: float, tool: str, finding: Optional[str]) -> Dict[str, Any]:
        """Score one probability against the measured threshold for this tool+finding.

        `margin` is distance from the decision point, normalised to 0-1 on whichever
        side the value falls, so it is comparable across tools with different
        thresholds. A small margin means the tool is genuinely undecided -- which is
        not the same as being near 0.5.

        Whether a reading counts as a vote is decided by that pair's measured threshold
        interval, not by the margin. PATCH: it used to be `margin >= 0.15`, a guessed
        constant that ignored discrimination entirely, so the classifier reading 0.50 on
        pneumothorax -- a tool at AUC 0.622, which is the exact failure this code path
        exists to prevent -- came back "supports YES, margin 0.23". Inside thr_ci, a
        plausible alternative threshold would flip the direction, so there is no vote to
        count. Outside it, `auc` still governs how much the vote is worth, in
        _ceiling_from_scored. One question per number.
        """
        info = cls._reliability(tool, finding)
        threshold = info["threshold"] if info else UNMEASURED_THRESHOLD
        auc = info["auc"] if info else None
        thr_ci = info.get("thr_ci") if info else None
        supports_yes = p >= threshold
        margin = ((p - threshold) / (1 - threshold) if supports_yes and threshold < 1
                  else (threshold - p) / threshold if threshold > 0 else 1.0)
        if thr_ci:
            informative = p < thr_ci[0] or p > thr_ci[1]
        else:
            informative = not (ASSUMED_DEAD_ZONE[0] <= p <= ASSUMED_DEAD_ZONE[1])
        return {"supports": "YES" if supports_yes else "NO",
                "margin": round(margin, 3),
                "informative": informative,
                "threshold": threshold, "thr_ci": thr_ci,
                "auc": auc, "measured": info is not None, "tool": tool,
                # "probability" or "binary"; decides whether margin scales the evidence
                # in _strength. Absent for an unmeasured pair, which is treated as
                # graded -- with the fallback AUC of 0.6 it cannot exceed Medium anyway.
                "output_type": (info or {}).get("output_type"),
                # Measured worth of an answer in the direction this reading actually
                # takes, and the prevalence it has to beat to mean anything.
                "polarity": (info or {}).get("yes" if supports_yes else "no"),
                "prevalence": (info or {}).get("prevalence")}

    @classmethod
    def _dependence(cls, finding: Optional[str], a: str, b: str) -> float:
        """How much two tools' errors coincide on this finding. 0 if unmeasured, which
        is the generous reading -- it treats the second opinion as fully independent.

        KNOWN LIMITATION, on the backlog rather than fixed. This is PAIRWISE error
        correlation, which is not the same thing as the conditional dependence of the
        evidence. For three tools A, B and C, a set of pairwise discounts does not fully
        describe their joint error structure: three tools can be pairwise mildly
        correlated and still fail together far more often than those pairs imply, and
        the reverse is also possible. Corroboration here is therefore discounted
        approximately, and on this data the approximation errs toward over-crediting
        the third opinion.

        It is kept simple deliberately. Measured pairwise dependence is already a long
        way better than the assumption it replaces -- that agreeing tools are
        independent -- and the failures of a simple rule stay visible, which is how
        most of the real problems on this branch were found.
        """
        if a == b:
            return 1.0
        first, second = sorted((a, b))
        return DEPENDENCE.get((finding or "", first, second), 0.0)

    @classmethod
    def _synthesise(cls, scored: List[Dict[str, Any]],
                    finding: Optional[str]) -> Dict[str, Any]:
        """Combine evidence into a net position. Deliberately NOT a vote.

        The strongest piece of evidence anchors each side. Further evidence can only
        add, never replace, and adds less the more its errors coincide with the anchor's:

            combined = 1 - (1 - combined) * (1 - strength * (1 - dependence))

        That is noisy-OR, the standard combination for independent evidence. It is
        bounded at 1, it has diminishing returns by construction, and the dependence
        term is measured rather than assumed. No constant decides how much a second
        opinion is worth.

        Why not count agreeing tools: three mediocre tools at 0.20, fully independent,
        reach 0.49 -- still below one excellent tool at 0.78. On atelectasis, where the
        three fail in lockstep at 0.60+, the same three reach only 0.32. That is the
        outvoting failure this whole code path exists to prevent, and it is prevented
        arithmetically rather than by instruction.

        `net` is support minus contradiction. It is deliberately NOT mapped to
        High/Medium/Low here: where those boundaries belong is a question for
        scripts/calibrate_claims.py, which measures how often a claim at a given net
        strength is actually correct, rather than for whoever picks a number.
        """
        live = [s for s in scored if s["informative"]]
        support = cls._combine([s for s in live if s["supports"] == "YES"], finding)
        against = cls._combine([s for s in live if s["supports"] == "NO"], finding)
        return {"support": round(support, 3), "against": round(against, 3),
                "net": round(support - against, 3),
                "n_support": sum(1 for s in live if s["supports"] == "YES"),
                "n_against": sum(1 for s in live if s["supports"] == "NO")}

    @classmethod
    def _combine(cls, entries: List[Dict[str, Any]], finding: Optional[str]) -> float:
        """Anchor on the strongest, then add independence-discounted corroboration."""
        if not entries:
            return 0.0
        ranked = sorted(entries, key=cls._strength, reverse=True)
        anchor = ranked[0]
        combined = cls._strength(anchor)
        for entry in ranked[1:]:
            dependence = cls._dependence(finding, anchor.get("tool", ""),
                                         entry.get("tool", ""))
            contribution = cls._strength(entry) * (1 - dependence)
            combined = 1 - (1 - combined) * (1 - contribution)
        return combined

    @classmethod
    def _ceilings_by_finding(cls, scored: List[Dict[str, Any]]) -> Dict[str, str]:
        """One ceiling per finding, for when no single finding is in question.

        PATCH: commit 484bad5 fixed a ceiling computed across unrelated findings --
        "an irrelevant Cardiomegaly=0.006 forced a High ceiling on a pneumothorax
        question" -- by filtering on _is_relevant(label, focus). That filter passes
        everything when focus is None, so an open-ended question reintroduced exactly
        the same bug through the back door.

        Observed: the Director concluded atelectasis and reported Medium. That Medium
        came from Pneumonia=0.0086, an unmeasured absent finding with nothing to say
        about atelectasis, whose own strength is 0.110 -- Low. A ceiling earned by one
        finding must not license confidence about another, so they are kept apart.
        """
        by_finding: Dict[str, List[Dict[str, Any]]] = {}
        for entry in scored:
            key = entry.get("scored_as") or entry.get("label") or "this finding"
            by_finding.setdefault(key, []).append(entry)
        return {finding: cls._ceiling_from_scored(entries)
                for finding, entries in by_finding.items()}

    @staticmethod
    def _strength(entry: Dict[str, Any]) -> float:
        """What one reading is worth as evidence.

        reliability = (auc - 0.5) * 2 puts discrimination on 0-1: chance scores 0,
        perfect scores 1. Read it for what it is -- a population-level property of this
        tool for this finding, NOT the probability that this particular answer is
        right. MedGemma's 0.792 on pneumothorax says it separates the classes usefully
        over 544 films. It does not say a positive MedGemma pneumothorax answer has a
        79.2% chance of being correct, and nothing here should be read as claiming so.

        The margin is applied only where it carries information:

          probability : a graded tool landing far from its decision point has told you
                        more than one landing near it, so margin scales the evidence.
          binary      : it has not. MedGemma puts 97% of its answers outside 0.05-0.95,
                        so its margin is ~0.99 no matter what it thinks. Multiplying by
                        it converts confident delivery into strong evidence -- the exact
                        chain this code path exists to break -- and would have returned
                        High for every finding it measures, atelectasis at AUC 0.747
                        included.
        """
        reliability = ((entry.get("auc") or 0.6) - 0.5) * 2
        if entry.get("output_type") == "binary":
            # No gradation to read, so the answer is worth what an answer in THIS
            # DIRECTION has measured. A MedGemma "no" on pneumothorax scores 0.198
            # despite NPV 95.9%, because at 5.1% prevalence refusing everything already
            # scores 94.9% -- and discrimination alone would have paid 0.584 for it.
            side = entry.get("polarity")
            return side["strength"] if side else reliability
        return entry["margin"] * reliability

    # A directional rate needs as many observations to mean something as a finding needs
    # positives, so this mirrors MIN_POSITIVES in scripts/findings.py rather than adding
    # a new number. MedGemma's pneumothorax PPV rests on 11 answers: a plausible-looking
    # 54.5% that four more mistakes would move to 40%.
    MIN_POLARITY_N = 20

    @classmethod
    def _strength_tier(cls, entry: Dict[str, Any]) -> str:
        strength = cls._strength(entry)
        tier = "strong" if strength > 0.45 else "moderate" if strength > 0.15 else "weak"
        polarity = entry.get("polarity")
        if tier == "strong" and polarity and polarity["n"] < cls.MIN_POLARITY_N:
            # Too few answers in this direction to license the top tier, however
            # favourable the rate computed from them looks.
            return "moderate"
        return tier

    @classmethod
    def _ceiling_from_scored(cls, scored: List[Dict[str, Any]]) -> str:
        """Ceiling from margin AND discrimination. A wide margin on a tool that barely
        separates this finding is not the same as a wide margin on one that does."""
        live = [s for s in scored if s["informative"]]
        if not scored:
            return "not computable (no probabilities; this tool is unvalidated)"
        if not live:
            return "Low"
        tier = cls._strength_tier(max(live, key=cls._strength))
        return {"strong": "High", "moderate": "Medium", "weak": "Low"}[tier]

    @classmethod
    def _ceiling(cls, probs: List[Tuple[str, float]]) -> str:
        """Confidence ceiling, computed -- never the model's self-assessment."""
        live = [p for _, p in probs if p is not None]
        if not probs:
            # A text-returning tool (the report generator) yields nothing measurable.
            # Reporting "Medium" implied a computed judgement that does not exist, and
            # this is the tool that fabricates most -- invented prior studies, devices
            # and contralateral findings all pass through here unchecked.
            return "not computable (no probabilities; this tool is unvalidated)"
        if not live:
            return "Low"  # every number is a coin flip
        margin = max(abs(p - 0.5) for p in live) * 2
        if margin > 0.6:
            return "High"
        return "Medium" if margin > 0.2 else "Low"

    # ---------------------------------------------------------------- prose

    @staticmethod
    def _image_path(args: Dict[str, Any]) -> Optional[str]:
        if isinstance(args.get("image_paths"), list) and args["image_paths"]:
            return args["image_paths"][0]
        path = args.get("image_path")
        return path if isinstance(path, str) else None

    def _visual_assessment(self, args: Dict[str, Any], claim: str) -> str:
        path = self._image_path(args)
        if not self.describe or not path:
            return "not assessed"
        try:
            with open(path, "rb") as handle:
                b64 = base64.b64encode(handle.read()).decode("utf-8")
            message = HumanMessage(content=[
                {"type": "text", "text": f"Tool claim to validate: {claim}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ])
            response = self.model.invoke([SystemMessage(content=_DESCRIBE_SYSTEM), message])
            return " ".join(str(response.content).split())
        except Exception as exc:  # never let validation break the tool loop
            return f"not assessed ({type(exc).__name__})"

    # ---------------------------------------------------------------- entry point

    # Findings this validator can recognise in a question. Maps a canonical name to
    # the substrings that identify it in either a question or a classifier label.
    FINDING_ALIASES = {
        "pneumothorax": ("pneumothorax",),
        "pleural effusion": ("pleural effusion", "effusion"),
        "cardiomegaly": ("cardiomegaly", "enlarged heart", "cardiac silhouette",
                         "cardiomediastinal silhouette", "heart size"),
        "pulmonary edema": ("pulmonary edema", "edema"),
        "nodule": ("nodule",),
        "mass": ("mass",),
        "lung opacity": ("lung opacity", "opacity"),
        "consolidation": ("consolidation",),
        "atelectasis": ("atelectasis",),
        "pneumonia": ("pneumonia",),
        "fracture": ("fracture",),
        "emphysema": ("emphysema",),
        "fibrosis": ("fibrosis",),
        "hernia": ("hernia",),
        "infiltration": ("infiltration",),
        "pleural thickening": ("pleural thickening",),
        "lung lesion": ("lung lesion",),
    }

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(str(text).lower().replace("_", " ").split())

    @classmethod
    def infer_focus(cls, args: Dict[str, Any], fallback_text: str = "") -> Optional[str]:
        """Which finding is this call about? Prefer the tool's own prompt, else the
        user's question. Returns a canonical finding name, or None if unrecognised."""
        for source in (args.get("prompt", ""), fallback_text):
            haystack = cls._normalise(source)
            if not haystack:
                continue
            hits = [(name, alias) for name, aliases in cls.FINDING_ALIASES.items()
                    for alias in aliases if alias in haystack]
            if hits:
                # longest alias wins: "pleural effusion" beats "effusion"
                return max(hits, key=lambda pair: len(pair[1]))[0]
        return None

    @classmethod
    def _finding_for_label(cls, label: str) -> Optional[str]:
        """Which measured finding does this probability label itself refer to?

        PATCH: the reverse of _is_relevant, and the fix for a silent, total failure.
        When the user asks an open-ended question ("identify any abnormalities") no
        focus can be inferred, and every value used to be scored against focus=None --
        which matches nothing in RELIABILITY, so all 18 classifier readings fell back
        to the guessed 0.40-0.60 band and the whole measured table was bypassed.

        Observed on a real run: Atelectasis=0.5656 was discarded as "too close to call"
        when its measured row (threshold 0.40, interval 0.35-0.55) makes it a clear YES
        at margin 0.28. The label alone is enough to know which row applies; the user's
        intent was never needed for it.
        """
        normalised = cls._normalise(label)
        hits = [(name, alias) for name, aliases in cls.FINDING_ALIASES.items()
                for alias in aliases if alias in normalised or normalised in alias]
        if not hits:
            return None
        # longest alias wins, so "pleural effusion" beats bare "effusion"
        return max(hits, key=lambda pair: len(pair[1]))[0]

    @classmethod
    def _is_relevant(cls, label: str, focus: Optional[str]) -> bool:
        """Does this probability bear on the finding being asked about?"""
        if focus is None:
            return True  # nothing to filter against
        normalised = cls._normalise(label)
        if normalised.startswith("p(yes)"):
            return True  # the VQA tool was asked the question directly
        for alias in cls.FINDING_ALIASES.get(focus, ()):
            if alias in normalised or normalised in alias:
                return True
        return False

    # Stock negations. A generated report says "no pleural effusion or pneumothorax"
    # in almost every normal study; it is template filler, not an observation about
    # this image. Measured: that sentence overrode a specialist reporting 0.995 on two
    # separate true-positive pneumothorax cases, because the prompt's instruction to
    # discount it was ignored. It is computed here instead of requested.
    _NEGATION_CUES = ("no ", "without ", "no evidence of ", "free of ", "negative for ",
                      "absence of ", "not identified", "no definite")
    # Negations that FOLLOW the finding. "the cardiomediastinal silhouette is normal"
    # denies cardiomegaly without using any of the cues above, and a backward-only scan
    # missed it -- the report then read as asserting the finding its own FINDINGS
    # section had denied.
    _NORMALITY_CUES = ("is normal", "are normal", "is unremarkable", "are unremarkable",
                       "within normal limits", "is within normal", "normal in size",
                       "is not enlarged", "are not enlarged")

    @classmethod
    def _text_stance(cls, text: str, focus: Optional[str]) -> Optional[Dict[str, Any]]:
        """Does this report assert or negate the finding? None if it is not mentioned."""
        if not focus:
            return None
        body = cls._normalise(text)
        aliases = cls.FINDING_ALIASES.get(focus, (focus,))

        # PATCH: scan EVERY mention, not just the first. The report generator's FINDINGS
        # and IMPRESSION come from two separate models that never see each other, so one
        # report can deny a finding in FINDINGS and confirm it in IMPRESSION. Returning
        # on the first hit always reported the FINDINGS stance and silently discarded
        # the other -- on effusion_CXR2046 it called the report a negation when the
        # IMPRESSION said "small right pleural effusion".
        hits, spans = [], []
        for alias in aliases:
            start = 0
            while True:
                index = body.find(alias, start)
                if index < 0:
                    break
                # Confine negation detection to the same clause. A fixed-width window
                # reached back into neighbouring sentences: on effusion_CXR2046 the "no"
                # in "no evidence of pneumonia." was read as negating the effusion
                # asserted in the next clause ("2. small right pleural effusion").
                window_start = max(0, index - 60)
                clause_start = max(body.rfind(ch, window_start, index) + 1
                                   for ch in ".;:")
                before = body[max(clause_start, window_start):index]
                clause_end = min([e for e in (body.find(ch, index) for ch in ".;:")
                                  if e >= 0] or [len(body)])
                after = body[index + len(alias):clause_end]
                negated = (any(cue in before for cue in cls._NEGATION_CUES)
                           or any(cue in after for cue in cls._NORMALITY_CUES))
                words = body[window_start:index + len(alias) + 20].split()
                if window_start > 0 and len(words) > 1:
                    words = words[1:]
                span = (index, index + len(alias))
                # Skip a match that overlaps one already recorded, so "effusion" does
                # not double-count inside "pleural effusion". Do NOT stop after the
                # first alias: a report can deny the finding under one name
                # ("the cardiomediastinal silhouette is normal") and assert it under
                # another ("mild cardiomegaly"), and breaking early hid that.
                if not any(span[0] < e and s < span[1] for s, e in spans):
                    spans.append(span)
                    hits.append({"alias": alias, "negated": negated,
                                 "quote": " ".join(words).strip()})
                start = index + len(alias)

        if not hits:
            return {"mentioned": None, "stance": "SILENT", "quote": ""}

        negations = [h for h in hits if h["negated"]]
        assertions = [h for h in hits if not h["negated"]]
        if negations and assertions:
            return {"mentioned": hits[0]["alias"], "stance": "CONTRADICTS",
                    "quote": f"negates: \"{negations[0]['quote']}\" | "
                             f"asserts: \"{assertions[0]['quote']}\""}
        chosen = (negations or assertions)[0]
        return {"mentioned": chosen["alias"],
                "stance": "NEGATES" if chosen["negated"] else "ASSERTS",
                "quote": chosen["quote"]}

    def _ground(self, path: Optional[str], focus: Optional[str]) -> List[Dict[str, Any]]:
        """Ask CheXagent to localise the finding. Cached per (image, finding) so the
        same question is not re-asked once per tool in a turn."""
        if not (self.grounder and path and focus):
            return []
        key = (path, focus)
        if key in self._grounding_cache:
            # Already localised for this image and finding in this turn. Return it for
            # the record, but flag it so the model-facing block does not repeat the same
            # coordinates once per tool.
            self._grounding_repeat = True
            return self._grounding_cache[key]
        self._grounding_repeat = False
        try:
            out, _ = self.grounder._run(
                image_paths=[path],
                prompt=self.GROUNDING_PROMPT.format(finding=focus),
                max_new_tokens=96,
            )
            regions = out.get("regions", []) or []
        except Exception:
            regions = []
        self._grounding_cache[key] = regions
        return regions

    def assess(self, call: Dict[str, Any], result: Any,
               focus: Optional[str] = None) -> Dict[str, Any]:
        """Build the validation record for one tool call. Always runs; cannot be skipped.

        Returns a dict so the same assessment can be rendered two ways: a compact block
        for the model, and a readable block for the log.
        """
        name = call.get("name", "unknown_tool")
        args = call.get("args", {}) or {}
        asked = str((args or {}).get("prompt", ""))
        yes_no = self._is_yes_no_prompt(asked)
        probs = self._probabilities(result, yes_no=yes_no)
        payload = result[0] if isinstance(result, tuple) and result else result
        claim = str(payload)[:2000]
        # Say so when a confidence was discarded. Silently dropping it makes the tool
        # look like one that reported no number, rather than one that was asked wrongly.
        dropped_confidence = (
            not yes_no and isinstance(payload, dict)
            and self._as_probability(payload.get("confidence")) is not None
        )

        # PATCH: only probabilities bearing on the finding in question may drive the
        # ceiling. Previously the max margin over all 18 classifier outputs was used,
        # so an irrelevant Cardiomegaly=0.006 forced a High ceiling on a pneumothorax
        # question -- which suppressed the hedging that the un-validated run produced.
        relevant = [(l, p) for l, p in probs if self._is_relevant(l, focus)]
        scored = []
        for label, p in relevant:
            # Score against the finding the question is about; failing that, against
            # the finding this value is itself reporting on. Only fall through to the
            # assumed band when neither is a pair we have measured.
            scored_as = focus or self._finding_for_label(label)
            entry = self._classify(p, name, scored_as)
            entry.update({"label": label, "value": round(p, 4), "scored_as": scored_as})
            scored.append(entry)
        informative = [(s["label"], s["value"]) for s in scored if s["informative"]]
        uninformative = [(s["label"], s["value"]) for s in scored if not s["informative"]]

        # PATCH: when the validator does not run its own visual pass, say nothing about
        # it being "disabled". The Director reads that as "I cannot look" and stops
        # reporting visible signs -- but it can see the image perfectly well, since
        # interface.py sends it as base64 on every message. None here means "the
        # Director does this itself", not "nobody does it".
        supports = self._visual_assessment(args, claim) if self.describe else None

        # Refuting evidence that can be computed without a model: this tool's own
        # numbers pointing the other way, and any value that is really a coin flip.
        # PATCH: a probability too close to its decision point is ABSENCE of evidence,
        # not evidence against. Listing it as refuting pushed the Director toward
        # negative verdicts on exactly the borderline cases where it should stay neutral.
        refuting = []
        uninformative_notes = [
            f"{s['label']}={s['value']:.3f} falls inside the uncertainty around this "
            f"tool's decision point for {s.get('scored_as') or 'this finding'} "
            + (f"(measured 95% interval {s['thr_ci'][0]:.2f}-{s['thr_ci'][1]:.2f})"
               if s.get("thr_ci") else
               f"(ASSUMED {ASSUMED_DEAD_ZONE[0]}-{ASSUMED_DEAD_ZONE[1]}; this tool has "
               "never been measured for this finding)")
            + ": which side of the line it falls on is not settled by the data, so do "
              "not count it as a vote in either direction"
            for s in scored if not s["informative"]
        ]
        if dropped_confidence:
            uninformative_notes.insert(0, (
                "this tool returned a 'confidence' value, but that number is P(the "
                f"answer is \"Yes\") and the prompt asked was {asked!r} -- a question "
                "with no yes/no answer, so the value does not mean what its name says "
                "and has been discarded rather than counted as a vote. To obtain a "
                "usable probability, ask it 'Does this chest X-ray contain a FINDING?'"
            ))
        stance = None
        claims: List[Dict[str, Any]] = []
        if not probs and isinstance(payload, str):
            # PATCH: with no focus, scan every finding rather than none. _text_stance
            # returns None the moment focus is None, so on an open-ended question the
            # report generator -- the weakest of the three tools, and the one the
            # Director leans on hardest -- passed through entirely unvalidated: no
            # self-contradiction check, no boilerplate-negation flag, no measured
            # precision. Observed doing exactly that on a film labelled normal.
            for target in ([focus] if focus else list(self.FINDING_ALIASES)):
                found = self._text_stance(payload, target)
                if found and found["stance"] != "SILENT":
                    claims.append(dict(found, finding=target,
                                       reliability=REPORT_RELIABILITY.get(target)))
            stance = next((c for c in claims if c["finding"] == focus), None)

            # Negations are collapsed into one entry. A normal-template report negates
            # five or six findings in a single sentence, and repeating the same warning
            # per finding buried the one line that matters -- the measured precision of
            # whatever it actually asserted -- under a wall of identical paragraphs.
            negated = [c["finding"] for c in claims if c["stance"] == "NEGATES"]
            if negated:
                refuting.insert(0, (
                    f"this tool NEGATES {', '.join(negated)} in text, with no probability "
                    "attached to any of them and no way to validate them. Stock negations "
                    "of exactly this form appear in most generated reports regardless of "
                    "the image: it is normal-template wording, not an observation about "
                    "this study. None of them may outweigh a specialist reporting a high "
                    "probability"))
            for claim_made in claims:
                if claim_made["stance"] == "CONTRADICTS":
                    refuting.insert(0, (
                        f"this tool CONTRADICTS ITSELF about {claim_made['finding']} "
                        f"({claim_made['quote']}). Its FINDINGS and IMPRESSION sections "
                        "are generated by two separate models that never see each "
                        "other's output. A self-contradicting report is evidence that "
                        "this tool is unreliable here, not evidence about the image; do "
                        "not treat either half as decisive"))

        # fallback last, so it never sits beside a computed item
        if not refuting:
            refuting.append("none computed from this tool's output")

        return {
            "tool": name,
            "text_stance": stance,
            "text_claims": claims,
            "grounded_regions": self._ground(self._image_path(args), focus),
            "grounding_repeat": self._grounding_repeat,
            "grounding_attempted": bool(self.grounder and focus
                                        and self._image_path(args)),
            "args": args,
            "raw_output": claim,
            "conclusion": self._conclusion(
                name, payload, [s for s in scored if s["informative"]], relevant),
            # PATCH: score a probability against the finding's threshold only when it
            # actually bears on that finding. Previously every value was scored against
            # the focus threshold, so on a pleural-effusion question the classifier's
            # Cardiomegaly reading was judged against the effusion decision point.
            "probabilities": [
                (dict(self._classify(p, name, focus or self._finding_for_label(l)),
                      label=l, value=round(p, 4), relevant=True,
                      scored_as=focus or self._finding_for_label(l))
                 if self._is_relevant(l, focus) else
                 {"label": l, "value": round(p, 4), "relevant": False,
                  "informative": False, "supports": None, "margin": 0.0,
                  "threshold": None, "thr_ci": None, "auc": None,
                  "measured": False, "scored_as": None})
                for l, p in probs
            ],
            "report_reliability": REPORT_RELIABILITY.get(focus or ""),
            "supportive_evidence": supports,
            "refuting_evidence": refuting,
            "uninformative_notes": uninformative_notes,
            "focus": focus,
            # With a focus, one ceiling for that finding. Without, the scalar is capped
            # by whatever the tool actually asserts present -- an affirmative conclusion
            # is the risky direction -- and the per-finding map carries the detail.
            "confidence_ceiling": self._ceiling_from_scored(
                scored if focus else
                [s for s in scored if s["supports"] == "YES"] or scored),
            "ceilings_by_finding": None if focus else self._ceilings_by_finding(scored),
        }

    @staticmethod
    def _conclusion(name: str, payload: Any, informative: List[Dict[str, Any]],
                    probs_all: List[Tuple[str, float]]) -> str:
        """One-line restatement of what this tool actually claimed.

        PATCH: "present" now means the value cleared its own measured decision point,
        not that it exceeded 0.5. The last place in the scoring path still assuming
        0.5 -- with the classifier's edema threshold at 0.15, a reading of 0.45 is a
        clear positive that this line used to omit, and the summary then disagreed
        with the per-value verdicts printed directly beneath it.
        """
        if isinstance(payload, dict) and "response" in payload:
            answer = str(payload["response"]).strip()
            if informative:
                first = informative[0]
                return f"{name} answered {answer!r} ({first['label']}={first['value']:.3f})"
            return f"{name} answered {answer!r}"
        if informative:
            # A multi-label classifier has no single "conclusion"; picking the value
            # furthest from its threshold just surfaces whatever is most confidently
            # absent. Report what it calls present, and how much of it is undecided.
            positives = sorted((s for s in informative if s["supports"] == "YES"),
                               key=lambda s: -s["value"])
            undecided = len(probs_all) - len(informative)
            suffix = (f"; {undecided} of {len(probs_all)} relevant value(s) undecided"
                      if undecided else "")
            if positives:
                # PATCH: carry the weight in the headline, not only in the detail lines.
                # This line read "reports present: Atelectasis=0.57, Pneumothorax=0.51"
                # and the Director turned it into a two-finding diagnosis -- while the
                # pneumothorax vote came from a tool measured at AUC 0.62, barely above
                # chance. A bare list of numbers reads as a list of findings.
                listed = ", ".join(
                    f"{s['label']}={s['value']:.2f} ({EvidenceValidator._strength_tier(s)}"
                    + (f", auc {s['auc']:.2f} where 0.5 is chance)" if s.get("auc")
                       else ", this tool is unmeasured here)")
                    for s in positives[:4])
                more = f" (+{len(positives) - 4} more)" if len(positives) > 4 else ""
                weak = [s["label"] for s in positives[:4]
                        if EvidenceValidator._strength_tier(s) == "weak"]
                caveat = (f"; {', '.join(weak)} rest on weak evidence and should be "
                          "reported as uncertain or not at all, never as findings"
                          if weak else "")
                return f"{name} reports present: {listed}{more}{suffix}{caveat}"
            return (f"{name} reports nothing above its measured decision point for "
                    f"this finding{suffix}")
        if probs_all:
            # Relevant values exist, but all sit too near their decision points to call.
            listed = ", ".join(f"{l}={p:.3f}" for l, p in probs_all[:3])
            return f"{name} is undecided on this finding ({listed})"
        text = " ".join(str(payload).split())
        return f"{name} returned text, no probability: {text[:160]}"

    @staticmethod
    def render_for_model(record: Dict[str, Any]) -> str:
        """Compact block injected into the model's next turn."""
        focus = record.get("focus")
        header = f" finding=\"{focus}\"" if focus else ""
        lines = [f"<validation tool=\"{record['tool']}\"{header}>",
                 "Computed from the tool output (not a model judgement):"]
        # Show only probabilities bearing on the finding asked about. Listing all 18
        # classifier outputs buries the relevant one in noise.
        shown = [pr for pr in record["probabilities"] if pr.get("relevant", True)]
        hidden = len(record["probabilities"]) - len(shown)
        if shown:
            for pr in shown:
                ci = pr.get("thr_ci")
                basis = (f"measured threshold {pr['threshold']:.2f} "
                         f"(95% {ci[0]:.2f}-{ci[1]:.2f}), AUC {pr['auc']:.2f}"
                         if pr.get("measured") and ci else
                         f"measured threshold {pr['threshold']:.2f}, AUC {pr['auc']:.2f}"
                         if pr.get("measured") else
                         f"threshold {pr['threshold']:.2f} ASSUMED - this tool has not "
                         "been measured for this finding")
                verdict = (f"supports {pr['supports']} (margin {pr['margin']:.2f}; {basis})"
                           if pr["informative"] else
                           f"TOO CLOSE TO CALL - inside the decision point's own "
                           f"uncertainty ({basis}), so it is not a vote either way")
                # The number that answers "how much is THIS claim worth" -- which AUC
                # does not. Stated absolutely, because a lift over a low base rate can
                # look impressive while the claim is still close to a coin flip.
                pol = pr.get("polarity")
                if pol and pr["informative"]:
                    stat = pol.get("ppv", pol.get("npv"))
                    verdict += (f". When this tool answers {pr['supports']} for this "
                                f"finding it is correct {stat:.0%} of the time "
                                f"({pol['n']} such answers in 544 films)")
                lines.append(f"  {pr['label']} = {pr['value']:.3f} -> {verdict}")
        else:
            lines.append("  this tool reported no probability about "
                         + (focus or "the finding in question"))
        if hidden:
            lines.append(f"  ({hidden} other value(s) omitted: unrelated to {focus})")
        stance = record.get("text_stance")
        if stance and stance["stance"] != "SILENT":
            lines.append(f"  this tool's text {stance['stance']} {record.get('focus')}: "
                         f"\"{stance['quote']}\"")
            if stance["stance"] == "CONTRADICTS":
                lines.append("  -> SELF-CONTRADICTORY: FINDINGS and IMPRESSION come from two "
                             "separate models and disagree. Treat this tool as unreliable "
                             "for this finding rather than as evidence either way.")
            elif stance["stance"] == "NEGATES":
                lines.append("  -> UNVALIDATED NEGATION: no probability accompanies it, and "
                             "this phrasing is boilerplate in most generated reports. It must "
                             "not outweigh a specialist reporting a high probability.")
        lines.append(f"  confidence ceiling: {record['confidence_ceiling']} "
                     f"(you may report lower, never higher)")
        by_finding = record.get("ceilings_by_finding") or {}
        if len(by_finding) > 1:
            lines.append("  no single finding was in question, so that ceiling is per "
                         "finding. A ceiling earned by one finding does NOT license the "
                         "same confidence about another -- match the ceiling to whatever "
                         "you actually conclude:")
            for finding, tier in sorted(by_finding.items(), key=lambda kv: kv[0]):
                lines.append(f"    {finding}: {tier}")
        regions = record.get("grounded_regions") or []
        if regions and record.get("grounding_repeat"):
            lines.append(f"Localisation for {record.get('focus')} was reported above; "
                         "it is one result for this image, not separate corroboration "
                         "from each tool.")
        elif regions:
            lines.append(f"Localised by chest_xray_expert (radiology-trained, boxes are "
                         f"percentages of image width/height):")
            for region in regions:
                x1, y1, x2, y2 = region["box_pct"]
                lines.append(f"  {region['label']}: ({x1},{y1})-({x2},{y2}), "
                             f"{region['side_if_frontal']} on a frontal view")
            # PATCH: measured on 84 image x finding pairs, 28 images. Grounding is NOT
            # independent corroboration -- it is the same model's localisation head,
            # and it tracks the binary head at a lower threshold (17% of pairs grounded
            # below P(yes)=0.2, 91% between 0.2 and 0.5, 100% above). It found every
            # true positive including two the binary head missed, but drew a box on 31%
            # of images that did not have the finding. Describing it as corroboration
            # would double-count one model's opinion as two.
            lines.append("  Cite this location in your supportive evidence, naming it as "
                         "the localisation from chest_xray_expert -- it is the only "
                         "localised evidence available from a radiology-trained model.")
            lines.append("  CAUTION: it is the same model's localisation head, not a "
                         "second opinion. Measured on this dataset it draws a box on 31% "
                         "of images that do NOT have the finding, and grounds almost "
                         "anything the binary head scores above 0.2. So it tells you WHERE "
                         "the finding would be, not THAT it is present: do not count it as "
                         "a separate agreeing tool or let it raise your confidence.")
        elif record.get("grounding_attempted"):
            lines.append(f"chest_xray_expert was asked to localise {record.get('focus')} "
                         "and returned no region. On this dataset an absent localisation "
                         "was a stronger negative signal than a present one is a positive: "
                         "it grounded 19 of 19 true positives, so failing to ground weighs "
                         "against the finding, though it remains one model's opinion.")
        # PATCH: driven by every claim the text makes, not only the focus finding.
        # Keyed on focus alone, this said nothing at all on an open-ended question --
        # so the Director was never told that the tool it was quoting is the weak one.
        claims = record.get("text_claims") or []
        if claims and record["tool"] == "chest_xray_report_generator":
            lines.append("  what this report's text claims, and how reliable this tool "
                         "has been measured to be for each. It is the weakest of the "
                         "three tools; weight its wording accordingly:")
            for claim in claims:
                rel = claim.get("reliability")
                if not rel:
                    basis = "never measured for this finding"
                elif claim["stance"] == "ASSERTS":
                    # For a claim, what matters is how often such claims are right.
                    basis = (f"of the reports that assert it, only {rel['precision']:.0%} "
                             "are correct")
                else:
                    # For a denial, what matters is how much this tool misses. Quoting
                    # precision here would describe assertions it did not make.
                    basis = (f"this tool detects only {rel['recall']:.0%} of true cases, "
                             f"so a denial from it misses roughly {1 - rel['recall']:.0%}")
                lines.append(f"    {claim['stance']} {claim['finding']} -- {basis}")
            if any(c["finding"] == "atelectasis" and c["stance"] == "ASSERTS"
                   for c in claims):
                lines.append("    note: across 544 reports this tool never once negated "
                             "atelectasis -- it asserts it or stays silent. Its assertion "
                             "of it is therefore close to a coin flip, and carries no "
                             "weight from the fact that it was volunteered.")
        if record.get("supportive_evidence"):
            lines.append(f"Visual assessment by validator: {record['supportive_evidence']}")
        else:
            lines.append("Visual assessment: not performed by this validator. You can see "
                         "the X-ray yourself. Report ONLY signs you actually observe in this "
                         "image, with their location. If you cannot identify any, write "
                         "'none visible' -- do not describe what such signs would look like "
                         "in general, and do not say that visual assessment is needed.")
            # PATCH: "none visible" from a generalist reader is not evidence against a
            # specialist. Without this the Director downgraded correct findings to Low
            # purely because it could not see a mild or subtle sign itself, which is
            # expected: it is not trained on radiology and the specialists are.
            lines.append("You ARE a multimodal model and the radiograph is attached to "
                         "this conversation. Do not answer that you rely on tool outputs "
                         "rather than visual assessment, or that you cannot assess images "
                         "-- look, and report what you see.")
            lines.append("If you genuinely see nothing, write 'none visible'. That is a "
                         "limit of your own general vision, NOT evidence against the "
                         "finding: you are not radiology-trained and the specialist tools "
                         "are. Do not lower your confidence or change your conclusion "
                         "because you personally could not see a subtle sign.")
        lines.append("</validation>")
        return "\n".join(lines)

    @staticmethod
    def render_for_log(record: Dict[str, Any]) -> str:
        """Human-readable block written to logs/session_*.log."""
        lines = [f"  TOOL: {record['tool']}",
                 f"  ARGS: {record['args']}",
                 f"  RAW OUTPUT: {record['raw_output']}",
                 f"  CONCLUSION: {record['conclusion']}"]
        if record.get("focus"):
            lines.append(f"  FINDING IN QUESTION: {record['focus']}")
        if record["probabilities"]:
            lines.append("  PROBABILITIES (* = bears on the finding in question):")
            skipped = 0
            for pr in record["probabilities"]:
                if not pr.get("relevant", True):
                    skipped += 1
                    continue
                ci = pr.get("thr_ci")
                if pr["informative"]:
                    tag = f"supports {pr['supports']}, margin {pr['margin']:.2f}"
                else:
                    tag = "TOO CLOSE TO CALL, inside the threshold's own interval"
                if not pr.get("relevant", True):
                    tag, basis = "not related to this finding", "not scored"
                elif pr.get("measured"):
                    span = f" 95% {ci[0]:.2f}-{ci[1]:.2f}" if ci else ""
                    basis = f"thr {pr['threshold']:.2f}{span} auc {pr['auc']:.2f}"
                else:
                    basis = f"thr {pr['threshold']:.2f} UNMEASURED"
                lines.append(f"    * {pr['label']} = {pr['value']:.4f}  [{tag}; {basis}]")
            if skipped:
                lines.append(f"      ({skipped} other value(s) not related to "
                             f"{record.get('focus')}, not scored)")
        else:
            lines.append("  PROBABILITIES: none reported by this tool")
        for claim in record.get("text_claims") or []:
            rel = claim.get("reliability")
            basis = (f" [measured precision {rel['precision']:.0%}, "
                     f"recall {rel['recall']:.0%}]" if rel else " [unmeasured]")
            lines.append(f"  TEXT STANCE: {claim['stance']} {claim['finding']}{basis} "
                         f"-> \"{claim['quote']}\"")
        # PATCH: localisation is one cached result per image+finding. render_for_model
        # was de-duplicated but this was not, so the console still printed the same
        # coordinates under every tool, reading as three separate confirmations.
        if record.get("grounded_regions") and record.get("grounding_repeat"):
            lines.append("  GROUNDED REGION: (same result as above, not re-queried)")
        else:
            for region in record.get("grounded_regions") or []:
                x1, y1, x2, y2 = region["box_pct"]
                lines.append(f"  GROUNDED REGION: {region['label']} at ({x1},{y1})-({x2},{y2}) "
                             f"pct, {region['side_if_frontal']} on a frontal view")
        lines.append("  SUPPORTIVE EVIDENCE: " + (
            record["supportive_evidence"] or "(validator did not assess; Director reports "
                                             "this from the image itself)"))
        lines.append("  REFUTING EVIDENCE:")
        for item in record["refuting_evidence"]:
            lines.append(f"      {item}")
        if record.get("uninformative_notes"):
            lines.append("  NO OPINION (absence of evidence, not evidence against):")
            for item in record["uninformative_notes"]:
                lines.append(f"      {item}")
        lines.append(f"  CONFIDENCE (computed ceiling): {record['confidence_ceiling']}")
        by_finding = record.get("ceilings_by_finding") or {}
        if len(by_finding) > 1:
            listed = ", ".join(f"{f}={t}" for f, t in sorted(by_finding.items()))
            lines.append(f"  CEILING PER FINDING: {listed}")
        return "\n".join(lines)

    def validate(self, call: Dict[str, Any], result: Any) -> str:
        """Back-compatible helper: assess and render for the model."""
        return self.render_for_model(self.assess(call, result))
