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
RELIABILITY: Dict[Tuple[str, str], Dict[str, float]] = {
    #                                                                            thr 95% CI
    ("chest_xray_expert", "cardiomegaly"):         {"auc": 0.909, "threshold": 0.45},  # .45-.60
    ("chest_xray_expert", "pleural effusion"):     {"auc": 0.951, "threshold": 0.70},  # .50-.80
    ("chest_xray_expert", "pneumothorax"):         {"auc": 0.948, "threshold": 0.25},  # .10-.60
    ("chest_xray_expert", "consolidation"):        {"auc": 0.810, "threshold": 0.25},  # .15-.40
    ("chest_xray_expert", "pulmonary edema"):      {"auc": 0.897, "threshold": 0.50},  # .20-.75
    ("chest_xray_expert", "atelectasis"):          {"auc": 0.819, "threshold": 0.40},  # .35-.65
    ("chest_xray_classifier", "cardiomegaly"):     {"auc": 0.852, "threshold": 0.45},  # .25-.55
    ("chest_xray_classifier", "pleural effusion"): {"auc": 0.887, "threshold": 0.50},  # .40-.60
    # Near chance, and the finding this whole code path was built around. Its AUC interval
    # is 0.510-0.728: it clears chance by a hundredth. Kept rather than dropped because
    # dropping it would make the pair ASSUMED at 0.5, which claims more than this does --
    # and the AUC weighting in _ceiling_from_scored already discounts 0.622 to nearly
    # nothing. This is the measurement behind the original complaint that three tools
    # outvoted a correct specialist on pneumothorax.
    ("chest_xray_classifier", "pneumothorax"):     {"auc": 0.622, "threshold": 0.35},  # .05-.50
    ("chest_xray_classifier", "consolidation"):    {"auc": 0.762, "threshold": 0.50},  # .50-.50
    ("chest_xray_classifier", "pulmonary edema"):  {"auc": 0.807, "threshold": 0.15},  # .05-.40
    ("chest_xray_classifier", "atelectasis"):      {"auc": 0.699, "threshold": 0.40},  # .35-.55
}

# CheXagent beats the classifier on five of six findings, paired on the same bootstrap
# resamples. Consolidation is the exception at +0.047 (CI -0.001 to +0.100) -- within
# noise. An earlier claim that it wins on *every* finding was two AUCs compared by eye.
#
#   pneumothorax +0.325   cardiomegaly +0.056   consolidation +0.047 (no separation)
#   atelectasis  +0.118   effusion     +0.064
#   edema        +0.090

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

    @staticmethod
    def _probabilities(result: Any) -> List[Tuple[str, float]]:
        """Pull every probability a tool reported. Tools return (output, metadata)."""
        payload = result[0] if isinstance(result, tuple) and result else result
        if not isinstance(payload, dict):
            return []
        found: List[Tuple[str, float]] = []
        confidence = EvidenceValidator._as_probability(payload.get("confidence"))
        if confidence is not None:
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
        """
        info = cls._reliability(tool, finding)
        threshold = info["threshold"] if info else UNMEASURED_THRESHOLD
        auc = info["auc"] if info else None
        supports_yes = p >= threshold
        margin = ((p - threshold) / (1 - threshold) if supports_yes and threshold < 1
                  else (threshold - p) / threshold if threshold > 0 else 1.0)
        return {"supports": "YES" if supports_yes else "NO",
                "margin": round(margin, 3),
                "informative": margin >= 0.15,
                "threshold": threshold, "auc": auc, "measured": info is not None}

    @classmethod
    def _ceiling_from_scored(cls, scored: List[Dict[str, Any]]) -> str:
        """Ceiling from margin AND discrimination. A wide margin on a tool that barely
        separates this finding is not the same as a wide margin on one that does."""
        live = [s for s in scored if s["informative"]]
        if not scored:
            return "not computable (no probabilities; this tool is unvalidated)"
        if not live:
            return "Low"
        best = max(live, key=lambda s: s["margin"] * ((s["auc"] or 0.6) - 0.5) * 2)
        strength = best["margin"] * ((best["auc"] or 0.6) - 0.5) * 2
        if strength > 0.45:
            return "High"
        return "Medium" if strength > 0.15 else "Low"

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
        probs = self._probabilities(result)
        payload = result[0] if isinstance(result, tuple) and result else result
        claim = str(payload)[:2000]

        # PATCH: only probabilities bearing on the finding in question may drive the
        # ceiling. Previously the max margin over all 18 classifier outputs was used,
        # so an irrelevant Cardiomegaly=0.006 forced a High ceiling on a pneumothorax
        # question -- which suppressed the hedging that the un-validated run produced.
        relevant = [(l, p) for l, p in probs if self._is_relevant(l, focus)]
        scored = []
        for label, p in relevant:
            entry = self._classify(p, name, focus)
            entry.update({"label": label, "value": round(p, 4)})
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
            f"{s['label']}={s['value']:.3f} is only {s['margin']:.2f} from this tool's "
            f"{'measured' if s['measured'] else 'ASSUMED (unmeasured)'} decision point "
            f"of {s['threshold']:.2f} for {focus}: too close to call, do not count it as "
            "a vote in either direction"
            for s in scored if not s["informative"]
        ]
        stance = None
        if not probs and isinstance(payload, str):
            stance = self._text_stance(payload, focus)
            if stance and stance["stance"] == "CONTRADICTS":
                refuting.insert(0, (
                    f"this tool CONTRADICTS ITSELF about {focus} ({stance['quote']}). Its "
                    "FINDINGS and IMPRESSION sections are generated by two separate models "
                    "that never see each other's output. A self-contradicting report is "
                    "evidence that this tool is unreliable here, not evidence about the "
                    "image; do not treat either half as decisive"))
            elif stance and stance["stance"] == "NEGATES":
                refuting.insert(0, (
                    f"this tool NEGATES {focus} (\"{stance['quote']}\") but reports no "
                    "probability and cannot be validated. Stock negations of this form "
                    "appear in most generated reports regardless of the image; do not let "
                    "it outweigh a specialist reporting a high probability"))

        # fallback last, so it never sits beside a computed item
        if not refuting:
            refuting.append("none computed from this tool's output")

        return {
            "tool": name,
            "text_stance": stance,
            "grounded_regions": self._ground(self._image_path(args), focus),
            "grounding_repeat": self._grounding_repeat,
            "grounding_attempted": bool(self.grounder and focus
                                        and self._image_path(args)),
            "args": args,
            "raw_output": claim,
            "conclusion": self._conclusion(name, payload, informative, relevant),
            # PATCH: score a probability against the finding's threshold only when it
            # actually bears on that finding. Previously every value was scored against
            # the focus threshold, so on a pleural-effusion question the classifier's
            # Cardiomegaly reading was judged against the effusion decision point.
            "probabilities": [
                (dict(self._classify(p, name, focus), label=l, value=round(p, 4),
                      relevant=True)
                 if self._is_relevant(l, focus) else
                 {"label": l, "value": round(p, 4), "relevant": False,
                  "informative": False, "supports": None, "margin": 0.0,
                  "threshold": None, "auc": None, "measured": False})
                for l, p in probs
            ],
            "report_reliability": REPORT_RELIABILITY.get(focus or ""),
            "supportive_evidence": supports,
            "refuting_evidence": refuting,
            "uninformative_notes": uninformative_notes,
            "focus": focus,
            "confidence_ceiling": self._ceiling_from_scored(scored),
        }

    @staticmethod
    def _conclusion(name: str, payload: Any, informative: List[Tuple[str, float]],
                    probs_all: List[Tuple[str, float]]) -> str:
        """One-line restatement of what this tool actually claimed."""
        if isinstance(payload, dict) and "response" in payload:
            answer = str(payload["response"]).strip()
            if informative:
                return f"{name} answered {answer!r} ({informative[0][0]}={informative[0][1]:.3f})"
            return f"{name} answered {answer!r}"
        if informative:
            # A multi-label classifier has no single "conclusion"; picking the value
            # furthest from 0.5 just surfaces whatever is most confidently absent.
            # Report what it calls present, and how much of it is undecided.
            positives = sorted(((l, p) for l, p in informative if p > 0.5),
                               key=lambda kv: -kv[1])
            undecided = len(probs_all) - len(informative)
            suffix = (f"; {undecided} of {len(probs_all)} relevant value(s) undecided"
                      if undecided else "")
            if positives:
                listed = ", ".join(f"{l}={p:.2f}" for l, p in positives[:4])
                more = f" (+{len(positives) - 4} more)" if len(positives) > 4 else ""
                return f"{name} reports present: {listed}{more}{suffix}"
            return f"{name} reports nothing above 0.50 for this finding{suffix}"
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
                basis = (f"measured threshold {pr['threshold']:.2f}, AUC {pr['auc']:.2f}"
                         if pr.get("measured") else
                         f"threshold {pr['threshold']:.2f} ASSUMED - this tool has not "
                         "been measured for this finding")
                verdict = (f"supports {pr['supports']} (margin {pr['margin']:.2f}; {basis})"
                           if pr["informative"] else
                           f"TOO CLOSE TO CALL (margin {pr['margin']:.2f}; {basis}) - "
                           "do not count as a vote")
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
        rr = record.get("report_reliability")
        if rr and record["tool"] == "chest_xray_report_generator":
            lines.append(f"  measured reliability of this tool for {record.get('focus')}: "
                         f"recall {rr['recall']:.0%}, precision {rr['precision']:.0%}. "
                         "It is the weakest of the three tools; weight it accordingly.")
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
                if pr["informative"]:
                    tag = f"supports {pr['supports']}, margin {pr['margin']:.2f}"
                else:
                    tag = f"TOO CLOSE TO CALL, margin {pr['margin']:.2f}"
                if not pr.get("relevant", True):
                    tag, basis = "not related to this finding", "not scored"
                elif pr.get("measured"):
                    basis = f"thr {pr['threshold']:.2f} auc {pr['auc']:.2f}"
                else:
                    basis = f"thr {pr['threshold']:.2f} UNMEASURED"
                lines.append(f"    * {pr['label']} = {pr['value']:.4f}  [{tag}; {basis}]")
            if skipped:
                lines.append(f"      ({skipped} other value(s) not related to "
                             f"{record.get('focus')}, not scored)")
        else:
            lines.append("  PROBABILITIES: none reported by this tool")
        stance = record.get("text_stance")
        if stance and stance["stance"] != "SILENT":
            lines.append(f"  TEXT STANCE: {stance['stance']} -> \"{stance['quote']}\"")
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
        return "\n".join(lines)

    def validate(self, call: Dict[str, Any], result: Any) -> str:
        """Back-compatible helper: assess and render for the model."""
        return self.render_for_model(self.assess(call, result))
