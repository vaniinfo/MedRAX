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

# A probability inside this band expresses no opinion. Treating it as a vote is what
# made three tools outvote a correct specialist on every pneumothorax case.
DEAD_ZONE: Tuple[float, float] = (0.40, 0.60)

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

    def __init__(self, model: Optional[BaseLanguageModel] = None, describe: bool = True):
        """
        Args:
            model: a vision-capable chat model used only for the descriptive half.
            describe: set False to skip the LLM call entirely (deterministic only).
        """
        self.model = model
        self.describe = describe and model is not None

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
    def _is_uninformative(p: float) -> bool:
        return DEAD_ZONE[0] <= p <= DEAD_ZONE[1]

    @classmethod
    def _ceiling(cls, probs: List[Tuple[str, float]]) -> str:
        """Confidence ceiling, computed -- never the model's self-assessment."""
        live = [p for _, p in probs if not cls._is_uninformative(p)]
        if not probs:
            return "Medium"  # no numbers reported; nothing to cap on
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

    def assess(self, call: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """Build the validation record for one tool call. Always runs; cannot be skipped.

        Returns a dict so the same assessment can be rendered two ways: a compact block
        for the model, and a readable block for the log.
        """
        name = call.get("name", "unknown_tool")
        args = call.get("args", {}) or {}
        probs = self._probabilities(result)
        payload = result[0] if isinstance(result, tuple) and result else result
        claim = str(payload)[:400]

        informative, uninformative = [], []
        for label, p in probs:
            (uninformative if self._is_uninformative(p) else informative).append((label, p))

        supports = self._visual_assessment(args, claim) if self.describe else (
            "not assessed (LLM visual assessment disabled; set MEDRAX_VALIDATE_DESCRIBE=1)"
        )

        # Refuting evidence that can be computed without a model: this tool's own
        # numbers pointing the other way, and any value that is really a coin flip.
        refuting = []
        for label, p in uninformative:
            refuting.append(f"{label}={p:.3f} is inside the {DEAD_ZONE[0]}-{DEAD_ZONE[1]} "
                            "dead zone: no opinion, must not be counted as a vote")
        if not refuting:
            refuting.append("none computed from this tool's output")

        return {
            "tool": name,
            "args": args,
            "raw_output": claim,
            "conclusion": self._conclusion(name, payload, informative, probs),
            "probabilities": [{"label": l, "value": round(p, 4),
                               "informative": not self._is_uninformative(p),
                               "supports": None if self._is_uninformative(p) else ("YES" if p > 0.5 else "NO")}
                              for l, p in probs],
            "supportive_evidence": supports,
            "refuting_evidence": refuting,
            "confidence_ceiling": self._ceiling(probs),
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
            undecided = sum(1 for _, p in probs_all if EvidenceValidator._is_uninformative(p))
            if positives:
                listed = ", ".join(f"{l}={p:.2f}" for l, p in positives[:4])
                more = f" (+{len(positives) - 4} more)" if len(positives) > 4 else ""
                return (f"{name} reports present: {listed}{more}; "
                        f"{undecided} value(s) in the dead zone")
            return (f"{name} reports nothing above 0.50; "
                    f"{undecided} value(s) in the dead zone")
        text = " ".join(str(payload).split())
        return f"{name} returned text, no probability: {text[:160]}"

    @staticmethod
    def render_for_model(record: Dict[str, Any]) -> str:
        """Compact block injected into the model's next turn."""
        lines = [f"<validation tool=\"{record['tool']}\">",
                 "Computed from the tool output (not a model judgement):"]
        if record["probabilities"]:
            for pr in record["probabilities"]:
                verdict = (f"informative, supports {pr['supports']}" if pr["informative"]
                           else "UNINFORMATIVE - near 0.5, expresses no opinion, "
                                "do not count as a vote")
                lines.append(f"  {pr['label']} = {pr['value']:.3f} -> {verdict}")
        else:
            lines.append("  no probabilities reported by this tool")
        lines.append(f"  confidence ceiling: {record['confidence_ceiling']} "
                     f"(you may report lower, never higher)")
        lines.append(f"Visual assessment of the image: {record['supportive_evidence']}")
        lines.append("</validation>")
        return "\n".join(lines)

    @staticmethod
    def render_for_log(record: Dict[str, Any]) -> str:
        """Human-readable block written to logs/session_*.log."""
        lines = [f"  TOOL: {record['tool']}",
                 f"  ARGS: {record['args']}",
                 f"  RAW OUTPUT: {record['raw_output'][:300]}",
                 f"  CONCLUSION: {record['conclusion']}"]
        if record["probabilities"]:
            lines.append("  PROBABILITIES:")
            for pr in record["probabilities"]:
                tag = f"supports {pr['supports']}" if pr["informative"] else "UNINFORMATIVE (dead zone)"
                lines.append(f"      {pr['label']} = {pr['value']:.4f}  [{tag}]")
        else:
            lines.append("  PROBABILITIES: none reported by this tool")
        lines.append(f"  SUPPORTIVE EVIDENCE: {record['supportive_evidence']}")
        lines.append("  REFUTING EVIDENCE:")
        for item in record["refuting_evidence"]:
            lines.append(f"      {item}")
        lines.append(f"  CONFIDENCE (computed ceiling): {record['confidence_ceiling']}")
        return "\n".join(lines)

    def validate(self, call: Dict[str, Any], result: Any) -> str:
        """Back-compatible helper: assess and render for the model."""
        return self.render_for_model(self.assess(call, result))
