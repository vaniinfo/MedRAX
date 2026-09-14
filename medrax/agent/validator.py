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
        if isinstance(payload.get("confidence"), (int, float)):
            answer = str(payload.get("response", "")).strip()
            found.append((f"P(yes) [answered {answer!r}]", float(payload["confidence"])))
        for key, value in payload.items():
            if key != "confidence" and isinstance(value, float) and 0.0 <= value <= 1.0:
                found.append((key, float(value)))
        return found

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

    def validate(self, call: Dict[str, Any], result: Any) -> str:
        """Return a validation block for one tool call. Always runs; cannot be skipped."""
        name = call.get("name", "unknown_tool")
        args = call.get("args", {}) or {}
        probs = self._probabilities(result)

        lines = [f"<validation tool=\"{name}\">", "Computed from the tool output (not a model judgement):"]
        if probs:
            for label, p in probs:
                if self._is_uninformative(p):
                    verdict = "UNINFORMATIVE - near 0.5, expresses no opinion, do not count as a vote"
                else:
                    verdict = f"informative, supports {'YES' if p > 0.5 else 'NO'}"
                lines.append(f"  {label} = {p:.3f} -> {verdict}")
        else:
            lines.append("  no probabilities reported by this tool")
        lines.append(f"  confidence ceiling: {self._ceiling(probs)} "
                     f"(you may report lower, never higher)")

        claim = str(result[0] if isinstance(result, tuple) and result else result)[:400]
        lines.append(f"Visual assessment of the image: {self._visual_assessment(args, claim)}")
        lines.append("</validation>")
        return "\n".join(lines)
