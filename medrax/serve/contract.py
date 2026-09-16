"""The wire contract between a model service and the agent.

One shape for every model, so the agent and the EvidenceValidator do not care which
process, which environment or which machine answered. This is what makes the
dependency conflicts go away: CheXagent needs transformers 4.40 and silently stops
attending to the image on anything newer, MedGemma needs >=4.50, MAIRA-2 needs newer
still. In one process those are irreconcilable -- MAIRA-2 already fails to load for
exactly this reason. Across processes each keeps its own pin and none of them can
break another.

Nothing in medrax/agent changes to support this. A service reply is mapped back into
the payload shape the in-process tools already return, so RELIABILITY keys, thresholds
and scoring are untouched:

    classify -> {"Atelectasis": 0.57, ...}          label -> probability
    vqa      -> {"response": "Yes", "confidence": 0.83}
    report   -> "CHEST X-RAY REPORT ..."            plain text

`tool` on a service is the RELIABILITY key. Two models answering the same kind of
question must NOT share one, or they share a threshold and an AUC -- and the whole
point of the measurement is that they differ: on pneumothorax CheXagent measures
AUC 0.948 against the classifier's 0.622.
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

# Tasks a service can advertise. The task decides how its reply is mapped back to a
# payload, not what the agent is allowed to ask it.
TASKS = ("vqa", "classify", "report")


class PredictReply(BaseModel):
    """What every model service returns from /predict."""

    model: str = Field(..., description="the actual checkpoint, e.g. StanfordAIMI/CheXagent-2-3b")
    task: str = Field(..., description="one of: vqa, classify, report")
    answer: str = Field("", description="free text the model produced")
    # Only meaningful when the question was genuinely yes/no. The validator discards a
    # probability of "Yes" when no yes/no question was asked, so say so honestly here
    # rather than returning a number that cannot mean what its name says.
    p_yes: Optional[float] = Field(None, description="P(the answer is yes), yes/no questions only")
    yes_no: bool = Field(False, description="was this treated as a yes/no question")
    probabilities: Dict[str, float] = Field(default_factory=dict,
                                            description="label -> probability, for classifiers")


class HealthReply(BaseModel):
    """What every model service returns from /health.

    `vision_ok` is not boilerplate. CheXagent on the wrong transformers version loads
    cleanly, produces fluent clinical text, and ignores the image -- no exception, no
    warning. scripts/verify_vqa_vision.py exists because of it. A service that can
    self-test that property must report it, so the agent can refuse a model that is
    confidently describing nothing.
    """

    tool: str
    model: str
    task: str
    ready: bool
    transformers: Optional[str] = None
    vision_ok: Optional[bool] = Field(None, description="None if the service cannot self-test")
    detail: str = ""


def to_payload(reply: PredictReply) -> Any:
    """Map a service reply to the payload shape the in-process tools already return.

    The validator's _probabilities() reads `confidence` and any label -> float pairs,
    and _text_stance() reads a plain string. Matching those exactly is what keeps this
    a transport change rather than a validation change.
    """
    if reply.task == "classify":
        return dict(reply.probabilities)
    if reply.task == "report":
        return reply.answer
    payload: Dict[str, Any] = {"response": reply.answer}
    # Attach the probability only when the service says the question was yes/no. The
    # validator would drop it anyway, but a service should not emit a number whose name
    # is a lie and leave the client to clean up after it.
    if reply.yes_no and reply.p_yes is not None:
        payload["confidence"] = reply.p_yes
    return payload


def metadata(reply: PredictReply, url: str) -> Dict[str, Any]:
    """Provenance travels with every answer, so a log says which model actually spoke."""
    return {"served_by": url, "model": reply.model, "task": reply.task,
            "yes_no": reply.yes_no}
