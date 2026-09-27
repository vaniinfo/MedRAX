"""CXRAgent-style Evidence-Driven Validator (EDV).

Port of CXRAgent (arXiv:2510.21324, medAgent/agent/agent.py): after every tool result a
separate call to the LLM, given the X-ray, lists supporting and refuting visual evidence
for each of the tool's conclusions. The critiques are not shown to the tool-calling
director; a final responder writes the answer from the image, the tool conclusions and
the critiques. The prompts are CXRAgent's, verbatim.

The LLM both finds the evidence and judges its credibility here. Measured on the cxr-edv
project, that design validates almost everything it is shown; it is ported faithfully so
it can be compared, not because it is trusted.
"""

import base64
import mimetypes
from typing import Dict, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

EXPLAIN_PROMPT = (
    "Critically evaluate each individual conclusion made by the tool based on the provided chest X-ray images.\n"
    "Do not group multiple conclusions into one evaluation—analyze each separately.\n"
    "Thoroughly examine the image to identify any subtle or overlooked findings that may support or contradict the conclusion.\n"
    "Be especially attentive to subtle signs.\n"
    "For each conclusion:\n"
    "- Clearly restate the specific conclusion.\n"
    "- Describe visual evidence in the image that supports it.\n"
    "- Describe visual evidence that may contradict or refute it.\n"
    "- Assess the overall consistency of the conclusion with the image findings.\n"
    "- End with a brief one-sentence judgment on its credibility.\n"
    "Format your evaluation using one structured block per conclusion, as follows:\n"
    "[Conclusion] — [Supporting Evidence] — [Refuting Evidence] — [Credibility Evaluation]"
)

# Not CXRAgent: a critique written the way a radiologist reports. Supportive signs are named
# radiographic signs; Refuting states evidence against the conclusion itself and then the
# signs that rule out the main alternative explanations; confidence is split between the
# imaging finding and its cause. Every sign must be visible, since a model asked for named
# signs will otherwise supply them.
RADIOLOGIST_PROMPT = (
    "You are a radiologist validating another model's reading of a chest X-ray. You are "
    "given the X-ray and the model's result. Boxes in the result are [x1, y1, x2, y2] in "
    "percent of image width and height; the left of the image is the patient's right.\n"
    "Write exactly these four sections, each starting on a new line with its label:\n"
    "Conclusion: one sentence stating the finding, its location (patient side, lobe or "
    "zone) and its most likely cause, as supported by the image. If the image does not "
    "support the model's claim, say so here.\n"
    "Supportive: the named radiographic signs visible on this image that support the "
    "finding and its location (for example air bronchograms, silhouette sign, distribution "
    "of the opacity).\n"
    "Refuting: first, any visible sign that argues against the conclusion itself, or "
    "'None seen'. Then, for each main alternative explanation (such as atelectasis, "
    "pleural effusion, pulmonary edema, mass), the visible sign that argues against it.\n"
    "Confidence: two levels, each High, Medium or Low with a short reason: one for the "
    "presence and location of the imaging finding, one for its cause. State that the "
    "cause needs clinical correlation where it does.\n"
    "Rules: cite only signs you can actually see on this image. If a sign cannot be "
    "assessed on this image, say 'not assessable' rather than guessing. Do not restate "
    "the model's scores. Be concise, in the style of a radiology report."
)

RESPONDER_PROMPT = """
As the final responder, you need to review the tool call results and team results in the history record. Other information can serve as reference, but you must make the most accurate judgment. Do not mention any tools or other conclusions. Please provide the final result.
        """


def _image_parts(image_paths: List[str]) -> List[Dict]:
    """Encode images as OpenAI-style image_url content parts."""
    parts = []
    for path in image_paths:
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_base64}"}})
    return parts


class EvidenceValidator:
    """Runs CXRAgent's per-tool-result critique and its final responder."""

    def __init__(
        self, model: BaseLanguageModel, max_tokens_explain: int = 600, style: str = "cxragent"
    ):
        """
        Args:
            model: Chat model without tools bound (the director's model)
            max_tokens_explain: Token budget for each critique, as in CXRAgent
            style: "cxragent" (CXRAgent's prompt, verbatim) or "radiologist"
                (Conclusion / Supportive / Refuting / Confidence, report style)
        """
        if style not in ("cxragent", "radiologist"):
            raise ValueError(f"Unknown critique style {style!r}")
        self.model = model
        self.max_tokens_explain = max_tokens_explain
        self.prompt = RADIOLOGIST_PROMPT if style == "radiologist" else EXPLAIN_PROMPT

    def explain(self, tool_result: str, image_paths: List[str]) -> str:
        """Critique one tool result against the images (CXRAgent `_explain_func`)."""
        content = _image_parts(image_paths) + [
            {"type": "text", "text": "Tool result: " + tool_result}
        ]
        response = self.model.invoke(
            [SystemMessage(content=self.prompt), HumanMessage(content=content)],
            max_tokens=self.max_tokens_explain,
        )
        return response.content

    @staticmethod
    def note(tool_result: str, explanation: str) -> str:
        """Format a tool result and its critique the way CXRAgent stores them."""
        return (
            "\n<Tool conclusion>\n"
            f"{tool_result}\n"
            "</Tool conclusion>\n"
            "<Tool explanation>\n"
            f"{explanation}\n"
            "</Tool explanation>\n"
        )

    def respond(self, question: str, image_paths: List[str], notes: List[str]) -> str:
        """Write the final answer from the image, tool conclusions and critiques
        (CXRAgent `response_func`, without the expert-team results)."""
        messages = [SystemMessage(content=RESPONDER_PROMPT)]
        messages += [HumanMessage(content=note) for note in notes]
        messages.append(
            HumanMessage(content=_image_parts(image_paths) + [{"type": "text", "text": question}])
        )
        return self.model.invoke(messages, max_tokens=2048).content
