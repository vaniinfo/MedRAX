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

    def __init__(self, model: BaseLanguageModel, max_tokens_explain: int = 600):
        """
        Args:
            model: Chat model without tools bound (the director's model)
            max_tokens_explain: Token budget for each critique, as in CXRAgent
        """
        self.model = model
        self.max_tokens_explain = max_tokens_explain

    def explain(self, tool_result: str, image_paths: List[str]) -> str:
        """Critique one tool result against the images (CXRAgent `_explain_func`)."""
        content = _image_parts(image_paths) + [
            {"type": "text", "text": "Tool result: " + tool_result}
        ]
        response = self.model.invoke(
            [SystemMessage(content=EXPLAIN_PROMPT), HumanMessage(content=content)],
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
