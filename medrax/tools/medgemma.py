import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI


class MedGemmaInput(BaseModel):
    """Input schema for the MedGemma opinion tool."""

    image_path: str = Field(..., description="Path to the chest X-ray image (JPG or PNG)")
    question: str = Field(..., description="Question about the chest X-ray to give an opinion on")


FINDINGS_PROMPT = (
    "Below is a radiologist's opinion on a chest X-ray. List only the abnormal findings "
    "the opinion says it SEES on this image, as present or suspected.\n"
    "Do not include:\n"
    "- findings stated as absent, not seen, ruled out, or that cannot be determined;\n"
    "- signs the opinion says it would look for, or lists as criteria or general "
    'explanations (e.g. "to assess for pneumonia I would look for consolidation");\n'
    '- locations or sides on their own (e.g. "lower lung fields", "right side");\n'
    "- normal structures and recommendations.\n"
    "Use short radiology terms of one to four words without the location, for example "
    '"pneumonia", "consolidation", "pleural effusion", "cardiomegaly". If the opinion does '
    "not state that it sees any abnormality, output []. Output a JSON list of strings "
    "only.\n\nOpinion:\n"
)

# Qualifiers and sides at the start of a term, and trailing location phrases, are removed
# so the locator gets a clean phrase ("possible right-sided pleural effusion in the base"
# -> "pleural effusion")
_LEADING_QUALIFIERS = re.compile(
    r"^(?:(?:possible|possibly|probable|suspected|likely|questionable|mild|moderate|severe|"
    r"small|large|increased|new|right-sided|left-sided|right|left|bilateral)\s+)+"
)
_TRAILING_LOCATION = re.compile(r"\s+(?:in|on|at|of|within|involving)\s+(?:the\s+)?.*$")

def describe_location(box: List[float]) -> str:
    """Describe where a box lies, so the summary never has to work it out from numbers.

    Side comes from the box's horizontal centre (the left of a frontal radiograph is the
    patient's right); a box extending well past the midline on both sides is "both sides".
    Zones are the image's upper, middle and lower thirds, an approximation of the lung
    zones; every third holding at least a quarter of the box's height is named.

    Args:
        box: [x1, y1, x2, y2] in percent of image width and height
    Returns:
        e.g. "patient's right, upper zone" or "patient's left, middle-to-lower zones"
    """
    x1, y1, x2, y2 = box
    if x1 < 40 and x2 > 60:
        side = "both sides"
    else:
        side = "patient's right" if (x1 + x2) / 2 < 50 else "patient's left"

    height = max(y2 - y1, 1e-6)
    thirds = [("upper", 0, 100 / 3), ("middle", 100 / 3, 200 / 3), ("lower", 200 / 3, 100)]
    zones = [
        name for name, top, bottom in thirds
        if (min(y2, bottom) - max(y1, top)) / height >= 0.25
    ]
    if not zones:  # a thin box: use the third holding its centre
        centre = (y1 + y2) / 2
        zones = [name for name, top, bottom in thirds if top <= centre <= bottom][:1]
    zone = f"{zones[0]} zone" if len(zones) == 1 else f"{zones[0]}-to-{zones[-1]} zones"
    return f"{side}, {zone}"


# Terms made only of these words are locations, not findings, and are never located
_LOCATION_WORDS = {
    "right", "left", "bilateral", "both", "upper", "middle", "mid", "lower", "lung", "lungs",
    "field", "fields", "zone", "zones", "side", "sides", "lobe", "lobes", "base", "bases",
    "basal", "apex", "apices", "apical", "hilar", "perihilar", "region", "area", "the", "and",
}


class MedGemmaTool(BaseTool):
    """Tool that asks MedGemma (served by Ollama) for an opinion on a chest X-ray, with
    visual proof from a locator model.

    MedGemma runs in Ollama rather than in-process: its transformers requirement (>= 4.50)
    conflicts with the version pinned for CheXagent in this environment.

    If a locator (the CheXagent tool) is given, the findings MedGemma states are present
    are boxed by the locator and drawn on the image. Findings the locator cannot box are
    reported as not localized. The locator only shows where; it does not give an opinion.
    """

    name: str = "medgemma_opinion"
    description: str = (
        "Asks MedGemma, a medical vision-language model, for an expert opinion on a chest "
        "X-ray. Input is the image path and a natural language question. Output is "
        "MedGemma's written opinion and, for each finding MedGemma states is present, "
        "visual proof from CheXagent: bounding boxes (box_normalized as percent of image "
        "width/height, x1,y1,x2,y2) drawn on the X-ray, or 'not localized'."
    )
    args_schema: Type[BaseModel] = MedGemmaInput
    model: Any = None
    locator: Any = None
    extractor: Any = None
    system_prompt: str = "You are an expert radiologist."
    max_tokens: int = 512
    max_findings: int = 5

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        locator: Any = None,
        extractor: Any = None,
        **kwargs: Any,
    ):
        """
        Args:
            model_name: Ollama model tag. Defaults to MEDGEMMA_MODEL, then "medgemma:4b".
            base_url: OpenAI-compatible endpoint. Defaults to OLLAMA_BASE_URL, then the
                local Ollama server.
            locator: Optional XRayVQATool (CheXagent) used to box MedGemma's findings.
            extractor: Optional chat model that lists the findings MedGemma states (text
                only). MedGemma itself is used if not given, but follows the extraction
                rules less reliably.
        """
        super().__init__(**kwargs)
        self.model = ChatOpenAI(
            model=model_name or os.getenv("MEDGEMMA_MODEL", "medgemma:4b"),
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",  # Ollama ignores the key, but the client requires one
            temperature=0,
        )
        self.locator = locator
        self.extractor = extractor

    def _extract_findings(self, opinion: str) -> List[str]:
        """List the findings the opinion states it sees (a text-only call to the extractor,
        or to MedGemma if no extractor was given)."""
        extractor = self.extractor or self.model
        response = extractor.invoke([HumanMessage(content=FINDINGS_PROMPT + opinion)], max_tokens=200)
        # MedGemma 1.5 writes its reasoning before the answer
        text = response.content.rsplit("Final Answer:", 1)[-1]
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\[.*?\]", text, re.S)
        try:
            findings = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            findings = None
        if not isinstance(findings, list):
            # Unquoted ([opacity, consolidation]) or unclosed lists: split on commas and
            # strip any leftover brackets and quotes
            body = match.group(0) if match else text if text.startswith("[") else ""
            findings = [f.strip(" \"'\n[]") for f in body.split(",")]
        unique = []
        for finding in findings:
            finding = str(finding).strip().lower()
            finding = _TRAILING_LOCATION.sub("", _LEADING_QUALIFIERS.sub("", finding)).strip()
            if not finding or finding in unique:
                continue
            # Safety net: drop pure locations such as "lower lung fields" or "right side"
            if set(re.findall(r"[a-z]+", finding)) <= _LOCATION_WORDS:
                continue
            unique.append(finding)
        return unique[: self.max_findings]

    def _run(
        self,
        image_path: str,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Ask MedGemma the question about the image."""
        try:
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            response = self.model.invoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(
                        content=[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{img_base64}"},
                            },
                            {"type": "text", "text": question},
                        ]
                    ),
                ],
                max_tokens=self.max_tokens,
            )

            output = {"response": response.content}

            # Visual proof: box each finding MedGemma asserts; skip what cannot be boxed
            if self.locator is not None:
                findings = self._extract_findings(response.content)
                if findings:
                    proof, viz_path = self.locator.locate_findings(image_path, findings)
                    output["visual_proof"] = [
                        {
                            "finding": p["finding"],
                            "boxes": [
                                {
                                    "box_normalized": b["box_normalized"],
                                    "location": describe_location(b["box_normalized"]),
                                    "box_pixels": b["box_pixels"],
                                    "p_present": b.get("p_present"),
                                }
                                for b in p["boxes"]
                            ],
                        }
                        if p["localized"]
                        else {
                            "finding": p["finding"],
                            "status": "not localized",
                            # CheXagent boxed something else instead, e.g. "pneumothorax"
                            **({"chexagent_boxed_instead": p["substituted_labels"]}
                               if p.get("substituted_labels") else {}),
                        }
                        for p in proof
                    ]
                    if viz_path:
                        output["visualization_path"] = viz_path

            metadata = {
                "image_path": image_path,
                "question": question,
                "model": self.model.model_name,
                "analysis_status": "completed",
            }
            return output, metadata

        except Exception as e:
            return {"error": str(e)}, {
                "image_path": image_path,
                "question": question,
                "analysis_status": "failed",
            }

    async def _arun(
        self,
        image_path: str,
        question: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path, question)
