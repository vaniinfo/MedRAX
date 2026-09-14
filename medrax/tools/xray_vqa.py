from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
from pydantic import BaseModel, Field

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


# PATCH: CheXagent-2-3b is only compatible with transformers 4.40.x. On newer versions
# the model still loads and still emits fluent, clinically plausible text, but it stops
# attending to the injected image embeddings -- it returns the same answer for a real
# chest X-ray and for a blank image, with no exception and no warning.
#
# The previous code here spoofed `transformers.__version__ = "4.40.0"` to get past the
# model's own compatibility check. That silenced the alarm without fixing the fault and
# turned a loud failure into a silent one. We now verify the real version and refuse to
# start instead, because confident fabricated radiology is worse than a crash.
_REQUIRED_TRANSFORMERS = "4.40"


def _require_compatible_transformers() -> None:
    """Raise if the installed transformers version breaks CheXagent's vision path."""
    installed = transformers.__version__
    if not installed.startswith(_REQUIRED_TRANSFORMERS):
        raise RuntimeError(
            f"XRayVQATool requires transformers {_REQUIRED_TRANSFORMERS}.x, but "
            f"{installed} is installed.\n"
            "On other versions CheXagent-2-3b SILENTLY IGNORES THE IMAGE: it produces "
            "confident, well-formed answers that are unrelated to the X-ray. This fails "
            "without raising, so the tool refuses to start rather than fabricate.\n"
            "Fix: pip install 'transformers==4.40.0' 'tokenizers>=0.19,<0.20'\n"
            "Verify afterwards with: python scripts/verify_vqa_vision.py"
        )


class XRayVQAToolInput(BaseModel):
    """Input schema for the CheXagent Tool."""

    image_paths: List[str] = Field(
        ..., description="List of paths to chest X-ray images to analyze"
    )
    prompt: str = Field(..., description="Question or instruction about the chest X-ray images")
    max_new_tokens: int = Field(
        512, description="Maximum number of tokens to generate in the response"
    )


class XRayVQATool(BaseTool):
    """Tool that leverages CheXagent for comprehensive chest X-ray analysis."""

    name: str = "chest_xray_expert"
    description: str = (
        "A versatile tool for analyzing chest X-rays. "
        "Can perform multiple tasks including: visual question answering, report generation, "
        "abnormality detection, comparative analysis, anatomical description, "
        "and clinical interpretation. Input should be paths to X-ray images "
        "and a natural language prompt describing the analysis needed. "
        # PATCH: tell the orchestrator the confidence number exists, otherwise it
        # never appears in the agent's reasoning.
        "For yes/no questions the output also includes 'confidence', the model's "
        "probability that the answer is yes (0.0-1.0). Use it: 0.97 and 0.51 are both "
        "reported as 'Yes' but mean very different things."
    )
    args_schema: Type[BaseModel] = XRayVQAToolInput
    return_direct: bool = True
    cache_dir: Optional[str] = None
    device: Optional[str] = None
    dtype: torch.dtype = torch.bfloat16
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None
    # PATCH: token ids for "Yes"/"No", used to turn a binary answer into a probability
    yes_token_id: Optional[int] = None
    no_token_id: Optional[int] = None

    def __init__(
        self,
        model_name: str = "StanfordAIMI/CheXagent-2-3b",
        device: Optional[str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the XRayVQATool.

        Args:
            model_name: Name of the CheXagent model to use
            device: Device to run model on (cuda/cpu)
            dtype: Data type for model weights
            cache_dir: Directory to cache downloaded models
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)

        _require_compatible_transformers()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.cache_dir = cache_dir

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.model = self.model.to(dtype=self.dtype)
        self.model.eval()

        # PATCH: resolve "Yes"/"No" token ids once, for confidence extraction
        self.yes_token_id = self._single_token_id("Yes")
        self.no_token_id = self._single_token_id("No")

    def _single_token_id(self, word: str) -> Optional[int]:
        """Token id for `word`, or None if it does not encode to a single token."""
        ids = self.tokenizer.encode(word)
        return int(ids[0]) if len(ids) >= 1 else None

    def _generate_response(
        self, image_paths: List[str], prompt: str, max_new_tokens: int
    ) -> Tuple[str, Optional[float]]:
        """Generate response using CheXagent model.

        Args:
            image_paths: List of paths to chest X-ray images
            prompt: Question or instruction about the images
            max_new_tokens: Maximum number of tokens to generate
        Returns:
            str: Model's response
        """
        query = self.tokenizer.from_list_format(
            [*[{"image": path} for path in image_paths], {"text": prompt}]
        )
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        ).to(device=self.device)

        # Run inference.
        # PATCH: request scores so a binary answer can be converted into a probability.
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        output = generated.sequences[0]
        response = self.tokenizer.decode(output[input_ids.size(1) : -1])
        confidence = self._yes_probability(generated.scores, output, input_ids.size(1))
        return response, confidence

    def _yes_probability(self, scores, sequence, prompt_length: int) -> Optional[float]:
        """Probability that a yes/no answer is "Yes", from the first generated token.

        CheXagent answers binary questions with a bare "Yes" or "No", which discards how
        certain it was -- 0.97 and 0.51 both print as "Yes". Softmaxing the logits of the
        two candidate tokens recovers that. Returns None when the reply is not binary,
        so open-ended prompts are unaffected.
        """
        if not scores or self.yes_token_id is None or self.no_token_id is None:
            return None
        if sequence.size(0) <= prompt_length:
            return None
        first_token = int(sequence[prompt_length])
        if first_token not in (self.yes_token_id, self.no_token_id):
            return None
        logits = scores[0][0].float()
        pair = torch.softmax(
            torch.stack([logits[self.yes_token_id], logits[self.no_token_id]]), dim=0
        )
        return round(float(pair[0]), 4)

    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Execute the chest X-ray analysis.

        Args:
            image_paths: List of paths to chest X-ray images
            prompt: Question or instruction about the images
            max_new_tokens: Maximum number of tokens to generate
            run_manager: Optional callback manager

        Returns:
            Tuple[Dict[str, Any], Dict]: Output dictionary and metadata dictionary
        """
        try:
            # Verify image paths
            for path in image_paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Image file not found: {path}")

            response, confidence = self._generate_response(image_paths, prompt, max_new_tokens)

            output = {
                "response": response,
            }
            # PATCH: only present for yes/no answers; omitted for open-ended replies
            if confidence is not None:
                output["confidence"] = confidence

            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "completed",
            }

            return output, metadata

        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "failed",
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_paths, prompt, max_new_tokens)
