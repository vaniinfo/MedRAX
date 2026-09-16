from typing import Any, Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field

import re
import torch

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from PIL import Image


from medrax.llava.conversation import conv_templates
from medrax.llava.model.builder import load_pretrained_model
from medrax.llava.mm_utils import tokenizer_image_token, process_images
from medrax.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


class LlavaMedInput(BaseModel):
    """Input for the LLaVA-Med Visual QA tool. Only supports JPG or PNG images."""

    question: str = Field(..., description="The question to ask about the medical image")
    image_path: Optional[str] = Field(
        None,
        description="Path to the medical image file (optional), only supports JPG or PNG images",
    )


class LlavaMedTool(BaseTool):
    """Tool that performs medical visual question answering using LLaVA-Med.

    This tool uses a large language model fine-tuned on medical images to answer
    questions about medical images. It can handle both image-based questions and
    general medical questions without images.
    """

    name: str = "llava_med_qa"
    description: str = (
        "A tool that answers questions about biomedical images and general medical questions using LLaVA-Med. "
        "While it can process chest X-rays, it may not be as reliable for detailed chest X-ray analysis. "
        "Input should be a question and optionally a path to a medical image file."
    )
    args_schema: Type[BaseModel] = LlavaMedInput
    tokenizer: Any = None
    model: Any = None
    image_processor: Any = None
    context_len: int = 200000

    def __init__(
        self,
        model_path: str = "microsoft/llava-med-v1.5-mistral-7b",
        cache_dir: str = "/model-weights",
        low_cpu_mem_usage: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            cache_dir=cache_dir,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
            device=device,
            **kwargs,
        )
        self.model.eval()

    def _process_input(
        self, question: str, image_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.model.config.mm_use_im_start_end:
            question = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + question
            )
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # PATCH: was .cuda(), which hard-fails on any machine without CUDA rather than
        # falling back. Follow the loaded model instead, so the tool works wherever the
        # model was placed.
        device = self.model.device

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(device)
        )

        image_tensor = None
        if image_path:
            image = Image.open(image_path)
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            image_tensor = image_tensor.unsqueeze(0).half().to(device)

        return input_ids, image_tensor

    @staticmethod
    def _answered_yes_no(answer: str) -> bool:
        """Did the model actually open with yes or no?

        P(yes) is read off the FIRST generated token, so it only means anything if that
        token was the answer. MedGemma taught this one: asked a yes/no question it
        replied "Based on the chest X-ray provided..." and P(yes) came back 0.0001 on an
        image whose text asserted the finding -- a maximal NO pointing the opposite way
        to what the model said. First whole word, not a prefix, or "Nodular" reads as a
        no; first NON-EMPTY word, or markdown "**Yes**" reads as neither.
        """
        words = [w for w in re.split(r"[^a-z]+", str(answer).lower()) if w]
        return bool(words) and words[0] in ("yes", "no")

    def _yes_probability(self, scores, answer: str) -> Optional[float]:
        """P(the answer is yes), or None when that quantity does not exist."""
        if not scores or not self._answered_yes_no(answer):
            return None
        ids = [self.tokenizer.encode(t, add_special_tokens=False) for t in ("Yes", "yes")]
        first = [i[-1] for i in ids if i]
        if not first:
            return None
        probabilities = torch.softmax(scores[0][0].float(), dim=-1)
        return float(max(probabilities[i] for i in first))

    def _run(
        self,
        question: str,
        image_path: Optional[str] = None,
        max_new_tokens: int = 500,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Any, Dict]:
        """Answer a medical question, optionally based on an input image.

        Args:
            question (str): The medical question to answer.
            image_path (Optional[str]): The path to the medical image file (if applicable).
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.

        Returns:
            Tuple[str, Dict]: A tuple containing the model's answer and any additional metadata.

        Raises:
            Exception: If there's an error processing the input or generating the answer.
        """
        try:
            input_ids, image_tensor = self._process_input(question, image_path)
            input_ids = input_ids.to(device=self.model.device)
            image_tensor = image_tensor.to(device=self.model.device, dtype=self.model.dtype)

            # PATCH: request scores so a yes/no answer can carry a probability, the way
            # XRayVQATool already does. Without one this tool could never be measured:
            # RELIABILITY needs a continuous score to find a decision point, so a
            # text-only tool falls through to the assumed 0.5 and is capped at Medium
            # forever, however good it actually is.
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0.2,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            sequences = generated.sequences
            output = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()
            metadata = {
                "question": question,
                "image_path": image_path,
                "analysis_status": "completed",
            }
            confidence = self._yes_probability(generated.scores, output)
            if confidence is None:
                return output, metadata
            # Same shape XRayVQATool returns, so the validator scores it identically.
            return {"response": output, "confidence": confidence}, metadata
        except Exception as e:
            return f"Error generating answer: {str(e)}", {
                "question": question,
                "image_path": image_path,
                "analysis_status": "failed",
            }

    async def _arun(
        self,
        question: str,
        image_path: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Asynchronously answer a medical question, optionally based on an input image.

        This method currently calls the synchronous version, as the model inference
        is not inherently asynchronous. For true asynchronous behavior, consider
        using a separate thread or process.

        Args:
            question (str): The medical question to answer.
            image_path (Optional[str]): The path to the medical image file (if applicable).
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.

        Returns:
            Tuple[str, Dict]: A tuple containing the model's answer and any additional metadata.

        Raises:
            Exception: If there's an error processing the input or generating the answer.
        """
        return self._run(question, image_path)
