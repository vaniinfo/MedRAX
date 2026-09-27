from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import re
import uuid
from pydantic import BaseModel, Field

import torch
from PIL import Image, ImageDraw, ImageFont
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


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
        "The model answers only one task per call. A Yes to a presence question of the form "
        "'Is there <finding>?' automatically includes the finding's location (boxes and "
        "visualization_path), so no separate locate call is needed for it. "
        "To get the location of a finding or structure as visual evidence, use a prompt of "
        "exactly the form 'Please locate the following phrase: <finding>' (e.g. 'Please locate "
        "the following phrase: pleural effusion') with a single image; if the finding is absent "
        "the model says so instead of returning a box. Open-ended 'where' questions return text "
        "only. When the model localizes a finding, the output "
        "also contains bounding boxes (box_normalized on a 0-100 scale, box_pixels in image "
        "pixels) and a visualization_path to the annotated image, which is shown to the user. "
        "Each box has p_present, the model's own yes/no probability that the finding exists; "
        "locate prompts can box absent findings, so describe boxes with p_present < 0.5 as "
        "likely absent / probable false positive, never as confirmed findings. "
        "For yes/no questions (e.g. 'Is there pneumonia?') the output includes "
        "yes_no_confidence with the model's relative probability of Yes vs No; it is an "
        "uncalibrated model score, so report it as e.g. 'Yes (model p=0.83)', not as a "
        "clinical probability."
    )
    args_schema: Type[BaseModel] = XRayVQAToolInput
    return_direct: bool = True
    cache_dir: Optional[str] = None
    device: Optional[str] = None
    dtype: torch.dtype = torch.bfloat16
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None
    temp_dir: Path = Path("temp")

    def __init__(
        self,
        model_name: str = "StanfordAIMI/CheXagent-2-3b",
        device: Optional[str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        temp_dir: Optional[str] = "temp",
        **kwargs: Any,
    ) -> None:
        """Initialize the XRayVQATool.

        Args:
            model_name: Name of the CheXagent model to use
            device: Device to run model on (cuda/cpu)
            dtype: Data type for model weights
            cache_dir: Directory to cache downloaded models
            temp_dir: Directory to save grounding visualizations
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)

        self.temp_dir = Path(temp_dir or "temp")
        self.temp_dir.mkdir(exist_ok=True)

        # Dangerous code, but works for now
        import transformers

        original_transformers_version = transformers.__version__
        transformers.__version__ = "4.40.0"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.cache_dir = cache_dir

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        # Load straight in self.dtype: loading in float32 first leaves ~12 GB reserved by
        # PyTorch's allocator even after the conversion below
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=self.dtype,
        )
        self.model = self.model.to(dtype=self.dtype)
        self.model.eval()

        # CheXagent only encodes the images when forward() gets past_key_values=None.
        # Newer transformers pre-creates an empty DynamicCache for cache-class models, which
        # silently skips image encoding (the model answers from the text alone). Disabling
        # the flag makes generate() pass None on the first step; the model's forward then
        # builds its own cache.
        self.model._supports_cache_class = False

        transformers.__version__ = original_transformers_version

    def _generate_response(
        self, image_paths: List[str], prompt: str, max_new_tokens: int
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate response using CheXagent model.

        Args:
            image_paths: List of paths to chest X-ray images
            prompt: Question or instruction about the images
            max_new_tokens: Maximum number of tokens to generate
        Returns:
            Tuple[str, Optional[Dict]]: Model's response, and a yes/no confidence
                (see _yes_no_confidence) if the answer starts with Yes or No
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

        # Run inference
        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )
            new_tokens = output.sequences[0][input_ids.size(1) :]
            response = self.tokenizer.decode(new_tokens[:-1])

            # Final-layer hidden state at the position that predicts the first answer token
            first_hidden = output.hidden_states[0][-1][0, -1]
            yes_no = None
            # Only score actual yes/no answers ("Yes", "No, ..."), not e.g. "No significant findings"
            if re.match(r"^\s*(yes|no)\s*(?:[.,!]|$)", response, re.IGNORECASE):
                yes_no = self._yes_no_confidence(first_hidden, int(new_tokens[0]))

            return response, yes_no

    def _yes_no_confidence(
        self, hidden: torch.Tensor, first_token_id: int
    ) -> Optional[Dict[str, Any]]:
        """Compute P(Yes) vs P(No) for the first answer token.

        The logits are recomputed in float32 for the Yes/No token variants only: the
        bf16 lm_head output is too coarse (~0.125 logit steps) and collapses the
        probabilities to a few discrete values. Probabilities are renormalized over
        the Yes/No variants, so they are not calibrated clinical probabilities.

        Args:
            hidden: Final-layer hidden state that predicts the first answer token
            first_token_id: The first token the model actually generated
        Returns:
            Optional[Dict]: {"answer", "probability", "p_yes", "p_no"}, or None if the
                answer does not start with Yes or No
        """
        yes_ids = self._first_token_ids(["Yes", " Yes", "yes", " yes", "YES"])
        no_ids = self._first_token_ids(["No", " No", "no", " no", "NO"])
        if first_token_id not in yes_ids + no_ids:
            return None

        lm_head = self.model.get_output_embeddings()
        ids = torch.tensor(yes_ids + no_ids, device=hidden.device)
        logits = hidden.float() @ lm_head.weight[ids].float().T
        if lm_head.bias is not None:
            logits = logits + lm_head.bias[ids].float()
        probs = torch.softmax(logits, dim=-1)

        p_yes = float(probs[: len(yes_ids)].sum())
        answer = "Yes" if first_token_id in yes_ids else "No"
        return {
            "answer": answer,
            "probability": round(p_yes if answer == "Yes" else 1 - p_yes, 3),
            "p_yes": round(p_yes, 3),
            "p_no": round(1 - p_yes, 3),
        }

    def _first_token_ids(self, words: List[str]) -> List[int]:
        """Return the distinct first token ids of the given words."""
        ids = []
        for word in words:
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if token_ids and token_ids[0] not in ids:
                ids.append(token_ids[0])
        return ids

    def _parse_grounding(self, response: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract CheXagent grounding tags from a response.

        CheXagent emits Qwen-VL style tags, e.g.
        '<|ref|> Heart <|/ref|> <|box|> (38,50),(80,72) <|/box|>', with coordinates
        (x1,y1),(x2,y2) normalized to a 0-100 range (not Qwen-VL's 0-1000; verified
        empirically on CheXagent-2-3b). A ref applies to every box that follows it.

        Args:
            response: Raw decoded model response
        Returns:
            Tuple[str, List[Dict]]: Response with tags removed, and list of
                {"phrase", "box_normalized"} entries
        """
        tag_pattern = re.compile(
            r"<\|ref\|>(.*?)<\|/ref\|>"
            r"|<\|box\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|/box\|>"
        )
        boxes = []
        phrase = None
        for match in tag_pattern.finditer(response):
            if match.group(1) is not None:
                phrase = match.group(1).strip()
            else:
                box = [int(v) for v in match.group(2, 3, 4, 5)]
                boxes.append({"phrase": phrase, "box_normalized": box})

        clean_text = re.sub(r"<\|ref\|>\s*(.*?)\s*<\|/ref\|>", r"\1", response)
        clean_text = re.sub(r"<\|box\|>.*?<\|/box\|>", "", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        return clean_text, boxes

    def _visualize_boxes(self, image_path: str, boxes: List[Dict[str, Any]]) -> str:
        """Draw bounding boxes on the image and save the visualization.

        Args:
            image_path: Path to the image the boxes refer to
            boxes: Parsed boxes; each entry gets a "box_pixels" key added
        Returns:
            str: Path to the saved visualization
        """
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        line_width = max(2, width // 200)
        font = ImageFont.load_default(size=max(12, width // 50))

        for entry in boxes:
            x1, y1, x2, y2 = entry["box_normalized"]
            box_pixels = [
                int(x1 / 100 * width),
                int(y1 / 100 * height),
                int(x2 / 100 * width),
                int(y2 / 100 * height),
            ]
            entry["box_pixels"] = box_pixels

            # Red for likely present, orange for boxes the presence check marks unlikely
            p_present = entry.get("p_present")
            color = "orange" if p_present is not None and p_present < 0.5 else "red"
            label = entry["phrase"] or ""
            if p_present is not None:
                label += f" p={p_present:.2f}" + (" (likely absent)" if p_present < 0.5 else "")

            draw.rectangle(box_pixels, outline=color, width=line_width)
            if label:
                draw.text(
                    (box_pixels[0] + line_width, box_pixels[1] + line_width),
                    label,
                    fill=color,
                    font=font,
                )

        # Use forward slashes so the UI's temp-path scrubbing matches on Windows too
        viz_path = (self.temp_dir / f"vqa_grounding_{uuid.uuid4().hex[:8]}.png").as_posix()
        image.save(viz_path)
        return viz_path

    @staticmethod
    def _yes_no_finding(prompt: str) -> Optional[str]:
        """Extract X from yes/no presence questions like "Is there X?" or "Does this chest
        X-ray show X?"; None for any other prompt."""
        # The finding ends at the first "?", so trailing requests ("...? Show me where.")
        # are not taken as part of it
        patterns = [
            r"^\s*(?:is|are) there (?:an? |any |evidence of )?([^?]+?)\s*(?:\?|$)",
            r"^\s*does (?:this|the) (?:chest )?x-?ray show (?:an? |any |evidence of )?([^?]+?)\s*(?:\?|$)",
        ]
        for pattern in patterns:
            match = re.match(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _check_presence(self, image_path: str, boxes: List[Dict[str, Any]]) -> None:
        """Ask the model a yes/no question for each boxed phrase and store P(present).

        Locate prompts presuppose the finding, so the model sometimes draws a box for an
        absent finding. Each box gets "p_present" (None if the answer was not yes/no).

        Args:
            image_path: Path to the image the boxes refer to
            boxes: Parsed boxes, updated in place
        """
        p_by_phrase = {}
        for entry in boxes:
            phrase = entry["phrase"]
            if phrase and phrase not in p_by_phrase:
                _, yes_no = self._generate_response([image_path], f"Is there {phrase}?", 8)
                p_by_phrase[phrase] = yes_no["p_yes"] if yes_no else None
            entry["p_present"] = p_by_phrase.get(phrase)

    def locate_findings(
        self, image_path: str, findings: List[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Visual proof for findings named by another model: box each one on the image.

        Each finding gets a "Please locate the following phrase: <finding>" call. Findings
        the model does not box are reported as not localized rather than guessed. A box the
        model labels as a different finding is not counted, and p_present always scores
        the finding that was asked about.

        Args:
            image_path: Path to the chest X-ray
            findings: Short finding phrases, e.g. ["pneumonia", "pleural effusion"]
        Returns:
            Tuple of per-finding results ({"finding", "localized", "boxes"}) and the path to
            an image with all boxes drawn, or None if nothing was localized
        """
        results, all_boxes = [], []
        for finding in findings:
            response, _ = self._generate_response(
                [image_path], f"Please locate the following phrase: {finding}", 128
            )
            _, boxes = self._parse_grounding(response)

            # The model sometimes boxes a different finding than the one asked about
            # (e.g. "Pneumothorax" for "pneumonia"); such a box is no evidence for it
            key = finding.strip().lower()
            matching, substituted = [], []
            for box in boxes:
                label = (box["phrase"] or finding).strip().lower()
                (matching if key in label or label in key else substituted).append(box)

            if matching:
                # Score the finding that was asked about, not the model's label: the label
                # can be a different finding, and even its capitalization shifts the score
                _, yes_no = self._generate_response([image_path], f"Is there {finding}?", 8)
                for box in matching:
                    box["phrase"] = finding
                    box["p_present"] = yes_no["p_yes"] if yes_no else None
                all_boxes.extend(matching)

            result = {"finding": finding, "localized": bool(matching), "boxes": matching}
            if substituted:
                result["substituted_labels"] = sorted({b["phrase"] for b in substituted})
            results.append(result)

        viz_path = self._visualize_boxes(image_path, all_boxes) if all_boxes else None
        return results, viz_path

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

            response, yes_no = self._generate_response(image_paths, prompt, max_new_tokens)
            response, boxes = self._parse_grounding(response)

            output = {
                "response": response,
            }
            if yes_no:
                output["yes_no_confidence"] = yes_no

            # A Yes to "Is there X?" also fetches where X is, so a positive finding always
            # comes with the model's location rather than relying on a follow-up call
            finding = self._yes_no_finding(prompt)
            if yes_no and yes_no["answer"] == "Yes" and finding and not boxes:
                locate_prompt = f"Please locate the following phrase: {finding}"
                located, _ = self._generate_response(image_paths[:1], locate_prompt, max_new_tokens)
                location_text, boxes = self._parse_grounding(located)
                output["auto_location"] = {"prompt": locate_prompt, "response": location_text}

            if boxes:
                # Boxes refer to the first image in the prompt
                self._check_presence(image_paths[0], boxes)
                output["visualization_path"] = self._visualize_boxes(image_paths[0], boxes)
                output["boxes"] = boxes

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
