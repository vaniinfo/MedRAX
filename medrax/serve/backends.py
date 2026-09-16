"""Model adapters. Each one runs in ITS OWN environment, with its own transformers pin.

A backend is deliberately tiny: hold a model, answer a question, report whether it can
still see. Everything about scoring, thresholds and confidence lives in the agent, on
the far side of the wire, and none of it belongs here.

Where a tested in-process tool already exists, the backend WRAPS it rather than
reimplementing inference. That is not laziness, it is the only safe option: the first
version of this file re-derived CheXagent's call and passed the prompt to the
tokenizer without the image. It would have loaded cleanly, written fluent radiology
prose, and never looked at the pixels -- the precise failure scripts/verify_vqa_vision.py
exists to catch. CheXagent takes its image through `tokenizer.from_list_format`, which
is easy to miss and impossible to notice from the output.

Run one per model, each in its own venv:

    python -m medrax.serve.server --backend chexagent --port 8101   # transformers 4.40
    python -m medrax.serve.server --backend medgemma  --port 8102   # transformers >=4.50
    python -m medrax.serve.server --backend densenet  --port 8103   # no transformers
"""
import os
import tempfile
from typing import Optional

from .contract import PredictReply

# Openers that make a prompt a genuine yes/no question. A "confidence" meaning P(answer
# is "Yes") is meaningless for anything else, and the agent discards it -- so the
# service is honest about which kind of question it was handed.
_YES_NO_OPENERS = ("does ", "do ", "did ", "is ", "are ", "was ", "were ", "has ",
                   "have ", "can ", "could ", "should ", "will ", "would ")


def is_yes_no(prompt: str) -> bool:
    return " ".join(str(prompt).lower().split()).startswith(_YES_NO_OPENERS)


# The probe for "is this model actually reading the image", taken from
# scripts/verify_vqa_vision.py, which is the tested version of this check.
#
# It has to be a question whose answer MUST differ between a real radiograph and a
# blank frame. The first version here asked "does this chest X-ray contain a
# pneumothorax?" of a normal film and of a black square -- both correctly answer "no",
# so a perfectly healthy CheXagent was reported as blind and refused at startup. View
# is high-entropy by comparison: a real frontal is PA or AP, a blank frame is not.
_VISION_PROBE = "What is the view of this chest X-ray? Options: (a) PA, (b) AP, (c) LATERAL"


def _cache_dir() -> str:
    return os.getenv("MEDRAX_MODEL_DIR", os.path.expanduser("~/model-weights"))


def _device() -> str:
    if os.getenv("MEDRAX_DEVICE"):
        return os.environ["MEDRAX_DEVICE"]
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Backend:
    """Interface every adapter implements."""

    tool = "override_me"        # the RELIABILITY key -- one per MODEL, never shared
    model = "override_me"
    task = "vqa"

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, image_bytes: bytes, prompt: str) -> PredictReply:
        raise NotImplementedError

    def transformers_version(self) -> Optional[str]:
        try:
            import transformers
            return transformers.__version__
        except Exception:
            return None

    def vision_ok(self) -> Optional[bool]:
        """Does the answer actually depend on the image? None if not self-testable."""
        return None

    @staticmethod
    def _to_file(image_bytes: bytes) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        handle.write(image_bytes)
        handle.close()
        return handle.name


class DenseNetBackend(Backend):
    """TorchXRayVision DenseNet-121, wrapping ChestXRayClassifierTool.

    No transformers dependency, so it can share an environment with anything. Included
    mainly because it makes the whole contract testable without a gated download.
    """

    tool = "chest_xray_classifier"
    model = "densenet121-res224-all"
    task = "classify"

    def load(self) -> None:
        from medrax.tools import ChestXRayClassifierTool
        self.impl = ChestXRayClassifierTool(device=_device())

    def predict(self, image_bytes: bytes, prompt: str) -> PredictReply:
        path = self._to_file(image_bytes)
        try:
            probabilities, _ = self.impl._run(path)
        finally:
            os.unlink(path)
        return PredictReply(model=self.model, task=self.task, answer="",
                            probabilities={k: float(v) for k, v in probabilities.items()})


class CheXagentBackend(Backend):
    """StanfordAIMI/CheXagent-2-3b, wrapping XRayVQATool. Requires transformers 4.40.x.

    XRayVQATool refuses to start on any other version rather than fabricate, so this
    service simply fails to come up in a wrong environment instead of serving text
    that has nothing to do with the image.
    """

    tool = "chest_xray_expert"
    model = "StanfordAIMI/CheXagent-2-3b"
    task = "vqa"

    def load(self) -> None:
        from medrax.tools import XRayVQATool
        self.impl = XRayVQATool(cache_dir=_cache_dir(), device=_device())

    def _ask(self, path: str, prompt: str, max_new_tokens: int = 64):
        output, _ = self.impl._run(image_paths=[path], prompt=prompt,
                                   max_new_tokens=max_new_tokens)
        return str(output.get("response", "")).strip(), output.get("confidence")

    def predict(self, image_bytes: bytes, prompt: str) -> PredictReply:
        path = self._to_file(image_bytes)
        try:
            answer, p_yes = self._ask(path, prompt, max_new_tokens=256)
        finally:
            os.unlink(path)
        yes_no = is_yes_no(prompt)
        return PredictReply(model=self.model, task=self.task, answer=answer,
                            p_yes=p_yes if yes_no else None, yes_no=yes_no)

    def vision_ok(self) -> Optional[bool]:
        """Ask the same question of a real X-ray and of a blank image. Matching answers
        mean the model is not looking. Set MEDRAX_VISION_CHECK_IMAGE to enable."""
        sample = os.getenv("MEDRAX_VISION_CHECK_IMAGE")
        if not sample or not os.path.isfile(sample):
            return None
        try:
            from PIL import Image
            blank_path = self._to_file(b"")
            os.unlink(blank_path)
            Image.new("RGB", (512, 512), "black").save(blank_path)
            try:
                real, _ = self._ask(sample, _VISION_PROBE, max_new_tokens=16)
                blank, _ = self._ask(blank_path, _VISION_PROBE, max_new_tokens=16)
            finally:
                os.path.isfile(blank_path) and os.unlink(blank_path)
            if not real.strip() or not blank.strip():
                return None          # inconclusive is not the same as failing
            return real.lower() != blank.lower()
        except Exception:
            return None


class MedGemmaBackend(Backend):
    """google/medgemma-4b-it. Requires transformers>=4.50, so it CANNOT share an
    environment with CheXagent. That conflict is why this package exists.

    Gated on Hugging Face like MAIRA-2: run `huggingface-cli login` and accept the
    Health AI terms first, or loading fails with a 403 that reads like a permissions
    bug rather than a missing login.

    No in-process tool exists to wrap, so this is a real implementation -- and it is
    therefore the one to distrust until /health reports vision_ok, or until it has
    measured RELIABILITY rows of its own.
    """

    tool = "chest_xray_expert_gemma"   # its OWN key, never shared with CheXagent
    model = "google/medgemma-4b-it"
    task = "vqa"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._torch = torch
        self.device = _device()
        self.processor = AutoProcessor.from_pretrained(self.model, cache_dir=_cache_dir())
        self.net = AutoModelForImageTextToText.from_pretrained(
            self.model, cache_dir=_cache_dir(),
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

    def _ask(self, image, prompt: str, max_new_tokens: int = 128):
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            generated = self.net.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False, output_scores=True,
                                          return_dict_in_generate=True)
        start = inputs["input_ids"].shape[-1]
        text = self.processor.decode(generated.sequences[0][start:],
                                     skip_special_tokens=True).strip()
        p_yes = None
        if is_yes_no(prompt) and generated.scores:
            ids = [self.processor.tokenizer.encode(t, add_special_tokens=False)
                   for t in ("Yes", " Yes", "yes")]
            first = [i[0] for i in ids if i]
            probabilities = self._torch.softmax(generated.scores[0][0].float(), dim=-1)
            p_yes = float(max(probabilities[i] for i in first)) if first else None
        return text, p_yes

    def predict(self, image_bytes: bytes, prompt: str) -> PredictReply:
        import io
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        answer, p_yes = self._ask(image, prompt)
        yes_no = is_yes_no(prompt)
        return PredictReply(model=self.model, task=self.task, answer=answer,
                            p_yes=p_yes if yes_no else None, yes_no=yes_no)

    def vision_ok(self) -> Optional[bool]:
        sample = os.getenv("MEDRAX_VISION_CHECK_IMAGE")
        if not sample or not os.path.isfile(sample):
            return None
        try:
            from PIL import Image
            real, _ = self._ask(Image.open(sample).convert("RGB"), _VISION_PROBE, 16)
            blank, _ = self._ask(Image.new("RGB", (512, 512), "black"), _VISION_PROBE, 16)
            if not real.strip() or not blank.strip():
                return None
            return real.lower() != blank.lower()
        except Exception:
            return None


BACKENDS = {
    "densenet": DenseNetBackend,
    "chexagent": CheXagentBackend,
    "medgemma": MedGemmaBackend,
}
