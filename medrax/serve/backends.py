"""Model adapters. Each one runs in ITS OWN environment, with its own transformers pin.

A backend is deliberately tiny: load a model, answer a question, report whether it can
still see. Everything about scoring, thresholds and confidence lives in the agent, on
the other side of the wire, and none of it belongs here.

Run one per model, each in its own venv:

    # CheXagent -- needs transformers==4.40.x, and breaks silently above it
    python -m medrax.serve.server --backend chexagent --port 8101

    # MedGemma -- needs transformers>=4.50, cannot coexist with the above
    python -m medrax.serve.server --backend medgemma --port 8102

    # DenseNet -- torchxrayvision, no transformers dependency at all
    python -m medrax.serve.server --backend densenet --port 8103
"""
import io
import os
from typing import Any, Dict, Optional

from .contract import PredictReply

# Openers that make a prompt a genuine yes/no question. A "confidence" that means
# P(answer is "Yes") is meaningless for anything else, and the agent discards it --
# so the service is honest about which kind of question it was given.
_YES_NO_OPENERS = ("does ", "do ", "did ", "is ", "are ", "was ", "were ", "has ",
                   "have ", "can ", "could ", "should ", "will ", "would ")


def is_yes_no(prompt: str) -> bool:
    return " ".join(str(prompt).lower().split()).startswith(_YES_NO_OPENERS)


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


class DenseNetBackend(Backend):
    """TorchXRayVision DenseNet-121. No transformers dependency, so it can share an
    environment with anything -- included mainly because it makes the contract
    testable end to end without downloading a gated model."""

    tool = "chest_xray_classifier"
    model = "densenet121-res224-all"
    task = "classify"

    def load(self) -> None:
        import torch
        import torchvision
        import torchxrayvision as xrv
        self._xrv = xrv
        self._torch = torch
        self.device = os.getenv("MEDRAX_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = xrv.models.DenseNet(weights=self.model).to(self.device).eval()
        self.transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

    def _tensor(self, image_bytes: bytes):
        import numpy as np
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        array = self._xrv.datasets.normalize(np.array(image), 255)
        array = self.transform(array[None, ...])
        return self._torch.from_numpy(array)[None, ...].to(self.device)

    def predict(self, image_bytes: bytes, prompt: str) -> PredictReply:
        with self._torch.no_grad():
            scores = self.net(self._tensor(image_bytes))[0]
        probabilities = {name: float(scores[i])
                         for i, name in enumerate(self.net.pathologies) if name}
        return PredictReply(model=self.model, task=self.task,
                            answer="", probabilities=probabilities)


class CheXagentBackend(Backend):
    """StanfordAIMI/CheXagent-2-3b. Requires transformers 4.40.x.

    On newer versions it still loads and still writes confident radiology prose, but
    stops attending to the image. That is why vision_ok is implemented here rather
    than assumed: the service proves the answer depends on the pixels by asking the
    same question of the real image and of a blank one.
    """

    tool = "chest_xray_expert"
    model = "StanfordAIMI/CheXagent-2-3b"
    task = "vqa"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        cache = os.getenv("MEDRAX_MODEL_DIR", os.path.expanduser("~/model-weights"))
        self.device = os.getenv("MEDRAX_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True,
                                                       cache_dir=cache)
        self.net = AutoModelForCausalLM.from_pretrained(
            self.model, trust_remote_code=True, cache_dir=cache,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

    def predict(self, image_bytes: bytes, prompt: str) -> PredictReply:
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        answer, p_yes = self._ask(image, prompt)
        return PredictReply(model=self.model, task=self.task, answer=answer,
                            p_yes=p_yes, yes_no=is_yes_no(prompt))

    def _ask(self, image, prompt: str):
        """Returns (answer, P(yes)). P(yes) is None unless the question was yes/no."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated = self.net.generate(**inputs, max_new_tokens=64, do_sample=False,
                                      output_scores=True, return_dict_in_generate=True)
        text = self.tokenizer.decode(generated.sequences[0][inputs.input_ids.size(1):],
                                     skip_special_tokens=True).strip()
        p_yes = None
        if is_yes_no(prompt) and generated.scores:
            yes_ids = [self.tokenizer.encode(t, add_special_tokens=False)
                       for t in ("Yes", " Yes", "yes")]
            flat = [i[0] for i in yes_ids if i]
            probs = self._torch.softmax(generated.scores[0][0].float(), dim=-1)
            p_yes = float(max(probs[i] for i in flat)) if flat else None
        return text, p_yes

    def vision_ok(self) -> Optional[bool]:
        """Real X-ray vs blank image: if the answers match, it is not looking."""
        try:
            from PIL import Image
            sample = os.getenv("MEDRAX_VISION_CHECK_IMAGE")
            if not sample or not os.path.isfile(sample):
                return None
            question = "Does this chest X-ray contain a pneumothorax?"
            real, _ = self._ask(Image.open(sample).convert("RGB"), question)
            blank, _ = self._ask(Image.new("RGB", (512, 512), "black"), question)
            return real.strip().lower() != blank.strip().lower()
        except Exception:
            return None


class MedGemmaBackend(Backend):
    """google/medgemma-4b-it. Requires transformers>=4.50, so it CANNOT share an
    environment with CheXagent. That is the whole reason this package exists.

    Gated on Hugging Face like MAIRA-2: run `huggingface-cli login` and accept the
    Health AI terms first, or loading fails with a 403 that reads like a permissions
    bug rather than a missing login.
    """

    tool = "chest_xray_expert_gemma"   # its OWN key, never shared with CheXagent
    model = "google/medgemma-4b-it"
    task = "vqa"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._torch = torch
        cache = os.getenv("MEDRAX_MODEL_DIR", os.path.expanduser("~/model-weights"))
        self.device = os.getenv("MEDRAX_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(self.model, cache_dir=cache)
        self.net = AutoModelForImageTextToText.from_pretrained(
            self.model, cache_dir=cache,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

    def predict(self, image_bytes: bytes, prompt: str) -> PredictReply:
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            generated = self.net.generate(**inputs, max_new_tokens=128, do_sample=False,
                                          output_scores=True, return_dict_in_generate=True)
        start = inputs["input_ids"].shape[-1]
        text = self.processor.decode(generated.sequences[0][start:],
                                     skip_special_tokens=True).strip()
        p_yes = None
        if is_yes_no(prompt) and generated.scores:
            ids = [self.processor.tokenizer.encode(t, add_special_tokens=False)
                   for t in ("Yes", " Yes", "yes")]
            flat = [i[0] for i in ids if i]
            probs = self._torch.softmax(generated.scores[0][0].float(), dim=-1)
            p_yes = float(max(probs[i] for i in flat)) if flat else None
        return PredictReply(model=self.model, task=self.task, answer=text,
                            p_yes=p_yes, yes_no=is_yes_no(prompt))


BACKENDS = {
    "densenet": DenseNetBackend,
    "chexagent": CheXagentBackend,
    "medgemma": MedGemmaBackend,
}
