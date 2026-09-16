"""A tool backed by a model service instead of an in-process model.

Imports `requests` and nothing else. No torch, no transformers, no model weights --
which is the point: a remote tool cannot drag a dependency into the agent's
environment, so each model keeps whatever pin it needs and none of them collide.

The reply is mapped back into the payload shape the in-process tools already return,
so the EvidenceValidator scores a remote model exactly as it scores a local one. Its
`name` is the RELIABILITY key, so a served model needs its own measured rows the same
as any other tool -- until it has them, the validator marks it ASSUMED and caps it at
a Medium ceiling, which is the honest description of a tool nobody has evaluated.
"""
from typing import Any, Dict, List, Optional, Tuple, Type

import requests
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from medrax.serve.contract import PredictReply, metadata, to_payload


class RemoteToolInput(BaseModel):
    """Accepts either key, so a served model is a drop-in for the tool it replaces:
    the VQA tools take `image_paths`, the classifier takes `image_path`."""

    image_path: Optional[str] = Field(None, description="Path to a chest X-ray image")
    image_paths: Optional[List[str]] = Field(None, description="Paths to chest X-ray images")
    prompt: str = Field("", description="Question or instruction about the image")


class RemoteModelTool(BaseTool):
    """Calls one model service. One instance per served model."""

    name: str = "remote_model"
    description: str = "A chest X-ray model served in its own process."
    args_schema: Type[BaseModel] = RemoteToolInput
    url: str = ""
    task: str = "vqa"
    model_id: str = ""
    timeout: int = 180

    @classmethod
    def from_health(cls, url: str, timeout: int = 180) -> "RemoteModelTool":
        """Build from the service's own /health, so the tool name and task come from
        the process that actually holds the model rather than from a config file that
        can drift away from it."""
        info = requests.get(f"{url.rstrip('/')}/health", timeout=15).json()
        if not info.get("ready"):
            raise RuntimeError(f"{url} not ready: {info.get('detail') or 'unknown'}")
        if info.get("vision_ok") is False:
            # Refuse rather than warn. A model that answers without looking produces
            # fluent, confident, image-independent text -- the failure this project
            # keeps a verification script for.
            raise RuntimeError(f"{url} reports vision_ok=False: {info.get('detail')}")
        return cls(name=info["tool"], task=info.get("task", "vqa"),
                   model_id=info.get("model", ""), url=url.rstrip("/"), timeout=timeout,
                   description=(f"Chest X-ray {info.get('task', 'vqa')} served by "
                                f"{info.get('model', 'a remote model')}. "
                                "For yes/no questions the output includes 'confidence', "
                                "the model's probability that the answer is yes."))

    def _call(self, image_path: str, prompt: str) -> Tuple[Any, Dict[str, Any]]:
        with open(image_path, "rb") as handle:
            response = requests.post(f"{self.url}/predict",
                                     files={"file": (image_path, handle.read())},
                                     data={"prompt": prompt}, timeout=self.timeout)
        response.raise_for_status()
        reply = PredictReply(**response.json())
        return to_payload(reply), metadata(reply, self.url)

    def _run(self, image_path: Optional[str] = None, image_paths: Optional[List[str]] = None,
             prompt: str = "",
             run_manager: Optional[CallbackManagerForToolRun] = None) -> Tuple[Any, Dict[str, Any]]:
        target = image_path or (image_paths[0] if image_paths else None)
        if not target:
            return {"error": "no image path supplied"}, {"served_by": self.url}
        try:
            return self._call(target, prompt)
        except Exception as exc:
            # Surface the failure as the tool's output. The validator then reports no
            # probability and no ceiling, rather than the agent silently continuing
            # with one fewer opinion than it believes it has.
            return {"error": f"{self.name} unreachable at {self.url}: {exc}"}, {
                "served_by": self.url, "failed": True}

    async def _arun(self, image_path: Optional[str] = None,
                    image_paths: Optional[List[str]] = None, prompt: str = "",
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None
                    ) -> Tuple[Any, Dict[str, Any]]:
        return self._run(image_path, image_paths, prompt)
