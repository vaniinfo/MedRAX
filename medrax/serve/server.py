"""One model, one process, one port. Start as many as you have models.

    python -m medrax.serve.server --backend densenet  --port 8103
    python -m medrax.serve.server --backend chexagent --port 8101   # transformers 4.40
    python -m medrax.serve.server --backend medgemma  --port 8102   # transformers >=4.50

The agent talks to all of them over the same two endpoints and never imports a model
library, so the pin that a model needs stops being the pin the whole project needs.

  GET  /health   -> HealthReply, including the transformers version this process runs
                    and, where the backend can self-test, whether it still sees.
  POST /predict  -> PredictReply. multipart: file=<image>, prompt=<text>
"""
import argparse
import sys

from fastapi import FastAPI, File, Form, UploadFile

from .backends import BACKENDS
from .contract import HealthReply, PredictReply


def build_app(backend) -> FastAPI:
    app = FastAPI(title=f"medrax-serve:{backend.tool}")
    state = {"ready": False, "detail": "", "vision_ok": None}

    @app.on_event("startup")
    def _load() -> None:
        try:
            backend.load()
            state["ready"] = True
            # Self-test once at startup rather than per request: it costs a generation
            # or two, and the answer cannot change while the process lives.
            state["vision_ok"] = backend.vision_ok()
            if state["vision_ok"] is False:
                state["detail"] = ("MODEL IS NOT LOOKING AT THE IMAGE: it returned the "
                                   "same answer for a real X-ray and a blank one. Check "
                                   "the transformers version in this environment.")
        except Exception as exc:
            state["ready"] = False
            state["detail"] = f"{type(exc).__name__}: {exc}"

    @app.get("/health", response_model=HealthReply)
    def health() -> HealthReply:
        return HealthReply(tool=backend.tool, model=backend.model, task=backend.task,
                           ready=state["ready"],
                           transformers=backend.transformers_version(),
                           vision_ok=state["vision_ok"], detail=state["detail"])

    @app.post("/predict", response_model=PredictReply)
    async def predict(file: UploadFile = File(...), prompt: str = Form(""),
                      max_new_tokens: int = Form(0)) -> PredictReply:
        if not state["ready"]:
            # Fail loudly. A service that answers while broken is worse than one that
            # is down, because the agent cannot tell the difference from the reply.
            raise RuntimeError(f"{backend.tool} is not ready: {state['detail']}")
        # Bounding generation matters for measurement: P(yes) is read off the first
        # token, so a 3264-call sweep has no reason to pay for the explanation after it.
        return backend.predict(await file.read(), prompt,
                               max_new_tokens=max_new_tokens or None)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    import uvicorn
    backend = BACKENDS[args.backend]()
    print(f"serving {backend.tool} ({backend.model}) on "
          f"http://{args.host}:{args.port}", flush=True)
    uvicorn.run(build_app(backend), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
