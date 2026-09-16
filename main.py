import os
import warnings
from typing import *
from dotenv import load_dotenv
import torch  # PATCH: added for automatic device selection (see select_device below)
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


# PATCH: added helper. Upstream hardcodes device="cuda"; this picks the best
# available backend so the project also runs on Apple Silicon (mps) and CPU-only
# machines. Override with the MEDRAX_DEVICE env var, e.g. MEDRAX_DEVICE=cpu.
def select_device() -> str:
    """Return the best available torch device: cuda > mps > cpu."""
    if forced := os.getenv("MEDRAX_DEVICE"):
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/model-weights",
    temp_dir="temp",
    device="cuda",
    model="chatgpt-4o-latest",
    temperature=0.7,
    top_p=0.95,
    openai_kwargs={}
):
    """Initialize the MedRAX agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "chatgpt-4o-latest".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        openai_kwargs (dict, optional): Additional keyword arguments for OpenAI API, such as API key and base URL.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    # PATCH: quantisation is a per-machine decision, not a constant. 8-bit is the
    # upstream default; a 12-16GB card running LLaVA-Med alongside CheXagent needs
    # 4-bit. Both require bitsandbytes, which needs CUDA -- on Apple Silicon these
    # tools cannot run at all, so the value is irrelevant there.
    quant = os.getenv("MEDRAX_QUANT", "8bit").lower()
    # PATCH: "none" must pass both flags False explicitly. load_pretrained_model's
    # signature defaults load_in_4bit=True, so an empty dict still quantised.
    if quant == "4bit":
        quant_kwargs = {"load_in_4bit": True, "load_in_8bit": False}
    elif quant in ("none", "off", "full"):
        quant_kwargs = {"load_in_4bit": False, "load_in_8bit": False}
    else:
        quant_kwargs = {"load_in_8bit": True, "load_in_4bit": False}

    prompts = load_prompts_from_file(prompt_file)
    # PATCH: the prompt section is now selectable. MEDICAL_ASSISTANT_EDV adds the
    # evidence-driven validation protocol from CXRAgent (arXiv:2510.21324): consult
    # >=2 tools of different architectures, quote their numbers verbatim, and assign
    # a confidence anchored to tool agreement. It is the default because it beat the
    # original prompt on both cases tested so far -- correctly Medium rather than
    # High on a borderline case, and refusing a non-radiograph without calling any
    # tool. That is only two cases; revert with MEDRAX_PROMPT=MEDICAL_ASSISTANT.
    prompt_key = os.getenv("MEDRAX_PROMPT", "MEDICAL_ASSISTANT_EDV")
    if prompt_key not in prompts:
        raise KeyError(f"{prompt_key!r} not in {prompt_file}; found {list(prompts)}")
    print(f"Using system prompt: {prompt_key}")
    prompt = prompts[prompt_key]

    all_tools = {
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device),
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, **quant_kwargs),
        "XRayVQATool": lambda: XRayVQATool(cache_dir=model_dir, device=device),
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(
            cache_dir=model_dir, device=device
        ),
        "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(
            cache_dir=model_dir, temp_dir=temp_dir, device=device, **quant_kwargs
        ),
        "ChestXRayGeneratorTool": lambda: ChestXRayGeneratorTool(
            model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device
        ),
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),
    }

    # Initialize only selected tools or all if none specified.
    # PATCH: a tool that cannot load must not take the whole application down with it.
    # Weights can be gated (microsoft/maira-2 needs approval), absent (RoentGen is not
    # public), too large for the card, or blocked by a version conflict. Previously any
    # one of those raised during startup and nothing ran at all.
    tools_dict = {}
    failed = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name not in all_tools:
            print(f"  ! unknown tool {tool_name!r}, skipping")
            continue
        try:
            tools_dict[tool_name] = all_tools[tool_name]()
        except Exception as exc:
            failed[tool_name] = exc
            reason = str(exc).split("\n")[0][:160]
            print(f"  ! {tool_name} unavailable, continuing without it: {reason}")

    # PATCH: models served out of process. MEDRAX_REMOTE_TOOLS is a comma-separated
    # list of service URLs; each is asked what it is via /health, so the tool name and
    # task come from the process actually holding the weights.
    #
    # A remote tool REPLACES a local one of the same name, which is how a model moves
    # out of this environment without anything else changing -- and why the transformers
    # pin can be per model rather than per project. CheXagent needs 4.40 and breaks
    # silently above it, MedGemma needs >=4.50, MAIRA-2 newer still; in one process
    # those cannot all be satisfied, which is why XRayPhraseGroundingTool does not load.
    for url in filter(None, (u.strip() for u in
                             os.getenv("MEDRAX_REMOTE_TOOLS", "").split(","))):
        try:
            from medrax.tools.remote import RemoteModelTool
            remote = RemoteModelTool.from_health(url)
            # Match on the TOOL name, not the dict key: locals are keyed by class
            # (ChestXRayClassifierTool) and remotes by tool name (chest_xray_classifier),
            # so a key comparison leaves both loaded and the agent gets two opinions
            # from one model -- correlated errors wearing the costume of corroboration.
            superseded = [key for key, tool in tools_dict.items()
                          if getattr(tool, "name", None) == remote.name]
            for key in superseded:
                del tools_dict[key]
            tools_dict[remote.name] = remote
            note = f" (replacing local {superseded[0]})" if superseded else ""
            print(f"  + {remote.name} served by {remote.model_id} at {url}{note}")
        except Exception as exc:
            failed[url] = exc
            print(f"  ! remote tool at {url} unavailable: {str(exc).splitlines()[0][:160]}")

    if failed:
        print(f"\n{len(failed)} tool(s) could not load: {', '.join(map(str, failed))}")
        for tool_name, exc in failed.items():
            if "gated repo" in str(exc).lower() or "403" in str(exc):
                # The 403 says "not in the authorized list" even when the account does
                # have access, if the request was anonymous. Login is the usual fix.
                print(f"  {tool_name}: gated weights. Run `huggingface-cli login` first — "
                      f"the 403 says 'not in the authorized list' even when you do have "
                      f"access, if you are not logged in.")
    if not tools_dict:
        raise RuntimeError("No tools could be initialized; refusing to start.")

    # The definitive list, after remote services have superseded local tools. Printed
    # because "Tools:" above is the REQUESTED set, and a served model that failed to
    # register is otherwise invisible until you notice the Director never calls it.
    served = [t for t in tools_dict.values() if type(t).__name__ == "RemoteModelTool"]
    print(f"\nAgent tools ({len(tools_dict)}): "
          f"{', '.join(sorted(getattr(t, 'name', k) for k, t in tools_dict.items()))}")
    if served:
        print(f"  of which served remotely: "
              f"{', '.join(f'{t.name} ({t.model_id})' for t in served)}")
    elif os.getenv("MEDRAX_REMOTE_TOOLS"):
        print("  MEDRAX_REMOTE_TOOLS was set but no remote tool registered -- see the "
              "errors above")
    else:
        print("  no remote models: MEDRAX_REMOTE_TOOLS is not set in this shell")

    # PATCH: forced evidence validation. Runs as a function call on every tool result
    # rather than as a system-prompt request the model may ignore. Uses temperature=0
    # and a separate un-tool-bound model so validation cannot itself call tools.
    # Disable with MEDRAX_VALIDATE=0.
    validator = None
    if os.getenv("MEDRAX_VALIDATE", "1") == "1":
        # MEDRAX_VALIDATE_DESCRIBE=0 drops the LLM visual assessment, leaving purely
        # deterministic validation. The descriptive half can hurt: GPT-4o answering
        # "no supporting signs visible" gets used as evidence against a specialist
        # reporting 0.995, which is the generalist-overrules-specialist failure again.
        describe = os.getenv("MEDRAX_VALIDATE_DESCRIBE", "0") == "1"
        validator = EvidenceValidator(
            model=ChatOpenAI(model=model, temperature=0, **openai_kwargs),
            describe=describe,
            # CheXagent localises a named finding with bounding boxes. Passing the
            # tool in lets the validator get visual evidence from a radiology-trained
            # model instead of only from the generalist orchestrator.
            grounder=tools_dict.get("XRayVQATool"),
        )
        print(f"Evidence validation: ON (forced per tool call, describe={describe})")

    checkpointer = MemorySaver()
    model = ChatOpenAI(model=model, temperature=temperature, top_p=top_p, **openai_kwargs)
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
        validator=validator,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the MedRAX application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    # PATCH: resolve the weights directory and compute device at runtime instead of
    # hardcoding "/model-weights" (root-owned on macOS) and "cuda". Must happen before
    # the tool list is chosen, since some tools are CUDA-only.
    model_dir = os.getenv("MEDRAX_MODEL_DIR", os.path.expanduser("~/model-weights"))
    os.makedirs(model_dir, exist_ok=True)
    device = select_device()
    print(f"Using device: {device} | model_dir: {model_dir}")

    # PATCH: tool selection is now chosen for the machine instead of hand-edited.
    # LlavaMedTool and XRayPhraseGroundingTool need bitsandbytes quantisation, which
    # requires CUDA -- they cannot run on Apple Silicon or CPU at all. ChestXRayGeneratorTool
    # is never on by default: RoentGen weights are not publicly downloadable.
    # Override with MEDRAX_TOOLS as a comma-separated list.
    PORTABLE_TOOLS = [
        "ImageVisualizerTool",
        "DicomProcessorTool",
        "ChestXRayClassifierTool",
        "ChestXRaySegmentationTool",
        "ChestXRayReportGeneratorTool",
        "XRayVQATool",
    ]
    CUDA_ONLY_TOOLS = [
        "XRayPhraseGroundingTool",
    ]
    # PATCH: LlavaMedTool is no longer loaded by default. Measured over the same 544
    # films as the other tools, it is at or below chance on every finding:
    #
    #   pneumothorax 0.619   cardiomegaly 0.531   effusion 0.514
    #   edema        0.499   atelectasis  0.434   consolidation 0.429
    #
    # Five of the six are excluded by analyze_reliability.py's own gates -- two for a
    # confidence interval spanning chance, two for lying below it, one for being worse
    # than chance outright -- and CheXagent beats it by +0.33 to +0.44 AUC on every
    # finding, the widest margins in the table. It had been loaded in every session of
    # this project, occupying roughly 8GB, never once called by the Director, and with
    # no reliability rows it could have contributed through if it had been.
    #
    # This measures binary finding detection with CheXagent's question template. The
    # model was built for open-ended medical VQA and may well be better at that; the
    # result says it does not belong in this pipeline, not that the model is poor.
    # Re-enable with MEDRAX_TOOLS if you want it back.

    if tools_env := os.getenv("MEDRAX_TOOLS"):
        selected_tools = [name.strip() for name in tools_env.split(",") if name.strip()]
    else:
        selected_tools = list(PORTABLE_TOOLS)
        if device == "cuda":
            selected_tools += CUDA_ONLY_TOOLS
    print(f"Tools: {', '.join(selected_tools)}")

    # Collect the ENV variables
    openai_kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key

    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    agent, tools_dict = initialize_agent(
        "medrax/docs/system_prompts.txt",
        tools_to_use=selected_tools,
        model_dir=model_dir,  # PATCH: was "/model-weights"; set MEDRAX_MODEL_DIR to override
        temp_dir="temp",  # Change this to the path of the temporary directory
        device=device,  # PATCH: was "cuda"; auto-detected, set MEDRAX_DEVICE to override
        model="gpt-4o",  # Change this to the model you want to use, e.g. gpt-4o-mini
        temperature=0.7,
        top_p=0.95,
        openai_kwargs=openai_kwargs
    )
    demo = create_demo(agent, tools_dict)

    # PATCH: share=True opens a public gradio.live tunnel to this machine; default
    # to local-only. Set MEDRAX_SHARE=1 to restore the upstream public-link behaviour.
    demo.launch(
        server_name="0.0.0.0",
        server_port=8585,
        share=os.getenv("MEDRAX_SHARE", "0") == "1",
    )
