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

    # Initialize only selected tools or all if none specified
    tools_dict = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

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
        "LlavaMedTool",
        "XRayPhraseGroundingTool",
    ]

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
