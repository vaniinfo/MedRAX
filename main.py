import os
import warnings
from typing import *
from dotenv import load_dotenv
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


DIRECTORS = ["gpt-4o", "qwen"]


def get_director(name, ollama_model=None):
    """Return the model name and OpenAI client kwargs for the director LLM.

    Both directors go through ChatOpenAI: GPT-4o via the OpenAI API, Qwen via Ollama's
    OpenAI-compatible endpoint.

    Args:
        name (str): "gpt-4o" or "qwen"
        ollama_model (str, optional): Ollama model tag for "qwen". Defaults to the
            OLLAMA_MODEL env var, then "qwen3.8:latest" (27B, ~16 GB; fits on a 24 GB GPU
            next to CheXagent loaded in bf16). Also used as the EDV validator, where the
            smaller qwen3-vl:8b invented findings on a normal film.

    Returns:
        Tuple[str, dict]: Model name and openai_kwargs for initialize_agent
    """
    if name == "gpt-4o":
        openai_kwargs = {}
        if api_key := os.getenv("OPENAI_API_KEY"):
            openai_kwargs["api_key"] = api_key
        if base_url := os.getenv("OPENAI_BASE_URL"):
            openai_kwargs["base_url"] = base_url
        return "gpt-4o", openai_kwargs

    if name == "qwen":
        model = ollama_model or os.getenv("OLLAMA_MODEL", "qwen3.8:latest")
        openai_kwargs = {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "api_key": "ollama",  # Ollama ignores the key, but the client requires one
        }
        return model, openai_kwargs

    raise ValueError(f"Unknown director {name!r}, expected one of {DIRECTORS}")


def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/model-weights",
    temp_dir="temp",
    device="cuda",
    model="chatgpt-4o-latest",
    temperature=0.7,
    top_p=0.95,
    openai_kwargs={},
    edv="off",
    prompt_name="MEDICAL_ASSISTANT",
    edv_style="cxragent",
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
        edv (str, optional): Evidence validation mode: "off", "cxragent" (final responder
            decides), "advisory" (critiques shown only) or "inline" (director summarizes
            each result with its critique). Defaults to "off".
        prompt_name (str, optional): Section of the prompt file to use as the director's
            system prompt. Defaults to "MEDICAL_ASSISTANT".
        edv_style (str, optional): Critique prompt: "cxragent" (verbatim) or "radiologist"
            (Conclusion / Supportive / Refuting / Confidence). Defaults to "cxragent".

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts[prompt_name]

    model = ChatOpenAI(model=model, temperature=temperature, top_p=top_p, **openai_kwargs)

    # CheXagent is used both as a tool and as MedGemma's visual-proof locator; load it once
    shared = {}

    def chexagent():
        if "chexagent" not in shared:
            shared["chexagent"] = XRayVQATool(cache_dir=model_dir, device=device, temp_dir=temp_dir)
        return shared["chexagent"]

    all_tools = {
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device),
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=True),
        "XRayVQATool": chexagent,
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(
            cache_dir=model_dir, device=device
        ),
        "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(
            cache_dir=model_dir, temp_dir=temp_dir, load_in_8bit=True, device=device
        ),
        "ChestXRayGeneratorTool": lambda: ChestXRayGeneratorTool(
            model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device
        ),
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),
        # The director's model lists MedGemma's findings; MedGemma alone follows the
        # extraction rules unreliably
        "MedGemmaTool": lambda: MedGemmaTool(locator=chexagent(), extractor=model),
    }

    # Initialize only selected tools or all if none specified
    tools_dict = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    checkpointer = MemorySaver()
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
        edv=edv,
        edv_style=edv_style,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the MedRAX application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    # Example: initialize with only specific tools
    # Here three tools are commented out, you can uncomment them to use them
    selected_tools = [
        "ImageVisualizerTool",
        "DicomProcessorTool",
        "ChestXRayClassifierTool",
        "ChestXRaySegmentationTool",
        "ChestXRayReportGeneratorTool",
        "XRayVQATool",
        # "LlavaMedTool",
        # "XRayPhraseGroundingTool",
        # "ChestXRayGeneratorTool",
    ]

    # Collect the ENV variables
    openai_kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key

    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    agent, tools_dict = initialize_agent(
        "medrax/docs/system_prompts.txt",
        tools_to_use=selected_tools,
        model_dir="/model-weights",  # Change this to the path of the model weights
        temp_dir="temp",  # Change this to the path of the temporary directory
        device="cuda",  # Change this to the device you want to use
        model="gpt-4o",  # Change this to the model you want to use, e.g. gpt-4o-mini
        temperature=0.7,
        top_p=0.95,
        openai_kwargs=openai_kwargs
    )
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name="0.0.0.0", server_port=8585, share=True)
