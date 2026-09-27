import argparse
import os

from main import DIRECTORS, get_director, initialize_agent  # importing main also loads .env
from interface import create_demo

if __name__ == "__main__":
    """
    Launch the MedRAX UI with only the CheXagent VQA tool (plus lightweight helpers).

    The director LLM can be GPT-4o (OpenAI) or Qwen (local Ollama):
        python run_vqa.py                                   # GPT-4o
        python run_vqa.py --director qwen                   # qwen3.8:latest on Ollama
        python run_vqa.py --director qwen --no-image        # director never sees the X-ray
        python run_vqa.py --director qwen --edv advisory    # show evidence critiques only
        python run_vqa.py --director qwen --edv cxragent    # critiques + final responder (CXRAgent)
    Defaults can also come from MEDRAX_DIRECTOR and OLLAMA_MODEL in the environment or .env.
    """
    parser = argparse.ArgumentParser(description="MedRAX VQA UI")
    parser.add_argument(
        "--director",
        choices=DIRECTORS,
        default=os.getenv("MEDRAX_DIRECTOR", "gpt-4o"),
        help="LLM that plans tool calls and writes the answer (default: gpt-4o)",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama model tag for --director qwen (default: OLLAMA_MODEL or qwen3.8:latest)",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Do not send the X-ray to the director; it only sees tool outputs",
    )
    parser.add_argument(
        "--edv",
        choices=["off", "advisory", "cxragent"],
        default=os.getenv("MEDRAX_EDV", "off"),
        help="Evidence validation: every CheXagent result is critiqued against the image. "
        "'advisory' only shows the critiques and keeps the director's answer; 'cxragent' "
        "lets a final responder rewrite the answer from them (default: off)",
    )
    args = parser.parse_args()

    model, openai_kwargs = get_director(args.director, args.ollama_model)
    print(f"Starting VQA server with director {model}...")

    agent, tools_dict = initialize_agent(
        "medrax/docs/system_prompts.txt",
        tools_to_use=["ImageVisualizerTool", "DicomProcessorTool", "XRayVQATool"],
        model_dir="C:/Users/hari/model-weights",  # Change this to the path of the model weights
        temp_dir="temp",
        device="cuda",
        model=model,
        temperature=0.7,
        top_p=0.95,
        openai_kwargs=openai_kwargs,
        edv=args.edv,
    )
    edv_labels = {
        "advisory": " · EDV: advisory critiques (answer not changed)",
        "cxragent": " · EDV: CXRAgent validator + final responder",
    }
    label = model + edv_labels.get(args.edv, "")
    demo = create_demo(
        agent, tools_dict, send_image=not args.no_image, director_label=label
    )

    demo.launch(server_name="127.0.0.1", server_port=8585, share=False)
