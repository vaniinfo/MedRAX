import argparse
import os

from main import DIRECTORS, get_director, initialize_agent  # importing main also loads .env
from interface import create_demo

if __name__ == "__main__":
    """
    Launch the MedGemma + CheXagent visual validation pipeline:
        director (never sees the image)
          -> calls MedGemma for an opinion
          -> director's model lists the findings MedGemma states it sees
          -> CheXagent, the visual validator, boxes each finding (with p_present);
             findings it cannot box are reported as "not localized"
          -> director summarizes MedGemma's opinion and CheXagent's visual validation
        The director only directs and summarizes; no LLM judges the image.

        python run_medgemma.py                              # Qwen as director/summarizer
        python run_medgemma.py --director gpt-4o
    MedGemma and Qwen run on the local Ollama; MEDGEMMA_MODEL overrides "medgemma:4b".
    """
    parser = argparse.ArgumentParser(description="MedRAX MedGemma + CheXagent visual validation UI")
    parser.add_argument(
        "--director",
        choices=DIRECTORS,
        default=os.getenv("MEDRAX_DIRECTOR", "qwen"),
        help="LLM that calls MedGemma and summarizes (default: qwen)",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama model tag for --director qwen (default: OLLAMA_MODEL or qwen3.8:latest)",
    )
    args = parser.parse_args()

    model, openai_kwargs = get_director(args.director, args.ollama_model)
    print(f"Starting MedGemma + CheXagent server with director {model}...")

    agent, tools_dict = initialize_agent(
        "medrax/docs/system_prompts.txt",
        tools_to_use=["DicomProcessorTool", "MedGemmaTool"],
        model_dir="C:/Users/hari/model-weights",  # CheXagent weights (visual proof)
        temp_dir="temp",
        device="cuda",
        model=model,
        temperature=0.7,
        top_p=0.95,
        openai_kwargs=openai_kwargs,
        edv="off",  # no LLM judges the image; CheXagent is the visual validator
        prompt_name="VISUAL_PROOF_SUMMARIZER",
    )
    demo = create_demo(
        agent,
        tools_dict,
        send_image=False,
        director_label=(
            f"{model} (director and summarizer) · opinion: MedGemma "
            "· visual validation: CheXagent"
        ),
    )

    demo.launch(server_name="127.0.0.1", server_port=8586, share=False)
