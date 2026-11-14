<h1 align="center">
🤖 MedRAX: Medical Reasoning Agent for Chest X-ray
</h1>
<p align="center"> <a href="https://arxiv.org/abs/2502.02673" target="_blank"><img src="https://img.shields.io/badge/arXiv-ICML 2025-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a> <a href="https://github.com/bowang-lab/MedRAX"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a> <a href="https://huggingface.co/datasets/wanglab/chest-agent-bench"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Dataset"></a> </p>

![](assets/demo_fast.gif?autoplay=1)

<br>

## Abstract
Chest X-rays (CXRs) play an integral role in driving critical decisions in disease management and patient care. While recent innovations have led to specialized models for various CXR interpretation tasks, these solutions often operate in isolation, limiting their practical utility in clinical practice. We present MedRAX, the first versatile AI agent that seamlessly integrates state-of-the-art CXR analysis tools and multimodal large language models into a unified framework. MedRAX dynamically leverages these models to address complex medical queries without requiring additional training. To rigorously evaluate its capabilities, we introduce ChestAgentBench, a comprehensive benchmark containing 2,500 complex medical queries across 7 diverse categories. Our experiments demonstrate that MedRAX achieves state-of-the-art performance compared to both open-source and proprietary models, representing a significant step toward the practical deployment of automated CXR interpretation systems.
<br><br>


## MedRAX
MedRAX is built on a robust technical foundation:
- **Core Architecture**: Built on LangChain and LangGraph frameworks
- **Language Model**: Uses GPT-4o with vision capabilities as the backbone LLM
- **Deployment**: Supports both local and cloud-based deployments
- **Interface**: Production-ready interface built with Gradio
- **Modular Design**: Tool-agnostic architecture allowing easy integration of new capabilities

### Integrated Tools
- **Visual QA**: Utilizes CheXagent and LLaVA-Med for complex visual understanding and medical reasoning
- **Segmentation**: Employs MedSAM and PSPNet model trained on ChestX-Det for precise anatomical structure identification
- **Grounding**: Uses Maira-2 for localizing specific findings in medical images
- **Report Generation**: Implements SwinV2 Transformer trained on CheXpert Plus for detailed medical reporting
- **Disease Classification**: Leverages DenseNet-121 from TorchXRayVision for detecting 18 pathology classes
- **X-ray Generation**: Utilizes RoentGen for synthetic CXR generation
- **Utilities**: Includes DICOM processing, visualization tools, and custom plotting capabilities
<br><br>


## ChestAgentBench
We introduce ChestAgentBench, a comprehensive evaluation framework with 2,500 complex medical queries across 7 categories, built from 675 expert-curated clinical cases. The benchmark evaluates complex multi-step reasoning in CXR interpretation through:

- Detection
- Classification
- Localization
- Comparison
- Relationship
- Diagnosis
- Characterization

Download the benchmark: [ChestAgentBench on Hugging Face](https://huggingface.co/datasets/wanglab/chest-agent-bench)
```
huggingface-cli download wanglab/chestagentbench --repo-type dataset --local-dir chestagentbench
```

Unzip the Eurorad figures to your local `MedMAX` directory.
```
unzip chestagentbench/figures.zip
```

To evaluate with GPT-4o, set your OpenAI API key and run the quickstart script.
```
export OPENAI_API_KEY="<your-openai-api-key>"
python quickstart.py \
    --model chatgpt-4o-latest \
    --temperature 0.2 \
    --max-cases 2 \
    --log-prefix chatgpt-4o-latest \
    --use-urls
```


<br>

## Installation
### Prerequisites
- Python 3.8+
- CUDA/GPU for best performance

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/bowang-lab/MedRAX.git
cd MedRAX

# Install package
pip install -e .
```

### Getting Started
```bash
# Start the Gradio interface
python main.py
```
or if you run into permission issues
```bash
sudo -E env "PATH=$PATH" python main.py
```
You need to setup the `model_dir` inside `main.py` to the directory where you want to download or already have the weights of above tools from Hugging Face.
Comment out the tools that you do not have access to.
Make sure to setup your OpenAI API key in `.env` file!
<br><br><br>


## Tool Selection and Initialization

MedRAX supports selective tool initialization, allowing you to use only the tools you need. Tools can be specified when initializing the agent (look at `main.py`):

```python
selected_tools = [
    "ImageVisualizerTool",
    "ChestXRayClassifierTool",
    "ChestXRaySegmentationTool",
    # Add or remove tools as needed
]

agent, tools_dict = initialize_agent(
    "medrax/docs/system_prompts.txt",
    tools_to_use=selected_tools,
    model_dir="/model-weights"
)
```

<br><br>
## Automatically Downloaded Models

The following tools will automatically download their model weights when initialized:

### Classification Tool
```python
ChestXRayClassifierTool(device=device)
```

### Segmentation Tool
```python
ChestXRaySegmentationTool(device=device)
```

### Grounding Tool
```python
XRayPhraseGroundingTool(
    cache_dir=model_dir, 
    temp_dir=temp_dir, 
    load_in_8bit=True, 
    device=device
)
```
- Maira-2 weights download to specified `cache_dir`
- 8-bit and 4-bit quantization available for reduced memory usage

### LLaVA-Med Tool
```python
LlavaMedTool(
    cache_dir=model_dir, 
    device=device, 
    load_in_8bit=True
)
```
- Automatic weight download to `cache_dir`
- 8-bit and 4-bit quantization available for reduced memory usage

### Report Generation Tool
```python
ChestXRayReportGeneratorTool(
    cache_dir=model_dir, 
    device=device
)
```

### Visual QA Tool
```python
XRayVQATool(
    cache_dir=model_dir, 
    device=device
)
```
- CheXagent weights download automatically

### MedSAM Tool
```
Support for MedSAM segmentation will be added in a future update.
```

### Utility Tools
No additional model weights required:
```python
ImageVisualizerTool()
DicomProcessorTool(temp_dir=temp_dir)
```
<br>

## Manual Setup Required

### Image Generation Tool
```python
ChestXRayGeneratorTool(
    model_path=f"{model_dir}/roentgen", 
    temp_dir=temp_dir, 
    device=device
)
```
- RoentGen weights require manual setup:
  1. Contact authors: https://github.com/StanfordMIMI/RoentGen
  2. Place weights in `{model_dir}/roentgen`
  3. Optional tool, can be excluded if not needed
<br>

## Configuration Notes

### Required Parameters
- `model_dir` or `cache_dir`: Base directory for model weights that Hugging Face uses
- `temp_dir`: Directory for temporary files
- `device`: "cuda" for GPU, "cpu" for CPU-only

### Memory Management
- Consider selective tool initialization for resource constraints
- Use 8-bit quantization where available
- Some tools (LLaVA-Med, Grounding) are more resource-intensive
<br>

### Local LLMs
If you are running a local LLM using frameworks like [Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/), you need to configure your environment variables accordingly. For example:
```
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
```
<br>

### Optional: OpenAI-compatible Providers

MedRAX supports OpenAI-compatible APIs, allowing regional or local LLM providers to serve as alternative backends.

For example, to use **Qwen3-VL** via [Alibaba Cloud DashScope](https://bailian.console.aliyun.com/?tab=model#/model-market), set the following environment variables:

```bash
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_API_KEY="<your-dashscope-api-key>"
export OPENAI_MODEL="qwen3-vl-235b-a22b-instruct"
```
<br>

## Research Extension: Review Guide

This fork adds a research pipeline for **claim-blinded evidence verification** of chest X-ray reports: a trained
MedGemma writes (longitudinal) reports, and CheXagent independently checks each stated finding without being told the
claim. The components below are listed in the order data flows through the pipeline, so each one's inputs have
already been reviewed when you reach it. If time is short, review **1, 4 and 7** first: that is where the paper's main
claim lives.

### 1. Protocols and pre-registration
- **Purpose:** Before running a test, we write down exactly what we will measure and what counts as success, and save it with a date. That way nobody can change the rules after seeing the results, like a teacher writing the answer key before the exam instead of after.
- Files: [`experiments/*_protocol.md`](experiments/), [`experiments/evidence_engine_design.md`](experiments/evidence_engine_design.md), git tags `vindr-amendment-1` and
  `vindr-amendment-2`, [`experiments/results/frozen_dev_params.json`](experiments/results/frozen_dev_params.json) (+ `.sha256`).
- Check: every reported result traces to a protocol committed before the test data was read; frozen thresholds
  match what the code actually loads; post-hoc analyses are labelled as such.

### 2. Report generator (trained MedGemma)
- **Purpose:** This is the AI that looks at a chest X-ray and writes a report, like a student writing an answer. Our team trained it, so we must make sure we always use our version and never quietly swap or re-run it until the answers look better.
- Files: [`experiments/edv_trained.py`](experiments/edv_trained.py), [`experiments/edv_trained_protocol.md`](experiments/edv_trained_protocol.md).
- Check: only the team's trained MedGemma is used (never the base model); prompts and decoding are fixed; reports
  are generated once and cached, never regenerated after results were seen.

### 3. Claim extraction (report text to findings)
- **Purpose:** Reports are written in sentences. This step turns each report into a simple checklist (for example: fluid in the lungs, yes, left side). If this "translator" makes mistakes, every score after it is wrong too, so we check it against answers marked by hand.
- The reader model turns free text into (condition, side, change) labels, for both model and reference reports.
- Files: [`experiments/reader_validation_protocol.md`](experiments/reader_validation_protocol.md), `experiments/results/pilot_*_auto_labels.csv`.
- Check: reader accuracy against hand labels; handling of "possible" and hedged wording; laterality parsing. Errors
  here propagate into every downstream number, including the reference column of the radiologist workbook.

### 4. Evidence Engine (CheXagent verification, the core of the paper)
- **Purpose:** A second AI (CheXagent) checks each finding the first AI wrote. The key trick: we ask it "is there fluid?" without telling it what the first AI said, so it cannot just copy. Like a second doctor giving an opinion without reading the first doctor's notes.
- Files: [`experiments/build_evidence_engine_v1.py`](experiments/build_evidence_engine_v1.py), [`experiments/verify_evidence_engine_v1.py`](experiments/verify_evidence_engine_v1.py),
  [`experiments/evidence_engine_d1_d2.py`](experiments/evidence_engine_d1_d2.py), [`experiments/evidence_engine_d3.py`](experiments/evidence_engine_d3.py), [`medrax/tools/xray_vqa.py`](medrax/tools/xray_vqa.py).
- Check:
  - claim-blinding: the prompt is the condition name only, never the report's side or wording;
  - calibration: support and evidence-against cut-offs; values between them are Indeterminate (shown on the forms
    as "Not agree" together with Evidence against);
  - the image-ignored fix in [`xray_vqa.py`](medrax/tools/xray_vqa.py);
  - possible training-data overlap (VinDr AUCs of 0.98-0.999 suggest CheXagent saw VinDr during training).

### 5. Grounding and laterality
- **Purpose:** This checks whether the AI points to the right place on the image, especially the right versus left side. We found that if you tell an AI "left side", it will point left even when that is wrong, so we never give it the side in the question.
- Files: [`experiments/condition_aware_grounding.py`](experiments/condition_aware_grounding.py), [`experiments/grounding_recall_protocol.md`](experiments/grounding_recall_protocol.md), VinDr Part B results.
- Check: locate prompts contain no side words; the side-withheld vs side-named comparison (H1/H2); MAIRA-2's
  default to the patient's right. Wording: "claim-blinded" grounding, never "independent" or "unbiased".

### 6. Longitudinal / interval change (EDV)
- **Purpose:** Patients often have an older X-ray and a new one. This part asks: did things get better, worse or stay the same? It compares the two pictures for each finding, like comparing two photos of a cut a week apart.
- Files: [`experiments/edv_longitudinal.py`](experiments/edv_longitudinal.py), [`experiments/edv_change.py`](experiments/edv_change.py) and their protocols.
- Check: how prior/current pairs are selected (NIH pairs); how change claims are scored; the change check built from
  separate prior and current presence verdicts.

### 7. Radiologist evaluation
- **Purpose:** Real X-ray doctors (radiologists) graded the AI's answers from 1 (wrong) to 5 (right). This is the human check of whether the whole system actually helps, and we must make sure their scores are counted fairly and the same way every time.
- Files: [`experiments/radiologist_evaluation_protocol.md`](experiments/radiologist_evaluation_protocol.md); form builders ([`experiments/focused_conditions_report.py`](experiments/focused_conditions_report.py),
  [`experiments/regional_focal_reports_v2.py`](experiments/regional_focal_reports_v2.py)); score extraction and workbook ([`experiments/doctor_scores_to_excel.py`](experiments/doctor_scores_to_excel.py),
  [`experiments/doctor_scores_analysis_tabs.py`](experiments/doctor_scores_analysis_tabs.py), [`experiments/disorder_agreement_tab.py`](experiments/disorder_agreement_tab.py)).
- Check:
  - the rule turning a 1-5 score into a present/absent call (model-vs-doctor columns count 3-5 as agreement;
    reference-vs-doctor and doctor-vs-doctor columns exclude 3 as uncertain);
  - precedence when the same radiologist scored a finding on two forms;
  - excluded conditions (emphysema, pleural thickening: not reliably assessable on chest X-ray);
  - what the radiologists actually saw on the forms;
  - whether a radiologist's score is independent of the model output shown to them.

### 8. Statistics and reproducibility
- **Purpose:** Numbers can look good by luck, especially with few cases. Statistics tell us how sure we can be, and reproducibility means someone else can rerun our code and get the same numbers.
- Files: [`experiments/results/REPRODUCE.md`](experiments/results/REPRODUCE.md) and the results `.md` files.
- Check: bootstrap CIs, McNemar tests and Holm correction; unit of analysis (per finding, not pooled rows); reported
  numbers are computed from the cached model outputs (`*.jsonl`); model revisions and hashes are pinned.

### 9. Data governance
- **Purpose:** Some X-rays come from real hospital patients and can only be used under strict rules. This keeps private data out of shared places and makes sure downloaded AI code cannot run anything unsafe on our computers.
- MIMIC-CXR outputs stay in the git-ignored `logs/` folder (PhysioNet data use agreement); only NIH and VinDr numbers
  go in [`experiments/results/`](experiments/results/).
- Downloaded model code is reviewed and loaded by path; no `trust_remote_code` for new models.

### Lower priority
- **Purpose:** These parts exist in the project but do not affect the paper's results, so they can wait.
- The upstream MedRAX agent and Gradio UI ([`main.py`](main.py), [`interface.py`](interface.py), [`medrax/agent/`](medrax/agent/)): the paper's results do not
  go through them.
- Exploratory visualizations: the BioViL-T heat map is an exploratory overlay only; prior-vs-current subtraction
  failed validation and is not used.
<br>

## Star History
<div align="center">
  
[![Star History Chart](https://api.star-history.com/svg?repos=bowang-lab/MedRAX&type=Date)](https://star-history.com/#bowang-lab/MedRAX&Date)

</div>
<br>


## Authors
- **Adibvafa Fallahpour**¹²³⁴ * (adibvafa.fallahpour@mail.utoronto.ca)
- ****Jun Ma****²³ *
- **Alif Munim**³⁵ *
- ****Hongwei Lyu****³
- ****Bo Wang****¹²³⁶

¹ Department of Computer Science, University of Toronto, Toronto, Canada <br>
² Vector Institute, Toronto, Canada <br>
³ University Health Network, Toronto, Canada <br>
⁴ Cohere, Toronto, Canada <br>
⁵ Cohere Labs, Toronto, Canada <br>
⁶ Department of Laboratory Medicine and Pathobiology, University of Toronto, Toronto, Canada

<br>
* Equal contribution
<br><br>


## Citation
If you find this work useful, please cite our paper:
```bibtex
@misc{fallahpour2025medraxmedicalreasoningagent,
      title={MedRAX: Medical Reasoning Agent for Chest X-ray}, 
      author={Adibvafa Fallahpour and Jun Ma and Alif Munim and Hongwei Lyu and Bo Wang},
      year={2025},
      eprint={2502.02673},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.02673}, 
}
```

---
<p align="center">
Made with ❤️ at University of Toronto, Vector Institute, and University Health Network
</p>
