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

| | requirement |
|---|---|
| Python | **3.11 exactly.** Not 3.12 or 3.13 — this project pins `numpy<2` and `tokenizers 0.19.1`, and neither publishes wheels for 3.12+, so `pip` tries to build them from source and fails |
| Disk | ~20 GB for model weights (~35 GB if you enable LLaVA-Med and MAIRA-2) |
| GPU | Optional. NVIDIA (CUDA) is fastest and unlocks two extra tools; Apple Silicon (MPS) works; CPU works but is slow |
| API key | An OpenAI key — GPT-4o orchestrates the tools and nothing runs without it |

---

### Windows with an NVIDIA GPU

This is the configuration that runs **every** tool except the image generator.

**1. Install Python 3.11** from [python.org](https://www.python.org/downloads/release/python-3119/).
Tick **"Add Python to PATH"** in the installer.

```powershell
py -3.11 --version        # must print 3.11.x
nvidia-smi                # confirms the driver and shows your VRAM
```

**2. Clone and create a virtual environment**

```powershell
git clone https://github.com/vaniinfo/MedRAX.git
cd MedRAX
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

> If PowerShell refuses with *"running scripts is disabled on this system"*, run this
> once and then activate again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> On `cmd.exe` the activate command is `.venv\Scripts\activate.bat` instead.

Your prompt should now start with `(.venv)`. Confirm the right interpreter is active —
if this path does not contain `.venv`, activation failed and you will install into
system Python:

```powershell
where python
```

**3. Install CUDA PyTorch FIRST**

This step is the one people miss. The default PyTorch wheel on Windows is **CPU-only**;
installing it first pins the CUDA build so the next step does not replace it. Check
[pytorch.org](https://pytorch.org/get-started/locally/) for the index URL matching your
CUDA version:

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> Use `python -m pip`, not bare `pip`, on Windows. Windows cannot replace `pip.exe`
> while it is running, so `pip install --upgrade pip` fails with
> *"To modify pip, please run the following command"*.

**4. Install MedRAX**

```powershell
python -m pip install -e .
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

That must print `cuda: True`. If it prints `False`, step 3 did not take — uninstall torch
and redo it.

**5. Add your OpenAI key**

Create a file named `.env` in the project root with one line. The encoding matters —
PowerShell's default adds a BOM that breaks parsing:

```powershell
Set-Content -Path .env -Value 'OPENAI_API_KEY=sk-your-key-here' -Encoding utf8
```

**6. Run**

```powershell
python main.py
```

Open **http://localhost:8585**. The first run downloads the model weights and will sit
quiet for several minutes before the URL appears.

---

### macOS (Apple Silicon) and Linux

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
echo 'OPENAI_API_KEY=sk-your-key-here' > .env
python main.py                      # http://localhost:8585
```

On Linux with an NVIDIA GPU, install CUDA PyTorch first as in Windows step 3.

---

### Which tools run where

Tool selection is automatic. `main.py` prints the device and the tool list at startup.

| tool | CUDA | Apple Silicon | CPU |
|---|:---:|:---:|:---:|
| ChestXRayClassifierTool | ✅ | ✅ | ✅ |
| ChestXRaySegmentationTool | ✅ | ✅ | ✅ |
| ChestXRayReportGeneratorTool | ✅ | ✅ | ✅ |
| XRayVQATool (CheXagent) | ✅ | ✅ | ✅ |
| ImageVisualizerTool / DicomProcessorTool | ✅ | ✅ | ✅ |
| **LlavaMedTool** | ✅ | ❌ | ❌ |
| **XRayPhraseGroundingTool** (MAIRA-2) | ✅ | ❌ | ❌ |
| ChestXRayGeneratorTool (RoentGen) | manual | manual | manual |

The two CUDA-only tools need `bitsandbytes` quantisation, which requires CUDA. RoentGen
is never enabled automatically because its weights are not publicly downloadable — you
must [request them from the authors](https://github.com/StanfordMIMI/RoentGen).

**VRAM budget** (approximate, with 8-bit quantisation):

| | VRAM |
|---|---|
| CheXagent + classifier + report generator | ~8 GB |
| \+ LLaVA-Med (7B) | ~8 GB |
| \+ MAIRA-2 (7B) | ~8 GB |

A 24 GB card runs everything. On 12–16 GB, use 4-bit or run fewer tools:

```powershell
$env:MEDRAX_QUANT="4bit"; python main.py
```

---

### Configuration

All optional, all environment variables. No code editing required.

| variable | default | purpose |
|---|---|---|
| `MEDRAX_DEVICE` | auto (`cuda`→`mps`→`cpu`) | force a device, e.g. `cpu` |
| `MEDRAX_MODEL_DIR` | `~/model-weights` | where weights are downloaded |
| `MEDRAX_TOOLS` | auto by device | comma-separated tool list |
| `MEDRAX_QUANT` | `8bit` | `4bit`, `8bit` or `none` for the CUDA-only tools |
| `MEDRAX_PROMPT` | `MEDICAL_ASSISTANT_EDV` | `MEDICAL_ASSISTANT` for the original prompt |
| `MEDRAX_VALIDATE` | `1` | `0` disables forced evidence validation |
| `MEDRAX_SHARE` | `0` | `1` publishes a public `gradio.live` link |
| `MEDRAX_LOG_CONSOLE` | `1` | `0` sends validation logs to file only |

Setting one for a single run:

```powershell
$env:MEDRAX_DEVICE="cpu"; python main.py      # PowerShell
set MEDRAX_DEVICE=cpu && python main.py       # cmd.exe
MEDRAX_DEVICE=cpu python main.py              # macOS / Linux
```

---

### Verify the install

CheXagent can load, generate fluent clinical text, and **ignore the image entirely** —
returning the same answer for a real X-ray and a blank one, with no error and no
warning. This asserts that answers actually depend on the image. Run it after any change
to the environment, the transformers version, or the device:

```powershell
python scripts/verify_vqa_vision.py      # exit 0 = healthy, 1 = broken
```

---

### Troubleshooting

| symptom | cause |
|---|---|
| `No matching distribution found` during install | Wrong Python. `python --version` inside the venv must say 3.11 |
| `torch.cuda.is_available()` is `False` | CPU-only PyTorch. Redo step 3 with the CUDA index URL |
| `running scripts is disabled` | PowerShell execution policy — see step 2 |
| Imports fail although install succeeded | The venv is not activated. Check `where python` |
| `OPENAI_API_KEY` error | `.env` missing, named `.env.txt`, or saved with a BOM |
| `XRayVQATool requires transformers 4.40.x` | Something upgraded transformers. See the note below |
| VQA answers look plausible but ignore the image | Run `scripts/verify_vqa_vision.py` |
| Port 8585 in use | Change `server_port` at the bottom of `main.py` |

### Gated and quantised models

**MAIRA-2 needs you to be logged in.** `microsoft/maira-2` is a gated Hugging Face repo,
so downloading it requires authentication:

```powershell
huggingface-cli login          # paste a token from hf.co/settings/tokens
```

If you are not logged in you get a 403 saying *"you are not in the authorized list"* —
**even when your account does have access**, because the request arrives anonymous. Log
in first; only if it still fails do you need to request access on
[the model page](https://huggingface.co/microsoft/maira-2).

Either way the tool is skipped with a message and the rest of MedRAX runs normally.

**Load LLaVA-Med unquantised if you have the VRAM.** `transformers==4.40` calls
accelerate's `dispatch_model` whenever a `device_map` is set, with no guard for quantised
models, and accelerate takes a `model.to(device)` shortcut for a single-device map --
which bitsandbytes rejects:

```
ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models
```

This affects any single-GPU bitsandbytes load under the pinned transformers. On a 24 GB
card, skip quantisation entirely:

```powershell
$env:MEDRAX_QUANT="none"; python main.py
```

Verified working on an RTX 3090: CheXagent (~6 GB) + LLaVA-Med bf16 (~14 GB) + the small
tools (~2 GB). On a 12-16 GB card you will need quantisation, and therefore will hit the
conflict above -- run without LLaVA-Med instead:

```powershell
$env:MEDRAX_TOOLS="ImageVisualizerTool,DicomProcessorTool,ChestXRayClassifierTool,ChestXRaySegmentationTool,ChestXRayReportGeneratorTool,XRayVQATool"
```

A tool that cannot load no longer stops startup -- it is reported and skipped.

> **A pinned dependency worth understanding.** CheXagent-2-3b only works with
> `transformers==4.40.x`. On newer versions it still loads and still produces confident,
> well-formed radiology text, but it stops attending to the image — silently, with no
> exception. `XRayVQATool` therefore refuses to start on any other version rather than
> fabricate. If you upgrade transformers for another model, CheXagent will stop working,
> and that is deliberate.
>
> **MAIRA-2 is untested against this pin.** It may need a newer transformers than
> CheXagent allows, in which case the two cannot run in the same environment. If you
> enable it and hit version errors, that is why.
<br><br><br>


## Tool Selection and Initialization

Tool selection is automatic — `main.py` picks the set your hardware can actually run and
prints it at startup. CUDA machines additionally get `LlavaMedTool` and
`XRayPhraseGroundingTool`, which need bitsandbytes quantisation and therefore cannot run
on Apple Silicon or CPU.

To choose explicitly, set `MEDRAX_TOOLS` rather than editing the source:

```powershell
$env:MEDRAX_TOOLS="ImageVisualizerTool,ChestXRayClassifierTool,XRayVQATool"
python main.py
```

```bash
MEDRAX_TOOLS="ImageVisualizerTool,ChestXRayClassifierTool,XRayVQATool" python main.py
```

Embedding MedRAX in your own code works as before:

```python
agent, tools_dict = initialize_agent(
    "medrax/docs/system_prompts.txt",
    tools_to_use=["ImageVisualizerTool", "ChestXRayClassifierTool"],
    model_dir="/path/to/weights",
    device="cuda",
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
