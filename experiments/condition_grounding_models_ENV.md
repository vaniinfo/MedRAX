# Condition-specific grounding models: reproducible environment

Script: `experiments/condition_grounding_models.py` (`validate`, `cases`). Runs in the project `.venv`.

## Packages (as used, 2026-11-08)
| Package | Version |
|---|---|
| torch | 2.6.0+cu124 (CUDA 12.4) |
| timm | 1.0.30 |
| segmentation_models_pytorch | 0.6.0 (only new install; pulled no other changes) |
| safetensors | 0.8.0 |
| huggingface_hub | 0.36.2 |
| numpy 2.4.6, Pillow 11.3.0, scipy 1.17.1, scikit-image 0.26.0, torchxrayvision 1.5.4 (lung segmentation for the anatomy check) |

Hardware: NVIDIA RTX 3090 (24 GB). Each model needs < 2 GB GPU memory at 512 x 512; CPU inference also works (slower).

## Weights (downloaded into their own folders; nothing executed from them except the reviewed `unet.py`)
| Model | Hugging Face revision | Local folder | File | Loading |
|---|---|---|---|---|
| vasandesh/pneumothorax-unet-efficientnet-b4 | d9a9fb73a8bcefcce662905eba91b75ae12dfba4 | `C:/Users/hari/model-weights/siim_ptx_vasandesh` | `pytorch_model.bin` (81.7 MB, pickle) | `torch.load(weights_only=True)` into `smp.Unet("efficientnet-b4")` + `Linear(16,1)` head; all keys matched |
| ianpan/pneumothorax-cxr | f00a7af5cdd019b2a51d18ea585c947b7ecb27eb | `C:/Users/hari/model-weights/ianpan_pneumothorax-cxr` | `model.safetensors` (89 MB) | timm `tf_efficientnetv2_s` (features_only, 1 channel) + the repo's `unet.py` loaded by path (reviewed: model layers only, no I/O); **no `trust_remote_code`**; all keys matched |
| ianpan/pneumonia-cxr | 1ab2c07102f3a5eeaecc419d6a63bcef5522f314 | `C:/Users/hari/model-weights/ianpan_pneumonia-cxr` | `model.safetensors` (89 MB) | as above |

Download (reproduce): `huggingface_hub.hf_hub_download(repo, file, revision=<sha>, local_dir=<folder>)`.

## Preprocessing (from the model code / card; reversed exactly when rendering)
- vasandesh: grey -> [0,1] -> ImageNet mean/std, replicated to 3 channels, zero-padded to a square then 512 x 512 (SIIM images are square; padding keeps geometry). Output mask resized back to the padded square and cropped.
- ianpan (both): grey 0-255 stretched to 512 x 512 (albumentations `Resize`, as in `modeling.py`), scaled to [-1, 1]; output mask (upsampled to 512 inside the model) stretched back to the original size.
- Region = probability > 0.5 (no native threshold documented); components < 0.05% of the image removed.

## Not obtainable here
- `yolo5x_vindr.pt` (YOLOv5x trained on VinDr-CXR, in github.com/FinnBehrendt/node21-submit, Apache-2.0): GitHub's large-file host (media.githubusercontent.com) is blocked from this network (SSL error). It would also need the YOLOv5 code base and a pickle-safety review.
