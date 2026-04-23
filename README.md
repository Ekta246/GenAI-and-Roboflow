# GenAI-and-Roboflow

Utilities for combining Roboflow workflows, multimodal LLM verification, synthetic data generation, and annotation for fire/smoke security-surveillance use cases.

## What this repo includes

- Roboflow workflow inference + LLM scene assessment pipeline
- SAM3-based auto-annotation (Pascal VOC output)
- Synthetic image generation scripts
- Dataset download helper for Roboflow project versions
- Model quantization utility script

## Main scripts

- `practise-api.py`  
  Runs Roboflow workflow inference on images, draws detections, and gets structured threat assessment from an LLM.

- `annotate-generated-images.py`  
  Auto-annotates images using SAM3 (or YOLO backend) and writes Pascal VOC XML + optional visualizations.

- `generate-images-grok.py`  
  Generates surveillance-style synthetic fire/smoke images using xAI API.

- `sf-download.py`  
  Downloads a Roboflow dataset version from a workspace/project.

- `test-xai-api.py`  
  Quick connectivity/model test for xAI API.

- `quantize_model.py`  
  Quantization helper script for model optimization experiments.

## Environment setup

Create and fill `.env` in repo root:

```env
ROBOFLOW_API_KEY=...
ROBOFLOW_DATASET_API_KEY=...
OPENAI_API_KEY=...
XAI_API_KEY=...
GOOGLE_API_KEY=...
```

The code reads keys from environment variables / `.env`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install python-dotenv openai inference-sdk roboflow opencv-python numpy ultralytics
```

## Typical usage

### 1) Run workflow + LLM analysis

```bash
python practise-api.py
```

### 2) Auto-annotate generated images

```bash
python annotate-generated-images.py
```

### 3) Generate synthetic images

```bash
python generate-images-grok.py
```

### 4) Download dataset from Roboflow

```bash
python sf-download.py
```

## Notes

- Check script `CONFIG` blocks for input/output paths before running.
- `.env` is intentionally git-ignored; use `.env.example` pattern if sharing templates.
- Rotate API keys if they were ever committed or shared.
