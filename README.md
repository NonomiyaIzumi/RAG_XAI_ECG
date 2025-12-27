# RAG_ECG

Utilities for preprocessing PTB-XL ECGs, training/inference with a CNN, generating Grad-CAM visual explanations, and producing multimodal ECG interpretation reports using LLM/MLLM (Gemini/OpenAI).

## What’s in this repo

- `preprocessing.py`: Loads PTB-XL waveform records, applies basic filtering, extracts R-peaks, and exports:
  - processed arrays (`.npy`)
  - rendered ECG images (`.png`)
  - multi-label targets (`.npy`)

- `Model/Train/resnet50.py`: Example training script for a ResNet-based classifier.
- `Model/Predict/Predict.py`: Example inference script (generates predictions from processed inputs).

- `GradCAM.py`: Generates Grad-CAM heatmaps/overlays for processed ECG images using a trained CNN.

- `RAG_ECG_XAI.py`: Multimodal pipeline that combines:
  1) Grad-CAM (highest priority)
  2) the original processed ECG image
  3) a “facts pack” (predictions/probabilities)
  4) an ECG knowledge text (`ECG_Interpretation_Guide.txt`)

  and produces one JSON object per record.

- `MLLM_Referee/`: Scripts to compare/choose between outputs with Gemini or OpenAI.
- `Bertscore/Bertscore.py`: Computes BERTScore between generated impressions and PTB-XL reference reports.

## Folder structure

```text
RAG_ECG/
  ECG_Interpretation_Guide.txt
  ECG_Reading_Guide.py
  GradCAM.py
  preprocessing.py
  RAG_ECG_XAI.py
  scp_codes.jsonl
  requirements.txt
  README.md

  Bertscore/
    Bertscore.py

  Books/
    ... (PDFs used to build ECG_Interpretation_Guide.txt)

  Model/
    best_resnet50_fold4.pth
    best_resnet50_ecg_model.pth
    Predict/
      Predict.py
    Train/
      resnet50.py

  ecg_processed/
    ... (processed .npy arrays)

  processed_images/
    ... (rendered ECG images .png)

  processed_labels/
    ... (label arrays .npy)

  gradcam_out/
    ... (Grad-CAM overlays / heatmaps)

  MLLM_Referee/
    Gemini_choice.py
    Gemini_Score.py
    Gpt_choice.py
    ...

  ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/
    ... (PTB-XL dataset files)
```

## Setup (Windows)

### 1) Create a virtual environment

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bat
pip install -r requirements.txt
```

Notes:
- `torch`/`torchvision` may need a custom install (CUDA vs CPU). If you want CUDA, install the correct wheel from PyTorch’s official instructions, then re-run `pip install -r requirements.txt`.

## Data / paths

Several scripts use **hard-coded paths** at the top of the file (no CLI arguments). You will usually need to edit these constants before running:

- PTB-XL waveform root directory (e.g. `record_base` in `preprocessing.py`)
- PTB-XL metadata CSV paths (e.g. `ptbxl_database*.csv`, `scp_statements.csv`)
- Model weights path (e.g. `WEIGHTS` in `GradCAM.py`)
- Input/output directories (`processed_images`, `gradcam_out`, etc.)

## API keys

Different scripts expect keys in different places:

- OpenAI scripts (`ECG_Reading_Guide.py`, `MLLM_Referee/Gpt_choice.py`)
  - Set environment variable `OPENAI_API_KEY` (or `OPENAI_API_KEY_1` depending on the script).

- Gemini scripts (`RAG_ECG_XAI.py`, `MLLM_Referee/Gemini_choice.py`)
  - These currently use an in-code `API_KEYS = [...]` list. Put valid keys there (or refactor to read from env vars if you prefer).

## Typical run order

1) Preprocess PTB-XL into arrays/images:

```bat
python preprocessing.py
```

2) Train a model (optional):

```bat
python Model\Train\resnet50.py
```

3) Run inference (optional):

```bat
python Model\Predict\Predict.py
```

4) Generate Grad-CAM overlays:

```bat
python GradCAM.py
```

5) Generate multimodal JSON reports (Gemini):

```bat
python RAG_ECG_XAI.py
```

6) Evaluate with BERTScore (optional):

```bat
python Bertscore\Bertscore.py
```

## Outputs

Outputs are written to files/directories configured at the top of each script (e.g. `processed_images*`, `gradcam_out`, `New_method_outputs.jsonl`, `bertscore_*.csv`).
