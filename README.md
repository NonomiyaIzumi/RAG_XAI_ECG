# RAG-XAI-ECG: Enhancing Explainability in Cardiac Diagnosis

This repository contains the implementation for **Enhancing Explainability in Cardiac Diagnosis By Using Retrieval-Augmented Multimodal LLMs**. It provides a comprehensive pipeline for ECG analysis, combining Deep Learning (ResNet50), Explainable AI (Grad-CAM), and Retrieval-Augmented Generation (RAG) with Multimodal LLMs (Gemini/OpenAI) to generate interpretable cardiac diagnosis reports.

## 🌟 Key Features

- **ECG Preprocessing**: Automated filtering, R-peak detection, and conversion of PTB-XL waveform data into processed arrays and images.
- **Deep Learning Model**: ResNet50-based classifier for multi-label ECG diagnosis.
- **Explainable AI (XAI)**: Grad-CAM heatmap generation to visualize regions of interest in ECG signals.
- **RAG Pipeline**: A multimodal system that integrates:
  - Original ECG images
  - Grad-CAM visual explanations
  - Diagnostic predictions (Facts Pack)
  - Medical Knowledge Base (`ECG_Interpretation_Guide.txt`)
- **Automated Reporting**: Generates detailed clinical reports using Gemini 1.5 Flash/Pro or OpenAI models.
- **Evaluation**: BERTScore-based evaluation of generated reports against reference interpretations.

## 📂 Project Structure

```text
RAG_ECG/
├── Bertscore/                  # Evaluation scripts
│   └── Bertscore.py
├── Books/                      # Source materials for Knowledge Base
├── ecg_processed/              # Processed ECG numpy arrays (.npy)
├── gradcam_out/                # Generated Grad-CAM heatmaps
├── MLLM_Referee/               # Scripts for comparing LLM outputs
├── Model/                      # Model training and inference
│   ├── Predict/                # Inference scripts
│   ├── Train/                  # Training scripts
│   ├── best_resnet50_ecg_model.pth
│   └── best_resnet50_fold4.pth
├── processed_images/           # Rendered ECG images (.png)
├── ptb-xl-a-large.../          # PTB-XL Dataset folder
├── ECG_Interpretation_Guide.txt # RAG Knowledge Base
├── ECG_Reading_Guide.py        # Guide processing utility
├── GradCAM.py                  # Grad-CAM generation script
├── preprocessing.py            # Data preprocessing script
├── RAG_ECG_XAI.py              # Main RAG pipeline script
├── requirements.txt            # Python dependencies
└── scp_codes.jsonl             # SCP code definitions
```

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NonomiyaIzumi/RAG_XAI_ECG.git
    cd RAG_XAI_ECG
    ```

2.  **Create and activate a virtual environment (Optional but recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Dataset Setup:**
    - Download the [PTB-XL dataset](https://physionet.org/content/ptb-xl/).
    - Extract it into the repository folder (default expected path: `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1`).
    - Ensure `ptbxl_database.csv` is available in the dataset folder.

## 🛠️ Usage

### 1. Preprocessing
Convert raw PTB-XL waveforms into processed images and numpy arrays.
```bash
python preprocessing.py
```
*Outputs to `ecg_processed/` and `processed_images/`.*

### 2. Model Training & Inference
Train the ResNet50 model or run inference on processed data.
- **Training**: `python Model/Train/resnet50.py`
- **Inference**: `python Model/Predict/Predict.py`

### 3. Generate Explanations (Grad-CAM)
Generate visual heatmaps for the model's predictions.
```bash
python GradCAM.py
```
*Outputs to `gradcam_out/`.*

### 4. Run RAG Pipeline
Generate diagnostic reports using the Multimodal RAG system.
**Note**: You must configure your API keys in `RAG_ECG_XAI.py` before running.
```python
# In RAG_ECG_XAI.py
API_KEYS = [
    "YOUR_GEMINI_API_KEY_HERE",
]
```
Run the pipeline:
```bash
python RAG_ECG_XAI.py
```
*Outputs generated reports to `New_method_pipeline.jsonl`.*

### 5. Evaluation
Evaluate the quality of generated reports using BERTScore.
```bash
python Bertscore/Bertscore.py
```

## 📄 Citation
If you use this code or dataset in your research, please refer to the associated paper: `XAI_ECG_paper.pdf`.
