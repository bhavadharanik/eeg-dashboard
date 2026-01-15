# EEG Emotion Recognition Dashboard

A Streamlit dashboard for EEG-based emotion recognition using CNN and CLIP models with explainable AI (XAI) features.

## Features

- **Two Model Architectures**: CNN baseline and CLIP-based classifier
- **Explainable AI**: GradCAM and SHAP visualizations
- **Interactive Dashboard**: Real-time model comparison and analysis

## Project Structure

```
eeg-dashboard/
├── app.py                          # Streamlit dashboard
├── README.md
├── requirements.txt
│
├── scripts/                        # Experiment scripts
│   ├── experiments/
│   │   ├── run_cnn_experiment.py   # CNN baseline experiment
│   │   └── run_clip_experiment.py  # CLIP experiment
│   └── analysis/
│       └── generate_xai_comprehensive.py
│
├── src/                            # Source code
│   ├── models/                     # Model definitions
│   │   ├── cnn.py                  # CNN architecture
│   │   └── clip.py                 # CLIP wrapper
│   ├── utils/                      # Utilities
│   │   └── viz.py                  # Visualization functions
│   └── xai/                        # Explainability
│       ├── gradcam.py              # GradCAM implementation
│       ├── shap_explain.py         # SHAP explanations
│       └── clip_similarity.py      # CLIP similarity scoring
│
├── models/                         # Saved models
├── data/                           # Data (not in git)
└── results/                        # Outputs
```

## Installation

### Prerequisites
- Python 3.10+
- DEAP dataset (available from [Kaggle](https://www.kaggle.com/datasets/manh123df/deap-dataset))

### Setup

```bash
# Clone repository
git clone https://github.com/bhavadharanik/eeg-dashboard.git
cd eeg-dashboard

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Key packages:
- `torch` >= 2.0.0
- `open-clip-torch` >= 2.20.0
- `streamlit` >= 1.28.0
- `scikit-learn` >= 1.3.0
- `numpy`, `scipy`, `matplotlib`, `plotly`

## Usage

### Run Dashboard

```bash
streamlit run app.py
```

### Run Experiments

```bash
# CNN experiment (32-subject LOSO)
python scripts/experiments/run_cnn_experiment.py

# CLIP experiment (32-subject LOSO)
python scripts/experiments/run_clip_experiment.py
```

## Models

**CLIP + MLP**
- Frozen ViT-B/32 encoder (512-dim features)
- Trainable MLP classifier
- Focal loss + SMOTE + threshold optimization

**CNN Baseline**
- 3 Convolutional blocks with BatchNorm + ReLU
- Global Average Pooling + MLP classifier
- Balanced class weights

## Acknowledgements

- DEAP dataset: Koelstra et al. (2012)
- OpenCLIP: [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- Streamlit, PyTorch, scikit-learn
