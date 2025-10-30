# config.py
"""
Central configuration for dataset paths, model names, hyperparameters.
Edit these values for running different experiments or datasets.
"""
from pathlib import Path

ROOT = Path(__file__).parent

# Data
DATA_ROOT = ROOT / "data" / "raw"
BATCH_SIZE = 32
NUM_WORKERS = 4
IMG_SIZE = 224  # ViT usual input size

# Model
VIT_PRETRAINED = "google/vit-base-patch16-224-in21k"  # good default
NUM_LABELS = 2  # change for your dataset

# Explainability
LIME_SAMPLES = 1000
SHAP_BACKGROUND_SAMPLES = 50

# Device
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
