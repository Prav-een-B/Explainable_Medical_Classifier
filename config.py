# config.py
"""
Central configuration for dataset paths, model names, hyperparameters.
Edit these values for running different experiments or datasets.
"""
from pathlib import Path
import torch

ROOT = Path(__file__).parent

# Data
DATA_ROOT = ROOT / "data" / "raw"
BATCH_SIZE = 32
NUM_WORKERS = 4
IMG_SIZE = 224  # ViT usual input size

# --- Multimodal Configuration ---
# Map modalities to their characteristics and required model type
MODALITY_CONFIG = {
    "XRAY": {"dim": 2, "channels": 3, "size": 224, "model_type": "2D"},
    "ULTRASOUND": {"dim": 2, "channels": 3, "size": 224, "model_type": "2D"},
    # Conceptual sizes for 3D data (CT/MRI volumes)
    "CT": {"dim": 3, "channels": 1, "depth": 32, "size": 128, "model_type": "3D"}, 
    "MRI": {"dim": 3, "channels": 1, "depth": 32, "size": 128, "model_type": "3D"},
}
DEFAULT_MODALITY = "XRAY" 

# Model
VIT_2D_PRETRAINED = "google/vit-base-patch16-224-in21k"  # 2D ViT
VIT_3D_PRETRAINED = "mock/3d/vit/weights" # Placeholder for a 3D model
NUM_LABELS = 2  # change for your dataset (assuming binary classification)

# Explainability
LIME_SAMPLES = 1000
SHAP_BACKGROUND_SAMPLES = 50

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
