from pathlib import Path
import torch
import os

# ============================================================
# 🧭 PATHS & ROOT SETTINGS
# ============================================================

# Root directory for Kaggle (writable)
ROOT = Path("/kaggle/working")

# Default dataset roots — update only if you rename your Kaggle datasets
KAGGLE_PATHS = {
    "XRAY": Path("/kaggle/input/data"),  # NIH Chest X-ray
    "SKIN": Path("/kaggle/input/skin-cancer-mnist-ham10000"),
    "MRI":  Path("/kaggle/input/brain-mri-images-for-brain-tumor-detection"),
}

# Global data root
DATA_ROOT = ROOT / "data"
CLASS_MAP_JSON = DATA_ROOT / "class_map.json"

# ============================================================
# 🧮 DATA CONFIGURATION
# ============================================================

# 2D modalities (e.g., XRAY, SKIN)
IMG_SIZE_2D = 224

# 3D modalities (e.g., MRI)
IMG_SIZE_3D = (96, 96, 96)  # (Depth, Height, Width)

# Modality-specific config
MODALITY_CONFIG = {
    "XRAY": {"dim": 2, "channels": 3, "size": IMG_SIZE_2D, "model_type": "2D"},
    "SKIN": {"dim": 2, "channels": 3, "size": IMG_SIZE_2D, "model_type": "2D"},
    "MRI":  {"dim": 3, "channels": 1, "size": IMG_SIZE_3D, "model_type": "3D"},
}

# ============================================================
# ⚙️ TRAINING CONFIGURATION
# ============================================================

LEARNING_RATE = 2e-4
EPOCHS = 6

# Batches and workers (tuned for Kaggle)
BATCH_SIZE = 8
NUM_WORKERS = 2

# Optional K-Fold cross-validation (keep small for speed)
K_FOLDS = 3

# ============================================================
# 🧠 MODEL CONFIGURATION
# ============================================================

# Pretrained Vision Transformer (2D)
VIT_2D_PRETRAINED = "google/vit-base-patch16-224-in21k"

# Optional 3D model backbone (for MRI or MONAI)
VIT_3D_PRETRAINED = "monai/vitautoenc"  # not always used

# Local checkpoint paths (will auto-create)
SAVE_PATH = ROOT / "models" / "weights"
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Pretrained / resume checkpoints
MODEL_2D_CHECKPOINT = SAVE_PATH / "vit2d_checkpoint.pth"
MODEL_3D_CHECKPOINT = SAVE_PATH / "best_model_3d.pth"

# ============================================================
# 📊 EXPLAINABILITY & LOGGING
# ============================================================

LIME_SAMPLES = 1000
SHAP_NSAMPLES = 100

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 💻 DEVICE SETUP
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Config loaded | Device: {DEVICE}")
