"""
config.py
Global configuration file for Explainable_Medical_Classifier
"""

from pathlib import Path

# ============================================================
# 🧠 General Training Configuration
# ============================================================

DEVICE = "cuda"
IMG_SIZE_2D = 224              # Standard input resolution for ViT/ConvNeXt
BATCH_SIZE = 8                 # Batch size per GPU
NUM_WORKERS = 2                # DataLoader workers (Kaggle-safe)
EPOCHS = 6                     # Default number of epochs
LR = 2e-4                      # Default learning rate
USE_AMP = True                 # Enable mixed-precision training

# ============================================================
# 🧩 Dataset Configuration
# ============================================================

DATA_ROOT = Path("data")
CLASS_MAP_JSON = "class_map.json"

# Paths used by dataset_loader.py (Kaggle version)
KAGGLE_PATHS = {
    "XRAY": Path("/kaggle/input/data"),  # NIH Chest X-ray
    "SKIN": Path("/kaggle/input/skin-cancer-mnist-ham10000"),
    "MRI":  Path("/kaggle/input/brain-mri-images-for-brain-tumor-detection"),
}

# ============================================================
# 🎯 Modality-Specific Configuration
# ============================================================

MODALITY_CONFIG = {
    "XRAY": {
        "n_classes": 14,
        "loss": "BCEWithLogitsLoss",
        "task_type": "multi-label"
    },
    "SKIN": {
        "n_classes": 7,
        "loss": "CrossEntropyLoss",
        "task_type": "multi-class"
    },
    "MRI": {
        "n_classes": 3,
        "loss": "CrossEntropyLoss",
        "task_type": "multi-class"
    }
}

# ============================================================
# 🧠 Vision Transformer (ViT) Backbone
# ============================================================

# For HuggingFace Transformers
# Example: "google/vit-base-patch16-224-in21k"
VIT_2D_PRETRAINED = "google/vit-base-patch16-224-in21k"

# If using timm instead of transformers:
# VIT_2D_PRETRAINED = "vit_base_patch16_224"

# ============================================================
# 🧩 Other Model Backbones
# ============================================================

# For ConvNeXt-based multi-task model
CONVNEXT_VARIANT = "convnext_tiny"  # Options: tiny, base, small, large
CONVNEXT_PRETRAINED = True

# For EfficientNet (if needed)
EFFICIENT_VARIANT = "tf_efficientnet_b0"
EFFICIENT_PRETRAINED = True

# ============================================================
# 💾 Checkpointing & Logging
# ============================================================

CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = Path("/kaggle/working/train_log.csv")

# ============================================================
# 📈 Miscellaneous
# ============================================================

SEED = 42                       # Reproducibility
PRINT_FREQ = 50                 # Logging frequency per epoch
SAVE_EVERY = 1                  # Save checkpoint after every N epochs
