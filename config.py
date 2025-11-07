# config.py
from pathlib import Path
import torch

# writable root (Kaggle)
ROOT = Path("/kaggle/working")

# Kaggle dataset mounts (update only if you used different slugs)
KAGGLE_PATHS = {
    "XRAY": Path("/kaggle/input/data"),  # should contain Data_Entry_2017.csv and images_*/images/*.png
    "SKIN": Path("/kaggle/input/skin-cancer-mnist-ham10000"),
    "MRI":  Path("/kaggle/input/brain-mri-images-for-brain-tumor-detection"),
}

# training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8        # per step total batch (will be split across GPUs automatically)
NUM_WORKERS = 2
EPOCHS = 6
LR = 2e-4
USE_AMP = True

# image sizes
IMG_SIZE_2D = 224
# Save / checkpoints
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# pretrained ViT (transformers)
VIT_PRETRAINED = "google/vit-base-patch16-224-in21k"
