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
BATCH_SIZE = 16        # per step total batch
NUM_WORKERS = 2
EPOCHS = 4             # Set to a reasonable number for full training
LR = 2e-4
USE_AMP = True

# image sizes
IMG_SIZE_2D = 224
IMG_SIZE_3D = (96, 96, 96) # Note: This is unused by the current 2D-only train script

# Save / checkpoints
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --- NEW: Output directory for results ---
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- NEW: Explainer settings ---
LIME_SAMPLES = 100    # Number of samples for LIME
SHAP_NSAMPLES = 50    # Number of samples for SHAP (keep low for speed)


# pretrained ViT (transformers)
VIT_PRETRAINED = "google/vit-base-patch16-224-in21k"

# --- Default Transform (for Validation / Loading) ---
from torchvision import transforms

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE_2D, IMG_SIZE_2D)),
    transforms.CenterCrop((IMG_SIZE_2D, IMG_SIZE_2D)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
