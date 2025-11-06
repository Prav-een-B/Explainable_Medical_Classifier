# models/vit_model.py
"""
Model wrappers for Vision Transformers (2D and 3D) and a central Multimodal router.
"""

from transformers import ViTForImageClassification, AutoFeatureExtractor
import torch
import torch.nn.functional as F
import numpy as np
from config import (
    VIT_2D_PRETRAINED, NUM_LABELS, DEVICE, 
    MODALITY_CONFIG, IMG_SIZE_3D
)
from PIL import Image

# --- MONAI Imports for 3D ViT ---
from monai.networks.nets import ViT
from monai.utils import ensure_tuple

# --- 2D Model Wrapper (Original) ---
class ViT2DWrapper:
    def __init__(self, pretrained_name=VIT_2D_PRETRAINED, num_labels=NUM_LABELS):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_name)
        self.model = ViTForImageClassification.from_pretrained(pretrained_name, num_labels=num_labels)
        self.model.to(DEVICE)
        self.model.eval()

    def preprocess(self, pil_images):
        inputs = self.feature_extractor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        return pixel_values

    @torch.no_grad()
    def predict(self, pil_images):
        if not pil_images:
            return np.array([]).reshape(0, NUM_LABELS)
        pixel_values = self.preprocess(pil_images)
        outputs = self.model(pixel_values)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs
    
    # ... (save/load methods remain as before) ...

# --- 3D Model Wrapper (Replaced Mock with MONAI ViT) ---
class ViT3DWrapper:
    """Real 3D Vision Transformer wrapper using MONAI."""
    def __init__(self, num_labels=NUM_LABELS):
        cfg_3d = MODALITY_CONFIG["CT"] # Use CT config as default
        in_channels = cfg_3d["channels"]
        img_size_3d = ensure_tuple(cfg_3d["size"])

        # Initialize MONAI Vision Transformer
        # This is a standard ViT adapted for 3D input.
        self.model = ViT(
            in_channels=in_channels,
            img_size=img_size_3d,
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            classification=True,
            num_classes=num_labels,
            dropout_rate=0.1,
        ).to(DEVICE)
        
        self.model.eval()
        self.num_labels = num_labels

    def preprocess(self, volume_tensors):
        """
        Accepts a list of 3D torch tensors (C, D, H, W) 
        and stacks them into a batch.
        """
        # Ensure all tensors are on the correct device
        volume_batch = torch.stack(volume_tensors).to(DEVICE)
        return volume_batch # Shape (B, C, D, H, W)

    @torch.no_grad()
    def predict(self, volume_tensors):
        """Returns softmax probabilities for given list of 3D volume tensors."""
        if not volume_tensors:
            return np.array([]).reshape(0, self.num_labels)
        
        volume_batch = self.preprocess(volume_tensors)
        
        # MONAI ViT returns (logits, hidden_states)
        logits, _ = self.model(volume_batch)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

# --- Multimodal Router (Unchanged) ---
class MultimodalViTWrapper:
    """Routes inference to the correct specialized model based on modality."""
    def __init__(self):
        self.models = {
            "2D": ViT2DWrapper(),
            "3D": ViT3DWrapper()
        }
        print("MultimodalViTWrapper initialized with 2D (HuggingFace) and 3D (MONAI) models.")

    def predict(self, input_data, modality_type):
        """
        Predicts based on the modality_type.
        input_data can be a list of PIL images (2D) or list of Tensors (3D).
        """
        model_type = MODALITY_CONFIG.get(modality_type, {}).get("model_type")
        if not model_type or model_type not in self.models:
            raise ValueError(f"Unsupported or misconfigured modality: {modality_type}")

        return self.models[model_type].predict(input_data)
    
    def get_model(self, modality_type):
        """Helper to get the raw underlying model for explainers."""
        model_type = MODALITY_CONFIG.get(modality_type, {}).get("model_type")
        if model_type == "2D":
            return self.models["2D"].model
        elif model_type == "3D":
            return self.models["3D"].model
        return None
