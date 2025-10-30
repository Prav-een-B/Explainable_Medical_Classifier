# models/vit_model.py
"""
Model wrapper for Vision Transformer using HuggingFace transformers.
Provides a simple, consistent API for inference and optional fine-tuning.
"""

from transformers import ViTForImageClassification, AutoFeatureExtractor
import torch
import torch.nn.functional as F
from config import VIT_PRETRAINED, NUM_LABELS, DEVICE

class ViTWrapper:
    def __init__(self, pretrained_name=VIT_PRETRAINED, num_labels=NUM_LABELS):
        # Load feature extractor and model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_name)
        self.model = ViTForImageClassification.from_pretrained(pretrained_name, num_labels=num_labels)
        self.model.to(DEVICE)
        self.model.eval()

    def preprocess(self, pil_images):
        """
        Accepts a list of PIL images (or numpy arrays) and returns tensor
        ready to feed to model (on DEVICE).
        """
        # feature_extractor returns dict with 'pixel_values'
        inputs = self.feature_extractor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        return pixel_values

    @torch.no_grad()
    def predict(self, pil_images):
        """
        Returns softmax probabilities for given list of PIL images.
        Output: numpy array shape (N, num_labels)
        """
        pixel_values = self.preprocess(pil_images)
        outputs = self.model(pixel_values)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def save(self, path):
        self.model.save_pretrained(path)
        self.feature_extractor.save_pretrained(path)

    def load(self, path):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(path)
        self.model = ViTForImageClassification.from_pretrained(path)
        self.model.to(DEVICE)
        self.model.eval()
