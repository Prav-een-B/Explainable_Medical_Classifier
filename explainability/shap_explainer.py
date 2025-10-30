# explainability/shap_explainer.py
"""
SHAP explainer module.
We provide a wrapper using KernelExplainer for black-box models (works but can be slow).
For neural nets with gradients available, one could use GradientExplainer or DeepExplainer.
"""

import shap
import numpy as np
from config import SHAP_BACKGROUND_SAMPLES
from utils.visualization import show_image_with_mask

class SHAPExplainer:
    def __init__(self, predict_fn, background_data):
        """
        predict_fn: function mapping list/np.array of images -> probabilities (N x C)
        background_data: small set of images to use as background (np array)
        """
        self.predict_fn = predict_fn
        self.background = background_data
        # KernelExplainer expects a function that accepts 2D arrays; we'll wrap images to flat vectors inside the function.
        self.explainer = shap.KernelExplainer(self._wrapped_predict, self.background.reshape(len(self.background), -1))

    def _wrapped_predict(self, flat_images):
        """
        Accepts flat images with shape (N, D) and returns predictions
        """
        imgs = flat_images.reshape((-1, ) + self.background.shape[1:])  # (N, H, W, C)
        return self.predict_fn(imgs)

    def explain(self, images_to_explain, nsamples=100):
        """
        images_to_explain: np array shape (N,H,W,C)
        returns shap_values (list or array)
        """
        flat = images_to_explain.reshape(len(images_to_explain), -1)
        shap_values = self.explainer.shap_values(flat, nsamples=nsamples)
        return shap_values
