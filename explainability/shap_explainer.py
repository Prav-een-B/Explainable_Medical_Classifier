# explainability/shap_explainer.py
"""
SHAP explainer module for 2D and 3D data.
"""

import shap
import numpy as np
from config import SHAP_BACKGROUND_SAMPLES

# --- 2D SHAP Explainer (X-ray, Ultrasound) ---
class SHAP2DExplainer:
    def __init__(self, predict_fn, background_data):
        self.predict_fn = predict_fn
        self.background = background_data
        self.explainer = shap.KernelExplainer(self._wrapped_predict, self.background.reshape(len(self.background), -1))

    def _wrapped_predict(self, flat_images):
        imgs = flat_images.reshape((-1, ) + self.background.shape[1:])  # (N, H, W, C)
        return self.predict_fn(imgs)

    def explain(self, images_to_explain, nsamples=100):
        flat = images_to_explain.reshape(len(images_to_explain), -1)
        shap_values = self.explainer.shap_values(flat, nsamples=nsamples)
        return shap_values

# --- 3D SHAP Explainer (CT, MRI) ---
class SHAP3DExplainer:
    def __init__(self, predict_fn, background_data):
        self.predict_fn = predict_fn
        self.background = background_data
        
        # Flatten the 3D data for the explainer (D*H*W features)
        flat_background = self.background.reshape(len(self.background), -1)
        # Note: 3D SHAP is extremely slow/computationally expensive in practice
        self.explainer = shap.KernelExplainer(self._wrapped_predict, flat_background)

    def _wrapped_predict(self, flat_volumes):
        """Accepts flat volumes and returns predictions."""
        imgs_shape = self.background.shape[1:] 
        # Reshape flat vectors back to 3D volumes (D, H, W)
        volumes = flat_volumes.reshape((-1, ) + imgs_shape)
        return self.predict_fn(volumes)

    def explain(self, volumes_to_explain, nsamples=100):
        flat = volumes_to_explain.reshape(len(volumes_to_explain), -1)
        shap_values = self.explainer.shap_values(flat, nsamples=nsamples)
        return shap_values
