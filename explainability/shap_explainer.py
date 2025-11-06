# explainability/shap_explainer.py
"""
SHAP explainer module for 2D data.
"""

import shap
import numpy as np

# --- 2D SHAP Explainer (X-ray, Ultrasound) ---
class SHAP2DExplainer:
    def __init__(self, predict_fn, background_data):
        self.predict_fn = predict_fn
        self.background = background_data
        # Reshape background to (N, H*W*C) for KernelExplainer
        flat_background = self.background.reshape(len(self.background), -1)
        self.explainer = shap.KernelExplainer(self._wrapped_predict, flat_background)

    def _wrapped_predict(self, flat_images):
        # Reshape flat (N, D) back to (N, H, W, C) for the model
        imgs_shape = self.background.shape[1:]
        imgs = flat_images.reshape((-1, ) + imgs_shape)
        return self.predict_fn(imgs)

    def explain(self, images_to_explain, nsamples=100):
        flat = images_to_explain.reshape(len(images_to_explain), -1)
        shap_values = self.explainer.shap_values(flat, nsamples=nsamples)
        return shap_values
