# explainability/lime_explainer.py
"""
LIME explainer for 2D images.
"""

from lime import lime_image
import numpy as np
from config import LIME_SAMPLES

# --- 2D LIME Explainer (X-ray, Ultrasound) ---
class LIME2DExplainer:
    def __init__(self, predict_fn, top_labels=2, n_samples=LIME_SAMPLES):
        self.predict_fn = predict_fn
        self.explainer = lime_image.LimeImageExplainer()
        self.top_labels = top_labels
        self.n_samples = n_samples

    def explain(self, image_np, label=None, hide_color=0, num_features=10):
        explanation = self.explainer.explain_instance(
            image_np,
            classifier_fn=self.predict_fn,
            top_labels=self.top_labels,
            hide_color=hide_color,
            num_samples=self.n_samples
        )
        label_of_interest = label if label is not None else explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            label_of_interest, 
            positive_only=True, 
            num_features=num_features, 
            hide_rest=True
        )
        return explanation, temp, mask
