# explainability/lime_explainer.py
"""
LIME explainer for images.
LIME works by perturbing superpixels and calling a predict function returning probabilities.
"""

from lime import lime_image
import numpy as np
from skimage.segmentation import mark_boundaries
from utils.visualization import show_image_with_mask
from config import LIME_SAMPLES

class LIMEExplainer:
    def __init__(self, predict_fn, mode='classification', top_labels=2, n_samples=LIME_SAMPLES):
        """
        predict_fn: function that accepts a list/np.array of images and returns probabilities (N x C)
        """
        self.predict_fn = predict_fn
        self.explainer = lime_image.LimeImageExplainer()
        self.top_labels = top_labels
        self.n_samples = n_samples

    def explain(self, image_np, label=None, hide_color=0, num_features=10):
        """
        image_np: numpy array HxWxC (0-255)
        Returns the explanation and a visualization-ready mask.
        """
        explanation = self.explainer.explain_instance(
            image_np,
            classifier_fn=self.predict_fn,
            top_labels=self.top_labels,
            hide_color=hide_color,
            num_samples=self.n_samples
        )
        # get top label if not provided
        label_of_interest = label if label is not None else explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(label_of_interest, positive_only=True, num_features=num_features, hide_rest=True)
        return explanation, temp, mask

if __name__ == "__main__":
    # Demo usage in notebook or script: supply a model.predict wrapper and a sample image.
    pass
